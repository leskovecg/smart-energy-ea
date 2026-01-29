from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from app.schemas import RetrainRequest, RetrainResponse
from app.minio_io import (
    get_humaine_auth,
    download_object_to_path,
    upload_path_as_object,
)
from app.train import train_rf_and_save

app = FastAPI(title="Smart Energy Retraining API")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/retrain", response_model=RetrainResponse)
def retrain(req: RetrainRequest):
    # FIX: pravilna validacija env var-ov (HumAIne, ne klasični MinIO)
    needed = ["HUMAINE_API_BASE_URL", "HUMAINE_API_USERNAME", "HUMAINE_API_PASSWORD"]
    missing = [k for k in needed if not os.getenv(k)]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing env vars: {missing}")

    auth = get_humaine_auth()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # objektne poti v MinIO (key)
    # FIX: key strukturiran v folder: output_prefix/run_id/...
    model_key = f"{req.output_prefix}/{run_id}/model.joblib"
    metrics_key = f"{req.output_prefix}/{run_id}/metrics.json"

    with tempfile.TemporaryDirectory() as td:
        baseline_path = os.path.join(td, "baseline.csv")
        latest_path = os.path.join(td, "latest.csv")
        model_path = os.path.join(td, "model.joblib")
        metrics_path = os.path.join(td, "metrics.json")

        # download baseline
        try:
            download_object_to_path(auth, req.baseline_bucket, req.baseline_key, baseline_path)
        except Exception as e:
            # FIX: popravljeno sporočilo (brez "Can...om")
            raise HTTPException(
                status_code=404,
                detail=f"Cannot download baseline from MinIO: {req.baseline_bucket}/{req.baseline_key}. Error: {e}",
            )

        # download latest
        try:
            download_object_to_path(auth, req.results_bucket, req.latest_key, latest_path)
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Cannot download latest from MinIO: {req.results_bucket}/{req.latest_key}. Error: {e}",
            )

        # train + save artifacts locally
        try:
            artifacts = train_rf_and_save(
                baseline_csv_path=baseline_path,
                latest_csv_path=latest_path,
                model_out_path=model_path,
                metrics_out_path=metrics_path,
                test_size=req.test_size,
                n_estimators=req.n_estimators,
                random_state=req.random_state,
                label_col=req.label_col,
                drop_feature_cols=req.drop_feature_cols,
                drop_latest_columns=req.drop_latest_columns,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Training failed: {e}")

        # upload back to MinIO
        try:
            upload_path_as_object(auth, req.results_bucket, model_key, artifacts.model_path)
            upload_path_as_object(auth, req.results_bucket, metrics_key, artifacts.metrics_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload to MinIO failed: {e}")

        return RetrainResponse(
            ok=True,
            message="Retraining completed.",
            dataset_rows=artifacts.n_rows_all,
            dataset_rows_baseline=artifacts.n_rows_baseline,
            dataset_rows_latest=artifacts.n_rows_latest,
            model_object=f"{req.results_bucket}/{model_key}",
            metrics_object=f"{req.results_bucket}/{metrics_key}",
            metrics=artifacts.metrics,
        )
