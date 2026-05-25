from __future__ import annotations

import os
import json
import uuid
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Header

from app.schemas import RetrainRequest, RetrainResponse
from app.minio_io import (
    get_humaine_auth,
    download_object_to_path,
    upload_path_as_object,
)
from app.train import train_rf_and_save

app = FastAPI(title="Smart Energy Retraining API", version="0.1.0")

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DATASET_PATH = APP_ROOT / "data" / "base" / "simulation_security_labels_n-1.csv"
DEFAULT_APPENDED_LOCAL_PATH = (
    APP_ROOT / "data" / "appended" / "simulation_security_labels_n-1_appended_rows_latest.csv"
)
BASE_DATASET_LOCAL_PATH = Path(
    os.getenv("BASE_DATASET_LOCAL_PATH", str(DEFAULT_BASE_DATASET_PATH))
).resolve()
APPENDED_ROWS_LOCAL_PATH = Path(
    os.getenv("APPENDED_ROWS_LOCAL_PATH", str(DEFAULT_APPENDED_LOCAL_PATH))
).resolve()


# ----------------------------
# Minimal auth for remote calls
# ----------------------------
def verify_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    """
    If API_KEY env var is set -> require X-API-Key header to match.
    If API_KEY is not set -> allow (useful for local dev).
    """
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/retrain", response_model=RetrainResponse, dependencies=[Depends(verify_api_key)])
def retrain(req: RetrainRequest) -> RetrainResponse:
    """
    Retrain RF model from local base dataset + latest appended rows from MinIO.
    """
    auth = get_humaine_auth()

    appended_rows_key = req.appended_rows_key or req.latest_key
    if not appended_rows_key:
        raise HTTPException(
            status_code=422,
            detail="Missing input key. Provide 'appended_rows_key' (or legacy 'latest_key').",
        )

    if not BASE_DATASET_LOCAL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "Local base dataset not found at "
                f"'{BASE_DATASET_LOCAL_PATH}'. Prepare data/base/simulation_security_labels_n-1.csv first."
            ),
        )

    APPENDED_ROWS_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    model_key = f"{req.output_prefix}/{run_id}/model.joblib"
    metrics_key = f"{req.output_prefix}/{run_id}/metrics.json"

    with tempfile.TemporaryDirectory() as tmpdir:
        merged_path = os.path.join(tmpdir, "merged.csv")
        model_path = os.path.join(tmpdir, "model.joblib")
        metrics_path = os.path.join(tmpdir, "metrics.json")

        # 1) Download only appended rows latest file from MinIO.
        try:
            download_object_to_path(
                auth=auth,
                bucket=req.results_bucket,
                key=appended_rows_key,
                local_path=str(APPENDED_ROWS_LOCAL_PATH),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Download appended rows failed: {e}")

        # 2) Merge local base dataset + downloaded appended rows.
        try:
            df_base = pd.read_csv(BASE_DATASET_LOCAL_PATH)
            try:
                df_appended = pd.read_csv(APPENDED_ROWS_LOCAL_PATH)
            except pd.errors.EmptyDataError:
                df_appended = pd.DataFrame(columns=df_base.columns)

            df_merged = pd.concat([df_base, df_appended], ignore_index=True)
            df_merged.to_csv(merged_path, index=False)
            appended_rows_count = int(df_appended.shape[0])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Local dataset merge failed: {e}")

        # 3) Train from merged dataset.
        try:
            artifacts = train_rf_and_save(
                latest_csv_path=merged_path,
                model_out_path=model_path,
                metrics_out_path=metrics_path,
                test_size=req.test_size,
                n_estimators=req.n_estimators,
                random_state=req.random_state,
                label_col=req.label_col,
                drop_feature_cols=req.drop_feature_cols,
                drop_latest_columns=req.drop_latest_columns,
            )

            # Keep metrics semantics: latest rows now represent appended rows.
            artifacts.n_rows_latest = appended_rows_count
            artifacts.metrics["n_rows_latest"] = appended_rows_count
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(artifacts.metrics, f, indent=2)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Training failed: {e}")

        # 4) Upload artifacts.
        try:
            upload_path_as_object(auth, req.results_bucket, model_key, artifacts.model_path)
            upload_path_as_object(auth, req.results_bucket, metrics_key, artifacts.metrics_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload to MinIO failed: {e}")

        return RetrainResponse(
            ok=True,
            message="Retraining completed.",
            dataset_rows=artifacts.n_rows_all,
            dataset_rows_latest=appended_rows_count,
            model_object=f"{req.results_bucket}/{model_key}",
            metrics_object=f"{req.results_bucket}/{metrics_key}",
            metrics=artifacts.metrics,
        )
