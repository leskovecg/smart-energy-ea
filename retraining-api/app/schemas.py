"""
Pydantic schemas for the Smart Energy retraining API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RetrainRequest(BaseModel):
    # MinIO / HumAIne storage
    results_bucket: str = Field(
        default="smart-energy-results",
        description="Bucket used for appended rows input and for uploading artifacts.",
    )
    appended_rows_key: str = Field(
        default=(
            "al_training_dataset/appended_rows/"
            "simulation_security_labels_n-1_appended_rows_latest.csv"
        ),
        description="MinIO object key for appended rows latest CSV.",
    )
    latest_key: Optional[str] = Field(
        default=None,
        description="Legacy alias for appended_rows_key (kept for backward compatibility).",
    )

    # Output location
    output_prefix: str = Field(
        default="models/retraining_runs",
        description="Prefix for uploading run artifacts (model + metrics).",
    )

    # Training params
    n_estimators: int = Field(default=400, ge=1, description="Number of trees in RandomForest.")
    random_state: int = Field(default=42, description="Random seed.")
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0, description="Train/test split ratio.")

    # Dataset / feature handling
    label_col: str = Field(default="status", description="Name of the target label column.")
    drop_feature_cols: List[str] = Field(default_factory=list, description="Columns to drop from X before training.")
    drop_latest_columns: List[str] = Field(
        default_factory=lambda: ["created_at"],
        description="Columns to drop from merged dataset (metadata columns).",
    )


class RetrainResponse(BaseModel):
    ok: bool
    message: str

    dataset_rows: int
    dataset_rows_latest: int

    model_object: str
    metrics_object: str

    metrics: Dict[str, Any]
