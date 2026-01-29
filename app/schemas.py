"""
Pydantic sheme.
"""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class RetrainRequest(BaseModel):
    # Buckets + object keys
    results_bucket: str = Field(default="smart-energy-results")
    baseline_bucket: str = Field(default="smart-energy-results")

    baseline_key: str = Field(default="al_training_dataset/simulation_security_labels_n-1.csv")
    latest_key: str = Field(default="al_training_dataset/simulation_security_labels_n-1_latest.csv")

    # Kam shranit output (prefix folder)
    output_prefix: str = Field(default="models/retrained_rf")

    # Model params
    random_state: int = 42
    n_estimators: int = 400
    test_size: float = 0.2

    # Dataset params
    label_col: str = Field(default="status")

    # feature cleanup
    drop_feature_cols: List[str] = Field(default_factory=lambda: ["timestamp"])
    drop_latest_columns: List[str] = Field(default_factory=lambda: ["created_at"])


class RetrainResponse(BaseModel):
    ok: bool
    message: str
    dataset_rows: int
    dataset_rows_baseline: int
    dataset_rows_latest: int
    model_object: str
    metrics_object: str
    metrics: Dict[str, Any]
