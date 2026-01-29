"""
Trening RandomForest + shranjevanje metrik.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


@dataclass
class TrainArtifacts:
    model_path: str
    metrics_path: str
    metrics: Dict[str, Any]
    n_rows_all: int
    n_rows_baseline: int
    n_rows_latest: int


def _encode_labels(y: pd.Series) -> tuple[pd.Series, Dict[str, int]]:
    uniq = sorted(y.astype(str).unique().tolist())
    mapping = {name: i for i, name in enumerate(uniq)}
    return y.astype(str).map(mapping), mapping


def train_rf_and_save(
    baseline_csv_path: str,
    latest_csv_path: str,
    model_out_path: str,
    metrics_out_path: str,
    *,
    test_size: float = 0.2,
    n_estimators: int = 400,
    random_state: int = 42,
    label_col: str = "status",
    drop_feature_cols: Optional[List[str]] = None,
    drop_latest_columns: Optional[List[str]] = None,
) -> TrainArtifacts:
    drop_feature_cols = drop_feature_cols or []
    drop_latest_columns = drop_latest_columns or []

    df_base = pd.read_csv(baseline_csv_path)
    df_latest = pd.read_csv(latest_csv_path)

    # drop columns that exist only in latest (e.g. created_at)
    for c in drop_latest_columns:
        if c in df_latest.columns:
            df_latest = df_latest.drop(columns=[c])

    # concat
    df = pd.concat([df_base, df_latest], ignore_index=True, sort=False)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset columns.")

    # split X/y
    y_raw = df[label_col]
    X = df.drop(columns=[label_col])

    # drop non-feature columns if present
    for c in drop_feature_cols:
        if c in X.columns:
            X = X.drop(columns=[c])

    # ensure numeric (RF needs numeric)
    # if any non-numeric slipped in, try to coerce
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # fill NaNs (safe default)
    X = X.fillna(0.0)

    y, mapping = _encode_labels(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "label_mapping": mapping,
        "n_estimators": int(n_estimators),
        "random_state": int(random_state),
        "test_size": float(test_size),
        "n_features": int(X.shape[1]),
    }

    joblib.dump(clf, model_out_path)
    with open(metrics_out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return TrainArtifacts(
        model_path=model_out_path,
        metrics_path=metrics_out_path,
        metrics=metrics,
        n_rows_all=int(df.shape[0]),
        n_rows_baseline=int(df_base.shape[0]),
        n_rows_latest=int(df_latest.shape[0]),
    )
