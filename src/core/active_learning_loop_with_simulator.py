"""
active_learning_with_simulator.py
=================================

Active Learning (AL) loop z možnostjo poizvedb v simulator (on-demand).

- Strategije: uncertainty, entropy, margin, random
- RandomForest (class_weight="balanced")
- Vračilo: (metrics_per_iteration, duration_wall_sec, kpi_summary)
"""

from typing import Tuple, List, Dict, Any, Optional

import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from core.power_grid_simulator_interface import query_simulator_cached


# -----------------------------
# Query scoring (AL strategies)
# -----------------------------
def compute_query_scores(proba: np.ndarray, strategy: str) -> np.ndarray:
    """
    Compute query scores for unlabeled samples based on strategy.

    Args:
        proba (ndarray): predicted probabilities, shape = (n_samples, n_classes)
        strategy (str): one of ["uncertainty", "entropy", "margin", "random"]

    Returns:
        scores (ndarray): higher score = higher priority for querying
    """
    if strategy == "uncertainty":
        return 1.0 - proba.max(axis=1)
    elif strategy == "entropy":
        logp = np.log(proba + 1e-12)
        return -np.sum(proba * logp, axis=1)
    elif strategy == "margin":
        sorted_proba = np.sort(proba, axis=1)
        return -(sorted_proba[:, -1] - sorted_proba[:, -2])
    elif strategy == "random":
        return np.zeros(proba.shape[0])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def sample_to_dict(row: pd.Series) -> Dict[str, Any]:
    """Convert one row of a DataFrame into a {feature_name: value} dict."""
    return row.to_dict()


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return ROC AUC if y_true has >= 2 razreda, sicer np.nan (brez opozoril)."""
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


# -----------------------------------------
# Main: Active Learning with optional sim
# -----------------------------------------
def run_active_learning(
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    strategy: str,
    initial_size: int,
    batch_size: int,
    iterations: int,
    random_state: int = 42,
    simulate_on_demand: bool = False,
    avg_sim_time_sec: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    """
    Main Active Learning loop (with optional simulator queries).

    Args:
        X_pool: unlabeled feature pool
        y_pool: labels (used only if simulate_on_demand=False)
        X_val, y_val: validation set (TRUE labels)
        strategy: "uncertainty" | "entropy" | "margin" | "random"
        initial_size: initially labeled samples
        batch_size: new queries per iteration
        iterations: AL steps
        random_state: seed
        simulate_on_demand: if True → query simulator for labels
        avg_sim_time_sec: if given, also compute estimated sim time KPI

    Returns:
        metrics_per_iteration: list of dicts (metrics + KPI counters per iteration)
        duration: wall-clock duration of the AL run (seconds)
        kpi_summary: final KPI snapshot (dict)
    """
    rng = np.random.default_rng(random_state)

    # Clean indices
    X_pool = X_pool.reset_index(drop=True)
    if hasattr(y_pool, "reset_index"):
        y_pool = y_pool.reset_index(drop=True)

    n_samples = len(X_pool)
    if n_samples == 0:
        raise ValueError("X_pool is empty.")
    if initial_size <= 0:
        raise ValueError("initial_size must be >= 1.")

    # Init labeled mask
    labeled_mask = np.zeros(n_samples, dtype=bool)
    labeled_mask[rng.choice(n_samples, size=min(initial_size, n_samples), replace=False)] = True

    # Labels buffer
    if simulate_on_demand:
        y_labels = np.full(n_samples, np.nan, dtype=float)
    else:
        y_labels = y_pool.values if hasattr(y_pool, "values") else np.asarray(y_pool, dtype=float)

    # KPI counters
    sim_calls = 0
    sim_time_sec = 0.0
    wall_start = time.perf_counter()

    # Initial on-demand labels
    if simulate_on_demand:
        init_idx = np.where(labeled_mask)[0]
        for i in init_idx:
            if np.isnan(y_labels[i]):
                t0 = time.perf_counter()
                sim_label = query_simulator_cached(sample_to_dict(X_pool.iloc[i]))
                sim_time_sec += time.perf_counter() - t0
                sim_calls += 1
                y_labels[i] = 1 if sim_label == "secure" else 0

    metrics_per_iteration: List[Dict[str, Any]] = []

    # ---- Active Learning loop ----
    for it in tqdm(range(iterations), desc="Active Learning iterations"):
        labeled_indices = np.where(labeled_mask)[0]
        if labeled_indices.size == 0:
            break

        # Train model
        X_train = X_pool.iloc[labeled_indices]
        y_train = y_labels[labeled_indices].astype(int)

        train_t0 = time.perf_counter()
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        train_time_sec = time.perf_counter() - train_t0

        # Evaluate
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        total_labeled_now = int(labeled_mask.sum())
        metrics_per_iteration.append({
            "iteration": it + 1,
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
            "roc_auc": _safe_roc_auc(np.asarray(y_val), y_prob),
            # KPI counters (cumulative)
            "total_labeled": total_labeled_now,
            "sim_calls_cum": int(sim_calls),
            "sim_time_cum_sec": float(sim_time_sec),
            "train_time_sec": float(train_time_sec),
            "wall_time_cum_sec": float(time.perf_counter() - wall_start),
        })

        # Stop if no unlabeled samples remain
        if not (~labeled_mask).any():
            break

        # Select new samples
        unlabeled_indices = np.where(~labeled_mask)[0]
        if unlabeled_indices.size == 0:
            break

        take = min(batch_size, unlabeled_indices.size)
        if strategy == "random":
            query_indices = rng.choice(unlabeled_indices, size=take, replace=False)
        else:
            proba_unl = model.predict_proba(X_pool.iloc[unlabeled_indices])
            scores = compute_query_scores(proba_unl, strategy)
            top_local = np.argsort(scores)[::-1][:take]
            query_indices = unlabeled_indices[top_local]

        # Mark queried
        labeled_mask[query_indices] = True

        # On-demand labels
        if simulate_on_demand:
            for i in query_indices:
                if np.isnan(y_labels[i]):
                    t0 = time.perf_counter()
                    sim_label = query_simulator_cached(sample_to_dict(X_pool.iloc[i]))
                    sim_time_sec += time.perf_counter() - t0
                    sim_calls += 1
                    y_labels[i] = 1 if sim_label == "secure" else 0

    duration = time.perf_counter() - wall_start

    # Final KPI summary
    final_it = metrics_per_iteration[-1] if metrics_per_iteration else {}
    kpi_summary = {
        "final_iteration": int(final_it.get("iteration", 0)),
        "final_accuracy": float(final_it.get("accuracy", float("nan"))),
        "final_auc": float(final_it.get("roc_auc", float("nan"))),
        "total_labeled": int(final_it.get("total_labeled", int(np.sum(labeled_mask)))),
        "sim_calls": int(sim_calls),
        "sim_time_sec_measured": float(sim_time_sec),
        "runtime_wall_sec": float(duration),
        "strategy": strategy,
        "initial_size": int(initial_size),
        "batch_size": int(batch_size),
        "iterations_requested": int(iterations),
    }
    if avg_sim_time_sec is not None:
        kpi_summary["sim_time_sec_estimated"] = float(sim_calls) * float(avg_sim_time_sec)

    return metrics_per_iteration, duration, kpi_summary
