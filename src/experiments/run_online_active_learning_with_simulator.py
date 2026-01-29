"""
run_simulated_active_learning.py
================================
End-to-end Active Learning run with on-demand simulator labels.

- STRICT time-based split if 'timestamp' exists (no overlap):
    pool = past, validation = future
  Fallback to stratified split if no 'timestamp'.
- Whitelist of feature columns to avoid leakage (e.g., only exogenous inputs).
- Runs AL via active_learning_with_simulator.run_active_learning(simulate_on_demand=True)
- Saves per-iteration metrics and KPI summary to CSV + XLSX (multi-sheet)
"""

import os
import argparse
from typing import Optional
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from core.active_learning_loop_with_simulator import run_active_learning
from core.power_grid_simulator_interface import _cached_query_simulator_by_key


# ===================== DIAGNOSTICS UTILS =====================
def _fmt_counts(y):
    c = Counter(y)
    total = sum(c.values())
    parts = []
    for k in sorted(c.keys()):
        parts.append(f"{k}: {c[k]} ({(c[k] / total):.2%})")
    return ", ".join(parts), c, total


def check_split_diagnostics(
    y_pool,
    y_val,
    ts_pool: Optional[pd.Series] = None,
    ts_val: Optional[pd.Series] = None,
    title: str = "SPLIT",
):
    """
    Prints class balance and, if timestamps are provided, checks if split is time-separated.
    """
    print("\n" + "=" * 70)
    print(f"[DIAG] {title}")
    print("-" * 70)

    pool_str, pool_counts, pool_n = _fmt_counts(y_pool)
    val_str, val_counts, val_n = _fmt_counts(y_val)
    print(f"POOL   class balance: {pool_str}  | N={pool_n}")
    print(f"VALID. class balance: {val_str}   | N={val_n}")

    if len(pool_counts) < 2:
        print("WARN: POOL has < 2 classes (possibly unstratified initial sample).")
    if len(val_counts) < 2:
        print("WARN: VALIDATION has < 2 classes (metrics like AUC will be invalid).")

    if ts_pool is not None and ts_val is not None:
        try:
            ts_pool = pd.to_datetime(ts_pool)
            ts_val = pd.to_datetime(ts_val)
            pool_min, pool_max = ts_pool.min(), ts_pool.max()
            val_min, val_max = ts_val.min(), ts_val.max()

            print("-" * 70)
            print(f"POOL   time range: {pool_min}  ->  {pool_max}")
            print(f"VALID. time range: {val_min}   ->  {val_max}")

            strictly_time_separated = pool_max < val_min
            overlaps = not strictly_time_separated and not (val_max < pool_min)

            if strictly_time_separated:
                print("TIME CHECK: Strictly time-based forward split (no overlap).")
            elif overlaps:
                print("TIME CHECK: WARNING: time ranges OVERLAP (not time-based).")
            else:
                print("TIME CHECK: Time-based (reverse order), no overlap.")
        except Exception as e:
            print(f"TIME CHECK: could not parse timestamps ({e})")
    else:
        print("TIME CHECK: (no timestamps) - skipped.")
    print("=" * 70)
# =================== END DIAGNOSTICS UTILS ===================


def _select_feature_columns(df_like: pd.DataFrame) -> list[str]:
    """
    Whitelist feature selection to avoid leakage.
    Adjust ALLOWED_PREFIX to your exogenous inputs (extend as needed).
    If nothing matches, fallback to 'all except drop_cols_base'.
    """
    drop_cols_base = [c for c in ["timestamp", "status", "secure", "label"] if c in df_like.columns]
    allowed_prefix = ("load_", "gen_", "sgen_")  # TODO: extend (e.g., "pv_", "wind_", "weather_")
    feat_whitelist = [c for c in df_like.columns if any(c.startswith(p) for p in allowed_prefix)]
    if len(feat_whitelist) == 0:
        # Fallback: keep everything except drop/base label cols
        feat_whitelist = [c for c in df_like.columns if c not in drop_cols_base]
        print("[WARN] Whitelist matched 0 columns; using fallback (all except labels/timestamp).")
    return feat_whitelist


def main(
    data_path: str,
    strategy: str,
    initial_size: int,
    batch_size: int,
    iterations: int,
    test_size: float,
    random_state: int,
    avg_sim_time_sec: Optional[float],
    tables_dir: str,
) -> None:
    # 1) Load dataset
    df = pd.read_csv(data_path)

    if "status" not in df.columns:
        raise ValueError("Column 'status' not found in the dataset. Expected values: 'secure'/'insecure'.")

    # TRUE labels (for diagnostics and validation eval)
    y_true_all = df["status"].map({"secure": 1, "insecure": 0}).astype(int)

    # 2) Split (STRICT time-based if timestamp exists; otherwise stratified)
    drop_cols_base = [c for c in ["timestamp", "status", "secure", "label"] if c in df.columns]

    if "timestamp" in df.columns:
        # --- STRICT TIME-BASED SPLIT (no overlap) ---
        df_sorted = df.copy()
        df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"])
        df_sorted = df_sorted.sort_values("timestamp").reset_index(drop=True)

        feature_cols = _select_feature_columns(df_sorted)

        split_idx = int(len(df_sorted) * (1.0 - test_size))
        df_pool = df_sorted.iloc[:split_idx]   # past -> AL pool (unlabeled initially)
        df_val  = df_sorted.iloc[split_idx:]   # future -> validation (true labels)

        X_pool = df_pool[feature_cols].reset_index(drop=True)
        X_val  = df_val[feature_cols].reset_index(drop=True)

        y_pool_true = df_pool["status"].map({"secure": 1, "insecure": 0}).astype(int).reset_index(drop=True)
        y_val = df_val["status"].map({"secure": 1, "insecure": 0}).astype(int).reset_index(drop=True)

        # dummy pool labels (ignored in simulate_on_demand=True)
        y_pool_dummy = pd.Series(np.zeros(len(X_pool), dtype=int), name="status_binary")

        ts_pool = df_pool["timestamp"].reset_index(drop=True)
        ts_val = df_val["timestamp"].reset_index(drop=True)
    else:
        # --- FALLBACK: stratified split (if no timestamps) ---
        feature_cols = _select_feature_columns(df)
        X_all = df[feature_cols].copy()
        y_dummy_all = pd.Series(np.zeros(len(X_all), dtype=int), name="status_binary")

        X_pool, X_val, y_pool_dummy, _ = train_test_split(
            X_all,
            y_dummy_all,
            test_size=test_size,
            random_state=random_state,
            stratify=y_true_all,
        )

        y_pool_true = y_true_all.iloc[X_pool.index].reset_index(drop=True)
        y_val = y_true_all.iloc[X_val.index].reset_index(drop=True)

        ts_pool = df.iloc[X_pool.index]["timestamp"] if "timestamp" in df.columns else None
        ts_val = df.iloc[X_val.index]["timestamp"] if "timestamp" in df.columns else None

        X_pool = X_pool.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        y_pool_dummy = y_pool_dummy.reset_index(drop=True)

    # 3) Diagnostics (use TRUE labels for pool/val only for printing)
    check_split_diagnostics(
        y_pool_true,
        y_val,
        ts_pool=ts_pool,
        ts_val=ts_val,
        title=f"ONLINE | {strategy}",
    )

    # 4) Run Active Learning (online = simulate_on_demand=True)
    metrics, duration, kpi = run_active_learning(
        X_pool=X_pool,
        y_pool=y_pool_dummy,               # dummy; not used in online
        X_val=X_val,
        y_val=y_val,
        strategy=strategy,
        initial_size=initial_size,
        batch_size=batch_size,
        iterations=iterations,
        random_state=random_state,
        simulate_on_demand=True,
        avg_sim_time_sec=avg_sim_time_sec,
    )

    # 5) Save results (CSV + XLSX)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"metrics_simulated_{strategy}_init{initial_size}_b{batch_size}_it{iterations}_{ts_str}"

    csv_path = os.path.join(tables_dir, f"{base}.csv")
    xlsx_path = os.path.join(tables_dir, f"{base}.xlsx")

    df_iter = pd.DataFrame(metrics)
    df_iter.to_csv(csv_path, index=False)

    meta = {
        "data_path": data_path,
        "strategy": strategy,
        "initial_size": initial_size,
        "batch_size": batch_size,
        "iterations": iterations,
        "test_size": test_size,
        "random_state": random_state,
        "pool_size": len(X_pool),
        "val_size": len(X_val),
        "runtime_wall_sec": duration,
    }
    df_kpi = pd.DataFrame([{**meta, **kpi}])

    try:
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            df_iter.to_excel(writer, sheet_name="per_iteration", index=False)
            df_kpi.to_excel(writer, sheet_name="kpi_summary", index=False)
    except Exception:
        with pd.ExcelWriter(xlsx_path) as writer:
            df_iter.to_excel(writer, sheet_name="per_iteration", index=False)
            df_kpi.to_excel(writer, sheet_name="kpi_summary", index=False)

    print(f"Finished. Total runtime: {duration:.2f} s")
    print(f"Saved CSV : {csv_path}")
    print(f"Saved XLSX: {xlsx_path}")


if __name__ == "__main__":

    HERE = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(os.path.join(HERE, ".."))       
    DEFAULT_DATA = os.path.join(BASE_DIR, "data", "simulation_security_labels_n-1.csv")
    DEFAULT_TABLES = os.path.join(BASE_DIR, "tables")
    
    parser = argparse.ArgumentParser(description="Run AL with simulator (on-demand labels).")
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA,
        help="Path to dataset CSV.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="entropy",
        choices=["entropy", "uncertainty", "margin", "random"],
        help="Query strategy.",
    )
    parser.add_argument("--init", type=int, default=100, help="Initial labeled size.")
    parser.add_argument("--batch", type=int, default=50, help="Batch size per iteration.")
    parser.add_argument("--iters", type=int, default=40, help="Number of AL iterations.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Validation fraction (0–1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--avg-sim-sec",
        type=float,
        default=None,
        help="Average simulator time per label (sec) to compute estimated time KPI (optional).",
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default=DEFAULT_TABLES,
        help="Directory where result tables (CSV/XLSX) are saved.",
    )

    args = parser.parse_args()

    # clear simulator cache at the beginning of the run
    _cached_query_simulator_by_key.cache_clear()

    main(
        data_path=args.data,
        strategy=args.strategy,
        initial_size=args.init,
        batch_size=args.batch,
        iterations=args.iters,
        test_size=args.test_size,
        random_state=args.seed,
        avg_sim_time_sec=args.avg_sim_sec,
        tables_dir=args.tables_dir,
    )
