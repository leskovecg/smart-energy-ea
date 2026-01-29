"""
Active Learning Runner (Online) for Power-Grid Security Assessment

What this script does
---------------------
- Loads a labeled dataset of power-grid states (with a 'status' column: 'secure'/'insecure').
- Splits the data into:
  * POOL  : unlabeled pool used by the Active Learning (AL) loop (labels are queried on demand
            from an external simulator interface via `query_simulator_cached`).
  * VAL   : fixed validation split used only for evaluation each AL iteration.
- Runs an Active Learning loop with a RandomForest classifier:
  * Start by labeling an initial random subset from the POOL (size = --init).
  * Train on currently labeled data.
  * Score the remaining unlabeled pool with a chosen strategy: entropy / uncertainty / margin / random.
  * Query a batch (--batch) of highest-priority samples and fetch their labels from the simulator.
  * Repeat for --iters iterations while tracking metrics (ACC, F1, ROC AUC, timing, etc.).
- Saves:
  * Per-iteration metrics as a table (CSV).
  * Final KPIs and a small JSON with run metadata.
  * Filenames are structured via `build_run_slug` and saved with `save_with_meta`.

Key ideas / assumptions
-----------------------
- The simulator is the ground truth oracle for pool samples. Calls are cached.
- If your CSV has a 'timestamp' column, the split is time-aware (earlier → pool, later → validation).
- Feature columns are auto-selected (whitelist by prefixes 'load_', 'gen_', 'sgen_'; otherwise
  fallback to "all non-label columns").

CLI (examples)
--------------
python 10_run_al.py --strategy entropy --init 100 --batch 50 --iters 40
python 10_run_al.py --data ../data/simulation_security_labels_n-1.csv --test-size 0.1

Outputs
-------
- tables/<slug>__iter.csv         (per-iteration metrics)
- tables/<slug>__kpi.csv          (one-row final KPIs)
- tables/<slug>__meta.json        (configuration metadata)
"""

import os, time, argparse
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from core.power_grid_simulator_interface import query_simulator_cached, _cached_query_simulator_by_key

from utils.experiment_filename_namer import build_run_slug, save_with_meta

# ---------- AL strategies ----------
def compute_query_scores(proba: np.ndarray, strategy: str) -> np.ndarray:
    """
    Convert class-probabilities to "query scores" for ranking unlabeled samples.

    Args:
        proba: array of shape (n_samples, n_classes) with predicted probabilities.
        strategy: one of {"uncertainty", "entropy", "margin", "random"}.

    Returns:
        scores: array of shape (n_samples,). Higher score = higher priority to query.
    """

    # "Uncertainty" = 1 - max_class_prob
    if strategy == "uncertainty":
        return 1.0 - proba.max(axis=1)
    
    # "Entropy" = -sum(p * log p)
    if strategy == "entropy":
        logp = np.log(proba + 1e-12)
        return -(proba * logp).sum(axis=1)
    
    # "Margin" = -(p1 - p2) where p1 >= p2 are the top two probs
    if strategy == "margin":
        sp = np.sort(proba, axis=1)
        return -(sp[:, -1] - sp[:, -2])
    
    # "Random" = all zeros → selection will be random later
    if strategy == "random":
        return np.zeros(proba.shape[0])
    
    raise ValueError(f"Unknown strategy: {strategy}")

def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute ROC AUC safely. If only one class present or an error occurs, return NaN.
    """

    try:
        if len(np.unique(y_true)) < 2: return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")

# ---------- CORE: Active Learning ----------
def run_active_learning(
    X_pool: pd.DataFrame,
    y_pool_dummy: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    strategy: str,
    initial_size: int,
    batch_size: int,
    iterations: int,
    random_state: int = 42,
    simulate_on_demand: bool = True,
    avg_sim_time_sec: Optional[float] = None,
    rf_params: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    """
    Active Learning loop (online/oracle setting).

    Steps per iteration:
      1) Train RF on currently labeled pool items.
      2) Evaluate on fixed validation set (X_val, y_val).
      3) Score unlabeled pool using the chosen strategy.
      4) Select top-K (batch_size) and query simulator for their labels.
      5) Repeat.

    Returns:
        metrics: list of per-iteration dicts (accuracy, f1, auc, counts, timing).
        duration: total wall time in seconds.
        kpi: final KPIs/summary of the run (final accuracy, labels used, sim calls, etc.).
    """

    # Random generator for reproducible choices
    rng = np.random.default_rng(random_state)

    # Reset indices so that masks and iloc indexing match cleanly
    X_pool = X_pool.reset_index(drop=True)
    y_pool_dummy = y_pool_dummy.reset_index(drop=True)

    # Basic checks
    n = len(X_pool)
    if n == 0: 
        raise ValueError("X_pool is empty.")
    if initial_size <= 0: 
        raise ValueError("initial_size must be >= 1.")

    # Boolean mask: which pool items are already labeled
    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[rng.choice(n, size=min(initial_size, n), replace=False)] = True

    # Label buffer (np.nan means "unknown" until we query the simulator)
    y_labels = np.full(n, np.nan, dtype=float)

    # Track simulator usage and time
    sim_calls = 0
    sim_time = 0.0
    t0_wall = time.perf_counter()

    # Query simulator for the initial labeled set
    init_idx = np.where(labeled_mask)[0]
    for i in init_idx:
        if np.isnan(y_labels[i]):
            t0 = time.perf_counter()
            lab = query_simulator_cached(X_pool.iloc[i].to_dict())
            sim_time += time.perf_counter() - t0
            sim_calls += 1
            # Map simulator string to int {secure:1, insecure:0}
            y_labels[i] = 1 if lab == "secure" else 0

    # Store per-iteration metrics
    metrics: List[Dict[str, Any]] = []

    # Default RF params; override with rf_params if provided
    rf_defaults = dict(
        n_estimators=600,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    if rf_params: 
        rf_defaults.update(rf_params)

    # ---------------- Main Active Learning loop ----------------
    for it in tqdm(range(iterations), desc=f"AL-{strategy}"):
        
        # Get indices of currently labeled samples
        L = np.where(labeled_mask)[0]
        if L.size == 0: 
            break # Stop if no labeled data

        # Training data = pool samples with labels obtained so far
        X_tr = X_pool.iloc[L]
        y_tr = y_labels[L].astype(int)

        # Train model on currently labeled data
        t0_train = time.perf_counter()
        model = RandomForestClassifier(**rf_defaults)
        model.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0_train

        # Evaluate on the fixed validation set
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        # Save metrics for this iteration
        metrics.append({
            "iteration": it + 1,
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
            "roc_auc": _safe_auc(np.asarray(y_val), y_prob),
            "total_labeled": int(labeled_mask.sum()),
            "sim_calls_cum": int(sim_calls),
            "sim_time_cum_sec": float(sim_time),
            "train_time_sec": float(train_time),
            "wall_time_cum_sec": float(time.perf_counter() - t0_wall),
        })

        # Select next batch from the unlabeled set
        U = np.where(~labeled_mask)[0]
        if U.size == 0: 
            break # Stop if no unlabeled samples left

        # How many to select this round
        take = min(batch_size, U.size)

        # Select next batch (random or strategy-based)
        if strategy == "random":
            # Random pick when strategy == random
            Q = rng.choice(U, size=take, replace=False)
        else:
            # Score unlabeled pool and take top-K by score
            proba_unl = model.predict_proba(X_pool.iloc[U])
            scores = compute_query_scores(proba_unl, strategy)
            Q = U[np.argsort(scores)[::-1][:take]]

        # Mark selected items as labeled
        labeled_mask[Q] = True

        # Query simulator for each newly selected sample
        for i in Q:
            if np.isnan(y_labels[i]):
                t0 = time.perf_counter()
                lab = query_simulator_cached(X_pool.iloc[i].to_dict())
                sim_time += time.perf_counter() - t0
                sim_calls += 1
                y_labels[i] = 1 if lab == "secure" else 0

    # ---------------- Final summary (KPIs) ----------------
    duration = time.perf_counter() - t0_wall
    last = metrics[-1] if metrics else {}

    # Collect run-level KPIs
    kpi = {
        "final_iteration": int(last.get("iteration", 0)),
        "final_accuracy": float(last.get("accuracy", float("nan"))),
        "final_auc": float(last.get("roc_auc", float("nan"))),
        "total_labeled": int(last.get("total_labeled", int(labeled_mask.sum()))),
        "sim_calls": int(sim_calls),
        "sim_time_sec_measured": float(sim_time),
        "runtime_wall_sec": float(duration),
        "strategy": strategy,
        "initial_size": int(initial_size),
        "batch_size": int(batch_size),
        "iterations_requested": int(iterations),
    }

    # Optional "estimated" simulator time if a known avg is supplied
    if avg_sim_time_sec is not None:
        kpi["sim_time_sec_estimated"] = float(sim_calls) * float(avg_sim_time_sec)

    # Return full outputs: per-iteration metrics, total time, and KPI summary
    return metrics, duration, kpi

# ---------- util: split + diagnostics ----------
def _fmt_counts(y) -> Tuple[str, Dict[int, int], int]:
    """
    Helper function to quickly summarize class distribution.

    Args:
        y: iterable of labels (e.g. 0/1 classes).

    Returns:
        tuple:
          - str: nicely formatted summary like "0: 120 (60.0%), 1: 80 (40.0%)"
          - dict: raw counts per class {label: count}
          - int: total number of samples
    """

    c = Counter(y) # count occurrences of each class
    tot = sum(c.values()) # total samples
    s = ", ".join(f"{k}: {c[k]} ({c[k]/tot:.2%})" for k in sorted(c)) 
    return s, c, tot

def _select_feature_columns(df: pd.DataFrame) -> list:
    """
    Select which columns should be used as features for training.

    Rules:
      1) Prefer only columns that start with prefixes:
           - "load_", "gen_", "sgen_"
         (typical feature naming in power grid data).
      2) If no such columns are found, fall back to:
           - all columns except known non-features
             ("timestamp", "status", "secure", "label").

    Args:
        df: pandas DataFrame with both features and metadata columns.

    Returns:
        list: names of columns to use as features.
    """

    drop = [c for c in ["timestamp","status","secure","label"] if c in df.columns]
    allow_pref = ("load_", "gen_", "sgen_")
    feats = [c for c in df.columns if any(c.startswith(p) for p in allow_pref)]
    if not feats:
        feats = [c for c in df.columns if c not in drop]
        print("[WARN] whitelist 0 cols; using fallback (all except labels/timestamp).")
    return feats

def prepare_online_split(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Prepare POOL (unlabeled) and VAL (validation) splits for active learning.

    Logic:
      - The function always expects a column "status" with labels ("secure"/"insecure").
      - If the dataframe has a 'timestamp' column:
          * Sort rows chronologically.
          * Use the earliest (1 - test_size) fraction as POOL (to simulate "future unseen" data).
          * Use the most recent test_size fraction as VAL.
          * This simulates a realistic, time-aware scenario.
      - If there is no 'timestamp':
          * Perform a random stratified split based on 'status' so that class ratios are preserved.

    Notes:
      - The POOL labels (y_pool_dummy) are always initialized to zeros,
        because in online AL setting we assume the "oracle" (simulator)
        will provide labels later when queried.
      - VAL labels (y_val) are kept, because they are used only for evaluation.
      - Timestamps are only kept temporarily for debugging/diagnostics,
        but not returned.

    Args:
        df: DataFrame with features + metadata columns (must contain 'status').
        test_size: fraction of samples used for validation (0 < test_size < 1).
        seed: random seed for reproducibility (only used if no timestamp column).

    Returns:
        X_pool (pd.DataFrame): feature matrix of unlabeled pool
        y_pool_dummy (pd.Series): placeholder labels for pool (all zeros)
        X_val (pd.DataFrame): feature matrix of validation set
        y_val (pd.Series): true labels (0/1) for validation set
    """

    # Ensure we have the "status" column
    if "status" not in df.columns:
        raise ValueError("Missing 'status' column (expected 'secure'/'insecure').")
    
    # Map labels: "secure" → 1, "insecure" → 0
    y_all = df["status"].map({"secure":1, "insecure":0}).astype(int)

    if "timestamp" in df.columns:
        # --- TIME-AWARE SPLIT ---
        # earlier data → POOL, later data → VAL
        d = df.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"]) # ensure timestamp dtype
        d = d.sort_values("timestamp").reset_index(drop=True) # sort chronologically
        feats = _select_feature_columns(d) # choose feature columns
        cut = int(len(d)*(1.0-test_size)) # index where to split
        
        # Split into pool (earlier data) and validation (later data)
        df_pool, df_val = d.iloc[:cut], d.iloc[cut:]
        X_pool, X_val = df_pool[feats].reset_index(drop=True), df_val[feats].reset_index(drop=True)
        y_pool_true = df_pool["status"].map({"secure":1,"insecure":0}).astype(int).reset_index(drop=True)
        y_val = df_val["status"].map({"secure":1,"insecure":0}).astype(int).reset_index(drop=True)
        
        # Pool gets dummy labels (real labels will be provided by simulator later)
        y_pool_dummy = pd.Series(np.zeros(len(X_pool), dtype=int), name="status_binary")
        
        # Keep timestamps only for possible external diagnostics
        ts_pool, ts_val = df_pool["timestamp"].reset_index(drop=True), df_val["timestamp"].reset_index(drop=True)
    else:
        # --- STRATIFIED RANDOM SPLIT ---
        # No timestamps: standard stratified split
        feats = _select_feature_columns(df) # select features
        X_all = df[feats].copy()
        y_dummy_all = pd.Series(np.zeros(len(X_all), dtype=int), name="status_binary")
        
        # Stratified split ensures class proportions are preserved
        X_pool, X_val, y_pool_dummy, _ = train_test_split(
            X_all, y_dummy_all, test_size=test_size, random_state=seed, stratify=y_all
        )
        y_pool_true = y_all.iloc[X_pool.index].reset_index(drop=True)
        y_val = y_all.iloc[X_val.index].reset_index(drop=True)
        
        ts_pool = ts_val = None
        # Reset indices for clean alignment
        X_pool, X_val = X_pool.reset_index(drop=True), X_val.reset_index(drop=True)
        y_pool_dummy = y_pool_dummy.reset_index(drop=True)

    # Quick split diagnostics
    pool_s, pool_c, pool_n = _fmt_counts(y_pool_true)
    val_s, val_c, val_n = _fmt_counts(y_val)
    print("\n" + "="*70)
    print("[DIAG] ONLINE SPLIT")
    print(f"POOL   class balance: {pool_s}  | N={pool_n}")
    print(f"VALID. class balance: {val_s}   | N={val_n}")
    if "timestamp" in df.columns:
        pmx, vmn = df["timestamp"].min(), df["timestamp"].max()
        print("TIME CHECK:", "present (sorted split)")
    else:
        print("TIME CHECK: (no timestamps)")

    return X_pool, y_pool_dummy, X_val, y_val

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the Active Learning runner and set sane defaults.

    What this does
    --------------
    - Builds a parser with all knobs needed to reproduce an AL run.
    - Uses paths **relative to the project root** (../data, ../tables) so the script
      works no matter where you launch it from.
    - Returns a populated argparse.Namespace.

    Arguments exposed (flags)
    -------------------------
    --data            : path to the CSV with labeled historical samples
    --strategy        : AL scoring strategy: entropy | uncertainty | margin | random
    --init            : initial number of queried samples (seed set)
    --batch           : number of new samples to query each AL iteration
    --iters           : number of AL iterations
    --test-size       : fraction used for validation (if timestamp: last fraction; else stratified split)
    --seed            : random seed for reproducibility
    --avg-sim-sec     : optional avg simulator time per call (used only for estimated KPI)
    --tables-dir      : output directory for CSV/JSON results

    RandomForest overrides (optional):
    --n_estimators    : number of trees
    --max_depth       : max depth (None = unlimited)
    --min_samples_split : min samples to split an internal node
    --min_samples_leaf  : min samples at a leaf
    --class_weight    : e.g., balanced_subsample (helps with class imbalance)
    --n_jobs          : parallel jobs for RF (−1 = use all cores)

    Returns:
        argparse.Namespace with all parsed arguments.
    """

    p = argparse.ArgumentParser(description="Active Learning (online) runner – one-file.")

    # Compute project-root-relative paths:
    # HERE = this file's folder; BASE = project root (one level up)
    HERE = os.path.dirname(os.path.abspath(__file__))
    BASE = os.path.abspath(os.path.join(HERE, ".."))

    # Core inputs / run configuration
    p.add_argument("--data", type=str, default=os.path.join(BASE, "data", "simulation_security_labels_n-1.csv"))
    p.add_argument("--strategy", choices=["entropy","uncertainty","margin","random"], default="entropy")
    p.add_argument("--init", type=int, default=100)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--avg-sim-sec", type=float, default=None)
    p.add_argument("--tables-dir", type=str, default=os.path.join(BASE, "tables"))

    # RandomForest hyper-parameters (can be overridden via CLI)
    p.add_argument("--n_estimators", type=int, default=600)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--min_samples_split", type=int, default=None)
    p.add_argument("--min_samples_leaf", type=int, default=2)
    p.add_argument("--class_weight", type=str, default="balanced_subsample")
    p.add_argument("--n_jobs", type=int, default=-1)
    
    return p.parse_args()

def main() -> None:
    """
    Orchestrates the full experiment:
      - clear simulator cache
      - load data and prepare splits
      - run AL loop
      - save iteration metrics, KPIs, and metadata with a descriptive slug
    """

    args = parse_args()

    # Ensure no stale oracle results across runs
    _cached_query_simulator_by_key.cache_clear()

    # Load data and make sure output directory exists
    os.makedirs(args.tables_dir, exist_ok=True)
    df = pd.read_csv(args.data)

    # Prepare pool/validation splits
    X_pool, y_pool_dummy, X_val, y_val = prepare_online_split(
        df, test_size=args.test_size, seed=args.seed
    )

    # Collect RF hyperparameters from CLI
    rf_params = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=args.class_weight,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )

    # Run the AL process
    metrics, duration, kpi = run_active_learning(
        X_pool=X_pool, y_pool_dummy=y_pool_dummy, X_val=X_val, y_val=y_val,
        strategy=args.strategy, initial_size=args.init, batch_size=args.batch, iterations=args.iters,
        random_state=args.seed, simulate_on_demand=True, avg_sim_time_sec=args.avg_sim_sec,
        rf_params=rf_params
    )

    # Build a filename slug + metadata (for reproducibility)
    slug, meta = build_run_slug(
        mode="online",
        strategy=args.strategy,
        init=args.init, batch=args.batch, iters=args.iters,
        test_size=args.test_size, seed=args.seed,
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        min_samples_split=args.min_samples_split, min_samples_leaf=args.min_samples_leaf,
        class_weight=args.class_weight,
        avg_sim_sec=args.avg_sim_sec,
    )

    # Persist outputs (iter table, KPI row, meta JSON)
    paths = save_with_meta(
        out_dir=args.tables_dir,
        base_slug=slug,
        iter_df=metrics,
        kpi_row=kpi,
        meta=meta
    )

    print("Saved:", paths)
    print(f"Done in {duration:.2f}s")

if __name__ == "__main__":
    main()
