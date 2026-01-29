# 11_run_offline.py
# Offline baseline: treniramo RandomForest na vseh etiketiranih podatkih

"""
11_run_offline.py
=================

Offline baseline for power-grid security classification.

What this script does
---------------------
- Loads a fully labeled dataset (with a 'status' column: 'secure'/'insecure').
- Splits data into train/test (random, stratified by class).
- Trains a RandomForest **once** (single run) or over a small **grid** of RF settings.
- Evaluates on the test split and collects standard metrics (ACC, F1, ROC AUC, ...).
- Saves results using consistent filenames via `build_run_slug` + `save_with_meta`.

Why this exists
---------------
- Provides a simple, non-active-learning baseline to compare against AL runs.
- Useful for quick sanity checks and for tuning basic RF hyperparameters.

Typical usage
-------------
# Single run:
python 11_run_offline.py --data ../data/simulation_security_labels_n-1.csv --test-size 0.1

# Small grid search over RF hyperparams:
python 11_run_offline.py --grid --test-size 0.1

Outputs
-------
- tables/<slug>.csv         (one-row metrics for single run, or one row per grid combo)
- tables/offline_grid_summary_<timestamp>.csv  (only when --grid is used)
"""

import os, argparse, time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from utils.experiment_filename_namer import build_run_slug, save_with_meta

# ------------------ OFFLINE RUN ------------------
def run_offline(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
    rf_params: dict,
    ) -> dict:
    """
    Train and evaluate a RandomForest on labeled data (offline baseline).

    Steps:
      1) Validate presence of 'status' label column ('secure'/'insecure').
      2) Map labels to binary (secure=1, insecure=0).
      3) Drop non-feature columns (timestamp/status/secure/label).
      4) Stratified train/test split by label.
      5) Train RF, predict on test, compute metrics.

    Args:
        df: full labeled dataframe.
        test_size: fraction for test split (0<test_size<1).
        seed: RNG seed for reproducibility.
        rf_params: dict of RandomForest hyperparameters (overrides).

    Returns:
        dict with metrics and run info (e.g., accuracy, f1, roc_auc, sizes, runtime, rf_params).
    """

    # Ensure labels exist
    if "status" not in df.columns:
        raise ValueError("Missing 'status' column (expected 'secure'/'insecure').")

    # Map textual labels to {secure:1, insecure:0}
    y_all = df["status"].map({"secure":1,"insecure":0}).astype(int)

    # Remove non-feature columns (keep all other columns as features)
    drop_cols = [c for c in ["timestamp","status","secure","label"] if c in df.columns]
    X_all = df.drop(columns=drop_cols, errors="ignore")

    # Stratified split to preserve class balance in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=seed, stratify=y_all
    )

    # Build and train the RandomForest
    model = RandomForestClassifier(random_state=seed, n_jobs=-1, **rf_params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    dur = time.perf_counter() - t0

    # Predict labels and class-1 probabilities on the test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    # Collect standard metrics and simple run diagnostics
    metrics = dict(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, average="macro", zero_division=0),
        recall=recall_score(y_test, y_pred, average="macro", zero_division=0),
        f1=f1_score(y_test, y_pred, average="macro", zero_division=0),
        roc_auc=roc_auc_score(y_test, y_prob),
        n_train=len(X_train),
        n_test=len(X_test),
        runtime_sec=dur,
        rf_params=rf_params,
    )
    return metrics

# ------------------ CLI ------------------
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the offline baseline script.

    Flags:
      --data         : path to CSV with labeled samples
      --test-size    : test split fraction (e.g., 0.1)
      --seed         : random seed
      --n_estimators : number of trees
      --max_depth    : max depth (None = unlimited)
      --min_samples_split : min samples to split a node
      --min_samples_leaf  : min samples at a leaf
      --class_weight : e.g., balanced_subsample
      --n_jobs       : parallelism for RF (-1 = use all cores)
      --grid         : enable small grid search over RF params
      --tables-dir   : where to save outputs
    """

    p = argparse.ArgumentParser(description="Offline baseline (RandomForest).")

    # Project-root-relative defaults (works even if run from another folder)
    HERE = os.path.dirname(os.path.abspath(__file__))
    BASE = os.path.abspath(os.path.join(HERE, ".."))
    
    # Core run settings
    p.add_argument("--data", type=str, default=os.path.join(BASE, "data", "simulation_security_labels_n-1.csv"))
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    
    # RF hyperparams (basic)
    p.add_argument("--n_estimators", type=int, default=600)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--min_samples_leaf", type=int, default=2)
    p.add_argument("--class_weight", type=str, default="balanced_subsample")
    p.add_argument("--n_jobs", type=int, default=-1)
    
    # Grid search option (small, quick sweep)
    p.add_argument("--grid", action="store_true", help="Run small grid search over RF params.")
    p.add_argument("--tables-dir", type=str, default=os.path.join(BASE, "tables"))
    return p.parse_args()

def main():
    """
    Orchestrate single-run or grid search, and save outputs with metadata.

    Flow:
      - Read args and data.
      - (Optional) grid search over a small RF param grid:
          * train/eval for each combo
          * save each result with its own slug and meta
          * also write a summary CSV of all combos
      - Else: single run with provided RF params, then save one row of metrics.
    """

    args = parse_args()
    os.makedirs(args.tables_dir, exist_ok=True)

    # Load the labeled dataset
    df = pd.read_csv(args.data)

    if args.grid:
        # Small RF grid; adjust values to taste
        grid = {
            "n_estimators": [200, 600, 1000],
            "max_depth": [None, 20, 40],
            "min_samples_split": [2, 4],
            "min_samples_leaf": [1, 2],
            "class_weight": ["balanced_subsample"],
        }

        all_rows = []
        for params in ParameterGrid(grid):
            
            # Run one offline experiment with a specific hyperparam combo
            m = run_offline(df, args.test_size, args.seed, params)

            # Build a unique filename slug + metadata for this combo
            slug, meta = build_run_slug(
                mode="offline",
                test_size=args.test_size,
                seed=args.seed,
                n_estimators=params.get("n_estimators"),
                max_depth=params.get("max_depth"),
                min_samples_split=params.get("min_samples_split"),
                min_samples_leaf=params.get("min_samples_leaf"),
                class_weight=params.get("class_weight"),
            )

            # Save this single row of metrics (+ meta sidecar)
            paths = save_with_meta(
                out_dir=args.tables_dir,
                base_slug=slug,
                iter_df=[m],      # en “row” metrik
                kpi_row=None,
                meta=meta
            )
            print("Saved:", paths["iter_csv"])

            # Keep for the grid-level summary table
            row = dict(m)
            row.update(params)
            row["slug"] = slug
            all_rows.append(row)

        # Write grid summary across all combos
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(args.tables_dir, f"offline_grid_summary_{ts}.csv")
        pd.DataFrame(all_rows).to_csv(summary_path, index=False)
        print("Saved offline grid summary:", summary_path)

    else:
        # Single-run RF params from CLI
        rf_params = dict(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=args.class_weight,
            n_jobs=args.n_jobs,
        )
        
        # Train/evaluate once
        m = run_offline(df, args.test_size, args.seed, rf_params)

        # Create a slug + meta for this single run
        slug, meta = build_run_slug(
            mode="offline",
            test_size=args.test_size,
            seed=args.seed,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=args.class_weight,
        )

        # Save the single-row metrics (+ meta)
        paths = save_with_meta(
            out_dir=args.tables_dir,
            base_slug=slug,
            iter_df=[m],      # en “row” metrik
            kpi_row=None,
            meta=meta
        )
        print("Saved:", paths)
        print(m)
        
if __name__ == "__main__":
    main()
