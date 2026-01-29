"""
al_experiment_code.py
=====================

Offline Active Learning (brez klicev simulatorja) — baseline grid.

- Stratified split (stabilnejši razredi)
- RandomForest (class_weight="balanced")
- Varno računanje AUC (_safe_roc_auc)
- Rezultati: CSV + XLSX (multi-sheet: summary, per_iteration)
"""

import os
import time
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from collections import Counter

# ===================== DIAGNOSTICS UTILS =====================
def _fmt_counts(y):
    c = Counter(y)
    total = sum(c.values())
    # poskusi razumno mapo labelov (0/1 ali 'secure'/'insecure')
    keys = sorted(c.keys())
    parts = []
    for k in keys:
        parts.append(f"{k}: {c[k]} ({c[k]/total:.2%})")
    return ", ".join(parts), c, total

def check_split_diagnostics(
    y_pool, y_val,
    ts_pool: Optional[pd.Series] = None,
    ts_val:  Optional[pd.Series] = None,
    title: str = "SPLIT"
):
    """
    Izpiše class balance in, če so timestamps podani, preveri ali je split časovno ločen.

    Parameters
    ----------
    y_pool : array-like
        Labeli v POOL/Train delu (kar ostane za AL učenje).
    y_val : array-like
        Labeli v Validation delu (eval sklop).
    ts_pool : pd.Series or None
        Časi (datetime) poravnani z y_pool (opcijsko).
    ts_val : pd.Series or None
        Časi (datetime) poravnani z y_val (opcijsko).
    title : str
        Ime splita za lepši izpis.
    """
    print("\n" + "="*70)
    print(f"[DIAG] {title}")
    print("-"*70)

    # --- razredi in deleži
    pool_str, pool_counts, pool_n = _fmt_counts(y_pool)
    val_str,  val_counts,  val_n  = _fmt_counts(y_val)
    print(f"POOL   class balance: {pool_str}  | N={pool_n}")
    print(f"VALID. class balance: {val_str}   | N={val_n}")

    # --- sanity: oba razreda prisotna?
    if len(pool_counts) < 2:
        print("WARN: POOL ima < 2 razreda (ne-stratificiran začetni vzorec?).")
    if len(val_counts) < 2:
        print("WARN: VALIDATION ima < 2 razreda (težave pri metrikah kot je AUC).")

    # --- časovna diagnostika (če timestamps)
    if ts_pool is not None and ts_val is not None:
        try:
            ts_pool = pd.to_datetime(ts_pool)
            ts_val  = pd.to_datetime(ts_val)
            pool_min, pool_max = ts_pool.min(), ts_pool.max()
            val_min,  val_max  = ts_val.min(),  ts_val.max()

            print("-"*70)
            print(f"POOL   time range: {pool_min}  →  {pool_max}")
            print(f"VALID. time range: {val_min}   →  {val_max}")

            # striktno časovno ločen split?
            strictly_time_separated = pool_max < val_min
            overlaps = not strictly_time_separated and not (val_max < pool_min)

            if strictly_time_separated:
                print("TIME CHECK: Strictly time-based forward split (brez prekrivanja).")
            elif overlaps:
                print("TIME CHECK: Časovna območja se PREKRIVAJO (ni čist time-based).")
            else:
                # val_max < pool_min (tudi časovno ločen, samo obrnjen red)
                print("TIME CHECK: Time-based (reverse order) brez prekrivanja.")
        except Exception as e:
            print(f"TIME CHECK: ni uspelo parsat timestamps ({e})")
    else:
        print("TIME CHECK: (brez timestampov) — preskočeno.")

    print("="*70)
# =================== END DIAGNOSTICS UTILS ===================

def _safe_roc_auc(y_true, y_prob) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")

def load_dataset(csv_path: str):
    data = pd.read_csv(csv_path)

    # 1) timestamp: parse + sort (če obstaja)
    ts = None
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        data = data.sort_values("timestamp").reset_index(drop=True)
        ts = data["timestamp"].copy()  # shranimo za diagnostiko

    # 2) binarna labela
    data["status_binary"] = data["status"].map({"secure": 1, "insecure": 0})

    # 3) DROP nepotrebnih label-stolpcev + timestamp iz featurjev!
    columns_to_drop = [
        "status",
        "status_binary",
        "max_line_loading_percent_basecase",
        "min_bus_voltage_pu_basecase",
        "max_bus_voltage_pu_basecase",
        "max_line_loading_percent_contingency",
        "min_bus_voltage_pu_contingency",
        "max_bus_voltage_pu_contingency",
        "timestamp",  # <--- pomembno: timestamp ne sme ostati v X
    ]
    X = data.drop(columns=columns_to_drop, errors="ignore")
    y = data["status_binary"]
    return X, y, ts  # vrnemo ts za time-based split in diagnostiko

def split_dataset(X, y, test_size: float = 0.1, random_state: int = 42, timestamps: pd.Series = None):
    

    # 1) random stratified
    Xp_r, Xv_r, yp_r, yv_r = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2) sequential (ker smo v load_dataset že sortali po času, to je “časovno naprej”)
    split_idx = int((1 - test_size) * len(X))
    Xp_s, Xv_s = X.iloc[:split_idx], X.iloc[split_idx:]
    yp_s, yv_s = y.iloc[:split_idx], y.iloc[split_idx:]

    splits = {
        "random": (Xp_r, Xv_r, yp_r, yv_r),         # brez reset_index tukaj!
        "sequential": (Xp_s, Xv_s, yp_s, yv_s),     # brez reset_index tukaj!
    }

    # 3) explicit time-based split (če imamo timestamps)
    if timestamps is not None and timestamps.notna().any():
        cutoff = timestamps.quantile(1 - test_size)
        mask_pool = timestamps < cutoff
        Xp_t, Xv_t = X[mask_pool], X[~mask_pool]
        yp_t, yv_t = y[mask_pool], y[~mask_pool]
        splits["time"] = (Xp_t, Xv_t, yp_t, yv_t)

    return splits

def compute_query_scores(proba: np.ndarray, strategy: str) -> np.ndarray:
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
    ) -> Tuple[List[Dict[str, float]], float]:
    """
    Run the active learning loop.
    """
    
    rng = np.random.default_rng(random_state)
    n_samples = len(X_pool)
    labeled_mask = np.zeros(n_samples, dtype=bool)
    labeled_mask[rng.choice(n_samples, size=min(initial_size, n_samples), replace=False)] = True

    metrics_per_iteration = []
    start_time = time.perf_counter()

    for it in tqdm(range(iterations), desc=f"AL offline ({strategy})"):
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=random_state,
        )
        X_train = X_pool[labeled_mask]
        y_train = y_pool[labeled_mask]
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        metrics_per_iteration.append({
            "iteration": it + 1,
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
            "roc_auc": _safe_roc_auc(y_val, y_prob),
        })

        if not (~labeled_mask).any():
            break

        unlabeled_indices = np.where(~labeled_mask)[0]
        if strategy == "random":
            query_indices = rng.choice(unlabeled_indices, size=min(batch_size, len(unlabeled_indices)), replace=False)
        else:
            proba_unl = model.predict_proba(X_pool.iloc[unlabeled_indices])
            scores = compute_query_scores(proba_unl, strategy)
            top_local = np.argsort(scores)[::-1][:batch_size]
            query_indices = unlabeled_indices[top_local]

        labeled_mask[query_indices] = True

    duration = time.perf_counter() - start_time
    return metrics_per_iteration, duration

def ensure_directories(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def run_experiment_grid(
    csv_path: str,
    strategies: List[str],
    initial_sizes: List[int],
    batch_sizes: List[int],
    iteration_counts: List[int],
    test_size: float = 0.1,
    random_state: int = 42,
    figures_dir: str = "figures",
    tables_dir: str = "tables",
    avg_sim_time_sec: Optional[float] = None,
    ) -> pd.DataFrame:
    """
    Run a grid of AL experiments and save summary + per_iteration to CSV+XLSX.
    """

    ensure_directories(figures_dir, tables_dir)

    X, y, ts = load_dataset(csv_path)
    splits = split_dataset(X, y, test_size=test_size, random_state=random_state, timestamps=ts)

    for name, (X_pool, X_val, y_pool, y_val) in splits.items():
        ts_pool = ts.loc[X_pool.index] if ts is not None else None
        ts_val  = ts.loc[X_val.index]  if ts is not None else None
        check_split_diagnostics(y_pool, y_val, ts_pool=ts_pool, ts_val=ts_val, title=f"OFFLINE | {name}")

        # (po želji) za model lahko resetiraš indekse:
        X_pool_model = X_pool.reset_index(drop=True)
        X_val_model  = X_val.reset_index(drop=True)
        y_pool_model = y_pool.reset_index(drop=True)
        y_val_model  = y_val.reset_index(drop=True)

    timestamp_all = datetime.now().strftime("%Y%m%d_%H%M%S")
    combinations = list(product(initial_sizes, batch_sizes, iteration_counts, splits.items(), strategies))

    results_rows = []
    all_iteration_metrics = []
    start_all = time.perf_counter()

    for init_sz, batch_sz, iters, (split_name, (X_pool, X_val, y_pool, y_val)), strategy in tqdm(
        combinations, desc="Grid experiments"
    ):
        metrics_iter, duration = run_active_learning(
            X_pool, y_pool, X_val, y_val,
            strategy=strategy,
            initial_size=init_sz,
            batch_size=batch_sz,
            iterations=iters,
            random_state=random_state,
        )

        # enrich per-iteration
        for m in metrics_iter:
            m.update({
                "timestamp": timestamp_all,
                "strategy_type": strategy,
                "split_type": split_name,
                "initial_size": init_sz,
                "batch_size": batch_sz,
                "iterations": iters,
                "iteration_id": m["iteration"],
                "total_labeled": init_sz + (m["iteration"] - 1) * batch_sz,
            })
        all_iteration_metrics.extend(metrics_iter)

        accs = [m["accuracy"] for m in metrics_iter]
        aucs = [m["roc_auc"] for m in metrics_iter]
        total_labeled = init_sz + iters * batch_sz

        est_sim_time_used = total_labeled * avg_sim_time_sec if avg_sim_time_sec else None
        est_sim_time_full = (len(X_pool) + len(X_val)) * avg_sim_time_sec if avg_sim_time_sec else None
        est_time_saving = (est_sim_time_full - est_sim_time_used) if (est_sim_time_full and est_sim_time_used) else None

        results_rows.append({
            "timestamp": timestamp_all,
            "strategy_type": strategy,
            "split_type": split_name,
            "iterations": iters,
            "test_size": test_size,
            "initial_size": init_sz,
            "batch_size": batch_sz,
            "total_labeled_samples": total_labeled,
            "accuracy_final": accs[-1] if accs else None,
            "accuracy_mean": float(np.mean(accs)) if accs else None,
            "roc_auc_mean": float(np.mean(aucs)) if aucs else None,
            "duration_train_sec": duration,
            "est_sim_time_used_sec": est_sim_time_used,
            "est_sim_time_full_sec": est_sim_time_full,
            "est_sim_time_saved_sec": est_time_saving,
        })

    duration_all = time.perf_counter() - start_all

    df_results = pd.DataFrame(results_rows)
    df_detailed = pd.DataFrame(all_iteration_metrics)

    os.makedirs(tables_dir, exist_ok=True)
    csv_name = os.path.join(tables_dir, f"active_learning_results_{timestamp_all}.csv")
    df_results.to_csv(csv_name, index=False)

    xlsx_name = os.path.join(tables_dir, f"active_learning_results_{timestamp_all}.xlsx")
    with pd.ExcelWriter(xlsx_name, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, sheet_name="summary", index=False)
        df_detailed.to_excel(writer, sheet_name="per_iteration", index=False)

    detailed_csv_name = os.path.join(tables_dir, f"al_metrics_per_iteration_{timestamp_all}.csv")
    df_detailed.to_csv(detailed_csv_name, index=False)

    print(f"Rezultati shranjeni v: {csv_name}")
    print(f"Excel (summary + per_iteration): {xlsx_name}")
    print(f"Iteracije shranjene v: {detailed_csv_name}")
    print(f"Skupni čas izvajanja: {duration_all:.2f} s")

    return df_results


if __name__ == "__main__":
    # Primer default grid-a
    INITIAL_SIZES = [50, 100, 200]
    BATCH_SIZES = [10, 25, 50]
    ITERATION_COUNTS = [20, 40, 60]
    STRATEGIES = ["uncertainty", "entropy", "margin", "random"]

    HERE = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(os.path.join(HERE, ".."))

    TABLES_DIR = os.path.join(BASE_DIR, "tables")
    FIGURES_DIR = os.path.join(BASE_DIR, "figures")
    DATA_PATH = os.path.join(BASE_DIR, "data", "simulation_security_labels_n-1.csv")

    run_experiment_grid(
        csv_path=DATA_PATH,
        strategies=STRATEGIES,
        initial_sizes=INITIAL_SIZES,
        batch_sizes=BATCH_SIZES,
        iteration_counts=ITERATION_COUNTS,
        test_size=0.1,
        random_state=42,
        figures_dir=FIGURES_DIR,
        tables_dir=TABLES_DIR,
        avg_sim_time_sec=None,  # opcijsko: npr. 1.2
    )
