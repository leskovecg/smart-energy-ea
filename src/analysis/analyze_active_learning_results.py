#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
analysis.py
===========

Analyze Active Learning (AL) results and produce timestamped tables & figures.

What this script does (non-recursive over a single folder)
---------------------------------------------------------
- Scans a **single top-level** results folder (no subfolders) for AL outputs
  (CSV + optional *_kpi.csv + *.meta.json).
- Parses filename stems (end with `_YYYYMMDD_HHMMSS_<hash>`) and enriches them
  with metadata when the sidecar JSON is available.
- Builds **per-iteration** DataFrames, then computes **average curves** across
  seeds per strategy (e.g., Accuracy/AUC/Precision/Recall/F1/FNR vs. total labeled).
- Computes KPIs by strategy:
  * AULC (area under learning curve; for FNR we use AULC of (1-FNR)),
  * TTT (labels needed to reach user-defined metric targets; for FNR we use ≤),
  * Final metric values at the maximum label budget observed.
- Plots:
  * average curves per strategy (with optional std shading) for every metric present,
  * per-seed curves,
  * TTT bar charts for the requested targets.

Outputs (all **timestamped** in the filename)
---------------------------------------------
- Tables (XLSX+CSV) with KPIs and raw/averaged curves saved to `--tables-dir`.
- Figures (PNG): curves and TTT charts saved to `--figures-dir`.

Typical usage
-------------
python analysis.py --tables-dir ../tables --figures-dir ../figures \
  --acc-targets 0.90 0.92 --auc-targets 0.97 0.98 \
  --precision-targets 0.90 --recall-targets 0.90 --f1-targets 0.90 --fnr-targets 0.10
"""

import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional sklearn only for baseline demo (kept as-is; can ignore if unused)
try:
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ------------------------------ Config / regex -------------------------------
STEM_END_RE = re.compile(r".*_(\d{8})_(\d{6})_([0-9a-f]{6,16})$", re.IGNORECASE)

CW_MAP = {
    "cwbsub": "balanced_subsample",
    "cwbal": "balanced",
    "cwnone": "none",
    "cwbal_subsample": "balanced_subsample",
    "cwbalanced": "balanced",
    "cwbalsub": "balanced_subsample",
    "cwbalanceds": "balanced_subsample",
}

# Column candidates (case-insensitive)
ITER_COLS = ["iteration", "iter", "step", "round"]
ACC_COLS = ["accuracy", "acc"]
AUC_COLS = ["roc_auc", "auc", "auroc"]
PREC_COLS = ["precision", "prec"]
REC_COLS = ["recall", "tpr", "sensitivity"]
F1_COLS = ["f1", "f1_score"]
FNR_COLS = ["fnr", "miss_rate", "false_negative_rate"]

LABELED_COLS = ["total_labeled", "labeled", "labels", "sim_calls", "sim_calls_cum"]
SIMTIME_COLS = ["sim_time_sec_measured", "sim_time_cum_sec", "sim_time_sec", "sim_time"]


# ------------------------------ Small helpers --------------------------------

def require_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def _lower_map(cols: List[str]) -> Dict[str, str]:
    return {c.lower(): c for c in cols}

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lmap = _lower_map(df.columns.tolist())
    for cand in candidates:
        if cand.lower() in lmap:
            return lmap[cand.lower()]
    return None

def _read_csv_any(path: str) -> Optional[pd.DataFrame]:
    for sep in [",", ";", "\t", "|"]:
        try:
            return pd.read_csv(path, sep=sep)
        except Exception:
            continue
    return None

def _parse_stem(stem: str) -> Dict[str, Optional[object]]:
    out = {
        "mode": None, "strategy": None,
        "initial_size": None, "batch_size": None, "iterations": None,
        "test_ssize": None, "seed": None,
        "n_estimators": None, "max_depth": None,
        "min_samples_split": None, "min_samples_leaf": None,
        "class_weight": None, "timestamp": None, "hash": None, "stem": stem,
    }
    m = STEM_END_RE.match(stem)
    if m:
        out["timestamp"] = f"{m.group(1)}_{m.group(2)}"
        out["hash"] = m.group(3)

    toks = stem.split("_")
    if len(toks) >= 2:
        out["mode"] = toks[0]
        out["strategy"] = toks[1]

    i = 0
    while i < len(toks):
        t = toks[i]

        def take_int(prefix: str) -> Optional[int]:
            if t.startswith(prefix) and len(t) > len(prefix):
                try:
                    return int(t[len(prefix):])
                except Exception:
                    return None
            return None

        if t.startswith("i"):
            out["initial_size"] = take_int("i")
        elif t.startswith("b"):
            out["batch_size"] = take_int("b")
        elif t.startswith("it"):
            out["iterations"] = take_int("it")
        elif t == "ts" or t.startswith("ts"):
            if "_" in t:
                a, b = t[2:].split("_", 1)
                try:
                    out["test_ssize"] = float(f"{int(a)}.{int(b)}")
                except Exception:
                    pass
            else:
                try:
                    a = int(t[2:])
                    if i + 1 < len(toks):
                        b = int(toks[i + 1])
                        out["test_ssize"] = float(f"{a}.{b}")
                        i += 1
                except Exception:
                    pass
        elif t.startswith("s"):
            out["seed"] = take_int("s")
        elif t.startswith("ne"):
            out["n_estimators"] = take_int("ne")
        elif t.startswith("md"):
            out["max_depth"] = take_int("md")
        elif t.startswith("mss"):
            out["min_samples_split"] = take_int("mss")
        elif t.startswith("msl"):
            out["min_samples_leaf"] = take_int("msl")
        elif t.startswith("cw"):
            out["class_weight"] = CW_MAP.get(t, t.replace("cw", "", 1))
        i += 1

    return out


# ---------------------------- File collection --------------------------------

def collect_triplets_nonrecursive(folder: str) -> Dict[str, Dict[str, Optional[str]]]:
    trip: Dict[str, Dict[str, Optional[str]]] = {}
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isdir(path):
            continue
        if not (f.endswith(".csv") or f.endswith(".meta.json")):
            continue

        if f.endswith(".meta.json"):
            stem = f[:-10]
            if not STEM_END_RE.match(stem):
                continue
            entry = trip.setdefault(stem, {"main": None, "kpi": None, "meta": None})
            entry["meta"] = path
        elif f.endswith(".csv"):
            if f.endswith("_kpi.csv"):
                stem = f[:-8]
                if not STEM_END_RE.match(stem):
                    continue
                entry = trip.setdefault(stem, {"main": None, "kpi": None, "meta": None})
                entry["kpi"] = path
            else:
                stem = f[:-4]
                if not STEM_END_RE.match(stem):
                    continue
                entry = trip.setdefault(stem, {"main": None, "kpi": None, "meta": None})
                entry["main"] = path
    return trip


# --------------------------- Per-iteration build ------------------------------

def build_per_iteration_df(stem: str,
                           main_csv: Optional[str],
                           meta_json: Optional[str]) -> Optional[pd.DataFrame]:
    if not main_csv:
        return None
    df = _read_csv_any(main_csv)
    if df is None or df.empty:
        return None

    info = _parse_stem(stem)
    if meta_json and os.path.isfile(meta_json):
        try:
            with open(meta_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for k, v in {
                "initial_size": "init",
                "batch_size": "batch",
                "iterations": "iters",
                "test_ssize": "test_size",
                "seed": "seed",
                "n_estimators": "n_estimators",
                "max_depth": "max_depth",
                "min_samples_split": "min_samples_split",
                "min_samples_leaf": "min_samples_leaf",
                "class_weight": "class_weight",
            }.items():
                if v in meta and meta[v] is not None:
                    info[k] = meta[v]
        except Exception:
            pass

    iter_col = _pick_col(df, ITER_COLS)
    tl_col = _pick_col(df, LABELED_COLS)
    sim_col = _pick_col(df, SIMTIME_COLS)

    # metric columns (optional)
    acc_col  = _pick_col(df, ACC_COLS)
    auc_col  = _pick_col(df, AUC_COLS)
    prec_col = _pick_col(df, PREC_COLS)
    rec_col  = _pick_col(df, REC_COLS)
    f1_col   = _pick_col(df, F1_COLS)
    fnr_col  = _pick_col(df, FNR_COLS)

    out = pd.DataFrame()
    out["iteration"] = df[iter_col].values if iter_col else np.arange(len(df))
    if tl_col:
        out["total_labeled"] = pd.to_numeric(df[tl_col], errors="coerce")
    else:
        init = int(info.get("initial_size") or 0)
        batch = int(info.get("batch_size") or 1)
        it0 = out["iteration"].min()
        out["total_labeled"] = init + (out["iteration"] - it0) * batch

    # attach metrics (if present)
    out["accuracy"] = pd.to_numeric(df[acc_col],  errors="coerce") if acc_col  else np.nan
    out["roc_auc"]  = pd.to_numeric(df[auc_col],  errors="coerce") if auc_col  else np.nan
    out["precision"]= pd.to_numeric(df[prec_col], errors="coerce") if prec_col else np.nan
    out["recall"]   = pd.to_numeric(df[rec_col],  errors="coerce") if rec_col  else np.nan
    out["f1"]       = pd.to_numeric(df[f1_col],   errors="coerce") if f1_col   else np.nan
    out["fnr"]      = pd.to_numeric(df[fnr_col],  errors="coerce") if fnr_col  else np.nan

    out["sim_time_cum_sec"] = pd.to_numeric(df[sim_col], errors="coerce") if sim_col else np.nan

    out["strategy"] = str(info.get("strategy") or "unknown").lower()
    out["seed"] = int(info.get("seed") or 0)
    out["stem"] = stem
    out["source_file"] = os.path.basename(main_csv)
    out["rf_n_estimators"] = info.get("n_estimators")
    out["rf_max_depth"] = info.get("max_depth")
    out["rf_min_samples_split"] = info.get("min_samples_split")
    out["rf_min_samples_leaf"] = info.get("min_samples_leaf")
    out["rf_class_weight"] = info.get("class_weight")

    return out


# -------------------------- Aggregation + KPIs --------------------------------

def group_average(df: pd.DataFrame, y_col: str, x_col: str = "total_labeled",
                  group_cols: List[str] = ["strategy"]) -> pd.DataFrame:
    if df.empty or y_col not in df.columns or x_col not in df.columns:
        return pd.DataFrame(columns=group_cols + [x_col, y_col, f"{y_col}_std", "n"])
    g = (df.groupby(group_cols + [x_col])[y_col]
           .agg(["mean", "std", "count"])
           .reset_index()
           .rename(columns={"mean": y_col, "std": f"{y_col}_std", "count": "n"}))
    return g

def time_to_target(df_avg: pd.DataFrame, metric: str, target: float, mode: str = "ge") -> float:
    """
    Return the smallest x (labels) where:
      - mode="ge": metric >= target
      - mode="le": metric <= target
    NaN if never reached.
    """
    if df_avg.empty or metric not in df_avg.columns:
        return np.nan
    s = df_avg.sort_values("total_labeled")
    if mode == "le":
        hit = s[s[metric] <= target]
    else:
        hit = s[s[metric] >= target]
    if hit.empty:
        return np.nan
    return float(hit.iloc[0]["total_labeled"])

def aulc(df_avg: pd.DataFrame, metric: str, x_col: str = "total_labeled", invert: bool = False) -> float:
    """
    Area under learning curve (normalized by x-range).
    If invert=True (e.g., for FNR), we integrate (1 - metric) so that higher is better.
    """
    if df_avg.empty or metric not in df_avg.columns or x_col not in df_avg.columns:
        return np.nan
    s = df_avg.sort_values(x_col)
    x = s[x_col].values
    y = s[metric].values
    if invert:
        y = 1.0 - y
    if len(x) < 2 or x.max() == x.min():
        return np.nan
    area = np.trapz(y, x)
    return float(area / (x.max() - x.min()))


# -------------------------------- Plotting -----------------------------------

def plot_lines(df_avg: pd.DataFrame, y_col: str, out_path: str, y_label: str, title: str):
    plt.figure(figsize=(7, 4.2))
    if df_avg.empty:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        for strat in sorted(df_avg["strategy"].unique()):
            sub = df_avg[df_avg["strategy"] == strat].sort_values("total_labeled")
            x = sub["total_labeled"].values
            y = sub[y_col].values
            ystd = sub.get(f"{y_col}_std", pd.Series(np.zeros_like(y))).values
            plt.plot(x, y, label=strat)
            if np.any(np.nan_to_num(ystd) > 0):
                lo = y - ystd
                hi = y + ystd
                plt.fill_between(x, lo, hi, alpha=0.15, linewidth=0)
    plt.xlabel("Total labeled samples")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_lines_per_seed(df: pd.DataFrame, y_col: str, out_path: str, y_label: str, title: str):
    plt.figure(figsize=(8, 5))
    if df.empty:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        for strat in sorted(df["strategy"].unique()):
            sub_s = df[df["strategy"] == strat]
            for seed in sorted(sub_s["seed"].unique()):
                sub = sub_s[sub_s["seed"] == seed].sort_values("total_labeled")
                plt.plot(sub["total_labeled"].values,
                         sub[y_col].values,
                         label=f"{strat} (seed {seed})",
                         alpha=0.7)
    plt.xlabel("Total labeled samples")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# --------------------------- Baseline comparison ------------------------------
# (unchanged; optional)

def _clean_label_series(y: pd.Series) -> np.ndarray:
    if y.dtype == object:
        vset = set(str(v).strip().lower() for v in y.unique())
        if vset <= {"secure", "insecure"}:
            return y.astype(str).str.strip().str.lower().map({"secure": 0, "insecure": 1}).to_numpy()
        uniq = {val: idx for idx, val in enumerate(sorted(vset))}
        return y.astype(str).map(lambda v: uniq[str(v).strip().lower()]).to_numpy()
    return y.to_numpy()

def _prepare_features_df(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    X = df.drop(columns=[label_col], errors="ignore").copy()
    time_like = [c for c in X.columns if c.lower() in {"timestamp", "time", "datetime", "date"}]
    X = X.drop(columns=time_like, errors="ignore")
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = [c for c in X.columns if c not in num_cols]
    if num_cols:
        X[num_cols] = X[num_cols].apply(lambda s: s.fillna(s.median()))
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X

def _load_baseline_data(args) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if args.baseline_csv:
        df = pd.read_csv(args.baseline_csv)
        if args.label_col.lower() == "none":
            return None
        if args.label_col not in df.columns:
            raise ValueError(f"Label column '{args.label_col}' not found in baseline CSV.")
        y = _clean_label_series(df[args.label_col])
        Xdf = _prepare_features_df(df, args.label_col)
        nunique = Xdf.nunique()
        keep = nunique[nunique > 1].index.tolist()
        Xdf = Xdf[keep]
        X = Xdf.to_numpy(dtype=np.float32, copy=True)
        return X, y

    if args.baseline_npy_x and args.baseline_npy_y:
        X = np.load(args.baseline_npy_x)
        y = np.load(args.baseline_npy_y)
        return X, y

    return None

def _rf_params_from_row(row: pd.Series) -> Dict[str, Union[int, str, None]]:
    cw = row.get("rf_class_weight")
    if isinstance(cw, str) and cw.lower() in ["none", ""]:
        cw = None
    return {
        "n_estimators": int(row.get("rf_n_estimators") or 100),
        "max_depth": None if pd.isna(row.get("rf_max_depth")) else int(row.get("rf_max_depth")),
        "min_samples_split": int(row.get("rf_min_samples_split") or 2),
        "min_samples_leaf": int(row.get("rf_min_samples_leaf") or 1),
        "class_weight": cw,
    }

def baseline_random_same_budget(df_runs_last: pd.DataFrame,
                                args,
                                metric: str = "accuracy") -> Optional[pd.DataFrame]:
    if not SKLEARN_AVAILABLE:
        print("[INFO] sklearn not available, baseline skipped.")
        return None
    data = _load_baseline_data(args)
    if data is None:
        print("[INFO] baseline data not provided, baseline skipped.")
        return None
    X, y = data
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatched baseline shapes: X={X.shape}, y={y.shape}")

    best_idx = df_runs_last[metric].idxmax()
    best = df_runs_last.loc[best_idx]
    label_budget = int(best["total_labeled"])
    test_seed = int(args.baseline_test_seed)
    test_size = float(args.baseline_test_size)
    rf_params = _rf_params_from_row(best)

    from sklearn.model_selection import StratifiedShuffleSplit
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=test_seed)
    train_idx, test_idx = next(sss_test.split(X, y))
    X_train_all, X_test = X[train_idx], X[test_idx]
    y_train_all, y_test = y[train_idx], y[test_idx]

    results = []
    for s in args.baseline_seeds:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=label_budget, random_state=int(s))
        sub_idx, _ = next(sss.split(X_train_all, y_train_all))
        X_sub, y_sub = X_train_all[sub_idx], y_train_all[sub_idx]
        clf = RandomForestClassifier(**rf_params, random_state=int(s), n_jobs=-1)
        clf.fit(X_sub, y_sub)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
        results.append({"type": "baseline_random", "seed": int(s), "accuracy": acc, "roc_auc": auc})

    results.append({"type": "AL_best", "seed": int(best["seed"]),
                    "accuracy": float(best["accuracy"]), "roc_auc": float(best["roc_auc"])})
    return pd.DataFrame(results)


# ---------------------------------- Main -------------------------------------

def main(args):
    tables_dir = os.path.abspath(args.tables_dir)
    figures_dir = os.path.abspath(args.figures_dir)
    require_dir(figures_dir)

    now_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    trip = collect_triplets_nonrecursive(tables_dir)
    print(f"[INFO] Found {len(trip)} runs (stems):")
    for stem in sorted(trip.keys()):
        m = STEM_END_RE.match(stem)
        h = m.group(3) if m else "???"
        print(f"  - {stem}   (hash={h})")

    pit_rows = []
    for stem, paths in sorted(trip.items()):
        row = build_per_iteration_df(stem, paths.get("main"), paths.get("meta"))
        if row is not None and not row.empty:
            pit_rows.append(row)
    if not pit_rows:
        print("[WARN] No valid per-iteration CSVs found.")
        return

    df = pd.concat(pit_rows, ignore_index=True)

    # Average curves for all metrics we may have
    metric_defs = [
        ("accuracy", "Accuracy", ACC_COLS, False),
        ("roc_auc", "ROC AUC", AUC_COLS, False),
        ("precision", "Precision", PREC_COLS, False),
        ("recall", "Recall", REC_COLS, False),
        ("f1", "F1 score", F1_COLS, False),
        ("fnr", "FNR", FNR_COLS, True),   # invert=True for AULC and TTT<=
    ]

    avg_curves: Dict[str, pd.DataFrame] = {}
    for mname, _, _, _ in metric_defs:
        if mname in df.columns:
            avg = group_average(df, mname)
            if not avg.empty:
                avg_curves[mname] = avg

    # KPIs per strategy
    kpi_rows = []
    strategies = sorted(df["strategy"].unique())
    for strat in strategies:
        row = {"strategy": strat}

        for mname, _, _, invert in metric_defs:
            if mname not in avg_curves:
                continue
            curve = avg_curves[mname][avg_curves[mname]["strategy"] == strat]
            if curve.empty:
                continue

            # AULC
            row[f"AULC_{mname if mname!='roc_auc' else 'auc'}"] = aulc(curve, mname, invert=invert)

            # Targets → TTT
            targets = []
            mode = "ge"
            if mname == "accuracy":
                targets = args.acc_targets
            elif mname == "roc_auc":
                targets = args.auc_targets
            elif mname == "precision":
                targets = args.precision_targets
            elif mname == "recall":
                targets = args.recall_targets
            elif mname == "f1":
                targets = args.f1_targets
            elif mname == "fnr":
                targets = args.fnr_targets
                mode = "le"  # lower is better

            for t in targets:
                suffix = f"{t:.2f}".rstrip("0").rstrip(".")
                key_base = mname if mname != "roc_auc" else "auc"
                row[f"TTT_{key_base}@{suffix}"] = time_to_target(curve, mname, float(t), mode=mode)

            # Final @ max TL
            row[f"Final{('AUC' if mname=='roc_auc' else mname.capitalize())}@maxTL"] = \
                float(curve.sort_values("total_labeled").iloc[-1][mname])

        kpi_rows.append(row)
    kpis = pd.DataFrame(kpi_rows)

    # Last point per run
    df_runs_last = (df.sort_values(["stem", "total_labeled"])
                      .groupby("stem", as_index=False)
                      .tail(1))

    # Save tables (timestamped)
    out_xlsx = os.path.join(tables_dir, f"paper_kpis_summary_{now_stamp}.xlsx")
    out_csv  = os.path.join(tables_dir, f"paper_kpis_summary_{now_stamp}.csv")
    with pd.ExcelWriter(out_xlsx) as writer:
        kpis.to_excel(writer, sheet_name="online_kpis", index=False)
        for mname, title, _, _ in metric_defs:
            if mname in avg_curves:
                avg_curves[mname].to_excel(writer, sheet_name=f"{mname}_avg_curve", index=False)
        df_runs_last.to_excel(writer, sheet_name="per_run_last_points", index=False)
        df.to_excel(writer, sheet_name="online_per_iteration_raw", index=False)
    kpis.to_csv(out_csv, index=False)

    # Average curve plots (timestamped)
    for mname, title, _, _ in metric_defs:
        if mname not in avg_curves:
            continue
        plot_lines(avg_curves[mname], mname,
                   os.path.join(figures_dir, f"{mname}_vs_total_labeled_{now_stamp}.png"),
                   title, f"{title} vs. total labeled (avg across seeds)")

    # Per-seed plots (timestamped)
    for mname, title, _, _ in metric_defs:
        if mname not in df.columns:
            continue
        plot_lines_per_seed(df, mname,
                            os.path.join(figures_dir, f"{mname}_vs_total_labeled_per_seed_{now_stamp}.png"),
                            title, f"{title} vs. total labeled (per seed)")

    # TTT bar charts for requested targets (timestamped)
    def _plot_ttt(col_prefix: str, targets: List[float], label_fmt: str, file_stub: str):
        if kpis.empty:
            return
        for t in targets:
            suffix = f"{t:.2f}".rstrip("0").rstrip(".")
            col = f"{col_prefix}@{suffix}"
            if col not in kpis.columns:
                continue
            s = (kpis.set_index("strategy")[col].sort_values(na_position="last"))
            plt.figure(figsize=(6.5, 4.0))
            s.plot(kind="bar")
            plt.ylabel(label_fmt.format(t))
            plt.title(f"TTT ({label_fmt.format(t)})")
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f"{file_stub}_{suffix}_{now_stamp}.png"), dpi=200)
            plt.close()

    _plot_ttt("TTT_accuracy", args.acc_targets, "Labels to reach Accuracy ≥ {}", "ttt_accuracy")
    _plot_ttt("TTT_auc",      args.auc_targets, "Labels to reach AUC ≥ {}",      "ttt_auc")
    _plot_ttt("TTT_precision",args.precision_targets, "Labels to reach Precision ≥ {}", "ttt_precision")
    _plot_ttt("TTT_recall",   args.recall_targets,    "Labels to reach Recall ≥ {}",    "ttt_recall")
    _plot_ttt("TTT_f1",       args.f1_targets,        "Labels to reach F1 ≥ {}",        "ttt_f1")
    # For FNR: note we used ≤ targets. Label string indicates that.
    _plot_ttt("TTT_fnr",      args.fnr_targets,       "Labels to reach FNR ≤ {}",       "ttt_fnr")

    print(f"[OK] KPI tables: {out_xlsx} / {out_csv}")
    print(f"[OK] Figures written to: {figures_dir} (timestamp {now_stamp})")


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    BASE = os.path.abspath(os.path.join(HERE, ".."))
    DEFAULT_TABLES = os.path.join(BASE, "tables")
    DEFAULT_FIGS = os.path.join(BASE, "figures")

    ap = argparse.ArgumentParser(
        description="Analyze AL online results (non-recursive) with multi-metric KPIs and timestamped outputs."
    )
    ap.add_argument("--tables-dir", type=str, default=DEFAULT_TABLES, help="Folder with online_* files (top-level only).")
    ap.add_argument("--figures-dir", type=str, default=DEFAULT_FIGS, help="Folder for plots.")

    # Targets (you can pass multiples)
    ap.add_argument("--acc-targets", type=float, nargs="*", default=[0.90, 0.92], help="Accuracy targets for TTT (≥).")
    ap.add_argument("--auc-targets", type=float, nargs="*", default=[0.97, 0.98], help="AUC targets for TTT (≥).")
    ap.add_argument("--precision-targets", type=float, nargs="*", default=[0.90], help="Precision targets for TTT (≥).")
    ap.add_argument("--recall-targets", type=float, nargs="*", default=[0.90], help="Recall targets for TTT (≥).")
    ap.add_argument("--f1-targets", type=float, nargs="*", default=[0.90], help="F1 targets for TTT (≥).")
    ap.add_argument("--fnr-targets", type=float, nargs="*", default=[0.10], help="FNR targets for TTT (≤).")

    # Optional baseline (unchanged)
    ap.add_argument("--enable-baseline", action="store_true", help="Optional random baseline @ same label budget.")
    ap.add_argument("--baseline-csv", type=str, default="", help="CSV with features + label column (see --label-col).")
    ap.add_argument("--label-col", type=str, default="target", help="Label column in --baseline-csv.")
    ap.add_argument("--baseline-npy-x", type=str, default="", help="Path to X.npy alternative.")
    ap.add_argument("--baseline-npy-y", type=str, default="", help="Path to y.npy alternative.")
    ap.add_argument("--baseline-test-seed", type=int, default=42, help="Seed for test split.")
    ap.add_argument("--baseline-test-size", type=float, default=0.1, help="Test size fraction.")
    ap.add_argument("--baseline-seeds", type=int, nargs="*", default=[42, 1337, 7, 11, 101], help="Seeds for random subsampling.")

    args = ap.parse_args()
    main(args)
