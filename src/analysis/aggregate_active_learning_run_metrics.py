#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aggregate_metrics.py
====================

Aggregate Active Learning experiment outputs into a single master table.

What this script does
---------------------
- Scans a *single* results folder (non-recursive) and groups files by a shared stem:
  * <stem>.csv        : per-iteration metrics (main)
  * <stem>_kpi.csv    : one-row summary with final KPIs
  * <stem>.meta.json  : metadata (exact params used)
  The stem must end with: `_YYYYMMDD_HHMMSS_<hexhash>`.

- For each triplet:
  1) Parse key parameters from the filename stem (`parse_filename_stem`).
  2) Optionally pull richer metadata from the JSON sidecar (`pull_meta_from_json`)
     and prefer those values over the filename tokens (`merge_preferring_meta`).
  3) Pick a primary score column heuristically (`choose_metric`).
  4) Compute simple descriptive stats on that series (`compute_stats`).
  5) Extract final KPIs from the KPI CSV (or fall back to the main CSV) (`extract_kpis`).
  6) Build one consolidated row (`build_master_row`).

- Finally, writes timestamped files:
  * `master_results_YYYYMMDD_HHMMSS.xlsx` (sheet: "results")
  * `master_results_YYYYMMDD_HHMMSS.csv`

Why this is useful
------------------
- Produces one analysis-ready table across many runs.
- Robust to small naming variations (class_weight abbreviations, metric names).
- Filename parsing + metadata sidecar ensure reproducibility.

Assumptions / conventions
-------------------------
- Filenames follow the slug pattern produced by your runner (ending with timestamp + hash).
- Main CSV has an iteration axis and at least one numeric metric column.
- KPI CSV (optional) contains final-row KPIs; if missing, we fall back to main CSV.
- Non-recursive: only files directly under the specified folder are considered.

Typical usage
-------------
$ python aggregate_metrics.py ./tables
# optional custom outputs:
$ python aggregate_metrics.py ./tables --outfile my_master.xlsx --outcsv my_master.csv
"""

import os
import re
import json
import argparse
from typing import Dict, Optional, List

import pandas as pd
import numpy as np

# =========================
# Config / mapping
# =========================

# Map class_weight abbreviations found in slugs back to full names
CW_MAP = {
    "cwbsub": "balanced_subsample",
    "cwbal": "balanced",
    "cwnone": "none",
    "cwbal_subsample": "balanced_subsample",
    "cwbalanced": "balanced",
    "cwbalsub": "balanced_subsample",
    "cwbalanceds": "balanced_subsample",
}

# Preferred metric names in order; first match wins
PREFERRED_METRICS = [
    "accuracy", "acc",
    "roc_auc", "auc", "auroc",
    "f1", "f1_score",
]

# Possible names for the iteration column
ITERATION_COL_CANDIDATES = ["iteration", "iter", "step", "round"]

# Valid stem must end with: _YYYYMMDD_HHMMSS_<hexhash>
STEM_END_RE = re.compile(r".*_(\d{8})_(\d{6})_([0-9a-f]{6,16})$", re.IGNORECASE)

# --- NEW: alternate column name candidates for classification metrics
PREC_CANDS   = ["final_precision", "precision", "prec", "ppv"]
RECALL_CANDS = ["final_recall", "recall", "tpr", "sensitivity", "rec"]
F1_CANDS     = ["final_f1", "f1", "f1_score"]
FNR_CANDS    = ["final_fnr", "fnr", "miss_rate"]

# If KPI/main CSVs contain confusion matrix counts, use them to derive metrics
TP_CANDS = ["tp", "true_positives", "TP"]
FN_CANDS = ["fn", "false_negatives", "FN"]
FP_CANDS = ["fp", "false_positives", "FP"]
TN_CANDS = ["tn", "true_negatives", "TN"]


# =========================
# Helpers
# =========================

def _lower_map(cols: List[str]) -> Dict[str, str]:
    """Return a mapping {lower_name: original_name} to ease case-insensitive lookups."""
    return {c.lower(): c for c in cols}


def choose_metric(df: pd.DataFrame) -> Optional[str]:
    """
    Choose a primary metric column to summarize.

    Strategy:
      1) Try to find a preferred name (accuracy/roc_auc/f1... case-insensitive).
      2) Else, pick the first numeric column that is not blacklisted (timestamps, counts, etc.).
    """
    if df is None or df.empty:
        return None
    lmap = _lower_map(df.columns.tolist())
    for pref in PREFERRED_METRICS:
        if pref in lmap:
            return lmap[pref]
    # fallback: first reasonable numeric column
    blacklist = set(
        ["index", "idx", "seed", "timestamp",
         "total_labeled", "sim_calls", "sim_time_sec_measured",
         "runtime_wall_sec", "final_iteration", "final_accuracy", "final_auc"]
        + ITERATION_COL_CANDIDATES
    )
    for c in df.select_dtypes(include=[np.number]).columns:
        if c.lower() not in blacklist:
            return c
    return None


def compute_stats(series: Optional[pd.Series]) -> Dict[str, Optional[float]]:
    """
    Compute simple descriptive stats on the chosen metric series.

    Returns keys:
      - max_value, min_value, last_value
      - avg_last10, std_last10, avg_last5, std_last5
      - score : a lightweight composite score (0.5*max + 0.3*avg - 0.2*std)
    """
    keys = ["max_value","min_value","last_value","avg_last10","std_last10","avg_last5","std_last5","score"]
    if series is None or series.empty:
        return {k: None for k in keys}
    s = series.dropna().astype(float)
    if s.empty:
        return {k: None for k in keys}

    max_v = float(np.nanmax(s.values))
    min_v = float(np.nanmin(s.values))
    last_v = float(s.values[-1])

    def tail(n):
        t = s.tail(n)
        if len(t) == 0:
            return None, None
        return float(np.nanmean(t.values)), float(np.nanstd(t.values))

    avg10, std10 = tail(10)
    avg5, std5 = tail(5)
    # simple score (tweak if desired)
    avg = avg10 if avg10 is not None else (avg5 if avg5 is not None else last_v)
    std = std10 if std10 is not None else (std5 if std5 is not None else 0.0)
    score = 0.5 * max_v + 0.3 * avg - 0.2 * std

    return {
        "max_value": max_v, "min_value": min_v, "last_value": last_v,
        "avg_last10": avg10, "std_last10": std10, "avg_last5": avg5, "std_last5": std5,
        "score": score,
    }


def read_csv_any(path: str) -> Optional[pd.DataFrame]:
    """Try several common separators to read a CSV-like file; return None if all fail."""
    for sep in [",", ";", "\t", "|"]:
        try:
            return pd.read_csv(path, sep=sep)
        except Exception:
            continue
    return None


def parse_filename_stem(stem: str) -> Dict[str, Optional[object]]:
    """
    Parse parameter tokens from the filename stem (no extension).
    Expects suffix `_YYYYMMDD_HHMMSS_<hexhash>`; if missing, returns {"stem": stem}.

    Extracts (best-effort):
      mode, strategy, initial_size, batch_size, iterations,
      test_ssize, seed, n_estimators, max_depth,
      min_samples_split, min_samples_leaf, class_weight,
      timestamp, hash, stem
    """
    m = STEM_END_RE.match(stem)
    if not m:
        return {"stem": stem}

    tokens = stem.split("_")
    out = {
        "mode": None, "strategy": None,
        "initial_size": None, "batch_size": None, "iterations": None,
        "test_ssize": None, "seed": None,
        "n_estimators": None, "max_depth": None,
        "min_samples_split": None, "min_samples_leaf": None,
        "class_weight": None,
        "timestamp": f"{m.group(1)}_{m.group(2)}",
        "hash": m.group(3),
        "stem": stem,
    }

    if len(tokens) >= 2:
        out["mode"] = tokens[0]
        out["strategy"] = tokens[1]

    i = 0
    while i < len(tokens):
        t = tokens[i]

        def take_int(prefix):
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
            # forms: ts0_1 or "ts0" + "1"
            if "_" in t:
                a, b = t[2:].split("_", 1)
                try:
                    out["test_ssize"] = float(f"{int(a)}.{int(b)}")
                except Exception:
                    pass
            else:
                try:
                    a = int(t[2:])
                    if i + 1 < len(tokens):
                        b = int(tokens[i + 1])
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


def pull_meta_from_json(json_path: str) -> Dict[str, Optional[object]]:
    """
    Load the .meta.json file and translate its keys into the same schema
    used by parse_filename_stem (so we can merge them).
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return {}
    mapping = {
        "mode": "mode",
        "strategy": "strategy",
        "init": "initial_size",
        "batch": "batch_size",
        "iters": "iterations",
        "test_size": "test_ssize",
        "seed": "seed",
        "n_estimators": "n_estimators",
        "max_depth": "max_depth",
        "min_samples_split": "min_samples_split",
        "min_samples_leaf": "min_samples_leaf",
        "class_weight": "class_weight",
        "timestamp": "timestamp",
        "hash": "hash",
    }
    out = {}
    for k, v in meta.items():
        if k in mapping and v is not None:
            out[mapping[k]] = v
    return out


def merge_preferring_meta(parsed: Dict[str, object], meta: Dict[str, object]) -> Dict[str, object]:
    """Merge two dicts, preferring non-None values from metadata over filename tokens."""
    out = dict(parsed)
    for k, v in meta.items():
        if v is not None:
            out[k] = v
    return out


def extract_kpis(df_main: Optional[pd.DataFrame], df_kpi: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    """
    Extract final KPIs from the KPI CSV if available, otherwise fall back to the main CSV.

    KPIs extracted:
      final_accuracy, final_auc, total_labeled, sim_calls,
      sim_time_sec_measured, runtime_wall_sec, final_iteration,
      precision, recall, f1, fnr  (new)
    """
    res = {k: None for k in [
        "final_accuracy", "final_auc", "total_labeled", "sim_calls",
        "sim_time_sec_measured", "runtime_wall_sec", "final_iteration",
        "precision", "recall", "f1", "fnr"
    ]}

    def pick(df: pd.DataFrame, names: List[str]) -> Optional[float]:
        if df is None or df.empty:
            return None
        lmap = {c.lower(): c for c in df.columns}
        for n in names:
            key = n.lower()
            if key in lmap:
                try:
                    val = df[lmap[key]].iloc[-1]
                    return float(val) if isinstance(val, (int, float, np.floating)) else val
                except Exception:
                    continue
        return None

    # --- prefer *_kpi.csv
    if df_kpi is not None:
        res["final_accuracy"]        = pick(df_kpi, ["final_accuracy", "accuracy", "acc"])
        res["final_auc"]             = pick(df_kpi, ["final_auc", "roc_auc", "auc", "auroc"])
        res["total_labeled"]         = pick(df_kpi, ["total_labeled", "labeled", "labels"])
        res["sim_calls"]             = pick(df_kpi, ["sim_calls", "simulations"])
        res["sim_time_sec_measured"] = pick(df_kpi, ["sim_time_sec_measured", "sim_time_sec", "sim_time"])
        res["runtime_wall_sec"]      = pick(df_kpi, ["runtime_wall_sec", "runtime_sec", "runtime"])
        res["final_iteration"]       = pick(df_kpi, ["final_iteration", "iteration", "iter", "step"])

        # NEW classification metrics
        res["precision"] = pick(df_kpi, PREC_CANDS)
        res["recall"]    = pick(df_kpi, RECALL_CANDS)
        res["f1"]        = pick(df_kpi, F1_CANDS)
        res["fnr"]       = pick(df_kpi, FNR_CANDS)

        # derive from counts if available
        tp = pick(df_kpi, TP_CANDS); fn = pick(df_kpi, FN_CANDS)
        fp = pick(df_kpi, FP_CANDS); tn = pick(df_kpi, TN_CANDS)

        if res["recall"] is None and tp is not None and fn is not None and (tp + fn) > 0:
            res["recall"] = tp / (tp + fn)
        if res["precision"] is None and tp is not None and fp is not None and (tp + fp) > 0:
            res["precision"] = tp / (tp + fp)
        if res["f1"] is None and res["precision"] is not None and res["recall"] is not None:
            p, r = res["precision"], res["recall"]
            if (p + r) > 0:
                res["f1"] = 2 * p * r / (p + r)
        if res["fnr"] is None:
            if res["recall"] is not None:
                res["fnr"] = 1.0 - res["recall"]
            elif tp is not None and fn is not None and (tp + fn) > 0:
                res["fnr"] = fn / (tp + fn)

    # --- fallback to main CSV if still missing
    for k, cand in [
        ("final_accuracy", ["final_accuracy", "accuracy", "acc"]),
        ("final_auc", ["final_auc", "roc_auc", "auc", "auroc"]),
        ("total_labeled", ["total_labeled", "labeled", "labels"]),
        ("sim_calls", ["sim_calls", "simulations"]),
        ("sim_time_sec_measured", ["sim_time_sec_measured", "sim_time_sec", "sim_time"]),
        ("runtime_wall_sec", ["runtime_wall_sec", "runtime_sec", "runtime"]),
        ("final_iteration", ["final_iteration", "iteration", "iter", "step"]),
        ("precision", PREC_CANDS),
        ("recall", RECALL_CANDS),
        ("f1", F1_CANDS),
        ("fnr", FNR_CANDS),
    ]:
        if res[k] is None:
            res[k] = pick(df_main, cand)

    # derive any remaining classification metrics from counts in the main CSV
    if df_main is not None and (res["precision"] is None or res["recall"] is None or res["f1"] is None or res["fnr"] is None):
        tp = pick(df_main, TP_CANDS); fn = pick(df_main, FN_CANDS)
        fp = pick(df_main, FP_CANDS); tn = pick(df_main, TN_CANDS)

        if res["recall"] is None and tp is not None and fn is not None and (tp + fn) > 0:
            res["recall"] = tp / (tp + fn)
        if res["precision"] is None and tp is not None and fp is not None and (tp + fp) > 0:
            res["precision"] = tp / (tp + fp)
        if res["f1"] is None and res["precision"] is not None and res["recall"] is not None:
            p, r = res["precision"], res["recall"]
            if (p + r) > 0:
                res["f1"] = 2 * p * r / (p + r)
        if res["fnr"] is None:
            if res["recall"] is not None:
                res["fnr"] = 1.0 - res["recall"]
            elif tp is not None and fn is not None and (tp + fn) > 0:
                res["fnr"] = fn / (tp + fn)

    return res


def build_master_row(stem: str, main_csv: Optional[str], kpi_csv: Optional[str], meta_json: Optional[str]) -> Optional[Dict[str, object]]:
    """
    Build one consolidated row by combining:
      - filename stem parsing,
      - optional metadata JSON,
      - chosen metric stats,
      - extracted KPIs.
    """
    parsed = parse_filename_stem(stem)
    meta = pull_meta_from_json(meta_json) if meta_json else {}
    info = merge_preferring_meta(parsed, meta)

    df_main = read_csv_any(main_csv) if main_csv else None
    df_kpi = read_csv_any(kpi_csv) if kpi_csv else None

    metric_col = choose_metric(df_main) if df_main is not None else None
    stats = compute_stats(df_main[metric_col]) if (df_main is not None and metric_col in df_main.columns) else {
        k: None for k in ["max_value","min_value","last_value","avg_last10","std_last10","avg_last5","std_last5","score"]
    }

    kpis = extract_kpis(df_main, df_kpi)
    id_val = info.get("hash") or os.path.basename(info.get("stem", stem))

    row = {
        "ID": id_val,
        "strategy": info.get("strategy"),
        "initial_size": info.get("initial_size"),
        "batch_size": info.get("batch_size"),
        "iterations": info.get("iterations"),
        "test_ssize": info.get("test_ssize"),
        "seed": info.get("seed"),
        "n_estimators": info.get("n_estimators"),
        "max_depth": info.get("max_depth"),
        "min_samples_split": info.get("min_samples_split"),
        "min_samples_leaf": info.get("min_samples_leaf"),
        "class_weight": info.get("class_weight"),
        "score_metric": metric_col,
        **stats,
        "final_accuracy": kpis.get("final_accuracy"),
        "final_auc": kpis.get("final_auc"),
        # NEW metrics
        "precision": kpis.get("precision"),
        "recall": kpis.get("recall"),
        "f1": kpis.get("f1"),
        "fnr": kpis.get("fnr"),
        # bookkeeping
        "total_labeled": kpis.get("total_labeled"),
        "sim_calls": kpis.get("sim_calls"),
        "sim_time_sec_measured": kpis.get("sim_time_sec_measured"),
        "runtime_wall_sec": kpis.get("runtime_wall_sec"),
    }
    return row


def collect_triplets(folder: str) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Collect related files (main, kpi, meta) by stem from a single folder (non-recursive).

    Rules:
      - Ignore subdirectories entirely.
      - Consider only files ending with .csv or .meta.json.
      - Recognize stems that end with `_YYYYMMDD_HHMMSS_<hash>`.
      - Map each stem to up to three paths: {"main": ..., "kpi": ..., "meta": ...}
    """
    triplets: Dict[str, Dict[str, Optional[str]]] = {}

    # scan only the top-level files in `folder`
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isdir(path):
            continue

        if not (f.endswith(".csv") or f.endswith(".meta.json")):
            continue

        # compute stem
        if f.endswith(".meta.json"):
            stem_key = f[:-10]
        elif f.endswith(".csv"):
            if f.endswith("_kpi.csv"):
                stem_key = f[:-8]   # drop _kpi.csv
            else:
                stem_key = f[:-4]   # drop .csv
        else:
            continue

        # validate stem ending
        if not STEM_END_RE.match(stem_key):
            continue

        entry = triplets.setdefault(stem_key, {"main": None, "kpi": None, "meta": None})
        if f.endswith(".meta.json"):
            entry["meta"] = path
        elif f.endswith("_kpi.csv"):
            entry["kpi"] = path
        else:
            entry["main"] = path

    return triplets


def main():
    """CLI: aggregate all runs found in a folder into Excel + CSV master files."""
    ap = argparse.ArgumentParser(description="Aggregate AL experiment result files (non-recursive) into a master table.")
    ap.add_argument("folder", help="Folder with files (*.csv, *_kpi.csv, *.meta.json) – top-level only.")
    ap.add_argument("--outfile", default="master_results.xlsx", help="Output Excel (.xlsx)")
    ap.add_argument("--outcsv", default="master_results.csv", help="Optional CSV export")
    args = ap.parse_args()

    triplets = collect_triplets(args.folder)
    rows = []
    for stem, paths in sorted(triplets.items()):
        row = build_master_row(
            stem=stem,
            main_csv=paths.get("main"),
            kpi_csv=paths.get("kpi"),
            meta_json=paths.get("meta"),
        )
        if row:
            rows.append(row)

    if not rows:
        print("No rows aggregated (check filename patterns and folder).")
        return

    df = pd.DataFrame(rows)

    preferred = [
        "ID","strategy","initial_size","batch_size","iterations","test_ssize","seed",
        "n_estimators","max_depth","min_samples_split","min_samples_leaf","class_weight",
        "score_metric","score","max_value","min_value","last_value",
        "avg_last10","std_last10","avg_last5","std_last5",
        "final_accuracy","final_auc",
        "precision","recall","f1","fnr",  # NEW columns
        "total_labeled","sim_calls","sim_time_sec_measured","runtime_wall_sec"
    ]
    ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[ordered]

    # --- append a timestamp to output filenames (before the extension) ---
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    def _with_ts(path: str, ts: str) -> str:
        root, ext = os.path.splitext(path)
        return f"{root}_{ts}{ext}"

    in_dir   = os.path.abspath(args.folder)
    xlsxname = _with_ts(os.path.basename(args.outfile), ts)
    csvname  = _with_ts(os.path.basename(args.outcsv),  ts)

    out_xlsx = os.path.join(in_dir, xlsxname)
    out_csv  = os.path.join(in_dir, csvname)
    # ---------------------------------------------------------------------
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="results")
    df.to_csv(out_csv, index=False)

    print(f"Saved: {out_xlsx} and {out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
