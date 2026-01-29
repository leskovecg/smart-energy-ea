"""
filename_namer.py
=================

Utilities for naming and saving experiment runs in a consistent way.

What this module does
---------------------
- Build compact, unique, and filterable filename slugs for experiments.
  * Includes key parameters: mode (online/offline), strategy, init/batch/iters,
    test size, seed, RF hyperparameters, class_weight, simulator time, etc.
  * Appends a timestamp and a short hash to ensure uniqueness.
- Save experiment results and metadata together:
  * <slug>.csv       : iteration metrics
  * <slug>_kpi.csv   : summary row with final KPIs
  * <slug>.meta.json : JSON with full metadata and parameters

Why this matters
----------------
- Easy to filter and group runs by filename parts (e.g. grep "entropy" or "cwbal").
- Guaranteed uniqueness even if many runs share the same config.
- Metadata sidecar ensures full reproducibility.

Typical usage
-------------
slug, meta = build_run_slug(mode="online", strategy="entropy", init=100, batch=50, iters=40, seed=42)
paths = save_with_meta(out_dir="tables", base_slug=slug, iter_df=metrics, kpi_row=kpi, meta=meta)
print("Saved:", paths)
"""


import json, os, hashlib
from datetime import datetime

# --- Abbreviations for RF/class_weight (keeps names short & filterable) ---
# Maps scikit-learn class_weight values into short strings for filenames.
_CW_MAP = {
    None: "none",
    "None": "none",
    "balanced": "bal",
    "balanced_subsample": "bsub",
    "": "none"
}

def _abbr_class_weight(cw) -> str:
    """
    Return a compact string abbreviation for class_weight.
    Example: "balanced_subsample" -> "bsub".
    Used to keep filename slugs shorter and consistent.
    """

    return _CW_MAP.get(cw, str(cw).replace(" ", "").lower())

def _short_hash(d: dict, length: int = 8) -> str:
    """
    Compute a short, stable hash of a dictionary (sorted by keys).
    Used to guarantee uniqueness of filename slugs.
    """

    s = json.dumps(d, sort_keys=True, separators=(",",":"))
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()[:length]

def build_run_slug(
    mode: str,                    # "online" | "offline"
    strategy: str = None,         # "entropy" | "margin" | "uncertainty" | "random" (online only)
    init: int = None, batch: int = None, iters: int = None,  # AL knobs (online)
    test_size: float = None, seed: int = None,
    n_estimators: int = None, max_depth: int = None,
    min_samples_split: int = None, min_samples_leaf: int = None,
    class_weight: str = None,
    avg_sim_sec: float = None,
    timestamp: str = None,        # override if you want; else auto YYYYMMDD_HHMMSS
    extra: dict = None            # any extra key/values you want embedded in hash/meta
) -> tuple[str, dict]:
    """
    Build a compact, filterable filename slug for a run.

    Example slug (online, entropy, init=100, batch=50):
        online_entropy_i100_b50_it40_ts0_1_s42_ne600_cwbsub_20250907_103000_ab12cd34

    Returns:
        slug: filename-friendly string (without extension).
        meta: dict with full metadata (all params).
    """

    assert mode in ("online", "offline"), "mode must be 'online' or 'offline'"
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cw = _abbr_class_weight(class_weight)

    parts = [mode]

    # Strategy block (online only)
    if mode == "online" and strategy:
        parts.append(strategy)

    # Active learning knobs (online)
    if mode == "online":
        if init is not None:  parts.append(f"i{init}")
        if batch is not None: parts.append(f"b{batch}")
        if iters is not None: parts.append(f"it{iters}")

    # Common knobs
    if test_size is not None: parts.append(f"ts{str(test_size).replace('.', '_')}")
    if seed is not None:      parts.append(f"s{seed}")

    # RF
    if n_estimators is not None:      parts.append(f"ne{n_estimators}")
    if max_depth is not None:         parts.append(f"md{max_depth if max_depth is not None else 'None'}")
    if min_samples_split is not None: parts.append(f"mss{min_samples_split}")
    if min_samples_leaf is not None:  parts.append(f"msl{min_samples_leaf}")
    if cw is not None:                parts.append(f"cw{cw}")

    # Simulator (online only)
    if mode == "online" and avg_sim_sec is not None:
        parts.append(f"sim{str(avg_sim_sec).replace('.', '_')}s")

    # Short hash over all parameters to guarantee uniqueness
    meta = {
        "mode": mode,
        "strategy": strategy,
        "init": init, "batch": batch, "iters": iters,
        "test_size": test_size, "seed": seed,
        "n_estimators": n_estimators, "max_depth": max_depth,
        "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf,
        "class_weight": class_weight,
        "avg_sim_sec": avg_sim_sec,
        "timestamp": timestamp,
    }
    if extra: meta["extra"] = extra
    hid = _short_hash(meta, length=8)

    # Append timestamp and hash at the end (easy sorting, guaranteed uniqueness)
    parts.append(timestamp)
    parts.append(hid)

    return "_".join(parts), meta

def save_with_meta(out_dir: str, base_slug: str, iter_df=None, kpi_row: dict | None = None, meta: dict | None = None):
    """
    Save results (iteration metrics, KPIs, metadata) with consistent filenames.

    Produces up to three files:
        - <slug>.csv         : per-iteration metrics (if iter_df is provided)
        - <slug>_kpi.csv     : one-row summary with final KPIs (if kpi_row is provided)
        - <slug>.meta.json   : JSON file with full parameters and metadata

    Args:
        out_dir: directory to save files.
        base_slug: filename slug from build_run_slug().
        iter_df: iteration metrics (list of dicts or DataFrame).
        kpi_row: final KPIs (dict).
        meta: metadata dict.

    Returns:
        dict with paths to the saved files.
    """
    
    os.makedirs(out_dir, exist_ok=True)
    iter_path = os.path.join(out_dir, base_slug + ".csv")
    kpi_path  = os.path.join(out_dir, base_slug + "_kpi.csv")
    meta_path = os.path.join(out_dir, base_slug + ".meta.json")

    if iter_df is not None:
        import pandas as pd
        pd.DataFrame(iter_df).to_csv(iter_path, index=False)

    if kpi_row is not None:
        import pandas as pd
        pd.DataFrame([kpi_row]).to_csv(kpi_path, index=False)

    if meta is not None:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return {"iter_csv": iter_path, "kpi_csv": kpi_path, "meta_json": meta_path}
