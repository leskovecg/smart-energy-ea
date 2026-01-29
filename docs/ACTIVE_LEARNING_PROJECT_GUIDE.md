# Active Learning for Smart Energy — Project Guide

This document provides a detailed, file-by-file and concept-by-concept explanation
of the Smart Energy Active Learning project.

## 1. Big Picture
The project supports two execution modes:
- Online Active Learning with simulator-in-the-loop
- Offline benchmarking using pre-labeled datasets

## 2. Core Entry Points
- `run_online_active_learning_with_simulator.py`
- `run_offline_active_learning_grid.py`
- `streamlit_active_learning_dashboard.py`

## 3. Active Learning Loop
The AL loop:
1. Trains a classifier on labeled data
2. Scores unlabeled samples
3. Selects most informative samples
4. Queries simulator (optional)
5. Updates metrics and KPIs

## 4. Simulator Interface
`power_grid_simulator_interface.py` wraps pandapower and provides cached N-1 security checks.

## 5. Metrics & KPIs
- Accuracy, Precision, Recall, F1
- ROC-AUC (safe)
- Simulator calls and runtime

## 6. Extending the Project
- Add new AL strategies
- Swap classifiers
- Extend feature space

## 7. Troubleshooting
- ROC-AUC NaN → single-class validation
- Missing digital twin → check data paths








# Active Learning for Smart Energy — Project Guide

This guide explains what each script does, how they fit together, which functions matter, what they take as input and return as output, and how to run everything end‑to‑end. It’s written for **first‑time readers** and for **you** when you return to the project later.

---

## 1) Big Picture

You have two complementary ways to run experiments:

- **Online (with simulator calls)** — labels are obtained **on demand** by calling the digital‑twin simulator.  
  *Entry point:* `run_simulated_active_learning.py` → uses `active_learning_with_simulator.py` → calls `simulator_interface.py`.

- **Offline (no simulator calls)** — labels are taken from the CSV, used to benchmark Active Learning (AL) strategies quickly.  
  *Entry point:* `al_experiment_code.py`.

For a UI, use **`streamlit_al_dashboard.py`** to run both modes from a dashboard and download results.

---

## 2) File‑by‑File Overview

### `run_simulated_active_learning.py`
End‑to‑end **online** run that splits data, runs AL with **simulator labels on demand**, and saves results (CSV + XLSX).  
Key responsibilities:
- **Time‑based split** when a `timestamp` exists: pool = past, validation = future (no overlap). Falls back to stratified split otherwise.
- **Feature whitelist** to avoid leakage (e.g., keep only `load_*`, `gen_*`, `sgen_*` columns).
- Calls `active_learning_with_simulator.run_active_learning(simulate_on_demand=True)`.
- Writes **per‑iteration metrics** and a **KPI summary** to disk.

**Main CLI arguments**
```
--data <path>           Path to CSV (must contain 'status' = secure/insecure)
--strategy <str>        entropy | uncertainty | margin | random
--init <int>            initial labeled size
--batch <int>           queries per iteration
--iters <int>           number of AL iterations
--test-size <float>     validation fraction (0–1)
--seed <int>            random seed
--avg-sim-sec <float>   optional, to compute estimated simulator time
--tables-dir <path>     output folder for CSV/XLSX
```

**Outputs created**
- `tables/metrics_simulated_<strategy>_init<...>_b<...>_it<...>_<timestamp>.csv` (per‑iteration)
- corresponding `.xlsx` with sheets `per_iteration` and `kpi_summary`

---

### `active_learning_with_simulator.py`
Implements the **Active Learning loop** that can query the simulator **only for the samples you choose**.  
Core ideas:
- Train `RandomForestClassifier(class_weight="balanced")` on currently labeled pool.
- Score **unlabeled** points with a strategy (`uncertainty`, `entropy`, `margin`, `random`).
- Pick top‑K, **query labels via simulator** if `simulate_on_demand=True`, add them to labeled set.
- Track metrics over iterations (Accuracy, Macro‑Precision/Recall/F1, safe ROC‑AUC) and **KPI counters** (sim calls/time, wall time, etc.).

**Key functions**

```python
compute_query_scores(proba, strategy) -> np.ndarray
```
- Input: `proba` (N×2 class probabilities), `strategy` ∈ {uncertainty, entropy, margin, random}
- Output: **higher = more informative** score for each unlabeled sample

```python
run_active_learning(X_pool, y_pool, X_val, y_val, strategy,
                    initial_size, batch_size, iterations,
                    random_state=42,
                    simulate_on_demand=False,
                    avg_sim_time_sec=None)
 -> (metrics_per_iteration, duration_wall_sec, kpi_summary)
```
- If `simulate_on_demand=True`, labels for selected samples are fetched via the simulator (cached).
- Returns:
  - `metrics_per_iteration`: list of dicts with metrics + KPI counters per iteration  
  - `duration_wall_sec`: total wall‑clock time  
  - `kpi_summary`: final snapshot (accuracy/AUC, how many labels used, #sim calls, measured/estimated sim time, etc.)

---

### `simulator_interface.py`
Thin wrapper around the **pandapower** model of your grid (digital twin), with **robust path resolution** and **LRU‑cached** queries.

**Key pieces**
```python
query_simulator(sample: dict) -> "secure" | "insecure"
```
Runs base‑case + N‑1 contingencies; returns `"secure"` only if all checks pass (line loading within 100%, bus voltages within [0.9, 1.1] pu).

```python
query_simulator_cached(sample: dict) -> "secure" | "insecure"
```
Adds a stable cache key → **massively reduces repeated simulator work**.  

**Inputs expected in `sample`**
- Feature names like `load_<i>_p_mw`, `gen_<i>_p_mw`, `sgen_<i>_p_mw` mapped to floats.

---

### `al_experiment_code.py`
Implements **offline** AL sweeps (fast baselines). Labels are read from CSV.  
Highlights:
- `load_dataset()` parses & sorts by `timestamp` (if present), maps `status` → binary, and **drops target/timestamp from features**.
- Three split modes: **random (stratified)**, **sequential**, and **time‑based** (cut at quantile).
- `check_split_diagnostics()` prints **class balance** and **time‑range** info (helps debug AUC issues).
- `run_active_learning()` (offline variant) returns per‑iteration metrics and duration.
- `run_experiment_grid()` runs a **parameter grid** (strategies × init × batch × iters × split), then saves:
  - `tables/active_learning_results_<timestamp>.csv` (summary)
  - `tables/active_learning_results_<timestamp>.xlsx` (summary + `per_iteration` sheet)
  - `tables/al_metrics_per_iteration_<timestamp>.csv` (full curves)

---

### `streamlit_al_dashboard.py`
A simple **Streamlit** app to run either mode interactively and **download results**.

- **Mode 1: Single Run (Simulator)** — performs a stratified split by the true labels, then calls the online AL loop.
- **Mode 2: Offline Grid** — lets you choose strategies and grid params, runs `run_experiment_grid()`, previews a summary, and provides quick comparison charts.

**Run it**
```bash
streamlit run streamlit_al_dashboard.py
```

---

## 3) Data Expectations

Your CSV is expected to include:
- `status` column with values `"secure"` or `"insecure"` (mandatory)
- Optional `timestamp` (recommended for strict time‑based evaluation)
- Exogenous features such as `load_*`, `gen_*`, `sgen_*`, … (and other domain inputs like `pv_*`, `wind_*`, `weather_*` if you add them)

**Important**: We explicitly **drop** `status` and `timestamp` from the model’s input features to avoid leakage.

---

## 4) How to Run

### 4.1 Online (simulator) from CLI
```bash
python run_simulated_active_learning.py \
  --data "C:\path\to\simulation_security_labels_n-1.csv" \
  --strategy entropy \
  --init 100 \
  --batch 50 \
  --iters 40 \
  --test-size 0.1 \
  --seed 42 \
  --avg-sim-sec 2.3 \
  --tables-dir "tables"
```

### 4.2 Offline baseline grid from CLI
```bash
python al_experiment_code.py
```
*(Edit the `__main__` constants or call `run_experiment_grid()` from another script/notebook.)*

### 4.3 Streamlit dashboard
```bash
streamlit run streamlit_al_dashboard.py
```
Pick a mode in the sidebar, set parameters, click **Run**, and download CSV/XLSX.

---

## 5) Metrics & KPIs

Per iteration you get:
- **Accuracy, Macro‑Precision, Macro‑Recall, Macro‑F1**
- **ROC‑AUC (safe)** — returns `NaN` when only one class is present in validation to avoid misleading warnings
- **KPI counters (online mode)** — cumulative simulator calls, simulator time (measured), estimated simulator time (optional), training time, wall time, and total labeled count.

**Goal** of AL: achieve similar accuracy with **far fewer labeled samples**, translating to **lower simulator time**.

---

## 6) Extending the Project

- **Add new AL strategies**: implement a scorer in `compute_query_scores()` and add it to the accepted choice list.
- **Add more exogenous features**: extend the whitelist in `run_simulated_active_learning._select_feature_columns()` (e.g., `"pv_"`, `"wind_"`, `"weather_"`).
- **Swap models**: replace `RandomForestClassifier` with your model (keep `class_weight="balanced"` if classes are skewed).

---

## 7) Troubleshooting

- **ROC‑AUC is NaN** — validation contains only one class; use a time split with enough positives/negatives or expand the validation window.
- **digital_twin_ext_grid.json not found** — check `data/` path; the simulator loader tries multiple locations but you may need to drop the JSON into `data/`.
- **No features after whitelist** — you’ll see a warning and it will fall back to “all except labels/timestamp”. Prefer to fix the whitelist so only true exogenous inputs remain.

---

## 8) Quick Glossary

- **AL (Active Learning)** — iteratively selects the **most informative** samples to label next.
- **Uncertainty/Entropy/Margin** — three standard uncertainty‑based selection heuristics.
- **On‑demand labels** — ground truth obtained by **calling the simulator** as needed, not pre‑labeling everything.
- **N‑1** — grid security check under single‑element outages (lines/generators).

---

**Author notes**  
- Model defaults: `RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)`  
- All outputs are timestamped to keep experiment logs clean and comparable.
- Caching in the simulator layer dramatically speeds up repeated queries with identical features.

---
