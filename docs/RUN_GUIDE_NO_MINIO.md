# Active Learning (Smart Energy) – Run Guide (No MinIO)

This guide explains how to run Active Learning experiments **without MinIO**.
It is intended for fast experimentation, debugging, and interim evaluation.

## 1) One-shot simulated Active Learning run
Runs AL with an uncertainty-based strategy and queries the simulator on demand.

```bash
python run_online_active_learning_with_simulator.py
```

### Outputs
- `tables/metrics_simulated_<strategy>.csv`
- `tables/kpis_simulated_<strategy>.csv`

## 2) Streamlit dashboard
Interactive dashboard for running a single AL experiment and comparing it to a random baseline.

```bash
streamlit run streamlit_active_learning_dashboard.py --server.headless true
```

Notes:
- Dataset is loaded automatically from `data/` if present.
- Simulator calls are cached.

## 3) Offline grid experiments (no simulator)
Runs a grid of AL strategies using only offline labels.

```bash
python -c "from run_offline_active_learning_grid import run_experiment_grid; run_experiment_grid(...)"
```

### Outputs
- `tables/experiment_results_summary.csv`
- `tables/experiment_iteration_metrics.csv`
- Figures saved under `figures/`
