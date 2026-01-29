# Source Code Overview (`src/`)

This directory contains the **core implementation** of the Active Learning
pipeline for Smart Energy security assessment.

It includes reusable logic, experiment entrypoints, analysis utilities,
and lightweight applications used during research and experimentation.

---

## Directory structure

```text
src/
├── core/           # Core Active Learning logic and simulator interfaces
├── experiments/    # CLI entrypoints for running experiments
├── analysis/       # Post-hoc analysis and metric aggregation
├── utils/          # Shared helper utilities and configuration helpers
├── apps/           # User-facing applications (e.g. Streamlit dashboard)
└── integration/    # Integration scripts (e.g. MinIO upload helpers)
```

---

## Core modules (`core/`)

Contains the main domain logic:
- Active Learning loops
- Strategy selection
- Simulator / oracle interfaces

These modules are designed to be **importable and reusable** across
experiments, dashboards, and services.

---

## Experiments (`experiments/`)

Command-line entrypoints for running experiments, for example:
- Offline baselines (Random Forest)
- Offline Active Learning grid sweeps
- Online Active Learning with simulator-based labels

Typical usage (from repository root):
```bash
python src/experiments/run_online_active_learning_with_simulator.py
```

Experiment scripts are responsible for:
- configuring parameters
- orchestrating runs
- saving results (CSV / XLSX / metadata)

---

## Analysis (`analysis/`)

Scripts for post-processing experiment outputs:
- aggregating metrics across runs
- generating summary tables
- preparing results for plots or reports

These scripts operate on data produced by `experiments/`.

---

## Utilities (`utils/`)

Shared helper code, including:
- naming and run metadata helpers
- dataset and schema comparison utilities
- small reusable helper functions

These modules should contain **no experiment-specific logic**.

---

## Applications (`apps/`)

User-facing tools built on top of the experimental pipeline:
- Streamlit dashboards for monitoring and visualization

Typical usage:
```bash
streamlit run src/apps/streamlit_active_learning_dashboard.py
```

---

## Integration (`integration/`)

Scripts that interface with external systems, such as:
- MinIO dataset and artifact upload

These are typically run manually or via automation pipelines.

---

## Notes

- This directory is intended for **research and experimentation**.
- The code prioritizes clarity and reproducibility over production hardening.
- For API-related code, see `app/README.md`.

---
