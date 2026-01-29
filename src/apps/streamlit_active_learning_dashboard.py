"""
streamlit_al_dashboard.py
=========================

Streamlit dashboard za prikaz AL:
- Mode 1: Single Run (Simulator) — on-demand labels + KPI download (CSV/XLSX)
- Mode 2: Offline Grid — baseline sweep + summary/per-iteration
"""

import os
import io
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

# Online AL
from core.active_learning_loop_with_simulator import run_active_learning
# Offline grid
from experiments.run_offline_active_learning_grid import run_experiment_grid


def parse_int_list(text: str) -> List[int]:
    if not text.strip():
        return []
    parts = text.replace(",", " ").split()
    return [int(p) for p in parts]

# --- fixed project-relative paths ---
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_DATA    = os.path.join(BASE_DIR, "data", "simulation_security_labels_n-1.csv")
DEFAULT_TABLES  = os.path.join(BASE_DIR, "tables")
DEFAULT_FIGURES = os.path.join(BASE_DIR, "figures")

st.set_page_config(page_title="Active Learning Dashboard (Smart Energy)", layout="wide")
st.title("Active Learning Dashboard — Smart Energy (N-1)")

with st.sidebar:
    st.header("Controls")
    mode = st.radio(
        "Mode",
        ["Single Run (Simulator)", "Offline Grid (No Simulator)"],
        index=0,
        help="Choose on-demand simulator run or fast offline grid.",
    )

    st.markdown("### Data")
    data_path = st.text_input(
        "CSV path",
        value=DEFAULT_DATA,
        help="Path to the dataset CSV.",
    )

# --- Mode 1: Single Run (Simulator) ------------------------------------------
if mode == "Single Run (Simulator)":
    st.subheader("Single Run (on-demand simulator labels)")

    with st.sidebar:
        st.markdown("### Parameters")
        strategy = st.selectbox(
            "Query strategy",
            options=["entropy", "uncertainty", "margin", "random"],
            index=0,
            help="Uncertainty-based selection heuristic for AL.",
        )
        initial_size = st.number_input("Initial labeled size", min_value=1, value=10, step=1)
        batch_size = st.number_input("Batch size per iteration", min_value=1, value=10, step=1)
        iterations = st.number_input("Iterations", min_value=1, value=20, step=1)
        test_size = st.slider("Validation split (test_size)", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random seed", min_value=0, value=42, step=1)
        avg_sim_time_sec = st.number_input(
            "Avg sim time per label (sec) [optional]", min_value=0.0, value=0.0, step=0.1,
            help="If >0, we also compute estimated simulator time KPIs."
        )
        run_btn = st.button(" Run Single AL")

    if run_btn:
        try:
            with st.spinner("Loading data…"):
                df_full = pd.read_csv(data_path)
                if "status" not in df_full.columns:
                    st.error("Column 'status' not found in the dataset.")
                    st.stop()
                y_true = df_full["status"].map({"secure": 1, "insecure": 0})
                drop_cols = [c for c in ["timestamp", "status"] if c in df_full.columns]
                X = df_full.drop(columns=drop_cols)

            # Dummy pool y (unused online)
            y_dummy = np.zeros(len(X), dtype=int)

            # Split (stratify by true labels)
            X_pool, X_val, y_pool, _ = train_test_split(
                X, y_dummy, test_size=float(test_size), random_state=int(random_state), stratify=y_true
            )
            y_val_true = y_true.iloc[X_val.index].reset_index(drop=True)

            st.info(
                f"Pool size: {len(X_pool):,} — Validation size: {len(X_val):,} "
                f"(test_size={test_size})"
            )

            # Run AL
            with st.spinner("Running Active Learning with simulator…"):
                metrics, duration, kpi = run_active_learning(
                    X_pool=X_pool.reset_index(drop=True),
                    y_pool=pd.Series(y_pool).reset_index(drop=True),
                    X_val=X_val.reset_index(drop=True),
                    y_val=y_val_true.reset_index(drop=True),
                    strategy=strategy,
                    initial_size=int(initial_size),
                    batch_size=int(batch_size),
                    iterations=int(iterations),
                    random_state=int(random_state),
                    simulate_on_demand=True,
                    avg_sim_time_sec=(float(avg_sim_time_sec) if avg_sim_time_sec > 0 else None),
                )

            df_metrics = pd.DataFrame(metrics)
            st.success(f"Done in {duration:.2f} s")

            # Show table
            st.markdown("#### Per-iteration metrics")
            st.dataframe(df_metrics, use_container_width=True, height=350)

            # Simple charts
            c1, c2, c3 = st.columns(3)
            with c1:
                st.line_chart(df_metrics.set_index("iteration")["accuracy"], height=220)
            with c2:
                st.line_chart(df_metrics.set_index("iteration")["f1"], height=220)
            with c3:
                if "roc_auc" in df_metrics.columns:
                    st.line_chart(df_metrics.set_index("iteration")["roc_auc"], height=220)

            # Downloads
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"metrics_simulated_{strategy}_init{initial_size}_b{batch_size}_it{iterations}_{ts}"

            # CSV
            csv_buf = df_metrics.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download CSV: {base}.csv",
                data=csv_buf,
                file_name=f"{base}.csv",
                mime="text/csv",
            )

            # XLSX (per_iteration + kpi_summary)
            xlsx_buf = io.BytesIO()
            with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
                df_metrics.to_excel(writer, sheet_name="per_iteration", index=False)
                pd.DataFrame([kpi]).to_excel(writer, sheet_name="kpi_summary", index=False)
            st.download_button(
                "Download XLSX (with KPIs)",
                data=xlsx_buf.getvalue(),
                file_name=f"{base}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.caption(
                "Note: simulator labels are acquired on demand. "
                "Fewer labeled samples at similar accuracy ⇒ less simulator time."
            )

        except FileNotFoundError:
            st.error(f"File not found: {data_path}")
        except Exception as e:
            st.exception(e)

# --- Mode 2: Offline Grid (No Simulator) -------------------------------------
else:
    st.subheader("Offline Grid (baseline, uses CSV labels)")

    with st.sidebar:
        st.markdown("### Grid Parameters")

        strategies = st.multiselect(
            "Strategies",
            options=["uncertainty", "entropy", "margin", "random"],
            default=["uncertainty", "entropy", "random"],
        )

        init_text = st.text_input("Initial sizes (comma/space sep.)", value="50, 100, 200")
        batch_text = st.text_input("Batch sizes (comma/space sep.)", value="10, 25, 50")
        iter_text = st.text_input("Iteration counts (comma/space sep.)", value="20, 40, 60")

        grid_test_size = st.slider(
            "Validation split (test_size)",
            min_value=0.05, max_value=0.5, value=0.10, step=0.05
        )
        grid_seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        grid_avg_sim = st.number_input(
            "Avg sim time per label (sec) for estimates",
            min_value=0.0, value=0.0, step=0.1,
            help="Only used to estimate time saving vs. full sweep."
        )

        run_grid_btn = st.button("Run Offline Grid")

    if run_grid_btn:
        try:
            initial_sizes = parse_int_list(init_text)
            batch_sizes = parse_int_list(batch_text)
            iteration_counts = parse_int_list(iter_text)

            if not initial_sizes or not batch_sizes or not iteration_counts or not strategies:
                st.warning("Please provide at least one value for each parameter.")
            else:
                with st.spinner("Running offline grid experiments…"):
                    df_results = run_experiment_grid(
                        csv_path=data_path,
                        strategies=strategies,
                        initial_sizes=initial_sizes,
                        batch_sizes=batch_sizes,
                        iteration_counts=iteration_counts,
                        test_size=float(grid_test_size),
                        random_state=int(grid_seed),
                        figures_dir=DEFAULT_FIGURES,
                        tables_dir=DEFAULT_TABLES,
                        avg_sim_time_sec=(float(grid_avg_sim) if grid_avg_sim > 0 else None),
                    )

                st.success(f"Grid run complete. Results saved to: {DEFAULT_TABLES}")

                # Preview results
                st.markdown("#### Summary results (preview)")
                st.dataframe(df_results.head(100), use_container_width=True, height=400)

                # Quick comparison chart
                if {"accuracy_final", "total_labeled_samples", "strategy_type"}.issubset(df_results.columns):
                    st.markdown("#### Accuracy (final) vs. total labeled (by strategy)")
                    pivot = (
                        df_results
                        .groupby(["strategy_type", "total_labeled_samples"], as_index=False)["accuracy_final"]
                        .mean()
                        .sort_values(["strategy_type", "total_labeled_samples"])
                    )
                    for strategy_name in pivot["strategy_type"].unique():
                        sub = pivot[pivot["strategy_type"] == strategy_name].set_index("total_labeled_samples")
                        st.line_chart(sub["accuracy_final"], height=220, width=700)
                        st.caption(f"Strategy: {strategy_name}")

                st.caption(
                    f"Detailed per-iteration CSV is saved to: {DEFAULT_TABLES}. "
                    "Use it to plot full learning curves if needed."
                )

        except FileNotFoundError:
            st.error(f"File not found: {data_path}")
        except Exception as e:
            st.exception(e)

# --- Footer -------------------------------------------------------------------
st.write("---")
st.caption(
    "© Active Learning Smart-Energy Demo — RandomForest + Uncertainty strategies. "
    "Simulator querying available in the Single Run mode."
)
