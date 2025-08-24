import os
from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from active_learning_with_simulator import run_active_learning

# Nastavitve strani
st.set_page_config(page_title="Active Learning Dashboard", layout="wide")

st.title("Active Learning with Simulator")


# Naloži podatke
@st.cache_data
def load_data():
    df = pd.read_csv("data/simulation_security_labels_n-1.csv")
    df["status_binary"] = df["status"].map({"secure": 1, "insecure": 0})
    X = df.drop(columns=["timestamp", "status", "status_binary"])
    y = df["status_binary"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


X_pool, X_val, y_pool, y_val = load_data()

# Izbira parametrov (form-submit, brez rerun ob premiku sliderja)
with st.sidebar:
    st.header("Parametri")
    mode = st.radio("Način", ["Single run", "Grid run"], horizontal=True)
    with st.form("al_params"):
        strategy = st.selectbox("Strategija", ["entropy", "uncertainty", "margin", "random"])
        initial_size = st.slider("Začetna velikost", 5, 100, 20, step=5)
        batch_size = st.slider("Velikost batcha", 1, 100, 10, step=5)
        iterations = st.slider("Število iteracij", 1, 100, 30, step=1)
        submitted = st.form_submit_button("Zaženi")


# Zagon
if "submitted" in locals() and submitted:
    st.info("Začenjam simulacijo... prosim počakajte.")

    with st.spinner("Izvajam ..."):
        if mode == "Single run":
            metrics, duration = run_active_learning(
                X_pool=X_pool,
                y_pool=y_pool,
                X_val=X_val,
                y_val=y_val,
                strategy=strategy,
                initial_size=initial_size,
                batch_size=batch_size,
                iterations=iterations,
                simulate_on_demand=True,
            )

            df_metrics = pd.DataFrame(metrics)
            st.success(f"Dokončano v {duration:.2f} sekundah.")

            # Vizualizacija
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Točnost skozi iteracije")
                fig, ax = plt.subplots()
                ax.plot(df_metrics["iteration"], df_metrics["accuracy"], marker="o")
                ax.set_xlabel("Iteracija")
                ax.set_ylabel("Točnost")
                ax.set_title(f"Strategija: {strategy}")
                ax.grid(True)
                st.pyplot(fig)

            with col2:
                st.subheader("Povzetek")
                st.metric("Označeni vzorci", initial_size + iterations * batch_size)
                st.metric("Končna točnost", f"{df_metrics['accuracy'].iloc[-1]*100:.2f}%")
                st.metric("Čas izvajanja", f"{duration:.2f} s")
                st.metric(
                    "Prihranek vzorcev",
                    f"{100 - ((initial_size + iterations * batch_size) / len(X_pool))*100:.2f}%",
                )

            st.subheader("Podrobnosti po iteracijah")
            st.dataframe(df_metrics, use_container_width=True)

            # Shrani CSV in ponudi download
            os.makedirs("tables", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_csv = f"tables/streamlit_single_run_{strategy}_i{initial_size}_b{batch_size}_it{iterations}_{ts}.csv"
            df_metrics.to_csv(out_csv, index=False)
            st.download_button(
                "Prenesi CSV",
                data=df_metrics.to_csv(index=False),
                file_name=os.path.basename(out_csv),
                mime="text/csv",
            )

        else:
            # Grid run na offline datasetu (brez simulatorja)
            from al_experiment_code import run_experiment_grid

            df_results = run_experiment_grid(
                csv_path="data/simulation_security_labels_n-1.csv",
                strategies=["entropy", "uncertainty", "margin", "random"],
                initial_sizes=[10, 20],
                batch_sizes=[5, 10],
                iteration_counts=[10, 20],
                test_size=0.1,
                random_state=42,
                figures_dir="figures",
                tables_dir="tables",
            )
            st.success("Grid run končan.")
            st.dataframe(df_results, use_container_width=True)