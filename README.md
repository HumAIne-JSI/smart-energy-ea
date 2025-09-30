# Smart Energy ML Analysis – JSI E3

This repository contains my personal contributions and experimental work developed during my collaboration with the Artificial Intelligence Department (E3) at the Jožef Stefan Institute, as part of the [HumAIne-JSI](https://github.com/HumAIne-JSI) project.

Official repository: [energy-ea](https://github.com/HumAIne-JSI/energy-ea)

---

## Project context

The project focuses on exploratory data analysis and classification for smart energy systems. It is part of the EU-funded [HUMAINE](https://humaine-horizon.eu/) project, aiming to develop transparent, human-centered AI tools in the energy domain.

---

## My contributions

To be completed. 

---

## Repository structure

```text
smart-energy-ml-analysis-jsi/
├── data/    
├── docs/                                  
├── figures/      
├── models/     
├── notebooks/  
├── reports/                               
├── tables/  
├── README.md                              
```
---
# Active Learning (Smart Energy) – Run Guide (No MinIO)

This guide helps you run everything *today* without MinIO.

## 1) One-shot simulated AL run (with on-demand simulator)
Runs AL with `entropy` strategy, measures KPIs (sim calls, sim time), and saves CSVs.

```bash
python /mnt/data/run_simulated_active_learning.py
```

Outputs:
- `tables/metrics_simulated_entropy.csv`
- `tables/kpis_simulated_entropy.csv`

## 2) Streamlit dashboard (single run + baseline)
Interactive app showing learning curve vs random baseline and KPI cards.

```bash
streamlit run /mnt/data/streamlit_al_dashboard.py --server.headless true
```

Notes:
- The app automatically loads the dataset from either `data/simulation_security_labels_n-1.csv` or `/mnt/data/simulation_security_labels_n-1.csv`.
- Single run uses the simulator on-demand; baseline random run uses offline labels (no simulator calls).

## 3) Offline grid experiments (no simulator)
Run an experiment grid over strategies, initial sizes, batch sizes, and iteration counts.
Saves figures and CSV tables.

```bash
python -c "from al_experiment_code import run_experiment_grid; df=run_experiment_grid(csv_path='simulation_security_labels_n-1.csv', strategies=['entropy','uncertainty','margin','random'], initial_sizes=[10,20], batch_sizes=[5,10], iteration_counts=[10,20], test_size=0.1, random_state=42, figures_dir='figures', tables_dir='tables'); print(df.head())"
```

Outputs:
- `tables/experiment_results_summary.csv`
- `tables/experiment_iteration_metrics.csv`
- figures under `figures/` (if plotting is enabled in your code).

---
# Smart Energy Active Learning – Interim Report (No MinIO)

## 1. Objective
Reduce simulator usage/time via Active Learning (AL) while maintaining classification performance for N-1 security assessment.

## 2. Data & Digital Twin
- Dataset: `simulation_security_labels_n-1.csv` (secure/insecure)
- Digital twin: `digital_twin_ext_grid.json`

## 3. Methods
- Classifier: Random Forest (100 trees)
- Query strategies: entropy, uncertainty, margin, random
- AL loop with on-demand simulator labels (caching enabled)

## 4. KPIs
- Total labeled samples
- Sample saving (%) = 1 - labeled / pool_size
- Simulator calls (cumulative)
- Simulator time (cumulative seconds)

## 5. Experiments (to run today)
- **Single run**: entropy vs baseline random (same init, batch, iterations)
- **Grid**: small grid (entropy, uncertainty, margin, random) with two initial sizes and batches

## 6. Results (fill after runs)
- Final accuracy (AL): ___%
- Final accuracy (baseline): ___%
- Labeled samples (AL): ___ / Pool: ___ → saving: ___%
- Simulator calls/time (AL): ___ / ___ s
- Observation about random vs sequential splits: ___

## 7. Figures
- Learning curve (accuracy vs iterations): AL vs baseline
- KPI table (per iteration): labeled_count, sim_calls_cum, sim_time_sec_cum

## 8. Next Steps
- Integrate MinIO writes (results + models)
- Prepare HumAIne dashboard binding
- Expand simulator parameterization (seasonality scenarios)

---

## Tools & Technologies

- Python 
- Jupyter Notebooks
- Streamlit
- Git & GitHub
- VS Code

---

## Related links

- [HumAIne-JSI GitHub](https://github.com/HumAIne-JSI)
- [energy-ea (main repo)](https://github.com/HumAIne-JSI/energy-ea)
- [HUMAINE EU Project](https://humaine-horizon.eu/)

---

## Author

**Gašper Leskovec**  
MSc student in Electrical Engineering (ICT) – University of Ljubljana  
Contributor at E3, Jožef Stefan Institute  
GitHub: [@leskovecg](https://github.com/leskovecg)  







---






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

Happy experimenting! 🚀








