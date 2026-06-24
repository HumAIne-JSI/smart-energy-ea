# PowerShell Experiment Scripts

These scripts are **historical Windows experiment runners** from the offline/online Active Learning evaluation phase (August–September 2025). They were used to launch batches of `src/experiments/` Python scripts overnight with varying hyperparameters and seeds.

They are kept for reproducibility reference but are **not part of the deployed production workflow**.

> **Important:** All scripts contain hardcoded absolute paths (`C:\Users\gl8304\...`). Update `$BaseDir` / `$ROOT` to your own machine path before running.

---

## Script map

| Script | Internal name | What it runs | Key parameters |
|---|---|---|---|
| `01_online_margin_uncertainty.ps1` | `run_overnight.ps1` | Online AL with simulator: margin + uncertainty strategies, 3 seeds | init=100, batch=50, iters=20 |
| `02_quick_online_offline_entropy.ps1` | `hitri_test.ps1` | Quick test: online AL (entropy) + offline RF baseline back-to-back | Single seed, small run |
| `03_quick_online_uncertainty.ps1` | `quick_online_test.ps1` | Quick single run: online AL, uncertainty strategy only | init=100, batch=50, iters=20 |
| `04_quick_online_uncertainty.ps1` | `quick_online_test.ps1` | Same as 03, minor variant (check file diff if needed) | init=100, batch=50, iters=20 |
| `05_online_all.ps1` | `grid_all_strategies.ps1` | Grid sweep: 3 RF parameter combos × 4 strategies (uncertainty, entropy, margin, random) | seed=42 |
| `06_online_all.ps1` | *(unnamed)* | Similar grid sweep to 05 with updated Python path handling | seed=42 |
| `07_online_all.ps1` | `13_grid_online_alt_combos.ps1` | Online AL: alternative parameter combos E–H × 4 strategies, parallel jobs | Multiple RF configs |
| `08_online_all.ps1` | `14_grid_online_exploration.ps1` | Online AL: exploration combos I–L × 4 strategies, class_weight variants, 2 seeds | Broader RF sweep |
| `09_offline_all.ps1` | `09_offline_all.ps1` | Offline RF baseline: 4 configs (O5–O8) with parallel PowerShell jobs | MaxParallel=4 |

---

## Relationship to source code

All scripts call scripts under `src/experiments/`:

- `run_online_active_learning_with_simulator.py` — online AL with simulator loop
- `run_offline_random_forest_baseline.py` — offline RF baseline (no AL loop)

Results (CSV/XLSX) are written to `tables/`, logs to `tables/` or `logs/`. Both are gitignored.
