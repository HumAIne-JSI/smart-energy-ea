# Project Status
> Last updated: 2026-07-02 — refreshed after each milestone.
> For full docs navigation see [docs/INDEX.md](docs/INDEX.md).

## What this project is
This project supports an Active Learning workflow for power-grid N-1
security assessment. The operator selects informative/uncertain samples,
a Digital Twin provides ground-truth labels through N-1 simulation, and
a Random Forest model is retrained via a deployed API.

## Where we are now
Iteration 2 design phase: defining a backend-ranking workflow for the 24
day-ahead samples. Waiting for UBITECH to respond to the
Iteration 2 proposal. In parallel: SiKDD paper
sections are in progress, and HCI/XAI evaluation design is in progress
with internal partners.

## Deployed and working (Iteration 1)
- Retraining API running on Atena (Docker image
  `leskovecg/smart-energy-api:1.6.0`, deployed 2026-06-29).
- MinIO-based appended-rows delta workflow for dataset updates.
- Random Forest model trained on 265 features (`load_*`, `gen_*`,
  `sgen_*`), following the 2026-06-29 data leakage fix.
- Run logging (`data/retrain_log.jsonl`) and per-class metrics on every
  retrain.
- Current metrics (production model, random split): accuracy=0.940,
  recall_insecure=0.943.
- Temporal split analysis: accuracy=0.909, recall_insecure=0.838 — a
  more realistic estimate that motivates continual AL retraining.

## In progress / blocked
- SiKDD paper: Results, Figures/tables, System-level evaluation
  sections.
- Blocked on UBITECH: response to the Iteration 2 proposal
  (candidate delivery method, ranked-list display, Digital Twin
  metadata, raw input source and feature groups).
- Blocked on IJS: HCI/XAI evaluation input. 

## If you are new here, read in this order
1. This file
2. [README.md](README.md) — how to run
3. [docs/architecture.md](docs/architecture.md) — system map (Mermaid,
   renders on GitHub)
4. [docs/INDEX.md](docs/INDEX.md) — full documentation map
5. Then the specific doc for your task (see INDEX)
