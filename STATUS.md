# Project Status
> Last updated: 2026-07-15 — refreshed after each milestone.
> For full docs navigation see [docs/INDEX.md](docs/INDEX.md).

## What this project is
This project supports an Active Learning workflow for power-grid N-1
security assessment. The operator selects informative/uncertain samples,
a Digital Twin provides ground-truth labels through N-1 simulation, and
a Random Forest model is retrained via a deployed API.

## Where we are now
Iteration 2 implementation phase: the backend-ranking workflow was
finalized in a 2026-07-15 call with UBITECH (Kostas). JSI is implementing
a new `/rank` endpoint and a `model_latest.joblib` upload on `/retrain`,
targeting 2026-07-21. UBITECH is verifying production deployment of
`al_strategy` and will build the ranked-list frontend the week of
2026-07-27. SiKDD paper work continues in parallel, with a hard
wrap-up deadline of end of July.

## Deployed and working (Iteration 1)
- Retraining API running on Atena (Docker image
  `leskovecg/smart-energy-api:1.7.0`, deployed 2026-06-29).
- `al_strategy` metadata field (Phase 1) verified 2026-07-03 with a live
  retrain run: recorded in `retrain_log.jsonl`, echoed in the response,
  patched into `metrics.json`.
- MinIO-based appended-rows delta workflow for dataset updates.
- Random Forest model trained on 265 features (`load_*`, `gen_*`,
  `sgen_*`), following the 2026-06-29 data leakage fix.
- Run logging (`data/retrain_log.jsonl`) and per-class metrics on every
  retrain.
- Current metrics (production model, random split): accuracy=0.940,
  recall_insecure=0.943.
- Temporal split analysis: accuracy=0.909, recall_insecure=0.838 — a
  more realistic estimate that motivates continual AL retraining.

## Agreed for Iteration 2 (implementing now)
- `/rank` endpoint contract agreed with UBITECH: dashboard sends 24
  candidates + chosen AL strategy; backend returns each candidate's
  rank and `al_score` (no threshold, no `p_insecure`).
- Fix for the broken model loop: `/retrain` will also upload to a fixed
  key, `model_latest.joblib`, so the dashboard always serves the latest
  retrained model instead of a stale manually-uploaded one.
- Timeline: JSI backend by 2026-07-21; UBITECH frontend week of
  2026-07-27.

## In progress / blocked
- SiKDD paper: Results, Figures/tables, System-level evaluation
  sections.
- Waiting on UBITECH: confirmation that `al_strategy` is deployed to
  the production VM (target 2026-07-21).
- Waiting on UBITECH: frontend work for the ranked list and
  `model_latest` switch (week of 2026-07-27).
- Waiting on IJS (Jože/Ivana): timeline for the HCI/XAI evaluation
  section draft.

## If you are new here, read in this order
1. This file
2. [README.md](README.md) — how to run
3. [docs/architecture.md](docs/architecture.md) — system map (Mermaid,
   renders on GitHub)
4. [docs/INDEX.md](docs/INDEX.md) — full documentation map
5. Then the specific doc for your task (see INDEX)
