# Documentation Index

This file maps every document in `docs/` to its purpose and the situation in which you should read it.

---

## Active project docs (current, read these first)

| File | What it covers | Read when |
|---|---|---|
| [architecture.md](architecture.md) | System architecture overview — components, connections, MinIO paths (Mermaid flowchart) + Iteration 1 sequence diagram (now embedded here) | You need a visual map of all deployed components, their connections, or the message flow |
| [iteration-2-proposal.md](iteration-2-proposal.md) | Iteration 2 proposal: backend ranking of 24 day-ahead samples; proposed workflow, endpoint contract draft, open questions for Kostas | You are designing or extending the Iteration 2 AL ranking workflow |
| [retraining-api-current-state.md](retraining-api-current-state.md) | Snapshot of the deployed `retraining-api/` as of Jun 2026: endpoints, MinIO keys, payload schema, known limitations | You need to understand or extend the deployed API |
| [smart-energy-data-description.md](smart-energy-data-description.md) | Feature-by-feature description of the N-1 security dataset (columns, units, label definition) | You need to understand what the model is trained on or write the dataset section of the paper |
| [active-learning-project-guide.md](active-learning-project-guide.md) | Walk-through of `src/` code: experiment scripts, core AL loop, simulator interface, analysis | You need to run or modify offline/online AL experiments |
| [humaine-dashboard-guide.md](humaine-dashboard-guide.md) | How to use the deployed HumAIne dashboard UI for N-1 security monitoring and retraining | You need to demonstrate the system, write the paper's system description, or evaluate the UX |

---

## Reference / contract docs (in `retraining-api/`)

These live next to the code they describe, not in `docs/`.

| File | What it covers |
|---|---|
| [../retraining-api/INTEGRATION_CONTRACT.md](../retraining-api/INTEGRATION_CONTRACT.md) | API contract between the dashboard (UBITECH) and the retraining API: endpoint, payload, auth |
| [../retraining-api/ARCHITECTURE_SKETCH.txt](../retraining-api/ARCHITECTURE_SKETCH.txt) | Rough architecture diagram of the deployed system |
| [../retraining-api/README_DEPLOY.md](../retraining-api/README_DEPLOY.md) | How to build, run, and deploy the Docker container on Atena |

---

## Obsolete / historical (`docs/archive/`)

| File | Status |
|---|---|
| [archive/interim-report-no-minio.md](archive/interim-report-no-minio.md) | Pre-MinIO state; describes a workflow that no longer exists |
| [archive/run-guide-no-minio.md](archive/run-guide-no-minio.md) | Pre-MinIO run instructions; superseded by `retraining-api/README_DEPLOY.md` |
| [archive/datasets-and-minio-guide-slo.md](archive/datasets-and-minio-guide-slo.md) | Slovenian; superseded by `smart-energy-data-description.md` and `retraining-api-current-state.md` |
| [archive/day-ahead-load-and-security-explanation-slo.md](archive/day-ahead-load-and-security-explanation-slo.md) | Slovenian; superseded by `smart-energy-data-description.md` sections 13–15 |
| [archive/power-grid-dataset-learning-guide-slo.md](archive/power-grid-dataset-learning-guide-slo.md) | Slovenian personal learning resource; superseded by `smart-energy-data-description.md` |

- `archive/AGENT_INTAKE_PROMPT.md` — file removed; agent intake instructions are in Section 11 of `project_tracker.md` (gitignored).

---

## Historical / reference (`reports/`)

- `reports/architecture.txt` — early architecture sketch, predates `retraining-api/ARCHITECTURE_SKETCH.txt`. Historical only.
