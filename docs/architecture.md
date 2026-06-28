# System Architecture — Smart Energy Active Learning

> **Status:** Iteration 1 deployed. Iteration 2 (batch ranking via `/rank`) is proposed —
> see `docs/iteration-2-proposal.md`.

Six components in two layers. Steps 1–6 describe the current operator-driven
retraining workflow.

```mermaid
flowchart TD
    OP([Operator — TSO user])

    subgraph ubitech["UBITECH — dashboard.euprojects.net"]
        DASH[Dashboard UI]
        UBI[UBITECH Backend — Orchestrator]
        DT[Digital Twin — N-1 Simulator]
    end

    MINIO[(MinIO — smart-energy-results)]

    subgraph ijs["IJS — Atena"]
        API[Retraining API — Docker · :5004]
    end

    OP -->|1 · view & select| DASH
    DASH <-->|2 · trigger analysis| UBI
    UBI <-->|3 · N-1 simulation| DT
    UBI -->|4 · append delta CSV| MINIO
    UBI <-->|5 · POST /retrain| API
    API <-->|6 · download delta / upload model| MINIO
```

## Component summary

### Operator

| Property | Value |
|---|---|
| Role | Selects uncertain samples (p_insecure ≈ 0.5) to trigger Digital Twin simulation |
| Interface | `humaine-dashboard.euprojects.net` |
| Iteration 1 | Selects one sample at a time — no AL strategy ranking yet |
| Iteration 2 | Will review ranked list of 24 day-ahead candidates and pick a subset |

### Dashboard UI

| Property | Value |
|---|---|
| Owner | UBITECH — Kostas Mylonas, Magda Foti |
| URL | `humaine-dashboard.euprojects.net` |
| Displays | p_insecure per hour: green bars 00:00–23:00 |
| Iteration 2 | Needs to show ranked candidate list from `/rank` endpoint |

### UBITECH Backend

| Property | Value |
|---|---|
| Owner | UBITECH — Kostas Mylonas |
| Role | Orchestrates: DT simulation → MinIO append → API retrain |
| Calls our API | `http://atena.ijs.si:5004/retrain` |
| Open question | Does dashboard send `latest_key`? (pending Kostas confirmation) |

### Digital Twin

| Property | Value |
|---|---|
| Owner | UBITECH |
| Input | `load_*`, `gen_*`, `sgen_*` features + timestamp |
| Output | label: secure / insecure + physical metrics |
| Insecure if | Line >100%, voltage outside [0.90–1.10] pu, or non-convergence |
| Iteration 2 | Should also return: `contingency_id`, `overloaded_line_id` |

### MinIO

| Property | Value |
|---|---|
| Bucket | `smart-energy-results` |
| Delta CSV | `al_training_dataset/appended_rows/...appended_rows_latest.csv` |
| Full CSV | `al_training_dataset/simulation_security_labels_n-1.csv` (21.4 MB — NOT fetched per retrain) |
| Model out | `models/retraining_runs/<timestamp>/model.joblib` |
| Metrics out | `models/retraining_runs/<timestamp>/metrics.json` |

### Retraining API

| Property | Value |
|---|---|
| Owner | IJS |
| URL | `http://atena.ijs.si:5004` |
| Docker image | `leskovecg/smart-energy-api:1.3.0` |
| Ports | Host 5004 → Container 8000 |
| Volume | `./data:/app/data` (base + appended CSV accessible in container) |
| Base CSV | `/home/gleskovec/retraining-api/data/base/simulation_security_labels_n-1.csv` |
| Auth | X-API-Key header required for POST `/retrain` |
| Endpoints now | `GET /health` \| `POST /retrain` |
| Start script | `start-app.sh` (Atena only — not in git repo) |
| Pending | al_strategy logic, `/rank` endpoint, XAI |

## Key MinIO paths

| Artifact | Key |
|---|---|
| Delta CSV | `al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv` |
| Full base CSV | `al_training_dataset/simulation_security_labels_n-1.csv` (21.4 MB — not fetched per retrain) |
| Trained model | `models/retraining_runs/<timestamp>/model.joblib` |
| Metrics | `models/retraining_runs/<timestamp>/metrics.json` |

> Full API spec: `docs/retraining-api-current-state.md`

## Sequence Diagram — Iteration 1 Deployed Workflow

```mermaid
sequenceDiagram
    actor Operator
    participant Dashboard as Dashboard UI
    participant UBITECH as UBITECH Backend
    participant DT as Digital Twin
    participant MinIO
    participant API as Retraining API

    Operator->>Dashboard: Open dashboard
    Dashboard->>UBITECH: GET day-ahead p_insecure values
    UBITECH-->>Dashboard: p_insecure per hour (24 values)
    Dashboard-->>Operator: Show probability bars 00:00–23:00

    Note over Operator,Dashboard: Operator selects bar with p_insecure ≈ 0.5

    Operator->>Dashboard: Select uncertain operating point
    Dashboard->>UBITECH: Trigger analysis for selected point

    UBITECH->>DT: Run N-1 simulation
    Note over DT: Input: load_*, gen_*, sgen_* features + timestamp
    DT-->>UBITECH: label (secure/insecure) + physical metrics

    UBITECH->>MinIO: Append labeled row to delta CSV
    Note over MinIO: al_training_dataset/appended_rows/...appended_rows_latest.csv

    UBITECH->>API: POST /retrain
    Note over API: { latest_key, al_strategy } — al_strategy pending Phase 1

    API->>MinIO: Download appended rows delta
    API->>API: Merge with local base dataset
    API->>API: Retrain RandomForestClassifier (n_estimators=400)
    API->>MinIO: Upload model.joblib + metrics.json
    API-->>UBITECH: 200 OK — model_object, metrics_object, metrics

    UBITECH-->>Dashboard: Update status and metrics
    Dashboard-->>Operator: Show retraining result

    Note over Operator,API: Iteration 1 — operator-driven. No AL candidate ranking yet.
```

> For proposed Iteration 2 workflow see `docs/iteration-2-proposal.md`.
