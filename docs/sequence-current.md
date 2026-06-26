# Sequence Diagram — Iteration 1 Workflow (Deployed)

> **Status:** Deployed and running on Atena.
> AL strategies are **not yet used** for sample ranking. The operator selects samples manually.

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
    Note over API: { appended_rows_key, al_strategy } — al_strategy pending Phase 1

    API->>MinIO: Download appended rows delta
    API->>API: Merge with local base dataset
    API->>API: Retrain RandomForestClassifier (n_estimators=400)
    API->>MinIO: Upload model.joblib + metrics.json
    API-->>UBITECH: 200 OK — model_object, metrics_object, metrics

    UBITECH-->>Dashboard: Update status and metrics
    Dashboard-->>Operator: Show retraining result

    Note over Operator,API: Iteration 1 — operator-driven. No AL candidate ranking yet.
```

## Notes

- Ground-truth labels always come from the Digital Twin / N-1 simulation — the operator
  does not manually assign labels.
- The base dataset (21.4 MB) lives locally on Atena and is never re-downloaded from MinIO.
- **Phase 1 (pending):** `al_strategy` added to `/retrain` payload as metadata only.
- **Iteration 2 (proposed):** new `/rank` endpoint for batch ranking of 24 day-ahead
  candidates. See `docs/iteration-2-proposal.md`.
