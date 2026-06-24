# Iteration 2 Proposal — Backend Ranking of Day-Ahead Samples

**Status:** Sent to Kostas (UBITECH) and Magda on 2026-06-22. Jože Rožanec reviewed and approved 2026-06-23. Waiting for Kostas's response. See `project_tracker.md` Section 4 for current blockers.

---

## 1. Current Limitation (Iteration 1)

The deployed Iteration 1 workflow processes one sample at a time:

1. The operator views the 24-hour N-1 security classification graph in the dashboard.
2. The operator manually selects one operating point to trigger a simulation.
3. The Digital Twin runs an N-1 simulation for that single point and produces a ground-truth label.
4. The new labeled sample is appended to the dataset.
5. The retraining API retrains the Random Forest classifier.

**Problem:** Active Learning strategies (entropy, uncertainty, margin) require a *candidate pool* to rank meaningfully. Single-sample manual selection does not exploit the key AL advantage of selecting the *most informative* samples from a batch.

---

## 2. Proposed Iteration 2 Workflow

1. The dashboard collects all 24 day-ahead operating points (one per forecast hour).
2. The dashboard sends the full 24-sample candidate pool to our backend.
3. The backend scores each candidate using the selected AL strategy (entropy, uncertainty, margin, or random baseline).
4. The backend returns a ranked list with scores to the dashboard.
5. The operator reviews the ranked list and selects a subset (e.g., 5 samples) for simulation.
6. The selected samples are forwarded to the Digital Twin for N-1 simulation.
7. The Digital Twin returns ground-truth labels per sample.
8. New labeled samples are appended to the dataset in MinIO.
9. The retraining API is triggered (`POST /retrain`).

---

## 3. Draft Endpoint Contract

### New endpoint: `POST /rank`

**Purpose:** Receive a pool of day-ahead operating points; rank by informativeness; return the ranked list.

**Option A — samples inline in the request body:**

```json
{
  "al_strategy": "entropy",
  "candidates": [
    {
      "timestamp": "2026-06-23T08:00:00",
      "load_0_p_mw": 219.0,
      "gen_0_p_mw": 150.0,
      "sgen_0_p_mw": 30.0
    }
  ]
}
```

**Option B — samples stored in MinIO, key sent to backend:**

```json
{
  "al_strategy": "entropy",
  "candidates_bucket": "smart-energy-results",
  "candidates_key": "day_ahead/candidates_2026-06-23.csv"
}
```

**Response (draft):**

```json
{
  "ranked_candidates": [
    {
      "timestamp": "2026-06-23T08:00:00",
      "rank": 1,
      "al_score": 0.47,
      "p_insecure": 0.51
    }
  ]
}
```

**Open:** delivery method (Option A vs. B) and the exact metadata fields per sample are pending Kostas's answer — see Section 5.

---

## 4. Metadata per Ranked Sample

Minimum fields the backend can return:

| Field | Type | Description |
|---|---|---|
| `timestamp` | string | Hour of the operating point |
| `rank` | integer | AL rank (1 = most informative) |
| `al_score` | float | Raw informativeness score from the selected strategy |
| `p_insecure` | float | Model's predicted probability of insecurity |

Optional fields (pending Digital Twin support):

| Field | Type | Description |
|---|---|---|
| `contingency_id` | string | Critical N-1 contingency element |
| `overloaded_line_id` | string | ID of the line that would be overloaded |
| `bus_voltage_violation` | string | Bus ID with voltage outside [0.90, 1.10] pu |
| `line_loading_percent` | float | Loading percent of the critical line |

---

## 5. Open Questions for Kostas / UBITECH

1. Can the dashboard provide a pool of all 24 day-ahead operating points to our backend per request?
2. Preferred delivery method: samples inline in the request body (Option A), or as a MinIO object key (Option B)?
3. Can the dashboard display the ranked candidate list returned by our backend, so the operator can select a subset?
4. Can the Digital Twin return per-sample metadata after simulation (contingency ID, overloaded line ID, voltage-violating bus)?
5. Source of the raw `load`, `gen`, `sgen` input values for day-ahead operating points — ENTSO-E, SCADA, historical profiles? Are they generated on a scheduled day-ahead basis?
6. Which exact feature groups does the current dashboard prediction model use (`load_*`, `gen_*`, `sgen_*`, or a subset)?

---

## 6. Relation to Phase 1

Phase 1 (decided 2026-06-22, tracked in `project_tracker.md`) is a minimal step that does **not** require dashboard changes:

- `al_strategy` is added to the existing `/retrain` payload as a recorded metadata field.
- The operator continues to select samples manually.
- The strategy field is stored for tracking and future extension only.
- Phase 1 does **not** implement candidate ranking or the `/rank` endpoint.

Iteration 2 is the full batch-ranking workflow and requires UBITECH dashboard cooperation (questions in Section 5).
