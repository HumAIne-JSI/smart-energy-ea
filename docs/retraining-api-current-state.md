# Retraining API Current State

Last updated: 2026-06-27

Scope: Secrets from `.env` are redacted throughout.

## 1. What This API Does

The implemented API is a small FastAPI service for retraining the Smart Energy N-1 security classification model.

Implemented responsibilities:

- Exposes `GET /health`.
- Exposes `POST /retrain`.
- Requires `X-API-Key` for `/retrain` only when `API_KEY` is set in the environment.
- Logs into the HumAIne MinIO API wrapper using `HUMAINE_API_BASE_URL`, `HUMAINE_API_USERNAME`, and `HUMAINE_API_PASSWORD`.
- Downloads the latest appended-rows CSV from MinIO.
- Reads a fixed local base dataset CSV from disk.
- Concatenates base rows and appended rows locally.
- Trains a `sklearn.ensemble.RandomForestClassifier`.
- Writes a temporary `model.joblib` and `metrics.json`.
- Uploads both artifacts to MinIO under a timestamped run folder.

The current code implements retraining only. It does not implement prediction, candidate selection, active-learning ranking, XAI, or domain-aware sample selection endpoints.

## 2. How It Runs Locally and in Docker

### Docker

`Dockerfile`:

- Uses `python:3.10-slim`.
- Sets `WORKDIR /app`.
- Installs `requirements.txt`.
- Copies only `app/` and `vendor/`.
- Sets `PYTHONPATH=/app:/app/vendor`.
- Runs:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`docker-compose.yml`:

- Builds from `retraining-api/`.
- Starts container `retraining-api`.
- Loads `.env`.
- Maps `8000:8000`.
- Uses `restart: unless-stopped`.

Local Docker command from `retraining-api/`:

```bash
docker compose up -d --build
curl http://localhost:8000/health
```

### Important Docker Data Issue

The Dockerfile does not copy `data/`, and `.dockerignore` excludes `data/`. The compose file also has no volume mount. Therefore, a container built from the current checked-in files will not contain:

```text
/app/data/base/simulation_security_labels_n-1.csv
```

Unless Atena has a modified compose file, a manually injected file, or `BASE_DATASET_LOCAL_PATH` pointing to an accessible path inside the container, `POST /retrain` will fail with "Local base dataset not found".

Unknown / needs confirmation

### Manual Local Run

Because `app/minio_io.py` imports `minio_humaine_client` from `vendor/`, local manual execution should include `vendor` on `PYTHONPATH`.

PowerShell from `retraining-api/`:

```powershell
$env:PYTHONPATH = "$PWD;$PWD\vendor"
$env:HUMAINE_API_BASE_URL = "<REDACTED>"
$env:HUMAINE_API_USERNAME = "<REDACTED>"
$env:HUMAINE_API_PASSWORD = "<REDACTED>"
$env:API_KEY = "<REDACTED>"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Bash/Linux from `retraining-api/`:

```bash
export PYTHONPATH="$PWD:$PWD/vendor"
export HUMAINE_API_BASE_URL="<REDACTED>"
export HUMAINE_API_USERNAME="<REDACTED>"
export HUMAINE_API_PASSWORD="<REDACTED>"
export API_KEY="<REDACTED>"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The root `Makefile` `make run-api` target runs `cd retraining-api && PYTHONPATH=.:vendor python -m uvicorn app.main:app`. The working directory is changed to `retraining-api/` and `PYTHONPATH` includes both `retraining-api/` and `retraining-api/vendor/`, so the module path and vendored client resolve correctly.

## 3. Existing Endpoints

### `GET /health`

Purpose: simple health check.

Authentication: none.

Request JSON: none.

Response JSON:

```json
{
  "status": "ok"
}
```

Example curl:

```bash
curl http://localhost:8000/health
```

Likely Atena form:

```bash
curl http://atena.ijs.si:5004/health
```

Note: the project tracker says Atena is exposed at `http://atena.ijs.si:5004`, while the checked-in Docker Compose maps `8000:8000`. The exact external Atena port needs confirmation.

Unknown / needs confirmation

### `POST /retrain`

Purpose: download appended rows, merge with local base CSV, train a Random Forest classifier, upload model and metrics to MinIO.

Authentication: protected by `X-API-Key` only if `API_KEY` is set.

Minimal request JSON currently supported by defaults:

```json
{}
```

This uses default values from `RetrainRequest`, especially:

```json
{
  "results_bucket": "smart-energy-results",
  "latest_key": "al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv",
  "output_prefix": "models/retraining_runs",
  "n_estimators": 400,
  "random_state": 42,
  "test_size": 0.2,
  "label_col": "status",
  "drop_feature_cols": [
    "timestamp",
    "max_line_loading_percent_basecase",
    "min_bus_voltage_pu_basecase",
    "max_bus_voltage_pu_basecase",
    "max_line_loading_percent_contingency",
    "min_bus_voltage_pu_contingency",
    "max_bus_voltage_pu_contingency"
  ],
  "drop_latest_columns": ["created_at"]
}
```

Recommended explicit request JSON:

```json
{
  "results_bucket": "smart-energy-results",
  "latest_key": "al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv",
  "output_prefix": "models/retraining_runs",
  "n_estimators": 400,
  "random_state": 42,
  "test_size": 0.2,
  "label_col": "status",
  "drop_feature_cols": [
    "timestamp",
    "max_line_loading_percent_basecase",
    "min_bus_voltage_pu_basecase",
    "max_bus_voltage_pu_basecase",
    "max_line_loading_percent_contingency",
    "min_bus_voltage_pu_contingency",
    "max_bus_voltage_pu_contingency"
  ],
  "drop_latest_columns": ["created_at"]
}
```

> **Response schema:** see `retraining-api/INTEGRATION_CONTRACT.md`
> — Successful response example section.
> `INTEGRATION_CONTRACT.md` is the single source of truth for
> the request and response contract.

Example curl:

```bash
curl -X POST http://localhost:8000/retrain \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "latest_key": "al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv"
  }'
```

PowerShell:

```powershell
$body = @{
  latest_key = "al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv"
} | ConvertTo-Json

Invoke-WebRequest -Uri http://localhost:8000/retrain `
  -Method POST `
  -Headers @{"X-API-Key" = $env:API_KEY} `
  -ContentType "application/json" `
  -Body $body
```

Likely Atena form:

```bash
curl -X POST http://atena.ijs.si:5004/retrain \
  -H "X-API-Key: <REDACTED>" \
  -H "Content-Type: application/json" \
  -d '{
    "latest_key": "al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv"
  }'
```

Unknown / needs confirmation

## 4. What Happens When `POST /retrain` Is Called

Step-by-step from the current code:

1. FastAPI validates the JSON body against `RetrainRequest`.
2. If `API_KEY` is set, `verify_api_key()` requires `X-API-Key` to match it.
3. `get_humaine_auth()` reads MinIO wrapper credentials from environment variables and logs in to the HumAIne API.
4. The code chooses the input key:

```python
latest_key = req.latest_key
```

5. It checks whether the local base dataset exists at `BASE_DATASET_LOCAL_PATH`, defaulting to:

```text
retraining-api/data/base/simulation_security_labels_n-1.csv
```

6. It creates the local appended-rows parent directory, defaulting to:

```text
retraining-api/data/appended/
```

7. It creates a run ID:

```text
YYYYMMDDTHHMMSSZ_<8 hex chars>
```

8. It computes output object keys:

```text
<output_prefix>/<run_id>/model.joblib
<output_prefix>/<run_id>/metrics.json
```

9. It downloads the appended rows CSV from MinIO into:

```text
retraining-api/data/appended/simulation_security_labels_n-1_appended_rows_latest.csv
```

10. It reads the local base CSV with Pandas.
11. It reads the appended CSV with Pandas. If the appended CSV is empty, it creates an empty dataframe with the base columns.
12. It concatenates base + appended rows with `ignore_index=True`.
13. It writes the merged dataset to a temporary `merged.csv`.
14. It trains the model using `train_rf_and_save()`.
15. It overwrites `metrics["n_rows_latest"]` so it means appended-row count, not merged-row count.
16. It uploads `model.joblib` and `metrics.json` to MinIO.
17. It appends a JSON log entry to `data/retrain_log.jsonl` (relative to the Docker volume root). Each entry records the run_id, timestamp, full request parameters, response summary, and the complete metrics dict. Write errors are non-fatal — they are logged to stderr only and do not affect the API response.
18. It returns paths and metrics in the HTTP response.

## 5. Fields Accepted by `/retrain`

Fields defined in `RetrainRequest`:

| Field | Type | Default | Current behavior |
|---|---:|---|---|
| `results_bucket` | string | `smart-energy-results` | Used both for reading appended rows and uploading artifacts. |
| `latest_key` | string | `al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv` | Primary MinIO object key for appended rows. |
| `output_prefix` | string | `models/retraining_runs` | Prefix for uploaded run artifacts. |
| `n_estimators` | integer >= 1 | `400` | Passed to `RandomForestClassifier`. |
| `random_state` | integer | `42` | Used in train/test split and RF model. |
| `test_size` | float between 0 and 1 | `0.2` | Passed to `train_test_split`. |
| `label_col` | string | `status` | Target column. |
| `drop_feature_cols` | string list | `["timestamp", "max_line_loading_percent_basecase", ...]` | Columns excluded from feature matrix. Defaults exclude timestamp, post-simulation physical outputs (basecase and contingency line loading and voltages) which directly determine the label. Effective feature set: load_*, gen_*, sgen_* only. |
| `drop_latest_columns` | string list | `["created_at"]` | Dropped from the merged dataset if present. |

### `al_strategy`

**Phase 1 status (decision 2026-06-22):** `al_strategy` is being added to `RetrainRequest` as a metadata field for tracking purposes. Supported values: `entropy`, `uncertainty`, `margin`, `random`. In Phase 1 the field is stored as retraining metadata only and does not influence which samples are selected — the operator still selects points manually in the dashboard. The code change is tracked in the active sprint in `project_tracker.md`.

Prior to the Phase 1 update: the field was not defined in `RetrainRequest` and was silently ignored by Pydantic's default extra-field handling if sent.

## 6. Datasets and Files Read/Written

### Read from local disk

Default base dataset:

```text
retraining-api/data/base/simulation_security_labels_n-1.csv
```

The path can be overridden with:

```text
BASE_DATASET_LOCAL_PATH
```

Current checkout status: missing.

### Downloaded from MinIO and written locally

Default local appended rows path:

```text
retraining-api/data/appended/simulation_security_labels_n-1_appended_rows_latest.csv
```

The path can be overridden with:

```text
APPENDED_ROWS_LOCAL_PATH
```

This file is overwritten/refreshed by `/retrain`.

### Temporary files

Created inside a Python `TemporaryDirectory()` during each request:

```text
merged.csv
model.joblib
metrics.json
```

The temporary directory is deleted after the request finishes.

### Uploaded to MinIO

```text
<results_bucket>/<output_prefix>/<run_id>/model.joblib
<results_bucket>/<output_prefix>/<run_id>/metrics.json
```

Default:

```text
smart-energy-results/models/retraining_runs/<run_id>/model.joblib
smart-energy-results/models/retraining_runs/<run_id>/metrics.json
```

## 7. MinIO Buckets and Object Keys

Current default bucket:

```text
smart-energy-results
```

Current default appended rows object key:

```text
al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv
```

Current default output prefix:

```text
models/retraining_runs
```

Current output object keys:

```text
models/retraining_runs/<run_id>/model.joblib
models/retraining_runs/<run_id>/metrics.json
```

MinIO/HumAIne API endpoints used by the wrapper:

- Login: `<HUMAINE_API_BASE_URL>/auth/auth`
- List buckets: `<HUMAINE_API_BASE_URL>/main_ops/buckets`
- Download candidates:
  - `<HUMAINE_API_BASE_URL>/data/download?key=<key>`
  - `<HUMAINE_API_BASE_URL>/main_ops/download?bucket_name=<bucket>&object_name=<key>`
  - `<HUMAINE_API_BASE_URL>/main_ops/download/<bucket>/<key>`
- Upload: `<HUMAINE_API_BASE_URL>/main_ops/upload`

## 8. Model Trained and Metrics Returned

Model:

```python
sklearn.ensemble.RandomForestClassifier(
    n_estimators=<request value, default 400>,
    random_state=<request value, default 42>,
    n_jobs=-1
)
```

Training behavior:

- Reads merged CSV.
- Drops metadata columns listed in `drop_latest_columns`, default `["created_at"]`.
- Uses `label_col`, default `status`, as the target.
- Drops any columns listed in `drop_feature_cols`.
- Coerces non-numeric feature columns to numeric with `errors="coerce"`.
- Fills NaN feature values with `0.0`.
- Encodes labels by sorting string labels and assigning integer IDs.
- Uses `train_test_split()` with `test_size`, `random_state`, and stratification if more than one label is present.
- Trains a Random Forest classifier.
- Saves only the classifier object with `joblib.dump()`.

> Response schema: see `retraining-api/INTEGRATION_CONTRACT.md`
> — Successful response example and Response fields sections.

The model artifact does not include an explicit preprocessing pipeline or label-decoding wrapper. The metrics file includes `label_mapping`, but a downstream prediction service would need to handle feature preparation and label interpretation carefully.

## 9. Required Environment Variables

Actually required by current code for `/retrain`:

```text
HUMAINE_API_BASE_URL=<REDACTED>
HUMAINE_API_USERNAME=<REDACTED>
HUMAINE_API_PASSWORD=<REDACTED>
```

Used by current code for API protection:

```text
API_KEY=<REDACTED>
```

If `API_KEY` is unset, `/retrain` is not protected by `X-API-Key`. For Atena, the `.env` file contains `API_KEY`, with value redacted here.

Optional path overrides read by current code (listed in `.env.example`):

```text
BASE_DATASET_LOCAL_PATH=<path to base CSV>
APPENDED_ROWS_LOCAL_PATH=<path where appended CSV should be downloaded>
```

Both have sensible defaults (see `main.py` lines 24–34) and only need to be set to override the default paths inside the container.

## 10. Not Implemented Yet

Not implemented in `retraining-api/app`:

- AL strategy-based sample ranking (Iteration 2: backend receives a 24-sample candidate pool and returns a ranked list).
- Candidate pool selection endpoint.
- Entropy, margin, uncertainty, or random strategy endpoint.
- XAI/SHAP/feature importance endpoint.
- Prediction endpoint.
- `p_insecure` endpoint.
- Endpoint to return uncertain samples.
- Endpoint to return candidate samples for operator confirmation.
- Domain-aware or risk-aware selection using grid details.
- Per-sample grid details such as overloaded line ID, bus voltage violation, contingency ID, affected grid element, or affected segment.
- Model registry metadata beyond `model.joblib` and `metrics.json`.
- Monitoring, tracing, or middleware-level structured logging. Basic per-run logging for successful `/retrain` calls is implemented — see Section 15.
- CI/CD or deployment automation beyond Docker Compose.

The architecture sketch says the dashboard implements active-learning strategy selection, but this is not implemented in the retraining API.

## 12. Main Risks, Unclear Points, and Questions for Costas/Klemen

### Main Risks

1. Docker deployment may not include the local base dataset.
   - Current Dockerfile copies only `app/` and `vendor/`.
   - `.dockerignore` excludes `data/`.
   - `docker-compose.yml` has no data volume.
   - `/retrain` will fail if `/app/data/base/simulation_security_labels_n-1.csv` is absent.

2. Documentation drift between old snapshot flow and current delta flow.
   - `ARCHITECTURE_SKETCH.txt` and `INTEGRATION_CONTRACT.md` describe `latest_key` as a full dataset snapshot.
   - Current code and `README_DEPLOY.md` use `latest_key` as a delta/appended-rows file merged with local base data.

3. `latest_key` compatibility is likely broken or misleading.
   - Because `latest_key` has a default, a body containing only `latest_key` will not use that `latest_key`; it will use the default appended-rows key.

4. The external Atena port is unclear.
   - Tracker says `http://atena.ijs.si:5004`.
   - Compose maps `8000:8000`.

5. The trained model artifact is only the classifier.
   - There is no saved preprocessing pipeline.
   - There is no bundled label decoder beyond `label_mapping` in metrics.

6. Data validation is minimal.
   - The API concatenates base and appended data without checking schema compatibility, duplicate rows, timestamp consistency, label validity, or class coverage before train/test split.

7. Per-run logging is implemented for successful calls (`data/retrain_log.jsonl`).
   Error-level logging is still unstructured — failures are returned as HTTP errors but not written to a log file.

8. MinIO wrapper download path is defensive but uncertain.
   - It tries three download endpoint variants.
   - The comments suggest some uncertainty about which endpoint is correct.

9. No implemented AL/XAI workflow in this service.
   - The API currently retrains after samples already exist in MinIO. It does not choose or explain samples.

### Questions for Costas

1. Does the dashboard currently upload a delta file at:

```text
smart-energy-results/al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv
```

or a full snapshot at:

```text
smart-energy-results/al_training_dataset/simulation_security_labels_n-1_latest.csv
```

2. Which request body is the deployed dashboard actually sending to `/retrain`: `latest_key`, `latest_key`, or an empty body?

3. Is the dashboard using `http://atena.ijs.si:5004`, `:8000`, or another routed URL?

4. Does the dashboard expect `model_object` and `metrics_object` paths only, or does it also consume the returned inline `metrics` object?

5. Should `al_strategy` be sent to this API in the next iteration, or should strategy selection remain dashboard-side?

6. Is there an expected endpoint contract for candidate sample selection, e.g. `/candidates`, `/select`, or `/rank`?

7. Can the dashboard or Digital Twin provide per-sample grid details such as contingency ID, overloaded line ID, violated bus, voltage range violation, line loading, or affected network segment?

8. Should XAI outputs be per-candidate, per-prediction, or only global model-level feature importances?

### Questions for Klemen

1. Should the retraining API continue using local base dataset + appended delta, or return to full snapshot training from MinIO?

2. Should the model artifact include a preprocessing/metadata wrapper instead of only a bare Random Forest classifier?

3. Should we add schema validation for appended rows before training?

4. **Resolved (2026-06-28):** `precision_insecure`, `recall_insecure`, `f1_insecure`, and `roc_auc` are now returned in the `metrics` response object. `recall_insecure` is the most critical metric — a false negative (predicting secure when insecure) is the most dangerous error in N-1 classification. `pr_auc` was not added; `roc_auc` is sufficient for binary classification. See `INTEGRATION_CONTRACT.md` for the full response schema.

5. Should the next backend milestone be a candidate-selection endpoint with `al_strategy`, or should the paper describe this as future work until Costas/Magda confirm frontend support?

6. Should XAI for N-1 security classification be implemented now, or kept separate from day-ahead forecasting XAI until the dashboard contract is confirmed?

## 13. Design Note: Local Base Dataset and MinIO Delta

The `/retrain` endpoint merges a **local base dataset** with a small **delta CSV** downloaded from MinIO, rather than pulling the full dataset from MinIO on every request.

**Reason:** downloading the full `simulation_security_labels_n-1.csv` (~22 MB) from MinIO on every retrain caused timeout errors. The agreed solution (with Kostas) is to provision the base dataset locally once inside the container or via a mounted volume, and fetch only the small `appended_rows_latest.csv` delta per request.

## 14. Pre-Retrain Validation Checklist

Before triggering `/retrain`, verify:

- [ ] Local base dataset `simulation_security_labels_n-1.csv` exists at `BASE_DATASET_LOCAL_PATH`.
- [ ] MinIO object `simulation_security_labels_n-1_appended_rows_latest.csv` is accessible under `results_bucket`.
- [ ] Appended CSV contains the `status` column.
- [ ] `status` values are only `secure` or `insecure`.
- [ ] Feature columns in appended rows are compatible with the base dataset schema.
- [ ] Metadata columns such as `created_at` are listed in `drop_latest_columns` and will not enter model features.
- [ ] Merged dataset has enough rows of both classes for a stratified train/test split.
- [ ] After retraining, both `model.joblib` and `metrics.json` are uploaded to MinIO under the expected run path.

## 15. Local Run Log

After each successful `/retrain` call, one JSON line is appended to:

```text
data/retrain_log.jsonl
```

This file lives inside the Docker volume mount (`data/`) and persists across container restarts.

Each log entry contains:

- `run_id` — unique run identifier (`YYYYMMDDTHHMMSSZ_<8 hex chars>`)
- `called_at` — ISO 8601 UTC timestamp of when the entry was written
- `request` — full request parameters (`latest_key`, `results_bucket`, `output_prefix`, `n_estimators`, `random_state`, `test_size`, `label_col`, `drop_feature_cols`, `drop_latest_columns`)
- `response` — response summary (`ok`, `dataset_rows`, `dataset_rows_latest`, `model_object`, `metrics_object`)
- `metrics` — complete metrics dict (all fields returned in the API response `metrics` object)

Write errors are non-fatal: a log write failure is printed to stderr only and does not cause the `/retrain` endpoint to return an error.

View all runs in a readable format (from `retraining-api/` on Atena):

```bash
cat data/retrain_log.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    print(f\"{r['called_at']} | acc={r['metrics']['accuracy']:.3f} | recall_ins={r['metrics'].get('recall_insecure','?')} | rows={r['metrics']['n_rows_all']} | latest={r['metrics']['n_rows_latest']}\")
"
```

