# Smart Energy – Retraining API (Deploy)

This folder contains the FastAPI service that retrains the Random Forest model using a fixed local base dataset plus latest appended rows produced by the dashboard, and stores artifacts (model + metrics) back to MinIO.

## What it does
- Exposes:
  - `GET /health` – service health check
  - `POST /retrain` – triggers retraining
- Reads from MinIO:
  - appended rows latest CSV only (`al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv`)
- Reads from local disk:
  - base dataset (`data/base/simulation_security_labels_n-1.csv`)
- Writes to MinIO:
  - `model.joblib`
  - `metrics.json`
  - under a versioned run folder (`models/retraining_runs/<run_id>/...`)

## Code structure

Source files live in `app/`:

- `main.py` — FastAPI application, endpoint definitions, auth middleware
- `train.py` — Random Forest training and evaluation logic
- `minio_io.py` — MinIO I/O: download delta CSV, upload model and metrics
- `schemas.py` — Pydantic request and response schemas

## Prerequisites
- Docker + Docker Compose
- Access to HumAIne MinIO / API (credentials)

## Configuration
1) Copy the example env file:
```bash
cp .env.example .env
```

2) Edit `.env` and set:
- `HUMAINE_API_BASE_URL`
- `HUMAINE_API_USERNAME`
- `HUMAINE_API_PASSWORD`
- `API_KEY` (used to protect `/retrain`)

3) Prepare local base dataset once:
```bash
mkdir -p data/base data/appended
```
Windows PowerShell alternative:
```powershell
New-Item -ItemType Directory -Force data/base, data/appended | Out-Null
```
Copy the large base CSV to:
```text
data/base/simulation_security_labels_n-1.csv
```

## Build & run (local)
From inside `retraining-api/`:

```bash
docker compose up -d --build
```

Health check:
```bash
curl http://localhost:8000/health
```

## Trigger retraining

**Bash/Linux:**
```bash
curl -X POST http://localhost:8000/retrain \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "latest_key": "al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv"
  }'
```

**PowerShell (minimal):**
```powershell
$body = @{
  latest_key = "al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv"
} | ConvertTo-Json

$response = Invoke-WebRequest -Uri http://localhost:8000/retrain `
  -Method POST `
  -Headers @{"X-API-Key" = $env:API_KEY} `
  -ContentType "application/json" `
  -Body $body

$response.Content
```

**PowerShell (full options):**
```powershell
$body = @{
    results_bucket = "smart-energy-results"
    latest_key = "al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv"
    output_prefix = "models/retraining_runs"
    n_estimators = 400
    random_state = 42
    test_size = 0.2
    label_col = "status"
    drop_feature_cols = @()
    drop_latest_columns = @("created_at")
} | ConvertTo-Json

$response = Invoke-WebRequest -Uri http://localhost:8000/retrain `
  -Method POST `
  -Headers @{"X-API-Key" = $env:API_KEY} `
  -ContentType "application/json" `
  -Body $body

$response.Content
```

**Note:** `latest_key` has a default and normally does not need to be sent.

## Logs
```bash
docker logs -f retraining-api
```

## Verify data inside container
```bash
docker exec retraining-api ls -lh /app/data/base/
```

## View run log

Each successful `/retrain` call appends a structured JSON line to
`data/retrain_log.jsonl` inside the Docker volume (persists across
container restarts). View all runs in a readable format from
`retraining-api/` on Atena:

```bash
cat data/retrain_log.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    print(f\"{r['called_at']} | acc={r['metrics']['accuracy']:.3f} | recall_ins={r['metrics'].get('recall_insecure','?')} | rows={r['metrics']['n_rows_all']} | latest={r['metrics']['n_rows_latest']}\")
"
```

## Stop
```bash
docker compose down
```

## Deployment on Atena (summary)
1) Copy `retraining-api/` folder to Atena
2) Set `.env` on Atena
3) Run:
```bash
docker compose up -d --build
```
4) Share the API URL and `X-API-Key` with the dashboard integrator.
