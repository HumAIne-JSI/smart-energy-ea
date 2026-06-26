# Smart Energy – Retraining API Integration Contract

This document describes how the HumAIne dashboard integrates with the Smart Energy retraining API.

---

## Base URL

The API is deployed on Atena, exposed on host port **5004** (the container binds to port 8000 internally):

```
http://atena.ijs.si:5004
```

For local development (run from inside `retraining-api/`):

```
http://localhost:8000
```

---

## Authentication
All requests to protected endpoints must include the following header:

```
X-API-Key: <SHARED_SECRET>
```

If the header is missing or invalid, the API returns **401 Unauthorized**.

---

## Endpoints

### Health check
**GET** `/health`

Response:
```json
{
  "status": "ok"
}
```

This endpoint does not require authentication.

---

### Retrain model
**POST** `/retrain`

Triggers retraining of the Random Forest model by downloading the latest **appended rows (delta) CSV** from MinIO and merging it with a **local base dataset** stored on disk inside the container.

> **Important:** This API is not fully stateless. It requires the base dataset CSV to be provisioned inside the container at `data/base/simulation_security_labels_n-1.csv` (or at the path configured by `BASE_DATASET_LOCAL_PATH`). See deployment notes below.

#### Request headers
```
Content-Type: application/json
X-API-Key: <SHARED_SECRET>
```

#### Minimal request body (recommended)

Use `latest_key` to point to the delta CSV (appended rows only) produced by the dashboard.

```json
{
  "latest_key": "al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv"
}
```

#### Full request body (optional configuration)
```json
{
  "results_bucket": "smart-energy-results",
  "latest_key": "al_training_dataset/appended_rows/simulation_security_labels_n-1_appended_rows_latest.csv",
  "output_prefix": "models/retraining_runs",
  "n_estimators": 400,
  "random_state": 42,
  "test_size": 0.2,
  "label_col": "status",
  "drop_feature_cols": [],
  "drop_latest_columns": ["created_at"]
}
```

---

## Successful response example

```json
{
  "ok": true,
  "message": "Retraining completed.",
  "dataset_rows": 8773,
  "dataset_rows_latest": 2,
  "model_object": "smart-energy-results/models/retraining_runs/20260131T150800Z_a1b2c3d4/model.joblib",
  "metrics_object": "smart-energy-results/models/retraining_runs/20260131T150800Z_a1b2c3d4/metrics.json",
  "metrics": {
    "accuracy": 0.97,
    "f1_macro": 0.96,
    "dataset_mode": "single_input_csv",
    "n_rows_all": 8773,
    "n_rows_latest": 2,
    "n_features": 24,
    "n_estimators": 400,
    "random_state": 42,
    "test_size": 0.2
  }
}
```

`dataset_rows_latest` and `metrics.n_rows_latest` reflect the number of rows in the appended delta CSV, not the total dataset size.

### Response fields of interest for the dashboard
- **model_object** – MinIO path to the newly trained model artifact
- **metrics_object** – MinIO path to the metrics JSON
- **metrics** – training and evaluation statistics to be visualized in the dashboard

---

## Error handling

### 401 Unauthorized
Returned if the `X-API-Key` header is missing or invalid.

### 500 Download failed
Returned if the appended rows delta CSV cannot be downloaded from MinIO.

Example:
```json
{
  "detail": "Download appended rows failed: <reason>"
}
```

### 500 Training failed
Returned if model training fails due to invalid data, missing label column, or internal errors.

---

## Notes on dataset handling

The API uses a **two-part dataset** on every `/retrain` call:

1. **Local base dataset** – stored on disk inside the Docker container at `data/base/simulation_security_labels_n-1.csv`. This file must be provisioned once when the container is set up. It is not downloaded from MinIO on each request (this was changed to avoid timeout errors caused by transferring the full ~22 MB CSV on every retraining call).

2. **Appended rows (delta)** – a small CSV uploaded by the dashboard to MinIO containing only the newly labeled samples selected by the operator. The API downloads this on every `/retrain` call and concatenates it with the local base dataset before training.

Training always uses: `merged = base_dataset + appended_rows_delta`.

---

## Deployment notes
- The API is deployed as a Docker container on Atena
- Atena exposes the API on host port **5004**; the container itself binds to port 8000
- **The API is not fully stateless**: it requires the base dataset CSV to be present inside the container. Without it, `/retrain` returns 500.
- Trained model artifacts (`model.joblib`, `metrics.json`) are uploaded to MinIO after each retraining run under a versioned run folder

---

## Contact
For changes to the retraining logic or API contract, contact the Smart Energy retraining service maintainer.
