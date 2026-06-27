# Retraining API (FastAPI)

> For deployment, configuration, and run instructions
> see `retraining-api/README_DEPLOY.md`.

---

## Main files

- `main.py`  
  Defines the FastAPI application and API endpoints.

- `train.py`  
  Contains the model training and evaluation logic.

- `minio_io.py`  
  Handles input/output operations with MinIO (datasets, models, metrics).

- `schemas.py`  
  Defines request and response schemas for the API.

---

## Notes

- This API is a **prototype service** intended for research and experimentation.
- It is not designed for direct production deployment without additional hardening
  (authentication, monitoring, CI/CD, etc.).

---
