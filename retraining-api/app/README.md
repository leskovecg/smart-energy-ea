# Retraining API (FastAPI)

This module provides a **FastAPI-based retraining service** for the Smart Energy
Active Learning pipeline.

The API is responsible for retraining the power grid security classifier
based on updated labeled datasets and storing the resulting artifacts.

---

## Responsibilities

The retraining API handles:
- Reading the fixed base dataset from local disk (`data/base/...`)
- Downloading only the latest appended rows CSV from MinIO (`al_training_dataset/appended_rows/..._latest.csv`)
- Merging base + appended rows locally before training
- Retraining the security classification model (Random Forest)
- Computing and storing evaluation metrics
- Uploading trained models and metadata back to MinIO
- Exposing retraining functionality via HTTP endpoints

This service is intended to be triggered by:
- dashboards
- orchestration pipelines
- human-in-the-loop workflows

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

## API endpoints (overview)

Typical endpoints include:
- `GET /health` – health check
- `POST /retrain` – trigger model retraining with specified parameters

(See `main.py` for the exact API specification.)

---

## Running the API locally

### Using Makefile (recommended)

From the repository root:
```bash
make run-api
```

---

### Manual run

From the repository root:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Configuration

The API relies on environment variables defined in:
```text
.env
```

A template is provided in:
```text
.env.example
```

Make sure to configure:
- MinIO endpoint and credentials
- API authentication parameters
- Timeouts and other runtime settings

Also ensure local base dataset is prepared once:

```text
retraining-api/
  data/
    base/
      simulation_security_labels_n-1.csv
    appended/
      simulation_security_labels_n-1_appended_rows_latest.csv  # refreshed by /retrain downloads
```

---

## Notes

- This API is a **prototype service** intended for research and experimentation.
- It is not designed for direct production deployment without additional hardening
  (authentication, monitoring, CI/CD, etc.).

---
