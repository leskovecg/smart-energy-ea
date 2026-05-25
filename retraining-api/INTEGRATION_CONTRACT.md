# Smart Energy – Retraining API Integration Contract

This document describes how the HumAIne dashboard integrates with the Smart Energy retraining API.

---

## Base URL
```
http://<ATENA_HOST>:8000
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

Triggers retraining of the Random Forest model using the latest available dataset snapshot stored in MinIO.

#### Request headers
```
Content-Type: application/json
X-API-Key: <SHARED_SECRET>
```

#### Minimal request body (recommended)
The `latest_key` always points to the full dataset snapshot produced by the dashboard.

```json
{
  "latest_key": "al_training_dataset/simulation_security_labels_n-1_latest.csv"
}
```

#### Full request body (optional configuration)
```json
{
  "results_bucket": "smart-energy-results",
  "latest_key": "al_training_dataset/simulation_security_labels_n-1_latest.csv",
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
  "dataset_rows": 8771,
  "dataset_rows_latest": 8771,
  "model_object": "smart-energy-results/models/retraining_runs/20260131T150800Z_a1b2c3d4/model.joblib",
  "metrics_object": "smart-energy-results/models/retraining_runs/20260131T150800Z_a1b2c3d4/metrics.json",
  "metrics": {
    "accuracy": 0.97,
    "f1_macro": 0.96,
    "dataset_mode": "snapshot",
    "n_rows_all": 8771,
    "n_features": 24,
    "n_estimators": 400,
    "random_state": 42,
    "test_size": 0.2
  }
}
```

### Response fields of interest for the dashboard
- **model_object** – MinIO path to the newly trained model artifact
- **metrics_object** – MinIO path to the metrics JSON
- **metrics** – training and evaluation statistics to be visualized in the dashboard

---

## Error handling

### 401 Unauthorized
Returned if the `X-API-Key` header is missing or invalid.

### 500 Download failed
Returned if the latest dataset cannot be downloaded from MinIO.

Example:
```json
{
  "detail": "Download latest failed: <reason>"
}
```

### 500 Training failed
Returned if model training fails due to invalid data, missing label column, or internal errors.

---

## Notes on dataset handling
- `simulation_security_labels_n-1_latest.csv` is treated as a **full dataset snapshot**
- Retraining uses only `latest_key`

---

## Deployment notes
- The API is deployed as a Docker container on Atena
- Port **8000** must be accessible from the HumAIne dashboard
- The API is stateless; all artifacts are stored in MinIO

---

## Contact
For changes to the retraining logic or API contract, contact the Smart Energy retraining service maintainer.
