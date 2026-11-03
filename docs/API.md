# API documentation

Base URL during development:

```text
http://localhost:8000/api
```

## Health check

```http
GET /api/health/
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "mode": "real"
}
```

## Prediction

```http
POST /api/predict/
Content-Type: multipart/form-data
```

Form fields:

| Field | Type | Required | Description |
|---|---|---:|---|
| `image` | file | yes | Skin lesion image |
| `age` | number | no | Patient age |
| `sex` | string | no | Patient sex/gender |
| `anatom_site` | string | no | Body region |
| `smoke` | string | no | Smoking history |
| `drink` | string | no | Drinking history |

Example response:

```json
{
  "predicted_class": "BCC",
  "confidence": 0.87,
  "confidence_level": "High",
  "risk_level": "High risk",
  "probabilities": {
    "ACK": 0.03,
    "BCC": 0.87,
    "MEL": 0.04,
    "NEV": 0.02,
    "SCC": 0.03,
    "SEK": 0.01
  },
  "top_predictions": [
    {"class_name": "BCC", "probability": 0.87},
    {"class_name": "MEL", "probability": 0.04},
    {"class_name": "ACK", "probability": 0.03}
  ],
  "quality_warnings": [],
  "uncertainty_warnings": [],
  "disclaimer": "This AI result is for educational screening support only and is not a medical diagnosis."
}
```
