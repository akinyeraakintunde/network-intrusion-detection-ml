[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-brightgreen)](https://nid-ml-demo.onrender.com/docs)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com)
[![ML](https://img.shields.io/badge/Machine%20Learning-Production-blue)

## 🔴 Live Demo (Production)

This project is deployed as a live FastAPI service.

**Swagger UI:**  
👉 https://nid-ml-demo.onrender.com/docs

### Available Endpoints
- `GET /health` – Service health check
- `POST /predict` – Intrusion classification with confidence and signals

### Example Response
```json
{
  "prediction": "malicious",
  "confidence": 0.9458,
  "signals": {
    "score": 0.9458,
    "threshold": 0.5
  },
  "inputs_used": {
    "src_bytes": 200,
    "dst_bytes": 7000,
    "count": 4,
    "srv_count": 6
  }
}

This explains **why it’s simple now** and shows **clear roadmap thinking**.

---

## 3️⃣ Add Architecture explanation (text-first, no diagram yet)

Add this section:

```md
## 🧠 Architecture Overview

- **FastAPI** – API layer & OpenAPI schema
- **Pydantic** – Request/response validation
- **Rules-based Scoring Engine** – Placeholder for ML inference
- **Docker + Render** – Production deployment
- **Swagger UI** – Interactive demo for non-technical reviewers

The scoring engine is intentionally modular to allow drop-in replacement
with a trained model (e.g. RandomForest, XGBoost, or Neural Network).