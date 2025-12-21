from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, Any
import os
import math

app = FastAPI(
    title="Network Intrusion Detection (Live Demo)",
    description="FastAPI demo endpoint for intrusion classification. (Rules/heuristic placeholder until model is wired.)",
    version="1.0.0",
)

class PredictRequest(BaseModel):
    # Keep this flexible so the demo works even if your dataset features differ.
    # You can send any numeric features in a dict.
    features: Dict[str, float] = Field(..., example={"src_bytes": 181.0, "dst_bytes": 5450.0, "count": 2.0})

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    signals: Dict[str, Any]
    inputs_used: Dict[str, float]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def simple_intrusion_score(features: Dict[str, float]) -> float:
    """
    Lightweight heuristic scoring (demo-safe).
    Replace this with your trained model inference later.
    """
    # Basic signals that tend to correlate with suspicious traffic in many datasets
    src_bytes = features.get("src_bytes", 0.0)
    dst_bytes = features.get("dst_bytes", 0.0)
    count = features.get("count", 0.0)  # connection count to same host/service
    srv_count = features.get("srv_count", 0.0)

    # Hand-tuned weights (demo only)
    raw = (
        0.0008 * src_bytes +
        0.0004 * dst_bytes +
        0.25 * count +
        0.15 * srv_count
    )

    # squash to 0..1
    return _sigmoid(raw - 2.0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    score = simple_intrusion_score(payload.features)
    prediction = "malicious" if score >= 0.5 else "benign"

    signals = {
        "score": round(score, 4),
        "threshold": 0.5,
        "note": "Heuristic demo scorer (wire to trained model next)."
    }

    return PredictResponse(
        prediction=prediction,
        confidence=round(score if prediction == "malicious" else (1 - score), 4),
        signals=signals,
        inputs_used=payload.features
    )