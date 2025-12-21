from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = DATA_DIR / "intrusion_model.pkl"


# ----------------------------
# Request / Response Schemas
# ----------------------------
class PredictRequest(BaseModel):
    """
    Flexible payload:
    - features: a single record of numeric features
    - records: a list of records (batch)
    Provide either 'features' or 'records'.
    """
    features: Optional[Dict[str, float]] = Field(default=None, examples=[{"src_bytes": 200, "dst_bytes": 7000, "count": 4, "srv_count": 6}])
    records: Optional[List[Dict[str, float]]] = Field(default=None, examples=[[{"src_bytes": 200, "dst_bytes": 7000, "count": 4, "srv_count": 6}]])


class PredictItem(BaseModel):
    index: int
    prediction: str
    confidence: float
    score_attack: float


class PredictResponse(BaseModel):
    predictions: List[PredictItem]
    model_version: str = "v1.0.0"
    note: str


# ----------------------------
# App
# ----------------------------
app = FastAPI(
    title="Network Intrusion Detection (Live Demo)",
    version="1.0.0",
    description="FastAPI demo endpoint for intrusion classification. (Rules/heuristic fallback if model not available.)",
)

_model: Any = None
_feature_order: Optional[List[str]] = None


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _get_records(payload: PredictRequest) -> List[Dict[str, float]]:
    if payload.records and len(payload.records) > 0:
        return payload.records
    if payload.features and len(payload.features) > 0:
        return [payload.features]
    raise HTTPException(status_code=422, detail="Provide either 'features' (single record) or 'records' (batch).")


def _heuristic_score(record: Dict[str, float]) -> float:
    """
    Simple placeholder heuristic: higher bytes + higher counts => more suspicious.
    Replace with your trained model scoring later.
    """
    src_bytes = _safe_float(record.get("src_bytes", 0))
    dst_bytes = _safe_float(record.get("dst_bytes", 0))
    count = _safe_float(record.get("count", 0))
    srv_count = _safe_float(record.get("srv_count", 0))

    score = 0.0
    score += min(src_bytes / 5000.0, 1.0) * 0.35
    score += min(dst_bytes / 20000.0, 1.0) * 0.35
    score += min(count / 100.0, 1.0) * 0.15
    score += min(srv_count / 100.0, 1.0) * 0.15
    return max(0.0, min(score, 1.0))


def _load_model_if_present() -> None:
    global _model, _feature_order
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)

        # Optional: if you saved feature order alongside the model, set it here.
        # _feature_order = ["src_bytes", "dst_bytes", "count", "srv_count", ...]
        _feature_order = None


@app.on_event("startup")
def startup_event():
    _load_model_if_present()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.get("/demo")
def demo():
    """
    Human-friendly summary endpoint for non-technical viewers.
    """
    return {
        "project": "Network Intrusion Detection (Live Demo)",
        "what_it_does": "Scores network traffic-like features and returns a risk classification (benign vs malicious).",
        "how_to_try": {
            "swagger_ui": "/docs",
            "post_predict": {
                "endpoint": "/predict",
                "example_body": {
                    "features": {"src_bytes": 200, "dst_bytes": 7000, "count": 4, "srv_count": 6}
                },
            },
        },
        "notes": [
            "If a trained model file exists at data/intrusion_model.pkl, the API will use it.",
            "If not, it uses a simple heuristic scorer as a placeholder demo.",
        ],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    records = _get_records(payload)

    results: List[PredictItem] = []
    threshold = 0.5

    # If you have a trained model that supports predict_proba, you can wire it here.
    if _model is not None:
        try:
            # Build matrix in a stable column order if you have it; otherwise use keys sorted
            import pandas as pd  # local import to keep startup lighter

            if _feature_order:
                rows = [{k: _safe_float(r.get(k, 0.0)) for k in _feature_order} for r in records]
                df = pd.DataFrame(rows, columns=_feature_order)
            else:
                # fallback: union of keys, sorted
                all_keys = sorted({k for r in records for k in r.keys()})
                rows = [{k: _safe_float(r.get(k, 0.0)) for k in all_keys} for r in records]
                df = pd.DataFrame(rows, columns=all_keys)

            if hasattr(_model, "predict_proba"):
                proba = _model.predict_proba(df)
                # assume positive class is column 1 if binary
                attack_scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            else:
                # fallback to predict output, map to confidence-ish
                preds = _model.predict(df)
                attack_scores = [1.0 if int(p) == 1 else 0.0 for p in preds]

            for i, s in enumerate(attack_scores):
                s = float(s)
                label = "malicious" if s >= threshold else "benign"
                conf = s if label == "malicious" else (1.0 - s)
                results.append(PredictItem(index=i, prediction=label, confidence=conf, score_attack=s))

            return PredictResponse(
                predictions=results,
                note="Using trained model from data/intrusion_model.pkl",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # Heuristic fallback
    for i, r in enumerate(records):
        s = _heuristic_score(r)
        label = "malicious" if s >= threshold else "benign"
        conf = s if label == "malicious" else (1.0 - s)
        results.append(PredictItem(index=i, prediction=label, confidence=conf, score_attack=s))

    return PredictResponse(
        predictions=results,
        note="Using heuristic demo scorer (wire to trained model next).",
    )