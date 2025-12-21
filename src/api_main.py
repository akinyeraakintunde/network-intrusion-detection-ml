"""
Network Intrusion Detection API (FastAPI)

- Loads a trained model from: data/intrusion_model.pkl
- Exposes:
  - GET  /health
  - POST /predict   (accepts {"features": {...}} or a single flat JSON object)
  - GET  /demo      (non-technical friendly summary + sample request)

Run locally:
  uvicorn src.api_main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# -----------------------------
# Paths / config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = DATA_DIR / "intrusion_model.pkl"

# Optional helper file (recommended): a JSON list of model feature names.
# Example: data/feature_columns.json  -> ["src_bytes","dst_bytes","count","srv_count",...]
FEATURE_COLUMNS_PATH = DATA_DIR / "feature_columns.json"


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Network Intrusion Detection (Live Demo)",
    version="1.0.0",
    description=(
        "FastAPI demo endpoint for intrusion classification.\n\n"
        "POST /predict with feature values to get a prediction.\n"
        "This service supports a simplified demo payload like:\n"
        '{ "features": { "src_bytes": 200, "dst_bytes": 7000, "count": 4, "srv_count": 6 } }'
    ),
)

MODEL: Any = None
FEATURE_COLUMNS: Optional[List[str]] = None


# -----------------------------
# Schemas
# -----------------------------
class PredictionRequest(BaseModel):
    """
    Preferred request shape:
      { "features": { "src_bytes": 200, "dst_bytes": 7000, ... } }

    Also supported (flat JSON):
      { "src_bytes": 200, "dst_bytes": 7000, ... }
    """
    features: Dict[str, float] = Field(default_factory=dict, description="Feature-name -> numeric value map")


class PredictionItem(BaseModel):
    prediction: str = Field(..., description="Human label: 'normal' or 'attack'")
    score: float = Field(..., ge=0.0, le=1.0, description="Probability/confidence for the positive class (attack)")
    threshold: float = Field(0.5, description="Decision threshold used to convert score -> label")


class PredictionResponse(BaseModel):
    result: PredictionItem
    model_version: str = "v1.0.0"
    timestamp_utc: str
    inputs_used: Dict[str, float]
    notes: List[str] = Field(default_factory=list)


# -----------------------------
# Helpers
# -----------------------------
def _load_feature_columns() -> Optional[List[str]]:
    """Load feature column names (recommended) from data/feature_columns.json if present."""
    try:
        if FEATURE_COLUMNS_PATH.exists():
            import json
            cols = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))
            if isinstance(cols, list) and all(isinstance(x, str) for x in cols):
                return cols
    except Exception:
        # non-fatal
        return None
    return None


def _build_feature_dataframe(features: Dict[str, float]) -> pd.DataFrame:
    """
    Build a single-row dataframe for model inference.
    If FEATURE_COLUMNS is known, align to it and fill missing with 0.0.
    If unknown, use whatever keys we received (best-effort).
    """
    if not isinstance(features, dict) or not features:
        raise ValueError("No features provided. Provide a JSON payload with numeric feature values.")

    # Ensure numeric
    cleaned: Dict[str, float] = {}
    for k, v in features.items():
        try:
            cleaned[str(k)] = float(v)
        except Exception:
            raise ValueError(f"Feature '{k}' must be numeric. Got: {v!r}")

    if FEATURE_COLUMNS:
        row = {col: float(cleaned.get(col, 0.0)) for col in FEATURE_COLUMNS}
        df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    else:
        df = pd.DataFrame([cleaned])

    return df


def _get_attack_probability(model: Any, df: pd.DataFrame) -> float:
    """
    Returns attack probability score in [0,1].
    Tries predict_proba first, then decision_function as fallback.
    """
    # 1) predict_proba (preferred)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)
        # If binary classifier: proba[0][1] typically = positive class
        if hasattr(proba, "shape") and proba.shape[1] >= 2:
            return float(proba[0][1])
        return float(proba[0][0])

    # 2) decision_function fallback -> convert to pseudo-prob via sigmoid
    if hasattr(model, "decision_function"):
        import math
        score = float(model.decision_function(df)[0])
        return 1.0 / (1.0 + math.exp(-score))

    # 3) last resort: predict -> 0/1
    if hasattr(model, "predict"):
        pred = int(model.predict(df)[0])
        return 1.0 if pred == 1 else 0.0

    raise RuntimeError("Model does not support predict_proba, decision_function, or predict.")


def _normalize_request(payload: Union[PredictionRequest, Dict[str, Any]]) -> Dict[str, float]:
    """
    Accepts either:
      - PredictionRequest ({"features": {...}})
      - flat dict ({"src_bytes":..., "dst_bytes":...})
    and returns a normalized features dict.
    """
    if isinstance(payload, PredictionRequest):
        if payload.features:
            return payload.features
        return {}

    # flat dict support
    if isinstance(payload, dict):
        # If "features" is present, use it.
        if "features" in payload and isinstance(payload["features"], dict):
            return payload["features"]
        # Otherwise treat entire dict as features
        return {k: payload[k] for k in payload.keys()}

    return {}


# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
def startup() -> None:
    global MODEL, FEATURE_COLUMNS

    if not MODEL_PATH.exists():
        # Keep app running but clearly indicate missing model.
        # This makes /health work, and /predict gives a friendly error.
        MODEL = None
    else:
        MODEL = joblib.load(MODEL_PATH)

    FEATURE_COLUMNS = _load_feature_columns()


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "feature_columns_loaded": bool(FEATURE_COLUMNS),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/demo")
def demo() -> Dict[str, Any]:
    """
    A friendly endpoint recruiters can open quickly.
    """
    return {
        "what_this_is": "A live FastAPI demo for Network Intrusion Detection scoring.",
        "how_to_use": [
            "Open /docs to try it in Swagger UI.",
            "Call POST /predict with a JSON payload of numeric features.",
        ],
        "sample_request_body": {
            "features": {"src_bytes": 200, "dst_bytes": 7000, "count": 4, "srv_count": 6}
        },
        "sample_curl": (
            "curl -X POST 'http://localhost:8000/predict' "
            "-H 'accept: application/json' -H 'Content-Type: application/json' "
            "-d '{\"features\":{\"src_bytes\":200,\"dst_bytes\":7000,\"count\":4,\"srv_count\":6}}'"
        ),
        "notes": [
            "If feature_columns.json exists, missing features are auto-filled with 0.0.",
            "If feature columns are not provided, the API uses received keys (best-effort).",
        ],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: Dict[str, Any]) -> PredictionResponse:
    """
    Accepts either:
      {"features": {...}}
    or a flat JSON dict:
      {"src_bytes": 200, "dst_bytes": 7000, ...}
    """
    if MODEL is None:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Model not loaded. Expected model file at: {MODEL_PATH}. "
                "Ensure data/intrusion_model.pkl exists in the deployed environment."
            ),
        )

    features = _normalize_request(payload)
    if not features:
        raise HTTPException(
            status_code=422,
            detail="Invalid payload. Provide {'features': {...}} or a flat JSON object with numeric feature fields.",
        )

    try:
        df = _build_feature_dataframe(features)
        attack_proba = _get_attack_probability(MODEL, df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    threshold = 0.5
    label = "attack" if attack_proba >= threshold else "normal"

    notes: List[str] = []
    if FEATURE_COLUMNS:
        notes.append("Feature alignment: request was aligned to feature_columns.json (missing filled with 0.0).")
    else:
        notes.append("Feature alignment: feature_columns.json not found; using provided keys (best-effort).")

    return PredictionResponse(
        result=PredictionItem(
            prediction=label,
            score=float(round(attack_proba, 6)),
            threshold=threshold,
        ),
        timestamp_utc=datetime.utcnow().isoformat() + "Z",
        inputs_used={k: float(v) for k, v in features.items()},
        notes=notes,
    )


# Optional local runner
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api_main:app", host="0.0.0.0", port=8000, reload=True)