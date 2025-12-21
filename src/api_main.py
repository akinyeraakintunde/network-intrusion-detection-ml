from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# -------------------------
# Paths / config
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # repo_root/src -> repo_root
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = DATA_DIR / "intrusion_model.pkl"

# If your model uses 0=normal, 1=attack (binary)
LABEL_MAP = {0: "normal", 1: "attack"}


# -------------------------
# App
# -------------------------
app = FastAPI(
    title="Network Intrusion Detection (Live Demo)",
    description="FastAPI demo endpoint for intrusion classification.",
    version="1.0.0",
)

_model = None  # loaded at startup


# -------------------------
# Schemas
# -------------------------
class PredictionRequest(BaseModel):
    # Accept either:
    # 1) single feature dict -> { "src_bytes": 200, ... }
    # 2) list of feature dicts -> [ {...}, {...} ]
    features: Optional[Dict[str, float]] = Field(default=None, description="Single record features (dict).")
    records: Optional[List[Dict[str, float]]] = Field(default=None, description="Multiple records features (list).")


class PredictionItem(BaseModel):
    index: int
    label: int
    prediction: str
    confidence: float
    score_attack: Optional[float] = None


class PredictionResponse(BaseModel):
    predictions: List[PredictionItem]
    model_version: str = "v1.0.0"
    note: str = "Rules/heuristic placeholder until model wiring is verified."


# -------------------------
# Helpers
# -------------------------
def _load_model() -> Any:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def _infer_feature_order(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    """
    Align dataframe columns to the model's expected feature order if available.
    If model has feature_names_in_ (sklearn), we reindex to that.
    Otherwise we leave columns as-is.
    """
    if hasattr(model, "feature_names_in_"):
        cols = list(getattr(model, "feature_names_in_"))
        # add missing columns as zeros
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
        # drop extras and reorder
        df = df.reindex(columns=cols, fill_value=0.0)

    # ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def _build_df(payload: PredictionRequest) -> pd.DataFrame:
    if payload.records:
        rows = payload.records
    elif payload.features:
        rows = [payload.features]
    else:
        raise ValueError("Provide either 'features' (dict) or 'records' (list of dicts).")

    df = pd.DataFrame(rows).fillna(0.0)
    return df


def _predict(model: Any, df: pd.DataFrame) -> List[PredictionItem]:
    df = _infer_feature_order(model, df)

    # Prefer predict_proba for confidence
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)
        # If binary, proba shape -> (n,2)
        labels = np.argmax(proba, axis=1).astype(int)
        conf = np.max(proba, axis=1).astype(float)

        score_attack = None
        if proba.shape[1] >= 2:
            score_attack = proba[:, 1].astype(float)

        items: List[PredictionItem] = []
        for i, y in enumerate(labels):
            items.append(
                PredictionItem(
                    index=i,
                    label=int(y),
                    prediction=LABEL_MAP.get(int(y), str(int(y))),
                    confidence=float(conf[i]),
                    score_attack=float(score_attack[i]) if score_attack is not None else None,
                )
            )
        return items

    # Fallback: predict only (confidence unknown)
    preds = model.predict(df)
    items = []
    for i, y in enumerate(preds):
        yi = int(y)
        items.append(
            PredictionItem(
                index=i,
                label=yi,
                prediction=LABEL_MAP.get(yi, str(yi)),
                confidence=0.0,
                score_attack=None,
            )
        )
    return items


# -------------------------
# Startup
# -------------------------
@app.on_event("startup")
def startup_event() -> None:
    global _model
    _model = _load_model()


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    global _model
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        df = _build_df(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        preds = _predict(_model, df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # small friendly note for demo
    note = "Model is wired. If you see stable predictions, next step is to validate on held-out test set and add monitoring."
    return PredictionResponse(predictions=preds, note=note)


@app.get("/demo")
def demo() -> Dict[str, Any]:
    """
    Non-technical, pretty summary endpoint.
    """
    return {
        "product": "Network Intrusion Detection (Live Demo)",
        "what_it_does": "Classifies traffic-like features as normal vs attack.",
        "how_to_try": {
            "swagger_ui": "/docs",
            "predict_endpoint": "POST /predict",
            "example_payload": {
                "features": {
                    "src_bytes": 200,
                    "dst_bytes": 7000,
                    "count": 4,
                    "srv_count": 6,
                }
            },
        },
        "status": "ok",
    }