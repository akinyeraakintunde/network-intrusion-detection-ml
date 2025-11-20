from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd

# -------------------------------------------------
# Paths and config
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = DATA_DIR / "intrusion_model.pkl"
TRAIN_DATA_PATH = DATA_DIR / "nsl_kdd_train_binary.csv"  # used to infer feature schema

TARGET_COLUMN = "binary_label"   # target column used during training
POSITIVE_CLASS = 1               # 1 = attack


# -------------------------------------------------
# FastAPI app
# -------------------------------------------------

app = FastAPI(
    title="Network Intrusion Detection API",
    description=(
        "REST API for a Machine Learning–based Network Intrusion Detection System (IDS). "
        "Backed by a RandomForest model trained on the NSL-KDD dataset (binary: normal vs attack)."
    ),
    version="1.0.0",
)

# -------------------------------------------------
# Load model and feature schema at startup
# -------------------------------------------------

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

if not TRAIN_DATA_PATH.exists():
    raise RuntimeError(f"Training data file not found at {TRAIN_DATA_PATH}")

# Load trained model
model = joblib.load(MODEL_PATH)

# Infer feature columns from training data
train_sample = pd.read_csv(TRAIN_DATA_PATH, nrows=1)
feature_columns = [c for c in train_sample.columns if c != TARGET_COLUMN]

# Find index of positive (attack) class
if hasattr(model, "classes_"):
    if POSITIVE_CLASS not in model.classes_:
        raise RuntimeError(
            f"Model classes {model.classes_} do not contain POSITIVE_CLASS={POSITIVE_CLASS}"
        )
    POSITIVE_INDEX = list(model.classes_).index(POSITIVE_CLASS)
else:
    POSITIVE_INDEX = 1  # fallback – should not normally be needed


# -------------------------------------------------
# Pydantic models
# -------------------------------------------------

class PredictionRequest(BaseModel):
    records: List[Dict[str, float]]


class PredictionItem(BaseModel):
    index: int
    label: int
    label_name: str
    score_attack: float


class PredictionResponse(BaseModel):
    predictions: List[PredictionItem]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime


class MetadataResponse(BaseModel):
    model_name: str
    model_version: str
    framework: str
    trained_on_dataset: str
    features_count: int
    feature_names: List[str]


# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def build_feature_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of JSON records into a DataFrame with the same columns
    as the model was trained on. Missing columns are filled with 0.0.
    Extra columns are discarded.
    """
    if not records:
        raise ValueError("records must be a non-empty list")

    df = pd.DataFrame(records)

    # Ensure all expected feature columns are present, in correct order
    df = df.reindex(columns=feature_columns, fill_value=0.0)

    # Drop anything not in feature_columns (safety)
    df = df[feature_columns]

    return df


# -------------------------------------------------
# Endpoints
# -------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Simple health endpoint to verify that the API is reachable.
    """
    return HealthResponse(status="ok", timestamp=datetime.utcnow())


@app.get("/metadata", response_model=MetadataResponse, tags=["Health"])
def metadata():
    """
    Returns metadata about the deployed IDS model and dataset.
    """
    return MetadataResponse(
        model_name=type(model).__name__,
        model_version="v1.0.0",
        framework="scikit-learn",
        trained_on_dataset="NSL-KDD (binary: normal vs attack)",
        features_count=len(feature_columns),
        feature_names=feature_columns,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """
    Predict intrusion for one or more network records.
    0 = normal, 1 = attack.

    Each record should contain numeric feature values; any missing
    features will be filled with 0.0.
    """
    try:
        df = build_feature_dataframe(request.records)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        proba = model.predict_proba(df)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}",
        )

    preds = proba.argmax(axis=1)
    attack_scores = proba[:, POSITIVE_INDEX]

    results: List[PredictionItem] = []
    for idx, (label, score) in enumerate(zip(preds, attack_scores)):
        label_name = "attack" if label == 1 else "normal"
        results.append(
            PredictionItem(
                index=idx,
                label=int(label),
                label_name=label_name,
                score_attack=float(score),
            )
        )

    return PredictionResponse(
        predictions=results,
        model_version="v1.0.0",
    )