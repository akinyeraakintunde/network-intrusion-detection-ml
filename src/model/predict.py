import os
import joblib
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

_bundle = None

def load_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train it first (src/model/train.py)."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle

def predict(features: dict, threshold: float = DEFAULT_THRESHOLD):
    bundle = load_bundle()
    model = bundle["model"]
    cols = bundle["feature_columns"]

    # Align incoming features to training columns
    x = np.array([[float(features.get(c, 0.0)) for c in cols]])

    # Probability if available
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(x)[0][1]) if model.n_classes_ == 2 else float(np.max(model.predict_proba(x)[0]))
    else:
        proba = None

    pred = model.predict(x)[0]
    if proba is not None and model.n_classes_ == 2:
        label = "malicious" if proba >= threshold else "benign"
        confidence = proba if label == "malicious" else (1 - proba)
        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "signals": {"score": round(proba, 4), "threshold": threshold},
        }

    # Fallback: non-proba
    return {
        "prediction": str(pred),
        "confidence": 0.5,
        "signals": {"score": None, "threshold": threshold},
    }