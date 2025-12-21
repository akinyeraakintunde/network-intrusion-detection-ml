import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Adjust these to your dataset columns
LABEL_COL = os.getenv("LABEL_COL", "label")  # e.g. "label" or "target"
DATA_PATH = os.getenv("DATA_PATH", "data/dataset.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found. Columns: {list(df.columns)}")

    y = df[LABEL_COL]
    X = df.drop(columns=[LABEL_COL])

    # Basic split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = None
    try:
        proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    print("=== Classification Report ===")
    print(classification_report(y_test, preds))

    if proba is not None and y.nunique() == 2:
        try:
            auc = roc_auc_score(y_test, proba)
            print(f"ROC AUC: {auc:.4f}")
        except Exception:
            pass

    joblib.dump(
        {"model": model, "feature_columns": list(X.columns)},
        MODEL_PATH
    )
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()