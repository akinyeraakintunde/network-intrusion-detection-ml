"""
End-to-end training script for the Network Intrusion Detection System (IDS).

What it does:
- Loads dataset from data/dataset_clean.csv (or data/dataset.csv as fallback)
- Automatically detects a sensible target/label column
- Preprocesses data (drop NAs, encode categoricals)
- Splits into train/test sets
- Trains a RandomForestClassifier
- Prints accuracy, classification report, and confusion matrix
- Saves the trained model to models/intrusion_model.pkl
"""

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# ---------- 1. Load data ----------

def load_dataset():
    clean_path = os.path.join("data", "dataset_clean.csv")
    raw_path = os.path.join("data", "dataset.csv")

    if os.path.exists(clean_path):
        print(f"[INFO] Loading cleaned dataset: {clean_path}")
        df = pd.read_csv(clean_path)
    elif os.path.exists(raw_path):
        print(f"[INFO] Loading raw dataset: {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        raise FileNotFoundError(
            "No dataset found in data/ (expected dataset_clean.csv or dataset.csv)."
        )

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    return df


# ---------- 2. Detect target column automatically ----------

def detect_target_column(df):
    """
    Try to guess the target column from common names.
    If none match, fall back to using the last column.
    """

    candidates = [
        "label",
        "target",
        "class",
        "attack_type",
        "intrusion",
        "is_attack",
        "y",
    ]

    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand in lower_map:
            target = lower_map[cand]
            print(f"[INFO] Auto-detected target column: '{target}'")
            return target

    # Fallback: last column
    target = df.columns[-1]
    print(
        f"[WARN] Could not find a standard label column, "
        f"falling back to last column: '{target}'"
    )
    return target


# ---------- 3. Preprocess ----------

def preprocess(df, target_column=None):
    if target_column is None:
        target_column = detect_target_column(df)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Separate features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Basic cleaning
    X = X.copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(axis=0, inplace=True)

    # Align y with cleaned X
    y = y.loc[X.index]

    # Encode categoricals using one-hot encoding
    X = pd.get_dummies(X)

    print(f"[INFO] Features shape after encoding: {X.shape}")
    return X, y, target_column


# ---------- 4. Train / Evaluate / Save ----------

def train_and_evaluate(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Train size: {X_train.shape}, Test size: {X_test.shape}")

    # Create and train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    print("[INFO] Training RandomForest model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Accuracy: {acc:.4f}\n")

    print("[RESULT] Classification report:")
    print(classification_report(y_test, y_pred))

    print("[RESULT] Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "intrusion_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n[SAVED] Trained model saved to: {model_path}")

    return model


def main():
    df = load_dataset()
    X, y, target_col = preprocess(df)
    print(f"[INFO] Using target column: '{target_col}'")
    train_and_evaluate(X, y)


if __name__ == "__main__":
    main()