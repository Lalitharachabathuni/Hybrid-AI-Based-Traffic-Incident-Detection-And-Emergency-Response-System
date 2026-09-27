"""
train_model.py
---------------
Accident Detection module: trains the XGBoost severity classifier
(Normal / Minor / Moderate / Major) on the engineered traffic features.

Run:
    python backend/ml/train_model.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from preprocess import FEATURE_COLUMNS, build_dataset, PROCESSED_DIR

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(MODEL_DIR, exist_ok=True)

SEVERITY_LABELS = {0: "Normal", 1: "Minor", 2: "Moderate", 3: "Major"}


def load_or_build_dataset():
    path = os.path.join(PROCESSED_DIR, "traffic_features.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return build_dataset()


def main():
    df = load_or_build_dataset()
    X = df[FEATURE_COLUMNS]
    y = df["severity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # class_weight-style handling for XGBoost via sample_weight (severe class imbalance)
    class_counts = y_train.value_counts()
    total = len(y_train)
    sample_weight = y_train.map(lambda c: total / (len(class_counts) * class_counts[c]))

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    preds = model.predict(X_test)
    report = classification_report(
        y_test, preds, target_names=[SEVERITY_LABELS[i] for i in sorted(SEVERITY_LABELS)],
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test, preds).tolist()

    print(classification_report(
        y_test, preds, target_names=[SEVERITY_LABELS[i] for i in sorted(SEVERITY_LABELS)],
        zero_division=0,
    ))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(np.array(cm))

    model_path = os.path.join(MODEL_DIR, "xgb_incident_model.joblib")
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS, "labels": SEVERITY_LABELS}, model_path)

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"classification_report": report, "confusion_matrix": cm}, f, indent=2)

    print(f"\nSaved model      -> {model_path}")
    print(f"Saved metrics    -> {metrics_path}")


if __name__ == "__main__":
    main()
