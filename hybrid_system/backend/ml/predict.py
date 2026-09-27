"""
predict.py
-----------
Loads the trained XGBoost model and exposes a simple predict() function
used by both the FastAPI backend and the simulation engine.
"""
import os

import joblib
import pandas as pd

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "xgb_incident_model.joblib"))

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run 'python backend/ml/train_model.py' first."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_row(features: dict):
    """features must contain every key in FEATURE_COLUMNS."""
    bundle = _load()
    model = bundle["model"]
    cols = bundle["feature_columns"]
    labels = bundle["labels"]

    X = pd.DataFrame([[float(features.get(c, 0.0)) for c in cols]], columns=cols)
    proba = model.predict_proba(X)[0]
    pred_class = int(proba.argmax())
    return {
        "severity_code": pred_class,
        "severity_label": labels[pred_class],
        "confidence": float(proba[pred_class]),
        "probabilities": {labels[i]: float(p) for i, p in enumerate(proba)},
    }
