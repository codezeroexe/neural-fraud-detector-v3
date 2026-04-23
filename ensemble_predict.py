import numpy as np
import joblib
import tensorflow as tf
import os

WEIGHTS = {
    "dnn": 0.55,
    "xgb": 0.22,
    "rf": 0.23
}

THRESHOLD = 0.5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.keras")
XGB_PATH = os.path.join(BASE_DIR, "xgb_model.pkl")
RF_PATH = os.path.join(BASE_DIR, "rf_model.pkl")

_dnn_model = None
_xgb_model = None
_rf_model = None


def load_models():
    global _dnn_model, _xgb_model, _rf_model
    if _dnn_model is None:
        print("[*] Loading DNN model...")
        _dnn_model = tf.keras.models.load_model(MODEL_PATH)
    if _xgb_model is None:
        print("[*] Loading XGBoost model...")
        _xgb_model = joblib.load(XGB_PATH)
    if _rf_model is None:
        print("[*] Loading Random Forest model...")
        _rf_model = joblib.load(RF_PATH)
    return _dnn_model, _xgb_model, _rf_model


def predict_proba(X):
    dnn, xgb, rf = load_models()
    
    prob_dnn = dnn.predict(X, verbose=0).flatten()
    prob_xgb = xgb.predict_proba(X)[:, 1]
    prob_rf = rf.predict_proba(X)[:, 1]
    
    ensemble_prob = (
        WEIGHTS["dnn"] * prob_dnn +
        WEIGHTS["xgb"] * prob_xgb +
        WEIGHTS["rf"] * prob_rf
    )
    
    return ensemble_prob


def predict_proba_individual(X):
    dnn, xgb, rf = load_models()
    
    return {
        "dnn": dnn.predict(X, verbose=0).flatten().tolist(),
        "xgb": xgb.predict_proba(X)[:, 1].tolist(),
        "rf": rf.predict_proba(X)[:, 1].tolist()
    }


def predict_fraud(X, threshold=THRESHOLD):
    prob = predict_proba(X)
    predictions = (prob >= threshold).astype(int)
    return predictions, prob


def get_risk_tier(probability):
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.5:
        return "MEDIUM"
    elif probability >= 0.3:
        return "LOW"
    else:
        return "MINIMAL"


if __name__ == "__main__":
    print(f"Weights: {WEIGHTS}")
    print(f"Threshold: {THRESHOLD}")
    print("Ensemble predictor ready.")