"""
Flask Web Application for Credit Card Fraud Detection
======================================================
Multi-page dashboard with:
  1. Interactive Prediction Interface
  2. Model Architecture Visualization
  3. Training History Visualization
  4. Evaluation Results Dashboard
  5. Hyperparameter Tuning Results
"""

import os
import csv
import ast
import json
import math
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from ensemble_predict import predict_proba, predict_proba_individual, get_risk_tier, WEIGHTS

# ──────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────
app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.keras")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")
HYPERPARAM_PATH = os.path.join(BASE_DIR, "hyperparam_results.csv")
XGB_MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.pkl")
RF_MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")
THRESHOLD = 0.5

# ──────────────────────────────────────────────────────────
# Load artifacts once at startup
# ──────────────────────────────────────────────────────────
print("[*] Loading DNN model ...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[*] DNN model loaded.")

print("[*] Loading XGBoost model ...")
xgb_model = joblib.load(XGB_MODEL_PATH)
print("[*] XGBoost model loaded.")

print("[*] Loading Random Forest model ...")
rf_model = joblib.load(RF_MODEL_PATH)
print("[*] Random Forest model loaded.")

print("[*] Loading preprocessor ...")
preprocessor = joblib.load(PREPROCESSOR_PATH)
encoders = preprocessor["encoders"]
scaler = preprocessor["scaler"]
feature_cols = preprocessor["feature_cols"]
print(f"[*] Preprocessor loaded. Features: {feature_cols}")

# Check for an optimal threshold file (optional)
THRESHOLD_PATH = os.path.join(BASE_DIR, "optimal_threshold.txt")
if os.path.exists(THRESHOLD_PATH):
    with open(THRESHOLD_PATH, "r") as f:
        THRESHOLD = float(f.read().strip())
    print(f"[*] Using optimal threshold: {THRESHOLD}")
else:
    print(f"[*] Using default threshold: {THRESHOLD}")

# ──────────────────────────────────────────────────────────
# Load or compute LIVE evaluation metrics & training history
# ──────────────────────────────────────────────────────────
EVAL_JSON_PATH = os.path.join(BASE_DIR, "evaluation_results.json")
HISTORY_JSON_PATH = os.path.join(BASE_DIR, "training_history.json")
TEST_CSV_PATH = os.path.join(BASE_DIR, "fraudTest.csv")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "fraudTrain.csv")

cached_eval = None
cached_training = None

# Strategy 1: Load pre-computed results from JSON (saved by fraud_detection.py)
if os.path.exists(EVAL_JSON_PATH):
    with open(EVAL_JSON_PATH, "r") as f:
        cached_eval = json.load(f)
    print(f"[*] Loaded evaluation metrics from evaluation_results.json")

if os.path.exists(HISTORY_JSON_PATH):
    with open(HISTORY_JSON_PATH, "r") as f:
        cached_training = json.load(f)
    print(f"[*] Loaded training history from training_history.json")

# Test data loaded lazily on first ensemble eval request
X_test = None
y_test = None

def _ensure_test_data():
    """Load test data on first request (lazy load)."""
    global X_test, y_test
    if X_test is None and os.path.exists(TEST_CSV_PATH):
        try:
            from fraud_detection import preprocess_for_nn
            print("[*] Lazy-loading test data for ensemble evaluation...")
            _df = pd.read_csv(TEST_CSV_PATH)
            X_test, y_test, _, _, _ = preprocess_for_nn(_df, encoders=encoders, scaler=scaler, fit=False)
            del _df
            print(f"[*] Test data loaded: {X_test.shape[0]} samples")
        except Exception as e:
            print(f"[!] Failed to load test data: {e}")

# Strategy 2: If no eval JSON exists, compute LIVE from the test CSV + trained model
if cached_eval is None and os.path.exists(TEST_CSV_PATH):
    print("[*] No evaluation_results.json found. Computing LIVE metrics from fraudTest.csv ...")
    from fraud_detection import preprocess_for_nn

    _df_test = pd.read_csv(TEST_CSV_PATH)
    _X_test, _y_test, _, _, _ = preprocess_for_nn(
        _df_test, encoders=encoders, scaler=scaler, fit=False
    )
    del _df_test  # free ~150 MB

    _y_prob = model.predict(_X_test, batch_size=4096, verbose=0).flatten()
    _y_pred = (_y_prob >= THRESHOLD).astype(int)

    _cm = sk_confusion_matrix(_y_test, _y_pred)
    _tn, _fp, _fn, _tp = _cm.ravel()
    _accuracy = (_tn + _tp) / (_tn + _fp + _fn + _tp)

    _prec_fraud = float(_tp / (_tp + _fp)) if (_tp + _fp) > 0 else 0.0
    _rec_fraud  = float(_tp / (_tp + _fn)) if (_tp + _fn) > 0 else 0.0
    _f1_fraud   = (2 * _prec_fraud * _rec_fraud / (_prec_fraud + _rec_fraud)
                   if (_prec_fraud + _rec_fraud) > 0 else 0.0)
    _prec_legit = float(_tn / (_tn + _fn)) if (_tn + _fn) > 0 else 0.0
    _rec_legit  = float(_tn / (_tn + _fp)) if (_tn + _fp) > 0 else 0.0
    _f1_legit   = (2 * _prec_legit * _rec_legit / (_prec_legit + _rec_legit)
                   if (_prec_legit + _rec_legit) > 0 else 0.0)

    _roc = roc_auc_score(_y_test, _y_prob)
    _pr  = average_precision_score(_y_test, _y_prob)

    # Get training set stats efficiently (just the label column)
    _total_train, _fraud_rate_train = 0, 0.0
    if os.path.exists(TRAIN_CSV_PATH):
        _lbl = pd.read_csv(TRAIN_CSV_PATH, usecols=["is_fraud"])
        _total_train = len(_lbl)
        _fraud_rate_train = round(100 * float(_lbl["is_fraud"].mean()), 2)
        del _lbl

    cached_eval = {
        "confusion_matrix": {
            "tn": int(_tn), "fp": int(_fp), "fn": int(_fn), "tp": int(_tp)
        },
        "metrics": {
            "roc_auc":         round(float(_roc), 4),
            "pr_auc":          round(float(_pr), 4),
            "accuracy":        round(float(_accuracy), 4),
            "precision_fraud": round(_prec_fraud, 4),
            "recall_fraud":    round(_rec_fraud, 4),
            "f1_fraud":        round(_f1_fraud, 4),
            "precision_legit": round(_prec_legit, 4),
            "recall_legit":    round(_rec_legit, 4),
            "f1_legit":        round(_f1_legit, 4),
        },
        "dataset": {
            "total_train":      _total_train,
            "total_test":       int(len(_y_test)),
            "fraud_rate_train": _fraud_rate_train,
            "fraud_rate_test":  round(100 * float(np.mean(_y_test)), 2),
        }
    }

    # Store test data for ensemble evaluation
    X_test = _X_test
    y_test = _y_test

    # Save so next startup is instant
    with open(EVAL_JSON_PATH, "w") as f:
        json.dump(cached_eval, f, indent=2)
    print(f"[*] Live evaluation complete — saved to evaluation_results.json")
    print(f"    ROC-AUC: {_roc:.4f}  |  PR-AUC: {_pr:.4f}  |  Precision(Fraud): {_prec_fraud:.4f}  |  Recall(Fraud): {_rec_fraud:.4f}")
    del _X_test, _y_test, _y_prob, _y_pred  # free memory

elif cached_eval is None:
    print("[!] WARNING: No evaluation data available.")
    print("[!] Place fraudTest.csv in the project folder, or run 'python fraud_detection.py'.")

# ──────────────────────────────────────────────────────────
# Extract model architecture info at startup
# ──────────────────────────────────────────────────────────
def get_model_architecture():
    """Extract detailed layer info from the loaded Keras model."""
    layers_info = []
    total_params = 0
    trainable_params = 0

    for layer in model.layers:
        config = layer.get_config()
        layer_type = layer.__class__.__name__
        output_shape = layer.output_shape if hasattr(layer, 'output_shape') else None

        l_params = layer.count_params()
        l_trainable = sum(
            tf.size(w).numpy() for w in layer.trainable_weights
        ) if layer.trainable_weights else 0

        total_params += l_params
        trainable_params += l_trainable

        info = {
            "name": layer.name,
            "type": layer_type,
            "output_shape": str(output_shape),
            "params": int(l_params),
            "trainable": int(l_trainable),
            "config": {},
        }

        if layer_type == "Dense":
            info["config"]["units"] = config.get("units")
            info["config"]["activation"] = config.get("activation")
        elif layer_type == "Dropout":
            info["config"]["rate"] = config.get("rate")
        elif layer_type == "BatchNormalization":
            info["config"]["momentum"] = config.get("momentum")
        elif layer_type == "Activation":
            info["config"]["activation"] = config.get("activation")

        layers_info.append(info)

    return {
        "version": "v3",
        "ensemble": True,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(total_params - trainable_params),
        "models": {
            "dnn": {
                "name": "Deep Neural Network",
                "type": "MLP",
                "weight": WEIGHTS["dnn"],
                "description": "High recall anchor - captures complex patterns",
                "layers": layers_info,
                "total_params": int(total_params),
                "trainable_params": int(trainable_params),
                "architecture": "MLP [15→256→128→64→1]",
            },
            "xgb": {
                "name": "XGBoost",
                "type": "Gradient Boosting (Boosted Trees)",
                "weight": WEIGHTS["xgb"],
                "description": "Precision booster - sequential tree correction of errors",
                "n_estimators": int(xgb_model.n_estimators),
                "max_depth": int(xgb_model.max_depth),
                "learning_rate": float(xgb_model.learning_rate),
                "subsample": float(xgb_model.subsample) if xgb_model.subsample else 1.0,
                "colsample_bytree": float(xgb_model.colsample_bytree) if xgb_model.colsample_bytree else 1.0,
                "feature_importances": dict(zip(feature_cols, [float(x) for x in xgb_model.feature_importances_])),
                "top_features": [(f, float(imp)) for f, imp in sorted(zip(feature_cols, xgb_model.feature_importances_), key=lambda x: x[1], reverse=True)[:10]],
            },
            "rf": {
                "name": "Random Forest",
                "type": "Bagging Ensemble (Decision Trees)",
                "weight": WEIGHTS["rf"],
                "description": "Robust baseline - reduces variance via parallel bagging",
                "n_estimators": int(rf_model.n_estimators),
                "max_depth": int(rf_model.max_depth),
                "max_features": str(rf_model.max_features) if rf_model.max_features else "sqrt",
                "class_weight": str(rf_model.class_weight),
                "feature_importances": dict(zip(feature_cols, [float(x) for x in rf_model.feature_importances_])),
                "top_features": [(f, float(imp)) for f, imp in sorted(zip(feature_cols, rf_model.feature_importances_), key=lambda x: x[1], reverse=True)[:10]],
            }
        },
        "ensemble_weights": WEIGHTS,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "input_features": feature_cols,
        "architecture_summary": "3-Model Weighted Ensemble: DNN(55%) + XGBoost(22%) + Random Forest(23%)",
    }


# ──────────────────────────────────────────────────────────
# Load hyperparameter tuning results
# ──────────────────────────────────────────────────────────
def load_hyperparam_results():
    """Parse hyperparam_results.csv into structured data."""
    results = []
    if not os.path.exists(HYPERPARAM_PATH):
        return results

    with open(HYPERPARAM_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                params = ast.literal_eval(row["params"])
                results.append({
                    "trial": int(row["trial"]),
                    "hidden_layers": str(params.get("hidden_layers", [])),
                    "dropout_rate": params.get("dropout_rate", 0),
                    "learning_rate": params.get("learning_rate", 0),
                    "batch_size": params.get("batch_size", 0),
                    "epochs": params.get("epochs", 0),
                    "roc_auc": round(float(row["roc_auc"]), 4),
                    "pr_auc": round(float(row["pr_auc"]), 4),
                    "score": round(float(row["score"]), 4),
                })
            except (ValueError, SyntaxError):
                continue

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ──────────────────────────────────────────────────────────
# Helper: haversine distance
# ──────────────────────────────────────────────────────────
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


# ──────────────────────────────────────────────────────────
# Helper: preprocess a single transaction
# ──────────────────────────────────────────────────────────
def preprocess_single(data: dict) -> np.ndarray:
    """Convert raw form input into a scaled feature vector."""
    trans_dt = pd.to_datetime(data["trans_date_trans_time"])
    dob_dt = pd.to_datetime(data["dob"])

    hour = trans_dt.hour
    day_of_week = trans_dt.dayofweek
    month = trans_dt.month
    day_of_month = trans_dt.day
    distance_km = float(data["distance_km"])

    # Provide mean US coordinates for neutral location signal
    lat, long = 39.8, -98.5
    merch_lat, merch_long = 39.8, -98.5
    age = (trans_dt - dob_dt).days // 365

    def encode(col, value):
        le = encoders[col]
        val_str = str(value)
        if val_str in le.classes_:
            return le.transform([val_str])[0]
        return -1

    feature_map = {
        "category": encode("category", data["category"]),
        "amt": float(data["amt"]),
        "gender": encode("gender", data["gender"]),
        "state": encode("state", data["state"]),
        "lat": lat,
        "long": long,
        "city_pop": float(data["city_pop"]),
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "day_of_month": day_of_month,
        "distance_km": distance_km,
        "age": age,
    }

    vec = np.array([[feature_map[col] for col in feature_cols]])
    vec = scaler.transform(vec)

    # Clip extreme out-of-distribution values to prevent sigmoid collapse
    vec = np.clip(vec, -5.0, 5.0)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    return vec, distance_km


# ──────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main dashboard."""
    categories = list(encoders["category"].classes_)
    genders = list(encoders["gender"].classes_)
    states = list(encoders["state"].classes_)
    return render_template(
        "index.html",
        categories=categories,
        genders=genders,
        states=states,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Accept JSON transaction data, return fraud prediction."""
    try:
        data = request.get_json(force=True)

        required = [
            "amt", "category", "gender", "state",
            "distance_km", "city_pop",
            "dob", "trans_date_trans_time",
        ]
        missing = [f for f in required if f not in data or data[f] == ""]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        # Preprocess
        X, distance_km = preprocess_single(data)

        # Ensemble prediction
        probability = float(predict_proba(X)[0])
        individual_probs = predict_proba_individual(X)
        avg_prob = float(np.mean([
            individual_probs["dnn"][0],
            individual_probs["xgb"][0],
            individual_probs["rf"][0]
        ]))

        # International distance heuristic
        if distance_km > 3000:
            probability = max(probability, 0.85)

        # Decision
        prediction = "FRAUD" if probability >= THRESHOLD else "LEGITIMATE"
        risk = get_risk_tier(probability)

        return jsonify({
            "probability": round(probability * 100, 2),
            "prediction": prediction,
            "risk": risk,
            "threshold": round(THRESHOLD * 100, 2),
            "distance_km": round(distance_km, 1),
            "ensemble": True,
            "weights": WEIGHTS,
            "model_probs": {
                "dnn": round(individual_probs["dnn"][0] * 100, 2),
                "xgb": round(individual_probs["xgb"][0] * 100, 2),
                "rf": round(individual_probs["rf"][0] * 100, 2)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/architecture")
def api_architecture():
    """Return model architecture details as JSON."""
    return jsonify(get_model_architecture())


cached_ensemble_eval = None
ENSEMBLE_EVAL_JSON = os.path.join(BASE_DIR, "ensemble_eval.json")

# Pre-load ensemble eval if file exists
if os.path.exists(ENSEMBLE_EVAL_JSON):
    with open(ENSEMBLE_EVAL_JSON, "r") as f:
        cached_ensemble_eval = json.load(f)

@app.route("/api/evaluation")
def api_evaluation():
    """Return evaluation metrics — always read fresh from JSON files."""
    global cached_ensemble_eval
    response = {}

    # Load evaluation results fresh from disk
    if os.path.exists(EVAL_JSON_PATH):
        with open(EVAL_JSON_PATH, "r") as f:
            response = json.load(f)
    elif cached_eval is not None:
        response = dict(cached_eval)
    else:
        return jsonify({"error": "No evaluation data available. Run fraud_detection.py first."}), 503

    # Load training history fresh from disk
    if os.path.exists(HISTORY_JSON_PATH):
        with open(HISTORY_JSON_PATH, "r") as f:
            training_data = json.load(f)
        response.update(training_data)
    elif cached_training is not None:
        response.update(cached_training)
    else:
        # Provide empty structure so the frontend charts don't crash
        response["training"] = {
            "epochs_completed": 0, "best_epoch": 0, "early_stopped": False,
            "final_train_acc": 0, "final_train_auc": 0,
            "final_val_acc": 0, "final_val_auc": 0,
            "final_val_loss": 0, "learning_rate_final": 0,
        }
        response["history"] = {
            "epochs": [], "train_loss": [], "val_loss": [],
            "train_auc": [], "val_auc": [], "train_acc": [], "val_acc": [],
        }

    # Run ensemble evaluation only if not cached (lazy, one-time)
    if cached_ensemble_eval is None:
        cached_ensemble_eval = run_ensemble_evaluation()
    response["ensemble_eval"] = cached_ensemble_eval

    return jsonify(response)


def run_ensemble_evaluation():
    """Evaluate all three models individually and as ensemble."""
    try:
        _ensure_test_data()
        if X_test is None or y_test is None:
            return {"error": "Test data not available"}

        results = {}

        # Individual model predictions
        dnn_probs = model.predict(X_test, verbose=0).flatten()
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        rf_probs = rf_model.predict_proba(X_test)[:, 1]

        # Apply weights (normalize: weights are percentages like 55, 22, 23)
        w_total = WEIGHTS["dnn"] + WEIGHTS["xgb"] + WEIGHTS["rf"]
        w_dnn = WEIGHTS["dnn"] / w_total
        w_xgb = WEIGHTS["xgb"] / w_total
        w_rf = WEIGHTS["rf"] / w_total
        ensemble_probs = w_dnn * dnn_probs + w_xgb * xgb_probs + w_rf * rf_probs

        for name, probs in [("dnn", dnn_probs), ("xgb", xgb_probs), ("rf", rf_probs), ("ensemble", ensemble_probs)]:
            preds = (probs >= THRESHOLD).astype(int)
            tn = np.sum((preds == 0) & (y_test == 0))
            fp = np.sum((preds == 1) & (y_test == 0))
            fn = np.sum((preds == 0) & (y_test == 1))
            tp = np.sum((preds == 1) & (y_test == 1))

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            try:
                roc_auc = roc_auc_score(y_test, probs)
            except:
                roc_auc = 0

            try:
                pr_auc = average_precision_score(y_test, probs)
            except:
                pr_auc = 0

            results[name] = {
                "roc_auc": round(roc_auc, 4),
                "pr_auc": round(pr_auc, 4),
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "f1": round(f1, 4),
                "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            }

        return results
    except Exception as e:
        return {"error": str(e)}


@app.route("/api/tuning")
def api_tuning():
    """Return hyperparameter tuning trial results."""
    results = load_hyperparam_results()
    return jsonify({"trials": results, "total": len(results)})


# ──────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
