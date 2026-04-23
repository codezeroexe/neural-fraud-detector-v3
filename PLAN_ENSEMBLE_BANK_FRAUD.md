# Bank Fraud Detection: 3-Model Ensemble Implementation Plan

## Goal
- **Primary**: Maximize recall (catch 90%+ fraud → minimize money lost)
- **Secondary**: Boost precision (reduce false positives → reduce customer friction)

---

## Current State (DNN Only - v2)

| Metric | Value |
|--------|-------|
| Precision | 8.65% |
| Recall | 90.44% |
| F1-Score | 0.158 |
| False Positives | 20,482 |
| True Positives | 1,940 |

---

## Strategy: DNN-Led Weighted Ensemble

### Model Roles

| Model | Role | Weight |
|------|------|--------|
| DNN (Neural Network) | High recall anchor | **0.55-0.60** |
| XGBoost | Precision booster + noise filter | **0.20-0.25** |
| Random Forest | Precision booster + robust | **0.20-0.25** |

### Reasoning

```
DNN at 0.55-0.60 weight:
- Maintains high recall (~90%)
- Tree models add precision filtering
- Ensemble noise-cancels false positives

XGB + RF at combined 0.40-0.45:
- Different error patterns than DNN
- True fraud → all 3 agree (pass through)
- False positives → DNN overfits, trees don't (filtered)
```

---

## Implementation Plan

### Phase 1: Train Tree Models

#### 1.1 XGBoost (`xgb_train.py`)

```python
import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from fraud_detection import preprocess_for_nn

# Load data
df_train = pd.read_csv("fraudTrain.csv")
X, y, encoders, scaler, feature_cols = preprocess_for_nn(df_train, fit=True)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Calculate class weight
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=10)

# Save
joblib.dump(model, "xgb_model.pkl")
print("XGBoost model saved to xgb_model.pkl")
```

#### 1.2 Random Forest (`rf_train.py`)

```python
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from fraud_detection import preprocess_for_nn

# Load data
df_train = pd.read_csv("fraudTrain.csv")
X, y, encoders, scaler, feature_cols = preprocess_for_nn(df_train, fit=True)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Save
joblib.dump(model, "rf_model.pkl")
print("Random Forest model saved to rf_model.pkl")
```

---

### Phase 2: Ensemble Logic

#### 2.1 Ensemble Predictor (`ensemble_predict.py`)

```python
import numpy as np
import joblib
import tensorflow as tf

# Model weights (DNN-led for max recall)
WEIGHTS = {
    "dnn": 0.55,
    "xgb": 0.22,
    "rf": 0.23
}

# Threshold (unchanged = 0.5)
THRESHOLD = 0.5

def load_models():
    """Load all 3 models."""
    dnn = tf.keras.models.load_model("fraud_model.keras")
    xgb = joblib.load("xgb_model.pkl")
    rf = joblib.load("rf_model.pkl")
    return dnn, xgb, rf

def predict_proba(X):
    """Get ensemble probability."""
    dnn, xgb, rf = load_models()
    
    # Get probabilities from each model
    prob_dnn = dnn.predict(X, verbose=0).flatten()
    prob_xgb = xgb.predict_proba(X)[:, 1]
    prob_rf = rf.predict_proba(X)[:, 1]
    
    # Weighted average
    ensemble_prob = (
        WEIGHTS["dnn"] * prob_dnn +
        WEIGHTS["xgb"] * prob_xgb +
        WEIGHTS["rf"] * prob_rf
    )
    
    return ensemble_prob

def predict_fraud(X, threshold=THRESHOLD):
    """Predict fraud (binary)."""
    prob = predict_proba(X)
    predictions = (prob >= threshold).astype(int)
    return predictions, prob

def get_risk_tier(probability):
    """Risk tier for bank workflow."""
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.5:
        return "MEDIUM"
    elif probability >= 0.3:
        return "LOW"
    else:
        return "MINIMAL"

# CLI test
if __name__ == "__main__":
    print(f"Weights: {WEIGHTS}")
    print("Ensemble predictor ready.")
```

---

### Phase 3: UI Integration

#### 3.1 Update `app.py` (Integration)

```python
# Add to imports
from ensemble_predict import predict_proba, get_risk_tier

# Update predict endpoint (around line 330)
@app.route("/predict", methods=["POST"])
def predict():
    # ... existing preprocessing (lines 334-346) ...
    
    # OLD (single model)
    # probability = float(model.predict(X, verbose=0).flatten()[0])
    
    # NEW (ensemble - DNN-led)
    probability = predict_proba(X)[0]
    
    # Risk tier (bank workflow)
    risk = get_risk_tier(probability)
    
    # Decision
    prediction = "FRAUD" if probability >= THRESHOLD else "LEGITIMATE"
    
    # Risk level mapping for display
    if probability >= 0.7:
        risk = "HIGH"
    elif probability >= 0.5:
        risk = "MEDIUM"
    elif probability >= 0.3:
        risk = "LOW"
    else:
        risk = "LOW"
    
    return jsonify({
        "probability": round(probability * 100, 2),
        "prediction": prediction,
        "risk": risk,
        "threshold": round(THRESHOLD * 100, 2),
        "distance_km": round(distance_km, 1),
        "ensemble": True,
        "weights": {"dnn": 0.55, "xgb": 0.22, "rf": 0.23}
    })
```

---

### Phase 4: Evaluation

#### 4.1 Evaluation Script (`evaluate_ensemble.py`)

```python
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from fraud_detection import preprocess_for_nn

# Load test data
df_test = pd.read_csv("fraudTest.csv")
X_test, y_test, _, _, _ = preprocess_for_nn(df_test, encoders=encoders, scaler=scaler, fit=False)

# Load models and preprocessor
preprocessor = joblib.load("preprocessor.pkl")
encoders = preprocessor["encoders"]
scaler = preprocessor["scaler"]

# Import ensemble
from ensemble_predict import predict_proba, WEIGHTS, THRESHOLD

# Get predictions
y_prob = predict_proba(X_test)
y_pred = (y_prob >= THRESHOLD).astype(int)

# Metrics
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("=" * 60)
print("ENSEMBLE EVALUATION RESULTS")
print("=" * 60)
print(f"\nConfusion Matrix:")
print(f"  TN: {tn:,}  FP: {fp:,}")
print(f"  FN: {fn:,}  TP: {tp:,}")
print(f"\nWeights: {WEIGHTS}")
print(f"Threshold: {THRESHOLD}")
print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"\n--- vs Single DNN ---")
print(f"Precision: 0.0865 → {precision:.4f} (Delta: {precision - 0.0865:+.4f})")
print(f"Recall: 0.9044 → {recall:.4f} (Delta: {recall - 0.9044:+.4f})")
print(f"F1: 0.1579 → {f1:.4f} (Delta: {f1 - 0.1579:+.4f})")
print("=" * 60)
```

---

## Expected Results

| Metric | Before (DNN) | After (Ensemble) | Target |
|--------|--------------|---------------|--------|
| Recall | 90.44% | 88-91% | MAX (loss <2%) |
| Precision | 8.65% | 12-18% | +5-10% |
| F1-Score | 0.158 | 0.22-0.30 | +0.07-0.15 |
| False Positives | 20,482 | 12,000-16,000 | -25-40% |

---

## File Structure

```
v2/
├── xgb_train.py              # NEW - Train XGBoost
├── rf_train.py               # NEW - Train Random Forest
├── ensemble_predict.py        # NEW - Weighted ensemble
├── evaluate_ensemble.py     # NEW - Compare metrics
├── app.py               # UPDATED - Use ensemble
├── fraud_model.keras    # EXISTING - DNN
├── preprocessor.pkl   # EXISTING
├── xgb_model.pkl   # NEW
└── rf_model.pkl     # NEW
```

---

## Execution Order

1. ✅ `pip install xgboost` (if needed)
2. `python xgb_train.py` → xgb_model.pkl
3. `python rf_train.py` → rf_model.pkl
4. `python ensemble_predict.py` (verify loads)
5. `python app.py` → restart with ensemble
6. `python evaluate_ensemble.py` → see metrics

---

## Risk Tiers for Bank Dashboard

| Tier | Probability | Color | Action |
|------|-------------|-------|--------|
| HIGH | ≥70% | Red | Auto-block |
| MEDIUM | 50-70% | Orange | Same-day review |
| LOW | 30-50% | Yellow | Batch review |
| MINIMAL | <30% | Green | Approve |

---

## Questions Before Implementation

1. **Weights**: DNN at 0.55, XGB 0.22, RF 0.23 — acceptable?
2. **Run evaluation after**: Want full threshold sweep to optimize F1?
3. **File location**: Save .md to v2/ or root?