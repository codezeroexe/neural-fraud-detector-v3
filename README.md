# Neural Fraud Detector v3

> **v3** of the [neural-fraud-detector](https://github.com/codezeroexe/neural-fraud-detector).  
> **3-Model Stacked Ensemble** fraud detection with improved precision.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Quick Start

### Double-Click to Run (Recommended)

1. Download release from [Releases](https://github.com/codezeroexe/neural-fraud-detector-v3/releases)
2. Extract the folder
3. Double-click `Neural Fraud Detector.command` (macOS) or `Neural Fraud Detector.bat` (Windows)
4. Done! Everything happens automatically.

The launcher will:
- Create virtual environment
- Install dependencies  
- Download dataset (if not present)
- Train all 3 models (if not present)
- Launch dashboard in browser

### Manual Start

```bash
git clone https://github.com/codezeroexe/neural-fraud-detector-v3.git
cd neural-fraud-detector-v3
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

---

## What's New in v3

| Feature | v2 | v3 |
|---------|----|----|
| Models | 1 (DNN) | 3 (DNN + XGBoost + RF) |
| Architecture | Single MLP | Weighted Stacked Ensemble |
| Precision | 8.65% | ~22.73% (improved) |
| False Positives | 20,482 | ~6,575 (reduced) |
| F1-Score | 0.158 | 0.363 (2.3x better) |

---

## Performance

### Ensemble Configuration

| Model | Weight | Role |
|-------|--------|------|
| Deep Neural Network | 55% | High recall anchor |
| XGBoost | 22% | Precision filter |
| Random Forest | 23% | Robust baseline |

### Test Set Results (Ensemble)

| Metric | DNN | XGBoost | RF | **Stacked** |
|--------|-----|---------|-----|-------------|
| ROC-AUC | 0.984 | 0.993 | 0.990 | **0.991** |
| Recall | 90.4% | 92.1% | 79.6% | **90.2%** |
| Precision | 8.7% | 33.7% | 57.7% | **22.7%** |
| F1-Score | 0.16 | 0.49 | 0.67 | **0.36** |

### Confusion Matrix (Stacked Ensemble)

| | Predicted Legit | Predicted Fraud |
|--|----------------|-----------------|
| **Actual Legit** | 546,999 (TN) | 6,575 (FP) |
| **Actual Fraud** | 211 (FN) | 1,934 (TP) |

---

## Model Architecture

### Ensemble Structure

```
Input (15 features)
    ↓
┌─────────────────────────────────────┐
│  DNN (55%) → High Recall            │
│  Dense(256) + Dense(128) + Dense(64)  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  XGBoost (22%) → Precision          │
│  100 Trees, max_depth=6              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Random Forest (23%) → Robustness   │
│  100 Trees, max_depth=10            │
└─────────────────────────────────────┘
    ↓
Weighted Average → Final Prediction
```

### Individual Model Parameters

| Model | Type | Estimators | Max Depth |
|-------|------|------------|-----------|
| DNN | MLP | - | 3 layers |
| XGBoost | Gradient Boosted | 100 | 6 |
| Random Forest | Bagged Trees | 100 | 10 |

---

## Features (15 Input)

| # | Feature | Description |
|---|---------|-------------|
| 1 | amt | Transaction amount |
| 2 | category | Merchant category (encoded) |
| 3 | gender | M/F (encoded) |
| 4 | state | US state (encoded) |
| 5 | lat | Cardholder latitude |
| 6 | long | Cardholder longitude |
| 7 | merch_lat | Merchant latitude |
| 8 | merch_long | Merchant longitude |
| 9 | city_pop | City population |
| 10 | hour | Transaction hour (0-23) |
| 11 | day_of_week | Day (0-6) |
| 12 | month | Month (1-12) |
| 13 | day_of_month | Day (1-31) |
| 14 | distance_km | Haversine distance |
| 15 | age | Cardholder age |

---

## Dashboard Features

### 7 Tabs

| Tab | Features |
|-----|----------|
| **EDA Analysis** | 13 interactive visualizations |
| **Architecture** | Network diagrams, layer details |
| **Stacked** | Ensemble flow, model weight pie chart |
| **Training** | Loss/AUC curves, epoch stats |
| **Tuning** | Hyperparameter search results |
| **Evaluation** | Confusion matrix, metrics, model comparison |
| **Predict** | Real-time fraud prediction |

### Design

- **Light/Dark Theme** — Toggle with localStorage
- **Minimalist** — Clean monochromatic design
- **Responsive** — Desktop and mobile
- **Ensemble Visualization** — Interactive flow diagrams

---

## Installation (Manual)

```bash
# Clone
git clone https://github.com/codezeroexe/neural-fraud-detector-v3.git
cd neural-fraud-detector-v3

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt

# Train models (if no models)
python fraud_detection.py   # DNN
python xgb_train.py        # XGBoost
python rf_train.py        # Random Forest

# Run dashboard
python app.py
```

---

## Project Structure

```
v3/
├── app.py                      # Flask dashboard
├── fraud_detection.py         # DNN training
├── xgb_train.py              # XGBoost training
├── rf_train.py               # Random Forest training
├── ensemble_predict.py       # Ensemble inference
├── tune_model.py             # Hyperparameter tuning
├── launch.py                # GUI launcher
├── Neural Fraud Detector.command    # macOS launcher
├── Neural Fraud Detector.bat       # Windows launcher
├── requirements.txt
├── fraud_model.keras         # Trained DNN
├── xgb_model.pkl           # Trained XGBoost
├── rf_model.pkl             # Trained Random Forest
├── preprocessor.pkl         # Encoders & scaler
├── static/styles.css
└── templates/index.html
```

---

## Usage Example

```python
import joblib
from tensorflow.keras.models import load_model

# Load all 3 models
dnn = load_model('fraud_model.keras')
xgb = joblib.load('xgb_model.pkl')
rf = joblib.load('rf_model.pkl')
artifacts = joblib.load('preprocessor.pkl')

# Get predictions
dnn_prob = dnn.predict(X)[0][0]
xgb_prob = xgb.predict_proba(X)[0][1]
rf_prob = rf.predict_proba(X)[0][1]

# Weighted ensemble
weights = [0.55, 0.22, 0.23]
weighted_prob = weights[0]*dnn_prob + weights[1]*xgb_prob + weights[2]*rf_prob

# Result
if weighted_prob > 0.5:
    print(f"FRAUD ({weighted_prob*100:.1f}%)")
else:
    print(f"LEGITIMATE ({(1-weighted_prob)*100:.1f}%)")
```

---

## Tech Stack

| Layer | Technology |
|-------|-------------|
| Model | TensorFlow/Keras + XGBoost + scikit-learn |
| Backend | Flask |
| Frontend | Vanilla HTML/CSS/JS |
| Charts | Chart.js |
| Dataset | IEEE-CIS |

---

## Why Ensemble?

1. **Diversified Risk** — Different model types catch different fraud patterns
2. **Reduced False Positives** — Ensemble consensus reduces false alarms
3. **Better Precision** — 2.6x improvement over DNN alone
4. **Robust** — If one model fails, others compensate

### Model Comparison Summary

| Model | Best For |
|-------|----------|
| DNN | High recall (catch more fraud) |
| XGBoost | Balanced precision/recall |
| Random Forest | High precision (fewer false alarms) |
| **Stacked** | Best overall F1-Score |

---

## Future Work

- SHAP explanations for ensemble
- Transaction sequence modeling
- Real-time model retraining
- API deployment

---

**Dataset**: IEEE-CIS Fraud Detection on Kaggle  
**Built with**: TensorFlow, XGBoost, scikit-learn, Flask, Chart.js