# How to Run Neural Fraud Detector v3
## Complete Setup and Execution Guide

---

## Project Location

Replace `[PROJECT_PATH]` with your actual project folder path:

| Platform | Example Path |
|----------|--------------|
| **macOS** | `/Users/yourname/MLDL project` or `~/MLDL project` |
| **Windows** | `C:\Users\YourName\MLDL project` or `%USERPROFILE%\MLDL project` |
| **Linux** | `/home/yourname/MLDL project` or `~/MLDL project` |

---

## Quick Start

### Method 1: Double-Click (Easiest)

1. Download from [Releases](https://github.com/codezeroexe/neural-fraud-detector-v3/releases)
2. Extract the folder
3. Double-click `Neural Fraud Detector.command` (macOS) or `Neural Fraud Detector.bat` (Windows)

### Method 2: Terminal

```bash
# Navigate to project (use your actual path)
cd [PROJECT_PATH]

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies (first time only)
pip install -r requirements.txt

# Run the web dashboard
python app.py
```

Open http://127.0.0.1:5000

---

## Method 1: Terminal / Command Line

### macOS / Linux

**Step 1: Open Terminal**

- macOS: Press `Cmd + Space`, type "Terminal", press Enter
- Linux: Press `Ctrl + Alt + T`

**Step 2: Navigate to Project**

```bash
cd ~/MLDL\ project
```

**Step 3: Create Virtual Environment (First Time Only)**

```bash
python3 -m venv venv
```

**Step 4: Activate Virtual Environment**

```bash
source venv/bin/activate
```

**Step 5: Install Dependencies**

```bash
pip install -r requirements.txt
```

**Step 6: Run Dashboard**

```bash
python app.py
```

**Step 7: Open Browser**

Navigate to: http://127.0.0.1:5000

---

### Windows

**Step 1: Open Command Prompt**

- Press `Win + R`, type `cmd`, press Enter
- Or search for "Command Prompt"

**Step 2: Navigate to Project**

```bash
cd %USERPROFILE%\MLDL project
```

**Step 3: Create Virtual Environment (First Time Only)**

```bash
python -m venv venv
```

**Step 4: Activate Virtual Environment**

```bash
venv\Scripts\activate
```

**Step 5: Install Dependencies**

```bash
pip install -r requirements.txt
```

**Step 6: Run Dashboard**

```bash
python app.py
```

**Step 7: Open Browser**

Navigate to: http://127.0.0.1:5000

---

## Training Models (Optional)

Models are already trained and included. To retrain:

### Train DNN (Deep Neural Network)

```bash
python fraud_detection.py
```

### Train XGBoost

```bash
python xgb_train.py
```

### Train Random Forest

```bash
python rf_train.py
```

### Train All Three

```bash
python fraud_detection.py
python xgb_train.py
python rf_train.py
```

---

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **EDA Analysis** | 13 visualizations of data patterns |
| **Architecture** | DNN, XGBoost, RF architecture diagrams |
| **Stacked** | Ensemble visualization & model weights |
| **Training** | Training curves for DNN |
| **Tuning** | Hyperparameter search results |
| **Evaluation** | Model comparison & metrics |
| **Predict** | Real-time fraud prediction |

---

## Troubleshooting

### Port Already in Use

```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9  # macOS/Linux
netstat -aon | findstr :5000    # Windows
```

### Virtual Environment Issues

```bash
# Delete and recreate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Model Not Found

If you get errors about models not found, retrain them:

```bash
python fraud_detection.py
python xgb_train.py
python rf_train.py
```

---

## Stopping the Dashboard

- Press `Ctrl + C` in the terminal
- Or close the terminal window

---

## Files Generated

| File | Description |
|------|-------------|
| `fraud_model.keras` | Trained DNN model |
| `xgb_model.pkl` | Trained XGBoost model |
| `rf_model.pkl` | Trained Random Forest model |
| `preprocessor.pkl` | Encoders and scaler |
| `evaluation_results.json` | Evaluation metrics |
| `training_history.json` | DNN training history |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard |
| `/predict` | POST | Predict fraud |
| `/api/architecture` | GET | Model architecture |
| `/api/evaluation` | GET | Evaluation metrics |
| `/api/training` | GET | Training history |
| `/api/tuning` | GET | Hyperparameter trials |