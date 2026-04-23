# Code Guide: fraud_detection.py
## Block-by-Block Explanation

---

## Block 1: Imports (Lines 1-17)

```python
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, average_precision_score
)
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dense, Input, BatchNormalization, Dropout, Activation
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
```

**What each import does:**

| Import | Purpose |
|--------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `joblib` | Save/load preprocessor (encoders, scaler) |
| `os` | File path operations |
| `StandardScaler` | Normalize features to mean=0, std=1 |
| `LabelEncoder` | Convert categorical strings to numbers |
| `classification_report` | Precision, recall, F1 metrics |
| `confusion_matrix` | TN, FP, FN, TP counts |
| `roc_auc_score` | ROC-AUC metric |
| `average_precision_score` | Precision-Recall AUC |
| `tensorflow` | Deep learning framework |
| `Sequential` | Linear stack of layers |
| `load_model` | Load saved Keras model |
| `Dense` | Fully connected layer |
| `Input` | Input layer definition |
| `BatchNormalization` | Normalize layer inputs |
| `Dropout` | Regularization (prevent overfitting) |
| `Activation` | Activation function |
| `EarlyStopping` | Stop training if no improvement |
| `ReduceLROnPlateau` | Reduce learning rate on plateau |
| `ModelCheckpoint` | Save best model during training |
| `Adam` | Adaptive momentum optimizer |

---

## Block 2: Random Seeds (Lines 19-20)

```python
np.random.seed(42)
tf.random.set_seed(42)
```

**Why set random seeds?**
- Ensures reproducibility — same code produces same results every time
- NumPy generates same random arrays
- TensorFlow generates same weight initializations
- The number `42` is arbitrary (commonly used convention)

---

## Block 3: Haversine Distance Function (Lines 22-29)

```python
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c
```

**What it does:**
- Calculates distance between two GPS coordinates on Earth
- Uses the **haversine formula** — great circle distance

**Parameters:**
- `lat1, lon1` — Customer's location (degrees)
- `lat2, lon2` — Merchant's location (degrees)

**Returns:**
- Distance in kilometers

**Why this matters for fraud:**
- Fraudulent transactions often have large distances
- Legitimate transactions are usually close to home
- `distance_km` becomes a strong fraud indicator

---

## Block 4: Preprocessing Function (Lines 31-108)

```python
def preprocess_for_nn(df, encoders=None, scaler=None, fit=True):
    df = df.copy()
    
    # Feature engineering
    df['trans_datetime'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_datetime'].dt.hour
    df['day_of_week'] = df['trans_datetime'].dt.dayofweek
    df['month'] = df['trans_datetime'].dt.month
    df['day_of_month'] = df['trans_datetime'].dt.day
    
    df['distance_km'] = haversine_distance(
        df['lat'], df['long'],
        df['merch_lat'], df['merch_long']
    )
    
    df['dob_datetime'] = pd.to_datetime(df['dob'])
    df['age'] = (df['trans_datetime'] - df['dob_datetime']).dt.days // 365
    
    # Drop unnecessary columns
    drop_cols = [
        'merchant', 'job', 'first', 'last', 'street', 'city',
        'trans_num', 'cc_num', 'zip', 'unix_time',
        'trans_date_trans_time', 'trans_datetime', 'dob', 'dob_datetime',
        'Unnamed: 0'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Encode categorical variables
    categorical_cols = ['category', 'gender', 'state']
    
    if encoders is None:
        encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            if fit:
                encoders[col] = LabelEncoder()
                df[col] = encoders[col].fit_transform(df[col].astype(str))
            else:
                le = encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
    
    # Separate features and labels
    if 'is_fraud' in df.columns:
        y = df['is_fraud'].values
        df = df.drop(columns=['is_fraud'])
    else:
        y = None
    
    # Get numeric columns
    feature_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64', 'int32', 'float32']]
    X = df[feature_cols].values
    
    # Scale features
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    # Handle any NaN/inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, y, encoders, scaler, feature_cols
```

**Step-by-step breakdown:**

### Step 4a: Feature Engineering (Lines 35-53)
Creates new features from existing data:

| New Feature | Source | Why It Helps |
|-------------|--------|--------------|
| `hour` | Transaction time | Fraud often happens at unusual hours |
| `day_of_week` | Transaction time | Weekend fraud patterns |
| `month` | Transaction time | Seasonal fraud |
| `day_of_month` | Transaction time | Month-end patterns |
| `distance_km` | Customer vs merchant GPS | Long distance = suspicious |
| `age` | DOB vs transaction time | Age-related fraud patterns |

### Step 4b: Drop Columns (Lines 55-63)
Removes non-predictive columns:
- Unique identifiers (trans_num, cc_num)
- Free-text fields (merchant, job, street)
- Redundant columns (unix_time duplicates trans_date_trans_time)

### Step 4c: Label Encoding (Lines 65-81)
Converts categorical strings to numbers:

```
Before:  category = "grocery_pos", "gas_station", "entertainment"
After:   category = 0, 1, 2
```

- `fit=True`: Learn encoding from training data
- `fit=False`: Apply existing encoding to test data
- Handles unseen categories by assigning -1

### Step 4d: Separate Features/Labels (Lines 83-92)
- Extracts `is_fraud` as target variable `y`
- Remaining columns become features `X`

### Step 4e: Feature Scaling (Lines 94-101)
- `StandardScaler`: transforms to mean=0, std=1
- **CRITICAL**: Fit only on training data (prevents data leakage)
- Test data uses the same scaler fitted on training

### Step 4f: Handle NaN/Inf (Lines 103-105)
- Neural networks can't handle NaN or infinity
- Replaces with 0 (neutral value)

---

## Block 5: Build Model Function (Lines 110-133)

```python
def build_model(input_dim, 
                hidden_layers=[256, 128, 64],
                dropout_rate=0.3,
                learning_rate=0.001):
    model = Sequential()
    
    model.add(Input(shape=(input_dim,)))
    
    for i, neurons in enumerate(hidden_layers):
        model.add(Dense(neurons))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Activation('relu'))
    
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model
```

**What each layer does:**

### Input Layer
```python
model.add(Input(shape=(input_dim,)))
```
- Defines input shape (15 features for our data)
- `input_dim=15` means each sample has 15 numbers

### Hidden Layers (Loop)
```python
for i, neurons in enumerate(hidden_layers):
    model.add(Dense(neurons))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Activation('relu'))
```

| Layer | What It Does |
|-------|--------------|
| `Dense(neurons)` | Fully connected: every input connects to every neuron |
| `BatchNormalization` | Normalizes inputs to each neuron (stabilizes training) |
| `Dropout(0.3)` | Randomly "turns off" 30% of neurons each batch (prevents overfitting) |
| `Activation('relu')` | ReLU: `max(0, x)` — introduces non-linearity |

**Architecture used:** 256 → 128 → 64 neurons

### Output Layer
```python
model.add(Dense(1, activation='sigmoid'))
```
- 1 neuron for binary classification
- Sigmoid squashes output to 0-1 (probability)

### Compile
```python
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
```

| Parameter | Purpose |
|-----------|---------|
| `Adam` | Adaptive optimizer (good default choice) |
| `learning_rate=0.001` | Step size for weight updates |
| `binary_crossentropy` | Loss for binary classification |
| `AUC` | Track ROC-AUC during training |

---

## Block 6: Train Model Function (Lines 135-220)

```python
def train_model(X_train, y_train, 
                X_val=None, y_val=None,
                hidden_layers=[256, 128, 64],
                dropout_rate=0.3,
                learning_rate=0.001,
                batch_size=2048,
                epochs=50,
                class_weight=None,
                model_path='fraud_model.keras'):
```

### Step 6a: Handle Class Imbalance (Lines 141-154)

```python
if class_weight is None:
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    weight_pos = n_neg / n_pos
    class_weight = {0: 1.0, 1: weight_pos}
```

**Why class weights?**
- Fraud is rare (0.58% of data)
- Without weights, model predicts "not fraud" for everything
- Class weights make the model "pay more attention" to fraud cases

**Formula:**
```
weight_for_fraud = num_legitimate / num_fraud
```

For 172:1 imbalance → weight ≈ 172

### Step 6b: Build Model (Lines 156-163)

```python
model = build_model(
    input_dim=X_train.shape[1],
    hidden_layers=hidden_layers,
    dropout_rate=dropout_rate,
    learning_rate=learning_rate
)
```

Creates the neural network architecture.

### Step 6c: Setup Callbacks (Lines 165-192)

```python
has_validation = X_val is not None and y_val is not None
monitor_metric = 'val_loss' if has_validation else 'loss'

callbacks = [
    ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    EarlyStopping(
        monitor=monitor_metric,
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        model_path,
        monitor='val_auc' if has_validation else 'auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]
```

**Callbacks explained:**

| Callback | What It Does |
|----------|--------------|
| `ReduceLROnPlateau` | If no improvement for 3 epochs, cut learning rate in half |
| `EarlyStopping` | If no improvement for 5 epochs, stop training |
| `ModelCheckpoint` | Save the model with best AUC |

### Step 6d: Train (Lines 208-220)

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val) if has_validation else None,
    epochs=epochs,
    batch_size=batch_size,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)
```

**Key parameters:**

| Parameter | Purpose |
|-----------|---------|
| `epochs=50` | Max passes through data |
| `batch_size=2048` | Samples per weight update |
| `class_weight` | Handle imbalance |
| `verbose=1` | Show progress bar |

---

## Block 7: Evaluate Model Function (Lines 222-272)

```python
def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_pred_prob = model.predict(X_test, batch_size=4096, verbose=0).flatten()
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    pr_auc = average_precision_score(y_test, y_pred_prob)
    
    # Print results...
```

### Step 7a: Get Predictions (Lines 223-225)

```python
y_pred_prob = model.predict(X_test, batch_size=4096, verbose=0).flatten()
y_pred = (y_pred_prob >= threshold).astype(int)
```

- `model.predict()` returns probability (0 to 1)
- `.flatten()` converts from 2D to 1D array
- Compare against threshold (0.5 = 50%) to get binary prediction

### Step 7b: Calculate Metrics (Lines 227-235)

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| ROC-AUC | Area under ROC curve | Overall discrimination ability |
| PR-AUC | Area under PR curve | Precision-recall balance (better for imbalanced data) |
| Precision | TP / (TP + FP) | Of predicted fraud, how many are actual fraud |
| Recall | TP / (TP + FN) | Of actual fraud, how many did we catch? |
| F1 | 2 × P × R / (P + R) | Balance between precision and recall |

---

## Block 8: Predict Function (Lines 274-287)

```python
def predict(model, X, threshold=0.5):
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    probabilities = model.predict(X, batch_size=4096, verbose=0).flatten()
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions, probabilities
```

**Purpose:** Make predictions on new transactions

- Handles both single sample (1D) and batch (2D) input
- Returns both binary prediction and probability

---

## Block 9: Main Function (Lines 289-349)

```python
def main():
    print("="*60)
    print("CREDIT CARD FRAUD DETECTION - NEURAL NETWORK")
    print("="*60 + "\n")
    
    # Paths
    train_path = 'fraudTrain.csv'
    test_path = 'fraudTest.csv'
    model_path = 'fraud_model.keras'
    preprocessor_path = 'preprocessor.pkl'
    
    # Load data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Preprocess
    X_train, y_train, encoders, scaler, feature_cols = preprocess_for_nn(df_train, fit=True)
    X_test, y_test, _, _, _ = preprocess_for_nn(df_test, encoders=encoders, scaler=scaler, fit=False)
    
    # Save preprocessor
    joblib.dump({'encoders': encoders, 'scaler': scaler, 'feature_cols': feature_cols}, preprocessor_path)
    
    # Split for validation
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    
    # Train
    model, history = train_model(X_tr, y_tr, X_val, y_val, ...)
    
    # Save model
    model.save(model_path)
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test, threshold=0.5)
    
    return model, results
```

**Complete pipeline:**

1. **Load** training and test CSVs
2. **Preprocess** with feature engineering
3. **Save** preprocessor (for future predictions)
4. **Split** training into train/validation (90/10)
5. **Train** the model
6. **Save** the trained model
7. **Evaluate** on test set

---

## Summary: How the Blocks Connect

```
┌─────────────────────────────────────────────────────┐
│                    INPUT DATA                        │
│              fraudTrain.csv, fraudTest.csv          │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  BLOCK 4: preprocess_for_nn()                       │
│  • Feature engineering (hour, distance, age)        │
│  • Label encoding (category → numbers)              │
│  • Scaling (StandardScaler)                        │
│  → OUTPUT: X_train, y_train, encoders, scaler     │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  BLOCK 5: build_model()                             │
│  • Creates MLP architecture                        │
│  • 15 → 256 → 128 → 64 → 1                         │
│  → OUTPUT: model object (uncompiled)              │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  BLOCK 6: train_model()                             │
│  • Compiles model with Adam + binary_crossentropy  │
│  • Handles class imbalance with weights            │
│  • Uses callbacks (EarlyStopping, ReduceLROnPlateau)│
│  • Trains for up to 50 epochs                       │
│  → OUTPUT: trained model, history                  │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  BLOCK 7: evaluate_model()                          │
│  • Predicts on test set                            │
│  • Calculates ROC-AUC, PR-AUC, F1                  │
│  • Prints confusion matrix                         │
│  → OUTPUT: metrics dictionary                      │
└─────────────────────────────────────────────────────┘
```

---

## Running the Code

```bash
# Navigate to project (use your actual path)
cd [PROJECT_PATH]

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the script
python fraud_detection.py
```

**Expected output:**
- Loads ~1.3M training rows, ~555K test rows
- Trains for 30 epochs with early stopping
- Shows ROC-AUC, PR-AUC, confusion matrix
- Saves `fraud_model.keras` and `preprocessor.pkl`
