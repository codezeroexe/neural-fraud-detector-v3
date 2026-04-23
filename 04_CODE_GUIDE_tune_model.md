# Code Guide: tune_model.py
## Block-by-Block Explanation

---

## Overview

`tune_model.py` performs **hyperparameter tuning** using **Random Search**. Instead of testing every possible combination (which would take forever), it randomly samples from a predefined search space.

---

## Block 1: Imports (Lines 1-16)

```python
import numpy as np
import pandas as pd
import joblib
import os
import random
from itertools import product
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
```

**Key additions vs fraud_detection.py:**
- `random` — Randomly sample hyperparameters
- `product` — For grid search (not used in random search)
- `roc_auc_score`, `average_precision_score` — Evaluate each trial

---

## Block 2: Build Model (Lines 18-34)

```python
def build_model(input_dim, hidden_layers, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    
    for neurons in hidden_layers:
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

**Same as fraud_detection.py** — builds MLP architecture.

---

## Block 3: Train and Evaluate Single Trial (Lines 36-62)

```python
def train_and_evaluate(X_train, y_train, X_val, y_val, 
                       hidden_layers, dropout_rate, learning_rate, 
                       batch_size, epochs):
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    weight_pos = n_neg / n_pos
    class_weight = {0: 1.0, 1: weight_pos}
    
    model = build_model(
        input_dim=X_train.shape[1],
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    ]
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0
    )
    
    y_pred_prob = model.predict(X_val, batch_size=4096, verbose=0).flatten()
    roc_auc = roc_auc_score(y_val, y_pred_prob)
    pr_auc = average_precision_score(y_val, y_pred_prob)
    
    return model, roc_auc, pr_auc
```

**What this does:**

1. **Calculate class weights** (same as before)
2. **Build model** with given hyperparameters
3. **Train** with early stopping (verbose=0 to reduce output)
4. **Evaluate** on validation set
5. **Return** the model and both AUC scores

**Why return the model?**
- Could save each trial's model
- For now, we just extract metrics and discard the model

---

## Block 4: Random Search (Lines 64-118)

```python
def random_search(X_train, y_train, X_val, y_val, n_trials=20):
    param_space = {
        'hidden_layers': [
            [256, 128, 64],
            [512, 256, 128],
            [128, 64, 32],
            [256, 128],
            [512, 256, 128, 64]
        ],
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
        'batch_size': [1024, 2048, 4096],
        'epochs': [20, 30, 40]
    }
    
    results = []
    best_score = 0
    best_params = None
    
    print(f"\nStarting Random Search with {n_trials} trials...\n")
    
    for trial in range(n_trials):
        params = {
            'hidden_layers': random.choice(param_space['hidden_layers']),
            'dropout_rate': random.choice(param_space['dropout_rate']),
            'learning_rate': random.choice(param_space['learning_rate']),
            'batch_size': random.choice(param_space['batch_size']),
            'epochs': random.choice(param_space['epochs'])
        }
        
        print(f"Trial {trial+1}/{n_trials}: {params}")
        
        model, roc_auc, pr_auc = train_and_evaluate(
            X_train, y_train, X_val, y_val,
            **params
        )
        
        score = (roc_auc + pr_auc) / 2
        
        print(f"  ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, Score: {score:.4f}")
        
        results.append({
            'trial': trial + 1,
            'params': params,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_params = params
            print(f"  *** New best! ***")
    
    # Save results...
```

### Parameter Space Explained

| Parameter | Options | What It Controls |
|-----------|---------|------------------|
| `hidden_layers` | 5 architectures | Network depth/width |
| `dropout_rate` | 0.2, 0.3, 0.4, 0.5 | Regularization strength |
| `learning_rate` | 0.0001, 0.0005, 0.001, 0.005 | Step size for optimization |
| `batch_size` | 1024, 2048, 4096 | Samples per update |
| `epochs` | 20, 30, 40 | Max training passes |

### Total Combinations
- 5 × 4 × 4 × 3 × 3 = **720 possible combinations**
- Random search with 15-20 trials samples enough to find good hyperparameters

### Scoring
```python
score = (roc_auc + pr_auc) / 2
```
- Averages ROC-AUC and PR-AUC
- PR-AUC is more important for imbalanced data

---

## Block 5: Main Function (Lines 120-145)

```python
def main():
    print("="*60)
    print("HYPERPARAMETER TUNING - RANDOM SEARCH")
    print("="*60 + "\n")
    
    train_path = 'fraudTrain.csv'
    preprocessor_path = 'preprocessor.pkl'
    
    print("Loading data...")
    df_train = pd.read_csv(train_path)
    
    from fraud_detection import preprocess_for_nn
    
    print("Preprocessing...")
    X_train_full, y_train_full, encoders, scaler, feature_cols = preprocess_for_nn(
        df_train, fit=True
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.1, random_state=42, stratify=y_train_full
    )
    
    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
    
    best_params, results = random_search(
        X_train, y_train, X_val, y_val, n_trials=15
    )
    
    joblib.dump(best_params, 'best_hyperparams.pkl')
    print(f"\nSaved best params to best_hyperparams.pkl")
```

**Pipeline:**

1. Load training data
2. Preprocess (using fraud_detection.py function)
3. Split 90% train, 10% validation
4. Run random search with 15 trials
5. Save best parameters

---

## Hyperparameters Explained

### Hidden Layers
| Architecture | Best For |
|-------------|----------|
| [256, 128, 64] | Default, good balance |
| [512, 256, 128] | More capacity, may overfit |
| [128, 64, 32] | Simpler, faster |
| [256, 128] | Fewer layers |
| [512, 256, 128, 64] | Deeper network |

### Dropout Rate
| Rate | Effect |
|------|--------|
| 0.2 | Light regularization |
| 0.3 | Default, good balance |
| 0.4 | Strong regularization |
| 0.5 | Very strong (may underfit) |

### Learning Rate
| Rate | Effect |
|------|--------|
| 0.0001 | Slow, may not converge in time |
| 0.0005 | Moderate |
| 0.001 | Default, good for Adam |
| 0.005 | Fast, may overshoot |

### Batch Size
| Size | Effect |
|------|--------|
| 1024 | Slower updates, more noise |
| 2048 | Default, good balance |
| 4096 | Faster, may need more epochs |

---

## Running the Script

```bash
cd [PROJECT_PATH]

# Activate environment
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

python tune_model.py
```
```

**Expected output:**
```
============================================================
HYPERPARAMETER TUNING - RANDOM SEARCH
============================================================

Starting Random Search with 15 trials...

Trial 1/15: {'hidden_layers': [256, 128, 64], 'dropout_rate': 0.3, ...}
  ROC-AUC: 0.8542, PR-AUC: 0.3210, Score: 0.5876
  *** New best! ***
Trial 2/15: {'hidden_layers': [512, 256, 128], 'dropout_rate': 0.4, ...}
  ROC-AUC: 0.8612, PR-AUC: 0.3345, Score: 0.5979
  *** New best! ***
...
============================================================
RANDOM SEARCH RESULTS
============================================================

Best Parameters:
  hidden_layers: [512, 256, 128]
  dropout_rate: 0.4
  learning_rate: 0.001
  batch_size: 2048
  epochs: 30

Best Score: 0.6123

Saved results to hyperparam_results.csv
Saved best params to best_hyperparams.pkl
```

---

## Output Files

| File | Contents |
|------|----------|
| `hyperparam_results.csv` | All trials with params and scores |
| `best_hyperparams.pkl` | Best parameters for re-use |

---

## Why Random Search?

**vs Grid Search:**
- Grid: Test all 720 combinations (too slow)
- Random: Sample 15-20, find good params faster

**vs Bayesian Optimization:**
- Simpler to implement
- Good enough for this problem size
- No extra dependencies
