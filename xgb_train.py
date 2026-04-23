import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fraud_detection import preprocess_for_nn

print("[*] Loading training data...")
df_train = pd.read_csv("fraudTrain.csv")

print("[*] Preprocessing...")
X, y, encoders, scaler, feature_cols = preprocess_for_nn(df_train, fit=True)

print("[*] Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

print(f"[*] Train: {X_train.shape[0]:,} samples, Fraud: {y_train.sum():,}")
print(f"[*] Val: {X_val.shape[0]:,} samples, Fraud: {y_val.sum():,}")

scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
print(f"[*] Scale pos weight: {scale_pos_weight:.2f}")

print("[*] Training XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=25)

print("[*] Saving model...")
joblib.dump(model, "xgb_model.pkl")
print("[✓] XGBoost model saved to xgb_model.pkl")