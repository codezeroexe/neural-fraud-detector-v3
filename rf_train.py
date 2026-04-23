from sklearn.ensemble import RandomForestClassifier
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

print("[*] Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("[*] Saving model...")
joblib.dump(model, "rf_model.pkl")
print("[✓] Random Forest model saved to rf_model.pkl")