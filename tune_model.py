import numpy as np
import pandas as pd
import joblib
import os
import random
from itertools import product
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Input,
    BatchNormalization,
    Dropout,
    Activation,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)


def build_model(input_dim, hidden_layers, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for neurons in hidden_layers:
        model.add(Dense(neurons))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Activation("relu"))

    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model


def train_and_evaluate(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_layers,
    dropout_rate,
    learning_rate,
    batch_size,
    epochs,
):
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    weight_pos = n_neg / n_pos
    class_weight = {0: 1.0, 1: weight_pos}

    model = build_model(
        input_dim=X_train.shape[1],
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=0
        ),
        EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=0
        ),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )

    y_pred_prob = model.predict(X_val, batch_size=4096, verbose=0).flatten()
    roc_auc = roc_auc_score(y_val, y_pred_prob)
    pr_auc = average_precision_score(y_val, y_pred_prob)

    return model, roc_auc, pr_auc


def random_search(X_train, y_train, X_val, y_val, n_trials=20):
    param_space = {
        "hidden_layers": [
            [256, 128, 64],
            [512, 256, 128],
            [128, 64, 32],
            [256, 128],
            [512, 256, 128, 64],
        ],
        "dropout_rate": [0.2, 0.3, 0.4, 0.5],
        "learning_rate": [0.0001, 0.0005, 0.001, 0.005],
        "batch_size": [1024, 2048, 4096],
        "epochs": [20, 30, 40],
    }

    results = []
    best_score = 0
    best_params = None

    print(f"\nStarting Random Search with {n_trials} trials...\n")

    for trial in range(n_trials):
        params = {
            "hidden_layers": random.choice(param_space["hidden_layers"]),
            "dropout_rate": random.choice(param_space["dropout_rate"]),
            "learning_rate": random.choice(param_space["learning_rate"]),
            "batch_size": random.choice(param_space["batch_size"]),
            "epochs": random.choice(param_space["epochs"]),
        }

        print(f"Trial {trial + 1}/{n_trials}: {params}")

        model, roc_auc, pr_auc = train_and_evaluate(
            X_train, y_train, X_val, y_val, **params
        )

        score = (roc_auc + pr_auc) / 2

        print(f"  ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, Score: {score:.4f}")

        results.append(
            {
                "trial": trial + 1,
                "params": params,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "score": score,
            }
        )

        if score > best_score:
            best_score = score
            best_params = params
            print(f"  *** New best! ***")

    print("\n" + "=" * 60)
    print("RANDOM SEARCH RESULTS")
    print("=" * 60)
    print(f"\nBest Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\nBest Score: {best_score:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("hyperparam_results.csv", index=False)
    print(f"\nSaved results to hyperparam_results.csv")

    return best_params, results


def main():
    print("=" * 60)
    print("HYPERPARAMETER TUNING - RANDOM SEARCH")
    print("=" * 60 + "\n")

    train_path = "fraudTrain.csv"
    preprocessor_path = "preprocessor.pkl"

    print("Loading data...")
    df_train = pd.read_csv(train_path)

    from fraud_detection import preprocess_for_nn

    print("Preprocessing...")
    X_train_full, y_train_full, encoders, scaler, feature_cols = preprocess_for_nn(
        df_train, fit=True
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.1,
        random_state=42,
        stratify=y_train_full,
    )

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")

    best_params, results = random_search(X_train, y_train, X_val, y_val, n_trials=50)

    joblib.dump(best_params, "best_hyperparams.pkl")
    print(f"\nSaved best params to best_hyperparams.pkl")


if __name__ == "__main__":
    main()
