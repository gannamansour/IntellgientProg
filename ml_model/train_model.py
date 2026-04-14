"""
ml_model/train_model.py
Trains a Decision Tree Classifier on the heart disease dataset,
evaluates it, and saves the model with joblib.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)
import joblib


DATA_PATH  = "data/cleaned_data.csv"
MODEL_PATH = "ml_model/decision_tree_model.pkl"


def train_and_evaluate():
    # ── 1. Load cleaned data ─────────────────────────────────────────────
    df     = pd.read_csv(DATA_PATH)
    X      = df.drop(columns=["target"])
    y      = df["target"]

    # ── 2. Train / test split ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 3. Hyperparameter tuning (grid search) ───────────────────────────
    param_grid = {
        "max_depth":        [3, 5, 7, None],
        "min_samples_split": [2, 5, 10],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"[✓] Best params: {grid.best_params_}")

    # ── 4. Evaluate ──────────────────────────────────────────────────────
    y_pred = best_model.predict(X_test)
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
    }
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["No Disease", "Disease"]))

    # ── 5. Save model ────────────────────────────────────────────────────
    os.makedirs("ml_model", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"[✓] Model saved → {MODEL_PATH}")

    return best_model, metrics, X.columns.tolist()


if __name__ == "__main__":
    train_and_evaluate()
