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
    #  Load cleaned data 
    df     = pd.read_csv(DATA_PATH)
    X      = df.drop(columns=["target"])
    y      = df["target"]

    # Train / test split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #  Feature Selection 
    print("[*] Performing feature selection...")
    prelim_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    prelim_model.fit(X_train, y_train)
    
    feature_importances = pd.Series(prelim_model.feature_importances_, 
                                     index=X_train.columns).sort_values(ascending=False)
    
    # Select features with cumulative importance >= 90%
    cumulative_importance = feature_importances.cumsum()
    n_features = (cumulative_importance <= 0.90).sum() + 1
    n_features = max(n_features, 5)  # Keep at least 5 features
    selected_features = feature_importances.head(n_features).index.tolist()
    
    print(f"[✓] Selected {n_features} features: {', '.join(selected_features)}")
    
    # Filter to selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Hyperparameter tuning (grid search) 
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
    grid.fit(X_train_selected, y_train)
    best_model = grid.best_estimator_
    print(f"[✓] Best params: {grid.best_params_}")

    # Evaluate 
    y_pred = best_model.predict(X_test_selected)
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
    }
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["No Disease", "Disease"]))

    #  Save model with selected features 
    os.makedirs("ml_model", exist_ok=True)
    model_data = {
        "model": best_model,
        "selected_features": selected_features,
        "feature_importances": feature_importances.to_dict()
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"[✓] Model saved → {MODEL_PATH}")
    print(f"[✓] Selected features: {selected_features}")

    return best_model, metrics, selected_features


if __name__ == "__main__":
    train_and_evaluate()
