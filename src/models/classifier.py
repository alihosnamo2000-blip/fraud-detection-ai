# src/models/classifier.py

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from src.utils.metrics import compute_metrics, compute_auc

def train_model(X_train, y_train):
    """
    Train a Random Forest classifier.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
    Returns:
        model: Trained Random Forest model.
    """
    model = RandomForestClassifier(
        n_estimators=100,

        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using custom metrics functions.
    Args:
        model: Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels.
    """
    # التنبؤ بالفئة
    y_pred = model.predict(X_test)

    # التنبؤ بالاحتمالات (لـ AUC)
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]

    else:
        y_scores = y_pred  # fallback إذا لم تتوفر الاحتمالات

    # حساب المقاييس
    metrics = compute_metrics(y_test, y_pred)
    auc = compute_auc(y_test, y_scores)

    # عرض النتائج
    print("\n📊 Evaluation Metrics:")
    for key, value in metrics.items():
        if key == "Confusion Matrix":
            print(f"{key}:\n{value}")
        else:
            print(f"{key}: {value:.4f}")
    print(f"AUC Score: {auc:.4f}")

def save_model(model, path='models/random_forest_model.joblib'):
    """
    Save the trained model to disk.

    Args:
        model: Trained model.
        path (str): File path to save the model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(model, path)
