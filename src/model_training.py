"""
model_training.py — FailSafe AI Model Training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trains 3 models (Decision Tree, Random Forest, XGBoost) for binary
failure prediction, compares them, and saves the best performer.
Also trains a multi-class failure type classifier.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_config, pickle_dump, pickle_load


def load_binary_data(config: dict):
    """Load the preprocessed binary classification data."""
    d = config["train_test"]["directory"]
    X_train = pickle_load(d + config["train_test"]["X_train"])
    X_test  = pickle_load(d + config["train_test"]["X_test"])
    y_train = pickle_load(d + config["train_test"]["y_train"])
    y_test  = pickle_load(d + config["train_test"]["y_test"])
    return X_train, X_test, y_train, y_test


def load_failure_type_data(config: dict):
    """Load the preprocessed failure type classification data."""
    d = config["train_test"]["directory"]
    X_train = pickle_load(d + config["train_test"]["X_train_ft"])
    X_test  = pickle_load(d + config["train_test"]["X_test_ft"])
    y_train = pickle_load(d + config["train_test"]["y_train_ft"])
    y_test  = pickle_load(d + config["train_test"]["y_test_ft"])
    return X_train, X_test, y_train, y_test


def train_binary_models(X_train, X_test, y_train, y_test, config: dict) -> dict:
    """Train and evaluate Decision Tree, Random Forest, and XGBoost."""
    models_cfg = config["models"]
    results = {}

    # -- 1. Decision Tree --
    print("--- Training Decision Tree ---")
    dt = DecisionTreeClassifier(**models_cfg["decision_tree"]["params"])
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_prob = dt.predict_proba(X_test)[:, 1]

    dt_acc = accuracy_score(y_test, dt_pred)
    dt_f1 = f1_score(y_test, dt_pred)
    dt_auc = roc_auc_score(y_test, dt_prob)
    dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_prob)

    print(f"  Accuracy: {dt_acc:.4f} | F1: {dt_f1:.4f} | AUC: {dt_auc:.4f}")
    print(classification_report(y_test, dt_pred, target_names=["Normal", "Failure"]))

    results["Decision Tree"] = {
        "model": dt,
        "accuracy": dt_acc,
        "f1_score": dt_f1,
        "auc_roc": dt_auc,
        "fpr": dt_fpr.tolist(),
        "tpr": dt_tpr.tolist(),
        "y_pred": dt_pred,
        "y_prob": dt_prob,
        "confusion_matrix": confusion_matrix(y_test, dt_pred),
        "report": classification_report(y_test, dt_pred, target_names=["Normal", "Failure"], output_dict=True),
        "feature_importance": dt.feature_importances_,
    }

    # -- 2. Random Forest --
    print("--- Training Random Forest ---")
    rf = RandomForestClassifier(**models_cfg["random_forest"]["params"])
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_prob)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

    print(f"  Accuracy: {rf_acc:.4f} | F1: {rf_f1:.4f} | AUC: {rf_auc:.4f}")
    print(classification_report(y_test, rf_pred, target_names=["Normal", "Failure"]))

    results["Random Forest"] = {
        "model": rf,
        "accuracy": rf_acc,
        "f1_score": rf_f1,
        "auc_roc": rf_auc,
        "fpr": rf_fpr.tolist(),
        "tpr": rf_tpr.tolist(),
        "y_pred": rf_pred,
        "y_prob": rf_prob,
        "confusion_matrix": confusion_matrix(y_test, rf_pred),
        "report": classification_report(y_test, rf_pred, target_names=["Normal", "Failure"], output_dict=True),
        "feature_importance": rf.feature_importances_,
    }

    # -- 3. XGBoost --
    print("--- Training XGBoost ---")
    xgb = XGBClassifier(**models_cfg["xgboost"]["params"])
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_prob = xgb.predict_proba(X_test)[:, 1]

    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred)
    xgb_auc = roc_auc_score(y_test, xgb_prob)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_prob)

    print(f"  Accuracy: {xgb_acc:.4f} | F1: {xgb_f1:.4f} | AUC: {xgb_auc:.4f}")
    print(classification_report(y_test, xgb_pred, target_names=["Normal", "Failure"]))

    results["XGBoost"] = {
        "model": xgb,
        "accuracy": xgb_acc,
        "f1_score": xgb_f1,
        "auc_roc": xgb_auc,
        "fpr": xgb_fpr.tolist(),
        "tpr": xgb_tpr.tolist(),
        "y_pred": xgb_pred,
        "y_prob": xgb_prob,
        "confusion_matrix": confusion_matrix(y_test, xgb_pred),
        "report": classification_report(y_test, xgb_pred, target_names=["Normal", "Failure"], output_dict=True),
        "feature_importance": xgb.feature_importances_,
    }

    return results


def train_failure_type_model(X_train, X_test, y_train, y_test, config: dict) -> dict:
    """Train a Random Forest for multi-class failure type classification."""
    print("\n--- Training Failure Type Classifier (Random Forest) ---")

    # Encode string labels to integers
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train_enc)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test_enc, y_pred)
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

    result = {
        "model": rf,
        "label_encoder": le,
        "accuracy": acc,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_test_enc, y_pred),
        "report": classification_report(
            y_test_enc, y_pred, target_names=le.classes_, output_dict=True
        ),
        "feature_importance": rf.feature_importances_,
        "classes": le.classes_.tolist(),
    }

    return result


def run_training():
    """Execute the full training pipeline."""
    config = load_config()
    models_dir = config["models"]["directory"]

    # -- Binary classification --
    print("=" * 60)
    print("  BINARY FAILURE PREDICTION")
    print("=" * 60)
    X_train, X_test, y_train, y_test = load_binary_data(config)

    # Get feature names for display
    feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else config["dataset"]["all_features"]

    results = train_binary_models(X_train, X_test, y_train, y_test, config)

    # Select best model by F1 score (better for imbalanced data)
    best_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model = results[best_name]["model"]
    print(f"\n[BEST] Best model: {best_name} (F1={results[best_name]['f1_score']:.4f})")

    # Save all models
    for name, res in results.items():
        model_key = name.lower().replace(" ", "_")
        pickle_dump(res["model"], models_dir + config["models"][model_key]["name"])

    # Save best model separately
    pickle_dump(best_model, models_dir + config["models"]["best_model"])

    # Save metrics (without the model objects -- for the dashboard)
    metrics_to_save = {}
    for name, res in results.items():
        metrics_to_save[name] = {k: v for k, v in res.items() if k != "model"}
    metrics_to_save["best_model_name"] = best_name
    metrics_to_save["feature_names"] = feature_names
    metrics_to_save["y_test"] = y_test
    pickle_dump(metrics_to_save, models_dir + config["models"]["metrics"])

    # -- Failure type classification --
    print("\n" + "=" * 60)
    print("  FAILURE TYPE CLASSIFICATION")
    print("=" * 60)
    X_train_ft, X_test_ft, y_train_ft, y_test_ft = load_failure_type_data(config)

    ft_result = train_failure_type_model(X_train_ft, X_test_ft, y_train_ft, y_test_ft, config)

    pickle_dump(ft_result["model"], models_dir + config["models"]["failure_type_model"])

    # Save failure type metrics
    ft_metrics = {k: v for k, v in ft_result.items() if k != "model"}
    ft_metrics["feature_names"] = feature_names
    ft_metrics["y_test"] = y_test_ft
    pickle_dump(ft_metrics, models_dir + config["models"]["failure_type_metrics"])

    print("\n[OK] All models trained and saved to models/")


if __name__ == "__main__":
    run_training()
