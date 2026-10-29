"""
Module: models.py
Objective: Train, tune, and validate baseline ML classifiers:
1. Regularized Logistic Regression (L2)
2. Random Forest Classifier
3. Support Vector Machine (RBF kernel)
Features:
- Group-aware cross-validation (GroupKFold by cluster_id) on training data only
- Hyperparameter tuning without test set contamination
- Validation set evaluation and threshold calibration
- Model serialization with metadata (runtime, parameters, software versions)
"""

import sys
import time
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef
)

DATA_SPLITS_DIR = Path("data/splits")
RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"


def load_partitioned_data():
    """Load scaled feature matrices, labels, sequence UIDs, and cluster groups."""
    data = np.load(DATA_SPLITS_DIR / "features_scaled.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    
    # Load cluster IDs for group-aware cross-validation
    train_df = pd.read_csv(DATA_SPLITS_DIR / "train_indices.csv")
    val_df = pd.read_csv(DATA_SPLITS_DIR / "val_indices.csv")
    test_df = pd.read_csv(DATA_SPLITS_DIR / "test_indices.csv")
    
    with open(DATA_SPLITS_DIR / "feature_names.json", "r") as f:
        feature_names = json.load(f)
        
    return {
        "X_train": X_train, "y_train": y_train, "train_df": train_df,
        "X_val": X_val, "y_val": y_val, "val_df": val_df,
        "X_test": X_test, "y_test": y_test, "test_df": test_df,
        "feature_names": feature_names
    }


def compute_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive classification performance metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred))
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Find classification threshold maximizing F1-score on validation data."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_f1 = 0.0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
    return float(best_thresh), float(best_f1)


def train_logistic_regression(X_train, y_train, train_groups) -> Dict[str, Any]:
    """Train regularized Logistic Regression with GroupKFold tuning."""
    print("\n" + "=" * 60)
    print("TRAINING MODEL 1: REGULARIZED LOGISTIC REGRESSION (L2)")
    print("=" * 60)
    t0 = time.time()
    
    param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
    base_model = LogisticRegression(max_iter=1500, random_state=42)
    
    cv = GroupKFold(n_splits=5)
    grid = GridSearchCV(base_model, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train, groups=train_groups)
    
    train_time = time.time() - t0
    best_model = grid.best_estimator_
    
    print(f"  Best C parameter:       {grid.best_params_['C']}")
    print(f"  Mean 5-fold CV ROC-AUC: {grid.best_score_:.4f}")
    print(f"  Training Runtime:       {train_time:.2f} seconds")
    
    return {
        "model_name": "Logistic Regression",
        "best_estimator": best_model,
        "best_params": grid.best_params_,
        "cv_roc_auc": float(grid.best_score_),
        "train_time_sec": float(train_time)
    }


def train_random_forest(X_train, y_train, train_groups) -> Dict[str, Any]:
    """Train Random Forest classifier with GroupKFold tuning."""
    print("\n" + "=" * 60)
    print("TRAINING MODEL 2: RANDOM FOREST CLASSIFIER")
    print("=" * 60)
    t0 = time.time()
    
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [15, 25, None],
        "min_samples_split": [2, 5]
    }
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    cv = GroupKFold(n_splits=5)
    grid = GridSearchCV(base_model, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train, groups=train_groups)
    
    train_time = time.time() - t0
    best_model = grid.best_estimator_
    
    print(f"  Best Parameters:        {grid.best_params_}")
    print(f"  Mean 5-fold CV ROC-AUC: {grid.best_score_:.4f}")
    print(f"  Training Runtime:       {train_time:.2f} seconds")
    
    return {
        "model_name": "Random Forest",
        "best_estimator": best_model,
        "best_params": grid.best_params_,
        "cv_roc_auc": float(grid.best_score_),
        "train_time_sec": float(train_time)
    }


def train_svm_rbf(X_train, y_train, train_groups) -> Dict[str, Any]:
    """Train Support Vector Machine (RBF kernel) with GroupKFold tuning."""
    print("\n" + "=" * 60)
    print("TRAINING MODEL 3: SUPPORT VECTOR MACHINE (RBF KERNEL)")
    print("=" * 60)
    t0 = time.time()
    
    param_grid = {
        "C": [0.5, 1.0, 5.0],
        "gamma": ["scale", "auto"]
    }
    base_model = SVC(kernel="rbf", random_state=42)
    
    cv = GroupKFold(n_splits=5)
    grid = GridSearchCV(base_model, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train, groups=train_groups)
    
    train_time = time.time() - t0
    best_model = grid.best_estimator_
    
    print(f"  Best Parameters:        {grid.best_params_}")
    print(f"  Mean 5-fold CV ROC-AUC: {grid.best_score_:.4f}")
    print(f"  Training Runtime:       {train_time:.2f} seconds")
    
    return {
        "model_name": "SVM (RBF Kernel)",
        "best_estimator": best_model,
        "best_params": grid.best_params_,
        "cv_roc_auc": float(grid.best_score_),
        "train_time_sec": float(train_time)
    }


def run_training_pipeline() -> Dict[str, Any]:
    """Execute complete model training and validation pipeline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    data = load_partitioned_data()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    train_groups = data["train_df"]["cluster_id"].values
    
    print("=" * 70)
    print("PHASE 6: MODEL TRAINING & GROUP-AWARE CROSS-VALIDATION")
    print(f"Training Data: {X_train.shape[0]} samples across {len(np.unique(train_groups))} clusters")
    print(f"Validation Data: {X_val.shape[0]} samples")
    print("=" * 70)
    
    models_to_train = [
        ("logistic_regression", train_logistic_regression),
        ("random_forest", train_random_forest),
        ("svm_rbf", train_svm_rbf)
    ]
    
    trained_results = {}
    val_evaluation_summary = {}
    
    for key, train_fn in models_to_train:
        result = train_fn(X_train, y_train, train_groups)
        model = result["best_estimator"]
        
        # Predict on Validation set for threshold tuning and calibration
        if hasattr(model, "predict_proba"):
            val_probs = model.predict_proba(X_val)[:, 1]
        else:
            # Calibrate SVM continuous decision function via 1D Platt Scaling
            train_scores = model.decision_function(X_train).reshape(-1, 1)
            val_scores = model.decision_function(X_val).reshape(-1, 1)
            calibrator = LogisticRegression(random_state=42)
            calibrator.fit(train_scores, y_train)
            val_probs = calibrator.predict_proba(val_scores)[:, 1]
            result["calibrator"] = calibrator
            
        opt_thresh, opt_f1 = find_optimal_threshold(y_val, val_probs)
        val_preds = (val_probs >= opt_thresh).astype(int)
        
        val_metrics = compute_metrics_dict(y_val, val_preds, val_probs)
        val_metrics["optimal_threshold"] = opt_thresh
        val_metrics["val_f1_at_optimal_threshold"] = opt_f1
        
        result["val_metrics"] = val_metrics
        result["val_probs"] = val_probs
        result["val_preds"] = val_preds
        
        # Save model artifact
        model_save_path = MODELS_DIR / f"{key}_model.joblib"
        joblib.dump(result, model_save_path)
        print(f"[Saved] {result['model_name']} saved to {model_save_path}")
        print(f"  Validation ROC-AUC:    {val_metrics['roc_auc']:.4f}")
        print(f"  Validation PR-AUC:     {val_metrics['pr_auc']:.4f}")
        print(f"  Validation F1-score:   {val_metrics['f1_score']:.4f} (at threshold {opt_thresh:.2f})")
        print(f"  Validation MCC:        {val_metrics['mcc']:.4f}")
        
        trained_results[key] = result
        val_evaluation_summary[key] = {
            "model_name": result["model_name"],
            "cv_roc_auc": result["cv_roc_auc"],
            "best_params": result["best_params"],
            "train_time_sec": result["train_time_sec"],
            **val_metrics
        }
        
    summary_save_path = MODELS_DIR / "validation_model_comparison.json"
    with open(summary_save_path, "w", encoding="utf-8") as f:
        json.dump(val_evaluation_summary, f, indent=2)
        
    print("\n" + "=" * 70)
    print("PHASE 6 TRAINING & VALIDATION SELECTION COMPLETE")
    print(f"Summary saved to: {summary_save_path}")
    print("=" * 70)
    return trained_results


if __name__ == "__main__":
    run_training_pipeline()
