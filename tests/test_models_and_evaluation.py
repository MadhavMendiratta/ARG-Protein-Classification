"""
Module: test_models_and_evaluation.py
Objective: Automated pytest validation for Phase 6 (model training) and Phase 7 (evaluation & error analysis).
Validates:
- Trained model artifact existence, serialization, and metadata
- Real-time inference capability on feature matrices
- Output dimensions and mathematical bounds of predictions and probabilities
- Held-out test set evaluation metrics and confusion matrix sums
- Error analysis consistency (FP/FN partition constraints)
- Feature importance matrices and generated figures
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest

RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_SPLITS_DIR = Path("data/splits")


def test_saved_model_artifacts_exist_and_loadable():
    """Verify that all three trained model artifacts exist and load cleanly."""
    expected_models = [
        "logistic_regression_model.joblib",
        "random_forest_model.joblib",
        "svm_rbf_model.joblib"
    ]
    for model_fname in expected_models:
        model_path = MODELS_DIR / model_fname
        assert model_path.exists(), f"Model artifact {model_fname} missing!"
        data = joblib.load(model_path)
        assert isinstance(data, dict), f"Artifact {model_fname} is not a valid dict!"
        assert "best_estimator" in data
        assert "best_params" in data
        assert "cv_roc_auc" in data
        assert "train_time_sec" in data
        assert "val_metrics" in data
        assert data["cv_roc_auc"] > 0.5, f"{model_fname} CV ROC-AUC is not better than random!"


def test_validation_model_comparison_file():
    """Verify validation model comparison JSON exists and contains all 3 models."""
    val_json_path = MODELS_DIR / "validation_model_comparison.json"
    assert val_json_path.exists()
    with open(val_json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert "logistic_regression" in summary
    assert "random_forest" in summary
    assert "svm_rbf" in summary
    for k, v in summary.items():
        assert 0.0 <= v["roc_auc"] <= 1.0
        assert 0.0 <= v["pr_auc"] <= 1.0
        assert 0.0 <= v["f1_score"] <= 1.0
        assert -1.0 <= v["mcc"] <= 1.0


def test_model_inference_pipeline():
    """Verify models can run inference on test feature vectors without shape mismatches."""
    features = np.load(DATA_SPLITS_DIR / "features_scaled.npz")
    X_test = features["X_test"]
    assert X_test.shape == (1816, 428)
    
    # Test Random Forest prediction
    rf_data = joblib.load(MODELS_DIR / "random_forest_model.joblib")
    rf_estimator = rf_data["best_estimator"]
    rf_probs = rf_estimator.predict_proba(X_test)[:, 1]
    assert rf_probs.shape == (1816,)
    assert np.all((rf_probs >= 0.0) & (rf_probs <= 1.0))

    # Test Logistic Regression prediction
    lr_data = joblib.load(MODELS_DIR / "logistic_regression_model.joblib")
    lr_estimator = lr_data["best_estimator"]
    lr_probs = lr_estimator.predict_proba(X_test)[:, 1]
    assert lr_probs.shape == (1816,)
    assert np.all((lr_probs >= 0.0) & (lr_probs <= 1.0))


def test_test_predictions_file_integrity():
    """Verify test_predictions.csv matches exactly the 1,816 test samples with proper columns."""
    pred_path = METRICS_DIR / "test_predictions.csv"
    assert pred_path.exists()
    df_pred = pd.read_csv(pred_path)
    assert len(df_pred) == 1816
    expected_cols = [
        "seq_uid", "label", "cluster_id", "prob_Logistic Regression",
        "pred_Logistic Regression", "prob_Random Forest", "pred_Random Forest",
        "prob_SVM (RBF Kernel)", "pred_SVM (RBF Kernel)", "pred_best", "prob_best"
    ]
    for col in expected_cols:
        assert col in df_pred.columns, f"Missing column {col} in test_predictions.csv"
        
    # Check that labels are binary
    assert set(df_pred["label"].unique()) == {0, 1}
    assert (df_pred["label"] == 1).sum() == 908
    assert (df_pred["label"] == 0).sum() == 908


def test_model_comparison_benchmark():
    """Verify test benchmark CSV and ensure Random Forest achieves solid discriminative metrics."""
    comp_path = METRICS_DIR / "model_comparison.csv"
    assert comp_path.exists()
    df_comp = pd.read_csv(comp_path)
    assert len(df_comp) == 3
    assert set(df_comp["Model"]) == {"Logistic Regression", "Random Forest", "SVM (RBF Kernel)"}

    rf_row = df_comp[df_comp["Model"] == "Random Forest"].iloc[0]
    assert rf_row["ROC-AUC"] > 0.80, f"RF ROC-AUC {rf_row['ROC-AUC']} is lower than benchmark expectation!"
    assert rf_row["PR-AUC"] > 0.80, f"RF PR-AUC {rf_row['PR-AUC']} is lower than benchmark expectation!"
    assert rf_row["Accuracy"] > 0.70

    for _, row in df_comp.iterrows():
        assert 0.0 <= row["Accuracy"] <= 1.0
        assert 0.0 <= row["Precision"] <= 1.0
        assert 0.0 <= row["Recall (Sensitivity)"] <= 1.0
        assert 0.0 <= row["Specificity"] <= 1.0
        assert 0.0 <= row["F1-Score"] <= 1.0
        assert 0.0 <= row["ROC-AUC"] <= 1.0
        assert 0.0 <= row["PR-AUC"] <= 1.0
        assert -1.0 <= row["MCC"] <= 1.0
        
        # Verify confusion matrix sums to 1816
        cm_total = row["TP"] + row["TN"] + row["FP"] + row["FN"]
        assert cm_total == 1816, f"Confusion matrix for {row['Model']} does not sum to 1816 (got {cm_total})"
        assert row["TP"] + row["FN"] == 908, "True positives + false negatives must equal positive class count (908)"
        assert row["TN"] + row["FP"] == 908, "True negatives + false positives must equal negative class count (908)"


def test_error_analysis_tables():
    """Verify false positive and false negative tables correctly partition prediction errors."""
    fp_path = METRICS_DIR / "false_positives.csv"
    fn_path = METRICS_DIR / "false_negatives.csv"
    assert fp_path.exists()
    assert fn_path.exists()
    
    df_fp = pd.read_csv(fp_path)
    df_fn = pd.read_csv(fn_path)
    
    # False positives: true label = 0, pred = 1
    assert (df_fp["label"] == 0).all()
    assert (df_fp["pred_best"] == 1).all()
    
    # False negatives: true label = 1, pred = 0
    assert (df_fn["label"] == 1).all()
    assert (df_fn["pred_best"] == 0).all()
    
    # Verify exact counts matching Random Forest confusion matrix (FP=129, FN=380)
    assert len(df_fp) == 129
    assert len(df_fn) == 380


def test_feature_importance_tables():
    """Verify feature importances and coefficients cover all 428 features."""
    lr_coef_path = METRICS_DIR / "feature_coefficients_logreg.csv"
    rf_imp_path = METRICS_DIR / "feature_importances_rf.csv"
    assert lr_coef_path.exists()
    assert rf_imp_path.exists()
    
    df_lr = pd.read_csv(lr_coef_path)
    df_rf = pd.read_csv(rf_imp_path)
    
    assert len(df_lr) == 428
    assert len(df_rf) == 428
    
    # RF importances should sum approximately to 1.0
    assert pytest.approx(df_rf["gini_importance"].sum(), abs=1e-4) == 1.0


def test_generated_figures_exist():
    """Verify all 5 evaluation and error analysis figures exist and are non-empty."""
    expected_figures = [
        "roc_curves.png",
        "pr_curves.png",
        "confusion_matrices.png",
        "feature_importance_top25.png",
        "sequence_length_distribution_predictions.png"
    ]
    for fig_name in expected_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Figure {fig_name} missing!"
        assert fig_path.stat().st_size > 10000, f"Figure {fig_name} is unexpectedly small or empty!"
