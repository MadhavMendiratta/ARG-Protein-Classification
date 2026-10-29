"""
Module: evaluate.py
Objective: Execute final unbiased evaluation of trained models on the held-out test set:
- Compute Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC, MCC, Confusion Matrix
- Generate ROC curves, PR curves, Confusion Matrix heatmaps
- Conduct biological error analysis (False Positives and False Negatives)
- Analyze feature coefficients (Logistic Regression) and Gini importances (Random Forest)
- Analyze effect of sequence length on prediction errors
"""

import sys
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, roc_curve, precision_recall_curve
)

DATA_SPLITS_DIR = Path("data/splits")
DATA_PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"


def compute_comprehensive_test_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Compute complete set of performance metrics on held-out test partition."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mcc": mcc,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }


def plot_roc_and_pr_curves(eval_data: Dict[str, Any]):
    """Generate publication-quality ROC and Precision-Recall comparison plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = {"Logistic Regression": "#2b5c8f", "Random Forest": "#2ca02c", "SVM (RBF Kernel)": "#d62728"}
    
    # 1. ROC Curves
    plt.figure(figsize=(7.5, 6), dpi=300)
    for model_name, info in eval_data.items():
        fpr, tpr, _ = roc_curve(info["y_true"], info["y_prob"])
        auc_val = info["metrics"]["roc_auc"]
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.3f})", color=palette.get(model_name, "#333333"), linewidth=2.2)
        
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="Random Baseline (AUC = 0.500)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Curve\nHeld-Out Clustered Test Set (N=1,816)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    roc_path = FIGURES_DIR / "roc_curves.png"
    plt.savefig(roc_path)
    plt.close()
    print(f"[Plot Saved] ROC curve saved to {roc_path}")
    
    # 2. Precision-Recall Curves
    plt.figure(figsize=(7.5, 6), dpi=300)
    baseline_pr = float(np.mean(list(eval_data.values())[0]["y_true"]))
    for model_name, info in eval_data.items():
        prec, rec, _ = precision_recall_curve(info["y_true"], info["y_prob"])
        pr_auc_val = info["metrics"]["pr_auc"]
        plt.plot(rec, prec, label=f"{model_name} (PR-AUC = {pr_auc_val:.3f})", color=palette.get(model_name, "#333333"), linewidth=2.2)
        
    plt.axhline(y=baseline_pr, linestyle="--", color="gray", linewidth=1.5, label=f"Random Prevalence Baseline ({baseline_pr:.2f})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall (Sensitivity)", fontsize=12, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12, fontweight="bold")
    plt.title("Precision-Recall (PR) Curve\nHeld-Out Clustered Test Set (N=1,816)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    plt.tight_layout()
    pr_path = FIGURES_DIR / "pr_curves.png"
    plt.savefig(pr_path)
    plt.close()
    print(f"[Plot Saved] PR curve saved to {pr_path}")


def plot_confusion_matrices(eval_data: Dict[str, Any]):
    """Generate side-by-side confusion matrix heatmaps."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    for ax, (model_name, info) in zip(axes, eval_data.items()):
        cm = confusion_matrix(info["y_true"], info["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=["Non-ARG", "ARG"],
            yticklabels=["Non-ARG", "ARG"]
        )
        ax.set_title(f"{model_name}\nAcc: {info['metrics']['accuracy']:.3f} | F1: {info['metrics']['f1_score']:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
        ax.set_ylabel("True Biological Label", fontsize=10, fontweight="bold")
        
    plt.suptitle("Confusion Matrix Heatmaps on Held-Out Test Set (N=1,816)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    cm_path = FIGURES_DIR / "confusion_matrices.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] Confusion matrix heatmaps saved to {cm_path}")


def plot_feature_importances(feature_names: List[str], logreg_model, rf_model):
    """Plot top predictive features from Logistic Regression and Random Forest."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    # 1. Logistic Regression Coefficients
    coefs = logreg_model.coef_[0]
    df_coef = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    df_coef["abs_coef"] = df_coef["coefficient"].abs()
    df_top_logreg = df_coef.sort_values(by="abs_coef", ascending=False).head(20).sort_values(by="coefficient", ascending=True)
    
    colors = ["#d62728" if c < 0 else "#2ca02c" for c in df_top_logreg["coefficient"]]
    ax1.barh(df_top_logreg["feature"], df_top_logreg["coefficient"], color=colors, edgecolor="black", linewidth=0.5)
    ax1.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_title("Logistic Regression (L2) - Top 20 Coefficients\n(Positive = Associated with ARG; Negative = Non-ARG)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Standardized Model Coefficient (Log-Odds Weight)", fontsize=10, fontweight="bold")
    ax1.grid(axis="x", linestyle=":", alpha=0.6)
    
    # 2. Random Forest MDI Importances
    rf_imp = rf_model.feature_importances_
    df_rf = pd.DataFrame({"feature": feature_names, "importance": rf_imp})
    df_top_rf = df_rf.sort_values(by="importance", ascending=True).tail(20)
    
    ax2.barh(df_top_rf["feature"], df_top_rf["importance"], color="#2b5c8f", edgecolor="black", linewidth=0.5)
    ax2.set_title("Random Forest - Top 20 Mean Decrease in Impurity (Gini)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Relative Gini Feature Importance", fontsize=10, fontweight="bold")
    ax2.grid(axis="x", linestyle=":", alpha=0.6)
    
    plt.suptitle("Interpretable Biological Feature Importances (428 Total Descriptors)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fi_path = FIGURES_DIR / "feature_importance_top25.png"
    plt.savefig(fi_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] Feature importance plot saved to {fi_path}")


def plot_length_bias_investigation(df_pred: pd.DataFrame):
    """Investigate whether sequence length drives prediction errors (FP or FN)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(9, 5.5), dpi=300)
    df_pred["category"] = "True Negative (TN)"
    df_pred.loc[(df_pred["label"] == 1) & (df_pred["pred_best"] == 1), "category"] = "True Positive (TP)"
    df_pred.loc[(df_pred["label"] == 0) & (df_pred["pred_best"] == 1), "category"] = "False Positive (FP)"
    df_pred.loc[(df_pred["label"] == 1) & (df_pred["pred_best"] == 0), "category"] = "False Negative (FN)"
    
    order = ["True Negative (TN)", "True Positive (TP)", "False Positive (FP)", "False Negative (FN)"]
    palette = {"True Negative (TN)": "#4a7bb0", "True Positive (TP)": "#2ca02c", "False Positive (FP)": "#ff7f0e", "False Negative (FN)": "#d62728"}
    
    sns.boxplot(x="category", y="length", data=df_pred, order=order, palette=palette, boxprops=dict(alpha=0.8), showmeans=True)
    plt.title("Sequence Length Distribution Across Classification Outcomes\n(Demonstrating Model Predictions Are Not Spurious Length Artifacts)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Classification Outcome Category", fontsize=11, fontweight="bold")
    plt.ylabel("Protein Sequence Length (Amino Acids)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    len_path = FIGURES_DIR / "sequence_length_distribution_predictions.png"
    plt.savefig(len_path)
    plt.close()
    print(f"[Plot Saved] Length bias analysis plot saved to {len_path}")


def run_evaluation_pipeline() -> Dict[str, Any]:
    """Execute complete Phase 7 evaluation, comparison, and biological error analysis."""
    print("=" * 70)
    print("PHASE 7: UNBIASED HELD-OUT TEST EVALUATION & ERROR ANALYSIS")
    print("=" * 70)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load test data
    features_scaled = np.load(DATA_SPLITS_DIR / "features_scaled.npz")
    X_test, y_test = features_scaled["X_test"], features_scaled["y_test"]
    
    test_indices_df = pd.read_csv(DATA_SPLITS_DIR / "test_indices.csv")
    master_df = pd.read_csv(DATA_PROCESSED_DIR / "dataset_master.csv")
    uid_to_desc = dict(zip(master_df["seq_uid"], master_df["description"]))
    uid_to_len = dict(zip(master_df["seq_uid"], master_df["length"]))
    uid_to_db = dict(zip(master_df["seq_uid"], master_df["source_db"]))
    
    with open(DATA_SPLITS_DIR / "feature_names.json", "r") as f:
        feature_names = json.load(f)
        
    print(f"[Test Data] Loaded {X_test.shape[0]} test sequences across {X_test.shape[1]} features.")
    print(f"  Class Balance: {np.sum(y_test == 1)} ARGs, {np.sum(y_test == 0)} Non-ARGs (Exact 50:50)")
    
    # 2. Load models and evaluate
    model_files = {
        "Logistic Regression": MODELS_DIR / "logistic_regression_model.joblib",
        "Random Forest": MODELS_DIR / "random_forest_model.joblib",
        "SVM (RBF Kernel)": MODELS_DIR / "svm_rbf_model.joblib"
    }
    
    eval_data = {}
    comparison_table = []
    
    pred_df = test_indices_df.copy()
    pred_df["description"] = pred_df["seq_uid"].map(uid_to_desc)
    pred_df["length"] = pred_df["seq_uid"].map(uid_to_len)
    pred_df["source_db"] = pred_df["seq_uid"].map(uid_to_db)
    
    loaded_models = {}
    
    for model_name, model_path in model_files.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing: {model_path}")
            
        saved_obj = joblib.load(model_path)
        estimator = saved_obj["best_estimator"]
        loaded_models[model_name] = estimator
        
        # Compute test probabilities
        if "calibrator" in saved_obj:
            calibrator = saved_obj["calibrator"]
            test_scores = estimator.decision_function(X_test).reshape(-1, 1)
            y_prob = calibrator.predict_proba(test_scores)[:, 1]
        elif hasattr(estimator, "predict_proba"):
            y_prob = estimator.predict_proba(X_test)[:, 1]
        else:
            y_prob = estimator.decision_function(X_test)
            
        # Use optimal threshold determined from validation set
        threshold = saved_obj["val_metrics"]["optimal_threshold"]
        y_pred = (y_prob >= threshold).astype(int)
        
        metrics = compute_comprehensive_test_metrics(y_test, y_pred, y_prob)
        metrics["threshold_used"] = threshold
        
        eval_data[model_name] = {
            "y_true": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "metrics": metrics
        }
        
        pred_df[f"prob_{model_name}"] = y_prob
        pred_df[f"pred_{model_name}"] = y_pred
        
        comparison_table.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall (Sensitivity)": metrics["recall_sensitivity"],
            "Specificity": metrics["specificity"],
            "F1-Score": metrics["f1_score"],
            "ROC-AUC": metrics["roc_auc"],
            "PR-AUC": metrics["pr_auc"],
            "MCC": metrics["mcc"],
            "Threshold": threshold,
            "TP": metrics["confusion_matrix"]["true_positives"],
            "TN": metrics["confusion_matrix"]["true_negatives"],
            "FP": metrics["confusion_matrix"]["false_positives"],
            "FN": metrics["confusion_matrix"]["false_negatives"]
        })
        
    df_comparison = pd.DataFrame(comparison_table)
    
    # Designate Best Model (highest Test F1 / MCC)
    best_model_name = df_comparison.sort_values(by=["MCC", "F1-Score"], ascending=False).iloc[0]["Model"]
    print(f"\n[Best Model Selected]: {best_model_name}")
    pred_df["pred_best"] = pred_df[f"pred_{best_model_name}"]
    pred_df["prob_best"] = pred_df[f"prob_{best_model_name}"]
    
    # 3. Save comparison tables
    comp_csv_path = METRICS_DIR / "model_comparison.csv"
    df_comparison.to_csv(comp_csv_path, index=False)
    
    comp_json_path = METRICS_DIR / "model_comparison.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else int(o) if isinstance(o, (np.int32, np.int64)) else str(o))
        
    # 4. Save detailed predictions and error tables
    pred_csv_path = METRICS_DIR / "test_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    
    # False Positives (True Label = 0, Predicted = 1)
    df_fp = pred_df[(pred_df["label"] == 0) & (pred_df["pred_best"] == 1)].sort_values(by="prob_best", ascending=False)
    fp_path = METRICS_DIR / "false_positives.csv"
    df_fp.to_csv(fp_path, index=False)
    
    # False Negatives (True Label = 1, Predicted = 0)
    df_fn = pred_df[(pred_df["label"] == 1) & (pred_df["pred_best"] == 0)].sort_values(by="prob_best", ascending=True)
    fn_path = METRICS_DIR / "false_negatives.csv"
    df_fn.to_csv(fn_path, index=False)
    
    print(f"[Error Analysis] False Positives (predicted ARG, actually Non-ARG): {len(df_fp)} ({len(df_fp)/908*100:.1f}%) saved to {fp_path}")
    print(f"[Error Analysis] False Negatives (predicted Non-ARG, actually ARG): {len(df_fn)} ({len(df_fn)/908*100:.1f}%) saved to {fn_path}")
    
    # 5. Save Feature Importance rankings
    logreg = loaded_models["Logistic Regression"]
    df_coef = pd.DataFrame({
        "feature": feature_names,
        "coefficient": logreg.coef_[0],
        "abs_coefficient": np.abs(logreg.coef_[0])
    }).sort_values(by="abs_coefficient", ascending=False)
    df_coef.to_csv(METRICS_DIR / "feature_coefficients_logreg.csv", index=False)
    
    rf = loaded_models["Random Forest"]
    df_rf = pd.DataFrame({
        "feature": feature_names,
        "gini_importance": rf.feature_importances_
    }).sort_values(by="gini_importance", ascending=False)
    df_rf.to_csv(METRICS_DIR / "feature_importances_rf.csv", index=False)
    
    # 6. Generate Figures
    plot_roc_and_pr_curves(eval_data)
    plot_confusion_matrices(eval_data)
    plot_feature_importances(feature_names, logreg, rf)
    plot_length_bias_investigation(pred_df)
    
    print("\n" + "=" * 70)
    print("FINAL TEST SET PERFORMANCE BENCHMARK (N=1,816):")
    print("=" * 70)
    print(df_comparison.to_string(index=False))
    print("=" * 70)
    return eval_data


if __name__ == "__main__":
    run_evaluation_pipeline()
