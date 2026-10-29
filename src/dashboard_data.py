"""
Module: src/dashboard_data.py
Objective: Centralized data access and metrics caching for the ARG Classification Dashboard.
Reads genuine project artifacts without mock data or synthetic outputs.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SPLITS_DIR = DATA_DIR / "splits"
PROCESSED_DIR = DATA_DIR / "processed"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"


def get_validation_summary() -> Dict[str, Any]:
    """Load data preprocessing and raw exclusion summary."""
    path = PROCESSED_DIR / "validation_summary.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_split_report() -> Dict[str, Any]:
    """Load clustering, graph superclustering, and split validation report."""
    path = SPLITS_DIR / "split_validation_report.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_summary() -> Dict[str, Any]:
    """Load feature extraction breakdown and dataset matrix shapes."""
    path = SPLITS_DIR / "feature_summary.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_comparison_df() -> pd.DataFrame:
    """Load verified benchmark metrics comparing Logistic Regression, Random Forest, and SVM."""
    path = METRICS_DIR / "model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_test_predictions_df() -> pd.DataFrame:
    """Load genuine test set predictions and scores across all models."""
    path = METRICS_DIR / "test_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_dataset_master_df(limit: Optional[int] = None) -> pd.DataFrame:
    """Load the master cleaned dataset metadata."""
    path = PROCESSED_DIR / "dataset_master.csv"
    if not path.exists():
        return pd.DataFrame()
    if limit is not None:
        return pd.read_csv(path, nrows=limit)
    return pd.read_csv(path)


def get_rf_feature_importances() -> pd.DataFrame:
    """Load Random Forest Gini impurity feature importances."""
    path = METRICS_DIR / "feature_importances_rf.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_logreg_coefficients() -> pd.DataFrame:
    """Load Logistic Regression feature coefficients."""
    path = METRICS_DIR / "feature_coefficients_logreg.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_false_positives_df() -> pd.DataFrame:
    """Load false positive test set error analysis."""
    path = METRICS_DIR / "false_positives.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_false_negatives_df() -> pd.DataFrame:
    """Load false negative test set error analysis."""
    path = METRICS_DIR / "false_negatives.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_exclusion_log_df() -> pd.DataFrame:
    """Load data cleaning exclusion log."""
    path = PROCESSED_DIR / "exclusion_log.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_sequence_cluster_assignments_df() -> pd.DataFrame:
    """Load full sequence cluster and split assignments (lightweight metadata)."""
    path = SPLITS_DIR / "sequence_cluster_assignments.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
