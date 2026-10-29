"""
Test Suite: tests/test_app_inference.py
Objective: Comprehensive automated testing of the ARG Classification Dashboard
covering Tests A through G as specified in the research requirements.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from Bio import SeqIO
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, matthews_corrcoef

from src.inference import (
    ModelArtifactManager,
    clean_and_validate_sequence,
    parse_fasta_multi,
    predict_single_sequence,
    predict_batch_records
)
from src.dashboard_data import (
    get_validation_summary,
    get_split_report,
    get_feature_summary,
    get_model_comparison_df,
    get_test_predictions_df,
    get_exclusion_log_df
)
from src.feature_extraction import extract_features_for_sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# =========================================================================
# TEST A: Artifact Availability & Model Loading
# =========================================================================
def test_artifact_availability_and_loading():
    """Verify all required trained models, scaler, feature metadata, and reports exist and load."""
    assert (DATA_SPLITS_DIR / "feature_names.json").exists(), "feature_names.json missing"
    assert (DATA_SPLITS_DIR / "scaler.joblib").exists(), "scaler.joblib missing"
    assert (MODELS_DIR / "random_forest_model.joblib").exists(), "Random Forest model missing"
    assert (MODELS_DIR / "logistic_regression_model.joblib").exists(), "Logistic Regression model missing"
    assert (MODELS_DIR / "svm_rbf_model.joblib").exists(), "SVM RBF model missing"

    mgr = ModelArtifactManager.get_instance()
    assert len(mgr.feature_names) == 428, f"Expected 428 feature names, got {len(mgr.feature_names)}"
    assert mgr.scaler is not None, "Scaler object failed to load"
    assert "random_forest" in mgr.models, "Random Forest missing from loaded models"
    assert "logistic_regression" in mgr.models, "Logistic Regression missing from loaded models"
    assert "svm_rbf" in mgr.models, "SVM RBF missing from loaded models"

    # Verify model thresholds and calibration
    for k, m_info in mgr.models.items():
        assert m_info["model"] is not None, f"Model estimator for {k} is None"
        assert m_info["threshold"] == 0.10, f"Expected threshold 0.10 for {k}, got {m_info['threshold']}"
        if k == "svm_rbf":
            assert m_info["calibrator"] is not None, "SVM Platt calibrator is missing"


# =========================================================================
# TEST B: Feature Parity with Original Training Pipeline
# =========================================================================
def test_feature_parity_with_training_pipeline():
    """Verify that feature extraction produces exact mathematical parity with saved test matrices."""
    # Load first test sequence from dataset_clean.fasta
    fasta_path = PROCESSED_DIR / "dataset_clean.fasta"
    fasta_dict = {r.id: str(r.seq) for r in SeqIO.parse(fasta_path, "fasta")}
    
    # Load test indices to get exact sequence and matrix row
    test_indices_df = pd.read_csv(DATA_SPLITS_DIR / "test_indices.csv")
    first_seq_uid = test_indices_df.iloc[0]["seq_uid"]
    seq_str = fasta_dict[first_seq_uid]

    # 1. Extract features using runtime inference extractor
    feat_dict = extract_features_for_sequence(seq_str)
    mgr = ModelArtifactManager.get_instance()
    feat_vec = np.array([feat_dict[k] for k in mgr.feature_names], dtype=np.float64)

    # 2. Compare against saved raw feature matrix row
    raw_npz = np.load(DATA_SPLITS_DIR / "features_raw.npz")
    X_test_saved = raw_npz["X_test"][0]

    np.testing.assert_allclose(
        feat_vec,
        X_test_saved,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Extracted feature vector diverges from original training feature matrix!"
    )

    # 3. Compare scaled vector
    scaled_vec = mgr.scaler.transform(feat_vec.reshape(1, -1))[0]
    scaled_npz = np.load(DATA_SPLITS_DIR / "features_scaled.npz")
    X_test_scaled_saved = scaled_npz["X_test"][0]

    np.testing.assert_allclose(
        scaled_vec,
        X_test_scaled_saved,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Scaled feature vector diverges from original frozen training scaler!"
    )


# =========================================================================
# TEST C: Sequence Validation Rules
# =========================================================================
def test_sequence_validation_rules():
    """Verify input validation for valid, lowercase, whitespace, FASTA header, and invalid inputs."""
    valid_seq = "MKAYFIAILTLFTCIATVVRAQQMSELENRIDSLLNGKKATVGIAVWTDK"
    
    # Standard valid
    is_v, clean_s, s_id, msg = clean_and_validate_sequence(valid_seq, default_id="Test_Seq")
    assert is_v is True
    assert clean_s == valid_seq
    assert s_id == "Test_Seq"

    # Lowercase normalization
    is_v, clean_s, _, _ = clean_and_validate_sequence(valid_seq.lower())
    assert is_v is True
    assert clean_s == valid_seq

    # Whitespace and newline handling
    spaced_seq = f"  {valid_seq[:20]} \n {valid_seq[20:]} \t  "
    is_v, clean_s, _, _ = clean_and_validate_sequence(spaced_seq)
    assert is_v is True
    assert clean_s == valid_seq

    # FASTA formatted with header
    fasta_text = f">CustomHeader_001 Beta-lactamase protein\n{valid_seq}\n"
    is_v, clean_s, s_id, _ = clean_and_validate_sequence(fasta_text)
    assert is_v is True
    assert clean_s == valid_seq
    assert s_id == "CustomHeader_001"

    # Empty input
    is_v, _, _, msg = clean_and_validate_sequence("")
    assert is_v is False
    assert "empty" in msg.lower()

    # Non-canonical amino acids (X, B, Z, U)
    invalid_seq = "MKAYFIAXTLFTCZ"
    is_v, _, _, msg = clean_and_validate_sequence(invalid_seq)
    assert is_v is False
    assert "Non-canonical amino acid" in msg

    # Sequence length too short (<2 for DPC)
    is_v, _, _, msg = clean_and_validate_sequence("M")
    assert is_v is False
    assert "too short" in msg.lower()

    # Length advisory notice (<50 aa)
    short_seq = "MKAYFIAILTLFTCIATVVRAQQMSELENRI"  # 31 aa
    is_v, clean_s, _, msg = clean_and_validate_sequence(short_seq)
    assert is_v is True
    assert "Notice: Sequence length 31 aa is shorter" in msg


# =========================================================================
# TEST D: Inference Pipeline & Threshold Application Across All Models
# =========================================================================
def test_inference_pipeline_execution_and_thresholds():
    """Verify live inference, score ranges, finite values, and threshold application for all models."""
    test_seq = "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLL"
    mgr = ModelArtifactManager.get_instance()

    for model_key in ["random_forest", "logistic_regression", "svm_rbf"]:
        pred = predict_single_sequence(test_seq, model_key=model_key)
        
        assert pred["model_key"] == model_key
        assert pred["sequence_length"] == len(test_seq)
        assert np.isfinite(pred["score"])
        assert 0.0 <= pred["score"] <= 1.0, f"Score out of probability bounds [0, 1]: {pred['score']}"
        assert pred["threshold_used"] == 0.10
        assert pred["predicted_label"] in [0, 1]
        
        if pred["score"] >= 0.10:
            assert pred["predicted_label"] == 1
            assert pred["predicted_text"] == "Predicted ARG-like"
            assert pred["is_arg"] is True
        else:
            assert pred["predicted_label"] == 0
            assert pred["predicted_text"] == "Predicted candidate non-ARG-like"
            assert pred["is_arg"] is False

        assert len(pred["features_dict"]) == 428
        assert len(pred["features_scaled"]) == 428


# =========================================================================
# TEST E: Metrics Consistency & Confusion Matrix Totals
# =========================================================================
def test_metrics_consistency_and_confusion_matrix():
    """Recompute held-out test metrics from saved predictions and confirm exact match with model_comparison.csv."""
    test_preds_df = get_test_predictions_df()
    assert len(test_preds_df) == 1816, f"Expected 1,816 test rows, got {len(test_preds_df)}"
    
    comp_df = get_model_comparison_df()
    assert len(comp_df) == 3, f"Expected 3 models in comparison, got {len(comp_df)}"

    rf_comp = comp_df[comp_df["Model"] == "Random Forest"].iloc[0]
    y_true = test_preds_df["label"].values
    y_pred_rf = test_preds_df["pred_Random Forest"].values

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_rf).ravel()
    assert tp == int(rf_comp["TP"]) == 528
    assert tn == int(rf_comp["TN"]) == 779
    assert fp == int(rf_comp["FP"]) == 129
    assert fn == int(rf_comp["FN"]) == 380

    acc = accuracy_score(y_true, y_pred_rf)
    prec = precision_score(y_true, y_pred_rf)
    rec = recall_score(y_true, y_pred_rf)
    mcc = matthews_corrcoef(y_true, y_pred_rf)

    assert pytest.approx(acc, abs=1e-4) == float(rf_comp["Accuracy"])
    assert pytest.approx(prec, abs=1e-4) == float(rf_comp["Precision"])
    assert pytest.approx(rec, abs=1e-4) == float(rf_comp["Recall (Sensitivity)"])
    assert pytest.approx(mcc, abs=1e-4) == float(rf_comp["MCC"])


# =========================================================================
# TEST F: FASTA Batch Processing & Record ID Integrity
# =========================================================================
def test_fasta_batch_processing_and_record_integrity():
    """Verify parsing multi-record FASTA, mixed valid/invalid records, and DataFrame output format."""
    multi_fasta = (
        ">SeqA_Valid Beta-lactamase sample\n"
        "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLL\n"
        ">SeqB_Invalid Contaminated sequence with noncanonical\n"
        "MKAYFIAXTLFTCIATVVRAQQMSELENRI\n"
        ">SeqC_Valid Housekeeping protein sample\n"
        "MLEGKVKWFNSEKGFGFIEVEGQDDVFVHFSAIQGEGFKTLEEGQAVSFEIVEGNRGPQAANVTKEA\n"
    )

    records = parse_fasta_multi(multi_fasta)
    assert len(records) == 3
    assert records[0]["seq_id"] == "SeqA_Valid"
    assert records[0]["is_valid"] is True
    assert records[1]["seq_id"] == "SeqB_Invalid"
    assert records[1]["is_valid"] is False
    assert records[2]["seq_id"] == "SeqC_Valid"
    assert records[2]["is_valid"] is True

    # Run batch inference
    batch_df = predict_batch_records(records, model_key="random_forest")
    assert len(batch_df) == 3
    assert list(batch_df["seq_uid"]) == ["SeqA_Valid", "SeqB_Invalid", "SeqC_Valid"]
    assert batch_df.iloc[0]["status"] == "Success"
    assert batch_df.iloc[1]["status"] == "Invalid Input"
    assert batch_df.iloc[2]["status"] == "Success"

    # Confirm expected columns
    expected_cols = [
        "seq_uid", "description", "length", "status", "prediction",
        "predicted_label", "score", "score_type", "threshold", "model_used", "notes"
    ]
    for col in expected_cols:
        assert col in batch_df.columns, f"Missing expected batch column: {col}"


# =========================================================================
# TEST G: App Module Imports & Page Rendering Smoke Test
# =========================================================================
def test_app_modules_import_and_page_rendering():
    """Verify all dashboard pages and application entry points import without errors."""
    import app
    from app_pages.page_overview import render_overview
    from app_pages.page_predictor import render_predictor
    from app_pages.page_comparison import render_comparison
    from app_pages.page_visualizations import render_visualizations
    from app_pages.page_features import render_features
    from app_pages.page_dataset import render_dataset
    from app_pages.page_methodology import render_methodology
    from app_pages.page_about import render_about

    for func in [
        render_overview, render_predictor, render_comparison,
        render_visualizations, render_features, render_dataset,
        render_methodology, render_about
    ]:
        assert callable(func), f"Page render function {func.__name__} is not callable"

    # Test genuine inference on TEM-126 sample
    tem_seq = (
        "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLL"
        "CGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTA"
        "FLHNMGDHVTRLDRWEPELNEAIPNDERETTTPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSA"
        "LPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
    )
    pred_tem = predict_single_sequence(tem_seq, model_key="random_forest")
    assert pred_tem["predicted_label"] == 1
    assert pred_tem["predicted_text"] == "Predicted ARG-like"
    assert pred_tem["score"] >= 0.10
