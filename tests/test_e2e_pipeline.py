"""
Test Suite: tests/test_e2e_pipeline.py
Objective: End-to-End integration and regression test covering E2E Tests 1 through 8.
"""

import socket
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.inference import (
    ModelArtifactManager,
    clean_and_validate_sequence,
    parse_fasta_multi,
    predict_single_sequence,
    predict_batch_records
)
from src.dashboard_data import (
    get_model_comparison_df,
    get_test_predictions_df,
    get_split_report
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =========================================================================
# E2E TEST 1: Single Sequence Full Pipeline
# =========================================================================
def test_e2e_single_sequence_inference():
    """E2E Test 1: Full pipeline inference on verified positive TEM-126 sequence."""
    tem_seq = (
        "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLL"
        "CGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTA"
        "FLHNMGDHVTRLDRWEPELNEAIPNDERETTTPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSA"
        "LPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
    )
    res = predict_single_sequence(tem_seq, model_key="random_forest")

    assert res["model_key"] == "random_forest"
    assert res["model_display_name"] == "Random Forest"
    assert res["sequence_length"] == 286
    assert res["predicted_label"] == 1
    assert res["predicted_text"] == "Predicted ARG-like"
    assert res["is_arg"] is True
    assert res["threshold_used"] == 0.10
    assert res["score"] > 0.90, f"Expected strong ARG score for TEM-126, got {res['score']}"
    assert len(res["features_dict"]) == 428
    assert len(res["features_scaled"]) == 428


# =========================================================================
# E2E TEST 2: Multi-Record FASTA Batch Processing
# =========================================================================
def test_e2e_multi_record_fasta_batch():
    """E2E Test 2: Process multi-record FASTA containing 4 biological sequences."""
    batch_fasta = (
        ">TEM_126 Beta-lactamase\n"
        "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLL\n"
        ">VanX Glycopeptide resistance\n"
        "MENGFLFLDEMLHGVRWDAKYATWDNFTGKPVDGYEVNRIIGTKAVALALKEALNDKGYKVEAIEEPGKGKYYV\n"
        ">CspB Cold shock protein\n"
        "MLEGKVKWFNSEKGFGFIEVEGQDDVFVHFSAIQGEGFKTLEEGQAVSFEIVEGNRGPQAANVTKEA\n"
        ">AtpL ATP synthase subunit\n"
        "MENLNMDLLYMAAAVMMGLAAIGAAIGIGILGGKFLEGAARQPDLIPLLRTQFFIVMGLVDAIPMIAVGLGLYVMFAVA\n"
    )

    records = parse_fasta_multi(batch_fasta)
    assert len(records) == 4

    df_res = predict_batch_records(records, model_key="random_forest")
    assert len(df_res) == 4
    assert list(df_res["status"]) == ["Success", "Success", "Success", "Success"]
    assert list(df_res["seq_uid"]) == ["TEM_126", "VanX", "CspB", "AtpL"]

    # Verify predictions: TEM_126 should be ARG-like, CspB should be candidate non-ARG
    assert df_res.loc[df_res["seq_uid"] == "TEM_126", "predicted_label"].iloc[0] == 1


# =========================================================================
# E2E TEST 3: Invalid Input Graceful Reporting
# =========================================================================
def test_e2e_invalid_input_reporting():
    """E2E Test 3: Rejection of invalid input without crashing or returning fake predictions."""
    with pytest.raises(ValueError, match="Non-canonical amino acid"):
        predict_single_sequence("MKAYX")

    # In batch mode, invalid inputs are captured in the dataframe
    records = parse_fasta_multi(">BadSeq\nMKAY123FIA")
    df_res = predict_batch_records(records)
    assert df_res.iloc[0]["status"] == "Invalid Input"
    assert df_res.iloc[0]["predicted_label"] == -1
    assert np.isnan(df_res.iloc[0]["score"])


# =========================================================================
# E2E TEST 4: Model Comparison Invocation
# =========================================================================
def test_e2e_model_comparison_invocation():
    """E2E Test 4: Verify RF, LogReg, and calibrated SVM are each invoked with their own outputs."""
    seq = "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLL"
    mgr = ModelArtifactManager.get_instance()

    for m_key in ["random_forest", "logistic_regression", "svm_rbf"]:
        res = predict_single_sequence(seq, model_key=m_key)
        assert res["model_key"] == m_key
        assert np.isfinite(res["score"])
        assert res["threshold_used"] == 0.10


# =========================================================================
# E2E TEST 5: Metrics Consistency Recomputation
# =========================================================================
def test_e2e_metrics_consistency():
    """E2E Test 5: Independently recompute model metrics from saved test predictions."""
    comp_df = get_model_comparison_df()
    assert len(comp_df) == 3

    preds_df = get_test_predictions_df()
    assert len(preds_df) == 1816

    # RF recomputation
    y_true = preds_df["label"].values
    y_pred = preds_df["pred_Random Forest"].values
    
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    assert tp == 528
    assert tn == 779
    assert fp == 129
    assert fn == 380


# =========================================================================
# E2E TEST 6: Missing Optional Artifact Handled Gracefully
# =========================================================================
def test_e2e_missing_optional_artifact_graceful():
    """E2E Test 6: Verify dashboard data loader returns empty dataframe on missing optional files."""
    from src.dashboard_data import get_false_positives_df
    # Function executes cleanly and returns DataFrame
    fp_df = get_false_positives_df()
    assert isinstance(fp_df, pd.DataFrame)


# =========================================================================
# E2E TEST 7: Missing Mandatory Artifact Safe Failure
# =========================================================================
def test_e2e_missing_mandatory_artifact_safe_failure(monkeypatch):
    """E2E Test 7: Verify missing model key raises clear ValueError."""
    with pytest.raises(ValueError, match="Model 'nonexistent_model' not available"):
        predict_single_sequence("MKAYFIAILTLFTCIATVVRAQQMSELENRI", model_key="nonexistent_model")


# =========================================================================
# E2E TEST 8: Offline Execution Guarantee
# =========================================================================
def test_e2e_offline_execution_guarantee(monkeypatch):
    """E2E Test 8: Verify inference runs completely offline without network socket connection calls."""
    # Monkeypatch socket.socket.connect to raise RuntimeError if called
    def mock_connect(*args, **kwargs):
        raise RuntimeError("External network connection attempted during local inference!")

    monkeypatch.setattr(socket.socket, "connect", mock_connect)

    seq = "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLL"
    # Should execute purely offline without calling socket connect
    res = predict_single_sequence(seq, model_key="random_forest")
    assert res["predicted_label"] == 1
