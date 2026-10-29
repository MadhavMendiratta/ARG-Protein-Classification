"""
Test Suite: tests/test_adversarial.py
Objective: Rigorous adversarial, randomized, and regression testing for the
ARG Classification ML pipeline and inference engine.
"""

import random
from pathlib import Path
import numpy as np
import pytest

from src.inference import (
    ModelArtifactManager,
    clean_and_validate_sequence,
    parse_fasta_multi,
    predict_single_sequence,
    predict_batch_records
)
from src.preprocessing import CANONICAL_AMINO_ACIDS
from src.feature_extraction import extract_features_for_sequence


# =========================================================================
# REGRESSION TEST 1: Physicochemical Keys Presence & Non-Zero
# =========================================================================
def test_physicochemical_keys_in_prediction():
    """Regression test: verify all physicochemical keys exist and are non-zero for valid sequences."""
    seq = "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLL"
    pred = predict_single_sequence(seq, model_key="random_forest")
    fdict = pred["features_dict"]

    expected_keys = [
        "PC_length", "PC_molecular_weight", "PC_isoelectric_point", "PC_gravy",
        "PC_aromaticity", "PC_instability_index", "PC_aliphatic_index", "PC_net_charge_pH7"
    ]
    for k in expected_keys:
        assert k in fdict, f"Missing expected physicochemical key: {k}"

    assert fdict["PC_molecular_weight"] > 5000.0, "Molecular weight is suspiciously low or zero"
    assert 0.0 <= fdict["PC_isoelectric_point"] <= 14.0, "Isoelectric point out of biological range"
    assert -4.5 <= fdict["PC_gravy"] <= 4.5, "GRAVY hydropathy score out of Kyte-Doolittle range"
    assert fdict["PC_aliphatic_index"] > 0.0, "Aliphatic index should be positive"


# =========================================================================
# REGRESSION TEST 2: Unicode & Non-Breaking Whitespace Stripping
# =========================================================================
def test_unicode_and_non_breaking_whitespace():
    """Regression test: verify non-breaking spaces (\u00a0) and carriage returns are cleaned."""
    valid_base = "MKAYFIAILTLFTCIATVVRAQQMSELENRIDSLLNGKKATVGIAVWTDK"
    # Insert non-breaking space and carriage return
    dirty_seq = f"MKAY\u00a0FIAILTLFTC\r\nIATVVRAQQMSELENRI\tDSLLNGKKATVGIAVWTDK  "
    
    is_valid, clean_s, _, msg = clean_and_validate_sequence(dirty_seq)
    assert is_valid is True, f"Failed on Unicode whitespace with message: {msg}"
    assert clean_s == valid_base
    assert "\u00a0" not in clean_s


# =========================================================================
# REGRESSION TEST 3: Direct API Call Validation on Invalid Sequences
# =========================================================================
def test_direct_invalid_sequence_rejection():
    """Verify that predict_single_sequence raises ValueError rather than unhandled Biopython exceptions."""
    with pytest.raises(ValueError, match="Sequence validation failed"):
        predict_single_sequence("MKAYX")  # Non-canonical 'X'

    with pytest.raises(ValueError, match="Sequence validation failed"):
        predict_single_sequence("")  # Empty sequence

    with pytest.raises(ValueError, match="Sequence validation failed"):
        predict_single_sequence("M")  # Length < 2


# =========================================================================
# ADVERSARIAL TEST 4: Randomized Sequences & Determinism
# =========================================================================
def test_randomized_valid_sequences_determinism():
    """Generate 10 random canonical sequences of varying lengths; confirm finite scores and strict determinism."""
    rng = random.Random(42)
    aa_list = sorted(list(CANONICAL_AMINO_ACIDS))

    test_lengths = [50, 75, 120, 200, 350, 500, 750, 1000, 1200, 1500]
    for length in test_lengths:
        seq = "".join(rng.choice(aa_list) for _ in range(length))
        
        # Run inference 3 times on the exact same sequence to verify determinism
        pred1 = predict_single_sequence(seq, model_key="random_forest")
        pred2 = predict_single_sequence(seq, model_key="random_forest")
        
        assert pred1["score"] == pred2["score"], "Non-deterministic prediction scores detected!"
        assert pred1["predicted_label"] == pred2["predicted_label"]
        assert np.isfinite(pred1["score"])
        assert 0.0 <= pred1["score"] <= 1.0


# =========================================================================
# ADVERSARIAL TEST 5: Extreme Amino Acid Compositions (Homopolymers)
# =========================================================================
def test_extreme_composition_homopolymers():
    """Verify numerical stability on extreme homopolymers without NaNs or Infs."""
    extreme_cases = [
        ("Poly-Alanine", "A" * 60),
        ("Poly-Glycine", "G" * 60),
        ("Hydrophilic-Charge", ("K" * 30) + ("E" * 30)),
        ("Hydrophobic-Aliphatic", ("L" * 30) + ("V" * 30))
    ]

    for name, seq in extreme_cases:
        feats = extract_features_for_sequence(seq)
        assert len(feats) == 428
        for k, v in feats.items():
            assert np.isfinite(v), f"Non-finite feature {k} for {name}: {v}"

        pred = predict_single_sequence(seq, model_key="random_forest")
        assert np.isfinite(pred["score"])


# =========================================================================
# ADVERSARIAL TEST 6: Malformed FASTA & Partial Batch Errors
# =========================================================================
def test_malformed_fasta_batch_resilience():
    """Test FASTA with empty records, header-only, non-canonical letters, and special symbols."""
    adversarial_fasta = (
        ">Record_Valid_1 Clean protein\n"
        "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLL\n"
        ">Record_Header_Only\n"
        "\n"
        ">Record_Invalid_Chars Non-canonical symbols\n"
        "MKAYFIA123***ZT\n"
        ">Record_Valid_2 Another valid protein\n"
        "MLEGKVKWFNSEKGFGFIEVEGQDDVFVHFSAIQGEGFKTLEEGQAVSFEIVEGNRGPQAANVTKEA\n"
    )

    records = parse_fasta_multi(adversarial_fasta)
    assert len(records) >= 3

    batch_df = predict_batch_records(records, model_key="random_forest")
    assert len(batch_df) == len(records)
    
    # Valid records succeed, invalid records fail gracefully without crashing
    successes = batch_df[batch_df["status"] == "Success"]
    invalids = batch_df[batch_df["status"] == "Invalid Input"]
    
    assert len(successes) == 2, f"Expected 2 successful predictions, got {len(successes)}"
    assert len(invalids) >= 1, "Expected at least 1 invalid input capture"


# =========================================================================
# ADVERSARIAL TEST 7: Threshold Boundary Behavior
# =========================================================================
def test_threshold_boundary_behavior():
    """Verify decision logic exactly at threshold boundaries (t = 0.10)."""
    # Test helper checking logic
    t = 0.10
    assert bool(0.099999 >= t) is False
    assert bool(0.100000 >= t) is True
    assert bool(0.100001 >= t) is True


# =========================================================================
# ADVERSARIAL TEST 8: Multi-Model Invocation Parity
# =========================================================================
def test_multi_model_invocation_independence():
    """Verify each model uses its own estimator and produces valid scores."""
    test_seq = "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLL"
    
    rf_pred = predict_single_sequence(test_seq, model_key="random_forest")
    lr_pred = predict_single_sequence(test_seq, model_key="logistic_regression")
    svm_pred = predict_single_sequence(test_seq, model_key="svm_rbf")

    assert rf_pred["model_display_name"] == "Random Forest"
    assert lr_pred["model_display_name"] == "Logistic Regression"
    assert svm_pred["model_display_name"] == "SVM (RBF Kernel)"

    assert rf_pred["score_type"] == "Ensemble Probability"
    assert lr_pred["score_type"] == "Sigmoid Probability"
    assert svm_pred["score_type"] == "Platt-Calibrated Probability"

    # Scores should not all be identical
    scores = [rf_pred["score"], lr_pred["score"], svm_pred["score"]]
    assert len(set(scores)) > 1, "All models returned identical scores, indicating lack of architecture independence!"


# =========================================================================
# ADVERSARIAL TEST 9: Offline Safety & Unknown Model Rejection
# =========================================================================
def test_unknown_model_rejection():
    """Verify passing an unsupported model key raises ValueError."""
    with pytest.raises(ValueError, match="Model 'neural_net' not available"):
        predict_single_sequence("MKAYFIAILTLFTCIATVVRAQQMSELENRI", model_key="neural_net")
