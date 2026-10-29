"""
Automated tests for feature extraction dimensions, mathematical constraints,
numerical stability, and scaling discipline.
"""

from pathlib import Path
import json
import numpy as np
import pytest

from src.feature_extraction import (
    extract_features_for_sequence,
    compute_amino_acid_composition,
    compute_dipeptide_composition,
    compute_physicochemical_properties
)

DATA_SPLITS_DIR = Path("data/splits")


def test_single_sequence_feature_extraction_properties():
    sample_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDR"
    feats = extract_features_for_sequence(sample_seq)
    
    # Total features should be 428 (20 AAC + 400 DPC + 8 Physicochemical)
    assert len(feats) == 428
    
    # AAC sum check
    aac_vals = [v for k, v in feats.items() if k.startswith("AAC_")]
    assert len(aac_vals) == 20
    assert pytest.approx(sum(aac_vals), abs=1e-5) == 1.0
    
    # DPC sum check
    dpc_vals = [v for k, v in feats.items() if k.startswith("DPC_")]
    assert len(dpc_vals) == 400
    assert pytest.approx(sum(dpc_vals), abs=1e-5) == 1.0
    
    # Physicochemical sanity bounds
    assert feats["PC_length"] == len(sample_seq)
    assert feats["PC_molecular_weight"] > 5000.0
    assert 0.0 <= feats["PC_isoelectric_point"] <= 14.0
    assert -4.5 <= feats["PC_gravy"] <= 4.5
    assert 0.0 <= feats["PC_aromaticity"] <= 1.0
    assert feats["PC_instability_index"] > 0.0
    assert feats["PC_aliphatic_index"] > 0.0


def test_npz_feature_matrices_integrity():
    """Verify saved scaled feature matrices have correct shapes, alignment, and no NaNs."""
    scaled_npz = DATA_SPLITS_DIR / "features_scaled.npz"
    assert scaled_npz.exists(), "features_scaled.npz does not exist!"
    
    data = np.load(scaled_npz)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    # Check dimensions for exact 70:15:15 split
    assert X_train.shape == (8476, 428)
    assert y_train.shape == (8476,)
    assert X_val.shape == (1816, 428)
    assert y_val.shape == (1816,)
    assert X_test.shape == (1816, 428)
    assert y_test.shape == (1816,)
    
    # Check total sequences
    assert len(X_train) + len(X_val) + len(X_test) == 12108
    
    # Check no missing or infinite values
    assert not np.isnan(X_train).any(), "X_train contains NaNs!"
    assert not np.isnan(X_val).any(), "X_val contains NaNs!"
    assert not np.isnan(X_test).any(), "X_test contains NaNs!"
    assert not np.isinf(X_train).any(), "X_train contains Infs!"
    assert not np.isinf(X_val).any(), "X_val contains Infs!"
    assert not np.isinf(X_test).any(), "X_test contains Infs!"


def test_scaler_fitted_strictly_on_train():
    """Verify StandardScaler properties: mean 0 and std 1 on training set."""
    data = np.load(DATA_SPLITS_DIR / "features_scaled.npz")
    X_train = data["X_train"]
    
    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)
    
    # Every column in X_train should have mean ~ 0 and std ~ 1
    assert np.allclose(means, 0.0, atol=1e-3)
    assert np.allclose(stds, 1.0, atol=1e-3)
