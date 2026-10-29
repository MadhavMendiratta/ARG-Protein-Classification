"""
Module: feature_extraction.py
Objective: Extract 428 biological and numerical sequence features:
1. Amino Acid Composition (AAC: 20 features)
2. Dipeptide Composition (DPC: 400 features)
3. Physicochemical Properties (8 features: MW, pI, GRAVY, aromaticity, instability,
   aliphatic index, charge at pH 7, sequence length)
Ensures:
- Exact sequence alignment and 1-to-1 mapping with seq_uid and labels
- Zero NaNs, Infs, or missing values
- Feature scaling (StandardScaler) is strictly fitted ONLY on the training split.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

DATA_PROCESSED_DIR = Path("data/processed")
DATA_SPLITS_DIR = Path("data/splits")

STANDARD_AMINO_ACIDS = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]

ALL_DIPEPTIDES = [a1 + a2 for a1 in STANDARD_AMINO_ACIDS for a2 in STANDARD_AMINO_ACIDS]


def compute_amino_acid_composition(seq: str) -> Dict[str, float]:
    """Calculate relative frequency of each of the 20 standard amino acids."""
    length = len(seq)
    counts = {aa: 0 for aa in STANDARD_AMINO_ACIDS}
    for char in seq:
        if char in counts:
            counts[char] += 1
    return {f"AAC_{aa}": counts[aa] / length for aa in STANDARD_AMINO_ACIDS}


def compute_dipeptide_composition(seq: str) -> Dict[str, float]:
    """Calculate relative frequency of all 400 adjacent amino acid pairs."""
    length = len(seq)
    if length <= 1:
        return {f"DPC_{dp}": 0.0 for dp in ALL_DIPEPTIDES}
        
    counts = {dp: 0 for dp in ALL_DIPEPTIDES}
    for i in range(length - 1):
        pair = seq[i:i+2]
        if pair in counts:
            counts[pair] += 1
            
    total_pairs = length - 1
    return {f"DPC_{dp}": counts[dp] / total_pairs for dp in ALL_DIPEPTIDES}


def compute_physicochemical_properties(seq: str) -> Dict[str, float]:
    """
    Calculate 8 biologically grounded physicochemical descriptors:
    - Length
    - Molecular Weight (Daltons)
    - Theoretical Isoelectric Point (pI)
    - Grand Average of Hydropathicity (GRAVY; Kyte & Doolittle 1982)
    - Aromaticity (fraction of Phe, Trp, Tyr)
    - Instability Index (Guruprasad et al. 1990)
    - Aliphatic Index (Ikai 1980)
    - Net charge at physiological pH 7.0
    """
    length = len(seq)
    analysis = ProteinAnalysis(seq)
    
    # Biopython analysis methods
    mw = float(analysis.molecular_weight())
    pi = float(analysis.isoelectric_point())
    gravy = float(analysis.gravy())
    aromaticity = float(analysis.aromaticity())
    instability = float(analysis.instability_index())
    charge_ph7 = float(analysis.charge_at_pH(7.0))
    
    # Aliphatic index: X(Ala) + 2.9 * X(Val) + 3.9 * (X(Ile) + X(Leu))
    pct_A = (seq.count('A') / length) * 100.0
    pct_V = (seq.count('V') / length) * 100.0
    pct_I = (seq.count('I') / length) * 100.0
    pct_L = (seq.count('L') / length) * 100.0
    aliphatic_index = float(pct_A + 2.9 * pct_V + 3.9 * (pct_I + pct_L))
    
    return {
        "PC_length": float(length),
        "PC_molecular_weight": mw,
        "PC_isoelectric_point": pi,
        "PC_gravy": gravy,
        "PC_aromaticity": aromaticity,
        "PC_instability_index": instability,
        "PC_aliphatic_index": aliphatic_index,
        "PC_net_charge_pH7": charge_ph7
    }


def extract_features_for_sequence(seq: str) -> Dict[str, float]:
    """Extract complete 428-dimensional feature dictionary for a single protein sequence."""
    features = {}
    features.update(compute_amino_acid_composition(seq))
    features.update(compute_dipeptide_composition(seq))
    features.update(compute_physicochemical_properties(seq))
    return features


def extract_features_for_dataset(
    master_csv_path: Path = DATA_PROCESSED_DIR / "dataset_master.csv",
    batch_size: int = 1000
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Extract features for all sequences in dataset_master.csv.
    Returns: (X_matrix, y_vector, seq_uids, feature_names)
    """
    df = pd.read_csv(master_csv_path)
    n_seqs = len(df)
    print(f"[FeatureExtraction] Extracting 428 features for {n_seqs} sequences...")
    
    # Pre-extract first to get fixed feature ordering
    first_feat = extract_features_for_sequence(df["sequence"].iloc[0])
    feature_names = list(first_feat.keys())
    n_features = len(feature_names)
    
    X = np.empty((n_seqs, n_features), dtype=np.float64)
    y = df["label"].values.astype(np.int64)
    seq_uids = df["seq_uid"].tolist()
    
    for i, seq in enumerate(df["sequence"]):
        feat_dict = extract_features_for_sequence(seq)
        X[i, :] = [feat_dict[k] for k in feature_names]
        if (i + 1) % batch_size == 0 or (i + 1) == n_seqs:
            print(f"  Processed {i + 1}/{n_seqs} sequences ({(i + 1)/n_seqs*100:.1f}%)...")
            
    return X, y, seq_uids, feature_names


def fit_and_scale_splits(
    X: np.ndarray,
    y: np.ndarray,
    seq_uids: List[str],
    feature_names: List[str]
) -> dict:
    """
    Fit StandardScaler strictly on training split and transform all splits.
    Saves raw and scaled feature sets and verifies numerical stability.
    """
    from sklearn.preprocessing import StandardScaler
    
    DATA_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load split indices
    train_df = pd.read_csv(DATA_SPLITS_DIR / "train_indices.csv")
    val_df = pd.read_csv(DATA_SPLITS_DIR / "val_indices.csv")
    test_df = pd.read_csv(DATA_SPLITS_DIR / "test_indices.csv")
    
    uid_to_idx = {uid: idx for idx, uid in enumerate(seq_uids)}
    
    train_idx = np.array([uid_to_idx[uid] for uid in train_df["seq_uid"]])
    val_idx = np.array([uid_to_idx[uid] for uid in val_df["seq_uid"]])
    test_idx = np.array([uid_to_idx[uid] for uid in test_df["seq_uid"]])
    
    # Split raw data
    X_train_raw = X[train_idx]
    y_train = y[train_idx]
    
    X_val_raw = X[val_idx]
    y_val = y[val_idx]
    
    X_test_raw = X[test_idx]
    y_test = y[test_idx]
    
    print("[Preprocessing] Fitting StandardScaler ONLY on Training split...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    
    print("[Preprocessing] Transforming Validation and Test splits with fitted scaler...")
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Numerical stability assertions
    assert not np.isnan(X).any(), "NaN detected in raw feature matrix!"
    assert not np.isinf(X).any(), "Inf detected in raw feature matrix!"
    assert not np.isnan(X_train_scaled).any(), "NaN detected in scaled X_train!"
    assert not np.isnan(X_val_scaled).any(), "NaN detected in scaled X_val!"
    assert not np.isnan(X_test_scaled).any(), "NaN detected in scaled X_test!"
    
    # Verify train mean ~ 0 and std ~ 1
    train_means = np.mean(X_train_scaled, axis=0)
    train_stds = np.std(X_train_scaled, axis=0)
    print(f"  Training scaled mean range: [{np.min(train_means):.4f}, {np.max(train_means):.4f}]")
    print(f"  Training scaled std range:  [{np.min(train_stds):.4f}, {np.max(train_stds):.4f}]")
    
    # Save scaler
    scaler_path = DATA_SPLITS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"[Saved] Fitted StandardScaler saved to {scaler_path}")
    
    # Save feature names
    names_path = DATA_SPLITS_DIR / "feature_names.json"
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    print(f"[Saved] Feature names saved to {names_path}")
    
    # Save scaled NPZ dataset
    scaled_npz_path = DATA_SPLITS_DIR / "features_scaled.npz"
    np.savez_compressed(
        scaled_npz_path,
        X_train=X_train_scaled, y_train=y_train, train_uids=train_df["seq_uid"].values,
        X_val=X_val_scaled, y_val=y_val, val_uids=val_df["seq_uid"].values,
        X_test=X_test_scaled, y_test=y_test, test_uids=test_df["seq_uid"].values
    )
    print(f"[Saved] Scaled train/val/test matrices saved to {scaled_npz_path}")
    
    # Save raw NPZ dataset
    raw_npz_path = DATA_SPLITS_DIR / "features_raw.npz"
    np.savez_compressed(
        raw_npz_path,
        X_train=X_train_raw, y_train=y_train,
        X_val=X_val_raw, y_val=y_val,
        X_test=X_test_raw, y_test=y_test,
        all_X=X, all_y=y, all_uids=np.array(seq_uids)
    )
    print(f"[Saved] Raw feature matrices saved to {raw_npz_path}")
    
    summary = {
        "total_sequences": int(len(seq_uids)),
        "total_features": int(len(feature_names)),
        "feature_breakdown": {
            "amino_acid_composition_aac": 20,
            "dipeptide_composition_dpc": 400,
            "physicochemical_properties_pc": 8
        },
        "shapes": {
            "X_train": list(X_train_scaled.shape),
            "y_train": list(y_train.shape),
            "X_val": list(X_val_scaled.shape),
            "y_val": list(y_val.shape),
            "X_test": list(X_test_scaled.shape),
            "y_test": list(y_test.shape)
        },
        "class_distributions": {
            "train": {"positive": int((y_train == 1).sum()), "negative": int((y_train == 0).sum())},
            "val": {"positive": int((y_val == 1).sum()), "negative": int((y_val == 0).sum())},
            "test": {"positive": int((y_test == 1).sum()), "negative": int((y_test == 0).sum())}
        },
        "numerical_checks": {
            "raw_has_nans": bool(np.isnan(X).any()),
            "raw_has_infs": bool(np.isinf(X).any()),
            "scaled_has_nans": bool(np.isnan(X_train_scaled).any() or np.isnan(X_val_scaled).any() or np.isnan(X_test_scaled).any()),
            "scaler_fitted_on": "training_split_only"
        }
    }
    
    summary_path = DATA_SPLITS_DIR / "feature_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] Feature summary saved to {summary_path}")
    
    return summary


def run_feature_extraction_pipeline():
    """Orchestrate Task 6, 7, 8: Feature extraction, validation, and strict train scaling."""
    print("=" * 70)
    print("TASK 6-8: FEATURE EXTRACTION (AAC, DPC, PHYSICOCHEMICAL) & TRAIN SCALING")
    print("=" * 70)
    
    X, y, seq_uids, feature_names = extract_features_for_dataset()
    summary = fit_and_scale_splits(X, y, seq_uids, feature_names)
    
    print("=" * 70)
    print(f"FEATURE EXTRACTION COMPLETE:")
    print(f"Total Sequences:        {summary['total_sequences']}")
    print(f"Total Feature Columns:  {summary['total_features']} (20 AAC + 400 DPC + 8 Physicochemical)")
    print(f"Train Shape:            {summary['shapes']['X_train']}")
    print(f"Val Shape:              {summary['shapes']['X_val']}")
    print(f"Test Shape:             {summary['shapes']['X_test']}")
    print(f"Zero NaNs / Infs:       Confirmed")
    print(f"Scaling Protocol:       StandardScaler fitted exclusively on X_train")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    run_feature_extraction_pipeline()
