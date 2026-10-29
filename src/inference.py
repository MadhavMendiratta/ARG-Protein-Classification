"""
Module: src/inference.py
Objective: Production-grade, leak-free inference pipeline for the ARG Classification Dashboard.
Reuses the exact feature extraction, preprocessing alphabet, and model artifacts
trained during the research phase.
"""

import io
import json
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from Bio import SeqIO

from src.feature_extraction import extract_features_for_sequence
from src.preprocessing import CANONICAL_AMINO_ACIDS

# Project root aware paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class ModelArtifactManager:
    """Manages lazy-loading and caching of trained models, scaler, and feature metadata."""
    _instance: Optional['ModelArtifactManager'] = None

    def __init__(self):
        self.feature_names: List[str] = []
        self.scaler = None
        self.models: Dict[str, Dict[str, Any]] = {}
        self._load_core_artifacts()

    @classmethod
    def get_instance(cls) -> 'ModelArtifactManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_core_artifacts(self):
        """Load feature names, scaler, and models with robust error handling."""
        fnames_path = DATA_SPLITS_DIR / "feature_names.json"
        if not fnames_path.exists():
            raise FileNotFoundError(f"Feature names file missing: {fnames_path}")
        with open(fnames_path, "r", encoding="utf-8") as f:
            self.feature_names = json.load(f)

        scaler_path = DATA_SPLITS_DIR / "scaler.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Fitted scaler missing: {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        model_files = {
            "random_forest": ("Random Forest", MODELS_DIR / "random_forest_model.joblib"),
            "logistic_regression": ("Logistic Regression", MODELS_DIR / "logistic_regression_model.joblib"),
            "svm_rbf": ("SVM (RBF Kernel)", MODELS_DIR / "svm_rbf_model.joblib")
        }

        for key, (display_name, path) in model_files.items():
            if path.exists():
                data = joblib.load(path)
                self.models[key] = {
                    "display_name": display_name,
                    "model": data["best_estimator"],
                    "calibrator": data.get("calibrator"),
                    "threshold": float(data.get("val_metrics", {}).get("optimal_threshold", 0.10)),
                    "val_metrics": data.get("val_metrics", {}),
                    "best_params": data.get("best_params", {}),
                    "cv_roc_auc": data.get("cv_roc_auc", 0.0),
                    "train_time_sec": data.get("train_time_sec", 0.0)
                }
            else:
                print(f"[Warning] Model file missing: {path}")

    def get_available_models(self) -> Dict[str, str]:
        """Return dict of {model_key: display_name} for all loaded models."""
        return {k: v["display_name"] for k, v in self.models.items()}


def clean_and_validate_sequence(raw_input: str, default_id: str = "Query_Seq") -> Tuple[bool, str, str, str]:
    """
    Parse and validate a single protein sequence (plain sequence or FASTA format).
    Returns:
        (is_valid, clean_seq, seq_id, message)
    """
    if not raw_input or not raw_input.strip():
        return False, "", default_id, "Input sequence is empty."

    lines = raw_input.strip().splitlines()
    seq_id = default_id
    seq_lines = []

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        if line_str.startswith(">"):
            header_parts = line_str[1:].split()
            seq_id = header_parts[0] if header_parts else default_id
        else:
            seq_lines.append(line_str)

    raw_combined = "".join(seq_lines).upper()
    clean_seq = "".join(c for c in raw_combined if not c.isspace())

    if not clean_seq:
        return False, "", seq_id, "Sequence body contains no characters."

    # Validate against canonical 20 amino acids
    invalid_chars = set(clean_seq) - CANONICAL_AMINO_ACIDS
    if invalid_chars:
        return (
            False,
            clean_seq,
            seq_id,
            f"Non-canonical amino acid character(s) detected: {', '.join(sorted(invalid_chars))}. Only standard 20 amino acids are accepted."
        )

    # Length advisories
    length = len(clean_seq)
    if length < 2:
        return False, clean_seq, seq_id, f"Sequence length ({length} aa) is too short for dipeptide composition (min 2 aa required)."

    warning_msg = "Valid"
    if length < 50:
        warning_msg = f"Valid (Notice: Sequence length {length} aa is shorter than the training dataset range of 50–1,500 aa)."
    elif length > 1500:
        warning_msg = f"Valid (Notice: Sequence length {length} aa exceeds the training dataset range of 50–1,500 aa)."

    return True, clean_seq, seq_id, warning_msg


def parse_fasta_multi(content_str: str) -> List[Dict[str, Any]]:
    """
    Parse a multi-record FASTA string or file content into a list of records.
    Returns:
        List of dicts: {'seq_id': str, 'description': str, 'seq': str, 'is_valid': bool, 'msg': str}
    """
    records = []
    if not content_str or not content_str.strip():
        return records

    f_handle = io.StringIO(content_str)
    try:
        parsed_entries = list(SeqIO.parse(f_handle, "fasta"))
    except Exception as e:
        return [{"seq_id": "Error", "description": "Parse failure", "seq": "", "is_valid": False, "msg": f"FASTA format error: {str(e)}"}]

    if not parsed_entries:
        # Check if user entered a plain sequence without FASTA header
        is_valid, clean_seq, seq_id, msg = clean_and_validate_sequence(content_str, default_id="Seq_1")
        records.append({
            "seq_id": seq_id,
            "description": "Plain Sequence Input",
            "seq": clean_seq,
            "is_valid": is_valid,
            "msg": msg
        })
        return records

    for idx, entry in enumerate(parsed_entries, start=1):
        raw_seq = str(entry.seq).strip().upper()
        seq_id = entry.id if entry.id else f"Record_{idx}"
        desc = entry.description if entry.description else seq_id

        is_valid, clean_seq, _, msg = clean_and_validate_sequence(raw_seq, default_id=seq_id)
        records.append({
            "seq_id": seq_id,
            "description": desc,
            "seq": clean_seq,
            "is_valid": is_valid,
            "msg": msg
        })

    return records


def predict_single_sequence(
    seq: str,
    model_key: str = "random_forest"
) -> Dict[str, Any]:
    """
    Run the verified end-to-end inference pipeline on a single clean sequence:
    1. Validate input sequence.
    2. Extract 428 features (AAC, DPC, Physicochemical).
    3. Standardize features using the frozen training-only scaler.
    4. Run model inference with probability calculation / calibration.
    5. Apply validation-calibrated threshold (0.10).
    6. Return structured prediction results and key properties.
    """
    is_valid, clean_seq, seq_id, val_msg = clean_and_validate_sequence(seq, default_id="Query_Seq")
    if not is_valid:
        raise ValueError(f"Sequence validation failed: {val_msg}")
    seq = clean_seq

    mgr = ModelArtifactManager.get_instance()
    if model_key not in mgr.models:
        raise ValueError(f"Model '{model_key}' not available. Loaded models: {list(mgr.models.keys())}")

    model_info = mgr.models[model_key]
    model = model_info["model"]
    calibrator = model_info.get("calibrator")
    threshold = model_info["threshold"]
    display_name = model_info["display_name"]

    # 1. Feature Extraction
    feat_dict = extract_features_for_sequence(seq)

    # 2. Vector construction matching exact feature_names ordering
    feat_vector = np.array([feat_dict[k] for k in mgr.feature_names], dtype=np.float64).reshape(1, -1)

    # 3. Scaling
    feat_vector_scaled = mgr.scaler.transform(feat_vector)

    # 4. Model probability scoring
    if calibrator is not None:
        # SVM decision function + Platt scaling calibrator
        scores = model.decision_function(feat_vector_scaled).reshape(-1, 1)
        prob = float(calibrator.predict_proba(scores)[0, 1])
        score_type = "Platt-Calibrated Probability"
    elif hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(feat_vector_scaled)[0, 1])
        score_type = "Ensemble Probability" if "Forest" in display_name else "Sigmoid Probability"
    else:
        # Fallback to decision function if predict_proba is unavailable
        prob = float(model.decision_function(feat_vector_scaled)[0])
        score_type = "Decision Margin"

    # 5. Threshold application
    is_arg = bool(prob >= threshold)
    pred_label = 1 if is_arg else 0
    pred_text = "Predicted ARG-like" if is_arg else "Predicted candidate non-ARG-like"

    return {
        "model_key": model_key,
        "model_display_name": display_name,
        "sequence_length": len(seq),
        "score": prob,
        "score_type": score_type,
        "threshold_used": threshold,
        "predicted_label": pred_label,
        "predicted_text": pred_text,
        "is_arg": is_arg,
        "features_dict": feat_dict,
        "features_scaled": feat_vector_scaled[0]
    }


def predict_batch_records(
    records: List[Dict[str, Any]],
    model_key: str = "random_forest"
) -> pd.DataFrame:
    """
    Run inference on a batch of parsed FASTA records.
    Returns a pandas DataFrame ready for display and CSV download.
    """
    results = []
    for r in records:
        if not r["is_valid"]:
            results.append({
                "seq_uid": r["seq_id"],
                "description": r["description"],
                "length": len(r["seq"]),
                "status": "Invalid Input",
                "prediction": "Skipped (Invalid)",
                "predicted_label": -1,
                "score": np.nan,
                "score_type": "N/A",
                "threshold": np.nan,
                "model_used": model_key,
                "notes": r["msg"]
            })
            continue

        try:
            pred = predict_single_sequence(r["seq"], model_key=model_key)
            results.append({
                "seq_uid": r["seq_id"],
                "description": r["description"],
                "length": pred["sequence_length"],
                "status": "Success",
                "prediction": pred["predicted_text"],
                "predicted_label": pred["predicted_label"],
                "score": round(pred["score"], 4),
                "score_type": pred["score_type"],
                "threshold": pred["threshold_used"],
                "model_used": pred["model_display_name"],
                "notes": r["msg"] if "Notice" in r["msg"] else "Clean sequence"
            })
        except Exception as e:
            results.append({
                "seq_uid": r["seq_id"],
                "description": r["description"],
                "length": len(r["seq"]),
                "status": "Inference Error",
                "prediction": "Error",
                "predicted_label": -1,
                "score": np.nan,
                "score_type": "N/A",
                "threshold": np.nan,
                "model_used": model_key,
                "notes": str(e)
            })

    return pd.DataFrame(results)
