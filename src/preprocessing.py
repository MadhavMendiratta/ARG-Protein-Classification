"""
Module: preprocessing.py
Objective: Parse, validate, clean, and deduplicate real protein sequence data.
Implements rigorous biological QC:
- Validates canonical 20 amino acid alphabet
- Filters sequence length extremes
- Removes exact duplicates and detects conflicting labels
- Length-stratified sampling to eliminate sequence-length confounding bias
- Outputs master cleaned dataset, FASTA, exclusion audit log, and statistical summary.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from Bio import SeqIO

DATA_RAW_DIR = Path("data/raw")
DATA_PROCESSED_DIR = Path("data/processed")

CANONICAL_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def validate_protein_sequence(seq_str: str) -> Tuple[bool, str]:
    """
    Check if a sequence contains exclusively canonical 20 amino acids.
    Returns (is_valid, reason).
    """
    seq_upper = seq_str.upper().strip()
    if not seq_upper:
        return False, "Empty sequence"
    
    invalid_chars = set(seq_upper) - CANONICAL_AMINO_ACIDS
    if invalid_chars:
        return False, f"Contains non-canonical amino acids: {', '.join(sorted(invalid_chars))}"
        
    return True, "Valid"


def parse_and_validate_fasta(
    fasta_path: Path,
    label: int,
    source_db: str,
    min_length: int = 50,
    max_length: int = 1500
) -> Tuple[List[dict], List[dict]]:
    """
    Parse a FASTA file, validate sequences, check lengths, and collect exclusion reasons.
    """
    valid_records = []
    excluded_records = []
    
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        seq_str = str(record.seq).upper().strip()
        seq_id = record.id
        description = record.description
        seq_len = len(seq_str)
        
        # Check alphabet validity
        is_valid_aa, aa_reason = validate_protein_sequence(seq_str)
        if not is_valid_aa:
            excluded_records.append({
                "sequence_id": seq_id,
                "description": description,
                "length": seq_len,
                "source_db": source_db,
                "label": label,
                "reason": aa_reason
            })
            continue
            
        # Check length criteria
        if seq_len < min_length:
            excluded_records.append({
                "sequence_id": seq_id,
                "description": description,
                "length": seq_len,
                "source_db": source_db,
                "label": label,
                "reason": f"Sequence length {seq_len} below minimum threshold ({min_length} aa)"
            })
            continue
            
        if seq_len > max_length:
            excluded_records.append({
                "sequence_id": seq_id,
                "description": description,
                "length": seq_len,
                "source_db": source_db,
                "label": label,
                "reason": f"Sequence length {seq_len} exceeds maximum threshold ({max_length} aa)"
            })
            continue
            
        valid_records.append({
            "sequence_id": seq_id,
            "description": description,
            "sequence": seq_str,
            "length": seq_len,
            "label": label,
            "source_db": source_db
        })
        
    return valid_records, excluded_records


def deduplicate_and_resolve_conflicts(
    pos_records: List[dict],
    neg_records: List[dict]
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Deduplicate within datasets and detect any cross-dataset sequence collisions.
    Returns (cleaned_pos, cleaned_neg, duplicate_and_conflict_exclusions).
    """
    exclusions = []
    
    # 1. Deduplicate positives internally
    pos_seen = {}
    unique_pos = []
    for rec in pos_records:
        seq = rec["sequence"]
        if seq in pos_seen:
            exclusions.append({
                "sequence_id": rec["sequence_id"],
                "description": rec["description"],
                "length": rec["length"],
                "source_db": rec["source_db"],
                "label": rec["label"],
                "reason": f"Duplicate positive sequence (identical to {pos_seen[seq]['sequence_id']})"
            })
        else:
            pos_seen[seq] = rec
            unique_pos.append(rec)
            
    # 2. Deduplicate negatives internally and detect collisions with positives
    neg_seen = {}
    unique_neg = []
    for rec in neg_records:
        seq = rec["sequence"]
        # Collision check with positives
        if seq in pos_seen:
            exclusions.append({
                "sequence_id": rec["sequence_id"],
                "description": rec["description"],
                "length": rec["length"],
                "source_db": rec["source_db"],
                "label": rec["label"],
                "reason": f"Label conflict collision: sequence identical to positive entry {pos_seen[seq]['sequence_id']}"
            })
            continue
            
        if seq in neg_seen:
            exclusions.append({
                "sequence_id": rec["sequence_id"],
                "description": rec["description"],
                "length": rec["length"],
                "source_db": rec["source_db"],
                "label": rec["label"],
                "reason": f"Duplicate negative sequence (identical to {neg_seen[seq]['sequence_id']})"
            })
        else:
            neg_seen[seq] = rec
            unique_neg.append(rec)
            
    return unique_pos, unique_neg, exclusions


def balance_and_match_length(
    pos_records: List[dict],
    neg_records: List[dict],
    target_ratio: float = 1.0,
    random_seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    """
    Ensure the negative set is sampled to match the length distribution of positives
    using stratified length bins. This prevents classifiers from learning sequence length
    as a spurious distinguishing feature.
    """
    n_pos = len(pos_records)
    target_neg_count = int(n_pos * target_ratio)
    
    if len(neg_records) <= target_neg_count:
        return pos_records, neg_records
        
    df_pos = pd.DataFrame(pos_records)
    df_neg = pd.DataFrame(neg_records)
    
    # Create 10 quantile bins based on positive length distribution
    quantiles = np.linspace(0, 1, 11)
    bin_edges = np.quantile(df_pos["length"], quantiles)
    bin_edges[0] = 0
    bin_edges[-1] = np.inf
    
    df_pos["bin"] = pd.cut(df_pos["length"], bins=bin_edges, labels=False, include_lowest=True)
    df_neg["bin"] = pd.cut(df_neg["length"], bins=bin_edges, labels=False, include_lowest=True)
    
    bin_counts = df_pos["bin"].value_counts(normalize=True)
    
    sampled_neg_dfs = []
    rng = np.random.RandomState(random_seed)
    
    for bin_id, proportion in bin_counts.items():
        needed_for_bin = int(round(proportion * target_neg_count))
        candidates_in_bin = df_neg[df_neg["bin"] == bin_id]
        
        if len(candidates_in_bin) >= needed_for_bin:
            sampled_bin = candidates_in_bin.sample(n=needed_for_bin, random_state=rng)
        else:
            sampled_bin = candidates_in_bin
            
        sampled_neg_dfs.append(sampled_bin)
        
    df_sampled_neg = pd.concat(sampled_neg_dfs).drop_duplicates(subset=["sequence_id"])
    
    # If slight discrepancy due to rounding or bin exhaustion, sample remaining uniformly
    if len(df_sampled_neg) < target_neg_count:
        remaining_needed = target_neg_count - len(df_sampled_neg)
        remaining_pool = df_neg[~df_neg["sequence_id"].isin(df_sampled_neg["sequence_id"])]
        if len(remaining_pool) > 0:
            add_sample = remaining_pool.sample(min(remaining_needed, len(remaining_pool)), random_state=rng)
            df_sampled_neg = pd.concat([df_sampled_neg, add_sample])
    elif len(df_sampled_neg) > target_neg_count:
        df_sampled_neg = df_sampled_neg.sample(target_neg_count, random_state=rng)
        
    df_pos = df_pos.drop(columns=["bin"])
    df_sampled_neg = df_sampled_neg.drop(columns=["bin"])
    
    return df_pos.to_dict("records"), df_sampled_neg.to_dict("records")


def compute_length_summary(lengths: List[int]) -> dict:
    """Compute standard summary statistics for sequence lengths."""
    arr = np.array(lengths)
    if len(arr) == 0:
        return {}
    return {
        "count": int(len(arr)),
        "min": int(np.min(arr)),
        "q1": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "q3": float(np.percentile(arr, 75)),
        "max": int(np.max(arr)),
        "std": float(np.std(arr))
    }


def run_data_validation_and_preprocessing(
    pos_fasta: Path = DATA_RAW_DIR / "card_homolog_raw.fasta",
    neg_fasta: Path = DATA_RAW_DIR / "uniprot_negative_raw.fasta",
    min_length: int = 50,
    max_length: int = 1500,
    target_ratio: float = 1.0,
    random_seed: int = 42
) -> dict:
    """Orchestrate Phase 3 Data Validation and Preprocessing."""
    print("=" * 70)
    print("PHASE 3: SEQUENCE VALIDATION, QUALITY CONTROL & DEDUPLICATION")
    print("=" * 70)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Parse and validate positives
    print(f"[Validation] Parsing positive ARGs from {pos_fasta}...")
    valid_pos, excl_pos = parse_and_validate_fasta(pos_fasta, label=1, source_db="CARD", min_length=min_length, max_length=max_length)
    print(f"  Positives raw parsed: {len(valid_pos) + len(excl_pos)} | Valid: {len(valid_pos)} | Excluded: {len(excl_pos)}")
    
    # 2. Parse and validate negatives
    print(f"[Validation] Parsing candidate non-ARGs from {neg_fasta}...")
    valid_neg, excl_neg = parse_and_validate_fasta(neg_fasta, label=0, source_db="UniProtKB/Swiss-Prot", min_length=min_length, max_length=max_length)
    print(f"  Negatives raw parsed: {len(valid_neg) + len(excl_neg)} | Valid: {len(valid_neg)} | Excluded: {len(excl_neg)}")
    
    # 3. Deduplicate and detect conflicts
    print("[Validation] Checking for duplicate sequences and label conflicts...")
    unique_pos, unique_neg, dedup_exclusions = deduplicate_and_resolve_conflicts(valid_pos, valid_neg)
    print(f"  Positives after dedup: {len(unique_pos)}")
    print(f"  Negatives after dedup & conflict check: {len(unique_neg)}")
    print(f"  Duplicate/conflict exclusions: {len(dedup_exclusions)}")
    
    # 4. Length-stratified balance
    print(f"[Validation] Applying length-stratified sampling for balanced 1:{target_ratio:.1f} ratio...")
    final_pos, final_neg = balance_and_match_length(unique_pos, unique_neg, target_ratio=target_ratio, random_seed=random_seed)
    print(f"  Final Positive count: {len(final_pos)}")
    print(f"  Final Negative count: {len(final_neg)}")
    
    all_final_records = final_pos + final_neg
    for i, rec in enumerate(all_final_records):
        rec["seq_uid"] = f"SEQ_{i+1:06d}"
    df_master = pd.DataFrame(all_final_records)
    cols = ["seq_uid", "sequence_id", "description", "sequence", "length", "label", "source_db"]
    df_master = df_master[cols]
    
    # Save master dataset CSV
    master_csv_path = DATA_PROCESSED_DIR / "dataset_master.csv"
    df_master.to_csv(master_csv_path, index=False)
    print(f"[Output] Master dataset saved: {master_csv_path} ({len(df_master)} total records)")
    
    # Save master cleaned FASTA with unambiguous clean IDs (no pipe delimiters)
    master_fasta_path = DATA_PROCESSED_DIR / "dataset_clean.fasta"
    with open(master_fasta_path, "w", encoding="utf-8") as out_f:
        for rec in all_final_records:
            header = f">{rec['seq_uid']}"
            out_f.write(f"{header}\n{rec['sequence']}\n")
    print(f"[Output] Cleaned FASTA saved: {master_fasta_path}")
    
    # Save exclusion audit log
    all_exclusions = excl_pos + excl_neg + dedup_exclusions
    df_exclusions = pd.DataFrame(all_exclusions)
    excl_csv_path = DATA_PROCESSED_DIR / "exclusion_log.csv"
    df_exclusions.to_csv(excl_csv_path, index=False)
    print(f"[Output] Exclusion audit log saved: {excl_csv_path} ({len(df_exclusions)} total exclusions)")
    
    # Statistical summary
    pos_lengths = [r["length"] for r in final_pos]
    neg_lengths = [r["length"] for r in final_neg]
    all_lengths = [r["length"] for r in all_final_records]
    
    summary = {
        "raw_counts": {
            "card_positive_raw": len(valid_pos) + len(excl_pos),
            "uniprot_negative_raw": len(valid_neg) + len(excl_neg),
            "total_raw": len(valid_pos) + len(excl_pos) + len(valid_neg) + len(excl_neg)
        },
        "exclusions": {
            "invalid_amino_acids_or_length_pos": len(excl_pos),
            "invalid_amino_acids_or_length_neg": len(excl_neg),
            "duplicates_and_label_conflicts": len(dedup_exclusions),
            "total_exclusions": len(all_exclusions)
        },
        "final_dataset": {
            "total_sequences": len(all_final_records),
            "positive_args": len(final_pos),
            "negative_non_args": len(final_neg),
            "class_ratio": f"{len(final_pos)}:{len(final_neg)}"
        },
        "length_statistics": {
            "positive_args": compute_length_summary(pos_lengths),
            "negative_non_args": compute_length_summary(neg_lengths),
            "overall": compute_length_summary(all_lengths)
        }
    }
    
    summary_path = DATA_PROCESSED_DIR / "validation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Output] Summary statistics saved: {summary_path}")
    
    print("=" * 70)
    print(f"DATASET VALIDATION COMPLETE:")
    print(f"Positive ARGs (CARD):       {len(final_pos):>6} sequences")
    print(f"Negative Non-ARGs (UniProt):{len(final_neg):>6} sequences")
    print(f"Total Master Dataset:       {len(all_final_records):>6} sequences")
    print(f"Positive Median Length:     {summary['length_statistics']['positive_args']['median']:.1f} aa")
    print(f"Negative Median Length:     {summary['length_statistics']['negative_non_args']['median']:.1f} aa")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    run_data_validation_and_preprocessing()
