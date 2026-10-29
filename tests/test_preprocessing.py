"""
Unit tests for data validation and sequence cleaning logic.
"""

import pytest
from src.preprocessing import validate_protein_sequence, deduplicate_and_resolve_conflicts


def test_validate_protein_sequence_valid():
    # Valid standard 20 amino acid sequence
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDR"
    is_valid, reason = validate_protein_sequence(seq)
    assert is_valid is True
    assert reason == "Valid"


def test_validate_protein_sequence_non_canonical():
    # Sequence with ambiguous amino acid X
    seq_with_x = "MKTAXIAKQRQ"
    is_valid, reason = validate_protein_sequence(seq_with_x)
    assert is_valid is False
    assert "X" in reason

    # Sequence with premature stop codon *
    seq_with_stop = "MKT*YIAKQRQ"
    is_valid, reason = validate_protein_sequence(seq_with_stop)
    assert is_valid is False

    # Empty sequence
    is_valid, reason = validate_protein_sequence("   ")
    assert is_valid is False


def test_deduplicate_and_resolve_conflicts():
    pos_records = [
        {"sequence_id": "P1", "description": "ARG 1", "sequence": "MKTAYIAK", "length": 8, "source_db": "CARD", "label": 1},
        {"sequence_id": "P2", "description": "ARG 2 duplicate", "sequence": "MKTAYIAK", "length": 8, "source_db": "CARD", "label": 1},
        {"sequence_id": "P3", "description": "ARG 3", "sequence": "ACDEFGHIKL", "length": 10, "source_db": "CARD", "label": 1},
    ]
    neg_records = [
        {"sequence_id": "N1", "description": "Non-ARG 1", "sequence": "MNIDMNWLGQ", "length": 10, "source_db": "UniProt", "label": 0},
        {"sequence_id": "N2", "description": "Collision with P3", "sequence": "ACDEFGHIKL", "length": 10, "source_db": "UniProt", "label": 0},
        {"sequence_id": "N3", "description": "Duplicate of N1", "sequence": "MNIDMNWLGQ", "length": 10, "source_db": "UniProt", "label": 0},
    ]

    clean_pos, clean_neg, exclusions = deduplicate_and_resolve_conflicts(pos_records, neg_records)
    
    # P1 kept, P2 excluded as duplicate
    assert len(clean_pos) == 2
    assert [p["sequence_id"] for p in clean_pos] == ["P1", "P3"]
    
    # N1 kept, N2 excluded as collision with P3, N3 excluded as duplicate of N1
    assert len(clean_neg) == 1
    assert clean_neg[0]["sequence_id"] == "N1"
    
    # Exclusions count
    assert len(exclusions) == 3
    reasons = [e["reason"] for e in exclusions]
    assert any("Duplicate positive" in r for r in reasons)
    assert any("Label conflict collision" in r for r in reasons)
    assert any("Duplicate negative" in r for r in reasons)
