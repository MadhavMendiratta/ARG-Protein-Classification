"""
Automated tests for MMseqs2 clustering and cluster-aware splitting integrity.
"""

from pathlib import Path
import pandas as pd
import pytest

DATA_SPLITS_DIR = Path("data/splits")
DATA_PROCESSED_DIR = Path("data/processed")


def test_split_files_exist():
    assert (DATA_SPLITS_DIR / "train_indices.csv").exists()
    assert (DATA_SPLITS_DIR / "val_indices.csv").exists()
    assert (DATA_SPLITS_DIR / "test_indices.csv").exists()
    assert (DATA_SPLITS_DIR / "sequence_cluster_assignments.csv").exists()
    assert (DATA_SPLITS_DIR / "clusters_summary.csv").exists()


def test_partition_disjointness_and_completeness():
    """Verify that Train, Val, and Test are mutually exclusive and collectively exhaustive."""
    df_master = pd.read_csv(DATA_PROCESSED_DIR / "dataset_master.csv")
    train_df = pd.read_csv(DATA_SPLITS_DIR / "train_indices.csv")
    val_df = pd.read_csv(DATA_SPLITS_DIR / "val_indices.csv")
    test_df = pd.read_csv(DATA_SPLITS_DIR / "test_indices.csv")

    all_uids = set(df_master["seq_uid"])
    train_uids = set(train_df["seq_uid"])
    val_uids = set(val_df["seq_uid"])
    test_uids = set(test_df["seq_uid"])

    # Disjointness checks
    assert train_uids.isdisjoint(val_uids), "Train and Val share sequence UIDs!"
    assert train_uids.isdisjoint(test_uids), "Train and Test share sequence UIDs!"
    assert val_uids.isdisjoint(test_uids), "Val and Test share sequence UIDs!"

    # Completeness check
    combined = train_uids | val_uids | test_uids
    assert combined == all_uids, "Partitions do not sum to total master sequences!"
    assert len(train_df) + len(val_df) + len(test_df) == len(df_master) == 12108


def test_zero_cluster_splitting():
    """Verify that no cluster is ever split across multiple partitions."""
    df_seq = pd.read_csv(DATA_SPLITS_DIR / "sequence_cluster_assignments.csv")
    splits_per_cluster = df_seq.groupby("cluster_id")["split"].nunique()
    split_clusters = splits_per_cluster[splits_per_cluster > 1]
    assert len(split_clusters) == 0, f"Clusters split across partitions: {split_clusters.to_dict()}"


def test_class_balance_in_splits():
    """Verify that class balance is preserved (approximately 50% pos / 50% neg) in every split."""
    for split_name in ["train", "val", "test"]:
        df = pd.read_csv(DATA_SPLITS_DIR / f"{split_name}_indices.csv")
        n_pos = (df["label"] == 1).sum()
        n_neg = (df["label"] == 0).sum()
        ratio = n_pos / len(df)
        assert 0.45 <= ratio <= 0.55, f"Split {split_name} has unexpected class ratio: {ratio:.3f}"
        assert n_pos == n_neg, f"Split {split_name} is not exactly balanced: pos={n_pos}, neg={n_neg}"


def test_mixed_clusters_handled_indivisibly():
    """Verify mixed clusters were investigated and kept indivisible within a single split."""
    mixed_csv = DATA_SPLITS_DIR / "mixed_clusters_investigation.csv"
    assert mixed_csv.exists()
    df_mixed = pd.read_csv(mixed_csv)
    assert len(df_mixed) > 0, "No mixed clusters found to investigate!"
    
    # Check that mixed clusters are kept in a single split
    df_seq = pd.read_csv(DATA_SPLITS_DIR / "sequence_cluster_assignments.csv")
    mixed_cluster_ids = set(df_mixed["cluster_id"])
    for m_id in mixed_cluster_ids:
        sub = df_seq[df_seq["cluster_id"] == m_id]
        assert sub["split"].nunique() == 1, f"Mixed cluster {m_id} was split across partitions!"
