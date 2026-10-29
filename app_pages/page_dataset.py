"""
Page 6: Dataset Explorer
Detailed exploration of sequence counts, length distributions, provenance metadata,
curation exclusion audit, and interactive dataset record browser.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.dashboard_data import (
    get_validation_summary,
    get_split_report,
    get_exclusion_log_df,
    get_sequence_cluster_assignments_df
)


def render_dataset():
    st.markdown("## Dataset Explorer & Provenance Audit")
    st.markdown(
        "Complete transparency into the curated dataset of **12,108 protein sequences**, "
        "including database provenance, filtering exclusions, length distributions, and cluster allocations."
    )

    val_sum = get_validation_summary()
    split_rep = get_split_report()
    excl_df = get_exclusion_log_df()
    meta_df = get_sequence_cluster_assignments_df()

    # 1. Headline Dataset Overview
    st.markdown("### 1. Curated Dataset Composition")
    
    c1, c2, c3, c4 = st.columns(4)
    total_seqs = val_sum.get("final_dataset", {}).get("total_sequences", 12108)
    pos_seqs = val_sum.get("final_dataset", {}).get("positive_args", 6054)
    neg_seqs = val_sum.get("final_dataset", {}).get("negative_non_args", 6054)
    excl_total = val_sum.get("exclusions", {}).get("total_exclusions", 294)

    c1.metric("Final Dataset Size", f"{total_seqs:,}", "1:1 Balanced Ratio")
    c2.metric("Positive ARGs (CARD)", f"{pos_seqs:,}", "v4.0.2 Reference")
    c3.metric("Candidate Non-ARGs", f"{neg_seqs:,}", "Swiss-Prot 2024_06")
    c4.metric("Exclusions Filtered", f"{excl_total}", "Audit Documented")

    st.markdown("---")

    # 2. Sequence Length Distribution Analysis
    st.markdown("### 2. Sequence Length Distributions & Statistics")
    
    col_len_stats, col_len_chart = st.columns([1, 2])

    with col_len_stats:
        st.markdown("#### Summary Statistics (Amino Acids)")
        len_stats = val_sum.get("length_statistics", {})
        pos_len = len_stats.get("positive_args", {})
        neg_len = len_stats.get("negative_non_args", {})
        
        len_table_data = [
            {"Metric": "Count", "ARGs": f"{pos_len.get('count', 6054):,}", "Non-ARGs": f"{neg_len.get('count', 6054):,}"},
            {"Metric": "Min Length", "ARGs": f"{pos_len.get('min', 53)} aa", "Non-ARGs": f"{neg_len.get('min', 50)} aa"},
            {"Metric": "Q1 (25th %)", "ARGs": f"{pos_len.get('q1', 273):.0f} aa", "Non-ARGs": f"{neg_len.get('q1', 253):.0f} aa"},
            {"Metric": "Median", "ARGs": f"{pos_len.get('median', 290):.0f} aa", "Non-ARGs": f"{neg_len.get('median', 358):.0f} aa"},
            {"Metric": "Mean", "ARGs": f"{pos_len.get('mean', 320.1):.1f} aa", "Non-ARGs": f"{neg_len.get('mean', 398.9):.1f} aa"},
            {"Metric": "Q3 (75th %)", "ARGs": f"{pos_len.get('q3', 381):.0f} aa", "Non-ARGs": f"{neg_len.get('q3', 485):.0f} aa"},
            {"Metric": "Max Length", "ARGs": f"{pos_len.get('max', 1452)} aa", "Non-ARGs": f"{neg_len.get('max', 1500)} aa"},
            {"Metric": "Std Dev", "ARGs": f"{pos_len.get('std', 101.0):.1f}", "Non-ARGs": f"{neg_len.get('std', 222.8):.1f}"}
        ]
        st.dataframe(len_table_data, use_container_width=True, hide_index=True)

    with col_len_chart:
        if not meta_df.empty:
            fig_len_hist = px.histogram(
                meta_df,
                x="length",
                color="label",
                marginal="box",
                nbins=60,
                barmode="overlay",
                opacity=0.7,
                title="Sequence Length Distribution (Positive ARGs vs Candidate Non-ARGs)",
                labels={"length": "Length (aa)", "label": "Class Label"},
                color_discrete_map={0: "#64748B", 1: "#00B4D8"}
            )
            fig_len_hist.update_layout(height=360, legend=dict(title="Class (0=Non-ARG, 1=ARG)"))
            st.plotly_chart(fig_len_hist, use_container_width=True)

    st.markdown("---")

    # 3. Data Cleaning & Exclusion Audit Log
    st.markdown("### 3. Data Curation & Exclusion Audit Log")
    st.markdown("""
    To ensure pristine data quality and prevent machine learning artifacts, a multi-stage filtering protocol 
    was applied to raw downloads from CARD and Swiss-Prot. Sequences violating canonical biophysical constraints 
    were recorded in the project's permanent exclusion log.
    """)

    excl_summary = val_sum.get("exclusions", {})
    e_col1, e_col2, e_col3 = st.columns(3)
    e_col1.metric("CARD Positives Excluded", excl_summary.get("invalid_amino_acids_or_length_pos", 5), "Non-canonical / Outliers")
    e_col2.metric("Swiss-Prot Negatives Excluded", excl_summary.get("invalid_amino_acids_or_length_neg", 129), "Length bounds (<50 or >1500)")
    e_col3.metric("Duplicates & Conflicts Excluded", excl_summary.get("duplicates_and_label_conflicts", 160), "Identical sequences")

    if not excl_df.empty:
        with st.expander(f"📋 Inspect Exclusion Log Records ({len(excl_df)} entries)"):
            st.dataframe(excl_df.head(50), use_container_width=True, hide_index=True)
            st.caption("Displaying top 50 exclusions. Filter reasons include non-canonical characters (e.g. B, Z, X), sequence length <50 aa or >1500 aa, and identical sequence conflicts.")

    st.markdown("---")

    # 4. Interactive Dataset Record Browser
    st.markdown("### 4. Interactive Sequence Record Browser")
    st.markdown("Search and filter the curated metadata across all 12,108 sequences.")

    if not meta_df.empty:
        f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
        split_filter = f_col1.selectbox("Filter by Partition Split:", ["All Splits", "train", "val", "test"])
        class_filter = f_col2.selectbox("Filter by Class Label:", ["All Classes", "1 - Positive ARG", "0 - Candidate Non-ARG"])
        search_query = f_col3.text_input("Search ID or Description (e.g., `beta-lactamase`, `efflux`, `SEQ_000009`):", "")

        filtered_df = meta_df.copy()
        if split_filter != "All Splits":
            filtered_df = filtered_df[filtered_df["split"] == split_filter]
        if class_filter == "1 - Positive ARG":
            filtered_df = filtered_df[filtered_df["label"] == 1]
        elif class_filter == "0 - Candidate Non-ARG":
            filtered_df = filtered_df[filtered_df["label"] == 0]

        if search_query.strip():
            q = search_query.strip().lower()
            filtered_df = filtered_df[
                filtered_df["seq_uid"].str.lower().str.contains(q, na=False) |
                filtered_df["sequence_id"].str.lower().str.contains(q, na=False) |
                filtered_df["description"].str.lower().str.contains(q, na=False)
            ]

        row_limit = st.select_slider("Rows to display:", options=[25, 50, 100, 250], value=50)

        display_cols = ["seq_uid", "sequence_id", "cluster_id", "label", "length", "source_db", "split", "description"]
        st.dataframe(
            filtered_df[display_cols].head(row_limit),
            use_container_width=True,
            hide_index=True
        )
        st.caption(f"Showing {min(row_limit, len(filtered_df))} of {len(filtered_df):,} matching sequences.")

    st.info("""
    **Candidate Negatives Nomenclature:**
    The non-ARG negative class consists of candidate negative proteins retrieved from UniProtKB/Swiss-Prot 
    representing verified bacterial housekeeping genes (metabolic enzymes, ribosomal subunits, structural elements). 
    They are defined as *candidate negatives* because unannotated resistance functions cannot be 100% ruled out 
    without exhaustive wet-lab antibiotic susceptibility testing.
    """)
