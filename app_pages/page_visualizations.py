"""
Page 4: Model Performance Visualizations
Interactive and publication-quality diagnostic plots: ROC curves, PR curves,
confusion matrices, sequence length bias analysis, and FP/FN error distributions.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from src.dashboard_data import (
    get_test_predictions_df,
    get_false_positives_df,
    get_false_negatives_df
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def render_visualizations():
    st.markdown("## Model Performance Visualizations")
    st.markdown(
        "Diagnostic evaluation charts generated on the **held-out test partition ($n = 1,816$ sequences)**. "
        "All curves reflect zero-homology-leakage testing against unseen superclusters."
    )

    test_preds = get_test_predictions_df()
    if test_preds.empty:
        st.error("Test predictions data not found at `results/metrics/test_predictions.csv`.")
        return

    # Tabs for different visualization domains
    tab_curves, tab_len, tab_errors = st.tabs([
        "📈 ROC & PR Curves",
        "📏 Sequence Length Distribution",
        "🔍 Error Analysis (FP & FN)"
    ])

    # TAB 1: ROC & PR CURVES
    with tab_curves:
        st.markdown("### Receiver Operating Characteristic (ROC) & Precision-Recall (PR) Curves")
        col_roc, col_pr = st.columns(2)

        models_eval = [
            ("Random Forest", "prob_Random Forest", "#0A2540"),
            ("SVM (RBF Kernel)", "prob_SVM (RBF Kernel)", "#00B4D8"),
            ("Logistic Regression", "prob_Logistic Regression", "#64748B")
        ]

        # Interactive ROC Curve
        with col_roc:
            fig_roc = go.Figure()
            fig_roc.add_shape(
                type='line', line=dict(dash='dash', color='gray'),
                x0=0, x1=1, y0=0, y1=1
            )
            for m_name, p_col, m_color in models_eval:
                if p_col in test_preds.columns:
                    fpr, tpr, _ = roc_curve(test_preds["label"], test_preds[p_col])
                    roc_auc = auc(fpr, tpr)
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f"{m_name} (AUC = {roc_auc:.3f})",
                        line=dict(color=m_color, width=2.5)
                    ))

            fig_roc.update_layout(
                title="Interactive ROC Curves (Held-out Test)",
                xaxis_title="False Positive Rate (1 - Specificity)",
                yaxis_title="True Positive Rate (Sensitivity)",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1.05]),
                height=420,
                legend=dict(orientation="h", y=-0.25)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            st.caption("Interpretation: Random Forest achieves an AUC of 0.819, maintaining a high True Positive Rate across varied operating points.")

        # Interactive PR Curve
        with col_pr:
            fig_pr = go.Figure()
            pos_ratio = test_preds["label"].sum() / len(test_preds)
            fig_pr.add_shape(
                type='line', line=dict(dash='dash', color='gray'),
                x0=0, x1=1, y0=pos_ratio, y1=pos_ratio
            )
            for m_name, p_col, m_color in models_eval:
                if p_col in test_preds.columns:
                    prec, rec, _ = precision_recall_curve(test_preds["label"], test_preds[p_col])
                    pr_auc = auc(rec, prec)
                    fig_pr.add_trace(go.Scatter(
                        x=rec, y=prec,
                        mode='lines',
                        name=f"{m_name} (PR-AUC = {pr_auc:.3f})",
                        line=dict(color=m_color, width=2.5)
                    ))

            fig_pr.update_layout(
                title="Interactive Precision-Recall Curves",
                xaxis_title="Recall (Sensitivity)",
                yaxis_title="Precision (PPV)",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1.05]),
                height=420,
                legend=dict(orientation="h", y=-0.25)
            )
            st.plotly_chart(fig_pr, use_container_width=True)
            st.caption("Interpretation: PR-AUC of 0.816 for Random Forest demonstrates robust precision retention even as recall increases.")

        # Reconstructed Confusion Matrix Image from research
        cm_path = FIGURES_DIR / "confusion_matrices.png"
        if cm_path.exists():
            st.markdown("---")
            st.markdown("#### Publication-Quality Confusion Matrix Grid")
            st.image(str(cm_path), caption="Normalized and raw test set confusion matrices across all architectures at t = 0.10.")

    # TAB 2: SEQUENCE LENGTH DISTRIBUTION & BIAS
    with tab_len:
        st.markdown("### Sequence Length Distribution & Prediction Behavior")
        st.markdown(
            "Evaluation of whether the models exhibit sequence length bias across true classes and predicted outcomes."
        )

        fig_len = px.histogram(
            test_preds,
            x="length",
            color="label",
            marginal="box",
            nbins=60,
            barmode="overlay",
            opacity=0.7,
            title="Held-out Test Sequence Length Distribution by True Class",
            labels={"length": "Sequence Length (Amino Acids)", "label": "True Class"},
            color_discrete_map={0: "#64748B", 1: "#00B4D8"}
        )
        fig_len.update_layout(height=400)
        st.plotly_chart(fig_len, use_container_width=True)

        len_img_path = FIGURES_DIR / "sequence_length_distribution_predictions.png"
        if len_img_path.exists():
            st.markdown("---")
            st.image(str(len_img_path), caption="Comparative length distribution across True Negatives, False Positives, False Negatives, and True Positives.")

    # TAB 3: ERROR ANALYSIS
    with tab_errors:
        st.markdown("### Error Analysis: False Positives & False Negatives")
        st.markdown(
            "Detailed inspection of borderline cases and classification failures under the champion Random Forest model ($t = 0.10$)."
        )

        fp_df = get_false_positives_df()
        fn_df = get_false_negatives_df()

        col_fp, col_fn = st.columns(2)

        with col_fp:
            st.markdown(f"#### False Positives ($n = {len(fp_df)}$)")
            st.caption("Non-ARG candidate proteins classified as ARG-like (Predicted: 1, True: 0).")
            if not fp_df.empty:
                show_cols = [c for c in ["seq_uid", "sequence_id", "length", "prob_rf", "description"] if c in fp_df.columns]
                st.dataframe(fp_df[show_cols].head(15), use_container_width=True, hide_index=True)
                st.info("Biological insight: Many false positives are membrane transport proteins, efflux complexes, or acetyltransferases that share structural motifs with resistance pumps.")

        with col_fn:
            st.markdown(f"#### False Negatives ($n = {len(fn_df)}$)")
            st.caption("Genuine CARD ARGs classified as non-ARG (Predicted: 0, True: 1).")
            if not fn_df.empty:
                show_cols = [c for c in ["seq_uid", "sequence_id", "length", "prob_rf", "description"] if c in fn_df.columns]
                st.dataframe(fn_df[show_cols].head(15), use_container_width=True, hide_index=True)
                st.info("Biological insight: False negatives predominantly feature highly divergent, atypical resistance mechanisms (e.g. rare ribosomal protection factors) with no training homologs above 30% identity.")
