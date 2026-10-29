"""
Page 3: Model Comparison
Comprehensive comparative evaluation of Random Forest, Logistic Regression, and RBF-SVM
using verified metrics from held-out test evaluation.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.dashboard_data import get_model_comparison_df


def render_comparison():
    st.markdown("## Machine Learning Model Comparison")
    st.markdown(
        "Direct benchmark comparison of our three trained machine learning architectures evaluated on "
        "the **held-out test set ($n = 1,816$ sequences)** at the validation-calibrated threshold ($t = 0.10$)."
    )

    df_comp = get_model_comparison_df()
    if df_comp.empty:
        st.error("Model comparison metrics missing from `results/metrics/model_comparison.csv`.")
        return

    # Overview Metrics Table
    st.markdown("### 1. Verified Benchmark Metrics")
    display_df = df_comp.copy()
    metric_cols = ["Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1-Score", "ROC-AUC", "PR-AUC", "MCC"]
    
    # Format floats nicely for UI
    formatted_df = display_df.copy()
    for col in metric_cols:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(formatted_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 2. Interactive Comparison Visualizations
    st.markdown("### 2. Discrimination & Classification Performance Charts")
    
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # Grouped bar chart for AUC and Correlation metrics
        auc_mcc_metrics = ["ROC-AUC", "PR-AUC", "MCC", "F1-Score"]
        melted_auc = df_comp.melt(
            id_vars=["Model"],
            value_vars=auc_mcc_metrics,
            var_name="Metric",
            value_name="Score"
        )
        fig_auc = px.bar(
            melted_auc,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            text="Score",
            title="Overall Discrimination & Correlation (Higher is Better)",
            color_discrete_map={
                "Random Forest": "#0A2540",
                "SVM (RBF Kernel)": "#00B4D8",
                "Logistic Regression": "#64748B"
            }
        )
        fig_auc.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_auc.update_layout(yaxis=dict(range=[0, 1.05]), height=380, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_auc, use_container_width=True)

    with col_chart2:
        # Trade-off chart: Sensitivity vs Specificity vs Precision
        trade_metrics = ["Recall (Sensitivity)", "Specificity", "Precision", "Accuracy"]
        melted_trade = df_comp.melt(
            id_vars=["Model"],
            value_vars=trade_metrics,
            var_name="Metric",
            value_name="Score"
        )
        fig_trade = px.bar(
            melted_trade,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            text="Score",
            title="Operational Trade-offs: Recall vs Specificity vs Precision",
            color_discrete_map={
                "Random Forest": "#0A2540",
                "SVM (RBF Kernel)": "#00B4D8",
                "Logistic Regression": "#64748B"
            }
        )
        fig_trade.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_trade.update_layout(yaxis=dict(range=[0, 1.15]), height=380, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_trade, use_container_width=True)

    st.markdown("---")

    # 3. Confusion Matrix Breakdown
    st.markdown("### 3. Confusion Matrix Partition Distribution")
    cm_cols = st.columns(3)
    
    for idx, (_, row) in enumerate(df_comp.iterrows()):
        model_name = row["Model"]
        tp, tn, fp, fn = int(row["TP"]), int(row["TN"]), int(row["FP"]), int(row["FN"])
        with cm_cols[idx]:
            st.markdown(f"#### {model_name}")
            cm_matrix = [[tn, fp], [fn, tp]]
            fig_cm = px.imshow(
                cm_matrix,
                labels=dict(x="Predicted Class", y="Actual Label", color="Count"),
                x=["Non-ARG (0)", "ARG (1)"],
                y=["Non-ARG (0)", "ARG (1)"],
                text_auto=True,
                color_continuous_scale=["#F8FAFC", "#00B4D8", "#0A2540"],
                aspect="auto"
            )
            fig_cm.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=20, b=10),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            st.caption(f"**TP:** {tp} | **TN:** {tn} | **FP:** {fp} | **FN:** {fn}")

    st.markdown("---")

    # 4. In-Depth Scientific Analysis & Rationale
    st.markdown("### 4. Architectural Analysis & Selection Rationale")
    
    st.markdown(r"""
    #### Why Random Forest is the Designated Primary Champion
    1. **Superior Overall Discrimination:**
       Random Forest achieves the highest **ROC-AUC (0.8191)** and **PR-AUC (0.8161)** across all architectures on unseen homology clusters.
    2. **Balanced Sensitivity Under Homology Shift:**
       Because test sequences originate from disjoint homology superclusters ($<36.2\%$ identity), many ARG targets possess subtle 
       compositional signals. Random Forest captures **528 of 908 resistance proteins (Recall = 58.15%)** while maintaining high **Precision (80.37%)** 
       and an **MCC of 0.4572**.
    3. **Nonlinear Interaction Modeling:**
       Dipeptide composition (DPC) generates 400 features with complex pairwise interactions (e.g. active site motifs, charge distributions). 
       Decision trees naturally capture these hierarchical interactions without overfitting when regularized (`max_depth=20`, `min_samples_split=5`).

    #### RBF Support Vector Machine (SVM) Observations
    - **Ultra-Conservative High-Precision Filter:**
      SVM achieved **95.36% Precision** and **99.23% Specificity** with only 7 false positives out of 908 negative sequences.
    - **Lower Recall on Distant Homologs:**
      Its sensitivity on distant holdouts is lower (Recall = 15.86%), making it suitable as a high-confidence confirmatory screen rather than 
      a sensitive discovery tool.

    #### Logistic Regression Baseline Observations
    - As a linear model, L2-regularized logistic regression attained an ROC-AUC of 0.6299. It serves as a vital scientific baseline proving that 
      nonlinear interactions captured by tree ensembles are indispensable for sequence-based resistance classification.
    """)
