"""
Page 5: Feature Analysis
Breakdown of the 428 numerical features (AAC, DPC, Physicochemical),
Random Forest Gini importances, Logistic Regression coefficients, and searchable feature catalog.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.dashboard_data import (
    get_feature_summary,
    get_rf_feature_importances,
    get_logreg_coefficients
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def render_features():
    st.markdown("## Feature Engineering & Interpretability Analysis")
    st.markdown(
        "Detailed examination of the **428 sequence-derived numerical features** used to train the models: "
        "Amino Acid Composition (AAC), Dipeptide Composition (DPC), and Physicochemical Properties (PC)."
    )

    feat_sum = get_feature_summary()
    rf_imp = get_rf_feature_importances()
    logreg_coef = get_logreg_coefficients()

    # 1. Feature Group Breakdown
    st.markdown("### 1. Feature Space Architecture (428 Descriptors)")
    
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    col_metric1.metric("Total Feature Vector", "428 dimensions")
    col_metric2.metric("AAC (Amino Acid Comp)", "20 features", "Canonical AAs")
    col_metric3.metric("DPC (Dipeptide Comp)", "400 features", "20 x 20 Matrix")
    col_metric4.metric("Physicochemical Descriptors", "8 features", "Structural / Biophysical")

    col_pie, col_desc = st.columns([1, 2])
    with col_pie:
        fig_groups = go.Figure(data=[go.Pie(
            labels=["Dipeptide Comp (DPC)", "Amino Acid Comp (AAC)", "Physicochemical (PC)"],
            values=[400, 20, 8],
            hole=0.5,
            marker=dict(colors=["#0A2540", "#00B4D8", "#64DFDF"]),
            textinfo="label+value"
        )])
        fig_groups.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False
        )
        st.plotly_chart(fig_groups, use_container_width=True)

    with col_desc:
        st.markdown(r"""
        **How Global Composition Encodes Protein Sequences:**
        - **AAC (20 features):** Normalized frequency of each of the 20 canonical amino acids. Captures global proteomic biases (e.g. enrichment of basic or aromatic residues).
        - **DPC (400 features):** Normalized frequencies of all possible adjacent 2-residue permutations ($20 \\times 20$). Encodes local secondary structure propensities, dipeptide motifs, and active site sequence contexts without requiring structural alignment.
        - **Physicochemical Properties (8 features):** Sequence length, molecular weight, isoelectric point (pI), GRAVY (grand average of hydropathicity), aromaticity, instability index, aliphatic index, and net theoretical charge at physiological pH 7.0.
        """)

    st.markdown("---")

    # 2. Random Forest Top Feature Importances
    st.markdown("### 2. Random Forest Top Feature Importances (Gini Impurity)")
    
    if not rf_imp.empty:
        top_k = st.slider("Select Number of Top Features to Display:", min_value=10, max_value=50, value=25, step=5)
        top_rf = rf_imp.head(top_k).copy()
        
        # Add feature group tag
        def get_group(f_name):
            if f_name.startswith("AAC_"):
                return "Amino Acid Composition"
            elif f_name.startswith("DPC_"):
                return "Dipeptide Composition"
            elif f_name.startswith("PC_"):
                return "Physicochemical"
            return "Other"

        top_rf["Group"] = top_rf["feature"].apply(get_group)

        fig_bar = px.bar(
            top_rf.sort_values("gini_importance", ascending=True),
            x="gini_importance",
            y="feature",
            color="Group",
            orientation="h",
            title=f"Top {top_k} Features Ranked by Mean Decrease in Impurity (Gini)",
            labels={"gini_importance": "Gini Importance Score", "feature": "Feature Descriptor"},
            color_discrete_map={
                "Dipeptide Composition": "#0A2540",
                "Amino Acid Composition": "#00B4D8",
                "Physicochemical": "#64DFDF"
            },
            height=max(450, top_k * 18)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        rf_fig_path = FIGURES_DIR / "feature_importance_top25.png"
        if rf_fig_path.exists():
            with st.expander("🖼️ View Research Publication Figure (Top 25 Feature Importances)"):
                st.image(str(rf_fig_path), caption="Top 25 features identified during training model evaluation.")

    st.markdown("---")

    # 3. Logistic Regression Coefficients
    st.markdown("### 3. Linear Model Feature Coefficients (Logistic Regression)")
    if not logreg_coef.empty:
        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.markdown("#### Top Features Driving ARG Prediction (+)")
            pos_coef = logreg_coef.sort_values("coefficient", ascending=False).head(10)
            st.dataframe(
                pos_coef[["feature", "coefficient"]],
                use_container_width=True,
                hide_index=True,
                column_config={"coefficient": st.column_config.NumberColumn("Positive Weight", format="+%.4f")}
            )
        with col_neg:
            st.markdown("#### Top Features Driving Non-ARG Prediction (-)")
            neg_coef = logreg_coef.sort_values("coefficient", ascending=True).head(10)
            st.dataframe(
                neg_coef[["feature", "coefficient"]],
                use_container_width=True,
                hide_index=True,
                column_config={"coefficient": st.column_config.NumberColumn("Negative Weight", format="%.4f")}
            )

    st.markdown("---")

    # 4. Searchable 428-Feature Interactive Catalog
    st.markdown("### 4. Searchable Feature Catalog (All 428 Features)")
    search_q = st.text_input("Search feature by name (e.g. `DPC_W`, `AAC_H`, `net_charge`):", "")

    if not rf_imp.empty:
        full_catalog = rf_imp.copy()
        full_catalog["Group"] = full_catalog["feature"].apply(get_group)
        if search_q.strip():
            full_catalog = full_catalog[full_catalog["feature"].str.contains(search_q.strip(), case=False)]

        st.dataframe(
            full_catalog,
            use_container_width=True,
            hide_index=True,
            column_config={
                "gini_importance": st.column_config.NumberColumn("Gini Importance", format="%.6f")
            }
        )
        st.caption(f"Showing {len(full_catalog)} feature(s).")

    # 5. Scientific Disclaimer on Feature Importance
    st.warning("""
    **Crucial Methodological Caveat:**
    Feature importance metrics (Gini impurity and regression weights) identify statistical associations within the trained model 
    and dataset. They **do not prove biological causality** or imply that a specific dipeptide directly causes antibiotic resistance. 
    Rather, these features reflect characteristic global compositional signatures of resistance enzyme superfamilies 
    (e.g., beta-lactamases, aminoglycoside-modifying enzymes, tetracycline efflux pumps).
    """)
