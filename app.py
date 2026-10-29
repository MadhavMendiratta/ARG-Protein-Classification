"""
Application: Antibiotic Resistance Protein Classification Dashboard
Entry Point: app.py
Framework: Streamlit with Plotly, Scikit-learn, Biopython, Pandas, and NumPy.
Objective: Complete, polished interactive research dashboard for the ARG vs Candidate Non-ARG project.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in Python module path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import ModelArtifactManager
from app_pages.page_overview import render_overview
from app_pages.page_predictor import render_predictor
from app_pages.page_comparison import render_comparison
from app_pages.page_visualizations import render_visualizations
from app_pages.page_features import render_features
from app_pages.page_dataset import render_dataset
from app_pages.page_methodology import render_methodology
from app_pages.page_about import render_about


# Page Configuration
st.set_page_config(
    page_title="ARG Classifier | Bioinformatics ML Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Navy/Teal Bioinformatics Design System Styling
CUSTOM_CSS = """
<style>
    /* Global Typography & Palette */
    :root {
        --primary-navy: #0A192F;
        --secondary-navy: #172A45;
        --accent-teal: #00B4D8;
        --dark-teal: #0077B6;
        --light-teal: #E0F2FE;
        --card-bg: #F8FAFC;
        --text-dark: #0F172A;
    }

    /* Metric cards enhancement */
    [data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #475569;
        font-size: 0.85rem;
    }
    [data-testid="stMetricValue"] {
        color: #0A2540;
        font-weight: 700;
    }

    /* Sidebar custom styling */
    [data-testid="stSidebar"] {
        background-color: #0A192F;
    }
    [data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] .stRadio > div {
        background-color: #172A45;
        border-radius: 8px;
        padding: 8px;
    }

    /* Headers styling */
    h1, h2, h3 {
        color: #0A2540;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        padding: 8px 16px;
        font-weight: 600;
    }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background-color: #0077B6;
        border-color: #0077B6;
        color: white !important;
        font-weight: 600;
        border-radius: 6px;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #023E8A;
        border-color: #023E8A;
        box-shadow: 0 4px 12px rgba(0, 119, 182, 0.3);
    }

    /* Footer styling */
    .dashboard-footer {
        margin-top: 50px;
        padding: 20px 0 10px 0;
        border-top: 1px solid #E2E8F0;
        text-align: center;
        font-size: 0.85rem;
        color: #64748B;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main():
    # Load artifact manager in cache
    try:
        mgr = ModelArtifactManager.get_instance()
        loaded_models = list(mgr.models.keys())
    except Exception as e:
        st.error(f"Failed to initialize machine learning artifacts: {str(e)}")
        loaded_models = []

    # Sidebar Navigation & Context
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0 15px 0; text-align: left;">
                <span style="background-color: #00B4D8; color: #0A192F !important; font-size: 11px; font-weight: 800; padding: 4px 8px; border-radius: 4px; letter-spacing: 0.5px;">RESEARCH DASHBOARD</span>
                <h2 style="color: #FFFFFF !important; margin: 8px 0 2px 0; font-size: 1.4rem;">🧬 ARG Classifier</h2>
                <p style="color: #94A3B8 !important; font-size: 0.8rem; margin: 0;">Machine Learning Sequence Analytics</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        # 8-Page Navigation Selector
        pages = {
            "📊 1. Project Overview": render_overview,
            "🔬 2. Protein Predictor": render_predictor,
            "⚖️ 3. Model Comparison": render_comparison,
            "📈 4. Performance Visualizations": render_visualizations,
            "🧩 5. Feature Analysis": render_features,
            "🗃️ 6. Dataset Explorer": render_dataset,
            "📋 7. Methodology & Limitations": render_methodology,
            "ℹ️ 8. About & Governance": render_about
        }

        selected_page = st.radio(
            "Dashboard Navigation",
            options=list(pages.keys()),
            label_visibility="collapsed"
        )

        st.markdown("---")

        # System Status in Sidebar
        st.markdown("#### System & Artifact Status")
        st.markdown(
            f"""
            - **Models Loaded:** `{len(loaded_models)} / 3`
            - **Primary Champion:** `Random Forest`
            - **Feature Dimension:** `428 features`
            - **Decision Threshold:** `t = 0.10`
            - **Offline Mode:** `Active (Local)`
            """
        )

        st.markdown("---")
        st.caption("College Bioinformatics Data Analytics Project\nVersion 1.0 • Academic Prototype")

    # Render Active Page
    render_func = pages[selected_page]
    render_func()

    # Clean Academic Footer
    st.markdown(
        """
        <div class="dashboard-footer">
            Antibiotic Resistance Gene (ARG) vs Non-ARG Protein Classification Dashboard<br>
            Powered by Python 3.11 • Scikit-learn • Biopython • Streamlit • 100% Local Inference
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
