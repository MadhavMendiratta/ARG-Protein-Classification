"""
Page 2: Protein Sequence Predictor
Interactive inference tool supporting single sequence pasting and multi-record FASTA uploads.
Uses verified saved models, locked feature extraction, and validation-calibrated thresholds.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.inference import (
    ModelArtifactManager,
    clean_and_validate_sequence,
    parse_fasta_multi,
    predict_single_sequence,
    predict_batch_records
)

SAMPLE_SEQUENCES = {
    "TEM-126 Beta-Lactamase (Positive ARG)": (
        ">gb|AAT45742.1|ARO:3000988|TEM-126 beta-lactamase [Escherichia coli]\n"
        "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLL"
        "CGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTA"
        "FLHNMGDHVTRLDRWEPELNEAIPNDERETTTPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSA"
        "LPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
    ),
    "VanX Dipeptidase (Positive ARG)": (
        ">gb|AAB05628.1|ARO:3002950|vanX_in_vanB_cluster [Enterococcus faecalis]\n"
        "MENGFLFLDEMLHGVRWDAKYATWDNFTGKPVDGYEVNRIIGTKAVALALKEALNDKGYKVEAIEEPGKGKYYV"
        "GDQLLLYGTTKDGKQVFLKADGEFFDVVDGKTWTYQLDKYFEQKKDKTLLVDGGKTLYLSGEAALKDTVQKWTE"
        "ESGKRIEAIFPGYGVSEVEITQKDGDVYLLPSTKDGYELSLFKEGGFEKITIDGKPVTFEVDGEEFKYVI"
    ),
    "Cold Shock Protein CspB (Candidate Non-ARG)": (
        ">sp|P32081|CSPB_BACSU Cold shock protein CspB [Bacillus subtilis]\n"
        "MLEGKVKWFNSEKGFGFIEVEGQDDVFVHFSAIQGEGFKTLEEGQAVSFEIVEGNRGPQAANVTKEA"
    ),
    "ATP Synthase Subunit c (Candidate Non-ARG)": (
        ">sp|P68699|ATPL_ECOLI ATP synthase subunit c [Escherichia coli]\n"
        "MENLNMDLLYMAAAVMMGLAAIGAAIGIGILGGKFLEGAARQPDLIPLLRTQFFIVMGLVDAIPMIAVGLGLYVMFAVA"
    )
}


def render_predictor():
    st.markdown("## Protein Sequence Predictor")
    st.markdown(
        "Run genuine machine learning inference on target protein sequences using our trained, "
        "reproducible models. Sequences are standardized using the frozen training scaler and evaluated "
        "against the validation-calibrated threshold ($t = 0.10$)."
    )

    mgr = ModelArtifactManager.get_instance()
    avail_models = mgr.get_available_models()

    if not avail_models:
        st.error("No trained models could be loaded from `results/models/`. Please check disk artifacts.")
        return

    # Sidebar / Header configuration controls
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        # Default to Random Forest if available
        default_idx = list(avail_models.keys()).index("random_forest") if "random_forest" in avail_models else 0
        selected_model_key = st.selectbox(
            "Select Inference Model:",
            options=list(avail_models.keys()),
            format_func=lambda k: f"{avail_models[k]} {'(Primary Champion)' if k == 'random_forest' else ''}",
            index=default_idx,
            help="Random Forest demonstrated highest overall discrimination (ROC-AUC 0.8191) on held-out clusters."
        )
    with ctrl_col2:
        model_info = mgr.models[selected_model_key]
        st.markdown(f"**Selected Model:** `{model_info['display_name']}`")
        st.markdown(f"**Applied Threshold:** `{model_info['threshold']:.2f}` (Validation-Tuned)")

    st.markdown("---")

    # Input Method Tabs
    tab_paste, tab_upload = st.tabs(["Method A: Paste Sequence", "Method B: Upload FASTA File"])

    # State management for paste text
    if "paste_input_val" not in st.session_state:
        st.session_state.paste_input_val = ""

    # TAB 1: PASTE SEQUENCE
    with tab_paste:
        st.markdown("#### Paste Amino Acid Sequence or FASTA Record")
        
        # Quick sample loading buttons
        st.markdown("**Load Verified Test Sample:**")
        sample_cols = st.columns(len(SAMPLE_SEQUENCES))
        for idx, (s_name, s_seq) in enumerate(SAMPLE_SEQUENCES.items()):
            if sample_cols[idx].button(s_name, key=f"btn_sample_{idx}", use_container_width=True):
                st.session_state.paste_input_val = s_seq

        raw_text = st.text_area(
            "Enter Sequence (Plain or FASTA format with optional `>` header):",
            value=st.session_state.paste_input_val,
            height=160,
            placeholder=">header_optional\nMKAYFIAILTLFTCIATVVRAQQMSELENRIDSLLNGKKATVGIAVWTDK..."
        )

        btn_c1, btn_c2 = st.columns([1, 4])
        predict_single_btn = btn_c1.button("Predict Sequence", type="primary", use_container_width=True)
        if btn_c2.button("Clear Input", use_container_width=False):
            st.session_state.paste_input_val = ""
            st.rerun()

        if predict_single_btn:
            if not raw_text.strip():
                st.warning("Please enter or paste a sequence before running prediction.")
            else:
                with st.spinner("Extracting 428 features, scaling, and computing inference..."):
                    # Check if input has multiple records or single
                    records = parse_fasta_multi(raw_text)
                    if len(records) > 1:
                        st.info(f"Detected {len(records)} sequences in pasted text. Processing as batch...")
                        res_df = predict_batch_records(records, model_key=selected_model_key)
                        _render_batch_results(res_df)
                    elif len(records) == 1:
                        r = records[0]
                        if not r["is_valid"]:
                            st.error(f"**Validation Error:** {r['msg']}")
                        else:
                            try:
                                pred = predict_single_sequence(r["seq"], model_key=selected_model_key)
                                _render_single_prediction_result(
                                    seq_id=r["seq_id"],
                                    description=r["description"],
                                    seq=r["seq"],
                                    pred=pred,
                                    msg=r["msg"]
                                )
                            except Exception as e:
                                st.error(f"Inference pipeline execution error: {str(e)}")

    # TAB 2: UPLOAD FASTA FILE
    with tab_upload:
        st.markdown("#### Upload FASTA File (`.fasta`, `.fa`, `.faa`)")
        st.caption("Supports single or multi-sequence files. Each sequence receives its own independent prediction.")
        uploaded_file = st.file_uploader(
            "Choose a FASTA file from your local machine:",
            type=["fasta", "fa", "faa"],
            help="All processing runs entirely on your local machine. No sequences leave your computer."
        )

        if uploaded_file is not None:
            file_contents = uploaded_file.getvalue().decode("utf-8", errors="replace")
            records = parse_fasta_multi(file_contents)
            st.success(f"Parsed **{len(records)}** record(s) from `{uploaded_file.name}`.")

            if st.button("Run Batch Inference on Uploaded Records", type="primary"):
                with st.spinner(f"Computing predictions for {len(records)} sequence(s)..."):
                    res_df = predict_batch_records(records, model_key=selected_model_key)
                    _render_batch_results(res_df)


def _render_single_prediction_result(seq_id: str, description: str, seq: str, pred: dict, msg: str):
    """Render a polished result card and score visualization for a single sequence."""
    is_arg = pred["is_arg"]
    score = pred["score"]
    threshold = pred["threshold_used"]
    score_type = pred["score_type"]
    label_text = pred["predicted_text"]
    length = pred["sequence_length"]

    st.markdown("---")
    st.markdown("### Prediction Result")

    # Header card with distinct status color
    if is_arg:
        st.markdown(
            f"""
            <div style="background-color: #E6FFFA; border-left: 6px solid #00B4D8; padding: 18px; border-radius: 6px; margin-bottom: 20px;">
                <h3 style="color: #0077B6; margin: 0 0 8px 0;">🛡️ {label_text}</h3>
                <p style="margin: 0; font-size: 15px; color: #172A45;">
                    Sequence <strong>{seq_id}</strong> exhibits sequence composition and physicochemical attributes 
                    aligned with known Antibiotic Resistance Gene (ARG) proteins.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="background-color: #F8FAFC; border-left: 6px solid #64748B; padding: 18px; border-radius: 6px; margin-bottom: 20px;">
                <h3 style="color: #334155; margin: 0 0 8px 0;">🧬 {label_text}</h3>
                <p style="margin: 0; font-size: 15px; color: #172A45;">
                    Sequence <strong>{seq_id}</strong> does not exceed the resistance classification threshold ($t = {threshold:.2f}$) 
                    and is classified as candidate non-ARG-like.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Key Metrics Columns
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Sequence ID", seq_id)
    m_col2.metric("Sequence Length", f"{length} aa")
    m_col3.metric(f"Score ({score_type})", f"{score:.4f}", delta=f"{score - threshold:+.4f} vs threshold")
    m_col4.metric("Classification Threshold", f"{threshold:.2f}", "Validation Calibrated")

    if "Notice" in msg:
        st.info(msg)

    # Visual Score Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{score_type}</b><br><span style='font-size:0.8em;color:gray'>Threshold = {threshold:.2f}</span>"},
        delta={'reference': threshold, 'increasing': {'color': "#00B4D8"}, 'decreasing': {'color': "#64748B"}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#0A2540"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold], 'color': "#E2E8F0"},
                {'range': [threshold, 1.0], 'color': "#BAE6FD"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.8,
                'value': threshold
            }
        }
    ))
    fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Explanation and Biological Context
    comp_symbol = "≥" if is_arg else "<"
    with st.expander("ℹ️ Scientific Score Interpretation & Guidance", expanded=True):
        st.markdown(f"""
        - **Model Output:** The selected model (`{pred['model_display_name']}`) generated an estimated score of **{score:.4f}**.
        - **Decision Logic:** The validation-calibrated threshold is **{threshold:.2f}**. Since {score:.4f} {comp_symbol} {threshold:.2f}, 
          the model assigns the label **{label_text}**.
        - **Why threshold 0.10?** In homology-clustered held-out evaluation, distant resistance mechanisms often yield conservative scores 
          between 0.10 and 0.35. A standard 0.50 cutoff produces excessive false negatives; $t = 0.10$ maximizes sensitivity while retaining strong specificity (85.8%).
        - **Crucial Limitation:** A high score reflects global sequence compositional similarity to reference ARGs in CARD. It **does NOT** 
          prove in vitro catalytic activity or clinical antibiotic resistance. Experimental verification (e.g. minimum inhibitory concentration testing) is required.
        """)

    # Physicochemical Properties Sub-table
    with st.expander("🔬 Sequence Physicochemical Properties", expanded=False):
        fdict = pred["features_dict"]
        phys_data = [
            {"Property": "Molecular Weight", "Value": f"{fdict.get('PC_molecular_weight', fdict.get('molecular_weight', 0.0)):.2f} Da"},
            {"Property": "Isoelectric Point (pI)", "Value": f"{fdict.get('PC_isoelectric_point', fdict.get('isoelectric_point', 0.0)):.2f}"},
            {"Property": "GRAVY (Hydropathicity)", "Value": f"{fdict.get('PC_gravy', fdict.get('gravy', 0.0)):.3f}"},
            {"Property": "Aromaticity", "Value": f"{fdict.get('PC_aromaticity', fdict.get('aromaticity', 0.0)):.3f}"},
            {"Property": "Instability Index", "Value": f"{fdict.get('PC_instability_index', fdict.get('instability_index', 0.0)):.2f}"},
            {"Property": "Aliphatic Index", "Value": f"{fdict.get('PC_aliphatic_index', fdict.get('aliphatic_index', 0.0)):.2f}"},
            {"Property": "Net Charge at pH 7.0", "Value": f"{fdict.get('PC_net_charge_pH7', fdict.get('net_charge_ph7', 0.0)):.2f}"},
        ]
        st.dataframe(phys_data, use_container_width=True, hide_index=True)


def _render_batch_results(res_df: pd.DataFrame):
    """Render batch predictions table with filters and CSV export."""
    st.markdown("---")
    st.markdown("### Batch Prediction Results")

    # Summary metrics
    n_total = len(res_df)
    n_arg = (res_df["predicted_label"] == 1).sum()
    n_non_arg = (res_df["predicted_label"] == 0).sum()
    n_err = (res_df["predicted_label"] == -1).sum()

    b_c1, b_c2, b_c3, b_c4 = st.columns(4)
    b_c1.metric("Processed Records", n_total)
    b_c2.metric("Predicted ARG-like", n_arg)
    b_c3.metric("Candidate Non-ARG", n_non_arg)
    b_c4.metric("Invalid / Errors", n_err)

    # Interactive Table
    st.dataframe(
        res_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn("Score", format="%.4f"),
            "threshold": st.column_config.NumberColumn("Threshold", format="%.2f")
        }
    )

    # CSV Download Button
    csv_bytes = res_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Batch Results (CSV)",
        data=csv_bytes,
        file_name="arg_batch_predictions.csv",
        mime="text/csv",
        type="primary"
    )
