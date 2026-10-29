"""
Page 7: Methodology and Limitations
Comprehensive documentation of the 9-stage scientific workflow, MMseqs2 clustering,
MILP optimization, and critical bioinformatic and clinical limitations.
"""

import streamlit as st


def render_methodology():
    st.markdown("## Research Methodology & Scientific Limitations")
    st.markdown(
        "A rigorous, reproducible accounting of data curation, homology-leakage prevention, "
        "feature engineering, validation-based calibration, and academic limitations."
    )

    tab_method, tab_limits = st.tabs(["🔬 9-Stage Research Methodology", "⚠️ Scientific & Clinical Limitations"])

    # TAB 1: 9-STAGE RESEARCH METHODOLOGY
    with tab_method:
        st.markdown(r"""
        ### The 9-Stage Scientific Pipeline

        #### 1. CARD Positive Sequence Acquisition
        - **Source:** Comprehensive Antibiotic Resistance Database (CARD) v4.0.2.
        - **Query Criteria:** High-confidence curated protein sequences associated with Antibiotic Resistance Ontology (ARO) terms.
        - **Raw Yield:** 6,059 verified protein sequences representing beta-lactamases, aminoglycoside transferases, macrolide phosphotransferases, efflux pumps, and target alteration enzymes.

        #### 2. UniProtKB/Swiss-Prot Candidate-Negative Acquisition
        - **Source:** UniProtKB/Swiss-Prot release 2024_06 (manually reviewed, high-confidence annotations).
        - **Query Criteria:** Bacterial proteins (`taxonomy_id:2`) with explicit exclusion of keywords: `antibiotic`, `resistance`, `lactamase`, `efflux`, `transporter`, `antimicrobial`.
        - **Raw Yield:** 6,800 bacterial housekeeping sequences representing metabolic enzymes, ribosomal proteins, DNA replication machinery, and cell wall structural elements.

        #### 3. Sequence Curation, Deduplication & Exclusion Handling
        - **Alphabet Validation:** Strict enforcement of the 20 canonical amino acids (`ACDEFGHIKLMNPQRSTVWY`). Non-canonical or ambiguous symbols (`B, Z, X, J, U, O`) were rejected.
        - **Length Boundaries:** Sequence length restricted to 50–1,500 amino acids.
        - **Deduplication:** Exact sequence duplicates and cross-database conflicts were purged and recorded in `exclusion_log.csv` (294 total exclusions).
        - **Master Dataset:** Exactly 12,108 curated sequences (6,054 positive ARGs, 6,054 candidate non-ARGs; 1:1 balance).

        #### 4. Similarity-Aware Homology Clustering & Partitioning
        - **Clustering Engine:** MMseqs2 (`easy-cluster`) configured with `--min-seq-id 0.3` (30% sequence identity) and `-c 0.8` (80% alignment coverage).
        - **Graph Superclustering:** Representative graph union-find connected components resolved secondary transitive homologies, producing **3,631 disjoint superclusters**.
        - **Partitioning via MILP:** A Mixed-Integer Linear Programming optimizer assigned entire superclusters to partitions to achieve exact 70:15:15 target ratios without cluster splitting:
          - **Training:** 8,476 sequences (4,238 ARG / 4,238 non-ARG) across 3,126 clusters (70.0%).
          - **Validation:** 1,816 sequences (908 ARG / 908 non-ARG) across 268 clusters (15.0%).
          - **Held-out Test:** 1,816 sequences (908 ARG / 908 non-ARG) across 237 clusters (15.0%).
        - **Cross-Partition Similarity Audit:** Pairwise Smith-Waterman alignment verified that maximum cross-partition identity is **36.2%**, representing distant superfamily folds only.

        #### 5. Feature Engineering (428 Descriptors)
        - **Amino Acid Composition (AAC):** 20 normalized residue frequencies.
        - **Dipeptide Composition (DPC):** 400 adjacent residue pair frequencies ($20 \\times 20$).
        - **Physicochemical Properties:** 8 descriptors (length, MW, isoelectric point, GRAVY, aromaticity, instability index, aliphatic index, net charge at pH 7.0).

        #### 6. Model Training & Hyperparameter Tuning
        - **Standardization:** `StandardScaler` fitted strictly on the 8,476 training vectors; validation and test partitions transformed using frozen parameters.
        - **Architectures:**
          - **Random Forest:** 300 estimators, Gini criterion, `max_depth=20`, `min_samples_split=5`.
          - **Logistic Regression:** L2 regularized, `C=1.0`, liblinear solver.
          - **SVM (RBF Kernel):** `C=10.0`, `gamma='scale'`, continuous decision margins calibrated using Platt scaling on validation fold margins.
        - **Cross-Validation:** 5-fold stratified CV on training partition.

        #### 7. Validation-Based Threshold Calibration ($t = 0.10$)
        - In homology-split regimes, models evaluate sequences from novel superclusters with $<36\%$ identity to training examples.
        - Standard default thresholds ($0.50$) result in unacceptable false-negative rates ($>70\\%$).
        - Threshold $t = 0.10$ was selected strictly on the validation partition to optimize Matthews Correlation Coefficient (MCC) and sensitivity, while preserving high specificity ($85.8\\%$).
        - The test set remained completely untouched during threshold tuning.

        #### 8. Held-out Test Set Evaluation
        - Final benchmark executed once on the 1,816 unseen test sequences.
        - Random Forest proved best overall: **ROC-AUC = 0.8191, PR-AUC = 0.8161, MCC = 0.4572, Recall = 58.15%, Precision = 80.37%**.

        #### 9. Error Analysis & Length Bias Verification
        - False positives ($n = 129$) and false negatives ($n = 380$) were analyzed biologically.
        - Sequence length distributions across true positives, false positives, true negatives, and false negatives confirmed that predictions are not driven by sequence length artifacts.
        """)

    # TAB 2: SCIENTIFIC & CLINICAL LIMITATIONS
    with tab_limits:
        st.markdown("""
        ### Bioinformatic & Clinical Limitations

        #### 1. Candidate Negatives are Not Proven Susceptible Proteins
        The negative training instances are non-ARG bacterial enzymes derived from Swiss-Prot. While curated to avoid known resistance terms, 
        certain bacterial enzymes (e.g. uncharacterized efflux pumps, beta-lactamase homologs with low catalytic activity) could potentially 
        confer resistance under specific selective pressures. Therefore, negative labels represent *candidate negatives* rather than experimentally 
        verified susceptibility benchmarks.

        #### 2. In Silico Annotation vs. In Vitro Phenotype
        A positive classification indicates that a protein's global amino acid and dipeptide composition resembles documented resistance determinants in CARD. 
        It **does not establish**:
        - Whether the gene is actively expressed in vivo.
        - Whether promoter strength or copy number is sufficient to overcome clinical breakpoints.
        - Whether the enzyme exhibits catalytic turnover against specific antibiotic substrates.
        - Minimum Inhibitory Concentration (MIC) changes in clinical isolates.

        #### 3. Limitations of Global Compositional Descriptors
        AAC and DPC capture overall proteomic signature and local motif frequencies, but they **cannot**:
        - Resolve single-nucleotide point mutations (SNPs) that alter catalytic efficiency without changing global composition (e.g. GyrA fluoroquinolone resistance mutations).
        - Capture tertiary 3D fold geometry, active site cleft topography, or allosteric dynamics.

        #### 4. Homology Horizon & Gene Family Representation
        CARD contains substantial redundancy within well-studied families (e.g. beta-lactamases, aminoglycoside acetyltransferases) and sparse 
        representation of rare or uncultivated resistance families. Although 30% MMseqs2 clustering prevents direct sequence leakage, model performance 
        on completely novel protein folds not present in CARD cannot be guaranteed.

        #### 5. Strict Prohibition on Clinical Diagnostic Use
        **This software is an exploratory research prototype for academic and bioinformatic research.** It is NOT approved by the FDA, EMA, or any 
        regulatory body as a diagnostic tool or medical device. It must NEVER be used to guide patient antibiotic therapy, clinical prescribing decisions, 
        or infection control protocols without formal clinical laboratory validation (e.g., CLSI / EUCAST broth microdilution assays).
        """)
