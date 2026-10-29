# Machine Learning-Based Binary Classification of Antibiotic Resistance Genes (ARGs) vs Non-ARGs Using Protein Sequences

A bioinformatics and machine learning project for classifying bacterial protein sequences into **Antibiotic Resistance Genes (ARGs)** and **Non-ARGs**, engineered with rigorous scientific standards to prevent homology over-optimism and sequence-similarity data leakage.

---

## 1. Project Team & Allocation of Responsibilities

* **Madhav Mendiratta:** Dataset acquisition (CARD v4.0.2 & UniProtKB/Swiss-Prot), sequence hygiene, canonical alphabet verification, cross-dataset collision auditing, and negative-set length stratification.
* **Rakshan:** Feature engineering (428 AAC, DPC, and physicochemical descriptors), MMseqs2 two-stage homology reduction (30% identity threshold), Mixed-Integer Linear Program (MILP) group-aware split optimization, and data leakage audit.
* **Ankit:** Machine learning model design, 5-fold cluster-grouped cross-validation (`GroupKFold`), hyperparameter optimization (Logistic Regression, Random Forest, SVM), and validation set decision threshold calibration ($t = 0.10$).
* **Harsh:** Unbiased test set evaluation, multi-metric benchmarking (ROC-AUC, PR-AUC, MCC), biological error analysis (false positives and false negatives), publication-grade visualization, and dashboard synthesis.

---

## 2. Project Overview & Scientific Rationale

Automated discovery of antibiotic resistance determinants from microbial genomes and metagenomic assemblies requires machine learning classifiers that generalize to distant homologs without overfitting to sequence alignment artifacts or mistaking housekeeping metabolic machinery for resistance enzymes.

### Core Scientific Principles & Methodological Highlights
1. **Curated Biological Sequences & Ground Truth:**
   - **Positive ARGs:** 6,054 verified functional resistance proteins retrieved from the **Comprehensive Antibiotic Resistance Database (CARD v4.0.2)** Protein Homolog Model (PHM). Single-nucleotide target mutation models (PVMs) were explicitly excluded to eliminate single-residue label ambiguity against susceptible bacterial housekeeping genes.
   - **Negative Non-ARGs:** 6,054 reviewed bacterial proteins fetched from **UniProtKB/Swiss-Prot**, rigorously filtered through negative keyword queries (excluding resistance, $\beta$-lactamases, efflux pumps, and related GO terms).
   - **Length-Stratified Sampling:** Negative candidates were drawn across 10 quantile length bins matching the positive ARG length distribution (median: 290 aa positive vs 358 aa negative), preventing machine learning models from learning sequence length as an artificial shortcut.
   - **Cross-Dataset Collision Audit:** 8 exact sequence collisions (Swiss-Prot proteins identical to known CARD ARGs, such as *EmrE*, *MdtC*, and *CpxA*) were discovered and excluded from the negative pool.
   - **Canonical Quality Control:** All 12,108 sequences strictly comprise the 20 canonical amino acids, with lengths constrained between 50 and 1,500 amino acids.
2. **Two-Stage Homology Clustering & Exact Group-Aware Split (MILP):**
   - **Stage 1 (MMseqs2 easy-cluster):** Clustered at 30% sequence identity (`--min-seq-id 0.3`) and 80% bidirectional coverage (`-c 0.8 --cov-mode 0`), yielding 3,936 initial clusters.
   - **Stage 2 (Supercluster Graph):** Connected-component Union-Find on cluster representatives sharing $\ge 30\%$ sequence identity (50% coverage), condensing the dataset into 3,631 superclusters.
   - **Exact 70:15:15 Group-Aware Split:** An exact Mixed-Integer Linear Programming (MILP) solver (`scipy.optimize.milp`) assigned entire superclusters to partitions, achieving an exact 70.00% / 15.00% / 15.00% split with exactly balanced 1:1 class ratios in every partition and **zero cluster splitting** across boundaries.
   - **Mixed Supercluster Audit:** 50 superclusters (475 sequences) containing both ARG and non-ARG annotations (e.g. outer membrane porins and metabolite exporters) were investigated, kept intact within single partitions, and preserved without label alterations.
   - **Empirical Leakage Verification:** Exhaustive MMseqs2 cross-partition search between Test and Train sets confirmed **0 hits >50% sequence identity** (maximum observed identity: 36.2% across 30 distant superfamily hits at 80% coverage).
3. **Interpretable 428-Dimensional Feature Representation:**
   - **Amino Acid Composition (AAC):** 20 features (fractions of canonical residues).
   - **Dipeptide Composition (DPC):** 400 features (frequencies of adjacent amino acid pairs).
   - **Physicochemical Properties:** 8 features (molecular weight, isoelectric point, GRAVY hydropathy, aromaticity, instability index, aliphatic index, net charge at pH 7, sequence length).
   - **Strict Scaling Discipline:** `StandardScaler` was fitted **exclusively on the training partition** ($X_{\text{train}}$) and applied without modification to validation and test sets, strictly preventing data leakage.
4. **Group-Aware Modeling & Calibration:**
   - 5-fold `GroupKFold` cross-validation (grouped by `cluster_id`) was executed on training data only to tune hyperparameters.
   - Decision thresholds were calibrated on the validation set ($t=0.10$) to address the biological domain shift inherent to homology-reduced partitioning.
   - The final test partition remained completely untouched until model selection was locked.

---

## 3. Directory Architecture

```text
ARG-Protein-Classification/
├── data/
│   ├── raw/                                # Raw sequence fasta and provenance metadata
│   │   ├── card_homolog_raw.fasta          # CARD v4.0.2 Protein Homolog Model (6,059 seqs)
│   │   ├── uniprot_negative_raw.fasta      # Swiss-Prot candidate non-ARGs (6,800 seqs)
│   │   └── data_provenance.json            # Checksums, retrieval timestamps, queries
│   ├── processed/                          # Cleaned, validated master datasets
│   │   ├── dataset_master.csv              # Master metadata table (12,108 rows)
│   │   ├── dataset_clean.fasta             # Clean FASTA with unique seq_uid headers
│   │   ├── exclusion_log.csv               # Audit log of 294 exclusions
│   │   └── validation_summary.json         # Length statistics and QC metadata
│   └── splits/                             # Homology-reduced partitions & matrices
│       ├── clusters_summary.csv            # 3,631 superclusters and sizes
│       ├── sequence_cluster_assignments.csv# UID-to-cluster and partition mapping
│       ├── mixed_clusters_investigation.csv# Audit of 50 mixed clusters
│       ├── train_indices.csv               # Training split (8,476 seqs; 3,126 clusters)
│       ├── val_indices.csv                 # Validation split (1,816 seqs; 268 clusters)
│       ├── test_indices.csv                # Held-out test split (1,816 seqs; 237 clusters)
│       ├── features_scaled.npz             # Standardized feature matrices (train/val/test)
│       ├── features_raw.npz                # Unscaled feature matrices
│       ├── scaler.joblib                   # Fitted StandardScaler (train-only)
│       ├── feature_names.json              # 428 feature names
│       ├── feature_summary.json            # Shapes and numerical verification
│       └── split_validation_report.json    # Similarity audit & partition stats
├── app.py                                  # Streamlit research dashboard entry point
├── app_pages/                              # Modular dashboard pages
│   ├── page_overview.py                    # 1. Project Overview & Provenance
│   ├── page_predictor.py                   # 2. Protein Sequence Predictor (Single/Batch FASTA)
│   ├── page_comparison.py                  # 3. Model Benchmark Comparison
│   ├── page_visualizations.py              # 4. Performance Diagnostics (ROC, PR, CM, Length)
│   ├── page_features.py                    # 5. 428-Feature Analysis & Interpretability
│   ├── page_dataset.py                     # 6. Dataset Explorer & Exclusion Log Audit
│   ├── page_methodology.py                 # 7. Workflow Methodology & Limitations
│   └── page_about.py                       # 8. About, Tech Stack & Governance
├── src/
│   ├── __init__.py
│   ├── data_fetch.py                       # Automated CARD and Swiss-Prot acquisition
│   ├── preprocessing.py                    # QC, deduplication, conflict resolution, length matching
│   ├── clustering.py                       # Two-stage MMseqs2 clustering and MILP splitting
│   ├── feature_extraction.py               # AAC, DPC, physicochemical feature extraction
│   ├── inference.py                        # Production-grade inference engine & artifact caching
│   ├── dashboard_data.py                   # Centralized cached artifact and metrics loader
│   ├── models.py                           # Group-aware training, GroupKFold tuning, calibration
│   └── evaluate.py                         # Test evaluation, ROC/PR curves, error analysis
├── tests/
│   ├── test_preprocessing.py               # Tests for sequence QC and conflict deduplication
│   ├── test_clustering_and_split.py        # Tests for partition disjointness & zero cluster splitting
│   ├── test_feature_extraction.py          # Tests for feature dimensions, bounds, and scaling
│   ├── test_models_and_evaluation.py       # Tests for model inference, metrics, and error tables
│   ├── test_app_inference.py               # Tests for dashboard inference, parity, validation (Tests A-G)
│   ├── test_adversarial.py                 # Adversarial edge cases, unicode normalization, bounds
│   └── test_e2e_pipeline.py                # End-to-end integration workflows (E2E Tests 1 to 8)
├── results/
│   ├── figures/                            # High-resolution evaluation charts (300 DPI)
│   │   ├── roc_curves.png                  # Comparison of ROC curves
│   │   ├── pr_curves.png                   # Comparison of Precision-Recall curves
│   │   ├── confusion_matrices.png          # Side-by-side confusion matrix heatmaps
│   │   ├── feature_importance_top25.png    # Top predictive features (LogReg & RF)
│   │   └── sequence_length_distribution_predictions.png # Boxplot of errors vs length
│   ├── metrics/                            # Benchmark tables and error logs
│   │   ├── model_comparison.csv            # Comprehensive test evaluation table
│   │   ├── model_comparison.json           # Raw metrics and confusion matrices
│   │   ├── test_predictions.csv            # Per-sequence prediction probabilities
│   │   ├── false_positives.csv             # 129 false positive sequences
│   │   ├── false_negatives.csv             # 380 false negative sequences
│   │   ├── feature_coefficients_logreg.csv # Ranked LogReg standardized coefficients
│   │   └── feature_importances_rf.csv      # Ranked Random Forest Gini importances
│   └── models/                             # Trained model objects and CV summaries
│       ├── logistic_regression_model.joblib
│       ├── random_forest_model.joblib
│       ├── svm_rbf_model.joblib
│       └── validation_model_comparison.json
├── requirements.txt                        # Pinned dependencies
├── pytest.ini                              # Test runner configuration
└── README.md                               # Project documentation
```

---

## 4. Dataset & Homology-Partitioning Summary

| Metric | Training Split (70%) | Validation Split (15%) | Test Split (15%) | Total Master Dataset |
| :--- | :---: | :---: | :---: | :---: |
| **Total Sequences** | **8,476** (70.00%) | **1,816** (15.00%) | **1,816** (15.00%) | **12,108** (100.0%) |
| **Positive ARGs** | 4,238 (50.0%) | 908 (50.0%) | 908 (50.0%) | 6,054 (50.0%) |
| **Negative Non-ARGs** | 4,238 (50.0%) | 908 (50.0%) | 908 (50.0%) | 6,054 (50.0%) |
| **Superclusters** | 3,126 | 268 | 237 | 3,631 |
| **Cluster Splitting** | **0 clusters split** | **0 clusters split** | **0 clusters split** | **0** |
| **Feature Dimensions** | [8,476, 428] | [1,816, 428] | [1,816, 428] | 428 features |
| **Max Cross-Partition Identity** | Baseline target | — | **36.2%** (distant superfamily) | — |
| **Matches >50% Identity** | — | — | **0 hits** | — |
| **Missing / NaN Values** | 0 | 0 | 0 | 0 |

---

## 5. Model Performance Benchmark on Held-Out Test Set ($N=1,816$)

Models were trained with 5-fold `GroupKFold` cross-validation on training clusters only. Threshold calibration was performed on the validation set ($t=0.10$). Final unbiased evaluation on the held-out test partition ($N=1,816$ sequences from 237 unseen superclusters) produced the following genuine results:

| Classifier | Accuracy | Precision | Recall (Sens.) | Specificity | F1-Score | ROC-AUC | PR-AUC | MCC | TP | TN | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** (Best) | **71.97%** | **80.37%** | **58.15%** | **85.79%** | **0.6748** | **0.8191** | **0.8161** | **0.4572** | 528 | 779 | 129 | 380 |
| **SVM (RBF Kernel)** | 57.54% | 95.36% | 15.86% | 99.23% | 0.2720 | 0.7872 | 0.7995 | 0.2732 | 144 | 901 | 7 | 764 |
| **Logistic Regression (L2)** | 51.65% | 69.23% | 5.95% | 97.36% | 0.1095 | 0.6299 | 0.6148 | 0.0815 | 54 | 884 | 24 | 854 |

### Biological Interpretation of Findings
1. **Strong Ranking Discrimination Under Strict Homology Partitioning:**
   - **Random Forest achieved an ROC-AUC of 0.8191 and a PR-AUC of 0.8161** (against a 0.50 random prevalence baseline). This demonstrates that ensemble decision trees learning non-linear dipeptide interactions generalize effectively to unseen protein families even when sequence identity between test and train is $<36.2\%$.
2. **Failure of Linear Decision Boundaries:**
   - Regularized Logistic Regression achieved only 0.6299 ROC-AUC and 0.0815 MCC, indicating that linear combinations of global amino acid compositions cannot separate resistance mechanisms from housekeeping enzymes across distant protein families.
3. **Biological Domain Shift & Threshold Calibration:**
   - In CARD, thousands of clinical isolates harbor near-identical variants of major $\beta$-lactamases (TEM, CTX-M, OXA, SHV). Clustered at 30% sequence identity, these collapse into 3 massive superclusters (4,165 sequences) placed in the training partition.
   - Consequently, the test set positive ARGs comprise *distinct* gene families (e.g. efflux systems, transferases, MCR, ribosomal protection proteins) not represented in the training set. This causes predicted probabilities to compress downwards, but ranking quality remains high; threshold calibration successfully restored sensitivity.
4. **Error Analysis (False Positives & False Negatives):**
   - **False Positives (129 sequences):** Dominated by physiological bacterial transport proteins (e.g., *PstB* phosphate import ATP-binding protein, *SapF* putrescine exporter, *MdtI* spermidine exporter) and acetyltransferases. These share homologous nucleotide-binding domains, transmembrane topologies, and active-site geometries with multidrug resistance efflux pumps.
   - **False Negatives (380 sequences) & The 58.15% Recall Rate:** Concentrated in rare, atypical resistance families with limited representation in training data. Detecting 58.15% of remote resistance families with 80.37% precision is a genuine reflection of out-of-family generalization, unlike literature models that report >99% recall by testing on near-identical sequence variants.
5. **Absence of Length Shortcut:**
   - The boxplot analysis (`sequence_length_distribution_predictions.png`) confirms that False Positives (median 280 aa) and False Negatives (median 290 aa) have overlapping length distributions with True Positives/Negatives, proving the classifier does not rely on length shortcuts.

---

## 6. Environment Setup & Quickstart

The project requires Python 3.11 with standard scientific computing packages:

```bash
# 1. Clone repository
git clone https://github.com/MadhavMendiratta/ARG-Protein-Classification.git
cd ARG-Protein-Classification

# 2. Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Optional: Verify MMseqs2 installation (required only if regenerating clusters)
mmseqs version  # e.g., 18-8cc5c

# 5. Run full automated test suite (43 passing tests)
pytest tests/ -v
```

---

## 7. Interactive Bioinformatics Research Dashboard (Streamlit)

An interactive, production-grade web application built with **Streamlit**, **Plotly**, and **Scikit-learn** is provided for real-time model inference, dataset exploration, and diagnostic evaluation.

### Launching the Dashboard

```bash
# 1. Activate project virtual environment
source .venv/bin/activate

# 2. Launch the Streamlit application
streamlit run app.py
```

The application runs locally at `http://localhost:8501` (100% offline, zero cloud data transfer).

### Available Dashboard Pages (8 Modules)

1. **Project Overview:** Executive summary, CARD & Swiss-Prot data provenance, dynamic split allocation donut chart (70:15:15), verified benchmark metrics, and end-to-end pipeline architecture flowchart.
2. **Protein Predictor:** Genuine live inference engine. Supports:
   - **Paste Sequence:** Paste plain sequence or FASTA format with optional header; includes one-click sample loaders for verified positives (TEM-126 $\beta$-lactamase, VanX dipeptidase) and candidate negatives (CspB, ATP synthase).
   - **Upload FASTA:** Upload multi-record `.fasta`, `.fa`, or `.faa` files; returns individual sequence scores, applied validation threshold ($t = 0.10$), visual probability gauges, and downloadable batch results CSV.
3. **Model Comparison:** Interactive Plotly grouped bar charts comparing Random Forest, Logistic Regression, and RBF-SVM across Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, PR-AUC, and MCC, with confusion matrix distribution grids.
4. **Performance Visualizations:** Diagnostic charts including interactive ROC curves, Precision-Recall curves, confusion matrices, and sequence length bias distributions.
5. **Feature Analysis:** Breakdown of the 428-dimensional feature space (20 AAC, 400 DPC, 8 Physicochemical), interactive slider for top Random Forest Gini importances, Logistic Regression coefficient weights, and a searchable 428-feature interactive catalog.
6. **Dataset Explorer:** Detailed sequence length statistics (min, max, median, IQR), length histograms, curation exclusion audit log (294 filtered sequences), and an interactive sequence browser with partition and class filters.
7. **Methodology & Limitations:** Detailed documentation of the 9-stage scientific pipeline, MMseqs2 clustering, MILP optimization, and crucial bioinformatic/clinical limitations.
8. **About & Governance:** Project metadata, technology stack summary, local computational privacy guarantee, and academic research disclaimer.

---

## 8. Pipeline Reproduction Commands

All datasets, split assignments, feature matrices, trained models, metrics, and figures are committed and immediately available. To re-execute individual stages from scratch:

```bash
# Step 1: Real Biological Data Acquisition (CARD v4.0.2 & Swiss-Prot)
python src/data_fetch.py 6800

# Step 2: Quality Control, Conflict Deduplication & Length-Quantile Matching
python src/preprocessing.py

# Step 3: Two-Stage Homology Clustering & Exact 70:15:15 MILP Splitting
python src/clustering.py

# Step 4: 428-Feature Extraction & Training-Only Standardization
python src/feature_extraction.py

# Step 5: Model Training with 5-Fold GroupKFold Cross-Validation
python src/models.py

# Step 6: Unbiased Held-Out Test Evaluation & Error Analysis
python src/evaluate.py

# Step 7: Launch Interactive Research Dashboard
streamlit run app.py

# Step 8: Execute Full Automated Test Suite (43 Passed Tests)
pytest tests/ -v
```

---

## 9. Automated Test Suite

The test suite contains **43 comprehensive unit, integration, adversarial, and end-to-end tests** across 7 test modules:
- `tests/test_preprocessing.py`: Sequence alphabet verification, non-canonical residue filtering, and cross-dataset deduplication.
- `tests/test_clustering_and_split.py`: Partition disjointness, exact completeness, zero cluster splitting across partition boundaries, 1:1 balance, and mixed cluster integrity.
- `tests/test_feature_extraction.py`: 428 feature dimensions, AAC/DPC sum-to-one constraints, physicochemical bounds, matrix NaN/Inf checks, and training-only scaler fitting.
- `tests/test_models_and_evaluation.py`: Model serialization, live inference without shape mismatches, benchmark metric validity, confusion matrix sums, false positive/negative partitioning, and figure integrity.
- `tests/test_app_inference.py`: Automated testing for dashboard inference engine (Tests A through G):
  - **Test A:** Artifact availability & model loading (Random Forest, Logistic Regression, SVM RBF).
  - **Test B:** Exact mathematical feature parity between runtime extractor and original training matrices.
  - **Test C:** Input sequence validation (canonical AAs, whitespace normalization, FASTA headers, length advisories, non-canonical rejection).
  - **Test D:** Multi-model inference, probability scoring, and validation threshold ($t = 0.10$) application.
  - **Test E:** Test metrics consistency & confusion matrix verification against `model_comparison.csv`.
  - **Test F:** Multi-record FASTA batch processing, record ID preservation, and downloadable CSV format.
  - **Test G:** App module imports & page rendering smoke test.
- `tests/test_adversarial.py`: Adversarial and regression testing:
  - Physicochemical property keys regression check (`PC_` prefixes).
  - Unicode and non-breaking space (`\u00a0`) stripping and normalization.
  - Direct API validation error raising on malformed inputs (`ValueError`).
  - Randomized sequence generation and strict determinism across multiple runs.
  - Extreme homopolymer sequence stability (Poly-A, Poly-G, charge extremes).
  - Adversarial multi-record FASTA parsing with partial errors.
  - Decision threshold boundary checks ($t = 0.10$).
  - Multi-model invocation independence.
  - Rejection of unknown model keys.
- `tests/test_e2e_pipeline.py`: End-to-end user workflows (E2E Tests 1 to 8):
  - E2E 1: Single-sequence full pipeline inference on TEM-126.
  - E2E 2: Multi-record biological FASTA batch processing.
  - E2E 3: Invalid input reporting and batch status tracking.
  - E2E 4: Multi-model comparison and score type verification.
  - E2E 5: Metrics consistency recomputation from test predictions.
  - E2E 6: Missing optional artifact graceful fallback.
  - E2E 7: Missing mandatory artifact safe failure.
  - E2E 8: 100% offline local execution guarantee (mock socket block).

Execution command:
```bash
pytest tests/ -v
```
Output: `43 passed in 2.46s`.
