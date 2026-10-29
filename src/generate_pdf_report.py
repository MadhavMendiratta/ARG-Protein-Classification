"""
Module: generate_pdf_report.py
Objective: Generate a publication-quality academic PDF report from the project findings
using ReportLab, embedding real figures, benchmark tables, and structured sections.
Output: reports/final_project_report.pdf
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable, PageBreak
)
from reportlab.pdfgen import canvas

REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("results/figures")
METRICS_DIR = Path("results/metrics")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and render total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Suppress running header on cover/title page
        if self._pageNumber > 1:
            self.drawString(54, 11 * 72 - 36, "Machine Learning Binary Classification of ARGs vs Non-ARGs | Capstone Report")
            self.setStrokeColor(colors.HexColor("#cccccc"))
            self.setLineWidth(0.5)
            self.line(54, 11 * 72 - 40, 8.5 * 72 - 54, 11 * 72 - 40)

        # Running footer on all pages
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 30, page_str)
        self.drawString(54, 30, "Confidential — Academic Deliverable | Bioinformatics & Machine Learning Project")
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(54, 40, 8.5 * 72 - 54, 40)
        self.restoreState()


def build_pdf_report():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_filename = REPORTS_DIR / "final_project_report.pdf"
    
    doc = SimpleDocTemplate(
        str(pdf_filename),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1a365d"),
        alignment=0,
        spaceAfter=12
    )
    
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2b6cb0"),
        spaceAfter=20
    )
    
    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#4a5568")
    )
    
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1a365d"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#2c5282"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2d3748"),
        spaceAfter=6
    )
    
    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#742a2a"),
        backColor=colors.HexColor("#fff5f5"),
        borderColor=colors.HexColor("#e53e3e"),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=8,
        spaceAfter=8
    )
    
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Italic"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#4a5568"),
        alignment=1,
        spaceBefore=4,
        spaceAfter=10
    )
    
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1a202c")
    )
    
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    story = []
    
    # ----------------------------------------------------
    # COVER / HEADER BLOCK
    # ----------------------------------------------------
    story.append(Paragraph("Machine Learning-Based Binary Classification of Antibiotic Resistance Genes (ARGs) vs Non-ARGs", title_style))
    story.append(Paragraph("A Rigorous Homology-Reduced Protein Sequence Classification Benchmark", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a365d"), spaceAfter=14))
    
    team_info = """
    <b>Project Team Members & Contributions:</b><br/>
    • <b>Madhav Mendiratta:</b> Biological data acquisition (CARD v4.0.2 & UniProtKB/Swiss-Prot), canonical QC, collision deduplication, length stratification.<br/>
    • <b>Rakshan:</b> 428-dimensional feature engineering (AAC, DPC, physicochemical), MMseqs2 clustering, MILP split optimization.<br/>
    • <b>Ankit:</b> Machine learning pipeline, 5-fold GroupKFold cross-validation, hyperparameter tuning, validation threshold calibration.<br/>
    • <b>Harsh:</b> Held-out test evaluation, multi-metric benchmarking (ROC-AUC, PR-AUC, MCC), biological error analysis, reporting.
    """
    story.append(Paragraph(team_info, meta_style))
    story.append(Spacer(1, 14))
    
    # ----------------------------------------------------
    # ABSTRACT / EXECUTIVE SUMMARY
    # ----------------------------------------------------
    abstract_text = (
        "<b>Executive Summary & Abstract:</b> Automated identification of antibiotic resistance genes (ARGs) is critical "
        "for genomic surveillance of antimicrobial resistance (AMR). However, naive machine learning classifiers often exhibit "
        "severe homology over-optimism due to sequence similarity leakage between training and testing sets, yielding "
        "artificially inflated accuracies (>99%) that fail entirely when deployed on novel resistance families. "
        "In this study, we built an end-to-end, leakage-free binary classification pipeline evaluating 12,108 curated bacterial proteins "
        "(6,054 verified CARD v4.0.2 ARGs vs 6,054 length-matched UniProtKB/Swiss-Prot non-ARGs). "
        "Sequences were clustered at 30% sequence identity and 80% coverage using MMseqs2, followed by graph superclustering into "
        "3,631 superclusters. An exact Mixed-Integer Linear Program (MILP) partitioned superclusters into an exact 70:15:15 "
        "split (Train: 8,476; Val: 1,816; Test: 1,816) with zero cluster splitting across partition boundaries. "
        "Cross-partition similarity audits verified zero test hits exceeding 50% sequence identity (maximum cross-partition identity: 36.2%). "
        "From primary sequences, 428 numerical descriptors (20 AAC, 400 DPC, 8 physicochemical) were standardized using training-only statistics. "
        "Random Forest, Regularized Logistic Regression, and SVM were tuned using 5-fold GroupKFold cross-validation on training clusters, "
        "with decision thresholds calibrated on validation data (t=0.10). On the held-out test set (N=1,816), Random Forest demonstrated superior "
        "overall discrimination (ROC-AUC 0.8191, PR-AUC 0.8161, Accuracy 71.97%, Precision 80.37%, Specificity 85.79%, F1 0.6748, MCC 0.4572), "
        "with an ARG Recall of 58.15%. Error analysis revealed that False Positives (129 seqs) are dominated by physiological ABC transporters sharing "
        "efflux architectures, while False Negatives (380 seqs) represent divergent resistance families lacking training representation. "
        "All 19 unit and regression tests pass in 1.10 seconds."
    )
    story.append(Paragraph(abstract_text, body_style))
    story.append(Spacer(1, 12))
    
    # ----------------------------------------------------
    # SECTION 1: INTRODUCTION & BIOLOGICAL MOTIVATION
    # ----------------------------------------------------
    story.append(Paragraph("1. Introduction & Biological Background", h1_style))
    sec1_text = (
        "Antibiotic resistance is projected to cause 10 million annual deaths globally by 2050 if novel resistance mechanisms "
        "cannot be promptly discovered and contained. Resistance is biologically mediated through four primary mechanisms: "
        "(1) <i>Enzymatic Inactivation</i> (e.g., beta-lactamases hydrolyzing the beta-lactam ring, aminoglycoside transferases); "
        "(2) <i>Active Efflux</i> (e.g., membrane-bound ABC, MFS, and RND pumps lowering intracellular drug concentration); "
        "(3) <i>Target Protection & Alteration</i> (e.g., ribosomal protection proteins like tet(M), methylated 23S rRNA via cfr); and "
        "(4) <i>Target Replacement</i> (e.g., sulfonamide-resistant sul enzymes). While sequence alignment tools (BLAST) identify known variants, "
        "they falter on novel or environmentally mobilized resistance determinants. Machine learning provides alignment-free abstraction, "
        "but standard models frequently memorize strain-level sequence motifs rather than learning generalizable biophysical resistance rules."
    )
    story.append(Paragraph(sec1_text, body_style))
    story.append(Spacer(1, 10))
    
    # ----------------------------------------------------
    # SECTION 2: DATASET ENGINEERING & PROVENANCE
    # ----------------------------------------------------
    story.append(Paragraph("2. Biological Dataset Engineering & Provenance", h1_style))
    sec2_text = (
        "Ground truth data curation is the cornerstone of trustworthy bioinformatics modeling. "
        "Positive sequences were sourced from the <b>Comprehensive Antibiotic Resistance Database (CARD v4.0.2)</b> Protein Homolog Model (PHM). "
        "Crucially, Protein Variant Models (PVMs)—which acquire resistance through single-nucleotide target mutations (e.g., gyrA or rpoB)—were "
        "excluded because a 1-amino-acid mutation against susceptible housekeeping genes creates intractable label ambiguity for global sequence descriptors. "
        "Negative sequences were sourced from <b>UniProtKB/Swiss-Prot</b> (reviewed bacterial proteins) using strict exclusion queries filtering out "
        "resistance, beta-lactamase, efflux pump, and related Gene Ontology keywords. "
        "Programmatic QC eliminated non-canonical residues and constrained lengths between 50 and 1,500 aa. "
        "Cross-dataset exact sequence auditing discovered and removed 8 identical collisions in Swiss-Prot (e.g., EmrE, MdtC, CpxA). "
        "The negative set was sampled across 10 quantile bins matching the positive length distribution, producing an exact 1:1 balanced master dataset of 12,108 sequences."
    )
    story.append(Paragraph(sec2_text, body_style))
    story.append(Spacer(1, 8))
    
    # Dataset Table
    ds_data = [
        [Paragraph("<b>Dataset Partition</b>", table_header_style), Paragraph("<b>Sequences</b>", table_header_style), Paragraph("<b>Positive ARGs</b>", table_header_style), Paragraph("<b>Negative Non-ARGs</b>", table_header_style), Paragraph("<b>Superclusters</b>", table_header_style), Paragraph("<b>Proportion</b>", table_header_style)],
        [Paragraph("Training Split", table_cell_style), Paragraph("8,476", table_cell_style), Paragraph("4,238", table_cell_style), Paragraph("4,238", table_cell_style), Paragraph("3,126", table_cell_style), Paragraph("70.00%", table_cell_style)],
        [Paragraph("Validation Split", table_cell_style), Paragraph("1,816", table_cell_style), Paragraph("908", table_cell_style), Paragraph("908", table_cell_style), Paragraph("268", table_cell_style), Paragraph("15.00%", table_cell_style)],
        [Paragraph("Held-Out Test Split", table_cell_style), Paragraph("1,816", table_cell_style), Paragraph("908", table_cell_style), Paragraph("908", table_cell_style), Paragraph("237", table_cell_style), Paragraph("15.00%", table_cell_style)],
        [Paragraph("<b>Total Master Dataset</b>", table_cell_style), Paragraph("<b>12,108</b>", table_cell_style), Paragraph("<b>6,054</b>", table_cell_style), Paragraph("<b>6,054</b>", table_cell_style), Paragraph("<b>3,631</b>", table_cell_style), Paragraph("<b>100.00%</b>", table_cell_style)]
    ]
    t_ds = Table(ds_data, colWidths=[110, 75, 75, 85, 75, 75])
    t_ds.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1a365d")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor("#f7fafc")]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#edf2f7")),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4)
    ]))
    story.append(t_ds)
    story.append(Spacer(1, 12))
    
    # ----------------------------------------------------
    # SECTION 3: HOMOLOGY CLUSTERING & LEAKAGE AUDITING
    # ----------------------------------------------------
    story.append(Paragraph("3. Two-Stage Homology Clustering & Leakage-Free Splitting", h1_style))
    sec3_text = (
        "To enforce strict cluster-level independence, sequences were clustered using MMseqs2 at 30% sequence identity "
        "and 80% bidirectional coverage (Stage 1: 3,936 clusters), followed by representative graph Union-Find superclustering (Stage 2: 3,631 superclusters). "
        "Because CARD contains three massive beta-lactamase superclusters (4,165 positive sequences), a naive greedy split yielded an imbalanced 10.77% test set. "
        "We formulated the split as an exact <b>Mixed-Integer Linear Program (MILP)</b> using <code>scipy.optimize.milp</code>, successfully achieving an exact "
        "70.00% / 15.00% / 15.00% partition with exact 1:1 class balance and <b>zero cluster splitting</b> across boundaries. "
        "An exhaustive MMseqs2 similarity search across test and train partitions confirmed <b>0 hits >50% sequence identity</b>, with a maximum observed "
        "cross-partition identity of 36.2% (representing distant superfamily structural divergence). "
        "This audit establishes that test sequences cannot be classified by memorizing training sequence motifs. "
        "However, sequence-level audits cannot rule out remote 3D structural homology in the twilight zone (<20% identity) or short unaligned catalytic motifs. "
        "Fifty mixed-annotation superclusters (475 sequences) were identified and kept intact within single partitions without altering biological labels."
    )
    story.append(Paragraph(sec3_text, body_style))
    story.append(Spacer(1, 10))
    
    # ----------------------------------------------------
    # SECTION 4: FEATURE EXTRACTION & PIPELINE
    # ----------------------------------------------------
    story.append(Paragraph("4. Feature Representation & Training Discipline", h1_style))
    sec4_text = (
        "Each protein sequence was converted into 428 interpretable numerical descriptors: "
        "(1) <i>Amino Acid Composition (AAC — 20 features)</i>: normalized residue frequencies summing to 1.0; "
        "(2) <i>Dipeptide Composition (DPC — 400 features)</i>: adjacent residue pair frequencies summing to 1.0; and "
        "(3) <i>Physicochemical Properties (8 features)</i>: molecular weight, isoelectric point, GRAVY hydropathy index, aromaticity, "
        "instability index, aliphatic index, net charge at pH 7, and sequence length. "
        "To strictly eliminate data leakage, <code>StandardScaler</code> was fitted <b>exclusively on the training partition</b> (mean=0.000, std=1.000) "
        "and applied without modification to validation and test partitions."
    )
    story.append(Paragraph(sec4_text, body_style))
    story.append(Spacer(1, 12))
    
    # ----------------------------------------------------
    # SECTION 5: MODEL TRAINING & TEST BENCHMARK
    # ----------------------------------------------------
    story.append(Paragraph("5. Model Performance Benchmark on Held-Out Test Set (N=1,816)", h1_style))
    sec5_text = (
        "Classifiers were trained using 5-fold <code>GroupKFold</code> cross-validation grouped by supercluster. "
        "Due to the collapse of clinical beta-lactamases into training, validation and test positive sets consisted of distinct gene families, "
        "compressing raw predicted probabilities. Threshold calibration on the validation set determined an optimal decision threshold of <b>t = 0.10</b> "
        "for F1-score maximization across all models. Unbiased evaluation on the held-out test partition produced genuine performance metrics:"
    )
    story.append(Paragraph(sec5_text, body_style))
    story.append(Spacer(1, 8))
    
    # Benchmark Table
    bm_data = [
        [Paragraph("<b>Classifier</b>", table_header_style), Paragraph("<b>Accuracy</b>", table_header_style), Paragraph("<b>Precision</b>", table_header_style), Paragraph("<b>Recall</b>", table_header_style), Paragraph("<b>Specificity</b>", table_header_style), Paragraph("<b>F1-Score</b>", table_header_style), Paragraph("<b>ROC-AUC</b>", table_header_style), Paragraph("<b>PR-AUC</b>", table_header_style), Paragraph("<b>MCC</b>", table_header_style)],
        [Paragraph("<b>Random Forest</b> (Best)", table_cell_style), Paragraph("<b>71.97%</b>", table_cell_style), Paragraph("<b>80.37%</b>", table_cell_style), Paragraph("<b>58.15%</b>", table_cell_style), Paragraph("85.79%", table_cell_style), Paragraph("<b>0.6748</b>", table_cell_style), Paragraph("<b>0.8191</b>", table_cell_style), Paragraph("<b>0.8161</b>", table_cell_style), Paragraph("<b>0.4572</b>", table_cell_style)],
        [Paragraph("SVM (RBF Kernel)", table_cell_style), Paragraph("57.54%", table_cell_style), Paragraph("95.36%", table_cell_style), Paragraph("15.86%", table_cell_style), Paragraph("99.23%", table_cell_style), Paragraph("0.2720", table_cell_style), Paragraph("0.7872", table_cell_style), Paragraph("0.7995", table_cell_style), Paragraph("0.2732", table_cell_style)],
        [Paragraph("Logistic Regression (L2)", table_cell_style), Paragraph("51.65%", table_cell_style), Paragraph("69.23%", table_cell_style), Paragraph("5.95%", table_cell_style), Paragraph("97.36%", table_cell_style), Paragraph("0.1095", table_cell_style), Paragraph("0.6299", table_cell_style), Paragraph("0.6148", table_cell_style), Paragraph("0.0815", table_cell_style)]
    ]
    t_bm = Table(bm_data, colWidths=[110, 48, 48, 48, 52, 48, 48, 48, 45])
    t_bm.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1a365d")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4)
    ]))
    story.append(t_bm)
    story.append(Spacer(1, 12))
    
    # ----------------------------------------------------
    # EMBEDDED FIGURES: ROC & PR CURVES
    # ----------------------------------------------------
    story.append(Paragraph("6. Performance Visualizations & Error Analysis", h1_style))
    
    roc_img_path = str(FIGURES_DIR / "roc_curves.png")
    pr_img_path = str(FIGURES_DIR / "pr_curves.png")
    cm_img_path = str(FIGURES_DIR / "confusion_matrices.png")
    fi_img_path = str(FIGURES_DIR / "feature_importance_top25.png")
    len_img_path = str(FIGURES_DIR / "sequence_length_distribution_predictions.png")
    
    # ROC and PR curves side by side
    if Path(roc_img_path).exists() and Path(pr_img_path).exists():
        img_table_data = [
            [Image(roc_img_path, width=240, height=192), Image(pr_img_path, width=240, height=192)],
            [Paragraph("Figure 1: Receiver Operating Characteristic (ROC) Curves (Random Forest AUC = 0.819)", caption_style),
             Paragraph("Figure 2: Precision-Recall (PR) Curves (Random Forest PR-AUC = 0.816)", caption_style)]
        ]
        t_imgs = Table(img_table_data, colWidths=[245, 245])
        t_imgs.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 2)
        ]))
        story.append(KeepTogether([t_imgs]))
        story.append(Spacer(1, 10))
        
    # Confusion matrix heatmap
    if Path(cm_img_path).exists():
        story.append(KeepTogether([
            Image(cm_img_path, width=490, height=147),
            Paragraph("Figure 3: Confusion Matrix Heatmaps on Held-Out Test Set (N=1,816; Random Forest TP=528, TN=779, FP=129, FN=380)", caption_style)
        ]))
        story.append(Spacer(1, 10))
        
    # Feature importance plot
    if Path(fi_img_path).exists():
        story.append(KeepTogether([
            Image(fi_img_path, width=490, height=214),
            Paragraph("Figure 4: Top Predictive Biochemical Features. Left: Logistic Regression coefficients. Right: Random Forest Gini importances.", caption_style)
        ]))
        story.append(Spacer(1, 10))
        
    # Length bias plot
    if Path(len_img_path).exists():
        story.append(KeepTogether([
            Image(len_img_path, width=450, height=275),
            Paragraph("Figure 5: Sequence Length Distributions across Outcome Categories providing descriptive evidence of error independence from length shortcuts.", caption_style)
        ]))
        story.append(Spacer(1, 12))

    # ----------------------------------------------------
    # SECTION 7: BIOLOGICAL ERROR ANALYSIS
    # ----------------------------------------------------
    story.append(Paragraph("7. Biological Error Analysis & Clinical Considerations", h1_style))
    sec7_text = (
        "<b>False Positive Biology (129 sequences):</b> Approximately 65% of False Positives in the Random Forest model comprise physiological bacterial "
        "ABC transporters and export systems (e.g., PstB phosphate import ATP-binding protein, SapF putrescine transport ATPase, MdtI spermidine exporter). "
        "These proteins share nucleotide-binding cassettes, transmembrane topologies, and Walker motifs with multidrug resistance efflux pumps. "
        "The model detects this general export architecture, providing a plausible biological hypothesis based on structural homology. "
        "<br/><br/>"
        "<b>False Negative Biology (380 sequences) & The 58.15% Recall Rate:</b> Random Forest missed 380 ARGs because the positive test partition "
        "consists of divergent, less frequent resistance families (e.g. cfr rRNA methyltransferases, tet(M) ribosomal protection proteins, dfrA dihydrofolate reductases) "
        "whose direct sequence homologs were absent from the training set due to strict 30% homology clustering. "
        "Achieving 58.15% recall and 80.37% precision under strict homology reduction is a genuine reflection of out-of-cluster generalization, "
        "unlike literature models that report >99% recall by testing on near-identical sequence variants."
    )
    story.append(Paragraph(sec7_text, body_style))
    story.append(Spacer(1, 10))
    
    # Clinical governance alert
    gov_text = (
        "<b>CLINICAL GOVERNANCE NOTICE:</b> Computational classification of protein sequences produces predictive hypotheses, "
        "not clinical or diagnostic confirmation of antimicrobial resistance. Phenotypic resistance depends on in vivo gene expression levels, "
        "membrane permeability, and promoter strength, which must be experimentally determined via broth microdilution or disc diffusion assays."
    )
    story.append(Paragraph(gov_text, callout_style))
    story.append(Spacer(1, 12))
    
    # ----------------------------------------------------
    # SECTION 8: CONCLUSION & REFERENCES
    # ----------------------------------------------------
    story.append(Paragraph("8. Conclusion & References", h1_style))
    conc_text = (
        "This project establishes a reproducible, scientifically defensible benchmark for ARG classification. "
        "Random Forest proved to be the most effective architecture (ROC-AUC 0.8191, PR-AUC 0.8161, MCC 0.4572), demonstrating that "
        "non-linear dipeptide combinations capture generalizable biophysical properties of resistance proteins. "
        "Future work will evaluate pre-trained protein language model embeddings (e.g., ESM-2) to capture 3D active-site geometry directly from sequences."
        "<br/><br/>"
        "<b>Key References:</b><br/>"
        "1. Alcock, B. P., et al. (2023). CARD 2023: expanded curation at the Comprehensive Antibiotic Resistance Database. <i>Nucleic Acids Res</i>, 51(D1), D690–D699.<br/>"
        "2. The UniProt Consortium (2023). UniProt: the Universal Protein Knowledgebase in 2023. <i>Nucleic Acids Res</i>, 51(D1), D523–D531.<br/>"
        "3. Steinegger, M., & Söding, J. (2017). MMseqs2 enables sensitive protein sequence searching for massive data sets. <i>Nature Biotechnol</i>, 35(11), 1026–1028.<br/>"
        "4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. <i>J Mach Learn Res</i>, 12, 2825–2830."
    )
    story.append(Paragraph(conc_text, body_style))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[PDF Generated Successfully] {pdf_filename} ({pdf_filename.stat().st_size} bytes)")
    return pdf_filename


if __name__ == "__main__":
    build_pdf_report()
