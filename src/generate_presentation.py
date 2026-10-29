"""
Module: generate_presentation.py
Objective: Build a clean, formal 17-slide academic PowerPoint presentation (.pptx)
in reports/final_presentation.pptx.
Removes any "SENIOR DATA ANALYTICS & BIOINFORMATICS PROJECT" banner.
Balanced, presentation-friendly detail level: concise, punchy, highly legible bullet points,
clear biochemical and mathematical explanations, embedded figures, complete 13-row metric table,
and comprehensive speaker notes on every slide.
"""

import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("results/figures")


def build_presentation():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pptx_path = REPORTS_DIR / "final_presentation.pptx"

    prs = Presentation()
    # 16:9 Widescreen layout: 13.333 x 7.5 inches
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    # Academic Color Palette
    NAVY = RGBColor(15, 23, 42)          # #0F172A (Deep Slate Navy)
    SLATE_ACCENT = RGBColor(30, 58, 138) # #1E3A8A (Academic Blue)
    TEAL_ACCENT = RGBColor(13, 148, 136) # #0D9488 (Bioinformatics Teal)
    DARK_TEXT = RGBColor(30, 41, 59)     # #1E293B (Body text)
    WHITE = RGBColor(255, 255, 255)
    CARD_BG = RGBColor(255, 255, 255)
    CARD_BG_LIGHT = RGBColor(248, 250, 252) # #F8FAFC
    BORDER_COLOR = RGBColor(226, 232, 240)  # #E2E8F0
    BORDER_ACCENT = RGBColor(191, 219, 254) # #BFDBFE

    # Alert / Highlighting Tints
    BG_GREEN_TINT = RGBColor(240, 253, 244) # #F0FDF4
    BORDER_GREEN = RGBColor(187, 247, 208)  # #BBF7D0
    TEXT_GREEN = RGBColor(22, 101, 52)      # #166534

    BG_RED_TINT = RGBColor(254, 242, 242)   # #FEF2F2
    BORDER_RED = RGBColor(254, 202, 202)    # #FECACA
    TEXT_RED = RGBColor(153, 27, 27)        # #991B1B

    BG_BLUE_TINT = RGBColor(239, 246, 255)  # #EFF6FF

    def add_header(slide, title_text, section_breadcrumb=None):
        """
        Standardized clean top header.
        CRITICAL: Completely free of any 'SENIOR DATA ANALYTICS & BIOINFORMATICS PROJECT' banner.
        """
        # Top accent color bar
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.08))
        bar.fill.solid()
        bar.fill.fore_color.rgb = SLATE_ACCENT
        bar.line.fill.background()

        # Section breadcrumb
        if section_breadcrumb:
            cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.24), Inches(11.733), Inches(0.25))
            tf_cat = cat_box.text_frame
            tf_cat.word_wrap = True
            tf_cat.margin_left = tf_cat.margin_right = tf_cat.margin_top = tf_cat.margin_bottom = 0
            p_cat = tf_cat.paragraphs[0]
            p_cat.text = section_breadcrumb.upper()
            p_cat.font.name = "Arial"
            p_cat.font.size = Pt(9.5)
            p_cat.font.bold = True
            p_cat.font.color.rgb = TEAL_ACCENT

        # Main slide title
        top_offset = Inches(0.48) if section_breadcrumb else Inches(0.35)
        title_box = slide.shapes.add_textbox(Inches(0.8), top_offset, Inches(11.733), Inches(0.65))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = "Arial"
        p_title.font.size = Pt(21)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY

    def create_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=BORDER_COLOR):
        """Creates a stylized card container for structured readability."""
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.2)
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_top = Inches(0.18)
        tf.margin_bottom = Inches(0.18)
        tf.margin_left = Inches(0.22)
        tf.margin_right = Inches(0.22)
        return card, tf

    def add_card_header(text_frame, title_text, font_size=13, color=NAVY):
        """Adds a prominent bold header to a card text frame."""
        p = text_frame.paragraphs[0]
        p.text = title_text
        p.font.name = "Arial"
        p.font.size = Pt(font_size)
        p.font.bold = True
        p.font.color.rgb = color
        p.space_after = Pt(8)

    def add_bullet_point(text_frame, bold_prefix, body_text, font_size=10.5, space_before=6, color=DARK_TEXT):
        """Adds a clean, legible bullet point with bold lead-in prefix."""
        p = text_frame.add_paragraph()
        p.font.name = "Arial"
        p.font.size = Pt(font_size)
        p.space_before = Pt(space_before)
        
        # Bold lead-in
        run_bold = p.add_run()
        run_bold.text = f"• {bold_prefix}: "
        run_bold.font.bold = True
        run_bold.font.color.rgb = NAVY
        
        # Punchy, readable text
        run_text = p.add_run()
        run_text.text = body_text
        run_text.font.bold = False
        run_text.font.color.rgb = color

    def add_speaker_note(slide, note_text):
        """Attaches a comprehensive speaker note for defense/viva presentation."""
        notes_slide = slide.notes_slide
        tf = notes_slide.notes_text_frame
        tf.text = note_text.strip()

    card_top = Inches(1.25)
    card_h = Inches(5.75)
    col_w3 = Inches(3.72)
    gap3 = Inches(0.28)
    col_w2 = Inches(5.72)
    gap2 = Inches(0.29)

    # =========================================================================
    # SLIDE 1: Title & Project Team
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    t_box = s1.shapes.add_textbox(Inches(0.9), Inches(1.1), Inches(11.533), Inches(2.3))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Machine Learning-Based Binary Classification of\nAntibiotic Resistance Genes (ARGs) vs Non-ARGs"
    p1.font.name = "Arial"
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = WHITE

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "A Rigorous, Homology-Reduced Protein Sequence Classification Benchmark"
    p1_sub.font.name = "Arial"
    p1_sub.font.size = Pt(16)
    p1_sub.font.color.rgb = RGBColor(147, 197, 253)
    p1_sub.space_before = Pt(10)

    team_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.9), Inches(3.7), Inches(11.533), Inches(2.9))
    team_card.fill.solid()
    team_card.fill.fore_color.rgb = RGBColor(30, 41, 59)
    team_card.line.color.rgb = RGBColor(59, 130, 246)
    team_card.line.width = Pt(1.5)
    tf_team = team_card.text_frame
    tf_team.word_wrap = True
    tf_team.margin_top = Inches(0.2)
    tf_team.margin_bottom = Inches(0.2)
    tf_team.margin_left = Inches(0.3)
    tf_team.margin_right = Inches(0.3)

    p_th = tf_team.paragraphs[0]
    p_th.text = "PROJECT TEAM MEMBERS & DIVISION OF RESPONSIBILITIES"
    p_th.font.name = "Arial"
    p_th.font.size = Pt(13)
    p_th.font.bold = True
    p_th.font.color.rgb = WHITE
    p_th.space_after = Pt(8)

    team_roles = [
        ("Madhav Mendiratta", "Biological data acquisition (CARD v4.0.2 & UniProt), sequence hygiene, collision audit & length-matched sampling."),
        ("Rakshan", "428-feature engineering (AAC, DPC, physicochemical), MMseqs2 homology clustering & MILP split optimization."),
        ("Ankit", "Machine learning model development, 5-fold GroupKFold cross-validation & validation decision threshold calibration."),
        ("Harsh", "Held-out test evaluation, multi-metric benchmarking (ROC/PR/MCC), biological error analysis & reporting.")
    ]
    for member, role in team_roles:
        p_m = tf_team.add_paragraph()
        p_m.font.name = "Arial"
        p_m.font.size = Pt(10.5)
        p_m.space_before = Pt(5)
        run_name = p_m.add_run()
        run_name.text = f"• {member}: "
        run_name.font.bold = True
        run_name.font.color.rgb = RGBColor(253, 224, 71)
        run_role = p_m.add_run()
        run_role.text = role
        run_role.font.bold = False
        run_role.font.color.rgb = RGBColor(226, 232, 240)

    add_speaker_note(s1, (
        "Welcome professors and committee members. Today our team presents our capstone project: "
        "'Machine Learning-Based Binary Classification of Antibiotic Resistance Genes (ARGs) vs Non-ARGs Using Protein Sequences'. "
        "Our team consists of Madhav, Rakshan, Ankit, and Harsh. Our central mission was to build a scientifically rigorous, "
        "leakage-free benchmark that tests genuine out-of-family generalization without optimistic homology bias."
    ))

    # =========================================================================
    # SLIDE 2: AMR Global Crisis & Biological Stakes
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "The Global AMR Crisis & Biological Resistance Mechanisms", "Biological Foundations & Clinical Context")

    c1, tf_c1 = create_card(s2, Inches(0.8), card_top, col_w3, card_h)
    add_card_header(tf_c1, "1. Global Clinical Threat")
    add_bullet_point(tf_c1, "10 Million Deaths by 2050", "WHO projects AMR will surpass cancer deaths annually, causing over $100 trillion in global economic damage.")
    add_bullet_point(tf_c1, "Compromised Medicine", "Routine surgeries, chemotherapy, and transplants depend on effective prophylactic antibiotics.")
    add_bullet_point(tf_c1, "ESKAPE Pathogens", "Resistance concentrates in virulent hospital strains (E. faecium, S. aureus, K. pneumoniae, A. baumannii, P. aeruginosa, Enterobacter spp.).")
    add_bullet_point(tf_c1, "Diagnostic Bottleneck", "Standard phenotypic culture assays require 24–72 hours. Rapid genomic screening is essential for early targeted therapy.")

    c2, tf_c2 = create_card(s2, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h)
    add_card_header(tf_c2, "2. Four Biochemical Paradigms")
    add_bullet_point(tf_c2, "Enzymatic Inactivation", "Direct catalytic destruction or modification (e.g. beta-lactamases cleaving penicillins; aminoglycoside acetyltransferases).")
    add_bullet_point(tf_c2, "Active Efflux Systems", "Membrane transport pumps (ABC, MFS, RND) actively extrude antibiotics, lowering intracellular drug concentrations.")
    add_bullet_point(tf_c2, "Target Protection", "Structural modification prevents antibiotic binding while preserving function (e.g. tet(M) ribosome protection, mecA PBP2a).")
    add_bullet_point(tf_c2, "Target Bypass", "Expression of exogenous drug-insensitive metabolic enzymes that bypass the inhibited step (e.g. sul, dfrA).")

    c3, tf_c3 = create_card(s2, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h)
    add_card_header(tf_c3, "3. Computational ML Solution")
    add_bullet_point(tf_c3, "Metagenomic Expansion", "Next-generation sequencing produces massive uncharacterized contig libraries from hospitals and environmental reservoirs.")
    add_bullet_point(tf_c3, "Alignment Limitations", "Traditional alignment (BLAST) fails when novel resistance genes share low sequence identity with reference catalogs.")
    add_bullet_point(tf_c3, "Physicochemical Rules", "Machine learning abstracts generalized sequence grammar and biophysical properties without multiple sequence alignment.")
    add_bullet_point(tf_c3, "Early Warning Sieve", "Enables rapid in silico pre-screening to prioritize high-risk candidate genes for confirmatory laboratory testing.")

    add_speaker_note(s2, (
        "Antibiotic resistance is an existential medical challenge. Bacteria resist antimicrobial drugs through four primary biochemical strategies: "
        "enzymatic degradation, active efflux pumps, target alteration, and metabolic bypass. "
        "Traditional culture-based susceptibility testing takes up to 72 hours. While sequencing is rapid, traditional BLAST searches fail "
        "on divergent resistance determinants. Machine learning enables rapid pre-screening directly from primary amino acid sequences."
    ))

    # =========================================================================
    # SLIDE 3: Research Gap - BLAST Twilight Zone & Homology Over-Optimism
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "The Critical Research Gap: Why BLAST & Naive ML Models Fail", "Problem Formulation & Literature Limitations")

    c_blast, tf_blast = create_card(s3, Inches(0.8), card_top, col_w2, card_h)
    add_card_header(tf_blast, "The Alignment Bottleneck: The 30% Twilight Zone", color=SLATE_ACCENT)
    add_bullet_point(tf_blast, "High Identity Dependency", "Alignment tools (BLASTp, HMMER) rely on sequence conservation, working exceptionally well above 80% sequence identity.")
    add_bullet_point(tf_blast, "The 20–35% Twilight Zone", "In protein evolution, tertiary folds and catalytic functions diverge far slower than primary sequences. Below 35% identity, alignment sensitivity collapses.")
    add_bullet_point(tf_blast, "Environmental Invisibility", "Novel resistance genes mobilized from environmental resistomes (soil, wastewater) often share <30% identity with clinical catalogs, escaping BLAST detection.")
    add_bullet_point(tf_blast, "Compute Overhead", "Aligning millions of metagenomic contigs against gigabyte reference databases is computationally intensive for real-time surveillance.")

    c_leak, tf_leak = create_card(s3, Inches(0.8 + col_w2 + gap2), card_top, col_w2, card_h, bg_color=BG_RED_TINT, border_color=BORDER_RED)
    add_card_header(tf_leak, "The Fatal Flaw in Literature: Homology Over-Optimism", color=TEXT_RED)
    add_bullet_point(tf_leak, "Naive Random Splitting", "Published bioinformatics studies commonly use random 80:20 train/test splits, reporting artificially inflated test accuracies of >95–99%.", color=TEXT_RED)
    add_bullet_point(tf_leak, "Strain-Level Redundancy", "CARD contains hundreds of near-identical clinical variants of TEM or CTX-M beta-lactamases differing by only 1–3 amino acids (>98% identity).", color=TEXT_RED)
    add_bullet_point(tf_leak, "Severe Data Leakage", "Random splits scatter 90% of TEM variants into train and 10% into test; models simply memorize strain motifs rather than generalizable rules.", color=TEXT_RED)
    add_bullet_point(tf_leak, "Catastrophic Failure", "When deployed on genuinely novel, out-of-family resistance genes, these over-optimistic models fail completely.", color=TEXT_RED)
    add_bullet_point(tf_leak, "Our Benchmark Mandate", "Enforce strict <30% sequence identity between train and test partitions to benchmark authentic out-of-family generalization.", color=TEXT_RED)

    add_speaker_note(s3, (
        "Here is the core research gap: Sequence alignment works well above 80% identity, but fails in the 20-35% twilight zone. "
        "Machine learning should solve this, but the bioinformatics literature suffers from severe homology over-optimism. "
        "Standard random splits scatter near-identical variants across training and test sets. "
        "Models report 99% accuracy simply by memorizing TEM beta-lactamases, but collapse on unseen families. "
        "Our study eliminates this leakage by enforcing strict homology reduction."
    ))

    # =========================================================================
    # SLIDE 4: Project Objectives & Core Scientific Novelties
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Project Objectives & Core Scientific Novelties", "Research Strategy & Innovations")

    c_obj, tf_obj = create_card(s4, Inches(0.8), card_top, col_w2, card_h)
    add_card_header(tf_obj, "Five Core Research Objectives", color=SLATE_ACCENT)
    add_bullet_point(tf_obj, "1. Curate Unbiased Dataset", "Curate 6,054 verified ARGs from CARD v4.0.2 and 6,054 reviewed bacterial non-ARGs from Swiss-Prot across 10 matched length bins.")
    add_bullet_point(tf_obj, "2. Homology Clustering", "Cluster all 12,108 sequences at 30% sequence identity and 80% coverage using MMseqs2, condensing into 3,631 superclusters.")
    add_bullet_point(tf_obj, "3. Exact MILP Partitioning", "Formulate and solve a Mixed-Integer Linear Program to achieve an exact 70:15:15 split (8,476 / 1,816 / 1,816) with zero cluster splitting.")
    add_bullet_point(tf_obj, "4. 428-Dimensional Features", "Extract 20 AAC, 400 DPC, and 8 physicochemical properties with strict training-only standardization to prevent preprocessing leakage.")
    add_bullet_point(tf_obj, "5. Group-Aware Benchmarking", "Evaluate LogReg, Random Forest, and SVM under 5-fold GroupKFold CV and validation threshold calibration (t=0.10).")

    c_nov, tf_nov = create_card(s4, Inches(0.8 + col_w2 + gap2), card_top, col_w2, card_h, bg_color=BG_GREEN_TINT, border_color=BORDER_GREEN)
    add_card_header(tf_nov, "Four Key Scientific Novelties", color=TEXT_GREEN)
    add_bullet_point(tf_nov, "1. Zero-Leakage Guarantee", "Maximum cross-partition pairwise sequence identity strictly capped at 36.2% (0 hits >50%), establishing an honest generalization lower bound.", color=TEXT_GREEN)
    add_bullet_point(tf_nov, "2. MILP Split Optimization", "First implementation of Mixed-Integer Linear Programming to solve CARD's severe cluster-size skew, achieving exactly 70.00% / 15.00% / 15.00%.", color=TEXT_GREEN)
    add_bullet_point(tf_nov, "3. Collision-Audited Negatives", "Programmatic cross-dataset audit eliminated 8 identical Swiss-Prot collisions; 10-bin length matching removed sequence length shortcuts.", color=TEXT_GREEN)
    add_bullet_point(tf_nov, "4. Calibrated Decision Threshold", "Calibrated decision threshold (t=0.10) strictly on validation data to counter probability compression from biological domain shift.", color=TEXT_GREEN)

    add_speaker_note(s4, (
        "Our work establishes four major scientific novelties: First, a zero-leakage guarantee with maximum pairwise identity of 36.2%. "
        "Second, an exact MILP mathematical formulation that solves the severe combinatorial cluster-size imbalance in CARD. "
        "Third, collision-audited and length-matched negative curation. "
        "Fourth, principled validation-based threshold calibration."
    ))

    # =========================================================================
    # SLIDE 5: Positive Ground Truth - CARD v4.0.2
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Positive Ground Truth Curation: CARD v4.0.2 Dataset", "Data Curation & Provenance")

    c_card1, tf_card1 = create_card(s5, Inches(0.8), card_top, col_w3, card_h)
    add_card_header(tf_card1, "1. CARD Reference Database")
    add_bullet_point(tf_card1, "Gold-Standard Resource", "Sourced from Comprehensive Antibiotic Resistance Database (CARD release v4.0.2), maintained by McMaster University.")
    add_bullet_point(tf_card1, "ARO Ontology", "Organized around the controlled Antibiotic Resistance Ontology (ARO), defining molecular mechanisms and drug targets.")
    add_bullet_point(tf_card1, "Curated Sequences", "Downloaded the complete curated reference dataset comprising 6,059 canonical resistance protein sequences.")

    c_card2, tf_card2 = create_card(s5, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h, bg_color=CARD_BG_LIGHT)
    add_card_header(tf_card2, "2. Why Exclude Variant Models?")
    add_bullet_point(tf_card2, "PHM vs PVM Distinction", "Protein Homolog Models (PHMs) represent dedicated acquired resistance genes (e.g. beta-lactamases, efflux pumps). Presence confers resistance.")
    add_bullet_point(tf_card2, "Point Mutation Models", "Protein Variant Models (PVMs) are housekeeping genes where resistance arises via single SNPs (e.g. gyrA S83L conferring ciprofloxacin resistance).")
    add_bullet_point(tf_card2, ">99.8% Sequence Identity", "Susceptible wild-type gyrA and resistant mutant gyrA share >99.8% identity, differing by only 1 amino acid out of 800+.")
    add_bullet_point(tf_card2, "Severe Label Noise", "Global composition (AAC, DPC) cannot resolve single point mutations without 3D atomic coordinates. Excluding PVMs ensures every positive is a dedicated ARG.")

    c_card3, tf_card3 = create_card(s5, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h)
    add_card_header(tf_card3, "3. Positive Sequence Hygiene")
    add_bullet_point(tf_card3, "Canonical Alphabet", "Verified all sequences against standard 20 amino acids; eliminated sequences with ambiguous characters (X, B, Z, J, U, O, *).")
    add_bullet_point(tf_card3, "Length Boundaries", "Enforced biologically plausible functional range: 50 <= Length <= 1,500 amino acids, removing fragments and giant polyproteins.")
    add_bullet_point(tf_card3, "Clean Positive Pool", "5 sequences removed during QC, yielding exactly 6,054 verified functional resistance proteins.")

    add_speaker_note(s5, (
        "We sourced our positive cohort from CARD release 4.0.2. Crucially, we restricted sequences strictly to the Protein Homolog Model (PHM) "
        "and excluded Protein Variant Models (PVMs). In PVMs, resistance is caused by a single point mutation in a housekeeping gene like gyrA, "
        "sharing over 99.8% identity with susceptible versions. Global sequence composition cannot distinguish one mutation in 800 residues, "
        "which would introduce extreme label noise. Excluding PVMs guarantees that every positive instance is an authentic resistance gene."
    ))

    # =========================================================================
    # SLIDE 6: Negative Cohort Engineering & Sequence Hygiene
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Negative Cohort Engineering & Biological Sequence Hygiene", "Data Curation & Quality Control")

    c_neg1, tf_neg1 = create_card(s6, Inches(0.8), card_top, col_w3, card_h)
    add_card_header(tf_neg1, "1. Reviewed Bacterial Negatives")
    add_bullet_point(tf_neg1, "UniProtKB/Swiss-Prot", "Sourced strictly from manually reviewed bacterial proteins (taxonomy_id:2), preventing models from learning trivial eukaryotic taxonomic shortcuts.")
    add_bullet_point(tf_neg1, "Strict Exclusion Query", "Constructed multi-parameter REST query filtering out resistance keywords, beta-lactamases, efflux pumps, and related GO terms.")
    add_bullet_point(tf_neg1, "Candidate Harvest", "Retrieved an initial pool of 6,800 reviewed bacterial proteins for downstream quality control.")

    c_neg2, tf_neg2 = create_card(s6, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h, bg_color=BG_RED_TINT, border_color=BORDER_RED)
    add_card_header(tf_neg2, "2. Collision Audit (8 Purged)", color=TEXT_RED)
    add_bullet_point(tf_neg2, "Contamination Risk", "Swiss-Prot contains bacterial resistance proteins that occasionally lack explicit resistance keyword tags.", color=TEXT_RED)
    add_bullet_point(tf_neg2, "Exact String Audit", "Automated sequence identity comparison between 6,800 Swiss-Prot candidates and 6,054 CARD ARGs.", color=TEXT_RED)
    add_bullet_point(tf_neg2, "8 Identical Collisions", "Discovered 8 sequences 100% identical to CARD ARGs: EmrE multidrug transporter, MdtC pump component, and CpxA kinase.", color=TEXT_RED)
    add_bullet_point(tf_neg2, "Immediate Purge", "All 8 colliding entries were purged immediately, preventing false-negative contamination.", color=TEXT_RED)

    c_neg3, tf_neg3 = create_card(s6, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h)
    add_card_header(tf_neg3, "3. Length-Matched Sampling")
    add_bullet_point(tf_neg3, "Length Discrepancy", "CARD ARGs median length is 290 aa; raw Swiss-Prot median is 358 aa. Without correction, models learn that 'short proteins = ARGs'.")
    add_bullet_point(tf_neg3, "10-Quantile Bins", "Partitioned positive length distribution into 10 quantile bins: [53, 235], (235, 266], ..., (477, 1452] aa.")
    add_bullet_point(tf_neg3, "Stratified Sampling", "Sampled negative candidates within identical 10 bins to mirror the positive length distribution.")
    add_bullet_point(tf_neg3, "Exact 1:1 Balance", "Produced an exact 1:1 balanced master dataset of 12,108 sequences: 6,054 ARGs vs 6,054 Non-ARGs (exclusion_log.csv).")

    add_speaker_note(s6, (
        "For negative non-ARGs, we avoided arbitrary eukaryotic sequences and restricted our pool strictly to reviewed bacterial proteins in Swiss-Prot. "
        "Critically, our automated cross-dataset collision audit revealed that 8 Swiss-Prot proteins were 100% identical to CARD ARGs (like EmrE and MdtC) "
        "because they lacked explicit resistance keywords. We purged them immediately. "
        "Furthermore, we performed 10-bin length-quantile stratified sampling to match the positive length distribution, eliminating sequence length shortcuts."
    ))

    # =========================================================================
    # SLIDE 7: Homology Reduction & MMseqs2 Two-Stage Clustering
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Homology Reduction & MMseqs2 Two-Stage Clustering", "Homology-Aware Methodology")

    c_cl1, tf_cl1 = create_card(s7, Inches(0.8), card_top, col_w3, card_h)
    add_card_header(tf_cl1, "1. The 30% Twilight Zone")
    add_bullet_point(tf_cl1, "Biological Twilight Zone", "In protein structural biology, 30% sequence identity marks the boundary below which pairwise sequence alignment cannot infer homology.")
    add_bullet_point(tf_cl1, "Testing True Generalization", "Enforcing 30% separation forces the model to predict resistance across remote evolutionary families, simulating novel gene discovery.")
    add_bullet_point(tf_cl1, "MMseqs2 Engine", "Employed MMseqs2 (release 18-8cc5c) for sensitive, ultra-fast sequence clustering.")
    add_bullet_point(tf_cl1, "80% Coverage Constraint", "Required 80% bidirectional alignment coverage (-c 0.8 --cov-mode 0) to ensure full-domain homology rather than local peptide matches.")

    c_cl2, tf_cl2 = create_card(s7, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h)
    add_card_header(tf_cl2, "2. Stage 1: Easy-Cluster")
    add_bullet_point(tf_cl2, "Clustering Execution", "Clustered all 12,108 sequences at 30% identity and 80% coverage using mmseqs easy-cluster.")
    add_bullet_point(tf_cl2, "Initial Cluster Yield", "Grouped into 3,936 initial sequence clusters, each represented by a centroid sequence.")
    add_bullet_point(tf_cl2, "Cluster Size Skew", "2,750 clusters were singletons, while the largest cluster contained over 2,000 beta-lactamase variants.")
    add_bullet_point(tf_cl2, "Borderline Pairs", "Centroids can still share borderline similarity (e.g. 28–32% identity), requiring stage-two condensation.")

    c_cl3, tf_cl3 = create_card(s7, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h, bg_color=CARD_BG_LIGHT)
    add_card_header(tf_cl3, "3. Stage 2: Superclustering")
    add_bullet_point(tf_cl3, "Centroid Alignment", "Extracted all 3,936 centroid sequences and performed an all-against-all MMseqs2 alignment.")
    add_bullet_point(tf_cl3, "Graph Construction", "Constructed an undirected graph connecting any two clusters whose centroids shared >= 30% identity at >= 50% coverage.")
    add_bullet_point(tf_cl3, "Union-Find Condensation", "Applied disjoint-set (Union-Find) connected-components algorithm to merge transitively linked clusters.")
    add_bullet_point(tf_cl3, "3,631 Superclusters", "Condensed the 3,936 initial clusters into exactly 3,631 disjoint superclusters, guaranteeing zero transitive leakage.")

    add_speaker_note(s7, (
        "We implemented a two-stage clustering protocol. Stage 1 used MMseqs2 easy-cluster at 30% identity and 80% coverage, "
        "producing 3,936 initial clusters. In Stage 2, we performed an all-against-all alignment between centroids and applied Union-Find "
        "graph condensation to merge any clusters sharing borderline homology. This yielded 3,631 robust, isolated superclusters."
    ))

    # =========================================================================
    # SLIDE 8: Group-Aware Partitioning via MILP
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Exact 70:15:15 Group-Aware Partitioning via MILP", "Homology-Aware Methodology & Optimization")

    c_milp1, tf_milp1 = create_card(s8, Inches(0.8), card_top, col_w3, card_h, bg_color=BG_RED_TINT, border_color=BORDER_RED)
    add_card_header(tf_milp1, "1. The Skewed Cluster Dilemma", color=TEXT_RED)
    add_bullet_point(tf_milp1, "Extreme Cluster Skew", "In CARD, three massive beta-lactamase superclusters contain 4,165 positive sequences.", color=TEXT_RED)
    add_bullet_point(tf_milp1, "Zero-Splitting Rule", "To prevent leakage, all sequences in a supercluster must reside 100% within a single partition.", color=TEXT_RED)
    add_bullet_point(tf_milp1, "Greedy Heuristic Failure", "A standard greedy binning algorithm was forced to dump all large clusters into training, leaving only 10.77% in test (target: 15.00%).", color=TEXT_RED)
    add_bullet_point(tf_milp1, "Need for Optimization", "Mathematical optimization was required to achieve balanced partitions without splitting clusters.", color=TEXT_RED)

    c_milp2, tf_milp2 = create_card(s8, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h)
    add_card_header(tf_milp2, "2. MILP Mathematical Model")
    add_bullet_point(tf_milp2, "Formulation", "Formulated cluster assignment as a Mixed-Integer Linear Program solved via scipy.optimize.milp with HiGHS.")
    add_bullet_point(tf_milp2, "Decision Variable", "Binary variable x_{c,p} in {0, 1} assigning supercluster c to partition p in {train, val, test}.")
    add_bullet_point(tf_milp2, "Objective Function", "Minimize total partition size deviations and class imbalances with penalty lambda = 2.0.")
    add_bullet_point(tf_milp2, "Strict Constraint", "sum_{p} x_{c,p} = 1 for all 3,631 clusters. Zero cluster splitting allowed.")

    c_milp3, tf_milp3 = create_card(s8, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h, bg_color=BG_GREEN_TINT, border_color=BORDER_GREEN)
    add_card_header(tf_milp3, "3. Mathematically Exact Split", color=TEXT_GREEN)
    add_bullet_point(tf_milp3, "Training Partition", "8,476 sequences (exactly 70.00%): 4,238 Pos, 4,238 Neg; 3,126 superclusters.", color=TEXT_GREEN)
    add_bullet_point(tf_milp3, "Validation Partition", "1,816 sequences (exactly 15.00%): 908 Pos, 908 Neg; 268 superclusters.", color=TEXT_GREEN)
    add_bullet_point(tf_milp3, "Held-Out Test Partition", "1,816 sequences (exactly 15.00%): 908 Pos, 908 Neg; 237 superclusters.", color=TEXT_GREEN)
    add_bullet_point(tf_milp3, "Zero Cluster Splitting", "Every supercluster resides entirely within a single split with exact 1:1 class balance.", color=TEXT_GREEN)

    add_speaker_note(s8, (
        "Because CARD contains massive beta-lactamase superclusters, a standard greedy heuristic produced an imbalanced test partition "
        "containing only 10.77% of sequences. We formulated cluster assignment as an exact Mixed-Integer Linear Program using SciPy. "
        "The solver found a globally optimal assignment achieving an exact 70:15:15 split (8,476 / 1,816 / 1,816) with zero cluster splitting."
    ))

    # =========================================================================
    # SLIDE 9: Similarity Audit & Mixed Clusters Investigation
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Cross-Partition Similarity Audit & Mixed Cluster Investigation", "Homology-Aware Methodology & Auditing")

    c_audit1, tf_audit1 = create_card(s9, Inches(0.8), card_top, col_w3, card_h)
    add_card_header(tf_audit1, "1. Cross-Partition Audit")
    add_bullet_point(tf_audit1, "Audit Alignment", "Aligned all 1,816 test sequences against all 8,476 training sequences via MMseqs2 at E <= 10^-3.")
    add_bullet_point(tf_audit1, "Zero Hits >50% Identity", "Zero test sequences share >50% pairwise sequence identity with any training sequence.")
    add_bullet_point(tf_audit1, "Coverage Audit", "Out of 6,117 total local alignment hits, only 30 met the 80% bidirectional coverage criterion.")
    add_bullet_point(tf_audit1, "Max Observed Identity: 36.2%", "Maximum cross-partition identity is 36.2%, representing distant superfamily variants within the class-A beta-lactamase fold.")

    c_audit2, tf_audit2 = create_card(s9, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h, bg_color=CARD_BG_LIGHT)
    add_card_header(tf_audit2, "2. Scope: What Audit Proves")
    add_bullet_point(tf_audit2, "Eliminates Memorization", "Proves MILP partitioning completely eliminates sequence memorization shortcuts.")
    add_bullet_point(tf_audit2, "Boundary Verification", "Confirms no test/train pair violates the 30% identity / 80% coverage clustering boundary.")
    add_bullet_point(tf_audit2, "What It Cannot Prove: 3D Folds", "Cannot rule out 3D structural fold conservation (<20% identity), as tertiary structures diverge slower than sequences.")
    add_bullet_point(tf_audit2, "Short Motifs", "Cannot rule out shared local catalytic motifs (e.g. Walker A/B) falling below 80% coverage.")

    c_audit3, tf_audit3 = create_card(s9, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h)
    add_card_header(tf_audit3, "3. 50 Mixed Superclusters")
    add_bullet_point(tf_audit3, "Biological Dual Functions", "50 superclusters (475 sequences) contain both ARG and non-ARG annotations, reflecting biological multifunctionality.")
    add_bullet_point(tf_audit3, "Outer Membrane Porins", "Example: OmpF and OmpC porins, where general metabolite permeability mutations confer non-specific antibiotic resistance.")
    add_bullet_point(tf_audit3, "Metabolic Transporters", "Physiological ABC transporters with close structural homology to dedicated multidrug resistance efflux pumps.")
    add_bullet_point(tf_audit3, "Retained Intact", "All 50 mixed clusters were kept intact within single partitions and documented in mixed_clusters_investigation.csv.")

    add_speaker_note(s9, (
        "Our empirical similarity audit confirmed that zero test sequences share >50% identity with training data, with a maximum observed identity of 36.2%. "
        "This proves the elimination of sequence memorization shortcuts. We also audited 50 mixed superclusters representing dual-function "
        "proteins like outer membrane porins, retaining them intact within single partitions without silent relabeling."
    ))

    # =========================================================================
    # SLIDE 10: 428-Dimensional Feature Representation
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "428-Dimensional Alignment-Free Feature Representation", "Feature Engineering & Preprocessing Discipline")

    c_feat1, tf_feat1 = create_card(s10, Inches(0.8), card_top, col_w3, card_h)
    add_card_header(tf_feat1, "1. Amino Acid Composition (AAC)")
    add_bullet_point(tf_feat1, "20 AAC Descriptors", "Normalized frequency of each of the 20 standard amino acids: f(a) = N(a) / L, where sum f(a) = 1.0.")
    add_bullet_point(tf_feat1, "Length Invariant", "Normalized by sequence length, capturing global structural propensities.")
    add_bullet_point(tf_feat1, "Biological Significance", "Detects hydrophobic enrichment in transmembrane efflux pumps vs basic residues in nucleic-acid interacting proteins.")
    add_bullet_point(tf_feat1, "Instant Computation", "Extracted in microseconds directly from primary FASTA sequences without database searches.")

    c_feat2, tf_feat2 = create_card(s10, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h)
    add_card_header(tf_feat2, "2. Dipeptide Composition (DPC)")
    add_bullet_point(tf_feat2, "400 DPC Descriptors", "Frequencies of all 20 x 20 adjacent residue pairs: f(d) = N(d) / (L - 1), where sum f(d) = 1.0.")
    add_bullet_point(tf_feat2, "Local Secondary Structure", "Captures nearest-neighbor context and secondary structure elements (alpha-helices, beta-sheets, beta-turns).")
    add_bullet_point(tf_feat2, "Catalytic Motifs", "Encodes active-site catalytic dyads and loop flexibility without requiring 3D atomic coordinates.")
    add_bullet_point(tf_feat2, "High Information Density", "Provides a rich 400-dimensional continuous fingerprint of local peptide architecture.")

    c_feat3, tf_feat3 = create_card(s10, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h, bg_color=CARD_BG_LIGHT)
    add_card_header(tf_feat3, "3. Physicochemical Properties")
    add_bullet_point(tf_feat3, "8 Biophysical Descriptors", "Extracted via Biopython's ProteinAnalysis module:")
    add_bullet_point(tf_feat3, "Bulk Properties", "Molecular Weight (Mw), Isoelectric Point (pI), Sequence Length (L), Net Charge at pH 7.0.")
    add_bullet_point(tf_feat3, "Structural Stability", "GRAVY (Kyte-Doolittle hydropathy), Aliphatic Index (thermal stability), Aromaticity, Instability Index.")
    add_bullet_point(tf_feat3, "Training-Only Scaling", "StandardScaler fitted exclusively on X_train (mu=0.000, sigma=1.000); validation and test transformed using frozen scaler parameters. Zero leakage.")

    add_speaker_note(s10, (
        "We extracted 428 alignment-free descriptors directly from primary FASTA sequences: 20 Amino Acid Composition features, "
        "400 Dipeptide Composition features capturing local secondary structure and catalytic dyads, and 8 bulk Physicochemical descriptors. "
        "To prevent preprocessing leakage, StandardScaler was fitted strictly on the 8,476 training sequences only."
    ))

    # =========================================================================
    # SLIDE 11: Model Architectures & GroupKFold Cross-Validation
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Machine Learning Architectures & Group-Aware Validation", "Model Development & Optimization")

    c_m1, tf_m1 = create_card(s11, Inches(0.8), card_top, col_w3, card_h)
    add_card_header(tf_m1, "1. Three Algorithmic Paradigms")
    add_bullet_point(tf_m1, "Regularized Logistic Regression", "Linear baseline modeling log-odds: log(p/(1-p)) = beta_0 + sum beta_j z_j, with L2 Ridge penalty ((1/2C)||beta||_2^2) across 400 dipeptides.")
    add_bullet_point(tf_m1, "Random Forest Classifier", "Non-linear ensemble of 100 decorrelated decision trees using bagging and random feature subspace sampling, splitting nodes by Gini impurity.")
    add_bullet_point(tf_m1, "Support Vector Machine (SVM)", "Maximum-margin separator mapping features into Hilbert space via RBF kernel: K(x, x') = exp(-gamma ||x - x'||^2). Platt-scaled probabilities.")

    c_m2, tf_m2 = create_card(s11, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h, bg_color=CARD_BG_LIGHT)
    add_card_header(tf_m2, "2. 5-Fold GroupKFold Discipline")
    add_bullet_point(tf_m2, "The Flaw of Standard K-Fold", "Standard K-fold randomly splits sequences, leaking members of the same homology cluster across training and validation folds.")
    add_bullet_point(tf_m2, "Cluster-Grouped Folds", "We strictly used GroupKFold(n_splits=5) grouped by cluster_id across the 8,476 training sequences (3,126 clusters).")
    add_bullet_point(tf_m2, "Strict Cluster Separation", "Sequences from the same supercluster resided entirely within either fold training or validation, never split.")
    add_bullet_point(tf_m2, "Unbiased Model Selection", "CV scores reflected true out-of-cluster generalization performance during hyperparameter optimization.")

    c_m3, tf_m3 = create_card(s11, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h)
    add_card_header(tf_m3, "3. Hyperparameter Tuning")
    add_bullet_point(tf_m3, "Logistic Regression Grid", "Tuned C in [0.01, 0.1, 1.0, 10.0]. Optimal: C = 10.0 (mean CV ROC-AUC: 0.7998). Moderate regularization provided stability.")
    add_bullet_point(tf_m3, "Random Forest Grid", "Tuned N_trees in [100, 200], max_depth in [15, 25, None], min_split in [2, 5]. Optimal: N_trees = 100, depth = 15, split = 5 (CV ROC-AUC: 0.7102).")
    add_bullet_point(tf_m3, "SVM (RBF) Grid", "Tuned C in [0.5, 1.0, 5.0] and gamma in ['scale', 'auto']. Optimal: C = 0.5, gamma = 'auto' (CV ROC-AUC: 0.7741).")
    add_bullet_point(tf_m3, "Locked Test Partition", "Held-out test set remained completely locked during all training and hyperparameter search.")

    add_speaker_note(s11, (
        "We compared three contrasting paradigms: Logistic Regression as a linear baseline, Random Forest for non-linear feature interactions, "
        "and RBF-SVM for maximum-margin separation. To prevent cross-validation leakage, we used 5-fold GroupKFold grouped by cluster ID, "
        "ensuring that entire superclusters were held out in each fold. The test set remained in cold storage."
    ))

    # =========================================================================
    # SLIDE 12: Decision Threshold Calibration (t = 0.10)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Validation Decision Threshold Calibration (t = 0.10 Rationale)", "Model Calibration & Decision Strategy")

    c_th1, tf_th1 = create_card(s12, Inches(0.8), card_top, col_w3, card_h)
    add_card_header(tf_th1, "1. Biological Domain Shift")
    add_bullet_point(tf_th1, "Beta-Lactamase Skew", "Training absorbed 4,165 beta-lactamase sequences, leaving validation and test sets enriched in distinct, out-of-family resistance mechanisms.")
    add_bullet_point(tf_th1, "Out-of-Family Families", "Validation and test ARGs consisted of efflux pumps, transferases, and ribosomal protection sharing <36.2% identity with training.")
    add_bullet_point(tf_th1, "Probability Compression", "When evaluating remote sequence families, predicted confidence scores compressed downwards (median positive probability: ~0.15).")
    add_bullet_point(tf_th1, "Default Threshold Collapse", "Evaluating at default t = 0.50 resulted in near-zero recall (<2% for LogReg; <10% for SVM).")

    c_th2, tf_th2 = create_card(s12, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h, bg_color=CARD_BG_LIGHT)
    add_card_header(tf_th2, "2. Validation Optimization Sweep")
    add_bullet_point(tf_th2, "Optimization Protocol", "Swept thresholds t in [0.10, 0.90] exclusively on the 1,816 validation sequences to maximize F1-score.")
    add_bullet_point(tf_th2, "Harmonic Optimization", "F1-score balances precision and recall, establishing an optimal operating point for screening.")
    add_bullet_point(tf_th2, "Optimal Threshold t = 0.10", "An operating threshold of t = 0.10 was identified as optimal across all three classifiers on the validation cohort:")
    add_bullet_point(tf_th2, "Validation F1 Peaks", "Random Forest Val F1 peaked at 0.5371 (vs 0.3542 at t=0.50). SVM Val F1 peaked at 0.2614. LogReg Val F1 peaked at 0.2260.")

    c_th3, tf_th3 = create_card(s12, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h, bg_color=BG_BLUE_TINT, border_color=BORDER_ACCENT)
    add_card_header(tf_th3, "3. Threshold Freezing & Rationale", color=SLATE_ACCENT)
    add_bullet_point(tf_th3, "Freezing Before Test", "The calibrated threshold of t = 0.10 was permanently frozen in model metadata before opening the held-out test set.", color=SLATE_ACCENT)
    add_bullet_point(tf_th3, "Zero Test Peeking", "Test partition was evaluated exactly once using the frozen threshold. No post-hoc tuning on test data.", color=SLATE_ACCENT)
    add_bullet_point(tf_th3, "Clinical Cost Asymmetry", "In pathogen surveillance, false negatives (missing an active ARG) carry far higher clinical costs than false positives (which undergo confirmatory AST).", color=SLATE_ACCENT)
    add_bullet_point(tf_th3, "Screening Sensitivity", "A calibrated screening threshold of t = 0.10 provides the sensitivity required for actionable antimicrobial discovery.", color=SLATE_ACCENT)

    add_speaker_note(s12, (
        "Why calibrate the decision threshold to t = 0.10? Because training contained massive beta-lactamase superclusters, "
        "the validation and test sets were enriched in distinct, out-of-family resistance mechanisms. "
        "This biological domain shift compressed predicted probabilities downwards—the median positive probability was ~0.15. "
        "Using the arbitrary default threshold of 0.50 resulted in near-zero recall. "
        "We swept thresholds exclusively on the validation set to maximize F1-score, identifying t = 0.10 as optimal across all models. "
        "This threshold was frozen before evaluating the test set."
    ))

    # =========================================================================
    # SLIDE 13: Comprehensive Benchmark Results (Table + Insights)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Comprehensive Benchmark Results on Held-out Test Set (N=1,816)", "Experimental Results & Model Benchmarking")

    table_left = Inches(0.8)
    table_top = Inches(1.25)
    table_width = Inches(7.2)
    table_height = Inches(5.75)

    rows = 13
    cols = 4
    table_shape = s13.shapes.add_table(rows, cols, table_left, table_top, table_width, table_height)
    t = table_shape.table
    t.columns[0].width = Inches(2.7)
    t.columns[1].width = Inches(1.5)
    t.columns[2].width = Inches(1.5)
    t.columns[3].width = Inches(1.5)

    headers = ["Evaluation Metric", "Logistic Regression", "Random Forest (Best)", "SVM (RBF Kernel)"]
    for j, h in enumerate(headers):
        cell = t.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = "Arial"
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT

    table_data = [
        ("Accuracy", "51.65%", "71.97%", "57.54%"),
        ("Precision", "69.23%", "80.37%", "95.36%"),
        ("Recall (Sensitivity)", "5.95%", "58.15%", "15.86%"),
        ("Specificity", "97.36%", "85.79%", "99.23%"),
        ("F1-Score", "0.1095", "0.6748", "0.2720"),
        ("ROC-AUC", "0.6299", "0.8191", "0.7872"),
        ("PR-AUC (Avg Precision)", "0.6148", "0.8161", "0.7995"),
        ("Matthews Corr. (MCC)", "0.0815", "0.4572", "0.2732"),
        ("True Positives (TP)", "54", "528", "144"),
        ("True Negatives (TN)", "884", "779", "901"),
        ("False Positives (FP)", "24", "129", "7"),
        ("False Negatives (FN)", "854", "380", "764")
    ]
    for i, row in enumerate(table_data):
        for j, val in enumerate(row):
            cell = t.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(241, 245, 249) if i % 2 == 1 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Arial"
            p.font.size = Pt(9.5)
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
            if j == 2:
                p.font.bold = True
                p.font.color.rgb = SLATE_ACCENT

    c_res, tf_res = create_card(s13, Inches(8.3), table_top, Inches(4.233), table_height)
    add_card_header(tf_res, "Three Critical Analytical Takeaways")
    add_bullet_point(tf_res, "1. Random Forest Superiority", "Random Forest is the clear best overall classifier, achieving ROC-AUC of 0.8191, PR-AUC of 0.8161, F1-score of 0.6748, and MCC of 0.4572. It strikes the optimal biological balance, detecting 528 ARGs with 80.37% precision.")
    add_bullet_point(tf_res, "2. Deceptive Precision of SVM", "SVM achieves an astonishing 95.36% precision, but this is achieved through ultra-conservative under-prediction (Specificity 99.23%). It misses 764 out of 908 ARGs (Recall: 15.86%, F1: 0.2720), rendering it clinically ineffective as a screening tool.")
    add_bullet_point(tf_res, "3. Collapse of Linear Separation", "Logistic Regression completely collapses under homology reduction, missing 854 ARGs (Recall: 5.95%, MCC: 0.0815). This proves that resistance determinants cannot be linearly separated using global sequence frequencies.")

    add_speaker_note(s13, (
        "On the held-out test set of 1,816 sequences from 237 unseen superclusters, Random Forest achieved the best overall performance: "
        "ROC-AUC of 0.8191, PR-AUC of 0.8161, F1-score of 0.6748, and MCC of 0.4572. "
        "Notice that while SVM achieved 95.36% precision, it was an ultra-conservative model that missed 764 out of 908 ARGs (Recall: 15.86%). "
        "Logistic Regression collapsed to 5.95% recall, proving that linear models cannot separate resistance across remote homology boundaries."
    ))

    # =========================================================================
    # SLIDE 14: Visual Diagnostics - ROC, PR & Confusion Heatmaps
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Visual Diagnostics: ROC, Precision-Recall & Confusion Heatmaps", "Experimental Results & Visual Verification")

    left_fig_w = Inches(5.8)
    fig_roc_p = FIGURES_DIR / "roc_curves.png"
    fig_pr_p = FIGURES_DIR / "pr_curves.png"
    if fig_roc_p.exists():
        s14.shapes.add_picture(str(fig_roc_p), Inches(0.8), Inches(1.25), left_fig_w, Inches(2.75))
    if fig_pr_p.exists():
        s14.shapes.add_picture(str(fig_pr_p), Inches(0.8), Inches(4.15), left_fig_w, Inches(2.75))

    right_x = Inches(6.9)
    right_w = Inches(5.633)
    fig_cm_p = FIGURES_DIR / "confusion_matrices.png"
    if fig_cm_p.exists():
        s14.shapes.add_picture(str(fig_cm_p), right_x, Inches(1.25), right_w, Inches(2.2))

    c_diag, tf_diag = create_card(s14, right_x, Inches(3.6), right_w, Inches(3.4))
    add_card_header(tf_diag, "Diagnostic Interpretation of Curves & Matrices")
    add_bullet_point(tf_diag, "ROC Curve Dominance", "Random Forest (AUC = 0.819) consistently dominates SVM (0.787) and LogReg (0.630) across all false positive operating rates, proving superior ranking discrimination across unseen sequence families.")
    add_bullet_point(tf_diag, "PR Curve Robustness", "Against the 0.50 random prevalence baseline, Random Forest maintains high precision across substantial recall, achieving an Average Precision (PR-AUC) of 0.816.")
    add_bullet_point(tf_diag, "Confusion Matrix Trade-offs", "Random Forest correctly classifies 528 True Positives and 779 True Negatives. In contrast, SVM acts as a near-null predictor (764 FN), and Logistic Regression misses 854 ARGs.")

    add_speaker_note(s14, (
        "Slide 14 presents our visual diagnostics. On the left, the ROC and Precision-Recall curves show Random Forest's consistent dominance, "
        "achieving an ROC-AUC of 0.819 and PR-AUC of 0.816. "
        "On the right, the side-by-side confusion matrices visually illustrate the trade-offs: Random Forest achieves a balanced 528 True Positives "
        "and 779 True Negatives, whereas SVM and Logistic Regression produce massive false negative blocks."
    ))

    # =========================================================================
    # SLIDE 15: Interpretable Feature Analysis & Biochemical Drivers
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "Interpretable Feature Analysis & Biochemical Drivers", "Feature Interpretability & Biophysical Rationale")

    fig_fi_p = FIGURES_DIR / "feature_importance_top25.png"
    if fig_fi_p.exists():
        s15.shapes.add_picture(str(fig_fi_p), Inches(0.8), Inches(1.25), Inches(5.8), Inches(5.75))

    c_bio, tf_bio = create_card(s15, Inches(6.9), Inches(1.25), Inches(5.633), Inches(5.75))
    add_card_header(tf_bio, "Biochemical Drivers of Model Predictions")
    add_bullet_point(tf_bio, "1. Bulk Hydropathy (PC_gravy)", "GRAVY (Grand Average of Hydropathy) and Aliphatic Index rank among the top 10 most influential features in Random Forest. A large proportion of resistance proteins are membrane-spanning efflux pumps (RND, MFS, ABC) containing hydrophobic transmembrane alpha-helices with elevated aliphatic residues (Ala, Val, Ile, Leu).")
    add_bullet_point(tf_bio, "2. Proline & Glycine Dipeptides", "Dipeptides containing proline (DPC_PP, DPC_PA) and glycine (DPC_GG) were heavily weighted. Glycine and proline confer conformational flexibility in active-site loops of beta-lactamases and aminoglycoside transferases, allowing catalytic pockets to accommodate bulky antibiotic molecules.")
    add_bullet_point(tf_bio, "3. Electrostatic Properties (pI & Charge)", "Isoelectric point (pI) and Net Charge at pH 7.0 ranked prominently. Resistance enzymes often maintain basic surface residues to guide negatively charged antibiotic substrates toward catalytic clefts.")
    add_bullet_point(tf_bio, "4. Non-Linear Decision Splits", "Logistic Regression coefficients failed because resistance is not driven by a single amino acid. An active efflux pump requires both hydrophobic transmembrane anchors AND polar catalytic residues; decision trees capture these non-linear conditional splits naturally.")

    add_speaker_note(s15, (
        "Why does Random Forest succeed where linear models fail? "
        "First, bulk physicochemical descriptors like GRAVY hydropathy and aliphatic index are top predictors because many ARGs are "
        "membrane-bound active efflux pumps with hydrophobic transmembrane helices. "
        "Second, proline and glycine dipeptides provide conformational loop flexibility in catalytic clefts of beta-lactamases and transferases. "
        "Third, isoelectric point and net charge govern substrate attraction. "
        "Decision trees capture these multi-feature non-linear interactions naturally."
    ))

    # =========================================================================
    # SLIDE 16: Biological Error Analysis & Length Bias Audit
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    add_header(s16, "Biological Error Analysis & Sequence Length Bias Audit", "Model Limitations & Error Investigation")

    c_fp, tf_fp = create_card(s16, Inches(0.8), Inches(1.25), Inches(5.72), Inches(2.9), bg_color=CARD_BG_LIGHT)
    add_card_header(tf_fp, "1. False Positive Investigation (129 Cases)")
    add_bullet_point(tf_fp, "Physiological ABC Transporters (~65% of FPs)", "Proteins like PstB (phosphate import ATP-binding protein) and SapF (putrescine transport ATPase) share identical transmembrane topologies and Walker A/B motifs with multidrug resistance efflux pumps.")
    add_bullet_point(tf_fp, "Housekeeping Acetyltransferases", "Physiological bacterial acetyltransferases share active-site geometries and GNAT folds with aminoglycoside acetyltransferases (AACs), generating elevated dipeptide feature scores.")
    add_bullet_point(tf_fp, "Biological Reality", "The model detects genuine transport/catalytic architecture, but annotations classify them as metabolic rather than resistance pumps.")

    c_fn, tf_fn = create_card(s16, Inches(6.8), Inches(1.25), Inches(5.733), Inches(2.9), bg_color=CARD_BG_LIGHT)
    add_card_header(tf_fn, "2. False Negative Investigation (380 Cases)")
    add_bullet_point(tf_fn, "Homology Domain Shift", "Because 4,165 training sequences were beta-lactamases, the test set was enriched in rare, atypical resistance families whose direct homologs were absent from training data.")
    add_bullet_point(tf_fn, "Ribosomal Protection (tet(M), cfr)", "Ribosomal RNA methyltransferases (cfr) and protection proteins (tet(M)) act on the ribosome rather than degrading antibiotics, displaying distinct sequence signatures.")
    add_bullet_point(tf_fn, "Metabolic Bypass (dfrA)", "Trimethoprim-resistant dihydrofolate reductases are metabolic enzymes with minor active-site modifications that bypass drug inhibition.")
    add_bullet_point(tf_fn, "The 58.15% Recall Ceiling", "A global 428-descriptor model trained predominantly on beta-lactamases cannot resolve all divergent catalytic nuances without 3D structural coordinates.")

    fig_len_p = FIGURES_DIR / "sequence_length_distribution_predictions.png"
    if fig_len_p.exists():
        s16.shapes.add_picture(str(fig_len_p), Inches(0.8), Inches(4.3), Inches(5.72), Inches(2.7))

    c_len, tf_len = create_card(s16, Inches(6.8), Inches(4.3), Inches(5.733), Inches(2.7))
    add_card_header(tf_len, "3. Descriptive Sequence Length Audit")
    add_bullet_point(tf_len, "Median Across Outcomes", "True Positives: 290 aa; True Negatives: 358 aa; False Positives: 280 aa; False Negatives: 290 aa.")
    add_bullet_point(tf_len, "Significant Overlap", "The substantial overlap in sequence length distributions across all four outcomes confirms the classifier does not rely on a monotonic sequence length shortcut.")
    add_bullet_point(tf_len, "Honest Scientific Conclusion", "While multi-feature non-linear length interactions cannot be completely ruled out, length alone is not a primary confounding discriminator.")

    add_speaker_note(s16, (
        "In our error analysis: Over 65% of False Positives are physiological ABC transporters (like PstB and SapF) that share "
        "identical transmembrane topologies and Walker A/B motifs with multidrug efflux pumps. "
        "False Negatives were concentrated in atypical mechanisms like cfr rRNA methyltransferases, tet(M) ribosomal protection, "
        "and dfrA dihydrofolate reductases that lacked training representation. "
        "Our sequence length distribution audit confirms substantial overlap across all outcomes, proving the model did not learn sequence length as a shortcut."
    ))

    # =========================================================================
    # SLIDE 17: Live Dashboard, Clinical Governance & Future Horizons
    # =========================================================================
    s17 = prs.slides.add_slide(blank_layout)
    add_header(s17, "Live Interactive Dashboard, Clinical Governance & Future Horizons", "Deployment, Governance & Roadmap")

    c_dash, tf_dash = create_card(s17, Inches(0.8), card_top, col_w3, card_h)
    add_card_header(tf_dash, "1. Live Research Dashboard")
    add_bullet_point(tf_dash, "Production Streamlit App", "Developed a complete 6-page interactive web application (app.py) enabling researchers to explore models and test live sequences in real time.")
    add_bullet_point(tf_dash, "Live Sequence Predictor", "Accepts raw FASTA sequences, validates canonical alphabets, extracts 428 features, scales via frozen parameters, and computes multi-model probabilities with radar charts.")
    add_bullet_point(tf_dash, "Interactive Exploration", "Provides searchable CARD/UniProt browsers, interactive ROC/PR curves, confusion matrices, and feature importance rankings.")
    add_bullet_point(tf_dash, "Complete Documentation", "Embeds all 24 viva defense Q&As, exclusion logs, and reproducibility guides directly in the UI.")

    c_gov, tf_gov = create_card(s17, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h, bg_color=BG_RED_TINT, border_color=BORDER_RED)
    add_card_header(tf_gov, "2. Clinical Governance Caveat", color=TEXT_RED)
    add_bullet_point(tf_gov, "Computational Hypotheses Only", "In silico machine learning models produce predictive screening hypotheses, not clinical diagnoses or phenotypic confirmations.", color=TEXT_RED)
    add_bullet_point(tf_gov, "Phenotypic AST Mandatory", "In clinical microbiology, antimicrobial resistance is determined phenotypically by measuring Minimum Inhibitory Concentrations (MICs) via broth microdilution against CLSI/EUCAST breakpoints.", color=TEXT_RED)
    add_bullet_point(tf_gov, "Expression vs Presence", "A computational prediction of 'ARG' indicates sequence and biophysical propensity; it does not confirm active in vivo gene transcription, proper folding, or adequate expression levels.", color=TEXT_RED)
    add_bullet_point(tf_gov, "Screening Role", "Our pipeline serves as an early computational sieve to prioritize high-risk metagenomic contigs for phenotypic validation.", color=TEXT_RED)

    c_fut, tf_fut = create_card(s17, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h, bg_color=BG_GREEN_TINT, border_color=BORDER_GREEN)
    add_card_header(tf_fut, "3. Future Methodological Roadmap", color=TEXT_GREEN)
    add_bullet_point(tf_fut, "Protein Language Models", "Replace alignment-free hand-crafted features with 1280-dimensional ESM-2 transformer embeddings. ESM-2 captures 3D structural fold contacts and active-site cleft geometries directly from primary sequences, potentially improving out-of-family recall beyond 58.15%.", color=TEXT_GREEN)
    add_bullet_point(tf_fut, "Multi-Class Mechanism Split", "Decompose binary classification into multi-class prediction (Enzymatic vs Efflux vs Target Protection vs Bypass) so distinct biochemical architectures do not compete in a single feature space.", color=TEXT_GREEN)
    add_bullet_point(tf_fut, "Hybrid Ensemble Pipeline", "Combine ultra-fast BLAST searches (>80% identity) for known clinical variants with machine learning models for remote environmental homologs (<30% identity).", color=TEXT_GREEN)

    add_speaker_note(s17, (
        "In our final slide, we present our deployment architecture, clinical governance statement, and future roadmap. "
        "We built and verified a 6-page interactive Streamlit dashboard allowing users to input novel FASTA sequences and receive instant multi-model predictions, "
        "feature radar charts, and threshold comparisons. "
        "We conclude with an essential clinical governance reminder: In silico predictions are computational hypotheses, not clinical diagnoses. "
        "Antimicrobial resistance must always be confirmed through laboratory phenotypic susceptibility testing against CLSI and EUCAST breakpoints. "
        "Our model serves as a rapid screening tool to prioritize uncharacterized metagenomic contigs. "
        "Looking forward, integrating pre-trained protein language models like ESM-2, decomposing the task into multi-class mechanism classification, "
        "and deploying a hybrid BLAST-plus-ML pipeline represent the most promising frontiers for antimicrobial resistance surveillance. "
        "Thank you, and we welcome your questions."
    ))

    prs.save(str(pptx_path))
    print(f"Presentation successfully created with {len(prs.slides)} slides at: {pptx_path}")
    return pptx_path


if __name__ == "__main__":
    out_path = build_presentation()
    sys.exit(0)
