"""
Module: generate_presentation_clean.py
Objective: Build a clean, streamlined 17-slide academic PowerPoint presentation (.pptx)
in presentation/final_presentation.pptx.
Significantly reduced detailing: crisp, concise, high-impact bullet points, large fonts,
generous breathing room, zero clutter, and no top banner.
"""

import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

OUT_DIR = Path("presentation")
FIGURES_DIR = Path("results/figures")


def build_clean_presentation():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pptx_path = OUT_DIR / "final_presentation.pptx"

    prs = Presentation()
    # 16:9 Widescreen layout: 13.333 x 7.5 inches
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    # Clean Modern Palette
    NAVY = RGBColor(15, 23, 42)          # #0F172A
    SLATE = RGBColor(30, 58, 138)        # #1E3A8A
    TEAL = RGBColor(13, 148, 136)        # #0D9488
    DARK_TEXT = RGBColor(51, 65, 85)     # #334155
    WHITE = RGBColor(255, 255, 255)
    CARD_BG = RGBColor(255, 255, 255)
    CARD_BG_ALT = RGBColor(248, 250, 252) # #F8FAFC
    BORDER_COLOR = RGBColor(226, 232, 240) # #E2E8F0

    BG_GREEN_TINT = RGBColor(240, 253, 244) # #F0FDF4
    BORDER_GREEN = RGBColor(187, 247, 208)  # #BBF7D0
    TEXT_GREEN = RGBColor(22, 101, 52)      # #166534

    BG_RED_TINT = RGBColor(254, 242, 242)   # #FEF2F2
    BORDER_RED = RGBColor(254, 202, 202)    # #FECACA
    TEXT_RED = RGBColor(153, 27, 27)        # #991B1B

    BG_BLUE_TINT = RGBColor(239, 246, 255)  # #EFF6FF
    BORDER_BLUE = RGBColor(191, 219, 254)   # #BFDBFE

    def add_header(slide, title_text, category_breadcrumb=None):
        """Top clean header without any senior bioinformatics banner."""
        # Clean accent bar
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.08))
        bar.fill.solid()
        bar.fill.fore_color.rgb = SLATE
        bar.line.fill.background()

        # Breadcrumb
        if category_breadcrumb:
            b_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.24), Inches(11.733), Inches(0.25))
            tf_b = b_box.text_frame
            tf_b.word_wrap = True
            tf_b.margin_left = tf_b.margin_right = tf_b.margin_top = tf_b.margin_bottom = 0
            p_b = tf_b.paragraphs[0]
            p_b.text = category_breadcrumb.upper()
            p_b.font.name = "Arial"
            p_b.font.size = Pt(10)
            p_b.font.bold = True
            p_b.font.color.rgb = TEAL

        # Title
        top_offset = Inches(0.48) if category_breadcrumb else Inches(0.35)
        t_box = slide.shapes.add_textbox(Inches(0.8), top_offset, Inches(11.733), Inches(0.65))
        tf_t = t_box.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_right = tf_t.margin_top = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.name = "Arial"
        p_t.font.size = Pt(22)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY

    def create_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=BORDER_COLOR):
        """Creates a stylized, clean card container."""
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.2)
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_top = Inches(0.22)
        tf.margin_bottom = Inches(0.22)
        tf.margin_left = Inches(0.25)
        tf.margin_right = Inches(0.25)
        return card, tf

    def add_card_header(text_frame, title_text, font_size=14, color=NAVY):
        p = text_frame.paragraphs[0]
        p.text = title_text
        p.font.name = "Arial"
        p.font.size = Pt(font_size)
        p.font.bold = True
        p.font.color.rgb = color
        p.space_after = Pt(10)

    def add_bullet(text_frame, bold_prefix, text, font_size=11, space_before=8, color=DARK_TEXT):
        p = text_frame.add_paragraph()
        p.font.name = "Arial"
        p.font.size = Pt(font_size)
        p.space_before = Pt(space_before)
        
        run_b = p.add_run()
        run_b.text = f"• {bold_prefix}: "
        run_b.font.bold = True
        run_b.font.color.rgb = NAVY

        run_t = p.add_run()
        run_t.text = text
        run_t.font.bold = False
        run_t.font.color.rgb = color

    def add_notes(slide, note):
        notes_slide = slide.notes_slide
        tf = notes_slide.notes_text_frame
        tf.text = note.strip()

    card_top = Inches(1.25)
    card_h = Inches(5.75)
    col_w3 = Inches(3.72)
    gap3 = Inches(0.28)
    col_w2 = Inches(5.72)
    gap2 = Inches(0.29)

    # =========================================================================
    # SLIDE 1: Title & Team
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    t_box = s1.shapes.add_textbox(Inches(0.9), Inches(1.2), Inches(11.533), Inches(2.2))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Machine Learning Classification of\nAntibiotic Resistance Genes (ARGs) vs Non-ARGs"
    p1.font.name = "Arial"
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = WHITE

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "A Rigorous, Homology-Reduced Protein Sequence Benchmark"
    p1_sub.font.name = "Arial"
    p1_sub.font.size = Pt(16)
    p1_sub.font.color.rgb = RGBColor(147, 197, 253)
    p1_sub.space_before = Pt(10)

    team_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.9), Inches(3.8), Inches(11.533), Inches(2.8))
    team_card.fill.solid()
    team_card.fill.fore_color.rgb = RGBColor(30, 41, 59)
    team_card.line.color.rgb = RGBColor(59, 130, 246)
    team_card.line.width = Pt(1.5)
    tf_team = team_card.text_frame
    tf_team.word_wrap = True
    tf_team.margin_top = Inches(0.22)
    tf_team.margin_left = Inches(0.3)
    tf_team.margin_right = Inches(0.3)

    p_th = tf_team.paragraphs[0]
    p_th.text = "PROJECT TEAM MEMBERS & RESPONSIBILITIES"
    p_th.font.name = "Arial"
    p_th.font.size = Pt(13)
    p_th.font.bold = True
    p_th.font.color.rgb = WHITE
    p_th.space_after = Pt(10)

    team = [
        ("Madhav Mendiratta", "Data acquisition (CARD & Swiss-Prot), sequence hygiene, collision audit & length matching."),
        ("Rakshan", "428-feature extraction (AAC, DPC, physicochemical), MMseqs2 clustering & MILP split optimization."),
        ("Ankit", "Machine learning model development, 5-fold GroupKFold cross-validation & threshold calibration."),
        ("Harsh", "Held-out test evaluation, multi-metric benchmarking (ROC/PR/MCC) & biological error analysis.")
    ]
    for m, r in team:
        pm = tf_team.add_paragraph()
        pm.font.name = "Arial"
        pm.font.size = Pt(11)
        pm.space_before = Pt(6)
        r_b = pm.add_run()
        r_b.text = f"• {m}: "
        r_b.font.bold = True
        r_b.font.color.rgb = RGBColor(253, 224, 71)
        r_t = pm.add_run()
        r_t.text = r
        r_t.font.bold = False
        r_t.font.color.rgb = RGBColor(226, 232, 240)

    add_notes(s1, "Welcome committee members. Today we present our capstone benchmark on homology-reduced ARG classification.")

    # =========================================================================
    # SLIDE 2: AMR Crisis & 4 Biological Mechanisms
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "The Global AMR Crisis & Biological Resistance Mechanisms", "Biological Foundations")

    c1, tf1 = create_card(s2, Inches(0.8), card_top, col_w3, card_h)
    add_card_header(tf1, "1. The Global Threat")
    add_bullet(tf1, "10M Deaths/Year by 2050", "Projected to surpass cancer globally with >$100T economic impact.")
    add_bullet(tf1, "Compromised Medicine", "Chemotherapy, transplants, and surgeries rely on effective antibiotics.")
    add_bullet(tf1, "ESKAPE Pathogens", "Multi-drug resistant hospital strains (e.g., K. pneumoniae, P. aeruginosa).")
    add_bullet(tf1, "24–72h AST Delay", "Culture-based testing is slow; rapid sequence-based pre-screening is critical.")

    c2, tf2 = create_card(s2, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h)
    add_card_header(tf2, "2. Four Biochemical Paradigms")
    add_bullet(tf2, "Enzymatic Inactivation", "Catalytic cleavage (e.g., beta-lactamases, aminoglycoside transferases).")
    add_bullet(tf2, "Active Efflux Systems", "Membrane transport pumps (ABC, MFS, RND) that export antibiotics.")
    add_bullet(tf2, "Target Protection", "Structural alteration preventing drug binding (e.g., tet(M), cfr, mecA).")
    add_bullet(tf2, "Target Bypass", "Drug-insensitive metabolic enzymes that bypass the block (e.g., sul, dfrA).")

    c3, tf3 = create_card(s2, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h)
    add_card_header(tf3, "3. Computational ML Role")
    add_bullet(tf3, "Metagenomic Expansion", "Sequencing produces vast uncharacterized contigs from clinics and soils.")
    add_bullet(tf3, "Alignment Bottleneck", "BLAST fails when novel genes share low identity with reference databases.")
    add_bullet(tf3, "Feature-Based ML", "Captures generalized biophysical rules directly from amino acid sequences.")
    add_bullet(tf3, "Rapid Screening", "Prioritizes high-risk candidate genes for targeted laboratory confirmation.")

    add_notes(s2, "AMR threatens modern medicine. Bacteria use 4 biochemical strategies. ML offers fast screening where BLAST fails.")

    # =========================================================================
    # SLIDE 3: Research Gap - BLAST Twilight Zone & Homology Over-Optimism
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "The Critical Research Gap: Why BLAST & Naive ML Models Fail", "Problem Formulation")

    c_blast, tf_b = create_card(s3, Inches(0.8), card_top, col_w2, card_h)
    add_card_header(tf_b, "The Alignment Bottleneck (BLAST)", color=SLATE)
    add_bullet(tf_b, "High Identity Dependency", "BLAST works well above 80% sequence identity but fails on remote homologs.")
    add_bullet(tf_b, "The 20–35% Twilight Zone", "In protein evolution, folds diverge slower than sequences; alignment loses sensitivity.")
    add_bullet(tf_b, "Novel Resistomes Invisible", "Divergent genes mobilized from environmental reservoirs escape standard detection.")
    add_bullet(tf_b, "Compute Overhead", "Full-database alignment searches are computationally slow on big metagenomes.")

    c_leak, tf_l = create_card(s3, Inches(0.8 + col_w2 + gap2), card_top, col_w2, card_h, bg_color=BG_RED_TINT, border_color=BORDER_RED)
    add_card_header(tf_l, "Homology Over-Optimism in Literature", color=TEXT_RED)
    add_bullet(tf_l, "Naive Random Splits", "Studies routinely use random 80:20 splits, claiming inflated >95–99% accuracy.", color=TEXT_RED)
    add_bullet(tf_l, "Strain-Level Redundancy", "CARD contains hundreds of near-identical TEM/CTX-M variants (>98% identity).", color=TEXT_RED)
    add_bullet(tf_l, "Severe Data Leakage", "Random splits place variants in both train and test; models simply memorize strain motifs.", color=TEXT_RED)
    add_bullet(tf_l, "Deployment Failure", "Models collapse completely when evaluated against unrepresented resistance families.", color=TEXT_RED)

    add_notes(s3, "BLAST fails in the twilight zone. Naive ML splits suffer from homology leakage, producing fake 99% accuracies.")

    # =========================================================================
    # SLIDE 4: Objectives & Novelties
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Project Objectives & Core Scientific Novelties", "Objectives & Novelty")

    c_obj, tf_o = create_card(s4, Inches(0.8), card_top, col_w2, card_h)
    add_card_header(tf_o, "Five Project Objectives", color=SLATE)
    add_bullet(tf_o, "1. Curate Unbiased Dataset", "1:1 balanced cohort (6,054 ARGs vs 6,054 non-ARGs) with matched length bins.")
    add_bullet(tf_o, "2. Homology Clustering", "Cluster at 30% identity using MMseqs2, condensing into 3,631 superclusters.")
    add_bullet(tf_o, "3. Exact MILP Partitioning", "Solve Mixed-Integer Linear Program for exact 70:15:15 split with zero cluster splitting.")
    add_bullet(tf_o, "4. 428-Dimensional Features", "Extract AAC, DPC, and physicochemical properties with train-only scaling.")
    add_bullet(tf_o, "5. Honest Benchmarking", "Evaluate LogReg, Random Forest, and SVM under 5-fold GroupKFold and calibrated threshold.")

    c_nov, tf_n = create_card(s4, Inches(0.8 + col_w2 + gap2), card_top, col_w2, card_h, bg_color=BG_GREEN_TINT, border_color=BORDER_GREEN)
    add_card_header(tf_n, "Four Key Scientific Novelties", color=TEXT_GREEN)
    add_bullet(tf_n, "1. Zero-Leakage Guarantee", "Maximum cross-partition identity capped at 36.2% (0 hits >50%).", color=TEXT_GREEN)
    add_bullet(tf_n, "2. Exact MILP Formulation", "First exact optimization solving CARD's severe cluster-size skew without splitting.", color=TEXT_GREEN)
    add_bullet(tf_n, "3. Collision-Audited Negatives", "Purged 8 identical Swiss-Prot collisions and matched 10 length-quantile bins.", color=TEXT_GREEN)
    add_bullet(tf_n, "4. Calibrated Decision Threshold", "Calibrated t=0.10 strictly on validation data to counter domain-shift probability drop.", color=TEXT_GREEN)

    add_notes(s4, "We set 5 rigorous objectives and established 4 key novelties, guaranteeing zero homology leakage.")

    # =========================================================================
    # SLIDE 5: Positive Ground Truth - CARD v4.0.2
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Positive Ground Truth Curation: CARD v4.0.2 Dataset", "Data Curation")

    c1, tf1 = create_card(s5, Inches(0.8), card_top, col_w3, card_h)
    add_card_header(tf1, "1. CARD v4.0.2 Database")
    add_bullet(tf1, "Gold-Standard Resource", "Maintained by McMaster University; curated peer-reviewed reference.")
    add_bullet(tf1, "ARO Ontology", "Structured around the Antibiotic Resistance Ontology (mechanisms & drug targets).")
    add_bullet(tf1, "6,059 Raw Proteins", "Downloaded complete curated reference protein homolog sequences.")

    c2, tf2 = create_card(s5, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h, bg_color=CARD_BG_ALT)
    add_card_header(tf2, "2. Excluding Variant Models")
    add_bullet(tf2, "PHMs (Retained)", "Dedicated acquired resistance genes (e.g., beta-lactamases, efflux pumps).")
    add_bullet(tf2, "PVMs (Excluded)", "Housekeeping genes where single SNPs cause resistance (e.g., gyrA S83L).")
    add_bullet(tf2, ">99.8% Identity", "Susceptible wild-type and mutant gyrA differ by only 1 amino acid out of 800+.")
    add_bullet(tf2, "Severe Label Noise", "Global composition cannot resolve single point mutations; excluding PVMs ensures clean data.")

    c3, tf3 = create_card(s5, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h)
    add_card_header(tf3, "3. Positive Sequence QC")
    add_bullet(tf3, "Canonical Alphabet", "Enforced 20 standard amino acids; removed ambiguous characters (X, B, Z, *).")
    add_bullet(tf3, "Length Boundaries", "Restricted to functional proteins: 50 <= Length <= 1,500 amino acids.")
    add_bullet(tf3, "Final Clean Pool", "Purged 5 sequences during QC, leaving exactly 6,054 verified functional ARGs.")

    add_notes(s5, "We used CARD PHMs and excluded point-mutation variant models to prevent extreme label noise.")

    # =========================================================================
    # SLIDE 6: Negative Cohort Engineering
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Negative Cohort Engineering & Sequence Hygiene", "Data Curation")

    c1, tf1 = create_card(s6, Inches(0.8), card_top, col_w3, card_h)
    add_card_header(tf1, "1. Reviewed Bacterial Set")
    add_bullet(tf1, "Swiss-Prot Negatives", "Sourced from reviewed bacterial entries (taxonomy_id:2), avoiding eukaryotic bias.")
    add_bullet(tf1, "Strict REST Query", "Filtered out resistance keywords, beta-lactamases, efflux pumps, and GO terms.")
    add_bullet(tf1, "6,800 Candidates", "Harvested initial reviewed candidate pool for programmatic quality control.")

    c2, tf2 = create_card(s6, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h, bg_color=BG_RED_TINT, border_color=BORDER_RED)
    add_card_header(tf2, "2. Collision Audit (8 Purged)", color=TEXT_RED)
    add_bullet(tf2, "Label Contamination", "Swiss-Prot contains resistance proteins that occasionally lack explicit resistance tags.", color=TEXT_RED)
    add_bullet(tf2, "Exact String Audit", "Automated string comparison between 6,800 candidates and 6,054 CARD ARGs.", color=TEXT_RED)
    add_bullet(tf2, "8 Exact Collisions", "Identified 8 identical matches: EmrE transporter, MdtC pump, and CpxA kinase.", color=TEXT_RED)
    add_bullet(tf2, "Immediate Purge", "Purged all 8 colliding sequences to eliminate false-negative contamination.", color=TEXT_RED)

    c3, tf3 = create_card(s6, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h)
    add_card_header(tf3, "3. 10-Bin Length Matching")
    add_bullet(tf3, "Length Discrepancy", "CARD ARGs median is 290 aa; raw Swiss-Prot median is 358 aa.")
    add_bullet(tf3, "Quantile Sampling", "Partitioned positive lengths into 10 quantiles and sampled negatives to match.")
    add_bullet(tf3, "Exact 1:1 Balance", "Produced an exact 1:1 balanced master dataset: 6,054 ARGs vs 6,054 Non-ARGs (12,108 total).")

    add_notes(s6, "We curated bacterial Swiss-Prot negatives, purged 8 colliding ARGs, and matched 10 length bins for a clean 1:1 balance.")

    # =========================================================================
    # SLIDE 7: Homology Reduction & MMseqs2 Clustering
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Homology Reduction & MMseqs2 Two-Stage Clustering", "Clustering & Leakage Prevention")

    c1, tf1 = create_card(s7, Inches(0.8), card_top, col_w3, card_h)
    add_card_header(tf1, "1. 30% Twilight Zone")
    add_bullet(tf1, "Twilight Zone Boundary", "30% sequence identity is the established limit for functional sequence alignment.")
    add_bullet(tf1, "True Generalization", "Forces models to generalize across remote families rather than memorizing motifs.")
    add_bullet(tf1, "80% Coverage", "Required 80% bidirectional alignment coverage to guarantee full-domain homology.")

    c2, tf2 = create_card(s7, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h)
    add_card_header(tf2, "2. Stage 1: Easy-Cluster")
    add_bullet(tf2, "MMseqs2 Engine", "Clustered 12,108 sequences at 30% identity and 80% coverage.")
    add_bullet(tf2, "3,936 Initial Clusters", "Generated 3,936 clusters represented by centroid sequences.")
    add_bullet(tf2, "Severe Size Skew", "2,750 clusters were singletons, while large beta-lactamase clusters had >2,000 sequences.")

    c3, tf3 = create_card(s7, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h, bg_color=CARD_BG_ALT)
    add_card_header(tf3, "3. Stage 2: Superclustering")
    add_bullet(tf3, "Centroid Alignment", "Performed all-against-all alignment across all 3,936 centroids.")
    add_bullet(tf3, "Graph Construction", "Connected clusters sharing >= 30% identity at >= 50% coverage.")
    add_bullet(tf3, "3,631 Superclusters", "Union-Find condensation merged components into 3,631 disjoint superclusters.")

    add_notes(s7, "MMseqs2 easy-cluster plus Union-Find superclustering condensed 12,108 sequences into 3,631 isolated superclusters.")

    # =========================================================================
    # SLIDE 8: MILP Group-Aware Partitioning
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Exact 70:15:15 Group-Aware Partitioning via MILP", "Partition Optimization")

    c1, tf1 = create_card(s8, Inches(0.8), card_top, col_w3, card_h, bg_color=BG_RED_TINT, border_color=BORDER_RED)
    add_card_header(tf1, "1. Skewed Cluster Dilemma", color=TEXT_RED)
    add_bullet(tf1, "Massive Clusters", "3 beta-lactamase superclusters contain 4,165 sequences in CARD.", color=TEXT_RED)
    add_bullet(tf1, "Zero-Split Rule", "Whole clusters must stay together in a single partition to prevent leakage.", color=TEXT_RED)
    add_bullet(tf1, "Greedy Failure", "Heuristic binning left only 10.77% in test set (target: 15.00%).", color=TEXT_RED)

    c2, tf2 = create_card(s8, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h)
    add_card_header(tf2, "2. MILP Formulation")
    add_bullet(tf2, "Exact Optimization", "Formulated cluster assignment as a Mixed-Integer Linear Program via scipy.optimize.milp.")
    add_bullet(tf2, "Binary Variables", "Decision variable x_{c,p} in {0, 1} assigns cluster c to partition p in {train, val, test}.")
    add_bullet(tf2, "Objective", "Minimize partition size deviations and class imbalances with penalty lambda = 2.0.")

    c3, tf3 = create_card(s8, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h, bg_color=BG_GREEN_TINT, border_color=BORDER_GREEN)
    add_card_header(tf3, "3. Mathematically Exact Split", color=TEXT_GREEN)
    add_bullet(tf3, "Train (70.00%)", "8,476 sequences: 4,238 Pos / 4,238 Neg; 3,126 clusters.", color=TEXT_GREEN)
    add_bullet(tf3, "Validation (15.00%)", "1,816 sequences: 908 Pos / 908 Neg; 268 clusters.", color=TEXT_GREEN)
    add_bullet(tf3, "Held-Out Test (15.00%)", "1,816 sequences: 908 Pos / 908 Neg; 237 clusters.", color=TEXT_GREEN)
    add_bullet(tf3, "Zero Leakage", "100% cluster-isolated; zero cluster splitting across boundaries.", color=TEXT_GREEN)

    add_notes(s8, "Greedy binning produced an imbalanced 10.77% test set. MILP solved it, achieving exactly 70:15:15 with zero cluster splitting.")

    # =========================================================================
    # SLIDE 9: Similarity Audit & Mixed Clusters
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Cross-Partition Similarity Audit & Mixed Cluster Investigation", "Auditing & Robustness")

    c1, tf1 = create_card(s9, Inches(0.8), card_top, col_w3, card_h)
    add_card_header(tf1, "1. Cross-Partition Audit")
    add_bullet(tf1, "MMseqs2 Alignment", "Aligned all 1,816 test sequences against all 8,476 training sequences.")
    add_bullet(tf1, "Zero Hits >50% Identity", "Zero test sequences share >50% sequence identity with training data.")
    add_bullet(tf1, "Max Identity: 36.2%", "Maximum cross-partition identity is 36.2% (remote class-A beta-lactamase variants).")

    c2, tf2 = create_card(s9, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h, bg_color=CARD_BG_ALT)
    add_card_header(tf2, "2. Audit Scope & Limits")
    add_bullet(tf2, "Eliminates Memorization", "Proves models cannot exploit trivial sequence identity shortcuts.")
    add_bullet(tf2, "3D Folds Undetected", "Cannot rule out 3D structural fold conservation (<20% identity).")
    add_bullet(tf2, "Short Motifs", "Cannot rule out short active-site motifs (e.g. Walker A/B) below 80% coverage.")

    c3, tf3 = create_card(s9, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h)
    add_card_header(tf3, "3. 50 Mixed Superclusters")
    add_bullet(tf3, "Dual-Function Biology", "50 clusters (475 sequences) contain both ARG and non-ARG annotations.")
    add_bullet(tf3, "Porins & Transporters", "E.g., OmpF/OmpC porins and ABC transporters with dual metabolic roles.")
    add_bullet(tf3, "Kept Intact", "Retained intact within single partitions without silent relabeling.")

    add_notes(s9, "Max identity between test and train is 36.2% (0 hits >50%). We also investigated 50 mixed clusters and kept them intact.")

    # =========================================================================
    # SLIDE 10: 428-Dimensional Features & Scaling
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "428-Dimensional Alignment-Free Feature Representation", "Feature Engineering")

    c1, tf1 = create_card(s10, Inches(0.8), card_top, col_w3, card_h)
    add_card_header(tf1, "1. 20 AAC Features")
    add_bullet(tf1, "Normalized Frequency", "Frequency of 20 canonical amino acids: f(a) = N(a) / L.")
    add_bullet(tf1, "Length Invariant", "Captures global structural propensities (e.g. membrane hydrophobicity).")
    add_bullet(tf1, "Fast Extraction", "Computed in microseconds without multiple sequence alignments.")

    c2, tf2 = create_card(s10, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h)
    add_card_header(tf2, "2. 400 DPC Features")
    add_bullet(tf2, "Dipeptide Pairs", "Frequencies of all 20 x 20 adjacent pairs: f(d) = N(d) / (L - 1).")
    add_bullet(tf2, "Secondary Structure", "Encodes alpha-helices, beta-sheets, beta-turns, and loop flexibility.")
    add_bullet(tf2, "Catalytic Motifs", "Captures local active-site dyads without requiring 3D coordinates.")

    c3, tf3 = create_card(s10, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h, bg_color=CARD_BG_ALT)
    add_card_header(tf3, "3. 8 Physicochemical & Scaling")
    add_bullet(tf3, "Biochemical Descriptors", "Mw, pI, length, net charge, GRAVY hydropathy, aliphatic index, aromaticity.")
    add_bullet(tf3, "Train-Only Scaling", "StandardScaler fitted strictly on X_train (mu=0, sigma=1).")
    add_bullet(tf3, "Frozen Parameters", "Validation and test sets transformed using frozen training scaler (zero leakage).")

    add_notes(s10, "We extracted 428 alignment-free features (20 AAC, 400 DPC, 8 physicochemical) with strict train-only standardization.")

    # =========================================================================
    # SLIDE 11: Models & 5-Fold GroupKFold CV
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Machine Learning Architectures & Group-Aware Validation", "Model Development")

    c1, tf1 = create_card(s11, Inches(0.8), card_top, col_w3, card_h)
    add_card_header(tf1, "1. Three ML Paradigms")
    add_bullet(tf1, "Logistic Regression", "Linear baseline with L2 Ridge penalty ((1/2C)||beta||_2^2).")
    add_bullet(tf1, "Random Forest", "Ensemble of 100 decorrelated trees with Gini impurity splits.")
    add_bullet(tf1, "SVM (RBF Kernel)", "Non-linear maximum-margin classifier with 1D Platt probability scaling.")

    c2, tf2 = create_card(s11, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h, bg_color=CARD_BG_ALT)
    add_card_header(tf2, "2. 5-Fold GroupKFold")
    add_bullet(tf2, "Cluster-Grouped Folds", "Folds grouped strictly by cluster_id across the 8,476 training sequences.")
    add_bullet(tf2, "Zero Fold Leakage", "Guarantees superclusters are never split across training and validation folds.")
    add_bullet(tf2, "Honest Tuning", "Cross-validation scores reflect true out-of-cluster generalization.")

    c3, tf3 = create_card(s11, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h)
    add_card_header(tf3, "3. Best Configurations")
    add_bullet(tf3, "LogReg", "C = 10.0 (mean CV ROC-AUC: 0.7998).")
    add_bullet(tf3, "Random Forest", "N_trees = 100, depth = 15, split = 5 (mean CV ROC-AUC: 0.7102).")
    add_bullet(tf3, "SVM", "C = 0.5, gamma = 'auto' (mean CV ROC-AUC: 0.7741). Test set kept locked.")

    add_notes(s11, "We compared LogReg, Random Forest, and SVM under 5-fold GroupKFold to optimize hyperparameters without test set leakage.")

    # =========================================================================
    # SLIDE 12: Threshold Calibration (t = 0.10)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Validation Decision Threshold Calibration (t = 0.10 Rationale)", "Model Calibration")

    c1, tf1 = create_card(s12, Inches(0.8), card_top, col_w3, card_h)
    add_card_header(tf1, "1. Probability Compression")
    add_bullet(tf1, "Domain Shift", "Training absorbed massive beta-lactamases; test set has atypical, divergent mechanisms.")
    add_bullet(tf1, "Compressed Scores", "Model predicted probabilities compressed downwards (median positive probability: ~0.15).")
    add_bullet(tf1, "Default Collapse", "Using default t = 0.50 resulted in near-zero recall (<2% for LogReg).")

    c2, tf2 = create_card(s12, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h, bg_color=CARD_BG_ALT)
    add_card_header(tf2, "2. Validation Optimization")
    add_bullet(tf2, "Validation Sweep", "Swept thresholds t in [0.10, 0.90] exclusively on 1,816 validation sequences.")
    add_bullet(tf2, "F1-Score Maximization", "Selected t = 0.10 to maximize validation F1 across all three classifiers.")
    add_bullet(tf2, "Val F1 Gains", "Random Forest Val F1 peaked at 0.5371 (vs 0.3542 at default t = 0.50).")

    c3, tf3 = create_card(s12, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h, bg_color=BG_BLUE_TINT, border_color=BORDER_BLUE)
    add_card_header(tf3, "3. Threshold Freezing", color=SLATE)
    add_bullet(tf3, "Zero Test Peeking", "Threshold t = 0.10 frozen in model metadata before touching the held-out test set.", color=SLATE)
    add_bullet(tf3, "Screening Priority", "In clinical screening, false negatives carry far higher risk than false positives.", color=SLATE)
    add_bullet(tf3, "High Sensitivity", "t = 0.10 delivers necessary sensitivity for prospective resistance surveillance.", color=SLATE)

    add_notes(s12, "Out-of-family generalization compressed predicted probabilities. Sweeping validation data proved t=0.10 maximizes F1.")

    # =========================================================================
    # SLIDE 13: Comprehensive Benchmark Results (Table + Insights)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Comprehensive Benchmark Results on Held-out Test Set (N=1,816)", "Benchmark Results")

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
                p.font.color.rgb = SLATE

    c_res, tf_r = create_card(s13, Inches(8.3), table_top, Inches(4.233), table_height)
    add_card_header(tf_r, "Key Benchmark Insights")
    add_bullet(tf_r, "Random Forest Best Overall", "Achieves ROC-AUC of 0.8191, PR-AUC of 0.8161, F1 of 0.6748, and MCC of 0.4572.")
    add_bullet(tf_r, "Balanced Performance", "Detects 528 ARGs with 80.37% precision and 85.79% specificity.")
    add_bullet(tf_r, "SVM Precision Deceptive", "95.36% precision is achieved by under-predicting (99.23% spec), missing 764 ARGs (Recall: 15.86%).")
    add_bullet(tf_r, "Linear Model Collapses", "LogReg recall drops to 5.95%, proving resistance cannot be linearly separated.")

    add_notes(s13, "Random Forest is the clear best overall model with ROC-AUC 0.8191. SVM misses 764 ARGs, and LogReg collapses.")

    # =========================================================================
    # SLIDE 14: Visual Diagnostics - ROC, PR & Confusion Heatmaps
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Visual Diagnostics: ROC, Precision-Recall & Confusion Heatmaps", "Visual Verification")

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

    c_diag, tf_d = create_card(s14, right_x, Inches(3.6), right_w, Inches(3.4))
    add_card_header(tf_d, "Diagnostic Interpretation")
    add_bullet(tf_d, "ROC Dominance", "Random Forest (AUC = 0.819) consistently outperforms SVM (0.787) and LogReg (0.630).")
    add_bullet(tf_d, "PR Robustness", "Random Forest achieves 0.816 Average Precision against the 0.50 random prevalence baseline.")
    add_bullet(tf_d, "Confusion Trade-offs", "RF correctly classifies 528 TP / 779 TN, whereas SVM and LogReg suffer severe false negative blocks.")

    add_notes(s14, "ROC and PR curves show Random Forest's consistent ranking dominance across all operating thresholds.")

    # =========================================================================
    # SLIDE 15: Interpretable Feature Analysis & Biochemical Drivers
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "Interpretable Feature Analysis & Biochemical Drivers", "Feature Interpretability")

    fig_fi_p = FIGURES_DIR / "feature_importance_top25.png"
    if fig_fi_p.exists():
        s15.shapes.add_picture(str(fig_fi_p), Inches(0.8), Inches(1.25), Inches(5.8), Inches(5.75))

    c_bio, tf_bio = create_card(s15, Inches(6.9), Inches(1.25), Inches(5.633), Inches(5.75))
    add_card_header(tf_bio, "Biochemical Drivers of Predictions")
    add_bullet(tf_bio, "Hydropathy & Efflux", "GRAVY and Aliphatic Index in top 10 due to hydrophobic transmembrane helices in active efflux pumps.")
    add_bullet(tf_bio, "Loop Flexibility", "Proline (PP, PA) and glycine (GG) dipeptides confer loop mobility in catalytic clefts of beta-lactamases.")
    add_bullet(tf_bio, "Electrostatic Guidance", "pI and net charge guide polar antibiotic substrates into active sites.")
    add_bullet(tf_bio, "Non-Linear Synergy", "Decision trees successfully model combinations of hydrophobic anchors and polar catalytic residues.")

    add_notes(s15, "GRAVY hydropathy and proline/glycine dipeptides dominate feature importance, reflecting efflux helices and active-site loops.")

    # =========================================================================
    # SLIDE 16: Biological Error Analysis & Length Bias Audit
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    add_header(s16, "Biological Error Analysis & Sequence Length Bias Audit", "Error Investigation")

    c_fp, tf_fp = create_card(s16, Inches(0.8), Inches(1.25), Inches(5.72), Inches(2.9), bg_color=CARD_BG_ALT)
    add_card_header(tf_fp, "1. False Positives (129 Cases)")
    add_bullet(tf_fp, "ABC Transporters (~65% FPs)", "PstB and SapF share identical transmembrane topologies and Walker A/B motifs with efflux pumps.")
    add_bullet(tf_fp, "Housekeeping Transferases", "Bacterial acetyltransferases share active-site geometries with aminoglycoside modifying enzymes.")

    c_fn, tf_fn = create_card(s16, Inches(6.8), Inches(1.25), Inches(5.733), Inches(2.9), bg_color=CARD_BG_ALT)
    add_card_header(tf_fn, "2. False Negatives (380 Cases)")
    add_bullet(tf_fn, "Atypical Mechanisms", "cfr rRNA methyltransferases and tet(M) ribosomal protection act on ribosomes, lacking training representation.")
    add_bullet(tf_fn, "Metabolic Bypass", "dfrA trimethoprim bypass enzymes resemble standard housekeeping enzymes.")

    fig_len_p = FIGURES_DIR / "sequence_length_distribution_predictions.png"
    if fig_len_p.exists():
        s16.shapes.add_picture(str(fig_len_p), Inches(0.8), Inches(4.3), Inches(5.72), Inches(2.7))

    c_len, tf_len = create_card(s16, Inches(6.8), Inches(4.3), Inches(5.733), Inches(2.7))
    add_card_header(tf_len, "3. Sequence Length Audit")
    add_bullet(tf_len, "Median Across Outcomes", "TP: 290 aa; TN: 358 aa; FP: 280 aa; FN: 290 aa.")
    add_bullet(tf_len, "No Length Shortcut", "Substantial distributional overlap confirms the model did not learn sequence length as a shortcut.")

    add_notes(s16, "False positives are driven by physiological ABC transporters. Length distributions overlap, disproving length shortcuts.")

    # =========================================================================
    # SLIDE 17: Live Dashboard, Governance & Roadmap
    # =========================================================================
    s17 = prs.slides.add_slide(blank_layout)
    add_header(s17, "Live Interactive Dashboard, Clinical Governance & Future Horizons", "Deployment & Roadmap")

    c_dash, tf_d = create_card(s17, Inches(0.8), card_top, col_w3, card_h)
    add_card_header(tf_d, "1. Live Research App")
    add_bullet(tf_d, "Streamlit Dashboard", "Full 6-page interactive web application (app.py) for real-time model exploration.")
    add_bullet(tf_d, "FASTA Inference", "Accepts raw FASTA sequences, computes 428 features, and outputs probabilities with radar charts.")
    add_bullet(tf_d, "Built-In Documentation", "Embeds all 24 viva defense Q&As and full QC logs directly in the UI.")

    c_gov, tf_g = create_card(s17, Inches(0.8 + col_w3 + gap3), card_top, col_w3, card_h, bg_color=BG_RED_TINT, border_color=BORDER_RED)
    add_card_header(tf_g, "2. Clinical Governance", color=TEXT_RED)
    add_bullet(tf_g, "Computational Hypotheses", "Predictions are in silico screening hypotheses, not diagnostic confirmations.", color=TEXT_RED)
    add_bullet(tf_g, "Phenotypic AST Mandatory", "Clinical resistance requires phenotypic MIC confirmation (CLSI/EUCAST).", color=TEXT_RED)
    add_bullet(tf_g, "Presence vs Expression", "Predicting an ARG does not prove active in vivo transcription or expression.", color=TEXT_RED)

    c_fut, tf_f = create_card(s17, Inches(0.8 + 2 * (col_w3 + gap3)), card_top, col_w3, card_h, bg_color=BG_GREEN_TINT, border_color=BORDER_GREEN)
    add_card_header(tf_f, "3. Future Roadmap", color=TEXT_GREEN)
    add_bullet(tf_f, "Protein Language Models", "Leverage 1280-dim ESM-2 transformer embeddings to capture 3D fold contacts.", color=TEXT_GREEN)
    add_bullet(tf_f, "Multi-Class Architecture", "Decompose into Enzymatic vs Efflux vs Target Protection vs Bypass.", color=TEXT_GREEN)
    add_bullet(tf_f, "Hybrid Pipeline", "Combine fast BLAST (>80% identity) with ML models for remote homologs (<30%).", color=TEXT_GREEN)

    add_notes(s17, "Our 6-page Streamlit dashboard enables live testing. In silico predictions prioritize candidates for laboratory AST.")

    prs.save(str(pptx_path))
    print(f"Clean presentation created with {len(prs.slides)} slides at: {pptx_path}")
    return pptx_path


if __name__ == "__main__":
    out_path = build_clean_presentation()
    sys.exit(0)
