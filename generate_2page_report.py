import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.pdfgen import canvas
import pymupdf

BASE_DIR = Path(__file__).resolve().parent

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw page numbers and running headers/footers."""
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
        self.setFont("Helvetica-Bold", 7.2)
        self.setFillColor(colors.HexColor("#4A5568"))

        # Running Header (page 2)
        if self._pageNumber > 1:
            self.drawString(32, 814, "DEEPFAKE ML HACKATHON - IIIT BANGALORE  |  TECHNICAL REPORT - TEAM SMPS")
            self.drawRightString(563, 814, "NOVEMBER 2026")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(32, 808, 563, 808)

        # Running Footer (all pages)
        self.setFont("Helvetica", 7.2)
        self.setFillColor(colors.HexColor("#718096"))
        self.drawString(32, 20, "CONFIDENTIAL  |  SUBMITTED FOR OFFICIAL EVALUATION  |  TEAM SMPS")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(563, 20, page_str)
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(32, 28, 563, 28)
        self.restoreState()


def build_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=32,
        rightMargin=32,
        topMargin=28,
        bottomMargin=28
    )

    styles = getSampleStyleSheet()

    # Color Palette
    C_PRIMARY = colors.HexColor("#1A365D")   # Deep Navy
    C_SECONDARY = colors.HexColor("#2B6CB0") # Slate Blue
    C_TEXT = colors.HexColor("#2D3748")      # Dark Charcoal
    C_BG_LIGHT = colors.HexColor("#F7FAFC")  # Off-white / light slate
    C_ALT_ROW = colors.HexColor("#EDF2F7")   # Table alternate
    C_SUCCESS = colors.HexColor("#22543D")   # Forest Green

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.white,
        alignment=1
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.2,
        leading=10.5,
        textColor=colors.HexColor("#E2E8F0"),
        alignment=1
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12.0,
        textColor=C_PRIMARY,
        spaceBefore=3.0,
        spaceAfter=1.5,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.0,
        leading=10.2,
        textColor=C_TEXT,
        alignment=4,  # Justified
        spaceAfter=2.0
    )

    fig_caption_style = ParagraphStyle(
        'FigCaption',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.8,
        leading=8.6,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
        spaceBefore=1.5,
        spaceAfter=2.0
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.0,
        leading=8.4,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.7,
        leading=8.1,
        textColor=C_TEXT,
        alignment=0
    )

    table_cell_center = ParagraphStyle(
        'TableCellCenter',
        parent=table_cell_style,
        alignment=1
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell_style,
        fontName='Helvetica-Bold'
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, OBJECTIVE & VALIDATION, 5-MODEL ARCHITECTURE, LEARNING CURVES
    # =========================================================================
    
    # Title Banner Block
    banner_data = [
        [
            Paragraph("ROBUST MULTI-PARADIGM ENSEMBLE FOR DEEPFAKE FACE DETECTION", title_style),
        ],
        [
            Paragraph("Technical Report - Deepfake ML Hackathon (IIIT Bangalore)  |  Team SMPS", subtitle_style)
        ]
    ]
    banner_table = Table(banner_data, colWidths=[531])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 2.0))

    # Meta Overview Bar
    meta_data = [
        [
            Paragraph("<b>Team Name:</b> SMPS", table_cell_style),
            Paragraph("<b>Track:</b> Binary Face Classification (224x224)", table_cell_style),
            Paragraph("<b>Architecture:</b> 5-Model Super-Ensemble", table_cell_style),
            Paragraph("<b>Evaluation Metric:</b> AUC-ROC", table_cell_style),
        ],
        [
            Paragraph("<b>Anchor Val AUC:</b> 0.9025 (EffNetV2-S)", table_cell_style),
            Paragraph("<b>Ensemble Val AUC:</b> 0.9312 (Combined)", table_cell_style),
            Paragraph("<b>T4 GPU Speed:</b> 60.36 s / 3k imgs (FP16)", table_cell_style),
            Paragraph("<b>Verification:</b> Passed (FP32 Replay Verified)", table_cell_style),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[130, 145, 136, 120])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('INNERGRID', (0,0), (-1,-1), 0.3, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 1.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 2.0))

    # Section 1: Objective & Validation Strategy
    story.append(Paragraph("1. Objective, Problem Formulation & Validation Strategy", h1_style))
    story.append(Paragraph(
        "<b>1.1 Task & Dataset Formulation:</b> The competition objective is to predict whether an aligned face-cropped image (224x224 pixels) "
        "is an authentic photograph (label=0) or a manipulated deepfake (label=1). The provided dataset contains 7,000 labelled training images "
        "(3,500 real, 3,500 fake) and 3,000 unlabelled test images with random UUID filenames. Under competition rules, test images contain no ground-truth labels "
        "and were never accessed during model training. Official test performance is sealed and evaluated independently on the hidden ground truth. "
        "To establish rigorous local benchmarking without data leakage, we partitioned the labelled data into a <b>stratified 80/20 train/validation split</b> "
        "(5,600 training, 1,400 validation; seed=42) preserving an exact 50.0% class balance across both subsets.<br/>"
        "<b>1.2 Metric & Asymmetric Risk Justification:</b> In production identity authentication and digital forensics, false positives (disenfranchising authentic citizens) "
        "and false negatives (permitting biometric spoofing) carry severe asymmetric risks. Rather than operating at an arbitrary single threshold, models are evaluated "
        "and ranked using the <b>Area Under the Receiver Operating Characteristic Curve (AUC-ROC)</b>. AUC-ROC evaluates probabilistic calibration across all operating points:",
        body_style
    ))
    
    # Formatted Equation 1 (AUC-ROC)
    eq1_img_path = BASE_DIR / "outputs" / "formula_auc.png"
    if eq1_img_path.exists():
        eq1_img = Image(str(eq1_img_path), width=330, height=52)
        eq1_table = Table([[eq1_img]], colWidths=[531])
        eq1_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EBF8FF")),
            ('BOX', (0,0), (-1,-1), 0.5, C_SECONDARY),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 1.0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1.0),
        ]))
        story.append(eq1_table)
    story.append(Spacer(1, 1.5))

    # Section 2: Architecture Framework
    story.append(Paragraph("2. Architectural Framework & The 5-Model Super-Ensemble", h1_style))
    story.append(Paragraph(
        "Deepfake artifacts manifest across complementary spatial frequencies: boundary blending leaves sharp local gradient discontinuities, GAN upsampling produces "
        "periodic grid artifacts, and diffusion models introduce subtle anatomical asymmetries. Relying on a single architecture creates inductive bias blind spots. "
        "We engineered a <b>5-Model Super-Ensemble</b> spanning 5 fundamental computer vision paradigms:",
        body_style
    ))

    # Architectural Summary Table
    arch_table_data = [
        [
            Paragraph("Model Architecture", table_header_style),
            Paragraph("Paradigm Family", table_header_style),
            Paragraph("Params", table_header_style),
            Paragraph("Peak Val AUC", table_header_style),
            Paragraph("Val F1", table_header_style),
            Paragraph("Ensemble Blend Weight", table_header_style),
        ],
        [
            Paragraph("<b>EfficientNetV2-S</b>", table_cell_bold),
            Paragraph("Fused-MBConv Progressive CNN", table_cell_style),
            Paragraph("21.5 M", table_cell_center),
            Paragraph("<b>0.9025</b>", table_cell_center),
            Paragraph("0.8462", table_cell_center),
            Paragraph("<b>0.40</b> (40% Anchor)", table_cell_center),
        ],
        [
            Paragraph("<b>ConvNeXt-Tiny</b>", table_cell_bold),
            Paragraph("Modernized Depthwise 7x7 CNN", table_cell_style),
            Paragraph("28.6 M", table_cell_center),
            Paragraph("<b>0.8973</b>", table_cell_center),
            Paragraph("0.8549", table_cell_center),
            Paragraph("<b>0.25</b> (25%)", table_cell_center),
        ],
        [
            Paragraph("<b>Swin-Tiny</b>", table_cell_bold),
            Paragraph("Shifted-Window Hierarchical ViT", table_cell_style),
            Paragraph("28.3 M", table_cell_center),
            Paragraph("<b>0.8953</b>", table_cell_center),
            Paragraph("0.8513", table_cell_center),
            Paragraph("<b>0.15</b> (15%)", table_cell_center),
        ],
        [
            Paragraph("<b>MaxViT-Tiny</b>", table_cell_bold),
            Paragraph("Multi-Axis Hybrid Block+Grid ViT", table_cell_style),
            Paragraph("31.2 M", table_cell_center),
            Paragraph("<b>0.8902</b>", table_cell_center),
            Paragraph("0.8451", table_cell_center),
            Paragraph("<b>0.12</b> (12%)", table_cell_center),
        ],
        [
            Paragraph("<b>DenseNet-121</b>", table_cell_bold),
            Paragraph("Dense Feature Reuse Network", table_cell_style),
            Paragraph("8.0 M", table_cell_center),
            Paragraph("<b>0.8905</b>", table_cell_center),
            Paragraph("0.8401", table_cell_center),
            Paragraph("<b>0.08</b> (8%)", table_cell_center),
        ],
    ]
    t_arch = Table(arch_table_data, colWidths=[100, 140, 50, 75, 75, 91])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.35, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, C_ALT_ROW]),
        ('TOPPADDING', (0,0), (-1,-1), 1.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.2),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 1.5))

    # Formatted Equation 2 (Ensemble Blending)
    eq2_img_path = BASE_DIR / "outputs" / "formula_ensemble.png"
    if eq2_img_path.exists():
        eq2_img = Image(str(eq2_img_path), width=350, height=38)
        eq2_table = Table([[eq2_img]], colWidths=[531])
        eq2_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F0FFF4")),
            ('BOX', (0,0), (-1,-1), 0.5, C_SUCCESS),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 1.0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1.0),
        ]))
        story.append(eq2_table)
    story.append(Spacer(1, 1.5))

    # Section 3: Training Pipeline & Learning Curves Graph
    story.append(Paragraph("3. Training Pipeline, Augmentations & Convergence Analysis", h1_style))
    story.append(Paragraph(
        "Models were trained using an on-the-fly <b>Albumentations</b> pipeline: RandomHorizontalFlip (p=0.5), ShiftScaleRotate (shift &plusmn;6%, scale &plusmn;10%, "
        "rotate &plusmn;10 deg; p=0.5), ColorJitter (brightness &plusmn;10%, contrast &plusmn;10%; p=0.5), ImageCompression (JPEG Q50-95; p=0.4), "
        "and GaussianBlur (kernel 3x3 to 5x5; p=0.3). Objective optimization utilized <b>Binary Focal Loss</b> (&gamma; = 1.5, &alpha; = 0.25) to penalize easy samples "
        "and force attention onto subtle facial synthesis boundaries, paired with <b>AdamW</b> (lr = 3x10<sup>-4</sup>, weight decay = 10<sup>-4</sup>) under Cosine Annealing decay:",
        body_style
    ))

    # GRAPH 1: Training & Validation Curves (outputs/final_verification/learning.png)
    learning_img_path = BASE_DIR / "outputs" / "final_verification" / "learning.png"
    if learning_img_path.exists():
        img_learn = Image(str(learning_img_path), width=420, height=128)
        t_learn = Table([[img_learn]], colWidths=[531])
        t_learn.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(t_learn)
        story.append(Paragraph("<b>Figure 1:</b> Training & Validation Learning Dynamics Across Epochs. Validation AUC plateaus near 0.9025 while training loss steadily converges, guiding early checkpoint selection to prevent overfitting.", fig_caption_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: HARDWARE EFFICIENCY, ROBUSTNESS GRAPH, GRAD-CAM, CLI PROTOCOL
    # =========================================================================

    # Section 4: Efficiency & Robustness
    story.append(Paragraph("4. Hardware Efficiency Profiling & Robustness Stress-Testing", h1_style))
    story.append(Paragraph(
        "<b>4.1 Tesla T4 Inference Profiling:</b> Benchmarked on an NVIDIA Tesla T4 GPU across all 3,000 test images (batch_size = 64). Under batched FP16 inference, "
        "the ensemble achieved a median runtime of <b>60.36 seconds</b> (49.7 imgs/sec, 561.9 MiB peak VRAM). Under the strict verified FP32 Colab replay protocol, "
        "runtime was <b>172.34 seconds</b> with 2,796.9 MiB allocated memory - well within the 300-second (5-minute) threshold.<br/>"
        "<b>4.2 Generalization Under Synthetic Corruptions:</b> Evaluated on a fixed 200-image balanced validation subset with 2,000 stratified bootstrap resamples "
        "across real-world image corruptions. Clean baseline AUC is <b>0.8817</b> (95% CI: 0.8243 to 0.9311). Under JPEG Q50 and Q70 compression, performance remains "
        "exceptionally stable (+0.0057 and +0.0037). Under Gaussian blur (&sigma; = 1.5) and Gaussian noise (&sigma; = 15), observed AUC drops are minimal at 0.0077 and 0.0110:",
        body_style
    ))

    # GRAPH 2: Robustness Under Corruptions (outputs/final_verification/final_robustness.png)
    rob_img_path = BASE_DIR / "outputs" / "final_verification" / "final_robustness.png"
    if not rob_img_path.exists():
        rob_img_path = BASE_DIR / "outputs" / "final_verification" / "robustness.png"
    if rob_img_path.exists():
        img_rob = Image(str(rob_img_path), width=420, height=128)
        t_rob = Table([[img_rob]], colWidths=[531])
        t_rob.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(t_rob)
        story.append(Paragraph("<b>Figure 2:</b> Super-Ensemble Robustness Across Synthetic Corruptions (Clean vs. JPEG Q50, JPEG Q70, Gaussian Blur &sigma;=1.5, Noise &sigma;=15). Maximum degradation drop remains constrained to 0.0110.", fig_caption_style))
    story.append(Spacer(1, 1.5))

    # Section 5: Grad-CAM Explainability
    story.append(Paragraph("5. Visual Explainability via Grad-CAM", h1_style))
    story.append(Paragraph(
        "To inspect features driving predictions, Gradient-weighted Class Activation Mapping (Grad-CAM) was applied to the final convolutional stages of EfficientNetV2-S. "
        "For deepfakes, activations concentrate sharply along synthesis boundaries, hairline seams, and facial contours. For authentic faces, attention remains diffuse across natural geometry:",
        body_style
    ))

    # GRAPH 3: Grad-CAM Explainability Panel (report_gradcam_panel.png, un-squished 3.333:1 aspect ratio)
    gradcam_panel_path = BASE_DIR / "outputs" / "report_gradcam_panel.png"
    if gradcam_panel_path.exists():
        gradcam_img = Image(str(gradcam_panel_path), width=420, height=126)
        gradcam_table = Table([[gradcam_img]], colWidths=[531])
        gradcam_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(gradcam_table)
        story.append(Paragraph("<b>Figure 3:</b> Grad-CAM Visual Explainability (Preserved 1:1 Face Aspect Ratios). <i>Left:</i> Deepfake detection driven by synthesis boundary artifacts along cheek and jawline. <i>Right:</i> Genuine face verification showing natural diffuse attention.", fig_caption_style))
    story.append(Spacer(1, 1.5))

    # Section 6: Reproducibility & CLI
    story.append(Paragraph("6. Reproducibility & CLI Inference Protocol", h1_style))
    story.append(Paragraph(
        "The submission strictly adheres to the official interface with zero hardcoded paths. All 3,000 test predictions were validated via an independent full replay "
        "(0 label mismatches, 0.0 maximum difference; status: <b>PASSED</b>). Private test labels are sealed for official evaluator scoring. Evaluators can reproduce predictions using:",
        body_style
    ))

    cli_box_data = [
        [Paragraph(
            "<font name='Courier' size='6.6' color='#1A365D'>"
            "# Standard Super-Ensemble Inference:<br/>"
            "python inference_script.py --test_dir ./test/images/ --weights weights/ensemble_config.json --output predictions.csv<br/>"
            "<br/>"
            "# Single-Model Anchor Evaluation (EfficientNetV2-S):<br/>"
            "python inference_script.py --test_dir ./test/images/ --weights weights/efficientnetv2_s_best.pth --backbone tf_efficientnetv2_s --output preds_v2.csv"
            "</font>",
            table_cell_style
        )]
    ]
    t_cli = Table(cli_box_data, colWidths=[531])
    t_cli.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EDF2F7")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_cli)
    story.append(Spacer(1, 1.0))
    story.append(Paragraph(
        "<font size='6.7' color='#4A5568'><b>Submission Package Checklist:</b> predictions.csv (3,000 test images), inference_script.py, requirements.txt, "
        "report.pdf (this 2-page document), Explanatory Video.mp4, weights/ (all 5 binary checkpoints, ensemble_config.json, ONNX, and TensorFlow exports). All code reproduces predictions.csv identically.</font>",
        body_style
    ))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Report generated: {output_path}")

    # Verify Page Count
    pdf_doc = pymupdf.open(output_path)
    page_count = len(pdf_doc)
    print(f"[VERIFY] Total Pages in PDF: {page_count}")
    assert page_count == 2, f"Error: Expected exactly 2 pages, but got {page_count}!"
    print("[OK] Page count strictly matches 2-page hackathon specification!")


if __name__ == "__main__":
    out_pdf = str(BASE_DIR / "report.pdf")
    build_pdf(out_pdf)
