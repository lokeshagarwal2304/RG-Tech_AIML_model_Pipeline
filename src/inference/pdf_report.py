"""
GrainSight AI: Official PDF Inspection Report Generator
Project: Multi-Food-Grain Quality Analysis Using Diffusion Models and Deep Learning
Author: Senior ML Research Engineer

Generates professional, printable, publication-grade PDF inspection certificates
containing embedded optical crops, Grad-CAM XAI heatmaps, AGMARK/USDA grade badges,
and probabilistic defect breakdowns.
"""

import io
import os
import datetime
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
)


def generate_inspection_pdf(original_pil, gradcam_pil, prediction_result, grain_species="Rice (Oryza sativa)"):
    """Generates a professional single-page PDF inspection report in memory.
    
    Args:
        original_pil (PIL.Image.Image): Original cropped grain image.
        gradcam_pil (PIL.Image.Image): Grad-CAM attention heatmap image.
        prediction_result (dict): Output from RiceGrainPredictor.predict().
        grain_species (str): Name of grain species.
        
    Returns:
        bytes: Raw PDF file bytes ready for download or saving.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Clean Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=12
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )
    bold_label = ParagraphStyle(
        "BoldLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # 1. Header & Branding Banner
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cert_id = f"GS-{datetime.datetime.now().strftime('%Y%m%d')}-{prediction_result['predicted_code'][:2]}"

    header_data = [
        [
            Paragraph("🌾 <b>GrainSight AI</b> &bull; Quality Inspection Certificate", title_style),
            Paragraph(f"<font color='#64748b'><b>Report ID:</b> {cert_id}<br/><b>Date:</b> {now_str}</font>", ParagraphStyle("RightH", parent=body_style, alignment=2))
        ]
    ]
    header_table = Table(header_data, colWidths=[360, 180])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)

    story.append(Paragraph("AI-Powered Optical Defect Categorization &amp; Official Commercial Grading Report", subtitle_style))
    story.append(Spacer(1, 4))

    # 2. Executive Inspection Summary Table
    grading = prediction_result.get("grading", {})
    grade_badge = grading.get("grade_badge", "UNSPECIFIED")
    commercial_grade = grading.get("commercial_grade", "Standard Inspection")
    grade_color_hex = grading.get("grade_color", "#3b82f6")
    action_verdict = grading.get("actionable_verdict", "N/A")

    summary_rows = [
        [
            Paragraph("<b>Target Grain Species</b>", bold_label),
            Paragraph(grain_species, body_style),
            Paragraph("<b>Commercial Grade</b>", bold_label),
            Paragraph(f"<font color='{grade_color_hex}'><b>{grade_badge}</b></font> ({commercial_grade})", body_style)
        ],
        [
            Paragraph("<b>Predicted Defect Category</b>", bold_label),
            Paragraph(f"<b>{prediction_result['human_label']}</b>", body_style),
            Paragraph("<b>Model Certainty</b>", bold_label),
            Paragraph(f"<b>{prediction_result['confidence_percentage']:.1f}%</b> (Softmax Peak)", body_style)
        ],
        [
            Paragraph("<b>Dataset Class Code</b>", bold_label),
            Paragraph(f"<code>{prediction_result['predicted_code']}</code>", body_style),
            Paragraph("<b>Compliance Standards</b>", bold_label),
            Paragraph(grading.get("standards_body", "AGMARK &amp; USDA Commercial Grain Standards"), body_style)
        ]
    ]

    summary_table = Table(summary_rows, colWidths=[130, 140, 130, 140])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)

    story.append(Spacer(1, 8))

    # 3. Actionable Recommendation Callout Box
    rec_box_data = [[
        Paragraph(f"<b>Commercial Action &amp; Verdict:</b><br/>{action_verdict}", body_style)
    ]]
    rec_box = Table(rec_box_data, colWidths=[540])
    rec_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(rec_box)

    story.append(Spacer(1, 10))

    # 4. Visual Evidence Section (Side-by-Side: Original vs Grad-CAM)
    story.append(Paragraph("Visual Evidence &amp; Explainable AI (Grad-CAM Defect Attention Map)", section_style))

    # Convert PIL images to ReportLab Flowables via BytesIO
    orig_buf = io.BytesIO()
    original_pil.resize((200, 200)).save(orig_buf, format="PNG")
    orig_buf.seek(0)
    rl_orig = RLImage(orig_buf, width=170, height=170)

    if gradcam_pil is not None:
        cam_buf = io.BytesIO()
        gradcam_pil.resize((200, 200)).save(cam_buf, format="PNG")
        cam_buf.seek(0)
        rl_cam = RLImage(cam_buf, width=170, height=170)
    else:
        rl_cam = Paragraph("Grad-CAM Disabled", body_style)

    img_table_data = [
        [rl_orig, rl_cam],
        [
            Paragraph("<font size='8' color='#64748b'><b>Original Optical Crop</b> (SquarePad 224x224)</font>", body_style),
            Paragraph("<font size='8' color='#64748b'><b>Grad-CAM Hotspot Map</b> (Identified Defect Region)</font>", body_style)
        ]
    ]
    img_table = Table(img_table_data, colWidths=[270, 270])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(img_table)

    story.append(Spacer(1, 10))

    # 5. Top-3 Probabilistic Distribution Table
    story.append(Paragraph("Probabilistic Defect Spectrum (Top Predictions)", section_style))

    top3 = prediction_result.get("top3_predictions", [])
    prob_table_data = [
        [
            Paragraph("<b>Rank</b>", bold_label),
            Paragraph("<b>Class Code</b>", bold_label),
            Paragraph("<b>Defect Description</b>", bold_label),
            Paragraph("<b>Probability (%)</b>", bold_label)
        ]
    ]

    for rank, item in enumerate(top3, start=1):
        prob_table_data.append([
            Paragraph(f"#{rank}", body_style),
            Paragraph(f"<code>{item['class_code']}</code>", body_style),
            Paragraph(item["label"], body_style),
            Paragraph(f"<b>{item['percentage']:.1f}%</b>", body_style)
        ])

    prob_table = Table(prob_table_data, colWidths=[50, 90, 280, 120])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(prob_table)

    story.append(Spacer(1, 14))

    # 6. Quality Certification Footer
    footer_text = Paragraph(
        "<font size='7' color='#94a3b8'>"
        "GrainSight AI Optical Inspection System &bull; Evaluated using MobileNetV3-Small &amp; Grad-CAM XAI &bull; "
        "Confidence scores serve as automated quality screening evidence in accordance with commercial grain lot standards."
        "</font>",
        body_style
    )
    story.append(footer_text)

    # Build PDF document
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
