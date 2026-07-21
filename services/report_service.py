import os
import uuid
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from config import TEMP_FOLDER


def generate_report(user_name, file_name, prediction, confidence, gradcam_path, file_type, rgb_path=None):
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    report_filename = f"report_{uuid.uuid4().hex}.pdf"
    report_path = os.path.join(TEMP_FOLDER, report_filename)

    doc = SimpleDocTemplate(report_path, pagesize=A4,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=18,
                                 textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], fontSize=13,
                                   textColor=colors.HexColor("#16213e"), spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=11,
                                textColor=colors.HexColor("#333333"), spaceAfter=4)

    story = []

    story.append(Paragraph("Deepfake Detection Report", title_style))
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("Report Summary", heading_style))

    prediction_color = "#e74c3c" if prediction == "Fake" else "#27ae60"
    table_data = [
        ["Field", "Details"],
        ["User", user_name],
        ["File Name", file_name],
        ["File Type", file_type.capitalize()],
        ["Prediction", prediction],
        ["Confidence", f"{confidence:.2f}%"],
        ["Generated At", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]

    table = Table(table_data, colWidths=[2.5*inch, 4*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3*inch))

    if gradcam_path:
        full_gradcam = os.path.join(os.path.dirname(TEMP_FOLDER), gradcam_path)
        if os.path.exists(full_gradcam):
            story.append(Paragraph("GradCAM Visualization", heading_style))
            story.append(Spacer(1, 0.1*inch))
            img = RLImage(full_gradcam, width=5*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))

    if rgb_path:
        full_rgb = os.path.join(os.path.dirname(TEMP_FOLDER), rgb_path)
        if os.path.exists(full_rgb):
            story.append(Paragraph("RGB Channel Frequency Distribution", heading_style))
            rgb_note_style = ParagraphStyle("RGBNote", parent=styles["Normal"], fontSize=10,
                                            textColor=colors.HexColor("#555555"), spaceAfter=6)
            story.append(Paragraph(
                "The graph below shows the pixel intensity distribution across the Red, Green, and Blue channels. "
                "Unusual peaks, channel imbalance, or unnatural uniformity can indicate AI-generated or manipulated content.",
                rgb_note_style
            ))
            story.append(Spacer(1, 0.1*inch))
            rgb_img = RLImage(full_rgb, width=6*inch, height=2.5*inch)
            story.append(rgb_img)
            story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Interpretation", heading_style))
    if prediction == "Fake":
        interpretation = (
            f"The model has classified this {file_type} as FAKE with a confidence of {confidence:.2f}%. "
            "The GradCAM visualization highlights the regions that contributed most to this decision. "
            "These regions often correspond to facial manipulation artifacts, unnatural textures, or "
            "inconsistencies introduced during AI generation. "
            "Examine the RGB frequency graph for unusual channel distributions that may further confirm manipulation."
        )
    else:
        interpretation = (
            f"The model has classified this {file_type} as REAL with a confidence of {confidence:.2f}%. "
            "The GradCAM visualization shows the regions the model focused on to verify authenticity. "
            "No significant manipulation artifacts were detected in the highlighted regions. "
            "The RGB frequency distribution appears consistent with natural, unprocessed media."
        )

    story.append(Paragraph(interpretation, body_style))
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph("Disclaimer", heading_style))
    disclaimer = (
        "This report is generated by an automated deepfake detection system. "
        "While the model achieves high accuracy, results should be interpreted with caution. "
        "This tool is intended for research and educational purposes only."
    )
    story.append(Paragraph(disclaimer, body_style))

    doc.build(story)
    return f"temp/{report_filename}"