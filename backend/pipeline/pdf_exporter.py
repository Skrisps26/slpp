"""
PDF exporter for GCIS SOAP notes.
Uses ReportLab to generate professional PDFs with verification summary.
"""
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import io


class PDFExporter:
    """Generates PDF SOAP notes with verification annotations."""

    def export(self, soap, verification, patient_info: dict, transcript: str) -> bytes:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=LETTER,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=72,
        )
        styles = getSampleStyleSheet()

        story = []

        # ── Header ──
        story.append(Paragraph(
            f"GCIS SOAP Note — {patient_info.get('patient_name', 'Unknown Patient')}",
            styles["Title"]))
        story.append(Paragraph(
            f"Patient ID: {patient_info.get('patient_id', 'N/A')} | "
            f"Age: {patient_info.get('patient_age', 'N/A')}",
            styles["Normal"]))
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        story.append(Spacer(1, 8))

        # ── Verification Summary Block ──
        fa_pct = round(verification.faithfulness_score * 100)
        total = len(verification.sentence_results)
        entailed = sum(1 for s in verification.sentence_results if s.label == "ENTAILED")
        corrected = len(verification.hallucinated_sentences)

        fa_color = colors.green if fa_pct >= 85 else (
            colors.orange if fa_pct >= 70 else colors.red)

        style_heading3 = ParagraphStyle(
            name="Heading3_custom", parent=styles["Heading3"],
            textColor=colors.HexColor("#0c4a6e"), fontSize=11, spaceAfter=4)

        story.append(Paragraph("Verification Summary", style_heading3))
        story.append(Paragraph(
            f"Faithfulness Score: <b>{fa_pct}%</b> "
            f"({entailed}/{total} sentences verified against transcript). "
            f"Auto-corrected: {corrected} sentence(s). "
            f"Pipeline version: 2.0.0",
            styles["Normal"]))
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        story.append(Spacer(1, 8))

        # ── SOAP Sections ──
        for section_name in ["subjective", "objective", "assessment", "plan"]:
            story.append(Paragraph(section_name.upper(), styles["Heading2"]))
            text = getattr(soap, section_name, "")
            # Annotate unverified sentences
            for sv in verification.sentence_results:
                if sv.soap_section == section_name and sv.label == "NEUTRAL":
                    marker = f" <i>[unverified]</i>"
                    text = text.replace(sv.soap_sentence, sv.soap_sentence + marker, 1)

            story.append(Paragraph(text, styles["Normal"]))
            story.append(Spacer(1, 8))

        # ── Differentials ──
        if soap.differentials:
            story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
            story.append(Spacer(1, 8))
            story.append(Paragraph("DIFFERENTIAL DIAGNOSES", styles["Heading2"]))
            for diff in soap.differentials:
                story.append(Paragraph(
                    f"<b>{diff.diagnosis}</b> [{diff.likelihood.upper()}]: {diff.evidence}",
                    styles["Normal"]))
                if diff.kb_source:
                    story.append(Paragraph(
                        f"<i>Source: {diff.kb_source}</i>",
                        styles["Normal"]))
                story.append(Spacer(1, 4))

        doc.build(story)
        return buffer.getvalue()
