"""
PDF Builder
Generates a comprehensive, professionally formatted clinical encounter report using reportlab.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak,
)
from reportlab.platypus.flowables import Flowable

from soap_builder import SOAPNote


# ─────────────────────────── Color palette ──────────────────────────────────

NAVY      = colors.HexColor("#1A3557")
TEAL      = colors.HexColor("#1F7A8C")
LIGHT_BG  = colors.HexColor("#EEF4F8")
ACCENT    = colors.HexColor("#2E86AB")
MUTED     = colors.HexColor("#6B7C8D")
RED_FLAG  = colors.HexColor("#C0392B")
YELLOW    = colors.HexColor("#F39C12")
WHITE     = colors.white
BLACK     = colors.HexColor("#1A1A1A")
LIGHT_GRAY = colors.HexColor("#DEE6EC")
MID_GRAY  = colors.HexColor("#B0BEC5")
GREEN     = colors.HexColor("#27AE60")


# ─────────────────────────── Custom flowables ───────────────────────────────

class ColoredRect(Flowable):
    """A solid colored rectangle used as a section header background."""

    def __init__(self, width, height, color, text="", text_color=WHITE, font_size=11):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.color = color
        self.text = text
        self.text_color = text_color
        self.font_size = font_size

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        if self.text:
            self.canv.setFillColor(self.text_color)
            self.canv.setFont("Helvetica-Bold", self.font_size)
            self.canv.drawString(10, (self.height - self.font_size) / 2 + 2, self.text)


class SideBarRect(Flowable):
    """Left accent bar for subsection headers."""

    def __init__(self, width, height, bar_color, text="", font_size=10):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.bar_color = bar_color
        self.text = text
        self.font_size = font_size

    def draw(self):
        self.canv.setFillColor(self.bar_color)
        self.canv.rect(0, 0, 4, self.height, fill=1, stroke=0)
        self.canv.setFillColor(BLACK)
        self.canv.setFont("Helvetica-Bold", self.font_size)
        self.canv.drawString(12, (self.height - self.font_size) / 2 + 1, self.text)


# ─────────────────────────── Style registry ─────────────────────────────────

def build_styles() -> dict:
    base = getSampleStyleSheet()
    styles = {}

    styles["normal"] = ParagraphStyle(
        "Normal_Custom", parent=base["Normal"],
        fontName="Helvetica", fontSize=9,
        textColor=BLACK, leading=14, spaceAfter=3,
    )
    styles["body"] = ParagraphStyle(
        "Body_Custom", parent=base["Normal"],
        fontName="Helvetica", fontSize=9,
        textColor=BLACK, leading=14, spaceAfter=4,
        firstLineIndent=0, alignment=TA_JUSTIFY,
    )
    styles["label"] = ParagraphStyle(
        "Label", parent=base["Normal"],
        fontName="Helvetica-Bold", fontSize=8.5,
        textColor=MUTED, leading=12, spaceAfter=2,
    )
    styles["value"] = ParagraphStyle(
        "Value", parent=base["Normal"],
        fontName="Helvetica", fontSize=9,
        textColor=BLACK, leading=13, spaceAfter=3,
    )
    styles["flag_critical"] = ParagraphStyle(
        "FlagCritical", parent=base["Normal"],
        fontName="Helvetica-Bold", fontSize=9,
        textColor=RED_FLAG, leading=13, spaceAfter=2,
    )
    styles["flag_warning"] = ParagraphStyle(
        "FlagWarning", parent=base["Normal"],
        fontName="Helvetica-Bold", fontSize=9,
        textColor=YELLOW, leading=13, spaceAfter=2,
    )
    styles["flag_note"] = ParagraphStyle(
        "FlagNote", parent=base["Normal"],
        fontName="Helvetica", fontSize=9,
        textColor=ACCENT, leading=13, spaceAfter=2,
    )
    styles["bullet"] = ParagraphStyle(
        "Bullet_Custom", parent=base["Normal"],
        fontName="Helvetica", fontSize=9,
        textColor=BLACK, leading=14, spaceAfter=2,
        leftIndent=14, bulletIndent=4,
    )
    styles["icd_code"] = ParagraphStyle(
        "ICD_Code", parent=base["Normal"],
        fontName="Courier-Bold", fontSize=8,
        textColor=TEAL, leading=12,
    )
    styles["transcript"] = ParagraphStyle(
        "Transcript", parent=base["Normal"],
        fontName="Courier", fontSize=7.5,
        textColor=colors.HexColor("#444444"),
        leading=12, spaceAfter=2, leftIndent=8,
    )
    styles["footer"] = ParagraphStyle(
        "Footer", parent=base["Normal"],
        fontName="Helvetica", fontSize=7,
        textColor=MUTED, alignment=TA_CENTER,
    )
    styles["meta_key"] = ParagraphStyle(
        "MetaKey", parent=base["Normal"],
        fontName="Helvetica-Bold", fontSize=8,
        textColor=WHITE, leading=11,
    )
    styles["meta_val"] = ParagraphStyle(
        "MetaVal", parent=base["Normal"],
        fontName="Helvetica", fontSize=9,
        textColor=WHITE, leading=12,
    )
    styles["section_title"] = ParagraphStyle(
        "SectionTitle", parent=base["Normal"],
        fontName="Helvetica-Bold", fontSize=11,
        textColor=WHITE, leading=14,
    )
    styles["vital_value"] = ParagraphStyle(
        "VitalValue", parent=base["Normal"],
        fontName="Helvetica-Bold", fontSize=12,
        textColor=NAVY, leading=15, alignment=TA_CENTER,
    )
    styles["vital_label"] = ParagraphStyle(
        "VitalLabel", parent=base["Normal"],
        fontName="Helvetica", fontSize=7.5,
        textColor=MUTED, leading=10, alignment=TA_CENTER,
    )
    return styles


# ─────────────────────────── Page layout ────────────────────────────────────

PAGE_W, PAGE_H = letter
MARGIN = 0.65 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


def _header_footer(canvas, doc):
    canvas.saveState()
    # Top rule
    canvas.setStrokeColor(NAVY)
    canvas.setLineWidth(2)
    canvas.line(MARGIN, PAGE_H - 0.45 * inch, PAGE_W - MARGIN, PAGE_H - 0.45 * inch)

    # Page number
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(
        PAGE_W - MARGIN,
        0.35 * inch,
        f"Page {doc.page}  |  CONFIDENTIAL – PROTECTED HEALTH INFORMATION",
    )
    # Bottom rule
    canvas.setLineWidth(0.5)
    canvas.setStrokeColor(LIGHT_GRAY)
    canvas.line(MARGIN, 0.48 * inch, PAGE_W - MARGIN, 0.48 * inch)
    canvas.restoreState()


# ─────────────────────────── Section builder helpers ────────────────────────

def _section_header(title: str, color=NAVY) -> list:
    return [
        Spacer(1, 10),
        ColoredRect(CONTENT_W, 22, color, f"  {title}", font_size=10),
        Spacer(1, 6),
    ]


def _subsection(title: str, story: list, color=TEAL):
    story.append(SideBarRect(CONTENT_W, 17, color, title, font_size=9))
    story.append(Spacer(1, 4))


def _kv_table(rows: list[tuple], col_widths=(1.5 * inch, 4.5 * inch)) -> Table:
    styles_r = getSampleStyleSheet()
    key_style = ParagraphStyle("KS", parent=styles_r["Normal"],
                                fontName="Helvetica-Bold", fontSize=8.5, textColor=MUTED)
    val_style = ParagraphStyle("VS", parent=styles_r["Normal"],
                                fontName="Helvetica", fontSize=9, textColor=BLACK)
    table_data = [
        [Paragraph(k, key_style), Paragraph(str(v), val_style)]
        for k, v in rows
    ]
    t = Table(table_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _bullet(text: str, styles: dict) -> Paragraph:
    return Paragraph(f"&#x2022; &nbsp; {text}", styles["bullet"])


def _flag_paragraph(text: str, styles: dict) -> Paragraph:
    if "CRITICAL" in text or "RED FLAG" in text:
        return Paragraph(text, styles["flag_critical"])
    if "ALERT" in text or "HIGH-ALERT" in text:
        return Paragraph(text, styles["flag_warning"])
    return Paragraph(text, styles["flag_note"])


# ─────────────────────────── Main PDF builder ───────────────────────────────

class PDFBuilder:

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = build_styles()

    def build(self, note: SOAPNote) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() else "_" for c in note.patient_name)
        filename = f"MedScribe_{safe_name}_{timestamp}.pdf"
        output_path = str(self.output_dir / filename)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            topMargin=0.55 * inch,
            bottomMargin=0.65 * inch,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            title=f"Clinical Encounter Report – {note.patient_name}",
            author=note.physician_name,
            subject="Protected Health Information",
        )

        story = []
        story += self._build_cover(note)
        story += self._build_patient_banner(note)
        story += self._build_flags(note)
        story += self._build_subjective(note)
        story += self._build_objective(note)
        story += self._build_assessment(note)
        story += self._build_plan(note)
        story += self._build_supplemental(note)
        story += self._build_transcript(note)
        story += self._build_signature(note)

        doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
        return output_path

    # ── Cover / title ──────────────────────────────────────────────────────

    def _build_cover(self, note: SOAPNote) -> list:
        story = [Spacer(1, 6)]

        # Title block
        title_data = [
            [
                Paragraph(
                    "<font color='#1A3557'><b>CLINICAL ENCOUNTER REPORT</b></font>",
                    ParagraphStyle("TT", fontName="Helvetica-Bold", fontSize=18, textColor=NAVY, leading=22),
                ),
                Paragraph(
                    f"<font color='#1F7A8C'>MedScribe</font><br/>"
                    f"<font color='#6B7C8D' size='8'>AI-Powered Clinical Documentation</font>",
                    ParagraphStyle("ST", fontName="Helvetica", fontSize=11, textColor=TEAL,
                                   alignment=TA_RIGHT, leading=16),
                ),
            ]
        ]
        t = Table(title_data, colWidths=[CONTENT_W * 0.6, CONTENT_W * 0.4])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "BOTTOM")]))
        story.append(t)
        story.append(HRFlowable(width=CONTENT_W, thickness=2, color=NAVY, spaceAfter=6))

        # Encounter meta row
        meta = [
            ("Date of Encounter", note.encounter_date),
            ("Time", note.encounter_time),
            ("Encounter Type", note.encounter_type),
            ("Facility", note.facility),
        ]
        meta_data = [[
            Paragraph(f"<b>{k}</b><br/>{v}",
                      ParagraphStyle("M", fontName="Helvetica", fontSize=8.5, textColor=MUTED, leading=13))
            for k, v in meta
        ]]
        mt = Table(meta_data, colWidths=[CONTENT_W / 4] * 4)
        mt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
            ("BOX", (0, 0), (-1, -1), 0.5, LIGHT_GRAY),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, LIGHT_GRAY),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(mt)
        story.append(Spacer(1, 8))
        return story

    # ── Patient banner ─────────────────────────────────────────────────────

    def _build_patient_banner(self, note: SOAPNote) -> list:
        age_str = f"Age: {note.patient_age} yr" if note.patient_age else ""
        dob_str = f"DOB: {note.patient_dob}" if note.patient_dob != "Unknown" else ""

        left = [
            Paragraph(
                f"<b><font color='white' size='15'>{note.patient_name}</font></b>",
                ParagraphStyle("PN", fontName="Helvetica-Bold", fontSize=15, textColor=WHITE),
            ),
            Paragraph(
                f"<font color='#AAD0E0' size='8'>{age_str}   {dob_str}   MRN: {note.patient_id}</font>",
                ParagraphStyle("PD", fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#AAD0E0")),
            ),
        ]
        right = [
            Paragraph(
                f"<b><font color='white'>{note.physician_name}</font></b>",
                ParagraphStyle("DR", fontName="Helvetica-Bold", fontSize=10, textColor=WHITE, alignment=TA_RIGHT),
            ),
            Paragraph(
                f"<font color='#AAD0E0' size='8'>{note.physician_specialty}</font>",
                ParagraphStyle("DS", fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#AAD0E0"), alignment=TA_RIGHT),
            ),
        ]

        banner_data = [[left, right]]
        bt = Table(banner_data, colWidths=[CONTENT_W * 0.65, CONTENT_W * 0.35])
        bt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), NAVY),
            ("PADDING", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return [bt, Spacer(1, 8)]

    # ── Clinical flags ─────────────────────────────────────────────────────

    def _build_flags(self, note: SOAPNote) -> list:
        if not note.clinical_flags:
            return []

        story = _section_header("⚠  CLINICAL ALERTS & FLAGS", RED_FLAG)
        flag_data = [[_flag_paragraph(f, self.styles)] for f in note.clinical_flags]
        ft = Table(flag_data, colWidths=[CONTENT_W])
        ft.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF3F2")),
            ("BOX", (0, 0), (-1, -1), 1, RED_FLAG),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#FDDEDE")),
            ("PADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(KeepTogether(ft))
        story.append(Spacer(1, 6))
        return story

    # ── S – Subjective ─────────────────────────────────────────────────────

    def _build_subjective(self, note: SOAPNote) -> list:
        S = self.styles
        story = _section_header("S  –  SUBJECTIVE", NAVY)

        # Chief complaint
        _subsection("Chief Complaint", story)
        story.append(Paragraph(note.chief_complaint, S["body"]))
        story.append(Spacer(1, 6))

        # HPI
        _subsection("History of Present Illness (HPI)", story)
        story.append(Paragraph(note.hpi, S["body"]))
        story.append(Spacer(1, 6))

        # Symptoms table
        active = [s for s in (note.entities.symptoms if note.entities else []) if not s.negated]
        negated = [s for s in (note.entities.symptoms if note.entities else []) if s.negated]

        if active:
            _subsection("Reported Symptoms", story)
            header = [
                Paragraph("<b>Symptom</b>", S["label"]),
                Paragraph("<b>Severity</b>", S["label"]),
                Paragraph("<b>Duration</b>", S["label"]),
                Paragraph("<b>Character</b>", S["label"]),
                Paragraph("<b>Location</b>", S["label"]),
            ]
            rows = [header]
            for sym in active:
                rows.append([
                    Paragraph(sym.name, S["normal"]),
                    Paragraph(sym.severity or "–", S["normal"]),
                    Paragraph(sym.duration or "–", S["normal"]),
                    Paragraph(sym.character or "–", S["normal"]),
                    Paragraph(sym.location or "–", S["normal"]),
                ])
            col_w = [CONTENT_W * x for x in (0.28, 0.14, 0.18, 0.22, 0.18)]
            st = Table(rows, colWidths=col_w)
            st.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
                ("BOX", (0, 0), (-1, -1), 0.5, MID_GRAY),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, LIGHT_GRAY),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(st)
            story.append(Spacer(1, 4))

        if negated:
            neg_str = "; ".join(s.name for s in negated)
            story.append(Paragraph(f"<b>Denied:</b> {neg_str}", S["normal"]))
            story.append(Spacer(1, 4))

        # ROS
        ros = note.review_of_systems
        if ros:
            _subsection("Review of Systems", story)
            ros_rows = []
            for system, data in ros.items():
                pos = ", ".join(data.get("positive", [])) or "–"
                neg = ", ".join(data.get("negative", [])) or "–"
                ros_rows.append([
                    Paragraph(f"<b>{system}</b>", S["label"]),
                    Paragraph(f"(+) {pos}", S["normal"]),
                    Paragraph(f"(–) {neg}", S["normal"]),
                ])
            rt = Table(ros_rows, colWidths=[CONTENT_W * 0.22, CONTENT_W * 0.42, CONTENT_W * 0.36])
            rt.setStyle(TableStyle([
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LIGHT_BG]),
                ("BOX", (0, 0), (-1, -1), 0.5, MID_GRAY),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, LIGHT_GRAY),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(rt)
            story.append(Spacer(1, 4))

        # Allergies
        _subsection("Allergies", story)
        allergy_str = "; ".join(note.allergies) if note.allergies else "None reported / NKDA"
        story.append(Paragraph(allergy_str, S["body"]))
        story.append(Spacer(1, 4))

        # Current medications
        all_meds = note.medications_current
        if all_meds:
            _subsection("Current Medications", story)
            med_rows = [[
                Paragraph("<b>Medication</b>", S["label"]),
                Paragraph("<b>Dose</b>", S["label"]),
                Paragraph("<b>Frequency</b>", S["label"]),
                Paragraph("<b>Route</b>", S["label"]),
            ]]
            for m in all_meds:
                med_rows.append([
                    Paragraph(m.name, S["normal"]),
                    Paragraph(m.dose or "–", S["normal"]),
                    Paragraph(m.frequency or "–", S["normal"]),
                    Paragraph(m.route or "oral", S["normal"]),
                ])
            mt = Table(med_rows, colWidths=[CONTENT_W * 0.40, CONTENT_W * 0.18, CONTENT_W * 0.22, CONTENT_W * 0.20])
            mt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
                ("BOX", (0, 0), (-1, -1), 0.5, MID_GRAY),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, LIGHT_GRAY),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(mt)
            story.append(Spacer(1, 4))

        # Social & family history
        if note.social_history:
            _subsection("Social History", story)
            for cat, detail in note.so
