"""
PDF Builder
Generates a comprehensive, professionally formatted clinical encounter report using reportlab.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable

from soap_builder import SOAPNote

# ─────────────────────────── Color palette ──────────────────────────────────

NAVY = colors.HexColor("#1A3557")
TEAL = colors.HexColor("#1F7A8C")
LIGHT_BG = colors.HexColor("#EEF4F8")
ACCENT = colors.HexColor("#2E86AB")
MUTED = colors.HexColor("#6B7C8D")
RED_FLAG = colors.HexColor("#C0392B")
YELLOW = colors.HexColor("#F39C12")
WHITE = colors.white
BLACK = colors.HexColor("#1A1A1A")
LIGHT_GRAY = colors.HexColor("#DEE6EC")
MID_GRAY = colors.HexColor("#B0BEC5")
GREEN = colors.HexColor("#27AE60")


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
        "Normal_Custom",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=BLACK,
        leading=14,
        spaceAfter=3,
    )
    styles["body"] = ParagraphStyle(
        "Body_Custom",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=BLACK,
        leading=14,
        spaceAfter=4,
        firstLineIndent=0,
        alignment=TA_JUSTIFY,
    )
    styles["label"] = ParagraphStyle(
        "Label",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        textColor=MUTED,
        leading=12,
        spaceAfter=2,
    )
    styles["value"] = ParagraphStyle(
        "Value",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=BLACK,
        leading=13,
        spaceAfter=3,
    )
    styles["flag_critical"] = ParagraphStyle(
        "FlagCritical",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=RED_FLAG,
        leading=13,
        spaceAfter=2,
    )
    styles["flag_warning"] = ParagraphStyle(
        "FlagWarning",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=YELLOW,
        leading=13,
        spaceAfter=2,
    )
    styles["flag_note"] = ParagraphStyle(
        "FlagNote",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=ACCENT,
        leading=13,
        spaceAfter=2,
    )
    styles["bullet"] = ParagraphStyle(
        "Bullet_Custom",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=BLACK,
        leading=14,
        spaceAfter=2,
        leftIndent=14,
        bulletIndent=4,
    )
    styles["icd_code"] = ParagraphStyle(
        "ICD_Code",
        parent=base["Normal"],
        fontName="Courier-Bold",
        fontSize=8,
        textColor=TEAL,
        leading=12,
    )
    styles["transcript"] = ParagraphStyle(
        "Transcript",
        parent=base["Normal"],
        fontName="Courier",
        fontSize=7.5,
        textColor=colors.HexColor("#444444"),
        leading=12,
        spaceAfter=2,
        leftIndent=8,
    )
    styles["footer"] = ParagraphStyle(
        "Footer",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=7,
        textColor=MUTED,
        alignment=TA_CENTER,
    )
    styles["meta_key"] = ParagraphStyle(
        "MetaKey",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=WHITE,
        leading=11,
    )
    styles["meta_val"] = ParagraphStyle(
        "MetaVal",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=WHITE,
        leading=12,
    )
    styles["section_title"] = ParagraphStyle(
        "SectionTitle",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=WHITE,
        leading=14,
    )
    styles["vital_value"] = ParagraphStyle(
        "VitalValue",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=NAVY,
        leading=15,
        alignment=TA_CENTER,
    )
    styles["vital_label"] = ParagraphStyle(
        "VitalLabel",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        textColor=MUTED,
        leading=10,
        alignment=TA_CENTER,
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
    key_style = ParagraphStyle(
        "KS",
        parent=styles_r["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        textColor=MUTED,
    )
    val_style = ParagraphStyle(
        "VS",
        parent=styles_r["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=BLACK,
    )
    table_data = [
        [Paragraph(k, key_style), Paragraph(str(v), val_style)] for k, v in rows
    ]
    t = Table(table_data, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
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
                    ParagraphStyle(
                        "TT",
                        fontName="Helvetica-Bold",
                        fontSize=18,
                        textColor=NAVY,
                        leading=22,
                    ),
                ),
                Paragraph(
                    f"<font color='#1F7A8C'>MedScribe</font><br/>"
                    f"<font color='#6B7C8D' size='8'>AI-Powered Clinical Documentation</font>",
                    ParagraphStyle(
                        "ST",
                        fontName="Helvetica",
                        fontSize=11,
                        textColor=TEAL,
                        alignment=TA_RIGHT,
                        leading=16,
                    ),
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
        meta_data = [
            [
                Paragraph(
                    f"<b>{k}</b><br/>{v}",
                    ParagraphStyle(
                        "M",
                        fontName="Helvetica",
                        fontSize=8.5,
                        textColor=MUTED,
                        leading=13,
                    ),
                )
                for k, v in meta
            ]
        ]
        mt = Table(meta_data, colWidths=[CONTENT_W / 4] * 4)
        mt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
                    ("BOX", (0, 0), (-1, -1), 0.5, LIGHT_GRAY),
                    ("INNERGRID", (0, 0), (-1, -1), 0.3, LIGHT_GRAY),
                    ("PADDING", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
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
                ParagraphStyle(
                    "PN", fontName="Helvetica-Bold", fontSize=15, textColor=WHITE
                ),
            ),
            Paragraph(
                f"<font color='#AAD0E0' size='8'>{age_str}   {dob_str}   MRN: {note.patient_id}</font>",
                ParagraphStyle(
                    "PD",
                    fontName="Helvetica",
                    fontSize=8,
                    textColor=colors.HexColor("#AAD0E0"),
                ),
            ),
        ]
        right = [
            Paragraph(
                f"<b><font color='white'>{note.physician_name}</font></b>",
                ParagraphStyle(
                    "DR",
                    fontName="Helvetica-Bold",
                    fontSize=10,
                    textColor=WHITE,
                    alignment=TA_RIGHT,
                ),
            ),
            Paragraph(
                f"<font color='#AAD0E0' size='8'>{note.physician_specialty}</font>",
                ParagraphStyle(
                    "DS",
                    fontName="Helvetica",
                    fontSize=8,
                    textColor=colors.HexColor("#AAD0E0"),
                    alignment=TA_RIGHT,
                ),
            ),
        ]

        banner_data = [[left, right]]
        bt = Table(banner_data, colWidths=[CONTENT_W * 0.65, CONTENT_W * 0.35])
        bt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                    ("PADDING", (0, 0), (-1, -1), 10),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        return [bt, Spacer(1, 8)]

    # ── Clinical flags ─────────────────────────────────────────────────────

    def _build_flags(self, note: SOAPNote) -> list:
        if not note.clinical_flags:
            return []

        story = _section_header("⚠  CLINICAL ALERTS & FLAGS", RED_FLAG)
        flag_data = [[_flag_paragraph(f, self.styles)] for f in note.clinical_flags]
        ft = Table(flag_data, colWidths=[CONTENT_W])
        ft.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF3F2")),
                    ("BOX", (0, 0), (-1, -1), 1, RED_FLAG),
                    ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#FDDEDE")),
                    ("PADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
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
        active = [
            s
            for s in (note.entities.symptoms if note.entities else [])
            if not s.negated
        ]
        negated = [
            s for s in (note.entities.symptoms if note.entities else []) if s.negated
        ]

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
                rows.append(
                    [
                        Paragraph(sym.name, S["normal"]),
                        Paragraph(sym.severity or "–", S["normal"]),
                        Paragraph(sym.duration or "–", S["normal"]),
                        Paragraph(sym.character or "–", S["normal"]),
                        Paragraph(sym.location or "–", S["normal"]),
                    ]
                )
            col_w = [CONTENT_W * x for x in (0.28, 0.14, 0.18, 0.22, 0.18)]
            st = Table(rows, colWidths=col_w)
            st.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 8),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
                        ("BOX", (0, 0), (-1, -1), 0.5, MID_GRAY),
                        ("INNERGRID", (0, 0), (-1, -1), 0.25, LIGHT_GRAY),
                        ("PADDING", (0, 0), (-1, -1), 5),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
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
                ros_rows.append(
                    [
                        Paragraph(f"<b>{system}</b>", S["label"]),
                        Paragraph(f"(+) {pos}", S["normal"]),
                        Paragraph(f"(–) {neg}", S["normal"]),
                    ]
                )
            rt = Table(
                ros_rows,
                colWidths=[CONTENT_W * 0.22, CONTENT_W * 0.42, CONTENT_W * 0.36],
            )
            rt.setStyle(
                TableStyle(
                    [
                        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LIGHT_BG]),
                        ("BOX", (0, 0), (-1, -1), 0.5, MID_GRAY),
                        ("INNERGRID", (0, 0), (-1, -1), 0.25, LIGHT_GRAY),
                        ("PADDING", (0, 0), (-1, -1), 5),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(rt)
            story.append(Spacer(1, 4))

        # Allergies
        _subsection("Allergies", story)
        allergy_str = (
            "; ".join(note.allergies) if note.allergies else "None reported / NKDA"
        )
        story.append(Paragraph(allergy_str, S["body"]))
        story.append(Spacer(1, 4))

        # Current medications
        all_meds = note.medications_current
        if all_meds:
            _subsection("Current Medications", story)
            med_rows = [
                [
                    Paragraph("<b>Medication</b>", S["label"]),
                    Paragraph("<b>Dose</b>", S["label"]),
                    Paragraph("<b>Frequency</b>", S["label"]),
                    Paragraph("<b>Route</b>", S["label"]),
                ]
            ]
            for m in all_meds:
                med_rows.append(
                    [
                        Paragraph(m.name, S["normal"]),
                        Paragraph(m.dose or "–", S["normal"]),
                        Paragraph(m.frequency or "–", S["normal"]),
                        Paragraph(m.route or "oral", S["normal"]),
                    ]
                )
            mt = Table(
                med_rows,
                colWidths=[
                    CONTENT_W * 0.40,
                    CONTENT_W * 0.18,
                    CONTENT_W * 0.22,
                    CONTENT_W * 0.20,
                ],
            )
            mt.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 8),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
                        ("BOX", (0, 0), (-1, -1), 0.5, MID_GRAY),
                        ("INNERGRID", (0, 0), (-1, -1), 0.25, LIGHT_GRAY),
                        ("PADDING", (0, 0), (-1, -1), 5),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(mt)
            story.append(Spacer(1, 4))

        # Social & family history
        if note.social_history:
            _subsection("Social History", story)
            for cat, detail in note.social_history.items():
                story.append(
                    Paragraph(
                        f"<b>{cat.replace('_', ' ').title()}:</b> {detail}", S["normal"]
                    )
                )
            story.append(Spacer(1, 4))

        if note.family_history:
            _subsection("Family History", story)
            for item in note.family_history[:5]:
                story.append(_bullet(item, S))
            story.append(Spacer(1, 4))

        return story

    # ── O – Objective ──────────────────────────────────────────────────────

    def _build_objective(self, note: SOAPNote) -> list:
        S = self.styles
        story = _section_header("O  –  OBJECTIVE", NAVY)

        # Vitals
        _subsection("Vital Signs", story)
        vitals = note.vitals_summary
        if vitals:
            # Display vitals as cards
            vital_cells = []
            for v in vitals:
                flag_color = self._vital_color(v.name, v.value)
                cell = [
                    Paragraph(
                        f"<b>{v.value}</b>",
                        ParagraphStyle(
                            "VV",
                            fontName="Helvetica-Bold",
                            fontSize=14,
                            textColor=flag_color,
                            alignment=TA_CENTER,
                        ),
                    ),
                    Paragraph(
                        v.unit,
                        ParagraphStyle(
                            "VU",
                            fontName="Helvetica",
                            fontSize=7,
                            textColor=MUTED,
                            alignment=TA_CENTER,
                        ),
                    ),
                    HRFlowable(width="100%", thickness=0.5, color=LIGHT_GRAY),
                    Paragraph(
                        v.name,
                        ParagraphStyle(
                            "VL",
                            fontName="Helvetica-Bold",
                            fontSize=7.5,
                            textColor=NAVY,
                            alignment=TA_CENTER,
                        ),
                    ),
                ]
                vital_cells.append(cell)

            cols = min(len(vital_cells), 5)
            rows = [vital_cells[i : i + cols] for i in range(0, len(vital_cells), cols)]
            for row in rows:
                while len(row) < cols:
                    row.append([""])
                vt = Table([row], colWidths=[CONTENT_W / cols] * cols)
                vt.setStyle(
                    TableStyle(
                        [
                            ("BOX", (0, 0), (-1, -1), 0.5, LIGHT_GRAY),
                            ("INNERGRID", (0, 0), (-1, -1), 0.5, LIGHT_GRAY),
                            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
                            ("PADDING", (0, 0), (-1, -1), 8),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ]
                    )
                )
                story.append(vt)
                story.append(Spacer(1, 4))
        else:
            story.append(
                Paragraph("Vital signs not documented in transcript.", S["body"])
            )

        story.append(Spacer(1, 4))

        # Physical exam (placeholder if not extracted)
        _subsection("Physical Examination", story)
        story.append(
            Paragraph(
                "Physical examination findings as documented during the clinical encounter. "
                "Detailed examination notes should be added by the attending physician.",
                S["body"],
            )
        )
        story.append(Spacer(1, 4))

        return story

    def _vital_color(self, name: str, value: str) -> colors.Color:
        import re as _re

        try:
            if name == "Blood Pressure":
                sys_v = int(value.split("/")[0])
                if sys_v >= 180 or sys_v < 90:
                    return RED_FLAG
                if sys_v >= 140:
                    return YELLOW
                return GREEN
            if name == "O2 Saturation":
                v = float(_re.sub(r"[^\d.]", "", value))
                if v < 92:
                    return RED_FLAG
                if v < 95:
                    return YELLOW
                return GREEN
            if name == "Heart Rate":
                v = int(_re.sub(r"[^\d]", "", value)[:3])
                if v > 120 or v < 50:
                    return YELLOW
                return GREEN
            if name == "Temperature":
                v = float(_re.sub(r"[^\d.]", "", value))
                if v >= 103 or v < 96:
                    return RED_FLAG
                if v >= 100.4:
                    return YELLOW
                return GREEN
        except (ValueError, IndexError):
            pass
        return NAVY

    # ── A – Assessment ─────────────────────────────────────────────────────

    def _build_assessment(self, note: SOAPNote) -> list:
        S = self.styles
        story = _section_header("A  –  ASSESSMENT", NAVY)

        # Assessment narrative
        _subsection("Clinical Impression", story)
        story.append(Paragraph(note.assessment_narrative, S["body"]))
        story.append(Spacer(1, 6))

        # Diagnoses table
        all_dx = []
        if note.primary_diagnosis:
            all_dx.append(note.primary_diagnosis)
        all_dx.extend(note.secondary_diagnoses)

        if all_dx:
            _subsection("Diagnoses & ICD-10 Codes", story)
            dx_rows = [
                [
                    Paragraph("<b>#</b>", S["label"]),
                    Paragraph("<b>Diagnosis</b>", S["label"]),
                    Paragraph("<b>ICD-10</b>", S["label"]),
                    Paragraph("<b>Certainty</b>", S["label"]),
                    Paragraph("<b>Type</b>", S["label"]),
                ]
            ]
            _cert_hex = {
                "confirmed": "#27AE60",
                "possible": "#F39C12",
                "ruled-out": "#C0392B",
            }
            for i, dx in enumerate(all_dx, 1):
                cert_color = _cert_hex.get(dx.certainty, "#6B7C8D")
                dx_rows.append(
                    [
                        Paragraph(str(i), S["normal"]),
                        Paragraph(dx.name, S["normal"]),
                        Paragraph(dx.icd10 or "–", S["icd_code"]),
                        Paragraph(
                            f"<font color='{cert_color}'><b>{dx.certainty.upper()}</b></font>",
                            S["normal"],
                        ),
                        Paragraph(
                            "Primary" if dx.primary else "Secondary", S["normal"]
                        ),
                    ]
                )
            dt = Table(
                dx_rows,
                colWidths=[
                    CONTENT_W * 0.05,
                    CONTENT_W * 0.45,
                    CONTENT_W * 0.15,
                    CONTENT_W * 0.18,
                    CONTENT_W * 0.17,
                ],
            )
            dt.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
                        ("BOX", (0, 0), (-1, -1), 0.5, MID_GRAY),
                        ("INNERGRID", (0, 0), (-1, -1), 0.25, LIGHT_GRAY),
                        ("PADDING", (0, 0), (-1, -1), 5),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 8),
                    ]
                )
            )
            story.append(dt)
            story.append(Spacer(1, 6))

        # Differential diagnoses
        if note.differential_diagnoses:
            _subsection("Differential Diagnoses Considered", story)
            ddx_data = [
                [
                    Paragraph(f"{i}.", S["normal"]),
                    Paragraph(ddx, S["normal"]),
                ]
                for i, ddx in enumerate(note.differential_diagnoses, 1)
            ]
            ddt = Table(ddx_data, colWidths=[CONTENT_W * 0.06, CONTENT_W * 0.94])
            ddt.setStyle(
                TableStyle(
                    [
                        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LIGHT_BG]),
                        ("PADDING", (0, 0), (-1, -1), 4),
                        ("BOX", (0, 0), (-1, -1), 0.5, LIGHT_GRAY),
                    ]
                )
            )
            story.append(ddt)
            story.append(Spacer(1, 4))

        return story

    # ── P – Plan ───────────────────────────────────────────────────────────

    def _build_plan(self, note: SOAPNote) -> list:
        S = self.styles
        story = _section_header("P  –  PLAN", NAVY)

        # Plan narrative
        if note.plan_narrative:
            _subsection("Plan Summary", story)
            story.append(Paragraph(note.plan_narrative, S["body"]))
            story.append(Spacer(1, 6))

        # Plan items
        if note.plan_items:
            _subsection("Action Items", story)
            for item in note.plan_items:
                story.append(_bullet(item, S))
            story.append(Spacer(1, 6))

        # Prescribed medications
        if note.medications_prescribed:
            _subsection("Medications Prescribed / Changed", story)
            med_rows = [
                [
                    Paragraph("<b>Medication</b>", S["label"]),
                    Paragraph("<b>Dose</b>", S["label"]),
                    Paragraph("<b>Frequency</b>", S["label"]),
                    Paragraph("<b>Route</b>", S["label"]),
                    Paragraph("<b>Action</b>", S["label"]),
                ]
            ]
            for m in note.medications_prescribed:
                med_rows.append(
                    [
                        Paragraph(m.name, S["normal"]),
                        Paragraph(m.dose or "–", S["normal"]),
                        Paragraph(m.frequency or "–", S["normal"]),
                        Paragraph(m.route or "oral", S["normal"]),
                        Paragraph(
                            "Prescribe",
                            ParagraphStyle(
                                "PA",
                                fontName="Helvetica-Bold",
                                fontSize=8,
                                textColor=GREEN,
                            ),
                        ),
                    ]
                )
            for m in note.medications_discontinued:
                med_rows.append(
                    [
                        Paragraph(m.name, S["normal"]),
                        Paragraph("–", S["normal"]),
                        Paragraph("–", S["normal"]),
                        Paragraph("–", S["normal"]),
                        Paragraph(
                            "Discontinue",
                            ParagraphStyle(
                                "PD",
                                fontName="Helvetica-Bold",
                                fontSize=8,
                                textColor=RED_FLAG,
                            ),
                        ),
                    ]
                )
            mt = Table(
                med_rows,
                colWidths=[
                    CONTENT_W * 0.38,
                    CONTENT_W * 0.15,
                    CONTENT_W * 0.20,
                    CONTENT_W * 0.12,
                    CONTENT_W * 0.15,
                ],
            )
            mt.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 8),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
                        ("BOX", (0, 0), (-1, -1), 0.5, MID_GRAY),
                        ("INNERGRID", (0, 0), (-1, -1), 0.25, LIGHT_GRAY),
                        ("PADDING", (0, 0), (-1, -1), 5),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(mt)
            story.append(Spacer(1, 6))

        # Follow-up
        _subsection("Follow-Up", story)
        story.append(Paragraph(note.follow_up, S["body"]))
        story.append(Spacer(1, 4))

        return story

    # ── Supplemental sections ──────────────────────────────────────────────

    def _build_supplemental(self, note: SOAPNote) -> list:
        return []  # Social/family/ROS already in Subjective

    # ── Transcript ─────────────────────────────────────────────────────────

    def _build_transcript(self, note: SOAPNote) -> list:
        S = self.styles
        if not note.raw_transcript or note.raw_transcript.startswith("[No transcript"):
            return []

        story = [PageBreak()]
        story += _section_header("ENCOUNTER TRANSCRIPT (RAW)", MUTED)
        story.append(
            Paragraph(
                "<i>The following is the verbatim or manually-entered transcript from the clinical encounter. "
                "This section is included for documentation completeness and physician review.</i>",
                S["normal"],
            )
        )
        story.append(Spacer(1, 6))

        # Wrap long transcript into paragraphs
        lines = note.raw_transcript.split("\n")
        for line in lines:
            if line.strip():
                story.append(Paragraph(line.strip(), S["transcript"]))
            else:
                story.append(Spacer(1, 4))

        return story

    # ── Signature block ────────────────────────────────────────────────────

    def _build_signature(self, note: SOAPNote) -> list:
        S = self.styles
        story = [Spacer(1, 20)]
        story.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=MID_GRAY))
        story.append(Spacer(1, 8))

        sig_data = [
            [
                Paragraph(
                    f"Attending Physician: <b>{note.physician_name}</b><br/>"
                    f"Specialty: {note.physician_specialty}<br/>"
                    f"Date: {note.encounter_date}   Time: {note.encounter_time}",
                    ParagraphStyle(
                        "SL", fontName="Helvetica", fontSize=8.5, textColor=BLACK
                    ),
                ),
                Paragraph(
                    "Physician Signature: ______________________________<br/><br/>"
                    "Date Signed: ___________________",
                    ParagraphStyle(
                        "SR",
                        fontName="Helvetica",
                        fontSize=8.5,
                        textColor=MUTED,
                        alignment=TA_RIGHT,
                    ),
                ),
            ]
        ]
        st = Table(sig_data, colWidths=[CONTENT_W * 0.55, CONTENT_W * 0.45])
        st.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(st)
        story.append(Spacer(1, 10))
        story.append(
            Paragraph(
                "This report was generated by MedScribe — AI-Powered Clinical Documentation. "
                "All clinical content must be reviewed, verified, and signed by the attending physician. "
                "This document contains Protected Health Information (PHI) and is subject to HIPAA regulations.",
                ParagraphStyle(
                    "Disc",
                    fontName="Helvetica-Oblique",
                    fontSize=7,
                    textColor=MUTED,
                    alignment=TA_CENTER,
                    leading=10,
                ),
            )
        )
        return story
