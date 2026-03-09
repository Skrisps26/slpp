"""
DoctorSpeak – Streamlit UI
Fully local medical documentation system.
Run with: streamlit run app.py
"""

import base64
import io
import os
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Page config must be first ────────────────────────────────────────────────
st.set_page_config(
    page_title="DoctorSpeak – Clinical Documentation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project imports ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from nlp_engine import MedicalNLPEngine
from pdf_builder import PDFBuilder
from soap_builder import SOAPBuilder
from transcriber import load_transcript_from_file

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* ── Brand colors ─────────────────────── */
    :root {
        --navy:   #1A3557;
        --teal:   #1F7A8C;
        --accent: #2E86AB;
        --light:  #EEF4F8;
        --muted:  #6B7C8D;
        --red:    #C0392B;
        --green:  #27AE60;
        --yellow: #F39C12;
    }

    /* ── Global typography ────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* ── Sidebar ──────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--navy);
    }
    section[data-testid="stSidebar"] * {
        color: #e8eef4 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stTextArea label {
        color: #aac4d8 !important;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* ── Metric cards ─────────────────────── */
    [data-testid="metric-container"] {
        background: var(--light);
        border-radius: 10px;
        border-left: 4px solid var(--teal);
        padding: 12px 16px;
    }

    /* ── Tabs ─────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 2px solid var(--light);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: var(--navy) !important;
        color: white !important;
    }

    /* ── Section headers ──────────────────── */
    .section-header {
        background: var(--navy);
        color: white;
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.03em;
        margin: 18px 0 10px 0;
    }

    /* ── Alert boxes ──────────────────────── */
    .alert-critical {
        background: #fef2f2;
        border-left: 4px solid var(--red);
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-weight: 600;
        color: var(--red);
        font-size: 0.88rem;
    }
    .alert-warning {
        background: #fffbeb;
        border-left: 4px solid var(--yellow);
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-weight: 600;
        color: #92400e;
        font-size: 0.88rem;
    }
    .alert-info {
        background: #eff6ff;
        border-left: 4px solid var(--accent);
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        color: #1e40af;
        font-size: 0.88rem;
    }

    /* ── Symptom chip ─────────────────────── */
    .chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px 3px;
    }
    .chip-active   { background: #dcfce7; color: #166534; }
    .chip-negated  { background: #fee2e2; color: #991b1b; text-decoration: line-through; }
    .chip-med      { background: #dbeafe; color: #1e40af; }
    .chip-vital    { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }

    /* ── ICD badge ────────────────────────── */
    .icd-badge {
        background: #e0f2fe;
        color: #0369a1;
        font-family: monospace;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #bae6fd;
    }

    /* ── Vital card ───────────────────────── */
    .vital-card {
        background: var(--light);
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        border-top: 3px solid var(--teal);
        margin: 4px;
    }
    .vital-val  { font-size: 1.6rem; font-weight: 800; color: var(--navy); }
    .vital-unit { font-size: 0.72rem; color: var(--muted); }
    .vital-name { font-size: 0.78rem; font-weight: 600; color: var(--muted); margin-top: 4px; }
    .vital-ok   { color: var(--green) !important; }
    .vital-warn { color: var(--yellow) !important; }
    .vital-crit { color: var(--red) !important; }

    /* ── Recording indicator ──────────────── */
    .rec-indicator {
        display: inline-block;
        width: 12px; height: 12px;
        background: #ef4444;
        border-radius: 50%;
        margin-right: 8px;
        animation: blink 1s infinite;
    }
    @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.2; } }

    /* ── PDF download button ──────────────── */
    .stDownloadButton > button {
        background: var(--navy) !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 10px 24px !important;
        font-size: 0.95rem !important;
        width: 100%;
    }

    /* ── Expander ─────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: var(--navy);
    }

    /* ── Hide default Streamlit chrome ────── */
    #MainMenu, footer { visibility: hidden; }

    /* ── Divider ──────────────────────────── */
    hr { border-color: var(--light); margin: 18px 0; }
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def section(label: str, icon: str = ""):
    st.markdown(
        f'<div class="section-header">{icon}&nbsp; {label}</div>',
        unsafe_allow_html=True,
    )


def chip(text: str, kind: str = "active") -> str:
    return f'<span class="chip chip-{kind}">{text}</span>'


def vital_card(name: str, value: str, unit: str, status: str = "ok") -> str:
    return f"""
    <div class="vital-card">
        <div class="vital-val vital-{status}">{value}</div>
        <div class="vital-unit">{unit}</div>
        <div class="vital-name">{name}</div>
    </div>"""


def classify_vital(name: str, value: str) -> str:
    import re

    try:
        if "Blood Pressure" in name:
            sys_v = int(value.split("/")[0])
            if sys_v >= 180 or sys_v < 90:
                return "crit"
            if sys_v >= 140:
                return "warn"
            return "ok"
        if "O2" in name:
            v = float(re.sub(r"[^\d.]", "", value))
            return "crit" if v < 92 else "warn" if v < 95 else "ok"
        if "Heart Rate" in name:
            v = int(re.sub(r"[^\d]", "", value)[:3])
            return "warn" if (v > 120 or v < 50) else "ok"
        if "Temperature" in name:
            v = float(re.sub(r"[^\d.]", "", value))
            return "crit" if (v >= 103 or v < 96) else "warn" if v >= 100.4 else "ok"
        if "Glucose" in name:
            v = float(re.sub(r"[^\d.]", "", value))
            return (
                "crit"
                if (v > 400 or v < 60)
                else "warn"
                if (v > 250 or v < 70)
                else "ok"
            )
    except Exception:
        pass
    return "ok"


def render_flag(text: str):
    if "CRITICAL" in text or "RED FLAG" in text:
        st.markdown(
            f'<div class="alert-critical">🚨 {text}</div>', unsafe_allow_html=True
        )
    elif "ALERT" in text or "HIGH-ALERT" in text:
        st.markdown(
            f'<div class="alert-warning">⚠️ {text}</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(f'<div class="alert-info">📋 {text}</div>', unsafe_allow_html=True)


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio bytes with Whisper.
    Loads audio via scipy (no ffmpeg needed) and passes a float32
    numpy array directly to whisper.transcribe() — works on Streamlit Cloud.
    Returns transcript string, or None if Whisper is not installed.
    """
    try:
        import numpy as np
        import scipy.io.wavfile as wav_io
        import whisper
    except ImportError:
        return None

    try:
        with st.spinner("🎙️ Transcribing with Whisper..."):
            # Load WAV via scipy (no ffmpeg dependency)
            audio_buf = io.BytesIO(audio_bytes)
            sample_rate, audio_data = wav_io.read(audio_buf)

            # Convert to float32 in range [-1, 1] as Whisper expects
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_float = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.uint8:
                audio_float = (audio_data.astype(np.float32) - 128.0) / 128.0
            else:
                audio_float = audio_data.astype(np.float32)

            # Stereo to mono
            if audio_float.ndim == 2:
                audio_float = audio_float.mean(axis=1)

            # Resample to 16000 Hz if needed (Whisper requires 16kHz)
            if sample_rate != 16000:
                try:
                    import scipy.signal as signal

                    num_samples = int(len(audio_float) * 16000 / sample_rate)
                    audio_float = signal.resample(audio_float, num_samples)
                except Exception:
                    pass

            # Run Whisper on the numpy array directly
            model = whisper.load_model("base")
            result = model.transcribe(audio_float, language="en", fp16=False)
            return result["text"].strip()

    except Exception as e:
        st.warning(f"Transcription failed: {e}. Please paste the transcript manually.")
        return None


def run_pipeline(transcript: str, patient_info: dict) -> tuple:
    """Run NLP → SOAP → PDF and return (soap_note, pdf_bytes)."""
    pre_existing = {
        "current_medications": patient_info.get("current_medications", ""),
        "known_conditions": patient_info.get("known_conditions", ""),
        "allergies_list": [
            a.strip()
            for a in patient_info.get("allergies", "NKDA").split(",")
            if a.strip()
        ],
    }
    engine = MedicalNLPEngine()
    entities = engine.analyze(transcript)

    soap_note = SOAPBuilder().build(entities, transcript, patient_info, pre_existing)

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = PDFBuilder(output_dir=tmpdir).build(soap_note)
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

    return soap_note, pdf_bytes


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar – Patient & Encounter Info
# ═══════════════════════════════════════════════════════════════════════════


def sidebar_patient_form() -> dict:
    st.sidebar.markdown(
        """
    <div style="text-align:center;padding:16px 0 8px 0;">
        <span style="font-size:2rem;">🏥</span><br>
        <span style="font-size:1.25rem;font-weight:800;color:white;letter-spacing:0.04em;">MedScribe</span><br>
        <span style="font-size:0.72rem;color:#aac4d8;letter-spacing:0.08em;">LOCAL CLINICAL DOCUMENTATION</span>
    </div>
    <hr style="border-color:#2a4a6b;margin:8px 0 16px 0;">
    """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("**PATIENT INFORMATION**")
    patient_name = st.sidebar.text_input("Full Name", value="Anonymous Patient")
    patient_dob = st.sidebar.text_input("Date of Birth (YYYY-MM-DD)", value="")
    patient_id = st.sidebar.text_input(
        "MRN / Patient ID", value=f"PT-{datetime.now().strftime('%Y%m%d%H%M')}"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**ENCOUNTER**")
    doctor_name = st.sidebar.text_input("Physician Name", value="Dr. ")
    doctor_specialty = st.sidebar.selectbox(
        "Specialty",
        [
            "General Practice",
            "Internal Medicine",
            "Family Medicine",
            "Cardiology",
            "Pulmonology",
            "Neurology",
            "Orthopedics",
            "Endocrinology",
            "Gastroenterology",
            "Nephrology",
            "Psychiatry",
            "Oncology",
            "Emergency Medicine",
            "Other",
        ],
    )
    encounter_type = st.sidebar.selectbox(
        "Encounter Type",
        [
            "Office Visit",
            "Follow-up",
            "Urgent Care",
            "Telehealth",
            "Consultation",
            "Emergency",
        ],
    )
    facility = st.sidebar.text_input("Facility / Clinic", value="Medical Center")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**PRE-EXISTING CONTEXT** *(optional)*")
    known_conditions = st.sidebar.text_area(
        "Known Conditions", height=60, placeholder="e.g. Hypertension, Type 2 Diabetes"
    )
    current_medications = st.sidebar.text_area(
        "Current Medications",
        height=60,
        placeholder="e.g. Metformin 500mg, Lisinopril 10mg",
    )
    allergies = st.sidebar.text_input("Allergies", value="NKDA")

    return {
        "patient_name": patient_name,
        "patient_dob": patient_dob or "Unknown",
        "patient_id": patient_id,
        "doctor_name": doctor_name,
        "doctor_specialty": doctor_specialty,
        "encounter_type": encounter_type,
        "facility": facility,
        "known_conditions": known_conditions,
        "current_medications": current_medications,
        "allergies": allergies,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Report viewer – renders the SOAP note interactively in Streamlit
# ═══════════════════════════════════════════════════════════════════════════


def render_report(note, pdf_bytes: bytes):
    # ── Top: download + stats ────────────────────────────────────────────
    st.markdown("---")
    col_dl, col_s1, col_s2, col_s3, col_s4 = st.columns([2, 1, 1, 1, 1])

    safe = "".join(c if c.isalnum() else "_" for c in note.patient_name)
    fname = f"MedScribe_{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    with col_dl:
        st.download_button(
            "⬇️  Download PDF Report",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            use_container_width=True,
        )
    with col_s1:
        active = [s for s in note.entities.symptoms if not s.negated]
        st.metric("Symptoms", len(active))
    with col_s2:
        st.metric(
            "Medications",
            len(note.medications_current) + len(note.medications_prescribed),
        )
    with col_s3:
        st.metric("Diagnoses", len(note.entities.diagnoses))
    with col_s4:
        st.metric("🚨 Alerts", len(note.clinical_flags))

    # ── Clinical Alerts ──────────────────────────────────────────────────
    if note.clinical_flags:
        section("Clinical Alerts", "⚠️")
        for flag in note.clinical_flags:
            render_flag(flag)

    # ── Patient header ───────────────────────────────────────────────────
    age_str = f"&nbsp;·&nbsp; Age: {note.patient_age} yr" if note.patient_age else ""
    st.markdown(
        f"""
    <div style="background:#1A3557;border-radius:10px;padding:16px 20px;margin:12px 0;
                display:flex;justify-content:space-between;align-items:center;">
        <div>
            <div style="color:white;font-size:1.4rem;font-weight:800;">{note.patient_name}</div>
            <div style="color:#aac4d8;font-size:0.82rem;margin-top:2px;">
                DOB: {note.patient_dob} &nbsp;·&nbsp; MRN: {note.patient_id}{age_str}
            </div>
        </div>
        <div style="text-align:right;">
            <div style="color:white;font-weight:700;">{note.physician_name}</div>
            <div style="color:#aac4d8;font-size:0.8rem;">{note.physician_specialty} &nbsp;·&nbsp; {note.encounter_type}</div>
            <div style="color:#aac4d8;font-size:0.8rem;">{note.encounter_date} at {note.encounter_time}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── SOAP Tabs ────────────────────────────────────────────────────────
    tab_s, tab_o, tab_a, tab_p, tab_tx = st.tabs(
        [
            "📋 S – Subjective",
            "🩺 O – Objective",
            "🔍 A – Assessment",
            "📝 P – Plan",
            "🗣️ Transcript",
        ]
    )

    # ────────── S – Subjective ───────────────────────────────────────────
    with tab_s:
        col1, col2 = st.columns([3, 2])

        with col1:
            section("Chief Complaint", "💬")
            st.markdown(f"> {note.chief_complaint}")

            section("History of Present Illness", "📖")
            st.write(note.hpi)

            section("Symptoms", "🔴")
            active = [s for s in note.entities.symptoms if not s.negated]
            negated = [s for s in note.entities.symptoms if s.negated]

            if active:
                chips_html = " ".join(chip(s.name, "active") for s in active)
                st.markdown(
                    f'<div style="margin-bottom:10px;">{chips_html}</div>',
                    unsafe_allow_html=True,
                )

                # Detailed table
                import pandas as pd

                rows = [
                    {
                        "Symptom": s.name,
                        "Severity": s.severity or "–",
                        "Duration": s.duration or "–",
                        "Character": s.character or "–",
                        "Location": s.location or "–",
                    }
                    for s in active
                ]
                st.dataframe(
                    pd.DataFrame(rows), use_container_width=True, hide_index=True
                )

            if negated:
                st.markdown(
                    "**Patient denies:** "
                    + " ".join(chip(s.name, "negated") for s in negated),
                    unsafe_allow_html=True,
                )

        with col2:
            section("Allergies", "⚠️")
            allergy_str = "; ".join(note.allergies) if note.allergies else "NKDA"
            st.info(allergy_str)

            section("Current Medications", "💊")
            if note.medications_current:
                for m in note.medications_current:
                    parts = [m.name]
                    if m.dose:
                        parts.append(m.dose)
                    if m.frequency:
                        parts.append(m.frequency)
                    st.markdown(f"- {' · '.join(parts)}")
            else:
                st.caption("None documented")

            section("Review of Systems", "📋")
            ros = note.review_of_systems
            if ros:
                for system, data in ros.items():
                    pos = data.get("positive", [])
                    neg_list = data.get("negative", [])
                    if pos or neg_list:
                        with st.expander(system):
                            if pos:
                                st.markdown("✅ **Positive:** " + ", ".join(pos))
                            if neg_list:
                                st.markdown("❌ **Negative:** " + ", ".join(neg_list))
            else:
                st.caption("No ROS extracted")

            if note.social_history:
                section("Social History", "👤")
                for cat, detail in note.social_history.items():
                    st.markdown(f"**{cat.replace('_', ' ').title()}:** {detail}")

            if note.family_history:
                section("Family History", "👨‍👩‍👧")
                for item in note.family_history[:5]:
                    st.markdown(f"- {item}")

    # ────────── O – Objective ────────────────────────────────────────────
    with tab_o:
        section("Vital Signs", "📊")
        vitals = note.vitals_summary
        if vitals:
            cols = st.columns(min(len(vitals), 5))
            for i, v in enumerate(vitals):
                status = classify_vital(v.name, v.value)
                with cols[i % len(cols)]:
                    st.markdown(
                        vital_card(v.name, v.value, v.unit, status),
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("No vital signs extracted from transcript.")

        section("Physical Examination", "🩺")
        st.caption(
            "Physical examination findings as documented during the encounter. "
            "Add detailed exam notes here for physician review and signature."
        )
        st.text_area(
            "Examination Notes",
            height=120,
            placeholder="General: Alert and oriented x3, no acute distress\n"
            "HEENT: Normocephalic, atraumatic...\n"
            "CV: Regular rate and rhythm, no murmurs...",
            key="exam_notes",
        )

    # ────────── A – Assessment ───────────────────────────────────────────
    with tab_a:
        col1, col2 = st.columns([3, 2])

        with col1:
            section("Clinical Impression", "🧠")
            st.write(note.assessment_narrative)

            section("Diagnoses", "🏷️")
            all_dx = []
            if note.primary_diagnosis:
                all_dx.append(note.primary_diagnosis)
            all_dx.extend(note.secondary_diagnoses)

            if all_dx:
                import pandas as pd

                rows = []
                for i, dx in enumerate(all_dx, 1):
                    cert_colors = {
                        "confirmed": "🟢",
                        "possible": "🟡",
                        "ruled-out": "🔴",
                    }
                    rows.append(
                        {
                            "#": i,
                            "Diagnosis": dx.name,
                            "ICD-10": dx.icd10 or "–",
                            "Status": cert_colors.get(dx.certainty, "⚪")
                            + " "
                            + dx.certainty.title(),
                            "Type": "Primary" if dx.primary else "Secondary",
                        }
                    )
                st.dataframe(
                    pd.DataFrame(rows), use_container_width=True, hide_index=True
                )
            else:
                st.caption("No diagnoses extracted.")

        with col2:
            section("Differential Diagnoses", "🔄")
            if note.differential_diagnoses:
                for i, ddx in enumerate(note.differential_diagnoses, 1):
                    st.markdown(f"{i}. {ddx}")
            else:
                st.caption("None generated")

    # ────────── P – Plan ─────────────────────────────────────────────────
    with tab_p:
        col1, col2 = st.columns([3, 2])

        with col1:
            section("Plan Summary", "📝")
            st.write(note.plan_narrative)

            section("Action Items", "✅")
            if note.plan_items:
                for item in note.plan_items:
                    st.markdown(f"- {item}")
            else:
                st.caption("No specific plan items extracted.")

        with col2:
            if note.medications_prescribed or note.medications_discontinued:
                section("Medication Changes", "💊")
                for m in note.medications_prescribed:
                    parts = [m.name]
                    if m.dose:
                        parts.append(m.dose)
                    if m.frequency:
                        parts.append(m.frequency)
                    st.success("➕ Prescribe: " + " · ".join(parts))
                for m in note.medications_discontinued:
                    st.error("⛔ Discontinue: " + m.name)

            section("Follow-Up", "📅")
            st.info(note.follow_up or "As needed or per physician instruction")

    # ────────── Transcript ───────────────────────────────────────────────
    with tab_tx:
        section("Encounter Transcript", "🗣️")
        if note.raw_transcript and not note.raw_transcript.startswith("[No transcript"):
            st.text_area(
                "Raw Transcript", value=note.raw_transcript, height=500, disabled=True
            )
        else:
            st.caption("No transcript recorded.")


# ═══════════════════════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════════════════════


def main():
    # ── Session state init ───────────────────────────────────────────────
    for key in ["transcript", "soap_note", "pdf_bytes", "report_ready"]:
        if key not in st.session_state:
            st.session_state[key] = None
    if "report_ready" not in st.session_state:
        st.session_state.report_ready = False

    # ── Sidebar ──────────────────────────────────────────────────────────
    patient_info = sidebar_patient_form()

    # ── Main content ─────────────────────────────────────────────────────
    st.markdown(
        """
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;">
        <span style="font-size:2.2rem;">🏥</span>
        <div>
            <h1 style="margin:0;font-size:1.9rem;color:#1A3557;font-weight:800;">MedScribe</h1>
            <p style="margin:0;color:#6B7C8D;font-size:0.85rem;">
                Local AI Clinical Documentation &nbsp;·&nbsp; No API required &nbsp;·&nbsp;
                Record → Extract → SOAP Report
            </p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Input tabs ───────────────────────────────────────────────────────
    in_record, in_upload, in_paste, in_demo = st.tabs(
        [
            "🎙️ Record Encounter",
            "📁 Upload Audio",
            "📝 Paste Transcript",
            "🧪 Demo",
        ]
    )

    # ── Tab 1: Browser microphone ─────────────────────────────────────────
    with in_record:
        st.markdown(
            """
        <div class="alert-info">
            🎙️ Click <b>Start recording</b> below to capture the consultation directly in your browser.
            No PortAudio or system audio drivers needed.
            When done, click <b>Stop</b> — the audio will be transcribed automatically if
            <code>openai-whisper</code> is installed, or you can paste the transcript manually.
        </div>
        """,
            unsafe_allow_html=True,
        )

        audio_val = st.audio_input("Record the consultation", key="audio_recorder")

        if audio_val is not None:
            audio_bytes = audio_val.getvalue()
            st.success(f"✅ Audio captured ({len(audio_bytes) // 1024} KB)")
            st.audio(audio_val)

            # Try Whisper
            transcript_result = transcribe_audio(audio_bytes)
            if transcript_result:
                st.success("✅ Whisper transcription complete")
                st.session_state.transcript = transcript_result
                with st.expander("View transcript"):
                    st.write(transcript_result)
            else:
                st.warning(
                    "⚠️ **Whisper not installed** — automatic transcription unavailable.  \n"
                    "Install it with: `pip install openai-whisper`  \n"
                    "Or paste the transcript manually in the **Paste Transcript** tab."
                )

    # ── Tab 2: Upload audio / transcript file ────────────────────────────
    with in_upload:
        st.markdown("Upload a **WAV / MP3** recording, or a **TXT** transcript file.")

        uploaded = st.file_uploader(
            "Choose file",
            type=["wav", "mp3", "m4a", "txt"],
            label_visibility="collapsed",
        )
        if uploaded:
            if uploaded.name.endswith(".txt"):
                text = uploaded.read().decode("utf-8", errors="ignore")
                st.session_state.transcript = text
                st.success(f"✅ Transcript loaded ({len(text)} characters)")
                with st.expander("Preview"):
                    st.write(text[:2000])
            else:
                audio_bytes = uploaded.read()
                st.audio(audio_bytes)
                transcript_result = transcribe_audio(audio_bytes)
                if transcript_result:
                    st.session_state.transcript = transcript_result
                    st.success("✅ Whisper transcription complete")
                    with st.expander("View transcript"):
                        st.write(transcript_result)
                else:
                    st.warning(
                        "⚠️ Whisper not installed. "
                        "Paste the transcript manually in the **Paste Transcript** tab."
                    )

    # ── Tab 3: Manual paste ───────────────────────────────────────────────
    with in_paste:
        st.markdown("Paste or type the full encounter transcript below.")
        manual_text = st.text_area(
            "Encounter Transcript",
            height=320,
            placeholder=(
                "Doctor: What brings you in today?\n"
                "Patient: I've been having chest pain for the past three days...\n"
                "Doctor: On a scale of 1-10 how bad is the pain?\n"
                "..."
            ),
            label_visibility="collapsed",
        )
        if (
            st.button("Use This Transcript", use_container_width=True)
            and manual_text.strip()
        ):
            st.session_state.transcript = manual_text.strip()
            st.success("✅ Transcript saved")

    # ── Tab 4: Demo ───────────────────────────────────────────────────────
    with in_demo:
        st.markdown("""
        Run the full pipeline with a built-in demo consultation transcript —
        a follow-up visit for a patient with **hypertension, diabetes, and migraines**.
        """)
        if st.button("▶️  Run Demo", use_container_width=True, type="primary"):
            st.session_state.transcript = DEMO_TRANSCRIPT
            # Override sidebar info with demo patient
            patient_info.update(
                {
                    "patient_name": "Jane Smith",
                    "patient_dob": "1978-04-22",
                    "patient_id": "MRN-78042201",
                    "doctor_name": "Dr. Michael Chen",
                    "doctor_specialty": "Internal Medicine",
                    "facility": "Riverside Medical Center",
                    "encounter_type": "Office Visit",
                    "known_conditions": "Hypertension, Type 2 Diabetes",
                    "current_medications": "Metformin 500mg, Lisinopril 10mg",
                    "allergies": "Penicillin",
                }
            )
            st.session_state.report_ready = False  # force re-generation
            st.info("Demo transcript loaded. Click **Generate Report** below.")

    # ── Generate report ──────────────────────────────────────────────────
    st.markdown("---")

    transcript = st.session_state.get("transcript")
    has_transcript = bool(transcript and not transcript.startswith("[No transcript"))

    col_gen, col_clear = st.columns([4, 1])
    with col_gen:
        gen_disabled = not has_transcript
        if st.button(
            "🧠  Generate Clinical Report"
            if has_transcript
            else "🧠  Generate Report (load a transcript first)",
            use_container_width=True,
            type="primary" if has_transcript else "secondary",
            disabled=gen_disabled,
        ):
            with st.spinner("Running medical NLP pipeline..."):
                soap_note, pdf_bytes = run_pipeline(transcript, patient_info)
            st.session_state.soap_note = soap_note
            st.session_state.pdf_bytes = pdf_bytes
            st.session_state.report_ready = True
            st.rerun()

    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.transcript = None
            st.session_state.soap_note = None
            st.session_state.pdf_bytes = None
            st.session_state.report_ready = False
            st.rerun()

    # ── Show transcript status ────────────────────────────────────────────
    if has_transcript and not st.session_state.report_ready:
        st.markdown(
            f'<div class="alert-info">📄 Transcript ready ({len(transcript)} chars). Click <b>Generate Clinical Report</b> above.</div>',
            unsafe_allow_html=True,
        )

    # ── Render report ─────────────────────────────────────────────────────
    if st.session_state.report_ready and st.session_state.soap_note:
        render_report(st.session_state.soap_note, st.session_state.pdf_bytes)


# ═══════════════════════════════════════════════════════════════════════════
# Demo transcript
# ═══════════════════════════════════════════════════════════════════════════

DEMO_TRANSCRIPT = """Doctor: Good morning, Ms. Smith. What brings you in today?

Patient: Hi doctor. I've been having really bad headaches for the past two weeks. They're mostly on the right side, throbbing, and they get worse with bright light. I'd say the pain is about a seven out of ten.

Doctor: Any nausea or vomiting?

Patient: Yes, nausea with a few of them. No vomiting though.

Doctor: Any vision changes, weakness, or numbness?

Patient: No numbness or weakness. My vision gets a bit blurry during the worst ones.

Doctor: You're on lisinopril for hypertension, correct?

Patient: Yes, lisinopril 10mg daily. I checked at the pharmacy last week — my blood pressure was 158 over 92.

Doctor: Let me get your vitals. Blood pressure today is 162 over 94, heart rate 88, temperature 98.6, oxygen saturation 97 percent.

Doctor: How's your diabetes? Still on metformin?

Patient: Metformin 500mg twice daily. Morning blood sugar has been around 180 to 210. I haven't been great with my diet.

Doctor: Any fatigue, increased thirst, or frequent urination?

Patient: Definitely more tired than usual. More frequent urination — three or four times at night.

Doctor: Any fever, chills, or recent infections?

Patient: No fever or chills. I had a cold three weeks ago but it cleared up.

Doctor: Family history of migraines?

Patient: Yes, my mother had migraines her whole life.

Doctor: Any medication allergies?

Patient: Penicillin — I get a rash.

Doctor: Do you smoke or drink?

Patient: I quit smoking five years ago. I drink socially, maybe one glass of wine on weekends.

Doctor: Based on your history and symptoms, this presentation is consistent with migraines, likely exacerbated by suboptimal blood pressure control. Your diabetes management also needs adjustment given fasting glucose readings.

I'm prescribing sumatriptan 50mg for acute migraine attacks. I'll increase lisinopril to 20mg daily to better control the hypertension. Continue metformin.

I want labs — HbA1c, comprehensive metabolic panel, and lipid panel. Monitor blood pressure twice daily at home. If it exceeds 180 over 110, go to the emergency room immediately.

Patient: Okay, I understand.

Doctor: Follow up in four weeks to review your labs and assess headache response to sumatriptan. If headaches worsen or you develop sudden severe headache, vision loss, or new neurological symptoms, come in immediately or go to the ER.

Patient: Thank you, doctor."""


if __name__ == "__main__":
    main()
