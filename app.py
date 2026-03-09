"""
MedScribe – Streamlit UI
Fully local medical documentation system.
Run with: streamlit run app.py
"""

import io
import os
import sys
import time
import wave
import base64
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Page config must be first ────────────────────────────────────────────────
st.set_page_config(
    page_title="MedScribe – Clinical Documentation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project imports ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from nlp_engine import MedicalNLPEngine
from soap_builder import SOAPBuilder
from pdf_builder import PDFBuilder
from transcriber import load_transcript_from_file

@st.cache_resource(show_spinner="Loading medical NLP engine...")
def get_nlp_engine():
    """Cached so the spaCy model only loads once per Streamlit session."""
    return MedicalNLPEngine()

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
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
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def section(label: str, icon: str = ""):
    st.markdown(f'<div class="section-header">{icon}&nbsp; {label}</div>', unsafe_allow_html=True)


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
            if sys_v >= 180 or sys_v < 90: return "crit"
            if sys_v >= 140: return "warn"
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
            return "crit" if (v > 400 or v < 60) else "warn" if (v > 250 or v < 70) else "ok"
    except Exception:
        pass
    return "ok"


def render_flag(text: str):
    if "CRITICAL" in text or "RED FLAG" in text:
        st.markdown(f'<div class="alert-critical">🚨 {text}</div>', unsafe_allow_html=True)
    elif "ALERT" in text or "HIGH-ALERT" in text:
        st.markdown(f'<div class="alert-warning">⚠️ {text}</div>', unsafe_allow_html=True)
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
        import whisper
        import numpy as np
        import scipy.io.wavfile as wav_io
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
        "known_conditions":    patient_info.get("known_conditions", ""),
        "allergies_list": [
            a.strip()
            for a in patient_info.get("allergies", "NKDA").split(",")
            if a.strip()
        ],
    }
    engine = get_nlp_engine()
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
    st.sidebar.markdown("""
    <div style="text-align:center;padding:16px 0 8px 0;">
        <span style="font-size:2rem;">🏥</span><br>
        <span style="font-size:1.25rem;font-weight:800;color:white;letter-spacing:0.04em;">MedScribe</span><br>
        <span style="font-size:0.72rem;color:#aac4d8;letter-spacing:0.08em;">LOCAL CLINICAL DOCUMENTATION</span>
    </div>
    <hr style="border-color:#2a4a6b;margin:8px 0 16px 0;">
    """, unsafe_allow_html=True)

    st.sidebar.markdown("**PATIENT INFORMATION**")
    patient_name = st.sidebar.text_input("Full Name", value="Anonymous Patient")
    patient_dob  = st.sidebar.text_input("Date of Birth (YYYY-MM-DD)", value="")
    patient_id   = st.sidebar.text_input("MRN / Patient ID",
                                          value=f"PT-{datetime.now().strftime('%Y%m%d%H%M')}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**ENCOUNTER**")
    doctor_name      = st.sidebar.text_input("Physician Name", value="Dr. ")
    doctor_specialty = st.sidebar.selectbox("Specialty", [
        "General Practice", "Internal Medicine", "Family Medicine",
        "Cardiology", "Pulmonology", "Neurology", "Orthopedics",
        "Endocrinology", "Gastroenterology", "Nephrology", "Psychiatry",
        "Oncology", "Emergency Medicine", "Other",
    ])
    encounter_type = st.sidebar.selectbox("Encounter Type", [
        "Office Visit", "Follow-up", "Urgent Care",
        "Telehealth", "Consultation", "Emergency",
    ])
    facility = st.sidebar.text_input("Facility / Clinic", value="Medical Center")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**PRE-EXISTING CONTEXT** *(optional)*")
    known_conditions    = st.sidebar.text_area("Known Conditions", height=60,
                                                placeholder="e.g. Hypertension, Type 2 Diabetes")
    current_medications = st.sidebar.text_area("Current Medications", height=60,
                                                placeholder="e.g. Metformin 500mg, Lisinopril 10mg")
    allergies           = st.sidebar.text_input("Allergies", value="NKDA")

    return {
        "patient_name":        patient_name,
        "patient_dob":         patient_dob or "Unknown",
        "patient_id":          patient_id,
        "doctor_name":         doctor_name,
        "doctor_specialty":    doctor_specialty,
        "encounter_type":      encounter_type,
        "facility":            facility,
        "known_conditions":    known_conditions,
        "current_medications": current_medications,
        "allergies":           allergies,
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
        st.download_button("⬇️  Download PDF Report", data=pdf_bytes,
                           file_name=fname, mime="application/pdf",
                           use_container_width=True)
    with col_s1:
        active = [s for s in note.entities.symptoms if not s.negated]
        st.metric("Symptoms", len(active))
    with col_s2:
        st.metric("Medications", len(note.medications_current) + len(note.medications_prescribed))
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
    st.markdown(f"""
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
    """, unsafe_allow_html=True)

    # ── SOAP Tabs ────────────────────────────────────────────────────────
    tab_s, tab_o, tab_a, tab_p, tab_tx = st.tabs([
        "📋 S – Subjective",
        "🩺 O – Objective",
        "🔍 A – Assessment",
        "📝 P – Plan",
        "🗣️ Transcript",
    ])

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
                st.markdown(f'<div style="margin-bottom:10px;">{chips_html}</div>',
                            unsafe_allow_html=True)

                # Detailed table
                import pandas as pd
                rows = [{
                    "Symptom":   s.name,
                    "Severity":  s.severity or "–",
                    "Duration":  s.duration or "–",
                    "Character": s.character or "–",
                    "Location":  s.location or "–",
                } for s in active]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if negated:
                st.markdown("**Patient denies:** " +
                            " ".join(chip(s.name, "negated") for s in negated),
                            unsafe_allow_html=True)

        with col2:
            section("Allergies", "⚠️")
            allergy_str = "; ".join(note.allergies) if note.allergies else "NKDA"
            st.info(allergy_str)

            section("Current Medications", "💊")
            if note.medications_current:
                for m in note.medications_current:
                    parts = [m.name]
                    if m.dose:      parts.append(m.dose)
                    if m.frequency: parts.append(m.frequency)
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
                    st.markdown(f"**{cat.replace('_',' ').title()}:** {detail}")

            if not
