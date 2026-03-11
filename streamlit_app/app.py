"""
streamlit_app/app.py
MedScribe Ultra – Full Clinical AI Dashboard
Modular Streamlit UI with Ollama LLM, risk scores, differential diagnosis engine.
Run with: streamlit run streamlit_app/app.py
"""

import base64
import io
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="MedScribe Ultra – Clinical AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ── Lazy imports ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🏥 Loading MedScribe Ultra...")
def _get_pipeline():
    from streamlit_app.services.pipeline_runner import PipelineRunner
    return PipelineRunner()

def _get_ollama_status():
    try:
        from streamlit_app.services.ollama_client import check_ollama_health
        return check_ollama_health()
    except Exception:
        return {"available": False, "error": "Client not available"}

# ══════════════════════════════════════════════════════════════════════════════
# Global CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --navy:    #0F2440;
    --navy2:   #1A3557;
    --teal:    #1F7A8C;
    --accent:  #2E86AB;
    --light:   #EEF4F8;
    --muted:   #6B7C8D;
    --red:     #C0392B;
    --green:   #27AE60;
    --yellow:  #E67E22;
    --purple:  #8E44AD;
    --bg:      #F7FAFC;
}

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background: var(--bg);
}

/* ── Sidebar ─────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F2440 0%, #1A3557 100%);
}
section[data-testid="stSidebar"] * { color: #e8eef4 !important; }
section[data-testid="stSidebar"] label {
    color: #8ab4cc !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
section[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: white !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}

/* ── Metric cards ────────────────────────── */
[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    border-left: 4px solid var(--teal);
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* ── Tabs ────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--light);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 0.82rem;
    color: var(--muted);
}
.stTabs [aria-selected="true"] {
    background: var(--navy2) !important;
    color: white !important;
}

/* ── Section headers ──────────────────────── */
.section-header {
    background: linear-gradient(135deg, var(--navy2), var(--teal));
    color: white;
    padding: 10px 18px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 0.88rem;
    letter-spacing: 0.04em;
    margin: 18px 0 10px 0;
}

/* ── Alerts ───────────────────────────────── */
.alert-critical {
    background: linear-gradient(135deg, #fef2f2, #fff5f5);
    border-left: 4px solid var(--red);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-weight: 600;
    color: #991b1b;
    font-size: 0.85rem;
}
.alert-warning {
    background: linear-gradient(135deg, #fffbeb, #fef9ec);
    border-left: 4px solid var(--yellow);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-weight: 600;
    color: #78350f;
    font-size: 0.85rem;
}
.alert-info {
    background: linear-gradient(135deg, #eff6ff, #f0f7ff);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    color: #1e40af;
    font-size: 0.85rem;
}
.alert-success {
    background: linear-gradient(135deg, #f0fdf4, #f0fff4);
    border-left: 4px solid var(--green);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    color: #166534;
    font-size: 0.85rem;
}

/* ── Chips ────────────────────────────────── */
.chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 600;
    margin: 3px 3px;
    letter-spacing: 0.02em;
}
.chip-active  { background: #dcfce7; color: #166534; }
.chip-negated { background: #fee2e2; color: #991b1b; text-decoration: line-through; }
.chip-med     { background: #dbeafe; color: #1e40af; }
.chip-vital   { background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }
.chip-dx      { background: #f3e8ff; color: #6b21a8; }

/* ── Vital cards ──────────────────────────── */
.vital-card {
    background: white;
    border-radius: 14px;
    padding: 18px 12px;
    text-align: center;
    border-top: 4px solid var(--teal);
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin: 4px;
    transition: transform 0.2s;
}
.vital-card:hover { transform: translateY(-2px); }
.vital-val  { font-size: 1.8rem; font-weight: 800; color: var(--navy2); }
.vital-unit { font-size: 0.68rem; color: var(--muted); margin-top: 2px; }
.vital-name { font-size: 0.74rem; font-weight: 700; color: var(--muted); margin-top: 6px; text-transform: uppercase; letter-spacing: 0.06em; }
.vital-ok   { color: var(--green) !important; }
.vital-warn { color: var(--yellow) !important; }
.vital-crit { color: var(--red) !important; }

/* ── ICD badge ────────────────────────────── */
.icd-badge {
    background: #e0f2fe;
    color: #0369a1;
    font-family: 'Courier New', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid #bae6fd;
    margin-left: 6px;
}

/* ── Risk score cards ─────────────────────── */
.risk-card {
    background: white;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin: 8px 0;
    border-left: 5px solid var(--accent);
}
.risk-card.risk-low    { border-left-color: var(--green); }
.risk-card.risk-mod    { border-left-color: var(--yellow); }
.risk-card.risk-high   { border-left-color: var(--red); }
.risk-score-value { font-size: 2.4rem; font-weight: 800; }
.risk-score-value.low  { color: var(--green); }
.risk-score-value.mod  { color: var(--yellow); }
.risk-score-value.high { color: var(--red); }

/* ── Dx ranking table ─────────────────────── */
.dx-row {
    display: flex;
    align-items: center;
    padding: 10px 14px;
    border-radius: 8px;
    margin: 4px 0;
    background: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.dx-rank { font-size: 1.1rem; font-weight: 800; color: var(--navy2); min-width: 32px; }
.dx-name { font-weight: 600; font-size: 0.92rem; flex: 1; }
.dx-conf-bar {
    height: 8px;
    border-radius: 4px;
    background: var(--accent);
    display: inline-block;
}

/* ── Timeline ─────────────────────────────── */
.timeline-event {
    display: flex;
    gap: 14px;
    padding: 10px 0;
    align-items: flex-start;
}
.timeline-dot {
    width: 12px; height: 12px;
    border-radius: 50%;
    background: var(--teal);
    margin-top: 4px;
    flex-shrink: 0;
}
.timeline-time { font-size: 0.72rem; font-weight: 700; color: var(--teal); min-width: 80px; }
.timeline-text { font-size: 0.84rem; color: #374151; }

/* ── LLM assessment box ───────────────────── */
.llm-box {
    background: linear-gradient(135deg, #faf5ff, #f3e8ff);
    border: 1px solid #e9d5ff;
    border-radius: 12px;
    padding: 18px 20px;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #3b0764;
}

/* ── Ollama status badge ──────────────────── */
.ollama-online  { color: #22c55e; font-weight: 700; font-size: 0.78rem; }
.ollama-offline { color: #ef4444; font-weight: 700; font-size: 0.78rem; }

/* ── Header ───────────────────────────────── */
.ms-header {
    background: linear-gradient(135deg, #0F2440 0%, #1F7A8C 100%);
    border-radius: 16px;
    padding: 24px 30px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 20px;
}

/* ── Hide Streamlit chrome ────────────────── */
#MainMenu, footer { visibility: hidden; }
hr { border-color: var(--light); margin: 16px 0; }

/* ── Download button ──────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--navy2), var(--teal)) !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    font-size: 0.95rem !important;
    width: 100%;
    border: none !important;
    box-shadow: 0 4px 12px rgba(15,36,64,0.25) !important;
}

/* ── Progress bar ─────────────────────────── */
.confidence-bar-wrapper {
    background: var(--light);
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
    width: 120px;
    display: inline-block;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, var(--teal), var(--accent));
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

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


def render_confidence_bar(value: float) -> str:
    pct = int(value * 100)
    color = "#27AE60" if pct >= 60 else "#E67E22" if pct >= 35 else "#2E86AB"
    return (
        f'<div class="confidence-bar-wrapper">'
        f'<div class="confidence-bar-fill" style="width:{pct}%;background:{color};"></div>'
        f'</div> <span style="font-size:0.76rem;color:#6B7C8D;margin-left:6px;">{pct}%</span>'
    )


def risk_level_color(level: str) -> str:
    return {"LOW": "#27AE60", "MODERATE": "#E67E22", "HIGH": "#C0392B", "VERY_HIGH": "#7B0032"}.get(level, "#2E86AB")


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def sidebar_form() -> dict:
    st.sidebar.markdown("""
    <div style="text-align:center;padding:20px 0 12px 0;">
        <span style="font-size:2.5rem;">🏥</span><br>
        <span style="font-size:1.35rem;font-weight:800;color:white;letter-spacing:0.04em;">MedScribe</span><br>
        <span style="font-size:0.68rem;color:#8ab4cc;letter-spacing:0.12em;font-weight:600;">ULTRA · LOCAL AI</span>
    </div>
    <hr style="border-color:#2a4a6b;margin:8px 0 16px 0;">
    """, unsafe_allow_html=True)

    # ── Ollama status ─────────────────────────────────────────────────────
    if "ollama_status" not in st.session_state:
        st.session_state.ollama_status = None

    col_s, col_r = st.sidebar.columns([3, 1])
    with col_s:
        if st.session_state.ollama_status:
            health = st.session_state.ollama_status
            if health.get("available"):
                st.markdown(
                    f'<div class="ollama-online">🟢 Ollama · {health.get("model","")}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="ollama-offline">🔴 Ollama offline</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div style="color:#8ab4cc;font-size:0.75rem;">LLM status unknown</div>', unsafe_allow_html=True)

    with col_r:
        if st.sidebar.button("⟳", help="Check Ollama connection"):
            with st.spinner(""):
                st.session_state.ollama_status = _get_ollama_status()
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**PATIENT INFORMATION**")
    patient_name = st.sidebar.text_input("Full Name", value="Anonymous Patient")
    patient_dob = st.sidebar.text_input("Date of Birth (YYYY-MM-DD)", value="")
    patient_id = st.sidebar.text_input(
        "MRN / Patient ID",
        value=f"PT-{datetime.now().strftime('%Y%m%d%H%M')}"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**ENCOUNTER**")
    doctor_name = st.sidebar.text_input("Physician Name", value="Dr. ")
    doctor_specialty = st.sidebar.selectbox("Specialty", [
        "General Practice", "Internal Medicine", "Family Medicine",
        "Cardiology", "Pulmonology", "Neurology", "Orthopedics",
        "Endocrinology", "Gastroenterology", "Emergency Medicine", "Other",
    ])
    encounter_type = st.sidebar.selectbox("Encounter Type", [
        "Office Visit", "Follow-up", "Urgent Care",
        "Telehealth", "Consultation", "Emergency",
    ])
    facility = st.sidebar.text_input("Facility / Clinic", value="Medical Center")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**PRE-EXISTING CONTEXT**")
    known_conditions = st.sidebar.text_area("Known Conditions", height=55,
        placeholder="e.g. Hypertension, Type 2 Diabetes")
    current_medications = st.sidebar.text_area("Current Medications", height=55,
        placeholder="e.g. Metformin 500mg, Lisinopril 10mg")
    allergies = st.sidebar.text_input("Allergies", value="NKDA")

    # ── LLM toggle ────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    use_llm = st.sidebar.toggle("🤖 Ollama LLM Reasoning", value=True,
        help="Uses local Ollama model to generate clinical assessment. Requires Ollama running.")

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
        "use_llm": use_llm,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Report viewer with enhanced tabs
# ══════════════════════════════════════════════════════════════════════════════

def render_report(result: dict, patient_info: dict):
    note = result["soap_note"]
    pdf_bytes = result["pdf_bytes"]
    risk_scores = result.get("risk_scores", [])
    entities = result.get("entities")
    diagnoses = result.get("diagnoses", [])
    llm_assessment = result.get("llm_assessment")

    st.markdown("---")

    # ── Stats bar ─────────────────────────────────────────────────────────
    active_syms = [s for s in note.entities.symptoms if not s.negated] if note.entities else []
    col_dl, c1, c2, c3, c4, c5 = st.columns([2.2, 1, 1, 1, 1, 1])

    safe = "".join(c if c.isalnum() else "_" for c in note.patient_name)
    fname = f"MedScribe_{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    with col_dl:
        st.download_button(
            "⬇️  Download PDF Report",
            data=pdf_bytes, file_name=fname,
            mime="application/pdf",
            use_container_width=True,
        )
    with c1: st.metric("Symptoms", len(active_syms))
    with c2: st.metric("Medications", len(note.medications_current) + len(note.medications_prescribed))
    with c3: st.metric("Diagnoses", len(diagnoses))
    with c4: st.metric("🚨 Alerts", len(note.clinical_flags))
    with c5: st.metric("Risk Scores", len(risk_scores))

    # ── Alerts bar ────────────────────────────────────────────────────────
    if note.clinical_flags:
        section("Clinical Alerts", "⚠️")
        for flag in note.clinical_flags:
            render_flag(flag)

    # ── Patient header ────────────────────────────────────────────────────
    age_str = f"&nbsp;·&nbsp; Age: {note.patient_age} yr" if note.patient_age else ""
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0F2440,#1F7A8C);border-radius:14px;
                padding:20px 26px;margin:14px 0;display:flex;
                justify-content:space-between;align-items:center;">
        <div>
            <div style="color:white;font-size:1.5rem;font-weight:800;">{note.patient_name}</div>
            <div style="color:#aac4d8;font-size:0.8rem;margin-top:4px;">
                DOB: {note.patient_dob} &nbsp;·&nbsp; MRN: {note.patient_id}{age_str}
            </div>
        </div>
        <div style="text-align:right;">
            <div style="color:white;font-weight:700;font-size:1rem;">{note.physician_name}</div>
            <div style="color:#aac4d8;font-size:0.78rem;">{note.physician_specialty} &nbsp;·&nbsp; {note.encounter_type}</div>
            <div style="color:#aac4d8;font-size:0.78rem;">{note.encounter_date} at {note.encounter_time}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SOAP + Extended Tabs ───────────────────────────────────────────────
    tab_s, tab_o, tab_a, tab_p, tab_risk, tab_timeline, tab_llm, tab_tx = st.tabs([
        "📋 S – Subjective",
        "🩺 O – Objective",
        "🔍 A – Assessment",
        "📝 P – Plan",
        "📊 Risk Scores",
        "🕐 Timeline",
        "🤖 LLM Reasoning",
        "🗣️ Transcript",
    ])

    # ── S – Subjective ────────────────────────────────────────────────────
    with tab_s:
        col1, col2 = st.columns([3, 2])
        with col1:
            section("Chief Complaint", "💬")
            st.markdown(f"> {note.chief_complaint}")

            section("History of Present Illness", "📖")
            st.write(note.hpi)

            section("Symptoms", "🔴")
            if active_syms:
                chips_html = " ".join(chip(s.name, "active") for s in active_syms)
                st.markdown(f'<div style="margin-bottom:12px;">{chips_html}</div>', unsafe_allow_html=True)

                import pandas as pd
                rows = [{
                    "Symptom": s.name,
                    "Severity": s.severity or "–",
                    "Duration": s.duration or "–",
                    "Character": s.character or "–",
                    "Location": s.location or "–",
                    "ICD-10": getattr(s, "icd10", None) or "–",
                } for s in active_syms]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            negated = [s for s in note.entities.symptoms if s.negated] if note.entities else []
            if negated:
                st.markdown("**Patient denies:** " + " ".join(chip(s.name, "negated") for s in negated),
                            unsafe_allow_html=True)

        with col2:
            section("Allergies", "⚠️")
            st.info("; ".join(note.allergies) if note.allergies else "NKDA")

            section("Current Medications", "💊")
            if note.medications_current:
                for m in note.medications_current:
                    parts = [m.name]
                    if m.dose: parts.append(m.dose)
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
                            if pos: st.markdown("✅ **Positive:** " + ", ".join(pos))
                            if neg_list: st.markdown("❌ **Negative:** " + ", ".join(neg_list))
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

    # ── O – Objective ─────────────────────────────────────────────────────
    with tab_o:
        section("Vital Signs", "📊")
        vitals = note.vitals_summary
        if vitals:
            cols = st.columns(min(len(vitals), 5))
            for i, v in enumerate(vitals):
                status = classify_vital(v.name, v.value)
                with cols[i % len(cols)]:
                    st.markdown(vital_card(v.name, v.value, v.unit, status), unsafe_allow_html=True)
        else:
            st.caption("No vital signs extracted from transcript.")

        section("Physical Examination", "🩺")
        st.text_area("Examination Notes", height=120, key="exam_notes",
            placeholder="General: Alert and oriented x3, no acute distress\nCV: Regular rate and rhythm...")

    # ── A – Assessment ────────────────────────────────────────────────────
    with tab_a:
        col1, col2 = st.columns([3, 2])
        with col1:
            section("Clinical Impression", "🧠")
            st.write(note.assessment_narrative)

            section("Ranked Differential Diagnoses", "🏷️")
            if diagnoses:
                import pandas as pd
                cert_icons = {"confirmed": "🟢", "possible": "🟡", "ruled-out": "🔴"}
                for i, dx in enumerate(diagnoses[:8], 1):
                    conf_pct = int(dx.confidence * 100)
                    icd_badge = f'<span class="icd-badge">{dx.icd10}</span>' if dx.icd10 else ""
                    matched = ", ".join(dx.matched_symptoms[:3]) if dx.matched_symptoms else "—"
                    cert_icon = cert_icons.get(dx.certainty, "⚪")

                    st.markdown(f"""
                    <div class="dx-row" style="border-left: 3px solid {'#27AE60' if i==1 else '#e5e7eb'};">
                        <div class="dx-rank">#{i}</div>
                        <div class="dx-name">
                            {dx.name}{icd_badge}
                            <div style="font-size:0.72rem;color:#9ca3af;font-weight:400;">Matched: {matched}</div>
                        </div>
                        <div style="text-align:right;">
                            {render_confidence_bar(dx.confidence)}
                            <div style="font-size:0.72rem;color:#9ca3af;">{cert_icon} {dx.certainty}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No diagnoses generated.")

        with col2:
            section("Recommended Tests", "🔬")
            rec_tests = result.get("recommended_tests", [])
            if rec_tests:
                for t in rec_tests:
                    st.markdown(f"- {t}")
            else:
                st.caption("No specific tests recommended")

            red_flags = result.get("red_flags", [])
            if red_flags:
                section("Red Flags", "🚩")
                for rf in red_flags:
                    st.markdown(f'<div class="alert-warning">🚩 {rf}</div>', unsafe_allow_html=True)

    # ── P – Plan ──────────────────────────────────────────────────────────
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
                    if m.dose: parts.append(m.dose)
                    if m.frequency: parts.append(m.frequency)
                    st.success("➕ Prescribe: " + " · ".join(parts))
                for m in note.medications_discontinued:
                    st.error("⛔ Discontinue: " + m.name)

            section("Follow-Up", "📅")
            st.info(note.follow_up or "As needed or per physician instruction")

    # ── Risk Scores ───────────────────────────────────────────────────────
    with tab_risk:
        if risk_scores:
            for score in risk_scores:
                level_css = {"LOW": "risk-low", "MODERATE": "risk-mod", "HIGH": "risk-high", "VERY_HIGH": "risk-high"}.get(score.risk_level, "")
                color = risk_level_color(score.risk_level)
                score_css = {"LOW": "low", "MODERATE": "mod", "HIGH": "high"}.get(score.risk_level, "")

                st.markdown(f"""
                <div class="risk-card {level_css}">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div>
                            <div style="font-weight:800;font-size:1rem;color:#0F2440;">{score.name}</div>
                            <div style="font-size:0.82rem;color:#6B7C8D;margin-top:2px;">{score.interpretation}</div>
                        </div>
                        <div class="risk-score-value {score_css}">{score.score}<span style="font-size:1rem;color:#9ca3af;">/{score.max_score}</span></div>
                    </div>
                    <div style="margin-top:12px;padding:10px 14px;background:#f9fafb;border-radius:8px;font-size:0.82rem;color:#374151;">
                        💊 {score.recommendation}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"Score Breakdown — {score.name}"):
                    import pandas as pd
                    comp_rows = [{"Component": k, "Points": v} for k, v in score.components.items()]
                    if comp_rows:
                        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div class="alert-info">
                📊 No risk scores applicable for the current presentation.<br>
                Risk scores (Wells PE, HEART Score, CHA₂DS₂-VASc) are activated contextually
                based on symptoms detected.
            </div>
            """, unsafe_allow_html=True)

    # ── Timeline ──────────────────────────────────────────────────────────
    with tab_timeline:
        section("Clinical Timeline", "🕐")
        timeline = getattr(entities, "timeline", None) if entities else None
        if timeline and timeline.events:
            type_icons = {
                "onset": "🔴", "relative_past": "📅", "today": "⏰",
                "duration": "⌛", "progression": "📈", "treatment": "💊",
                "medical_event": "🏥", "test": "🔬", "general": "📋",
            }
            for event in timeline.events[:15]:
                icon = type_icons.get(event["type"], "📋")
                st.markdown(f"""
                <div class="timeline-event">
                    <div>
                        <div class="timeline-dot"></div>
                    </div>
                    <div>
                        <div class="timeline-time">{icon} {event['time']}</div>
                        <div class="timeline-text">{event['event'][:160]}</div>
                    </div>
                </div>
                <hr style="margin:4px 0;border-color:#f3f4f6;">
                """, unsafe_allow_html=True)
        else:
            st.caption("No temporal events extracted from transcript.")

    # ── LLM Reasoning ─────────────────────────────────────────────────────
    with tab_llm:
        section("Local LLM Clinical Reasoning", "🤖")
        if llm_assessment:
            st.markdown(f"""
            <div class="llm-box">
                <div style="font-size:0.7rem;font-weight:700;color:#7c3aed;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:8px;">
                    🤖 Generated by {st.session_state.get('ollama_status', {}).get('model', 'Local LLM')}
                </div>
                {llm_assessment.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
        else:
            health = _get_ollama_status()
            if health.get("available"):
                st.markdown('<div class="alert-info">🤖 LLM reasoning was not requested or returned empty. Enable the LLM toggle in the sidebar and re-run.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-warning">
                    🔴 <b>Ollama not connected</b> — LLM reasoning unavailable.<br><br>
                    To enable LLM reasoning:<br>
                    1. Install Ollama: <code>https://ollama.ai</code><br>
                    2. Run: <code>ollama pull llama3</code><br>
                    3. Start: <code>ollama serve</code><br>
                    4. Click ⟳ in the sidebar to refresh status.
                </div>
                """, unsafe_allow_html=True)

    # ── Transcript ────────────────────────────────────────────────────────
    with tab_tx:
        section("Encounter Transcript", "🗣️")
        if note.raw_transcript and not note.raw_transcript.startswith("[No transcript"):
            st.text_area("Raw Transcript", value=note.raw_transcript, height=500, disabled=True)
        else:
            st.caption("No transcript recorded.")


# ══════════════════════════════════════════════════════════════════════════════
# Demo transcript
# ══════════════════════════════════════════════════════════════════════════════

DEMO_TRANSCRIPT = """Doctor: Good morning, Ms. Smith. What brings you in today?

Patient: Hi doctor. I've been having really bad headaches for the past two weeks.
They're mostly on the right side, throbbing, and they get worse with bright light.
I'd say the pain is about a seven out of ten.

Doctor: Any nausea or vomiting?

Patient: Yes, nausea with a few of them. No vomiting though.

Doctor: Any vision changes, weakness, or numbness?

Patient: No numbness or weakness. My vision gets a bit blurry during the worst ones.

Doctor: You're on lisinopril for hypertension, correct?

Patient: Yes, lisinopril 10mg daily. I checked at the pharmacy last week —
my blood pressure was 158 over 92.

Doctor: Let me get your vitals. Blood pressure today is 162 over 94,
heart rate 88, temperature 98.6, oxygen saturation 97%.

Patient: Is that concerning?

Doctor: A bit high. How's your diabetes? Still taking metformin?

Patient: Metformin 500mg twice daily. Morning blood sugar has been around
180 to 210. I haven't been great with my diet.

Doctor: Any fatigue, increased thirst, or frequent urination?

Patient: Definitely more tired. More frequent urination — three or four times at night.

Doctor: Any fever, chills, recent infections?

Patient: No fever or chills. I had a cold three weeks ago but it cleared up.

Doctor: Family history of migraines?

Patient: Yes, my mother had migraines her whole life.

Doctor: Any medication allergies?

Patient: Penicillin — I get a rash.

Doctor: Do you smoke or drink?

Patient: I quit smoking five years ago. I drink socially, one glass of wine on weekends.

Doctor: Based on your history, symptoms, and family history, this presentation is
consistent with migraines. Your hypertension is also not optimally controlled,
and your blood glucose suggests your diabetes management needs adjustment.

I'm prescribing sumatriptan 50mg for acute migraine attacks.
I'll increase lisinopril to 20mg daily. Continue metformin.

I want labs — HbA1c, comprehensive metabolic panel, and lipid panel.
Monitor blood pressure twice daily at home. If it exceeds 180 over 110,
go to the emergency room immediately.

Patient: Okay, I understand.

Doctor: Follow up in four weeks to review your labs and headache response.
If headaches worsen or you develop sudden severe headache, vision loss,
or any new neurological symptoms, come in immediately or go to the ER.

Patient: Thank you, doctor."""

DEMO_PATIENT = {
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


# ══════════════════════════════════════════════════════════════════════════════
# Main app
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Session state ─────────────────────────────────────────────────────
    for key in ["result", "transcript"]:
        if key not in st.session_state:
            st.session_state[key] = None
    if "report_ready" not in st.session_state:
        st.session_state.report_ready = False

    # ── Sidebar ───────────────────────────────────────────────────────────
    patient_info = sidebar_form()

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="ms-header">
        <span style="font-size:3rem;">🏥</span>
        <div>
            <div style="color:white;font-size:2rem;font-weight:800;line-height:1.1;">MedScribe Ultra</div>
            <div style="color:#8ab4cc;font-size:0.85rem;margin-top:4px;">
                Local Clinical AI &nbsp;·&nbsp; No API Required &nbsp;·&nbsp;
                Audio → NLP → Diagnosis → Risk → Report
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input tabs ────────────────────────────────────────────────────────
    tab_record, tab_upload, tab_paste, tab_demo = st.tabs([
        "🎙️ Record", "📁 Upload Audio", "📝 Paste Text", "🧪 Demo",
    ])

    with tab_record:
        st.markdown('<div class="alert-info">🎙️ Use your browser microphone to record a consultation. Click <b>Start recording</b>, then <b>Stop</b> when done.</div>', unsafe_allow_html=True)
        audio_val = st.audio_input("Record encounter audio")
        if audio_val is not None:
            audio_bytes = audio_val.read()
            st.audio(audio_bytes, format="audio/wav")
            if st.button("🔬 Transcribe & Analyze", use_container_width=True, key="rec_run"):
                transcript = _transcribe(audio_bytes)
                if transcript:
                    st.session_state.transcript = transcript
                    _run_pipeline(transcript, patient_info)

    with tab_upload:
        uploaded = st.file_uploader("Upload audio file (WAV recommended)", type=["wav", "mp3", "m4a", "ogg"])
        if uploaded:
            audio_bytes = uploaded.read()
            st.audio(audio_bytes)
            if st.button("🔬 Transcribe & Analyze", use_container_width=True, key="up_run"):
                transcript = _transcribe(audio_bytes)
                if transcript:
                    st.session_state.transcript = transcript
                    _run_pipeline(transcript, patient_info)

    with tab_paste:
        transcript_text = st.text_area(
            "Paste encounter transcript",
            height=280,
            placeholder="Doctor: Good morning. What brings you in?\nPatient: I've been having chest pain for two days...",
        )
        if st.button("🔬 Analyze Transcript", use_container_width=True, key="paste_run"):
            if transcript_text.strip():
                st.session_state.transcript = transcript_text
                _run_pipeline(transcript_text, patient_info)
            else:
                st.warning("Please paste a transcript first.")

    with tab_demo:
        st.markdown("""
        <div class="alert-success">
            🧪 <b>Demo Mode</b> — Uses a pre-built doctor-patient consultation about migraines,
            hypertension, and diabetes. Activates all pipeline features including risk scoring
            and (if Ollama is running) LLM reasoning.
        </div>
        """, unsafe_allow_html=True)

        st.text_area("Demo Transcript Preview", value=DEMO_TRANSCRIPT[:600] + "...", height=180, disabled=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("▶️ Run Demo", use_container_width=True, key="demo_run"):
                merged_info = {**DEMO_PATIENT, "use_llm": patient_info.get("use_llm", True)}
                st.session_state.transcript = DEMO_TRANSCRIPT
                _run_pipeline(DEMO_TRANSCRIPT, merged_info)
        with col2:
            st.caption("Patient: Jane Smith, DOB: 1978-04-22\nPhysician: Dr. Michael Chen · Internal Medicine")

    # ── Report ────────────────────────────────────────────────────────────
    if st.session_state.report_ready and st.session_state.result:
        render_report(st.session_state.result, patient_info)


def _transcribe(audio_bytes: bytes):
    """Transcribe audio using Whisper if available."""
    try:
        import numpy as np
        import scipy.io.wavfile as wav_io
        import whisper

        with st.spinner("🎙️ Transcribing with Whisper..."):
            audio_buf = __import__("io").BytesIO(audio_bytes)
            sample_rate, audio_data = wav_io.read(audio_buf)

            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)

            if audio_float.ndim == 2:
                audio_float = audio_float.mean(axis=1)

            if sample_rate != 16000:
                import scipy.signal as signal
                num_samples = int(len(audio_float) * 16000 / sample_rate)
                audio_float = signal.resample(audio_float, num_samples)

            model = whisper.load_model("base")
            result = model.transcribe(audio_float, language="en", fp16=False)
            return result["text"].strip()

    except ImportError:
        st.warning("Whisper not installed. Please paste the transcript manually.")
        return None
    except Exception as e:
        st.warning(f"Transcription failed: {e}. Please paste the transcript manually.")
        return None


def _run_pipeline(transcript: str, patient_info: dict):
    """Run the full pipeline and store results in session state."""
    pipeline = _get_pipeline()
    use_llm = patient_info.get("use_llm", True)

    steps = [
        "🔬 Extracting clinical entities...",
        "🧬 Running diagnosis engine...",
        "📊 Computing risk scores...",
        "🚨 Generating alerts...",
        "🤖 Requesting LLM reasoning..." if use_llm else "🤖 LLM skipped",
        "📄 Building SOAP note...",
        "🖨️ Generating PDF...",
    ]

    progress_bar = st.progress(0)
    status_text = st.empty()

    def update(step_idx):
        progress_bar.progress((step_idx + 1) / len(steps))
        status_text.markdown(f'<div class="alert-info">{steps[step_idx]}</div>', unsafe_allow_html=True)
        time.sleep(0.1)

    try:
        for i in range(len(steps)):
            update(i)

        result = pipeline.run(transcript, patient_info, use_llm=use_llm)

        progress_bar.progress(1.0)
        status_text.markdown('<div class="alert-success">✅ Analysis complete!</div>', unsafe_allow_html=True)
        time.sleep(0.5)

        st.session_state.result = result
        st.session_state.report_ready = True
        progress_bar.empty()
        status_text.empty()
        st.rerun()

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Pipeline error: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
