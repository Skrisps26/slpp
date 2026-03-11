"""
streamlit_app/services/pipeline_runner.py
Orchestrates the full MedScribe Ultra pipeline:
  Transcript → NLP → Diagnosis Engine → Risk Scoring → Alerts → LLM → SOAP → PDF
"""

import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from nlp_engine.core import MedicalNLPEngine
from reasoning.diagnosis_engine import DiagnosisEngine
from reasoning.risk_scores import RiskScoreEngine
from reasoning.alerts import AlertEngine
from soap_builder import SOAPBuilder
from pdf_builder import PDFBuilder

# Optional LLM
try:
    from streamlit_app.services.ollama_client import (
        ask_llm, build_clinical_reasoning_prompt, check_ollama_health
    )
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False


class PipelineRunner:
    """
    Full MedScribe Ultra pipeline orchestrator.
    Call .run(transcript, patient_info) to get (soap_note, pdf_bytes, risk_scores, llm_assessment).
    """

    def __init__(self):
        self.nlp = MedicalNLPEngine()
        self.dx_engine = DiagnosisEngine()
        self.risk_engine = RiskScoreEngine()
        self.alert_engine = AlertEngine()

    def run(
        self,
        transcript: str,
        patient_info: dict,
        use_llm: bool = True,
    ) -> dict:
        """
        Full pipeline execution.
        Returns dict with keys:
          soap_note, pdf_bytes, risk_scores, llm_assessment, entities, diagnoses
        """
        result = {}

        # ── Step 1: NLP entity extraction ─────────────────────────────────
        entities = self.nlp.analyze(transcript)
        result["entities"] = entities

        # ── Step 2: Differential diagnosis ranking ────────────────────────
        diagnoses = self.dx_engine.rank(entities, top_n=8)
        # Merge with any already-extracted diagnoses, preferring engine output
        if diagnoses:
            entities.diagnoses = diagnoses
        result["diagnoses"] = diagnoses

        # ── Step 3: Risk scoring ──────────────────────────────────────────
        risk_scores = self.risk_engine.compute_all(entities, patient_info)
        result["risk_scores"] = risk_scores

        # ── Step 4: Enhanced alerts ───────────────────────────────────────
        enhanced_alerts = self.alert_engine.generate(entities, patient_info)
        result["enhanced_alerts"] = enhanced_alerts

        # ── Step 5: Recommended tests from top diagnosis ──────────────────
        if diagnoses:
            rec_tests = self.dx_engine.get_recommended_tests(diagnoses[0].name)
            result["recommended_tests"] = rec_tests
            red_flags = self.dx_engine.get_red_flags(diagnoses[0].name)
            result["red_flags"] = red_flags
        else:
            result["recommended_tests"] = []
            result["red_flags"] = []

        # ── Step 6: LLM reasoning (optional) ─────────────────────────────
        llm_assessment = None
        if use_llm and _LLM_AVAILABLE:
            try:
                health = check_ollama_health()
                if health.get("available"):
                    prompt = build_clinical_reasoning_prompt(
                        symptoms=entities.symptoms,
                        vitals=entities.vitals,
                        medications=entities.medications,
                        diagnoses=diagnoses[:5],
                        patient_info=patient_info,
                    )
                    llm_assessment = ask_llm(prompt, model=health.get("model"))
            except Exception:
                llm_assessment = None
        result["llm_assessment"] = llm_assessment

        # ── Step 7: Build SOAP note ───────────────────────────────────────
        pre_existing = {
            "current_medications": patient_info.get("current_medications", ""),
            "known_conditions": patient_info.get("known_conditions", ""),
            "allergies_list": [
                a.strip()
                for a in patient_info.get("allergies", "NKDA").split(",")
                if a.strip()
            ],
        }

        soap_note = SOAPBuilder().build(
            entities, transcript, patient_info, pre_existing
        )

        # Inject enhanced alerts into SOAP note
        if enhanced_alerts:
            existing = set(soap_note.clinical_flags)
            for alert in enhanced_alerts:
                if alert not in existing:
                    soap_note.clinical_flags.append(alert)

        # Attach risk scores and LLM assessment to soap note for PDF
        soap_note.risk_scores = risk_scores
        soap_note.llm_assessment = llm_assessment
        soap_note.recommended_tests = result.get("recommended_tests", [])

        result["soap_note"] = soap_note

        # ── Step 8: Generate PDF ──────────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = PDFBuilder(output_dir=tmpdir).build(soap_note)
            with open(pdf_path, "rb") as f:
                result["pdf_bytes"] = f.read()

        return result
