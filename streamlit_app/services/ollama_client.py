"""
streamlit_app/services/ollama_client.py
Local LLM client for Ollama.
Handles: connection, health check, prompt generation, streaming, graceful fallback.
"""

import os
import json
import time
from typing import Optional, Iterator

try:
    import requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
TIMEOUT = 120  # seconds


# ── Health check ──────────────────────────────────────────────────────────────

def check_ollama_health() -> dict:
    """
    Returns: {"available": bool, "model": str|None, "host": str, "error": str|None}
    """
    if not _REQUESTS:
        return {"available": False, "model": None, "host": OLLAMA_HOST, "error": "requests package not installed"}

    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if r.status_code == 200:
            data = r.json()
            models = [m["name"] for m in data.get("models", [])]
            # Check if our default model is available
            model_available = any(DEFAULT_MODEL in m for m in models)
            return {
                "available": True,
                "model": DEFAULT_MODEL if model_available else (models[0] if models else None),
                "models": models,
                "host": OLLAMA_HOST,
                "error": None,
            }
    except Exception as e:
        return {"available": False, "model": None, "host": OLLAMA_HOST, "error": str(e)}

    return {"available": False, "model": None, "host": OLLAMA_HOST, "error": "Unexpected response"}


# ── Core generation ───────────────────────────────────────────────────────────

def ask_llm(prompt: str, model: Optional[str] = None, stream: bool = False) -> Optional[str]:
    """
    Send a prompt to Ollama and return the response text.
    Returns None if Ollama is unavailable (should be used with fallback).
    """
    if not _REQUESTS:
        return None

    _model = model or DEFAULT_MODEL

    payload = {
        "model": _model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0.3,
            "num_predict": 800,
        },
    }

    try:
        if stream:
            return _stream_response(payload)

        r = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        return None

    except Exception:
        return None


def _stream_response(payload: dict) -> Iterator[str]:
    """Generator that yields text chunks from Ollama streaming API."""
    try:
        with requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            stream=True,
            timeout=TIMEOUT,
        ) as r:
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
    except Exception:
        return


# ── Clinical prompt builders ──────────────────────────────────────────────────

def build_clinical_reasoning_prompt(
    symptoms: list,
    vitals: list,
    medications: list,
    diagnoses: list,
    patient_info: dict,
) -> str:
    """
    Build a structured clinical prompt for LLM reasoning.
    The LLM is used ONLY for reasoning/summarization, not entity extraction.
    """
    lines = [
        "You are an expert clinical decision support assistant.",
        "Based on the extracted clinical data below, provide a concise assessment and plan.",
        "Be specific, clinical, and evidence-based. Limit to 3-4 paragraphs.",
        "",
        "=== PATIENT CONTEXT ===",
        f"Age/DOB: {patient_info.get('patient_dob', 'Unknown')}",
        f"Known Conditions: {patient_info.get('known_conditions', 'None documented')}",
        f"Allergies: {patient_info.get('allergies', 'NKDA')}",
        "",
        "=== ACTIVE SYMPTOMS ===",
    ]

    for s in symptoms:
        if not s.negated:
            parts = [s.name]
            if s.severity:
                parts.append(f"severity: {s.severity}")
            if s.duration:
                parts.append(f"duration: {s.duration}")
            if s.character:
                parts.append(f"character: {s.character}")
            lines.append(f"  - {' | '.join(parts)}")

    if vitals:
        lines += ["", "=== VITAL SIGNS ==="]
        for v in vitals:
            lines.append(f"  - {v.name}: {v.value} {v.unit}")

    if medications:
        lines += ["", "=== MEDICATIONS ==="]
        for m in medications:
            parts = [m.name]
            if m.dose:
                parts.append(m.dose)
            lines.append(f"  - {' '.join(parts)} ({m.status})")

    if diagnoses:
        lines += ["", "=== PRELIMINARY DIAGNOSES (ranked by probability) ==="]
        for i, dx in enumerate(diagnoses[:5], 1):
            conf = f" ({int(dx.confidence*100)}% confidence)" if dx.confidence else ""
            lines.append(f"  {i}. {dx.name} [{dx.icd10 or 'ICD pending'}]{conf}")

    lines += [
        "",
        "=== TASK ===",
        "1. Provide a brief clinical assessment (2-3 sentences).",
        "2. List the top 3 differential diagnoses with reasoning.",
        "3. Recommend immediate workup (labs, imaging, referrals).",
        "4. Suggest treatment plan components.",
        "",
        "Clinical Assessment:",
    ]

    return "\n".join(lines)


def build_soap_narrative_prompt(
    chief_complaint: str,
    symptoms: list,
    vitals: list,
    plan_items: list,
) -> str:
    """Build a prompt for generating SOAP narrative prose."""
    lines = [
        "You are a clinical documentation assistant. Write a formal SOAP note narrative.",
        "Be concise, professional, and use appropriate medical terminology.",
        "",
        f"Chief complaint: {chief_complaint}",
        "",
        "Active symptoms: " + ", ".join(s.name for s in symptoms if not s.negated),
        "",
        "Vitals: " + ", ".join(f"{v.name} {v.value} {v.unit}" for v in vitals),
        "",
        "Generate a 2-3 sentence Assessment and Plan paragraph:",
    ]
    return "\n".join(lines)
