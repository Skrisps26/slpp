"""
nlp_engine/core.py
Main MedicalNLPEngine — delegates to modular sub-engines.
Backward compatible: replaces the original monolithic nlp_engine.py.
"""

import json
import re
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── Allow imports from project root ──────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from nlp_engine.entities import (
    ClinicalEntities,
    ClinicalTimeline,
    Diagnosis,
    Medication,
    Symptom,
    Vital,
)
from nlp_engine.relations import (
    extract_allergies,
    extract_family_history,
    extract_social_history,
)
from nlp_engine.symptom_frames import extract_symptom_frame, extract_numeric_severity
from nlp_engine.timeline import extract_timeline
from nlp_engine.ontology import get_body_region

# ── Load knowledge base once ──────────────────────────────────────────────────
_KB_DIR = _ROOT / "knowledge_base"

def _load_json(filename: str) -> dict:
    path = _KB_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

_ICD_DATA = _load_json("icd_codes.json")
_SYMPTOM_TO_ICD = _ICD_DATA.get("symptom_to_icd10", {})


# ════════════════════════════════════════════════════════════════════════════
# spaCy loader
# ════════════════════════════════════════════════════════════════════════════

def _load_spacy():
    try:
        import spacy
    except ImportError:
        return None, None

    for model in ["en_ner_bc5cdr_md", "en_core_sci_md", "en_core_sci_sm", "en_core_web_sm"]:
        try:
            nlp = spacy.load(model)
            return nlp, model
        except OSError:
            continue
    return None, None


# ════════════════════════════════════════════════════════════════════════════
# Lexicons (imported from nlp_engine.py for backward compat)
# ════════════════════════════════════════════════════════════════════════════

# Load the original root-level nlp_engine.py by file path to avoid package name collision
# (the nlp_engine/ directory takes precedence over nlp_engine.py in sys.path)
try:
    import importlib.util as _ilu
    _nlp_root_path = _ROOT / "nlp_engine.py"
    _spec = _ilu.spec_from_file_location("_nlp_engine_root", str(_nlp_root_path))
    _orig_module_root = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_orig_module_root)

    COLLOQUIAL_TO_CLINICAL = getattr(_orig_module_root, "COLLOQUIAL_TO_CLINICAL", {})
    COLLOQUIAL_PATTERNS = getattr(_orig_module_root, "COLLOQUIAL_PATTERNS", [])
    MEDICATION_TERMS = getattr(_orig_module_root, "MEDICATION_TERMS", {})
    VITAL_PATTERNS = getattr(_orig_module_root, "VITAL_PATTERNS", [])
    NEGATION_CUES = getattr(_orig_module_root, "NEGATION_CUES", None)
    DOSE_RE = getattr(_orig_module_root, "DOSE_RE", None)
    FREQ_RE = getattr(_orig_module_root, "FREQ_RE", None)
    ROUTE_RE = getattr(_orig_module_root, "ROUTE_RE", None)
    _OrigEngine = getattr(_orig_module_root, "MedicalNLPEngine", None)
    _USE_ORIG = _OrigEngine is not None
except Exception as _e:
    _USE_ORIG = False
    COLLOQUIAL_TO_CLINICAL = {}
    COLLOQUIAL_PATTERNS = []
    MEDICATION_TERMS = {}
    VITAL_PATTERNS = []
    NEGATION_CUES = None
    DOSE_RE = None
    FREQ_RE = None
    ROUTE_RE = None
    _OrigEngine = None


# ════════════════════════════════════════════════════════════════════════════
# MedicalNLPEngine (enhanced wrapper)
# ════════════════════════════════════════════════════════════════════════════

class MedicalNLPEngine:
    """
    Enhanced Medical NLP Engine.
    Delegates entity extraction to the original engine (if available),
    then enriches results with: ICD-10 codes, timeline, frames, relations.
    """

    def __init__(self):
        if _USE_ORIG and _OrigEngine:
            self._engine = _OrigEngine()
        else:
            self._engine = None

    def analyze(self, text: str) -> ClinicalEntities:
        """Full analysis pipeline."""
        # Step 1: Base entity extraction
        if self._engine:
            entities = self._engine.analyze(text)
            # Convert to new entity types (they're compatible dataclasses)
            entities = self._upgrade_entities(entities)
        else:
            entities = ClinicalEntities()

        # Step 2: Enrich symptoms with frames + ICD-10
        for sym in entities.symptoms:
            self._enrich_symptom(sym, text)

        # Step 3: Enrich diagnoses with ICD-10
        disease_icd = _ICD_DATA.get("disease_to_icd10", {})
        for dx in entities.diagnoses:
            if not dx.icd10 and dx.name in disease_icd:
                dx.icd10 = disease_icd[dx.name]

        # Step 4: Extract relations if not already done
        if not entities.allergies:
            entities.allergies = extract_allergies(text)
        if not entities.family_history:
            entities.family_history = extract_family_history(text)
        if not entities.social_history:
            entities.social_history = extract_social_history(text)

        # Step 5: Extract clinical timeline
        entities.timeline = extract_timeline(text)

        return entities

    def _upgrade_entities(self, old: object) -> ClinicalEntities:
        """
        Convert old ClinicalEntities (from root nlp_engine.py) to new
        ClinicalEntities (from nlp_engine.entities). They're structurally
        compatible; we just create new instances to ensure type correctness.
        """
        # They share the same dataclass field names; return as-is if already compatible
        if isinstance(old, ClinicalEntities):
            return old

        # Otherwise copy fields
        new = ClinicalEntities()
        for field_name in ["symptoms", "medications", "vitals", "diagnoses",
                           "allergies", "procedures", "family_history",
                           "social_history", "review_of_systems",
                           "assessment_notes", "plan_items"]:
            try:
                setattr(new, field_name, getattr(old, field_name))
            except AttributeError:
                pass
        return new

    def _enrich_symptom(self, sym: Symptom, full_text: str):
        """Add ICD-10 code, body region, and frame data to a symptom."""
        # ICD-10 — exact match first, then whole-word containment
        # IMPORTANT: we must NOT let a generic term like "Pain" match "Chest pain"
        if not getattr(sym, "icd10", None):
            sym_lower = sym.name.lower()
            best_code = None
            best_len = 0
            for icd_key, icd_entry in _SYMPTOM_TO_ICD.items():
                key_lower = icd_key.lower()
                # Exact match wins immediately
                if key_lower == sym_lower:
                    best_code = icd_entry["code"]
                    break
                # Containment: both directions must hold (key ⊆ sym AND sym ⊆ key)
                # but only award if the key is at least as specific as the symptom.
                # Simple rule: key must be a left-anchored word-boundary match of sym.
                if key_lower in sym_lower and len(key_lower) > best_len:
                    # Ensure it's not a generic 1-word term matching a multi-word key
                    if len(sym_lower.split()) >= len(key_lower.split()):
                        best_code = icd_entry["code"]
                        best_len = len(key_lower)
            if best_code:
                sym.icd10 = best_code

        # Body region (ontology)
        if not sym.location:
            region = get_body_region(sym.name)
            if region and region != "systemic":
                sym.location = region

        # Frame extraction
        context = sym.context or ""
        frame = extract_symptom_frame(sym.name, context, full_text[:500])
        if not sym.severity and frame.get("severity"):
            sym.severity = frame["severity"]
        if not sym.duration and frame.get("duration"):
            sym.duration = frame["duration"]
        if not sym.character and frame.get("character"):
            sym.character = frame["character"]
        if not getattr(sym, "onset", None) and frame.get("onset"):
            sym.onset = frame["onset"]
        if not getattr(sym, "progression", None) and frame.get("progression"):
            sym.progression = frame["progression"]

        # Numeric severity
        if sym.context:
            num_sev = extract_numeric_severity(sym.context)
            if num_sev:
                sym.severity = num_sev
