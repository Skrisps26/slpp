"""
nlp_engine/entities.py
All clinical data-model dataclasses used throughout the NLP pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Symptom:
    name: str
    negated: bool = False
    severity: Optional[str] = None
    duration: Optional[str] = None
    frequency: Optional[str] = None
    location: Optional[str] = None
    character: Optional[str] = None
    context: str = ""
    source: str = "rule"  # "spacy" | "rule" | "colloquial"
    onset: Optional[str] = None
    progression: Optional[str] = None    # "worsening" | "improving" | "stable"
    icd10: Optional[str] = None


@dataclass
class Medication:
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    status: str = "mentioned"  # "current" | "prescribed" | "discontinued" | "mentioned"
    indication: Optional[str] = None


@dataclass
class Vital:
    name: str
    value: str
    unit: str
    raw: str
    status: str = "normal"   # "normal" | "warning" | "critical"


@dataclass
class Diagnosis:
    name: str
    icd10: Optional[str] = None
    certainty: str = "possible"  # "confirmed" | "possible" | "ruled-out"
    primary: bool = False
    confidence: float = 0.0
    matched_symptoms: List[str] = field(default_factory=list)


@dataclass
class ClinicalTimeline:
    """Ordered temporal events extracted from the transcript."""
    events: List[dict] = field(default_factory=list)
    # Each event: {"time": str, "event": str, "type": "onset"|"change"|"treatment"|"test"}


@dataclass
class ClinicalEntities:
    symptoms: List[Symptom] = field(default_factory=list)
    medications: List[Medication] = field(default_factory=list)
    vitals: List[Vital] = field(default_factory=list)
    diagnoses: List[Diagnosis] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    family_history: List[str] = field(default_factory=list)
    social_history: dict = field(default_factory=dict)
    review_of_systems: dict = field(default_factory=dict)
    assessment_notes: List[str] = field(default_factory=list)
    plan_items: List[str] = field(default_factory=list)
    timeline: Optional[ClinicalTimeline] = None
