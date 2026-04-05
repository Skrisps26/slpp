"""
Clinical entity dataclasses for GCIS extraction layer.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Entity:
    """A single extracted clinical entity."""
    text: str
    entity_type: str  # SYMPTOM, MEDICATION, DIAGNOSIS, VITAL
    start: int
    end: int
    confidence: float = 1.0
    negated: bool = False

    def to_dict(self) -> dict:
        return {**asdict(self)}


@dataclass
class TemporalEvent:
    """A temporal expression extracted from text."""
    text: str
    temporal_type: str  # DATE, DURATION, FREQUENCY, etc.
    normalized: str  # ISO 8601 / TIMEX3 format
    start: int
    end: int

    def to_dict(self) -> dict:
        return {**asdict(self)}


@dataclass
class DialogueAct:
    """A classified dialogue act for a sentence."""
    text: str
    label: str  # SYMPTOM_REPORT, QUESTION, DIAGNOSIS_STATEMENT, etc.
    confidence: float

    def to_dict(self) -> dict:
        return {**asdict(self)}


@dataclass
class ClinicalEntities:
    """Container for all extracted clinical information from a transcript."""
    symptoms: List[Entity] = field(default_factory=list)
    medications: List[Entity] = field(default_factory=list)
    diagnoses: List[Entity] = field(default_factory=list)
    vitals: List[Entity] = field(default_factory=list)
    negation_scopes: List[dict] = field(default_factory=list)
    dialogue_acts: List[DialogueAct] = field(default_factory=list)
    temporal_events: List[TemporalEvent] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symptoms": [e.to_dict() for e in self.symptoms],
            "medications": [e.to_dict() for e in self.medications],
            "diagnoses": [e.to_dict() for e in self.diagnoses],
            "vitals": [e.to_dict() for e in self.vitals],
            "negation_scopes": self.negation_scopes,
            "dialogue_acts": [d.to_dict() for d in self.dialogue_acts],
            "temporal_events": [t.to_dict() for t in self.temporal_events],
            "sentences": self.sentences,
        }
