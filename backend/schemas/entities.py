"""
Clinical entity dataclasses for GCIS extraction layer.
"""
from dataclasses import dataclass, field
import dataclasses


@dataclass
class ClinicalEntity:
    """A single extracted clinical entity."""
    text: str
    type: str           # SYMPTOM / MEDICATION / DIAGNOSIS / VITAL
    start: int
    end: int
    negated: bool = False
    uncertain: bool = False
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class DialogueAct:
    """A classified dialogue act for a sentence."""
    sentence: str
    sentence_index: int
    label: str
    confidence: float
    speaker: str = "unknown"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class TemporalEvent:
    """A temporal expression extracted from text."""
    text: str
    type: str           # DATE / TIME / DURATION / SET
    normalized: str     # ISO 8601
    start: int
    end: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class ClinicalEntities:
    """Container for all extracted clinical information from a transcript."""
    symptoms: list
    medications: list
    diagnoses: list
    vitals: list
    dialogue_acts: list
    temporal_events: list
    sentences: list
    negation_scopes: list

    def confirmed_symptoms(self):
        return [s for s in self.symptoms if not s.negated]

    def denied_symptoms(self):
        return [s for s in self.symptoms if s.negated]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
