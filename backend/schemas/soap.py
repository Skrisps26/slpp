"""
SOAP Note dataclasses and schemas for GCIS generation layer.
"""
from dataclasses import dataclass
import dataclasses


@dataclass
class Differential:
    """A differential diagnosis with evidence and KB source."""
    diagnosis: str
    evidence: str
    likelihood: str     # "high" / "moderate" / "low"
    kb_source: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class SOAPNote:
    """A structured SOAP note generated from clinical entities."""
    subjective: str
    objective: str
    assessment: str
    plan: str
    differentials: list

    @classmethod
    def from_dict(cls, data: dict, entities, retrieved_docs):
        """Create a SOAPNote from parsed JSON response."""
        diffs = [
            Differential(
                diagnosis=d.get("diagnosis", ""),
                evidence=d.get("evidence", ""),
                likelihood=d.get("likelihood", "moderate"),
                kb_source=retrieved_docs[i].title if i < len(retrieved_docs) else ""
            )
            for i, d in enumerate(data.get("differentials", []))
        ]
        return cls(
            subjective=data.get("subjective", ""),
            objective=data.get("objective", ""),
            assessment=data.get("assessment", ""),
            plan=data.get("plan", ""),
            differentials=diffs,
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
