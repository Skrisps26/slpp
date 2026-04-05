"""
SOAP Note dataclasses and schemas for GCIS generation layer.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from .entities import ClinicalEntities


@dataclass
class DifferentialDiagnosis:
    """A differential diagnosis with evidence and likelihood."""
    diagnosis: str
    evidence: str
    likelihood: str  # "High", "Moderate", "Low"

    def to_dict(self) -> dict:
        return {**asdict(self)}


@dataclass
class SOAPNote:
    """A structured SOAP note generated from clinical entities."""
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""
    differentials: List[DifferentialDiagnosis] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict, entities: ClinicalEntities = None,
                  retrieved_docs: list = None) -> "SOAPNote":
        """Create a SOAPNote from parsed JSON response."""
        differentials = []
        for d in data.get("differentials", []):
            differentials.append(DifferentialDiagnosis(
                diagnosis=d.get("diagnosis", ""),
                evidence=d.get("evidence", ""),
                likelihood=d.get("likelihood", "Unknown"),
            ))
        return cls(
            subjective=data.get("subjective", ""),
            objective=data.get("objective", ""),
            assessment=data.get("assessment", ""),
            plan=data.get("plan", ""),
            differentials=differentials,
        )

    def to_dict(self) -> dict:
        return {
            "subjective": self.subjective,
            "objective": self.objective,
            "assessment": self.assessment,
            "plan": self.plan,
            "differentials": [d.to_dict() for d in self.differentials],
        }
