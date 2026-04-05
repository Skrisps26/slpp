"""
GCISResponse schema — complete pipeline output.
"""
from dataclasses import dataclass
import dataclasses


@dataclass
class GCISResponse:
    """Complete pipeline response."""
    transcript: str
    patient_info: dict
    entities: dict
    soap: dict
    verification: dict
    refinement_iterations: int
    pipeline_version: str = "2.0.0"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
