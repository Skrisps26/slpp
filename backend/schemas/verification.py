"""
Verification result schemas for GCIS verification layer.
"""
from dataclasses import dataclass
import dataclasses


@dataclass
class SentenceVerification:
    """Verification result for a single SOAP sentence."""
    soap_sentence: str
    soap_section: str
    label: str          # ENTAILED / NEUTRAL / CONTRADICTED
    confidence: float
    source_transcript_sentence: str
    is_hallucinated: bool

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class VerificationResult:
    """Complete verification result for a SOAP note."""
    sentence_results: list
    faithfulness_score: float
    hallucinated_sentences: list

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
