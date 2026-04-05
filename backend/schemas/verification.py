"""
Verification result schemas for GCIS verification layer.
"""
from dataclasses import dataclass, field


@dataclass
class SentenceVerification:
    """Verification result for a single SOAP sentence."""
    soap_sentence: str
    soap_section: str
    label: str  # ENTAILED, NEUTRAL, CONTRADICTED
    confidence: float
    source_transcript_sentence: str
    is_hallucinated: bool = False

    def to_dict(self) -> dict:
        return {
            "soap_sentence": self.soap_sentence,
            "soap_section": self.soap_section,
            "label": self.label,
            "confidence": self.confidence,
            "source_transcript_sentence": self.source_transcript_sentence,
            "is_hallucinated": self.is_hallucinated,
        }


@dataclass
class AttributionMap:
    """Maps a SOAP sentence to its source in the transcript."""
    soap_sentence: str
    source_sentence: str
    similarity_score: float

    def to_dict(self) -> dict:
        return {**self.__dict__}


@dataclass
class VerificationResult:
    """Complete verification result for a SOAP note."""
    sentence_results: list = None
    faithfulness_score: float = 0.0
    hallucinated_sentences: list = None

    def __post_init__(self):
        if self.sentence_results is None:
            self.sentence_results = []
        if self.hallucinated_sentences is None:
            self.hallucinated_sentences = []

    def to_dict(self) -> dict:
        return {
            "sentence_results": [s.to_dict() for s in self.sentence_results],
            "faithfulness_score": round(self.faithfulness_score, 4),
            "hallucinated_sentences": [s.to_dict() for s in self.hallucinated_sentences],
        }
