"""
Stage 1: Extraction Layer.
Runs all 4 extraction models (NER, negation, dialogue acts, temporal).
All models run on CPU during inference.
"""
import re
from typing import List

from models.clinical_ner import ClinicalNERModel
from models.dialogue_act import DialogueActModel
from models.temporal import TemporalExtractor
from models.embedder import EmbedderModel
from schemas.entities import ClinicalEntities, Entity, DialogueAct, TemporalEvent


class ExtractionLayer:
    """Runs all extraction models and returns structured ClinicalEntities."""

    def __init__(self):
        self.ner: ClinicalNERModel = None
        self.dialogue: DialogueActModel = None
        self.temporal: TemporalExtractor = None
        self.embedder: EmbedderModel = None

    def load(self):
        """Load all extraction models."""
        print("[ExtractionLayer] Loading NER and negation model...")
        self.ner = ClinicalNERModel.load()

        print("[ExtractionLayer] Loading dialogue act classifier...")
        self.dialogue = DialogueActModel.load()

        print("[ExtractionLayer] Loading temporal extractor...")
        self.temporal = TemporalExtractor()

        print("[ExtractionLayer] Loading embedder...")
        self.embedder = EmbedderModel.get_instance()
        try:
            self.embedder.load()
        except Exception as e:
            print(f"[ExtractionLayer] Embedder load warning: {e}")

        print("[ExtractionLayer] All extraction models loaded.")

    def extract(self, transcript: str) -> ClinicalEntities:
        """Run full extraction pipeline on a transcript."""
        sentences = self._split_sentences(transcript)

        # Run NER + negation
        raw_entities = self.ner.extract_entities(transcript)
        negation_scopes = self.ner.detect_negation(transcript)

        # Apply negation to entities
        entities = self._apply_negation(raw_entities, negation_scopes)

        # Classify dialogue acts per sentence
        dialogue_acts = []
        for s in sentences:
            try:
                result = self.dialogue.classify(s)
                dialogue_acts.append(DialogueAct(
                    text=s,
                    label=result["label"],
                    confidence=result["confidence"],
                ))
            except Exception:
                dialogue_acts.append(DialogueAct(text=s, label="OTHER", confidence=0.0))

        # Extract temporal expressions
        temporal_events = []
        try:
            temporal_raw = self.temporal.extract(transcript)
            temporal_events = [
                TemporalEvent(
                    text=e.get("text", ""),
                    temporal_type=e.get("type", "DATE"),
                    normalized=e.get("normalized", ""),
                    start=e.get("start", 0),
                    end=e.get("end", 0),
                )
                for e in temporal_raw
            ]
        except Exception:
            pass

        return ClinicalEntities(
            symptoms=[e for e in entities if e.entity_type == "SYMPTOM"],
            medications=[e for e in entities if e.entity_type == "MEDICATION"],
            diagnoses=[e for e in entities if e.entity_type == "DIAGNOSIS"],
            vitals=[e for e in entities if e.entity_type == "VITAL"],
            negation_scopes=negation_scopes,
            dialogue_acts=dialogue_acts,
            temporal_events=temporal_events,
            sentences=sentences,
        )

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences using regex (no spacy dependency)."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _apply_negation(entities, negation_scopes) -> List:
        """Mark entities as negated if they fall within a negation scope."""
        for entity in entities:
            for scope in negation_scopes:
                if scope.get("start", 0) <= entity.start < scope.get("end", 0):
                    entity.negated = True
                    break
        return entities
