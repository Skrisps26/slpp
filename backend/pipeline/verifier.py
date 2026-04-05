"""
Stage 3: Verification Layer.
Sentence attribution + faithfulness scoring.
Uses embedding cosine similarity as primary metric
with NLI as secondary contradiction detection.
Runs on CPU using DeBERTa-v3-small.
"""
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models.nli import NLIModel
from models.embedder import EmbedderModel
from schemas.verification import VerificationResult, SentenceVerification

# Cosine similarity thresholds for sentence-level verification
ENTAILMENT_THRESHOLD = 0.65   # Similarity >= 0.65 → ENTAILED
CONTRADICTION_BELOW = 0.35    # Similarity < 0.35 AND NLI contradicts → CONTRADICTED


class VerificationLayer:
    """Verifies SOAP note sentences against the source transcript."""

    def __init__(self):
        self.nli: NLIModel = None
        self.embedder: EmbedderModel = None

    def load(self):
        """Load the NLI model and shared embedder."""
        print("[VerificationLayer] Loading NLI model...")
        self.nli = NLIModel()
        self.nli.load()

        print("[VerificationLayer] Loading shared embedder...")
        self.embedder = EmbedderModel.get_instance()
        self.embedder.load()

        print("[VerificationLayer] Verification models ready.")

    def verify(self, transcript: str, soap: "SOAPNote") -> VerificationResult:
        """Verify every SOAP sentence against the source transcript."""
        soap_sentences = self._extract_sentences(soap)
        if not soap_sentences:
            return VerificationResult(sentence_results=[], faithfulness_score=1.0)

        transcript_sentences = self._split_sentences(transcript)

        verified = []
        for soap_sent in soap_sentences:
            source_sentence, similarity = self._find_source_with_score(
                soap_sent["text"], transcript_sentences
            )

            # Primary: use cosine similarity for paraphrase detection
            if similarity >= ENTAILMENT_THRESHOLD:
                label = "ENTAILED"
                confidence = round(similarity, 4)
            elif similarity < CONTRADICTION_BELOW and source_sentence:
                # Only flag as contradicted if NLI confirms it
                nli_result = self.nli.score(source_sentence, soap_sent["text"])
                if nli_result["label"] == "CONTRADICTED":
                    label = "CONTRADICTED"
                    confidence = nli_result["confidence"]
                else:
                    label = "NEUTRAL"
                    confidence = round(similarity, 4)
            elif source_sentence:
                # Low-ish similarity but not below threshold — could be summary
                nli_result = self.nli.score(source_sentence, soap_sent["text"])
                if nli_result["label"] == "CONTRADICTED":
                    label = "CONTRADICTED"
                    confidence = nli_result["confidence"]
                else:
                    label = "NEUTRAL"
                    confidence = round(similarity, 4)
            else:
                # No matching sentence found at all
                label = "CONTRADICTED"
                confidence = 0.0

            sv = SentenceVerification(
                soap_sentence=soap_sent["text"],
                soap_section=soap_sent["section"],
                label=label,
                confidence=confidence,
                source_transcript_sentence=source_sentence or "",
                similarity=round(similarity, 4) if source_sentence else 0.0,
                is_hallucinated=(label == "CONTRADICTED"),
            )
            verified.append(sv)

        # Faithfulness: only contradictions reduce the score
        contradiction_count = sum(1 for v in verified if v.label == "CONTRADICTED")
        faithfulness_score = 1.0 - (contradiction_count / len(verified)) if verified else 1.0

        return VerificationResult(
            sentence_results=verified,
            faithfulness_score=round(faithfulness_score, 4),
            hallucinated_sentences=[v for v in verified if v.is_hallucinated],
        )

    def _find_source_with_score(self, soap_sentence: str, transcript_sentences: list):
        """Find the most similar transcript sentence and its similarity score."""
        if not transcript_sentences:
            return None, 0.0
        soap_emb = self.embedder.encode([soap_sentence])
        transcript_embs = self.embedder.encode(transcript_sentences)
        sims = cosine_similarity(soap_emb, transcript_embs)[0]
        best_idx = int(np.argmax(sims))
        return transcript_sentences[best_idx], float(sims[best_idx])

    @staticmethod
    def _split_sentences(text: str) -> list:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _extract_sentences(soap: "SOAPNote") -> list:
        """Flatten SOAP note into (section, sentence) pairs."""
        results = []
        for section_name in ["subjective", "objective", "assessment", "plan"]:
            text = getattr(soap, section_name, "")
            for sent in re.split(r'[.!?]\s+', text):
                sent = sent.strip()
                if sent:
                    results.append({"text": sent, "section": section_name})
        return results
