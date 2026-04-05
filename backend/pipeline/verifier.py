"""
Stage 3: Verification Layer.
NLI-based hallucination detection + sentence attribution.
Runs on CPU using DeBERTa-v3-small.
"""
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models.nli import NLIModel
from models.embedder import EmbedderModel
from schemas.verification import VerificationResult, SentenceVerification


class VerificationLayer:
    """Verifies SOAP note sentences against the source transcript using NLI."""

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

        # Split transcript into individual sentences for better NLI matching
        transcript_sentences = self._split_sentences(transcript)

        verified = []
        for soap_sent in soap_sentences:
            # NLI-score against the FULL transcript (catches contradictions across context)
            nli_full = self.nli.score(transcript, soap_sent["text"])
            
            # Find the best-matching transcript sentence
            source_sentence, similarity = self._find_source_with_score(soap_sent["text"], transcript_sentences)
            
            # NLI-score against the best-matching sentence (more precise)
            if source_sentence:
                nli_source = self.nli.score(source_sentence, soap_sent["text"])
            else:
                nli_source = nli_full

            # Use sentence-level if it's ENTAILED with high confidence,
            # otherwise use full transcript (safer — catches cross-sentence contradictions)
            if nli_source["label"] == "ENTAILED" and nli_source["confidence"] >= 0.75:
                nli_result = nli_source
            elif nli_source["label"] == "CONTRADICTED" and nli_source["confidence"] >= 0.5:
                nli_result = nli_source
            else:
                nli_result = nli_full

            sv = SentenceVerification(
                soap_sentence=soap_sent["text"],
                soap_section=soap_sent["section"],
                label=nli_result["label"],
                confidence=nli_result["confidence"],
                source_transcript_sentence=source_sentence or "",
                is_hallucinated=(nli_result["label"] == "CONTRADICTED"),
            )
            verified.append(sv)

        entailed = sum(1 for v in verified if v.label == "ENTAILED")
        faithfulness_score = entailed / len(verified) if verified else 0.0

        return VerificationResult(
            sentence_results=verified,
            faithfulness_score=faithfulness_score,
            hallucinated_sentences=[v for v in verified if v.is_hallucinated],
        )

    def _find_source_with_score(self, soap_sentence: str, transcript_sentences: list):
        """Find the most similar transcript sentence and its similarity score."""
        if not transcript_sentences:
            return "", 0.0
        soap_emb = self.embedder.encode([soap_sentence])
        transcript_embs = self.embedder.encode(transcript_sentences)
        sims = cosine_similarity(soap_emb, transcript_embs)[0]
        best_idx = np.argmax(sims)
        return transcript_sentences[best_idx], sims[best_idx]

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
