"""
Stage 3: Verification Layer.
NLI-based hallucination detection + sentence attribution.
Runs on CPU using DeBERTa-v3-small with auto-detected label order.
"""
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
        print(f"[VerificationLayer] NLI label map: {self.nli.label_map}")

        print("[VerificationLayer] Loading shared embedder...")
        self.embedder = EmbedderModel.get_instance()

        print("[VerificationLayer] Verification models ready.")

    def verify(self, transcript: str, soap) -> VerificationResult:
        """Verify every SOAP sentence against the source transcript."""
        soap_sentences = self._extract_sentences(soap)
        if not soap_sentences:
            return VerificationResult(
                sentence_results=[], faithfulness_score=1.0, hallucinated_sentences=[]
            )

        # Split transcript into individual sentences for attribution
        transcript_sentences = self._split_sentences(transcript)

        verified = []
        for soap_sent in soap_sentences:
            # NLI-score against the full transcript (primary — catches contradictions)
            nli_result = self.nli.score(transcript, soap_sent["text"])

            # Attribution: find the most similar transcript sentence via embedding
            source_sentence = self._find_source(soap_sent["text"], transcript_sentences)

            sv = SentenceVerification(
                soap_sentence=soap_sent["text"],
                soap_section=soap_sent["section"],
                label=nli_result["label"],
                confidence=nli_result["confidence"],
                source_transcript_sentence=source_sentence,
                is_hallucinated=(nli_result["label"] == "CONTRADICTED"),
            )
            verified.append(sv)

        # Faithfulness: fraction of non-contradicted sentences
        contradicted = sum(1 for v in verified if v.is_hallucinated)
        faithfulness_score = 1.0 - (contradicted / len(verified)) if verified else 1.0

        return VerificationResult(
            sentence_results=verified,
            faithfulness_score=round(faithfulness_score, 4),
            hallucinated_sentences=[v for v in verified if v.is_hallucinated],
        )

    def _find_source(self, soap_sentence: str, transcript_sentences: list) -> str:
        """Find the most similar transcript sentence via cosine similarity."""
        if not transcript_sentences:
            return ""
        soap_emb = self.embedder.encode([soap_sentence])
        transcript_embs = self.embedder.encode(transcript_sentences)
        sims = cosine_similarity(soap_emb, transcript_embs)[0]
        best_idx = int(np.argmax(sims))
        return transcript_sentences[best_idx]

    @staticmethod
    def _split_sentences(text: str) -> list:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _extract_sentences(self, soap) -> list:
        """Flatten SOAP note into (section, sentence) pairs."""
        results = []
        for section_name in ["subjective", "objective", "assessment", "plan"]:
            text = getattr(soap, section_name, "")
            for sent in re.split(r'[.!?]\s+', text):
                sent = sent.strip()
                if sent:
                    results.append({"text": sent, "section": section_name})
        return results
