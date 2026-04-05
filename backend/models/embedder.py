"""
Shared MiniLM embedder singleton.
Loaded once at startup, shared across RAG, dialogue act, and verifier.
"""
from sentence_transformers import SentenceTransformer
import numpy as np

_INSTANCE = None


class EmbedderModel:
    """Singleton that provides shared MiniLM embedding instance."""

    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    @classmethod
    def get_instance(cls) -> "EmbedderModel":
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = cls()
        return _INSTANCE

    def encode(self, texts: list) -> np.ndarray:
        """Encode texts to 384-dim unit vectors."""
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
