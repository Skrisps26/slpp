"""
Shared MiniLM embedder for both RAG and dialogue act classification.
Uses sentence-transformers/all-MiniLM-L6-v2 (80MB, CPU).
"""
from sentence_transformers import SentenceTransformer


class EmbedderModel:
    """Singleton that provides shared MiniLM embedding instance."""

    _instance: "EmbedderModel" = None

    def __init__(self):
        self.model: SentenceTransformer = None

    @classmethod
    def get_instance(cls) -> "EmbedderModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Load the MiniLM model. Shared across dialogue acts and RAG."""
        if self.model is None:
            self.model = SentenceTransformer(model_name)
        return self.model

    def encode(self, texts: list, **kwargs) -> "np.ndarray":
        """Embed a list of texts into dense vectors."""
        import numpy as np
        if self.model is None:
            self.load()
        return self.model.encode(texts, **kwargs)
