"""
RAG retriever with KBDocument dataclass and lazy index loading.
"""
import os
import numpy as np
from dataclasses import dataclass
from models.embedder import EmbedderModel
from rag.indexer import load_or_build_index

KB_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")


@dataclass
class KBDocument:
    title: str
    content: str
    source: str
    score: float


class RAGRetriever:
    """Queries FAISS index, returns KBDocument list."""

    def __init__(self):
        self.index, self.docs = load_or_build_index(KB_DIR)
        self.embedder = EmbedderModel.get_instance()

    def retrieve(self, query: str, top_k: int = 3) -> list:
        if not query.strip() or not self.docs:
            return []
        q_emb = self.embedder.encode([query]).astype(np.float32)
        actual_k = min(top_k, len(self.docs))
        distances, indices = self.index.search(q_emb, actual_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.docs):
                d = self.docs[idx]
                results.append(KBDocument(
                    title=d["title"], content=d["content"],
                    source=d["source"], score=float(dist),
                ))
        return results
