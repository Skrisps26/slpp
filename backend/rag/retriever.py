"""
Query FAISS index to retrieve relevant knowledge base documents.
"""
import os
import json
import faiss
import numpy as np
from typing import List
from models.embedder import EmbedderModel


class RAGRetriever:
    """Queries the FAISS index and returns top-k relevant docs."""

    def __init__(self, knowledge_base_dir: str = "backend/knowledge_base",
                 index_dir: str = "models/faiss_index"):
        self.knowledge_base_dir = knowledge_base_dir
        self.index_dir = index_dir
        self.index = None
        self.documents = []
        self.metadata = []
        self.embedder: EmbedderModel = EmbedderModel.get_instance()

    def _load_documents(self) -> list:
        """Load documents from knowledge base directory."""
        documents = []
        if not os.path.exists(self.knowledge_base_dir):
            return documents
        import glob
        patterns = [
            os.path.join(self.knowledge_base_dir, "*.txt"),
            os.path.join(self.knowledge_base_dir, "*.md"),
        ]
        for pattern in patterns:
            for filepath in sorted(glob.glob(pattern)):
                if os.path.isfile(filepath):
                    with open(filepath, encoding="utf-8") as f:
                        content = f.read()
                    title = os.path.splitext(os.path.basename(filepath))[0]
                    documents.append({
                        "title": title,
                        "content": content,
                    })
        return documents

    def _ensure_loaded(self):
        """Lazy-load the FAISS index and documents."""
        if self.index is not None:
            return

        index_path = os.path.join(self.index_dir, "faiss.index")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"[RAGRetriever] Loaded FAISS index from {index_path}")

        metadata_path = os.path.join(self.index_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)

        # Load documents from knowledge base
        self.documents = self._load_documents()
        print(f"[RAGRetriever] Loaded {len(self.documents)} documents")

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Retrieve top-k relevant docs for a query."""
        self._ensure_loaded()

        # If no index or docs, return empty
        if self.index is None or not self.documents:
            print(f"[RAGRetriever] No index or documents available, using all docs as fallback")
            if self.documents:
                return [{"title": d["title"], "content": d["content"][:300]}
                        for d in self.documents[:top_k]]
            return []

        if not query or query.strip() == "":
            return [{"title": d.get("title", ""), "content": d.get("content", "")[:300]}
                    for d in self.documents[:top_k]]

        self.embedder.load()
        query_embedding = self.embedder.encode([query])
        query_embedding = query_embedding.astype("float32")

        n_docs = len(self.documents)
        distances, indices = self.index.search(query_embedding, min(top_k, n_docs))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            score = 1.0 / (1.0 + dist)
            results.append({
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
                "score": round(score, 4),
            })

        return results
