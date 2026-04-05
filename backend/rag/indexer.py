"""
FAISS index builder for the clinical knowledge base.
Embeds documents from knowledge_base/ using MiniLM.
"""
import os
import json
import faiss
import numpy as np
from typing import List, Dict
from models.embedder import EmbedderModel


class RAGIndexer:
    """Builds and saves a FAISS index from knowledge base documents."""

    def __init__(self, knowledge_base_dir: str = "backend/knowledge_base",
                 index_dir: str = "models/faiss_index"):
        self.knowledge_base_dir = knowledge_base_dir
        self.index_dir = index_dir
        self.embedder = EmbedderModel.get_instance()
        self.documents: List[Dict] = []
        self.index = None

    def build(self) -> faiss.Index:
        """Load documents, embed them, and build the FAISS index."""
        self.documents = self._load_documents()
        if not self.documents:
            print("[RAGIndexer] No documents found in knowledge base.")
            self._build_empty_index()
            return self.index

        texts = [doc["content"] for doc in self.documents]
        self.embedder.load()
        embeddings = self.embedder.encode(texts)

        # Ensure 2D
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype("float32"))

        # Save index and metadata
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.index_dir, "faiss.index"))

        metadata = []
        for doc in self.documents:
            metadata.append({
                "title": doc.get("title", ""),
                "content_len": len(doc["content"]),
            })
        with open(os.path.join(self.index_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        print(f"[RAGIndexer] Built FAISS index with {len(self.documents)} documents.")
        return self.index

    def _build_empty_index(self):
        """Create an empty FAISS index."""
        dimension = 384  # MiniLM embedding dimension
        self.index = faiss.IndexFlatL2(dimension)
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.index_dir, "faiss.index"))

    def _load_documents(self) -> List[Dict]:
        """Load documents from knowledge_base directory."""
        documents = []
        if not os.path.exists(self.knowledge_base_dir):
            return documents

        for filename in os.listdir(self.knowledge_base_dir):
            filepath = os.path.join(self.knowledge_base_dir, filename)
            if not os.path.isfile(filepath):
                continue

            if filename.endswith(".txt") or filename.endswith(".md"):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append({
                    "title": os.path.splitext(filename)[0],
                    "content": content,
                    "source": filepath,
                })
            elif filename.endswith(".json"):
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        documents.append({
                            "title": item.get("title", filename),
                            "content": item.get("content", ""),
                            "source": filepath,
                        })
                elif isinstance(data, dict):
                    documents.append({
                        "title": data.get("title", filename),
                        "content": data.get("content", ""),
                        "source": filepath,
                    })

        return documents

    def get_documents(self) -> List[Dict]:
        """Return the loaded documents (call after build())."""
        return self.documents
