"""
RAG indexer — builds FAISS index from knowledge_base/ or loads existing.
"""
import faiss
import numpy as np
import json
import os
from pathlib import Path
from models.embedder import EmbedderModel

INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "faiss_index", "index.faiss")
DOCS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", "faiss_index", "docs.json")


def build_index(knowledge_base_dir: str):
    embedder = EmbedderModel.get_instance()
    docs = []
    for path in Path(knowledge_base_dir).rglob("*"):
        if path.is_file() and path.suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            words = text.split()
            for i in range(0, max(len(words), 300), 250):
                chunk = " ".join(words[i:i + 300])
                if len(chunk.strip()) > 20:
                    docs.append({"title": path.stem, "content": chunk, "source": str(path)})

    if not docs:
        print(f"[RAG] WARNING: No documents in {knowledge_base_dir}. RAG disabled.")
        return None, []

    embeddings = embedder.encode([d["content"] for d in docs]).astype(np.float32)
    index = faiss.IndexFlatL2(384)
    index.add(embeddings)
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "w") as f:
        json.dump(docs, f)
    print(f"[RAG] Index built with {len(docs)} chunks.")
    return index, docs


def load_or_build_index(knowledge_base_dir: str):
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH) as f:
            docs = json.load(f)
        print(f"[RAG] Loaded index with {len(docs)} chunks.")
        return index, docs
    return build_index(knowledge_base_dir) or (None, [])
