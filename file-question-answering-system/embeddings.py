# ── embeddings.py ─────────────────────────────────────────────────────────────
# BAAI embedder + cross-encoder reranker + FAISS index builder

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import EMBEDDER_NAME, RERANKER_NAME

embedder  = None
reranker  = None
index     = None
all_chunks = []


def load_embedder_and_reranker():
    """Load BAAI embedder and cross-encoder reranker."""
    global embedder, reranker

    print("Loading BAAI/bge-base-en-v1.5 embedder...")
    embedder = SentenceTransformer(EMBEDDER_NAME)

    print("Loading cross-encoder reranker...")
    reranker = CrossEncoder(RERANKER_NAME)

    print("Embedder + Reranker ready!")


def build_index(chunks: list):
    """Encode chunks and build FAISS IndexFlatIP (cosine similarity)."""
    global index, all_chunks
    all_chunks = chunks

    code_texts = [
        f"Type: {c['type']}\nName: {c['name']}\n\n{c['code']}"
        for c in chunks
    ]

    print("Encoding chunks...")
    embeddings = embedder.encode(
        code_texts,
        show_progress_bar=True,
        batch_size=16,
        normalize_embeddings=True
    )
    print(f"Embeddings shape: {embeddings.shape}")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings, dtype=np.float32))

    print(f"FAISS index ready with {index.ntotal} chunks")
