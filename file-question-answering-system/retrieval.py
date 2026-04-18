# ── retrieval.py ──────────────────────────────────────────────────────────────
# Two-stage retrieval: FAISS + cross-encoder reranker

import numpy as np
import embeddings as emb


def retrieve_and_rerank(query: str, top_k: int = 10, final_k: int = 5) -> list:
    """
    Two-stage retrieval pipeline:
    1. FAISS retrieves top_k candidates by embedding similarity.
    2. Cross-encoder reranker scores each (query, chunk) pair together
       and returns the final_k most relevant chunks.

    Handles indirect, paraphrased, and twisted questions because the
    cross-encoder sees the full query+chunk text together — not just
    independent embeddings.
    """
    query_emb = emb.embedder.encode(
        [query], normalize_embeddings=True
    ).astype(np.float32)

    k = min(top_k, emb.index.ntotal)
    D, I = emb.index.search(query_emb, k)
    candidates = [emb.all_chunks[i] for i in I[0]]

    if len(candidates) <= 1:
        return candidates

    pairs  = [(query, c["code"]) for c in candidates]
    scores = emb.reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    return [c for _, c in ranked[:final_k]]
