"""Compare dense retrieval alone vs. dense retrieval + cross-encoder re-ranking.

Dense (bi-encoder) search embeds the question and each chunk independently, so it
is fast but coarse. A cross-encoder instead reads the (question, chunk) pair
together and scores their relevance directly, which is far more accurate but too
slow to run over the whole collection. The standard pattern is therefore two
stage: retrieve a wider top-N with the fast dense search, then re-rank that
shortlist with the cross-encoder and keep the best few.

This script prints both rankings side by side so the effect of re-ranking is
visible.

Usage:
    python src/retrieval/test_rerank.py "your question here"
"""
import sys

import httpx
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "sec_filings"

RETRIEVE_N = 20  # wide shortlist from dense search
FINAL_K = 5      # keep this many after re-ranking
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_QUESTION = "How does NVIDIA describe the risk from U.S. export controls to China?"


def embed(text):
    response = httpx.post(
        OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60.0
    )
    response.raise_for_status()
    return response.json()["embedding"]


def source_label(payload):
    return f"{payload['ticker']} {payload['form']} {payload['filing_date']} / {payload['section']}"


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    print(f"Question: {question}\n")

    client = QdrantClient(url=QDRANT_URL)
    hits = client.query_points(
        collection_name=COLLECTION,
        query=embed(question),
        limit=RETRIEVE_N,
        with_payload=True,
    ).points

    print("--- Dense retrieval (top 5 of 20) ---")
    for rank, hit in enumerate(hits[:FINAL_K], 1):
        print(f"[{rank}] dense={hit.score:.4f}  |  {source_label(hit.payload)}")

    # Re-rank the 20 candidates with the cross-encoder.
    print(f"\nLoading cross-encoder ({RERANK_MODEL})...")
    reranker = CrossEncoder(RERANK_MODEL)
    pairs = [(question, hit.payload["text"]) for hit in hits]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)

    print("\n--- After cross-encoder re-ranking (top 5 of 20) ---")
    for rank, (score, hit) in enumerate(reranked[:FINAL_K], 1):
        snippet = hit.payload["text"][:200].replace("\n", " ")
        print(f"[{rank}] rerank={score:.4f}  |  {source_label(hit.payload)}")
        print(f"    {snippet}...")


if __name__ == "__main__":
    main()
