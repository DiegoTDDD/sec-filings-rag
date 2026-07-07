"""Quick retrieval test: embed a question, search Qdrant, print the top chunks.

This validates the core of the RAG (dense retrieval over the filings) before we
add re-ranking and the LLM. Each result is shown with its provenance so we can
see that citations will be traceable back to a specific filing and section.

Usage:
    python src/retrieval/test_search.py "your question here"
If no question is passed, a default NVIDIA export-control query is used.
"""
import sys

import httpx
from qdrant_client import QdrantClient

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "sec_filings"
TOP_K = 5

DEFAULT_QUESTION = "How does NVIDIA describe the risk from U.S. export controls to China?"


def embed(text):
    """Embed a piece of text with the same model used for indexing."""
    response = httpx.post(
        OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60.0
    )
    response.raise_for_status()
    return response.json()["embedding"]


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    print(f"Question: {question}\n")

    client = QdrantClient(url=QDRANT_URL)
    query_vector = embed(question)

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
    ).points

    print(f"Top {TOP_K} retrieved chunks:\n" + "=" * 70)
    for rank, point in enumerate(results, 1):
        p = point.payload
        source = f"{p['ticker']} {p['form']} {p['filing_date']} / {p['section']}"
        snippet = p["text"][:300].replace("\n", " ")
        print(f"\n[{rank}] score={point.score:.4f}  |  {source}")
        print(f"    {snippet}...")


if __name__ == "__main__":
    main()
