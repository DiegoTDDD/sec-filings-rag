"""Full RAG pipeline over SEC filings: retrieve, re-rank, generate, cite.

Pipeline stages:
  1. Embed the question (nomic-embed-text via Ollama) and dense-search Qdrant
     for a wide shortlist of candidate chunks. An optional ticker filter uses
     Qdrant's native metadata filtering to restrict search to one company.
  2. Re-rank that shortlist with a cross-encoder and keep the top few.
  3. Build a grounded prompt from those chunks and generate an answer with
     llama3.2 (via Ollama), instructed to answer only from the provided context.
  4. Return the prose answer followed by the list of sources actually used.

The grounding instruction (answer only from context; say when unknown) is what
keeps the system faithful to the filings rather than hallucinating - the property
RAGAS measures. The LLM is configurable via env vars so the same code runs on a
local Ollama model by default or a hosted API when LLM_BASE_URL / LLM_MODEL are set.

Usage:
    python src/retrieval/rag.py "your question here"
"""
import os
import sys

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import CrossEncoder

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "sec_filings"

RETRIEVE_N = 20
FINAL_K = 5
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2:3b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")

DEFAULT_QUESTION = "How does NVIDIA describe the risk from U.S. export controls to China?"

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer the question using ONLY the "
    "provided excerpts from SEC filings. If the excerpts do not contain the "
    "answer, say so plainly. Do not invent facts, figures, or filings. Write a "
    "clear, concise answer in prose."
)

_reranker = None


def embed(text):
    response = httpx.post(
        OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60.0
    )
    response.raise_for_status()
    return response.json()["embedding"]


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def retrieve(question, ticker=None):
    """Dense-search then cross-encoder re-rank; return the top FINAL_K chunks.

    If ticker is given, Qdrant restricts the search to that company's chunks via
    a metadata filter - the same mechanism that keeps a high-volume filer (JPM)
    from crowding out results for a company-specific question.
    """
    client = QdrantClient(url=QDRANT_URL)

    query_filter = None
    if ticker:
        query_filter = Filter(
            must=[FieldCondition(key="ticker", match=MatchValue(value=ticker))]
        )

    hits = client.query_points(
        collection_name=COLLECTION,
        query=embed(question),
        query_filter=query_filter,
        limit=RETRIEVE_N,
        with_payload=True,
    ).points

    if not hits:
        return []

    pairs = [(question, h.payload["text"]) for h in hits]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
    return [h for _, h in ranked[:FINAL_K]]


def build_context(chunks):
    blocks = []
    for i, h in enumerate(chunks, 1):
        p = h.payload
        src = f"{p['ticker']} {p['form']} filed {p['filing_date']} - {p['section']}"
        blocks.append(f"[{i}] ({src})\n{p['text']}")
    return "\n\n".join(blocks)


def generate(question, context):
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nExcerpts:\n{context}"},
        ],
        "temperature": 0.1,
    }
    response = httpx.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def format_sources(chunks):
    lines = []
    for i, h in enumerate(chunks, 1):
        p = h.payload
        lines.append(
            f"[{i}] {p['ticker']} {p['form']} filed {p['filing_date']} - {p['section']}"
        )
    return "\n".join(lines)


def answer_question(question, ticker=None):
    """Run the full pipeline and return (answer_text, source_chunks)."""
    chunks = retrieve(question, ticker=ticker)
    if not chunks:
        return "No relevant excerpts were found for that question.", []
    context = build_context(chunks)
    answer = generate(question, context)
    return answer, chunks


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    print(f"Question: {question}\n")
    print("Retrieving and generating...\n")

    answer, chunks = answer_question(question)

    print("=" * 70)
    print("ANSWER\n")
    print(answer)
    print("\n" + "=" * 70)
    print("SOURCES\n")
    print(format_sources(chunks))


if __name__ == "__main__":
    main()
