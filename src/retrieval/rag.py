"""Full RAG pipeline over SEC filings: retrieve, re-rank, generate, cite.

Pipeline stages:
  1. Hybrid retrieval: embed the question (nomic-embed-text via Ollama) and
     dense-search Qdrant, run a BM25 keyword search over the same chunks, and
     merge both shortlists with Reciprocal Rank Fusion. An optional ticker
     filter restricts both searches to one company. (RETRIEVAL_MODE=dense
     keeps dense search only; see eval/retrieval_results.md for why hybrid is
     the default.)
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

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import CrossEncoder

if TYPE_CHECKING:
    from strategies import BM25Index

log = logging.getLogger(__name__)

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "sec_filings"

RETRIEVE_N = 20
FINAL_K = 5
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "hybrid")  # "hybrid" or "dense"
CHUNKS_PATH = Path("data/processed/chunks.json")
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


class Retrieved(Protocol):
    """A retrieved chunk: a Qdrant point or a BM25 hit, both exposing `.payload`."""

    payload: dict[str, Any]


_reranker: CrossEncoder | None = None
_bm25: BM25Index | bool | None = None  # False = chunks file missing, do not retry


def embed(text: str) -> list[float]:
    """Embed one text with nomic-embed-text via the local Ollama server."""
    response = httpx.post(
        OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60.0
    )
    response.raise_for_status()
    return response.json()["embedding"]


def get_reranker() -> CrossEncoder:
    """Load the cross-encoder once and reuse it."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


class _Chunk:
    """Gives BM25 results the same `.payload` shape as Qdrant points."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload


def get_bm25() -> BM25Index | None:
    """BM25 index over data/processed/chunks.json, built once; None if the file is missing."""
    global _bm25
    if _bm25 is None:
        from strategies import BM25Index, load_chunks

        if CHUNKS_PATH.exists():
            _bm25 = BM25Index(load_chunks(CHUNKS_PATH))
        else:
            log.warning("%s not found: using dense retrieval only", CHUNKS_PATH)
            _bm25 = False
    return _bm25 or None


def retrieve(
    question: str, ticker: str | None = None, mode: str | None = None
) -> list[Retrieved]:
    """Hybrid (or dense) search, then cross-encoder re-rank; return the top FINAL_K chunks.

    If ticker is given, both searches are restricted to that company's chunks -
    the same mechanism that keeps a high-volume filer (JPM) from crowding out
    results for a company-specific question.
    """
    mode = mode or RETRIEVAL_MODE
    client = QdrantClient(url=QDRANT_URL)

    query_filter = None
    if ticker:
        query_filter = Filter(must=[FieldCondition(key="ticker", match=MatchValue(value=ticker))])

    hits = client.query_points(
        collection_name=COLLECTION,
        query=embed(question),
        query_filter=query_filter,
        limit=RETRIEVE_N,
        with_payload=True,
    ).points

    candidates: list[Retrieved] = list(hits)
    bm25 = get_bm25() if mode == "hybrid" else None
    if bm25 is not None:
        from strategies import rrf_fuse

        keyword = bm25.search(question, RETRIEVE_N, ticker=ticker)
        fused = rrf_fuse([[h.payload for h in hits], keyword], RETRIEVE_N)
        candidates = [_Chunk(payload) for payload in fused]

    if not candidates:
        return []

    pairs = [(question, c.payload["text"]) for c in candidates]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(scores, candidates, strict=True), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:FINAL_K]]


def build_context(chunks: Sequence[Retrieved]) -> str:
    """Number each excerpt and prefix it with its source, for the prompt."""
    blocks = []
    for i, h in enumerate(chunks, 1):
        p = h.payload
        src = f"{p['ticker']} {p['form']} filed {p['filing_date']} - {p['section']}"
        blocks.append(f"[{i}] ({src})\n{p['text']}")
    return "\n\n".join(blocks)


def generate(question: str, context: str) -> str:
    """Ask the LLM to answer from the context only (OpenAI-compatible chat API)."""
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


def format_sources(chunks: Sequence[Retrieved]) -> str:
    """Source list numbered like the context, so [n] in the answer can be traced."""
    lines = []
    for i, h in enumerate(chunks, 1):
        p = h.payload
        lines.append(f"[{i}] {p['ticker']} {p['form']} filed {p['filing_date']} - {p['section']}")
    return "\n".join(lines)


def answer_question(question: str, ticker: str | None = None) -> tuple[str, list[Retrieved]]:
    """Run the full pipeline and return (answer_text, source_chunks)."""
    chunks = retrieve(question, ticker=ticker)
    if not chunks:
        return "No relevant excerpts were found for that question.", []
    context = build_context(chunks)
    answer = generate(question, context)
    return answer, chunks


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    log.info("question: %s", question)
    answer, chunks = answer_question(question)
    # The answer is the program's output, so it goes to stdout, not the log.
    print("=" * 70)  # noqa: T201
    print(f"ANSWER\n\n{answer}\n")  # noqa: T201
    print("=" * 70)  # noqa: T201
    print(f"SOURCES\n\n{format_sources(chunks)}")  # noqa: T201


if __name__ == "__main__":
    main()
