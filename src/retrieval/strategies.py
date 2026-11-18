"""Retrieval strategies compared by the evaluation.

  - dense   : nomic-embed-text vectors searched in Qdrant (what rag.py uses)
  - bm25    : keyword search (Okapi BM25) over the same chunks
  - hybrid  : dense + bm25 merged with Reciprocal Rank Fusion (RRF)
  - any of the above followed by the cross-encoder re-ranker

Dense search is strong on paraphrases ("restrictions on selling chips to China")
but can miss exact terms (a product name, "Basel III", "CET1"); BM25 is the
reverse. RRF merges the two ranked lists without having to calibrate their
scores against each other: each list contributes 1 / (k + rank) per chunk.

Every chunk is identified by (source_file, chunk_index), the same fields stored
in the Qdrant payload and in data/processed/chunks.json, so results from the
different strategies can be compared directly.
"""

import json
import re
from pathlib import Path

CHUNKS_PATH = Path("data/processed/chunks.json")
RRF_K = 60  # standard constant from Cormack et al. (2009)

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by does for from has have how in is it its of on or "
    "that the their this to was what which with".split()
)


def chunk_key(chunk):
    """Stable identity of a chunk across Qdrant payloads and chunks.json records."""
    return (chunk["source_file"], chunk["chunk_index"])


def tokenize(text):
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def load_chunks(path=CHUNKS_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class BM25Index:
    """Okapi BM25 over the chunk texts (rank-bm25)."""

    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query, n, ticker=None):
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in order:
            if ticker and self.chunks[i]["ticker"] != ticker:
                continue
            results.append(self.chunks[i])
            if len(results) == n:
                break
        return results


def rrf_fuse(rankings, n, k=RRF_K):
    """Merge ranked chunk lists with Reciprocal Rank Fusion; return the top n chunks."""
    scores, chunks = {}, {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            key = chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(key, chunk)
    order = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [chunks[key] for key in order[:n]]


def dense_search(question, n, ticker=None):
    """Dense search in Qdrant; returns payload dicts (same shape as chunks.json)."""
    import rag
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    query_filter = None
    if ticker:
        query_filter = Filter(must=[FieldCondition(key="ticker", match=MatchValue(value=ticker))])
    hits = (
        QdrantClient(url=rag.QDRANT_URL)
        .query_points(
            collection_name=rag.COLLECTION,
            query=rag.embed(question),
            query_filter=query_filter,
            limit=n,
            with_payload=True,
        )
        .points
    )
    return [h.payload for h in hits]


def rerank(question, candidates, k):
    """Re-order candidates with the cross-encoder used by rag.py and keep the top k."""
    import rag

    if not candidates:
        return []
    scores = rag.get_reranker().predict([(question, c["text"]) for c in candidates])
    ranked = sorted(zip(scores, candidates, strict=True), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:k]]
