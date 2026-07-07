"""Embed every chunk with Ollama's nomic-embed-text and load it into Qdrant.

Reads data/processed/chunks.json, generates a 768-dim embedding per chunk via
the local Ollama server, and upserts each as a Qdrant point whose payload holds
the chunk text plus its provenance (ticker, form, filing_date, section). That
payload is what powers metadata-filtered retrieval and source citations.

nomic-embed-text caps input at 2048 tokens; a few dense chunks exceed that and
would make Ollama return 500. We defensively truncate each chunk to a safe
character budget before embedding, and retry transient failures. With truncation
in place the requests are safe to parallelize, so a small thread pool keeps the
CPU busy and cuts indexing time. The collection is recreated each run so
re-indexing never duplicates points.

Prereqs: Ollama running with nomic-embed-text pulled, Qdrant up on :6333.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

CHUNKS_PATH = Path("data/processed/chunks.json")
COLLECTION = "sec_filings"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
QDRANT_URL = "http://localhost:6333"

WORKERS = 4        # concurrent embedding requests; safe now that chunks are truncated
BATCH_SIZE = 200
MAX_CHARS = 6000   # ~4 chars/token keeps input under the 2048-token cap
MAX_RETRIES = 3

_client = httpx.Client(timeout=120.0)


def embed(text):
    """Return the embedding for one chunk, truncating over-long input and retrying."""
    safe_text = text[:MAX_CHARS]
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = _client.post(
                OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": safe_text}
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as exc:  # noqa: BLE001 - transient; retry a few times
            last_error = exc
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"embedding failed after {MAX_RETRIES} tries: {last_error}")


def main():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks.")

    dim = len(embed(chunks[0]["text"]))
    print(f"Embedding dimension: {dim}")

    client = QdrantClient(url=QDRANT_URL)
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION}' created (cosine, dim={dim}).")

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with {WORKERS} workers...")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        vectors = list(pool.map(embed, texts))

    print("Embeddings done. Upserting to Qdrant...")
    total = 0
    for start in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[start:start + BATCH_SIZE]
        batch_vectors = vectors[start:start + BATCH_SIZE]
        points = [
            PointStruct(
                id=start + j,
                vector=batch_vectors[j],
                payload={
                    "text": chunk["text"],
                    "ticker": chunk["ticker"],
                    "form": chunk["form"],
                    "filing_date": chunk["filing_date"],
                    "section": chunk["section"],
                    "chunk_index": chunk["chunk_index"],
                    "source_file": chunk["source_file"],
                },
            )
            for j, chunk in enumerate(batch_chunks)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        total += len(points)
        print(f"  upserted {total}/{len(chunks)}")

    count = client.count(collection_name=COLLECTION).count
    print(f"\nDone. Qdrant collection '{COLLECTION}' now holds {count} points.")


if __name__ == "__main__":
    main()
