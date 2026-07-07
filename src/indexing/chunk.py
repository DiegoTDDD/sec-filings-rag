"""Split each processed section into sentence-aware chunks with overlap.

Naive fixed-length chunking cuts sentences mid-thought, which blurs embeddings
and produces citations that start or end in the middle of a clause. Instead we
split on sentence boundaries (via NLTK) and greedily pack sentences into chunks
of roughly TARGET_CHARS, carrying OVERLAP_CHARS of trailing text into the next
chunk so ideas that straddle a boundary aren't lost.

Each chunk keeps its source metadata (ticker, form, filing date, section) plus a
positional index, which the indexer will attach to every vector so answers can
cite the exact filing and section they came from.

Input:  data/processed/*.txt + data/processed/metadata.json
Output: data/processed/chunks.json  (list of {text, metadata} records)
"""
import json
from pathlib import Path

import nltk

PROCESSED_DIR = Path("data/processed")
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"

TARGET_CHARS = 1000   # aim for ~1000 characters per chunk
OVERLAP_CHARS = 150   # carry ~150 characters of context into the next chunk


def ensure_punkt():
    """Make sure NLTK's sentence tokenizer data is available (downloads once)."""
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def chunk_sentences(sentences):
    """Greedily pack sentences into ~TARGET_CHARS chunks with OVERLAP_CHARS overlap."""
    chunks = []
    current = ""

    for sentence in sentences:
        # If adding this sentence would overflow and we already have content,
        # close the current chunk and start the next one with an overlap tail.
        if current and len(current) + len(sentence) + 1 > TARGET_CHARS:
            chunks.append(current.strip())
            tail = current[-OVERLAP_CHARS:]
            # Start the overlap at a word boundary so we don't begin mid-word.
            space = tail.find(" ")
            current = (tail[space + 1:] if space != -1 else tail) + " " + sentence
        else:
            current = f"{current} {sentence}".strip()

    if current.strip():
        chunks.append(current.strip())
    return chunks


def main():
    ensure_punkt()
    metadata = json.loads((PROCESSED_DIR / "metadata.json").read_text(encoding="utf-8"))

    records = []
    for entry in metadata:
        text = (PROCESSED_DIR / entry["filename"]).read_text(encoding="utf-8")
        sentences = nltk.sent_tokenize(text)
        chunks = chunk_sentences(sentences)

        for i, chunk_text in enumerate(chunks):
            records.append(
                {
                    "text": chunk_text,
                    "ticker": entry["ticker"],
                    "form": entry["form"],
                    "filing_date": entry["filing_date"],
                    "section": entry["section"],
                    "chunk_index": i,
                    "source_file": entry["filename"],
                }
            )
        print(f"  {entry['filename']:45} -> {len(chunks):3} chunks")

    CHUNKS_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nDone. {len(records)} chunks written to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
