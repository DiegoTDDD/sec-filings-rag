# 0003. Sentence-aware chunking with overlap

**Status:** Accepted

## Context
Fixed-length chunks cut sentences in half. That blurs the embedding of both halves and produces citations that begin or end mid-clause. Risk-factor paragraphs in SEC filings are long, so a chunk that is too small loses the subject of the sentence and one that is too large mixes several risks.

## Decision
Split each section into sentences with NLTK and pack them greedily into chunks of about 1,000 characters. The last ~150 characters of a chunk, starting at a word boundary, are repeated at the start of the next one (`src/indexing/chunk.py`). Chunks over 6,000 characters are truncated before embedding to stay under the embedding model's 2,048-token limit.

## Alternatives considered
- **Fixed character windows.** Simplest, but splits sentences.
- **Token-based windows (e.g. 512 tokens).** Ties chunk size to one tokenizer; characters are model-independent and easy to reason about.
- **Section-level or semantic chunking.** Sections are far too long to embed well; semantic splitting adds a model call per boundary for unclear gain at this size.

## Consequences
- Every chunk starts and ends on a sentence boundary, so citations read cleanly.
- The overlap means a fact near a boundary appears in two chunks. The evaluation therefore reports an "adjacent" column that also counts the neighbouring chunk as a hit.
