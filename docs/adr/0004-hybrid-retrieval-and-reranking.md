# 0004. Hybrid retrieval followed by a cross-encoder

**Status:** Accepted. Supersedes the original dense-only retrieval.

## Context
The first pipeline used dense search only (20 candidates) and re-ranked them with a cross-encoder. The retrieval evaluation showed the source chunk was missing from the 20 candidates for about a quarter of the questions (recall@20 0.76), and nothing after retrieval can recover a chunk that was never retrieved. The misses were mostly questions built on exact terms (product names, "CET1", "Basel III") that embeddings blur.

## Decision
Run dense search (Qdrant) and BM25 keyword search over the same chunks, merge the two lists with Reciprocal Rank Fusion (k = 60), then re-rank the fused 20 with the cross-encoder and keep five (`src/retrieval/rag.py`, `src/retrieval/strategies.py`). `RETRIEVAL_MODE=dense` keeps the old behaviour. If `chunks.json` is missing, the pipeline falls back to dense search and logs a warning.

## Alternatives considered
- **BM25 alone.** Scored highest in the evaluation, but the synthetic questions reuse the source chunk's words, which flatters keyword search. Real users paraphrase, and dense search is what handles that.
- **Weighted score fusion.** Needs dense and BM25 scores calibrated against each other; RRF uses ranks only and has one well-known constant.
- **More dense candidates (e.g. 50).** Raises recall a little but does not fix exact-term misses, and makes re-ranking slower.

## Consequences
Measured on the same 51 questions (`eval/retrieval_results.md`):

| | hit@5 | recall@20 |
|---|---|---|
| Dense + re-rank (before) | 0.73 | 0.76 |
| Hybrid + re-rank (now) | 0.80 | 0.96 |

- The BM25 index lives in memory (built once at API start-up from `chunks.json`), which is fine at 3,200 chunks but would need a search engine at millions.
- The re-ranker now sees better candidates, but it is still the slowest step on CPU.
