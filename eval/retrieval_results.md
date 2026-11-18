Retrieval on 51 synthetic questions, each written from one known chunk.

Questions written from a chunk tend to reuse its words, which favours BM25; compare strategies with each other rather than reading BM25's scores as absolute.

| Strategy | hit@1 | hit@5 | MRR@5 | recall@20 | hit@5 (adjacent) |
|---|---|---|---|---|---|
| Dense (nomic-embed-text) | 0.16 | 0.47 | 0.28 | 0.76 | 0.57 |
| BM25 keyword | 0.45 | 0.92 | 0.63 | 0.98 | 0.94 |
| Hybrid (dense + BM25, RRF) | 0.29 | 0.78 | 0.48 | 0.96 | 0.82 |
| Dense + cross-encoder re-rank (previous pipeline) | 0.39 | 0.73 | 0.52 | 0.76 | 0.73 |
| BM25 + cross-encoder re-rank | 0.43 | 0.92 | 0.61 | 0.98 | 0.92 |
| Hybrid + cross-encoder re-rank (current pipeline) | 0.41 | 0.80 | 0.56 | 0.96 | 0.82 |

Source: GitHub Actions run 44223836039 (rag-eval.yml), 2026-11-18.
