Retrieval on 60 synthetic questions, each written from one known chunk.

Questions written from a chunk tend to reuse its words, which favours BM25; compare strategies with each other rather than reading BM25's scores as absolute.

| Strategy | hit@1 | hit@5 | MRR@5 | recall@20 | hit@5 (adjacent) |
|---|---|---|---|---|---|
| Dense (nomic-embed-text) | 0.17 | 0.43 | 0.27 | 0.75 | 0.57 |
| BM25 keyword | 0.45 | 0.92 | 0.63 | 0.97 | 0.93 |
| Hybrid (dense + BM25, RRF) | 0.28 | 0.77 | 0.46 | 0.97 | 0.82 |
| Dense + cross-encoder re-rank (previous pipeline) | 0.37 | 0.70 | 0.49 | 0.75 | 0.73 |
| BM25 + cross-encoder re-rank | 0.43 | 0.90 | 0.59 | 0.97 | 0.92 |
| Hybrid + cross-encoder re-rank (current pipeline) | 0.43 | 0.80 | 0.57 | 0.97 | 0.83 |

Source: GitHub Actions run 44301667578 (rag-eval.yml), 2026-11-18.
The previous run (44223836039, 51 questions) gave the same picture: hybrid + re-rank hit@5 0.80 and recall@20 0.96, against 0.73 and 0.76 for dense + re-rank.
