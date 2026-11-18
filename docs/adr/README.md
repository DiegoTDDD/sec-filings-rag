# Architecture decision records

Each record states one decision, the context that forced it, the options weighed and what it costs. Numbers quoted come from the evaluation in [`eval/`](../../eval).

| # | Decision | Status |
|---|---|---|
| [0001](0001-local-models-only.md) | Run every model locally through Ollama; no paid APIs | Accepted |
| [0002](0002-qdrant-vector-store.md) | Qdrant as the vector store, with metadata filtering by company | Accepted |
| [0003](0003-sentence-aware-chunking.md) | Sentence-aware chunks of ~1,000 characters with 150 of overlap | Accepted |
| [0004](0004-hybrid-retrieval-and-reranking.md) | Hybrid retrieval (dense + BM25, RRF) followed by a cross-encoder | Accepted, supersedes dense-only retrieval |
| [0005](0005-separate-public-demo.md) | Public demo in a separate repository, serving precomputed answers | Accepted |
| [0006](0006-evaluation-method.md) | Synthetic questions from known chunks, judged by a larger local model | Accepted |
