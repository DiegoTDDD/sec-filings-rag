RAGAS scores. Answers by `llama3.2:3b`, judged by `qwen2.5:7b` (local, via Ollama).

| Questions | n | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|---|
| All | 57 | 0.81 | 0.46 | 0.72 | 0.92 |
| Curated (hand-written) | 6 | 0.99 | 0.73 | 0.98 | 0.91 |
| Synthetic | 51 | 0.79 | 0.43 | 0.69 | 0.92 |

Every question received all four scores (no unscored entries).
Source: GitHub Actions run 44223836039 (rag-eval.yml), 2026-11-18, 6 judge shards in parallel.
