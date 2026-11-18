RAGAS scores. Answers by `llama3.2:3b`, judged by `qwen2.5:7b` (local, via Ollama).

| Questions | n | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|---|
| All | 66 | 0.82 | 0.52 | 0.73 (1 unscored) | 0.93 |
| Curated (hand-written) | 6 | 0.94 | 0.58 | 0.97 (1 unscored) | 0.87 |
| Synthetic | 60 | 0.81 | 0.52 | 0.71 | 0.93 |

"Unscored" = the judge call failed every retry for that question and metric; it is reported, not dropped.
Source: GitHub Actions run 44301667578 (rag-eval.yml), 2026-11-18, 6 judge shards in parallel.
