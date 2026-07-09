# RAGAS Evaluation — Results and Notes

## Setup
- **Evaluation set:** 6 curated questions across the three companies (AAPL, NVDA, JPM),
  spanning Risk Factors and MD&A, each with a hand-written reference answer.
- **Judge / evaluator model:** local `llama3.2:3b` via Ollama (no external API, no cost).
- **Embeddings for RAGAS:** local `nomic-embed-text` via Ollama.
- **Pipeline under test:** dense retrieval (Qdrant) → cross-encoder re-ranking → llama3.2 generation.

## Results (local llama3.2:3b judge)

| Metric | Score | Reading |
|---|---|---|
| answer_relevancy | **0.9478** | Answers strongly address the question asked. |
| context_recall | **0.6701** | Retrieved chunks cover ~67% of what the reference answer needed. |
| faithfulness | n/a (timeout) | Judge could not complete within the time limit. |
| context_precision | n/a (timeout) | Judge could not complete within the time limit. |

Per-question breakdown saved to `data/processed/ragas_results.csv`.

## Why two metrics returned n/a — and what a stronger judge would change

faithfulness and context_precision are the most demanding metrics for the judge:
they require decomposing the answer into individual claims and verifying each one
against the retrieved context, which is many reasoning steps per question. The
local 3B model timed out on these while completing the lighter metrics
(answer_relevancy, context_recall).

This is a limitation of the **evaluator**, not of the RAG pipeline itself. With a
frontier-class judge (e.g. GPT-4-class or Claude), we would expect:
- all four metrics to complete reliably (no timeouts),
- more stable, less noisy scores across runs,
- sharper discrimination on faithfulness in particular, since detecting subtle
  unsupported claims is exactly where a larger model outperforms a 3B one.

The pipeline is judge-agnostic: swapping the evaluator LLM is a configuration
change, so these same metrics can be regenerated with a stronger judge later
without touching the RAG code.

## Takeaway
answer_relevancy ≈ 0.95 and context_recall ≈ 0.67 are directional evidence that
retrieval + generation are working well. The faithfulness/precision gap is an
evaluator-capacity artifact, reported transparently rather than hidden.
