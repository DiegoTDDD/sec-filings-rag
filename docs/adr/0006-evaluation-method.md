# 0006. Evaluation: synthetic questions from known chunks, judged by a larger local model

**Status:** Accepted

## Context
The first evaluation had six hand-written questions judged by the same 3B model that wrote the answers. Six questions cannot separate retrieval strategies, a model grading itself is biased, and two of the four RAGAS metrics timed out (`docs/EVALUATION_NOTES.md`).

## Decision
- **Question set.** For a sample of chunks stratified by company, form and section, `llama3.2:3b` writes one question the chunk answers plus a short reference answer. Questions that leak the setup ("according to the excerpt"), do not name the company or are malformed are discarded. Generation continues until 60 valid questions are kept (`src/evaluation/build_eval_set.py`). The six hand-written questions are kept and reported separately.
- **Retrieval metrics.** Because each question comes from a known chunk, retrieval is scored exactly: hit@1, hit@5, MRR@5 and recall@20, with and without counting the overlapping neighbour chunk.
- **Answer metrics.** RAGAS faithfulness, answer relevancy, context precision and context recall, judged by `qwen2.5:7b`, a larger model from a different family than the generator. Judge calls are limited to two at a time with a 30-minute timeout, and the questions are split across six GitHub runners.
- **Schedule.** `.github/workflows/rag-eval.yml` runs on demand, when the workflow changes and once a month.

## Alternatives considered
- **More hand-written questions.** The best signal, but slow to write and to keep up to date as filings change.
- **RAGAS test-set generation.** Heavier, needs a stronger generator to be reliable, and hides which chunk each question came from, which is what makes exact retrieval scoring possible.
- **Hosted judge (GPT-4-class).** More reliable judging, but paid (see [0001](0001-local-models-only.md)).

## Consequences
- On 66 questions the hybrid pipeline scores hit@5 0.80 and recall@20 0.97 for retrieval, and faithfulness 0.82 / context recall 0.93 for answers. One context-precision score was unscored and is reported as such. The earlier 57-question run gave the same picture.
- Synthetic questions reuse their chunk's wording, which favours BM25. The results compare strategies with each other; they are not absolute numbers for real users.
- Local generation on CPU is not exactly reproducible, so the question set changes slightly between runs. The run fails if fewer than 50 valid questions are produced.
- A full run takes about two and a half hours of runner time.
