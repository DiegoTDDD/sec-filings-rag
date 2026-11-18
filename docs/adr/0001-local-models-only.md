# 0001. Run every model locally through Ollama

**Status:** Accepted

## Context
The project has to be reproducible by anyone who clones it and has to run its full evaluation in CI, at no cost. Hosted LLM APIs need keys, cost money per call and can change behind the same model name, which makes results hard to compare over time.

## Decision
All models run locally:
- embeddings: `nomic-embed-text` (Ollama)
- answers: `llama3.2:3b` (Ollama, OpenAI-compatible endpoint)
- re-ranking: `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers)
- evaluation judge: `qwen2.5:7b` (Ollama)

The generator is reached through `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY`, so a hosted model can be swapped in without code changes.

## Alternatives considered
- **Hosted API (OpenAI, Anthropic) for generation and judging.** Better answers and faster judging, but paid, needs secrets in CI, and the model behind a name can change.
- **Hugging Face Transformers in-process.** No server, but slower on CPU and heavier to install than Ollama.

## Consequences
- Zero cost; the full evaluation runs on free GitHub runners (see `.github/workflows/rag-eval.yml`).
- A 3B generator is the main quality ceiling: answer relevancy is 0.46 on the synthetic set (0.73 on the hand-written questions) while faithfulness is 0.81. A larger generator is the first lever if quality matters more than cost.
- Judging on CPU is slow, so the RAGAS run is split across six parallel runners.
