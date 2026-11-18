# 0005. Public demo in a separate repository

**Status:** Accepted

## Context
The live pipeline needs Qdrant, Ollama with two models and PyTorch for the cross-encoder. Free hosts such as Streamlit Community Cloud provide about 1 GB of memory and no Docker, so they cannot run it. A recruiter or reviewer should still be able to try the system without installing anything.

## Decision
Run the real pipeline locally over a fixed set of questions (`src/retrieval/precompute_demo.py`) and save the answers and sources to `dashboard/demo_data.json`. A light Streamlit app (`dashboard/demo_app.py`) serves them. Both files are copied to the [sec-filings-rag-demo](https://github.com/DiegoTDDD/sec-filings-rag-demo) repository, which holds only what the host needs.

## Alternatives considered
- **Deploy the full stack on a paid host.** Works, but costs money and goes down when the budget does.
- **Use hosted APIs for the demo only.** The demo would then show a different system from the one in this repository.
- **Keep the demo app in this repository.** The host would install the full `requirements.txt` (PyTorch included) and fail.

## Consequences
- The demo always works and costs nothing, and its answers were genuinely produced by this pipeline.
- Visitors can only pick from the precomputed questions; the README says so.
- The two copies can drift, so the `demo-in-sync` job in `.github/workflows/tests.yml` fails whenever the demo repository's files differ from the ones here.
