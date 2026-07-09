"""Precompute RAG answers for the public demo.

The live system needs Qdrant + Ollama, which a free Streamlit Cloud host can't
run. So we run the real pipeline locally over a curated set of questions and save
each answer + sources to a JSON file. The public demo app then serves these
precomputed results directly — the answers are genuinely produced by this RAG,
just generated ahead of time.

Run locally (with Qdrant + Ollama up, in the rag env with SSL_CERT_FILE set):
    python src/retrieval/precompute_demo.py
Output: dashboard/demo_data.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag import answer_question  # noqa: E402

# Curated demo questions. `ticker` mirrors the dashboard's company filter.
DEMO_QUESTIONS = [
    {"question": "How does NVIDIA describe the risk from U.S. export controls to China?", "ticker": "NVDA"},
    {"question": "How did NVIDIA's risk factors change year over year?", "ticker": "NVDA"},
    {"question": "What supply chain risks does Apple disclose?", "ticker": "AAPL"},
    {"question": "What does Apple say about competition in its markets?", "ticker": "AAPL"},
    {"question": "What regulatory and legal risks does JPMorgan describe?", "ticker": "JPM"},
    {"question": "What risks does JPMorgan associate with economic and market conditions?", "ticker": "JPM"},
    {"question": "What cybersecurity risks are disclosed across these companies?", "ticker": None},
    {"question": "How do these companies describe risks from artificial intelligence?", "ticker": None},
]


def main():
    results = []
    for i, item in enumerate(DEMO_QUESTIONS, 1):
        q, ticker = item["question"], item["ticker"]
        print(f"[{i}/{len(DEMO_QUESTIONS)}] {q}  (ticker={ticker})")
        answer, chunks = answer_question(q, ticker=ticker)
        results.append(
            {
                "question": q,
                "ticker": ticker,
                "answer": answer,
                "sources": [
                    {
                        "ticker": c.payload["ticker"],
                        "form": c.payload["form"],
                        "filing_date": c.payload["filing_date"],
                        "section": c.payload["section"],
                    }
                    for c in chunks
                ],
            }
        )

    out = Path(__file__).resolve().parent.parent.parent / "dashboard" / "demo_data.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {len(results)} precomputed answers to {out}")


if __name__ == "__main__":
    main()
