"""Evaluate the RAG pipeline with RAGAS, using local Ollama models as the judge.

We run the full RAG (retrieve, re-rank, generate) over a small curated set of
questions spanning all three companies and several sections, then score four
RAGAS metrics:

  - faithfulness       : is the answer grounded in the retrieved context?
  - answer_relevancy   : does the answer address the question asked?
  - context_precision  : are the retrieved chunks relevant (low noise)?
  - context_recall     : do the retrieved chunks cover the reference answer?

IMPORTANT (documented limitation): the judge here is the local llama3.2:3b. A
small judge produces noisier, less discriminative scores than a frontier model
(e.g. GPT-4-class) would - especially for faithfulness and context_precision,
which hinge on fine-grained judgement. The numbers below are therefore directional
evidence that the pipeline works, not a precise leaderboard figure. With a
stronger judge the same harness would yield more stable, trustworthy scores.

This is slow: every metric calls the local LLM many times per question.
"""
import sys
from pathlib import Path

# Make rag.py importable from the retrieval package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
from rag import answer_question  # noqa: E402

from datasets import Dataset  # noqa: E402
from langchain_ollama import ChatOllama, OllamaEmbeddings  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# Curated evaluation questions with reference (ground-truth) answers.
EVAL_SET = [
    {
        "question": "How does NVIDIA describe the risk from U.S. export controls to China?",
        "ground_truth": (
            "NVIDIA states that U.S. export controls restrict sales of its data center "
            "GPUs and related products to China, can disproportionately harm NVIDIA, may "
            "reduce demand, and encourage customers to design out U.S. semiconductors or "
            "buy from competitors, hurting its competitive position and financial results."
        ),
    },
    {
        "question": "What supply chain risks does Apple disclose?",
        "ground_truth": (
            "Apple discloses dependence on outsourced manufacturing and component supply "
            "concentrated in a few regions, exposure to disruptions, component shortages, "
            "supplier concentration, and geopolitical and logistics risks that could raise "
            "costs or limit product availability."
        ),
    },
    {
        "question": "What regulatory and legal risks does JPMorgan describe?",
        "ground_truth": (
            "JPMorgan describes extensive regulation across jurisdictions, capital and "
            "liquidity requirements, potential fines and enforcement actions, litigation, "
            "and the risk that new or changing regulation increases compliance costs and "
            "constrains its business."
        ),
    },
    {
        "question": "How does NVIDIA describe demand for its data center products?",
        "ground_truth": (
            "NVIDIA describes strong demand for its data center and AI computing platforms "
            "driving revenue growth, while noting demand can be volatile and is exposed to "
            "supply constraints and export restrictions."
        ),
    },
    {
        "question": "What does Apple say about competition in its markets?",
        "ground_truth": (
            "Apple describes highly competitive markets characterized by rapid technological "
            "change, aggressive pricing, frequent new product introductions, and competitors "
            "with significant resources, which pressure margins and market share."
        ),
    },
    {
        "question": "What risks does JPMorgan associate with economic and market conditions?",
        "ground_truth": (
            "JPMorgan associates risks with macroeconomic and market conditions such as "
            "interest rate changes, inflation, recessions, market volatility, and credit "
            "deterioration, which can reduce revenue and increase credit losses."
        ),
    },
]


def build_dataset():
    """Run the RAG over each question and assemble the RAGAS dataset."""
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for i, item in enumerate(EVAL_SET, 1):
        q = item["question"]
        print(f"[{i}/{len(EVAL_SET)}] running RAG: {q}")
        answer, chunks = answer_question(q)
        rows["question"].append(q)
        rows["answer"].append(answer)
        rows["contexts"].append([c.payload["text"] for c in chunks])
        rows["ground_truth"].append(item["ground_truth"])
    return Dataset.from_dict(rows)


def main():
    print("Building evaluation dataset (this runs the full RAG per question)...\n")
    dataset = build_dataset()

    # Configure RAGAS to use local Ollama for both the judge LLM and embeddings.
    judge = LangchainLLMWrapper(ChatOllama(model="llama3.2:3b", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

    print("\nScoring with RAGAS (local judge = llama3.2:3b; this is slow)...\n")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embeddings,
    )

    print("\n" + "=" * 60)
    print("RAGAS RESULTS (local llama3.2:3b judge)")
    print("=" * 60)
    print(result)

    # Persist a per-question breakdown for the case study.
    df = result.to_pandas()
    out = Path("data/processed/ragas_results.csv")
    df.to_csv(out, index=False)
    print(f"\nPer-question scores written to {out}")


if __name__ == "__main__":
    main()
