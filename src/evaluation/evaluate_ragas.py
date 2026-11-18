"""Score the RAG answers with RAGAS, using a local Ollama model as the judge.

Four metrics per question:
  - faithfulness       : is every claim in the answer supported by the retrieved context?
  - answer_relevancy   : does the answer address the question asked?
  - context_precision  : are the retrieved chunks relevant (low noise)?
  - context_recall     : do the retrieved chunks cover the reference answer?

The judge (JUDGE_MODEL, default qwen2.5:7b) is a different and larger model than
the one that wrote the answers (llama3.2:3b), so the pipeline is not grading
itself. Local judging is slow on a CPU, so the questions can be split into
shards that run on separate machines:

    python src/evaluation/evaluate_ragas.py --shard 0 --shards 6

Why earlier runs returned n/a for two metrics: RAGAS sends up to 16 judge calls
at once with a 180-second limit each, while a local Ollama server answers them
one or two at a time, so queued calls timed out. Here the judge gets at most
MAX_WORKERS calls at once and each may take up to CALL_TIMEOUT seconds.

Input:  eval/answers.json (from generate_answers.py)
Output: eval/ragas/shard_<i>.csv  (merge with merge_ragas.py)
Needs:  pip install ragas==0.3.9 and python src/evaluation/fix_ragas_import.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ANSWERS = Path("eval/answers.json")
OUT_DIR = Path("eval/ragas")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
MAX_WORKERS = int(os.environ.get("JUDGE_WORKERS", "2"))
CALL_TIMEOUT = int(os.environ.get("JUDGE_TIMEOUT", "1800"))
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def shard_rows(rows: list[dict[str, Any]], shard: int, shards: int) -> list[dict[str, Any]]:
    """Every shards-th row starting at shard, so curated and synthetic questions spread evenly."""
    if not 0 <= shard < shards:
        raise ValueError(f"shard must be in [0, {shards})")
    return rows[shard::shards]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()

    from datasets import Dataset
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    rows = shard_rows(
        json.loads(ANSWERS.read_text(encoding="utf-8"))["rows"], args.shard, args.shards
    )
    log.info(
        "shard %d/%d: %d questions, judge %s", args.shard + 1, args.shards, len(rows), JUDGE_MODEL
    )

    dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "contexts": [r["contexts"] for r in rows],
            "ground_truth": [r["reference_answer"] for r in rows],
        }
    )
    judge = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBED_MODEL))
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(timeout=CALL_TIMEOUT, max_workers=MAX_WORKERS, max_retries=3),
    )

    df = result.to_pandas()
    df.insert(0, "origin", [r["origin"] for r in rows])
    df.insert(0, "id", [r["id"] for r in rows])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"shard_{args.shard}.csv"
    df.to_csv(out, index=False)
    log.info("%s", result)
    log.info("Per-question scores written to %s", out)


if __name__ == "__main__":
    main()
