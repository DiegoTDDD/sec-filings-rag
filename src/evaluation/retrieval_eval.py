"""Compare retrieval strategies on the synthetic evaluation set.

Each question was written from one known chunk, so a strategy is scored on
whether it puts that chunk in front of the LLM:

  - hit@1, hit@5 : the source chunk is the first / among the first 5 results
  - MRR@5        : mean of 1/rank of the source chunk within the top 5 (0 if absent)
  - recall@20    : the source chunk is in the 20-candidate shortlist (before re-ranking)

Chunks overlap by ~150 characters, so a question can also be fully answered by
the chunk right before or after its source. The "adjacent" columns count those
neighbours (same file, chunk_index +-1) as hits; the strict columns do not.

Known bias: a question written from a chunk tends to reuse that chunk's words,
which favours keyword search (BM25). The numbers compare strategies on the same
footing, but BM25's absolute scores here are optimistic for real users, who
paraphrase. That is why the pipeline uses hybrid retrieval, not BM25 alone.

Input:  eval/eval_set.json, data/processed/chunks.json, Qdrant + Ollama running
Output: eval/retrieval_results.json and eval/retrieval_results.md
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
from strategies import (  # noqa: E402
    BM25Index,
    Chunk,
    chunk_key,
    dense_search,
    load_chunks,
    rerank,
    rrf_fuse,
)

log = logging.getLogger(__name__)

Rank = int | None
Scores = dict[str, float]

EVAL_PATH = Path("eval/eval_set.json")
OUT_JSON = Path("eval/retrieval_results.json")
OUT_MD = Path("eval/retrieval_results.md")
SHORTLIST = 20
FINAL_K = 5

STRATEGIES = [
    ("dense", "Dense (nomic-embed-text)"),
    ("bm25", "BM25 keyword"),
    ("hybrid", "Hybrid (dense + BM25, RRF)"),
    ("dense_rerank", "Dense + cross-encoder re-rank (previous pipeline)"),
    ("bm25_rerank", "BM25 + cross-encoder re-rank"),
    ("hybrid_rerank", "Hybrid + cross-encoder re-rank (current pipeline)"),
]


def rank_of(results: list[Chunk], source: tuple[str, int], adjacent: bool = False) -> Rank:
    """1-based rank of the source chunk in results, or None."""
    source_file, source_index = source
    for rank, chunk in enumerate(results, start=1):
        file_name, index = chunk_key(chunk)
        if file_name != source_file:
            continue
        if index == source_index or (adjacent and abs(index - source_index) == 1):
            return rank
    return None


def summarize(ranks: list[Rank], shortlist_hits: list[bool]) -> Scores:
    """hit@1, hit@5, MRR@5 and recall@20 over all questions."""
    n = len(ranks)
    if n == 0:
        return {"hit@1": 0.0, "hit@5": 0.0, "mrr@5": 0.0, "recall@20": 0.0}
    return {
        "hit@1": sum(1 for r in ranks if r == 1) / n,
        "hit@5": sum(1 for r in ranks if r is not None and r <= FINAL_K) / n,
        "mrr@5": sum(1.0 / r for r in ranks if r is not None and r <= FINAL_K) / n,
        "recall@20": sum(shortlist_hits) / n,
    }


def run_strategies(
    question: str, bm25: BM25Index
) -> dict[str, tuple[list[Chunk], list[Chunk]]]:
    """For each strategy: (final top FINAL_K, SHORTLIST candidates before re-ranking)."""
    dense = dense_search(question, SHORTLIST)
    keyword = bm25.search(question, SHORTLIST)
    hybrid = rrf_fuse([dense, keyword], SHORTLIST)
    return {
        "dense": (dense[:FINAL_K], dense),
        "bm25": (keyword[:FINAL_K], keyword),
        "hybrid": (hybrid[:FINAL_K], hybrid),
        "dense_rerank": (rerank(question, dense, FINAL_K), dense),
        "bm25_rerank": (rerank(question, keyword, FINAL_K), keyword),
        "hybrid_rerank": (rerank(question, hybrid, FINAL_K), hybrid),
    }


def to_markdown(results: dict[str, dict[str, Scores]], n_questions: int) -> str:
    lines = [
        f"Retrieval on {n_questions} synthetic questions, each written from one known chunk.",
        "",
        "Questions written from a chunk tend to reuse its words, which favours BM25; "
        "compare strategies with each other rather than reading BM25's scores as absolute.",
        "",
        "| Strategy | hit@1 | hit@5 | MRR@5 | recall@20 | hit@5 (adjacent) |",
        "|---|---|---|---|---|---|",
    ]
    for key, label in STRATEGIES:
        s, a = results[key]["strict"], results[key]["adjacent"]
        lines.append(
            f"| {label} | {s['hit@1']:.2f} | {s['hit@5']:.2f} | {s['mrr@5']:.2f} "
            f"| {s['recall@20']:.2f} | {a['hit@5']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    questions = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["questions"]
    bm25 = BM25Index(load_chunks())
    ranks: dict[str, dict[str, list[Rank]]] = {
        k: {"strict": [], "adjacent": []} for k, _ in STRATEGIES
    }
    shortlist: dict[str, dict[str, list[bool]]] = {
        k: {"strict": [], "adjacent": []} for k, _ in STRATEGIES
    }

    start = time.time()
    for i, q in enumerate(questions, 1):
        source = (q["source_file"], q["chunk_index"])
        for key, (top, candidates) in run_strategies(q["question"], bm25).items():
            for mode, adjacent in (("strict", False), ("adjacent", True)):
                ranks[key][mode].append(rank_of(top, source, adjacent))
                shortlist[key][mode].append(rank_of(candidates, source, adjacent) is not None)
        log.info("[%d/%d] %s done", i, len(questions), q["id"])

    results = {
        key: {
            mode: summarize(ranks[key][mode], shortlist[key][mode])
            for mode in ("strict", "adjacent")
        }
        for key, _ in STRATEGIES
    }
    OUT_JSON.write_text(
        json.dumps(
            {
                "questions": len(questions),
                "seconds": round(time.time() - start),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    table = to_markdown(results, len(questions))
    OUT_MD.write_text(table, encoding="utf-8")
    log.info("\n%s", table)


if __name__ == "__main__":
    main()
