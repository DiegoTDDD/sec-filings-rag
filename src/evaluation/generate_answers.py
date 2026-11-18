"""Run the RAG pipeline over every evaluation question and save what RAGAS needs.

Questions come from two sets:
  - eval/curated_questions.json : 6 hand-written questions and reference answers
  - eval/eval_set.json          : the synthetic set (reference answer written from
                                  the source chunk by build_eval_set.py)

For each question the full pipeline runs (retrieve, re-rank, generate) and the
answer plus the retrieved contexts are stored. Scoring is a separate step
(evaluate_ragas.py) so it can be split across several machines.

Output: eval/answers.json
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
from rag import LLM_MODEL, answer_question  # noqa: E402

CURATED = Path("eval/curated_questions.json")
SYNTHETIC = Path("eval/eval_set.json")
OUT_PATH = Path("eval/answers.json")


def load_questions(curated=CURATED, synthetic=SYNTHETIC):
    """Curated questions first, then the synthetic ones, each tagged with its origin."""
    items = []
    for path, origin in ((curated, "curated"), (synthetic, "synthetic")):
        if not Path(path).exists():
            continue
        for q in json.loads(Path(path).read_text(encoding="utf-8"))["questions"]:
            items.append(
                {
                    "id": q["id"],
                    "origin": origin,
                    "question": q["question"],
                    "reference_answer": q["reference_answer"],
                }
            )
    return items


def main():
    questions = load_questions()
    if not questions:
        sys.exit("no evaluation questions found")
    rows, start = [], time.time()
    for i, q in enumerate(questions, 1):
        answer, chunks = answer_question(q["question"])
        rows.append(dict(q, answer=answer, contexts=[c.payload["text"] for c in chunks]))
        print(f"[{i}/{len(questions)}] {q['id']} answered ({len(chunks)} contexts)")
    OUT_PATH.write_text(
        json.dumps(
            {"generator": LLM_MODEL, "seconds": round(time.time() - start), "rows": rows},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} answers to {OUT_PATH}")


if __name__ == "__main__":
    main()
