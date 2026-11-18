"""Build a synthetic evaluation set from the indexed chunks.

For a stratified sample of chunks (balanced across company, form and section) a
local LLM writes one specific question that the chunk answers, plus a short
reference answer taken from that chunk. Because each question comes from a
known chunk, retrieval can be scored exactly: did the strategy find the chunk
the question was written from?

Questions that leak the setup ("according to the excerpt"), do not name the
company, or are malformed are discarded, so the set only keeps questions a real
user could ask.

Input:  data/processed/chunks.json
Output: eval/eval_set.json

Env: GEN_MODEL (default llama3.2:3b), OLLAMA_URL (default http://localhost:11434),
     EVAL_TARGET (default 60): stop once this many valid questions are kept.
     EVAL_CANDIDATES (default 160): most chunks tried. Generation is not exactly
     reproducible on CPU and roughly a third of questions are discarded, so the
     script keeps drawing chunks (in the same stratified order) until the target
     is met instead of trying a fixed number.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
from strategies import Chunk, chunk_key, load_chunks  # noqa: E402

log = logging.getLogger(__name__)

OUT_PATH = Path("eval/eval_set.json")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
GEN_MODEL = os.environ.get("GEN_MODEL", "llama3.2:3b")
TARGET = int(os.environ.get("EVAL_TARGET", "60"))
CANDIDATES = int(os.environ.get("EVAL_CANDIDATES", "160"))
MIN_CHUNK_CHARS = 600
SEED = 42

COMPANY_NAMES = {
    "AAPL": ("apple", "aapl"),
    "NVDA": ("nvidia", "nvda"),
    "JPM": ("jpmorgan", "jp morgan", "jpm"),
}
_LEAKS = re.compile(
    r"\b(excerpt|passage|the text|this text|above|this section|this document|the chunk|provided)\b",
    re.I,
)

PROMPT = """You are writing test questions for a search system over SEC filings.

Company: {company}
Filing: {form} filed {filing_date}, section "{section}"

Excerpt:
\"\"\"
{text}
\"\"\"

Write ONE specific question that this excerpt answers and that an analyst might ask.
Rules:
- Name the company ({company}) in the question.
- Ask about a concrete fact, risk, figure or claim stated in the excerpt.
- Do not mention "the excerpt", "the passage" or "the text".
Then write a short reference answer (1-3 sentences) using only the excerpt.

Reply with JSON only: {{"question": "...", "reference_answer": "..."}}"""

COMPANY_LABEL = {"AAPL": "Apple", "NVDA": "NVIDIA", "JPM": "JPMorgan Chase"}


def stratified_sample(chunks: list[Chunk], n: int, seed: int = SEED) -> list[Chunk]:
    """Round-robin over (ticker, form, section) groups so no filing dominates."""
    rng = random.Random(seed)
    groups: dict[tuple[str, str, str], list[Chunk]] = defaultdict(list)
    for c in chunks:
        if len(c["text"]) >= MIN_CHUNK_CHARS:
            groups[(c["ticker"], c["form"], c["section"])].append(c)
    for members in groups.values():
        rng.shuffle(members)
    keys = sorted(groups)
    sample: list[Chunk] = []
    i = 0
    while len(sample) < n and any(groups.values()):
        key = keys[i % len(keys)]
        if groups[key]:
            sample.append(groups[key].pop())
        i += 1
    return sample


def is_valid(item: dict[str, Any], ticker: str) -> bool:
    """Keep only well-formed questions a real user could ask."""
    question = (item.get("question") or "").strip()
    answer = (item.get("reference_answer") or "").strip()
    if not (25 <= len(question) <= 300 and question.endswith("?")):
        return False
    if not (20 <= len(answer) <= 800):
        return False
    if _LEAKS.search(question):
        return False
    return any(name in question.lower() for name in COMPANY_NAMES[ticker])


def ask_model(chunk: Chunk, client: httpx.Client) -> dict[str, Any]:
    """Ask the local LLM for one question + reference answer about this chunk."""
    prompt = PROMPT.format(
        company=COMPANY_LABEL[chunk["ticker"]],
        form=chunk["form"],
        filing_date=chunk["filing_date"],
        section=chunk["section"],
        text=chunk["text"],
    )
    response = client.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": GEN_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0, "seed": SEED},
        },
    )
    response.raise_for_status()
    return json.loads(response.json()["message"]["content"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    chunks = load_chunks()
    sample = stratified_sample(chunks, CANDIDATES)
    log.info(
        "%d chunks; up to %d candidates, target %d questions", len(chunks), len(sample), TARGET
    )

    items: list[dict[str, Any]] = []
    rejected = 0
    with httpx.Client(timeout=300.0) as client:
        for i, chunk in enumerate(sample, 1):
            if len(items) >= TARGET:
                break
            try:
                item = ask_model(chunk, client)
            except Exception as exc:  # noqa: BLE001 - a bad generation just drops that chunk
                log.warning("[%d/%d] generation failed: %s", i, len(sample), exc)
                rejected += 1
                continue
            if not is_valid(item, chunk["ticker"]):
                rejected += 1
                continue
            source_file, chunk_index = chunk_key(chunk)
            items.append(
                {
                    "id": f"q{len(items) + 1:03d}",
                    "question": item["question"].strip(),
                    "reference_answer": item["reference_answer"].strip(),
                    "ticker": chunk["ticker"],
                    "form": chunk["form"],
                    "section": chunk["section"],
                    "source_file": source_file,
                    "chunk_index": chunk_index,
                }
            )
            log.info(
                "[%d/%d] %s %s: %s", i, len(sample), chunk["ticker"], chunk["section"],
                items[-1]["question"],
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps({"generator": GEN_MODEL, "seed": SEED, "questions": items}, indent=2),
        encoding="utf-8",
    )
    log.info("Kept %d questions, rejected %d. Written to %s", len(items), rejected, OUT_PATH)


if __name__ == "__main__":
    main()
