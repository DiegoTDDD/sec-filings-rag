"""Merge the RAGAS shard results into one table.

Reports the mean of each metric overall and per question set (curated and
synthetic), plus how many questions got no score for a metric (a judge call that
failed every retry). A missing score is shown, never silently dropped.

Input:  eval/ragas/shard_*.csv, eval/answers.json
Output: eval/ragas_results.csv, eval/ragas_results.md
"""

import csv
import glob
import json
import math
import os
from pathlib import Path

from evaluate_ragas import METRIC_NAMES

OUT_CSV = Path("eval/ragas_results.csv")
OUT_MD = Path("eval/ragas_results.md")


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def summarize(rows):
    """Mean and missing count per metric for a list of CSV rows (dicts)."""
    summary = {}
    for metric in METRIC_NAMES:
        values = [_number(r.get(metric)) for r in rows]
        scored = [v for v in values if v is not None]
        summary[metric] = {
            "mean": sum(scored) / len(scored) if scored else None,
            "missing": len(values) - len(scored),
        }
    return summary


def to_markdown(groups, judge, generator):
    header = "| Questions | n | " + " | ".join(METRIC_NAMES) + " |"
    lines = [
        f"RAGAS scores. Answers by `{generator}`, judged by `{judge}` (local, via Ollama).",
        "",
        header,
        "|" + "---|" * (len(METRIC_NAMES) + 2),
    ]
    for label, rows in groups:
        cells = []
        for stats in summarize(rows).values():
            mean = "n/a" if stats["mean"] is None else f"{stats['mean']:.2f}"
            missing = f" ({stats['missing']} unscored)" if stats["missing"] else ""
            cells.append(mean + missing)
        lines.append(f"| {label} | {len(rows)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    files = sorted(glob.glob("eval/ragas/shard_*.csv"))
    rows = []
    for f in files:
        with open(f, newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise SystemExit("no shard results found")
    rows.sort(key=lambda r: r["id"])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    generator = json.loads(Path("eval/answers.json").read_text(encoding="utf-8"))["generator"]
    judge = os.environ.get("JUDGE_MODEL", "qwen2.5:7b")
    groups = [
        ("All", rows),
        ("Curated (hand-written)", [r for r in rows if r["origin"] == "curated"]),
        ("Synthetic", [r for r in rows if r["origin"] == "synthetic"]),
    ]
    table = to_markdown([g for g in groups if g[1]], judge, generator)
    OUT_MD.write_text(table, encoding="utf-8")
    print(f"merged {len(files)} shards, {len(rows)} questions\n")
    print(table)


if __name__ == "__main__":
    main()
