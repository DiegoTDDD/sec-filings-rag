"""Answer generation input, RAGAS sharding and result merging (no models needed)."""

import json

import evaluate_ragas
import generate_answers
import merge_ragas


def test_load_questions_tags_origin_and_keeps_curated_first(tmp_path):
    curated = tmp_path / "curated.json"
    synthetic = tmp_path / "synthetic.json"
    curated.write_text(
        json.dumps({"questions": [{"id": "c01", "question": "Q1?", "reference_answer": "A1"}]})
    )
    synthetic.write_text(
        json.dumps(
            {
                "questions": [
                    {"id": "q001", "question": "Q2?", "reference_answer": "A2", "ticker": "AAPL"}
                ]
            }
        )
    )
    items = generate_answers.load_questions(curated, synthetic)
    assert [(i["id"], i["origin"]) for i in items] == [("c01", "curated"), ("q001", "synthetic")]
    assert set(items[1]) == {"id", "origin", "question", "reference_answer"}


def test_load_questions_skips_missing_files(tmp_path):
    assert generate_answers.load_questions(tmp_path / "none.json", tmp_path / "none2.json") == []


def test_shards_cover_every_row_exactly_once():
    rows = list(range(23))
    parts = [evaluate_ragas.shard_rows(rows, i, 6) for i in range(6)]
    assert sorted(x for p in parts for x in p) == rows
    assert max(len(p) for p in parts) - min(len(p) for p in parts) <= 1


def test_invalid_shard_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        evaluate_ragas.shard_rows([1, 2], 3, 3)


def test_summarize_reports_missing_scores_instead_of_dropping_them():
    rows = [
        {
            "faithfulness": "1.0",
            "answer_relevancy": "0.8",
            "context_precision": "",
            "context_recall": "nan",
        },
        {
            "faithfulness": "0.5",
            "answer_relevancy": "0.6",
            "context_precision": "1",
            "context_recall": "0.5",
        },
    ]
    s = merge_ragas.summarize(rows)
    assert s["faithfulness"] == {"mean": 0.75, "missing": 0}
    assert s["context_precision"] == {"mean": 1.0, "missing": 1}
    assert s["context_recall"] == {"mean": 0.5, "missing": 1}


def test_markdown_shows_judge_generator_and_unscored_counts():
    rows = [
        {
            "faithfulness": "1",
            "answer_relevancy": "1",
            "context_precision": "",
            "context_recall": "1",
        }
    ]
    table = merge_ragas.to_markdown([("All", rows)], "qwen2.5:7b", "llama3.2:3b")
    assert "qwen2.5:7b" in table and "llama3.2:3b" in table
    assert "n/a (1 unscored)" in table
    assert "| All | 1 |" in table
