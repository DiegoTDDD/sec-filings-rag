"""Retrieval strategies and evaluation helpers (no Qdrant or Ollama needed)."""

import build_eval_set
import retrieval_eval
import strategies


def _chunk(text, ticker="NVDA", source_file="NVDA_10-K_risk.txt", index=0, section="risk_factors"):
    return {
        "text": text,
        "ticker": ticker,
        "form": "10-K",
        "filing_date": "2026-02-25",
        "section": section,
        "source_file": source_file,
        "chunk_index": index,
    }


def test_tokenize_lowercases_and_drops_stopwords():
    assert strategies.tokenize("What is the CET1 ratio of JPMorgan?") == [
        "cet1",
        "ratio",
        "jpmorgan",
    ]


def test_rrf_rewards_chunks_found_by_both_lists():
    a, b, c = _chunk("a", index=1), _chunk("b", index=2), _chunk("c", index=3)
    fused = strategies.rrf_fuse([[a, b], [c, b]], n=3)
    assert strategies.chunk_key(fused[0]) == strategies.chunk_key(b)
    assert {strategies.chunk_key(x) for x in fused} == {
        strategies.chunk_key(a),
        strategies.chunk_key(b),
        strategies.chunk_key(c),
    }


def test_rrf_respects_n_and_deduplicates():
    a, b = _chunk("a", index=1), _chunk("b", index=2)
    assert len(strategies.rrf_fuse([[a, b], [a, b], [b, a]], n=5)) == 2
    assert len(strategies.rrf_fuse([[a, b]], n=1)) == 1


def test_bm25_finds_exact_terms_and_filters_by_ticker():
    chunks = [
        _chunk("Export controls restrict sales of data center GPUs to China.", index=0),
        _chunk("Revenue from gaming products grew during the year.", index=1),
        _chunk(
            "Our CET1 capital ratio remained above regulatory minimums.",
            ticker="JPM",
            source_file="JPM_10-K_mda.txt",
        ),
        _chunk(
            "Apple depends on outsourcing partners in Asia.",
            ticker="AAPL",
            source_file="AAPL_10-K_risk.txt",
        ),
    ]
    index = strategies.BM25Index(chunks)
    assert index.search("CET1 capital ratio", n=1)[0]["ticker"] == "JPM"
    top = index.search("export controls China GPUs", n=2, ticker="NVDA")
    assert top[0]["chunk_index"] == 0
    assert all(c["ticker"] == "NVDA" for c in top)


def test_rank_of_strict_and_adjacent():
    results = [
        _chunk("x", index=7),
        _chunk("y", index=4),
        _chunk("z", source_file="other.txt", index=5),
    ]
    source = ("NVDA_10-K_risk.txt", 5)
    assert retrieval_eval.rank_of(results, source) is None
    assert retrieval_eval.rank_of(results, source, adjacent=True) == 2
    assert retrieval_eval.rank_of(results, ("NVDA_10-K_risk.txt", 7)) == 1


def test_summarize_metrics():
    summary = retrieval_eval.summarize([1, 2, None, 6], [True, True, False, True])
    assert summary["hit@1"] == 0.25
    assert summary["hit@5"] == 0.5
    assert summary["mrr@5"] == (1 + 0.5) / 4
    assert summary["recall@20"] == 0.75


def test_summarize_empty_set():
    assert retrieval_eval.summarize([], [])["hit@5"] == 0.0


def test_markdown_table_has_every_strategy():
    blank = {"hit@1": 0.1, "hit@5": 0.2, "mrr@5": 0.3, "recall@20": 0.4}
    results = {k: {"strict": blank, "adjacent": blank} for k, _ in retrieval_eval.STRATEGIES}
    table = retrieval_eval.to_markdown(results, 12)
    assert "12 synthetic questions" in table
    for _, label in retrieval_eval.STRATEGIES:
        assert label in table


def test_stratified_sample_balances_groups_and_is_reproducible():
    chunks = [
        _chunk("x" * 700, ticker=t, source_file=f"{t}_{s}.txt", index=i, section=s)
        for t in ("AAPL", "NVDA", "JPM")
        for s in ("risk_factors", "mda")
        for i in range(10)
    ]
    chunks.append(_chunk("too short", ticker="AAPL", source_file="short.txt"))
    sample = build_eval_set.stratified_sample(chunks, 12)
    assert len(sample) == 12
    assert all(len(c["text"]) >= build_eval_set.MIN_CHUNK_CHARS for c in sample)
    groups = {(c["ticker"], c["section"]) for c in sample}
    assert len(groups) == 6  # every company/section pair represented
    again = build_eval_set.stratified_sample(chunks, 12)
    assert [strategies.chunk_key(c) for c in again] == [strategies.chunk_key(c) for c in sample]


def test_generated_question_validation():
    good = {
        "question": "How does NVIDIA describe the impact of export controls on China sales?",
        "reference_answer": "NVIDIA says export controls limit its data center sales to China.",
    }
    assert build_eval_set.is_valid(good, "NVDA")
    assert not build_eval_set.is_valid(good, "AAPL")  # names the wrong company
    leak = dict(good, question="According to the excerpt, how does NVIDIA view export controls?")
    assert not build_eval_set.is_valid(leak, "NVDA")
    no_mark = dict(good, question="How does NVIDIA describe export controls")
    assert not build_eval_set.is_valid(no_mark, "NVDA")
    empty_answer = dict(good, reference_answer="")
    assert not build_eval_set.is_valid(empty_answer, "NVDA")
