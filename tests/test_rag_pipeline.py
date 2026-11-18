"""RAG pipeline logic with the external services (Qdrant, Ollama, LLM) mocked."""

import rag


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_context_numbers_each_excerpt_and_carries_its_source(make_point):
    chunks = [
        make_point("Export controls limit sales to China.", ticker="NVDA"),
        make_point(
            "Suppliers are concentrated in Asia.",
            ticker="AAPL",
            form="10-Q",
            filing_date="2026-01-30",
            section="mda",
        ),
    ]
    context = rag.build_context(chunks)

    assert context.startswith("[1] (NVDA 10-K filed 2026-02-25 - risk_factors)")
    assert "[2] (AAPL 10-Q filed 2026-01-30 - mda)" in context
    assert "Export controls limit sales to China." in context
    assert "Suppliers are concentrated in Asia." in context


def test_sources_list_matches_context_numbering(make_point):
    chunks = [make_point("a", ticker="JPM"), make_point("b", ticker="AAPL")]
    lines = rag.format_sources(chunks).splitlines()
    assert lines == [
        "[1] JPM 10-K filed 2026-02-25 - risk_factors",
        "[2] AAPL 10-K filed 2026-02-25 - risk_factors",
    ]


def test_no_retrieved_chunks_returns_a_clear_message_without_calling_the_llm(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda question, ticker=None: [])

    def _must_not_run(*args, **kwargs):
        raise AssertionError("LLM called without context")

    monkeypatch.setattr(rag, "generate", _must_not_run)
    answer, chunks = rag.answer_question("Anything?")
    assert chunks == []
    assert "No relevant excerpts" in answer


def test_answer_question_passes_ticker_and_grounded_context(monkeypatch, make_point):
    seen = {}
    chunks = [make_point("Export controls limit sales to China.")]

    def fake_retrieve(question, ticker=None):
        seen["ticker"] = ticker
        return chunks

    def fake_generate(question, context):
        seen["context"] = context
        return "Grounded answer."

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "generate", fake_generate)

    answer, used = rag.answer_question("Export risk?", ticker="NVDA")
    assert answer == "Grounded answer."
    assert used == chunks
    assert seen["ticker"] == "NVDA"
    assert "Export controls limit sales to China." in seen["context"]


def test_generate_sends_grounding_prompt_and_low_temperature(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, body=json)
        return _Response({"choices": [{"message": {"content": "  Answer.  "}}]})

    monkeypatch.setattr(rag.httpx, "post", fake_post)
    result = rag.generate("Question?", "[1] (src)\ntext")

    assert result == "Answer."
    assert captured["url"].endswith("/chat/completions")
    messages = captured["body"]["messages"]
    assert messages[0] == {"role": "system", "content": rag.SYSTEM_PROMPT}
    assert "ONLY" in rag.SYSTEM_PROMPT
    assert "Question?" in messages[1]["content"] and "[1] (src)" in messages[1]["content"]
    assert captured["body"]["temperature"] <= 0.2


def test_retrieve_applies_ticker_filter_and_keeps_best_reranked(monkeypatch, make_point):
    hits = [make_point(f"chunk {i}") for i in range(8)]
    captured = {}

    class FakeClient:
        def __init__(self, url):
            pass

        def query_points(self, **kwargs):
            captured.update(kwargs)
            return type("Result", (), {"points": hits})()

    class FakeReranker:
        def predict(self, pairs):
            # Higher score for higher chunk number: the reranker reverses dense order.
            return [int(text.split()[-1]) for _, text in pairs]

    monkeypatch.setattr(rag, "QdrantClient", FakeClient)
    monkeypatch.setattr(rag, "embed", lambda text: [0.0, 1.0])
    monkeypatch.setattr(rag, "get_reranker", lambda: FakeReranker())

    result = rag.retrieve("question", ticker="AAPL", mode="dense")

    assert captured["query_filter"] is not None
    assert "AAPL" in repr(captured["query_filter"])
    assert captured["limit"] == rag.RETRIEVE_N
    assert [p.payload["text"] for p in result] == [f"chunk {i}" for i in (7, 6, 5, 4, 3)]


def test_retrieve_without_ticker_does_not_filter(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, url):
            pass

        def query_points(self, **kwargs):
            captured.update(kwargs)
            return type("Result", (), {"points": []})()

    monkeypatch.setattr(rag, "QdrantClient", FakeClient)
    monkeypatch.setattr(rag, "embed", lambda text: [0.0])

    assert rag.retrieve("question", mode="dense") == []
    assert captured["query_filter"] is None


def test_hybrid_merges_keyword_hits_before_reranking(monkeypatch, make_point):
    import strategies

    dense_hits = [make_point("dense only chunk")]
    dense_hits[0].payload.update(source_file="a.txt", chunk_index=0)
    keyword_chunk = {
        "text": "CET1 capital ratio chunk",
        "ticker": "JPM",
        "form": "10-K",
        "filing_date": "2026-02-14",
        "section": "mda",
        "source_file": "b.txt",
        "chunk_index": 3,
    }

    class FakeClient:
        def __init__(self, url):
            pass

        def query_points(self, **kwargs):
            return type("Result", (), {"points": dense_hits})()

    class FakeReranker:
        def predict(self, pairs):
            return [1.0 if "CET1" in text else 0.0 for _, text in pairs]

    monkeypatch.setattr(rag, "QdrantClient", FakeClient)
    monkeypatch.setattr(rag, "embed", lambda text: [0.0])
    monkeypatch.setattr(rag, "get_reranker", lambda: FakeReranker())
    monkeypatch.setattr(rag, "get_bm25", lambda: strategies.BM25Index([keyword_chunk]))

    result = rag.retrieve("What is the CET1 capital ratio?", mode="hybrid")
    assert [c.payload["text"] for c in result] == ["CET1 capital ratio chunk", "dense only chunk"]


def test_hybrid_falls_back_to_dense_without_chunks_file(monkeypatch, tmp_path, make_point):
    monkeypatch.setattr(rag, "CHUNKS_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(rag, "_bm25", None)
    assert rag.get_bm25() is None

    class FakeClient:
        def __init__(self, url):
            pass

        def query_points(self, **kwargs):
            return type("Result", (), {"points": [make_point("only dense")]})()

    class FakeReranker:
        def predict(self, pairs):
            return [0.5 for _ in pairs]

    monkeypatch.setattr(rag, "QdrantClient", FakeClient)
    monkeypatch.setattr(rag, "embed", lambda text: [0.0])
    monkeypatch.setattr(rag, "get_reranker", lambda: FakeReranker())
    assert [c.payload["text"] for c in rag.retrieve("q", mode="hybrid")] == ["only dense"]
