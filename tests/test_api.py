"""HTTP layer: health check, ticker normalization and response shape."""
import pytest
from fastapi.testclient import TestClient

from api import main as api_main


@pytest.fixture
def client(monkeypatch, make_point):
    calls = []

    def fake_answer(question, ticker=None):
        calls.append({"question": question, "ticker": ticker})
        return "Grounded answer.", [make_point("text", ticker=ticker or "JPM")]

    monkeypatch.setattr(api_main.rag, "answer_question", fake_answer)
    test_client = TestClient(api_main.app)
    test_client.calls = calls
    return test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_and_sources(client):
    response = client.post("/ask", json={"question": "Export risk?", "ticker": "NVDA"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Grounded answer."
    assert body["ticker"] == "NVDA"
    assert body["sources"] == [
        {"ticker": "NVDA", "form": "10-K", "filing_date": "2026-02-25", "section": "risk_factors"}
    ]


def test_lowercase_ticker_is_normalized(client):
    client.post("/ask", json={"question": "q", "ticker": "aapl"})
    assert client.calls[-1]["ticker"] == "AAPL"


def test_unknown_ticker_falls_back_to_all_companies(client):
    response = client.post("/ask", json={"question": "q", "ticker": "TSLA"})
    assert response.json()["ticker"] is None
    assert client.calls[-1]["ticker"] is None


def test_missing_question_is_rejected(client):
    response = client.post("/ask", json={"ticker": "AAPL"})
    assert response.status_code == 422
