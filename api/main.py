"""FastAPI service exposing the SEC-filings RAG pipeline over HTTP.

Endpoints:
  GET  /health -> simple liveness check
  POST /ask    -> { "question": "...", "ticker": "AAPL"? } returns the generated
                  answer plus the source chunks used. The optional ticker filters
                  retrieval to a single company via Qdrant metadata filtering.

The pipeline itself lives in src/retrieval/rag.py; this module only wraps it in
an HTTP layer. The cross-encoder re-ranker is loaded once at startup and reused.

Run:
    uvicorn api.main:app --port 8000
"""
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "retrieval"))
import rag  # noqa: E402

app = FastAPI(
    title="SEC Filings RAG API",
    description="Ask questions about AAPL, NVDA and JPM 10-K/10-Q filings with cited sources.",
    version="1.0.0",
)

VALID_TICKERS = {"AAPL", "NVDA", "JPM"}


class AskRequest(BaseModel):
    question: str
    ticker: Optional[str] = None  # optional company filter: AAPL, NVDA or JPM


class Source(BaseModel):
    ticker: str
    form: str
    filing_date: str
    section: str


class AskResponse(BaseModel):
    question: str
    ticker: Optional[str]
    answer: str
    sources: list[Source]


@app.on_event("startup")
def warm_up():
    """Load the cross-encoder once at startup so the first request isn't slow."""
    rag.get_reranker()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    # Normalize/validate the optional ticker; ignore anything unexpected.
    ticker = request.ticker.upper() if request.ticker else None
    if ticker not in VALID_TICKERS:
        ticker = None

    answer, chunks = rag.answer_question(request.question, ticker=ticker)
    sources = [
        Source(
            ticker=c.payload["ticker"],
            form=c.payload["form"],
            filing_date=c.payload["filing_date"],
            section=c.payload["section"],
        )
        for c in chunks
    ]
    return AskResponse(
        question=request.question, ticker=ticker, answer=answer, sources=sources
    )
