# 0002. Qdrant as the vector store

**Status:** Accepted

## Context
About 3,200 chunks from nine filings (AAPL, NVDA, JPM). Questions are often about one company, and JPMorgan's filings are much longer than the others, so unfiltered search tends to fill the shortlist with JPM chunks. Every result must also carry its provenance (ticker, form, filing date, section) to be cited.

## Decision
Use Qdrant (Docker, cosine distance, 768 dimensions). Each point's payload holds the chunk text and its provenance, and an optional `ticker` filter is applied inside the search.

## Alternatives considered
- **FAISS.** Fast, but no payload storage or filtering; metadata would live in a side table and filtering would happen after the search, which loses results when one company dominates.
- **Chroma.** Simple, with metadata filters, but less mature server mode and tooling at the time.
- **pgvector.** Good if Postgres were already in the stack; it is not.

## Consequences
- Filtering happens before ranking, so a company-specific question always gets five chunks from that company.
- The payload doubles as the citation, so no join is needed at answer time.
- One more service to run (Docker), which is also why the public demo cannot run the live pipeline (see [0005](0005-separate-public-demo.md)).
