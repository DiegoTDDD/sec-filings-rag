"""Ingest SEC filings into clean, per-section text files using edgartools.

For each target company we take the two most recent 10-K filings (year-over-year
axis) and the most recent 10-Q. From each 10-K we extract Business, Risk Factors
and MD&A; from each 10-Q we extract MD&A and Risk Factors (a 10-Q has no Business
section). Every extracted section is written to data/processed/ as UTF-8 text,
and a metadata.json records provenance for the indexing step to consume.

Section access differs by form (attribute names verified against edgartools 5.40.1):
  - 10-K (TenK): tenk.business / tenk.risk_factors / tenk.management_discussion
  - 10-Q (TenQ): part-qualified key -> tenq["Part I, Item 2"]  (MD&A),
                                       tenq["Part II, Item 1A"] (Risk Factors)

Short sections (e.g. a 10-Q Risk Factors that merely cross-references the 10-K)
fall below MIN_SECTION_CHARS and are skipped to keep the index clean.
"""
import json
from pathlib import Path

from edgar import Company, set_identity

set_identity("Diego Leonardo Alves de Andrade diegoleonardo380@gmail.com")

COMPANIES = ["AAPL", "NVDA", "JPM"]
N_10K = 2  # current + prior year
N_10Q = 1  # most recent quarter

PROCESSED_DIR = Path("data/processed")
MIN_SECTION_CHARS = 1000  # skip cross-reference stubs and empty sections


def clean_filename(ticker, form, date, section):
    """Build a filesystem-safe, sortable name like AAPL_10-K_2025-10-31_risk_factors.txt."""
    safe_section = section.lower().replace(" ", "_").replace("&", "and")
    return f"{ticker}_{form}_{date}_{safe_section}.txt"


def save_section(records, ticker, form, date, section, text):
    """Write one section to disk if it has real content, and log its metadata."""
    if not text or len(text.strip()) < MIN_SECTION_CHARS:
        length = len(text.strip()) if text else 0
        print(f"    - {section:20} skipped (only {length} chars)")
        return
    filename = clean_filename(ticker, form, date, section)
    path = PROCESSED_DIR / filename
    path.write_text(text.strip(), encoding="utf-8")
    records.append(
        {
            "ticker": ticker,
            "form": form,
            "filing_date": str(date),
            "section": section,
            "filename": filename,
            "char_count": len(text.strip()),
        }
    )
    print(f"    - {section:20} OK ({len(text.strip()):,} chars) -> {filename}")


def ingest_10k(records, ticker, filing):
    """Extract Business, Risk Factors and MD&A from a 10-K filing."""
    date = filing.filing_date
    print(f"  10-K {date}")
    tenk = filing.obj()
    for section, getter in (
        ("business", lambda: tenk.business),
        ("risk_factors", lambda: tenk.risk_factors),
        ("mda", lambda: tenk.management_discussion),
    ):
        try:
            text = getter()
        except Exception as exc:  # noqa: BLE001 - real filings vary; keep going
            print(f"    - {section:20} error: {exc}")
            continue
        save_section(records, ticker, "10-K", date, section, text)


def ingest_10q(records, ticker, filing):
    """Extract MD&A and Risk Factors from a 10-Q using part-qualified keys."""
    date = filing.filing_date
    print(f"  10-Q {date}")
    tenq = filing.obj()
    for section, key in (
        ("mda", "Part I, Item 2"),
        ("risk_factors", "Part II, Item 1A"),
    ):
        try:
            text = tenq[key]
        except Exception as exc:  # noqa: BLE001
            print(f"    - {section:20} error: {exc}")
            continue
        save_section(records, ticker, "10-Q", date, section, text)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for ticker in COMPANIES:
        print(f"\n=== {ticker} ===")
        company = Company(ticker)

        tenk_filings = company.get_filings(form="10-K").head(N_10K)
        for filing in tenk_filings:
            ingest_10k(records, ticker, filing)

        tenq_filings = company.get_filings(form="10-Q").head(N_10Q)
        for filing in tenq_filings:
            ingest_10q(records, ticker, filing)

    metadata_path = PROCESSED_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nDone. Wrote {len(records)} sections + metadata.json into {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
