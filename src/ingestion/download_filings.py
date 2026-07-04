"""Download SEC filings (10-K and 10-Q) for the target companies from EDGAR.

For each company we fetch its submission history from the SEC, select the two
most recent 10-K filings (for a year-over-year axis) and the single most recent
10-Q, then download each primary document as HTML into data/raw/.

The SEC allows up to 10 requests/second and requires a descriptive User-Agent
with real contact info. We stay well under the limit with a small delay.
"""
import time
from pathlib import Path

import requests

# --- Configuration -------------------------------------------------------

HEADERS = {
    "User-Agent": "Diego Leonardo Alves de Andrade diegoleonardo380@gmail.com"
}

# ticker -> zero-padded 10-digit CIK (resolved earlier via test_edgar.py)
COMPANIES = {
    "AAPL": "0000320193",
    "NVDA": "0001045810",
    "JPM": "0000019617",
}

# How many of each form to keep, per company.
N_10K = 2  # current + prior year (year-over-year comparison)
N_10Q = 1  # most recent quarter

RAW_DIR = Path("data/raw")
REQUEST_DELAY = 0.3  # seconds between requests (well under SEC's 10 req/s limit)


# --- Helpers -------------------------------------------------------------

def get_submissions(cik):
    """Fetch the full submission history JSON for one company."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return response.json()


def select_filings(submissions):
    """Pick the N most recent 10-K and 10-Q filings from a submissions payload.

    EDGAR stores recent filings as parallel arrays under recent. We zip them
    into per-filing records, then filter and slice by form type. Records are
    already newest-first, so simple slicing gives the most recent ones.
    """
    recent = submissions["filings"]["recent"]
    records = []
    for form, date, accession, primary_doc in zip(
        recent["form"],
        recent["filingDate"],
        recent["accessionNumber"],
        recent["primaryDocument"],
    ):
        records.append(
            {
                "form": form,
                "date": date,
                "accession": accession,
                "primary_doc": primary_doc,
            }
        )

    tens_k = [r for r in records if r["form"] == "10-K"][:N_10K]
    tens_q = [r for r in records if r["form"] == "10-Q"][:N_10Q]
    return tens_k + tens_q


def build_document_url(cik, accession, primary_doc):
    """Construct the URL of a filing's primary HTML document.

    The Archives path uses the CIK without leading zeros and the accession
    number with its dashes removed.
    """
    cik_int = int(cik)  # strips leading zeros
    accession_nodashes = accession.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{accession_nodashes}/{primary_doc}"
    )


def download_document(url):
    """Download a filing document and return its raw HTML text."""
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return response.text


# --- Main ----------------------------------------------------------------

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    for ticker, cik in COMPANIES.items():
        print(f"\n=== {ticker} (CIK {cik}) ===")
        submissions = get_submissions(cik)
        filings = select_filings(submissions)

        for filing in filings:
            url = build_document_url(cik, filing["accession"], filing["primary_doc"])
            filename = f"{ticker}_{filing['form']}_{filing['date']}.html"
            out_path = RAW_DIR / filename

            print(f"  downloading {filing['form']} ({filing['date']}) ...", end=" ")
            try:
                html = download_document(url)
                out_path.write_text(html, encoding="utf-8")
                size_kb = len(html) / 1024
                print(f"OK  ->  {filename}  ({size_kb:,.0f} KB)")
                total += 1
            except requests.HTTPError as exc:
                print(f"FAILED ({exc})")

    print(f"\nDone. Downloaded {total} filings into {RAW_DIR}/")


if __name__ == "__main__":
    main()
