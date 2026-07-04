"""Minimal EDGAR connectivity test: resolve tickers to CIK numbers.

Run this first to confirm the SEC EDGAR API responds for your machine
before building the full download pipeline. It downloads the official
SEC ticker-to-CIK mapping and prints the CIK for each target company.
"""
import requests

# The SEC requires a descriptive User-Agent that includes real contact info.
# Requests without it (or with obviously fake contacts) get blocked (403).
HEADERS = {
    "User-Agent": "Diego Leonardo Alves de Andrade diegoleonardo380@gmail.com"
}

TICKERS = ["AAPL", "NVDA", "JPM"]


def load_ticker_map():
    """Download the SEC's official ticker-to-CIK mapping as a dict keyed by ticker."""
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    data = response.json()
    # The JSON is a dict of numbered records; re-key it by ticker for lookups.
    return {row["ticker"]: row for row in data.values()}


def main():
    ticker_map = load_ticker_map()
    print("EDGAR connectivity OK. Resolved CIK numbers:\n")
    for ticker in TICKERS:
        record = ticker_map[ticker]
        # A CIK must be zero-padded to 10 digits for later EDGAR API calls.
        cik = str(record["cik_str"]).zfill(10)
        print(f"  {ticker:6} -> CIK {cik} -> {record['title']}")


if __name__ == "__main__":
    main()
