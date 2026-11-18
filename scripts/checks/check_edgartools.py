"""Minimal edgartools test: pull clean Risk Factors text from Apple's latest 10-K.

This validates that edgartools returns readable prose (not the iXBRL noise we saw
with raw BeautifulSoup) before we rebuild the whole ingestion around it.
"""
from edgar import Company, set_identity

# The SEC requires identifying yourself. edgartools uses set_identity for this.
set_identity("Diego Leonardo Alves de Andrade diegoleonardo380@gmail.com")


def main():
    # Grab Apple's most recent 10-K.
    filing = Company("AAPL").get_filings(form="10-K").latest(1)
    print(f"Filing: {filing.form} filed {filing.filing_date} (accession {filing.accession_no})\n")

    # Convert to a structured TenK object and access the Risk Factors section.
    tenk = filing.obj()
    risk_text = tenk.risk_factors

    print(f"Risk Factors length: {len(risk_text):,} characters\n")
    print("First 800 characters of Risk Factors:")
    print("-" * 60)
    print(risk_text[:800])
    print("-" * 60)


if __name__ == "__main__":
    main()
