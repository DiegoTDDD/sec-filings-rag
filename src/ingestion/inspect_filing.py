"""Inspect one raw filing to understand its structure before writing the extractor.

Filings vary in HTML structure between companies. Rather than guess, this script
strips a filing down to plain text and reports basic stats plus a preview, and
searches for the section headers we care about (Item 1A Risk Factors, Item 7
MD&A, Item 1 Business) so we can see how they actually appear in the document.
"""
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")

# Section markers we plan to extract, by the standard SEC "Item" labels.
SECTION_PATTERNS = {
    "Item 1 - Business": r"item\s*1[^0-9a-z]",
    "Item 1A - Risk Factors": r"item\s*1a",
    "Item 7 - MD&A": r"item\s*7[^0-9a-z]",
}


def html_to_text(html):
    """Strip HTML to plain text, dropping script/style and collapsing whitespace."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)  # collapse runs of whitespace
    return text.strip()


def main():
    # Default to the first file in data/raw unless one is passed as an argument.
    if len(sys.argv) > 1:
        target = RAW_DIR / sys.argv[1]
    else:
        target = sorted(RAW_DIR.glob("*.html"))[0]

    print(f"Inspecting: {target.name}\n")
    html = target.read_text(encoding="utf-8")
    text = html_to_text(html)

    print(f"Plain-text length: {len(text):,} characters\n")
    print("First 500 characters of plain text:")
    print("-" * 60)
    print(text[:500])
    print("-" * 60)

    print("\nSection header occurrences (case-insensitive):")
    lowered = text.lower()
    for label, pattern in SECTION_PATTERNS.items():
        matches = [m.start() for m in re.finditer(pattern, lowered)]
        preview = f"{len(matches)} match(es)"
        if matches:
            preview += f"; first at char {matches[0]:,}"
        print(f"  {label:28} -> {preview}")


if __name__ == "__main__":
    main()
