"""Ingestion helpers: file naming and skipping cross-reference stubs."""
import ingest


def test_clean_filename_is_filesystem_safe_and_sortable():
    name = ingest.clean_filename("AAPL", "10-K", "2025-10-31", "Risk Factors & MD")
    assert name == "AAPL_10-K_2025-10-31_risk_factors_and_md.txt"
    assert " " not in name and "&" not in name


def test_short_section_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "PROCESSED_DIR", tmp_path)
    records = []
    ingest.save_section(records, "AAPL", "10-Q", "2026-01-30", "risk_factors",
                        "See the risk factors in our annual report.")
    assert records == []
    assert list(tmp_path.iterdir()) == []


def test_empty_section_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "PROCESSED_DIR", tmp_path)
    records = []
    ingest.save_section(records, "AAPL", "10-Q", "2026-01-30", "mda", None)
    assert records == []


def test_real_section_is_written_with_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "PROCESSED_DIR", tmp_path)
    records = []
    text = "  " + "Demand for data center products grew. " * 40 + "  "
    ingest.save_section(records, "NVDA", "10-K", "2026-02-25", "mda", text)

    assert len(records) == 1
    record = records[0]
    assert record["ticker"] == "NVDA"
    assert record["form"] == "10-K"
    assert record["filing_date"] == "2026-02-25"
    assert record["section"] == "mda"
    assert record["char_count"] == len(text.strip())

    written = (tmp_path / record["filename"]).read_text(encoding="utf-8")
    assert written == text.strip()


def test_minimum_length_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "PROCESSED_DIR", tmp_path)
    records = []
    exactly_min = "a" * ingest.MIN_SECTION_CHARS
    just_below = "a" * (ingest.MIN_SECTION_CHARS - 1)
    ingest.save_section(records, "JPM", "10-K", "2026-02-14", "business", just_below)
    ingest.save_section(records, "JPM", "10-K", "2026-02-14", "mda", exactly_min)
    assert [r["section"] for r in records] == ["mda"]
