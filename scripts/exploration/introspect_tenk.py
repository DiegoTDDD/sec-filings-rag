"""Introspect a TenK object to discover the real section access API in this version.

The docs described attributes (business_description, mda) that don't exist in the
installed edgartools 5.40.1. This lists what the object actually exposes so we can
fix the ingestion with the correct names instead of guessing.
"""
from edgar import Company, set_identity

set_identity("Diego Leonardo Alves de Andrade diegoleonardo380@gmail.com")


def main():
    filing = Company("AAPL").get_filings(form="10-K").latest(1)
    tenk = filing.obj()

    print("=== Public attributes / methods on TenK ===")
    members = [m for m in dir(tenk) if not m.startswith("_")]
    for m in members:
        print(f"  {m}")

    # Many edgartools report objects expose sections via .items or .sections.
    print("\n=== tenk.items (if present) ===")
    try:
        print(tenk.items)
    except Exception as exc:  # noqa: BLE001
        print(f"  not available: {exc}")

    print("\n=== tenk.sections (if present) ===")
    try:
        print(tenk.sections)
    except Exception as exc:  # noqa: BLE001
        print(f"  not available: {exc}")


if __name__ == "__main__":
    main()
