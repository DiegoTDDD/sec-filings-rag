"""Shared test setup.

The pipeline modules import heavy runtime dependencies at module level
(qdrant_client, sentence_transformers/PyTorch, edgartools). The unit tests only
exercise pure logic, so those packages are replaced with lightweight stand-ins
before the modules are imported. This keeps the test job fast and lets it run
without Qdrant, Ollama, model downloads or network access.
"""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for sub in ("src/indexing", "src/ingestion", "src/retrieval", "src/evaluation"):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules.setdefault(name, module)
    return sys.modules[name]


class _Unavailable:
    """Raises if a test accidentally reaches a real external service."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("External service used in a unit test")


_stub("qdrant_client", QdrantClient=_Unavailable)
_stub(
    "qdrant_client.models",
    FieldCondition=lambda **kw: ("FieldCondition", kw),
    Filter=lambda **kw: ("Filter", kw),
    MatchValue=lambda **kw: ("MatchValue", kw),
)
_stub("sentence_transformers", CrossEncoder=_Unavailable)
_stub("edgar", Company=_Unavailable, set_identity=lambda identity: None)


class FakePoint:
    """Mimics a Qdrant ScoredPoint: only the payload is used by the pipeline."""

    def __init__(self, text, ticker="NVDA", form="10-K", filing_date="2026-02-25",
                 section="risk_factors"):
        self.payload = {
            "text": text,
            "ticker": ticker,
            "form": form,
            "filing_date": filing_date,
            "section": section,
        }


@pytest.fixture
def make_point():
    return FakePoint
