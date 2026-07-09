"""Create a compatibility shim so RAGAS can import on modern langchain-community.

RAGAS (both 0.3.x and 0.4.x) imports ChatVertexAI from the old path
  langchain_community.chat_models.vertexai
but that module was removed from modern langchain-community (ChatVertexAI moved to
the separate langchain-google-vertexai package). Since this project uses a local
Ollama judge and never touches Vertex AI, we don't need the real class - we only
need the import to succeed.

This script writes a tiny stub module at the expected path that exposes a dummy
ChatVertexAI, satisfying RAGAS's import without pulling in Google dependencies.
Run once; it's idempotent.
"""
import os
from pathlib import Path

import langchain_community

# Locate the chat_models package inside the installed langchain_community.
base = Path(langchain_community.__file__).parent / "chat_models"
target = base / "vertexai.py"

stub = '''"""Compatibility stub: ChatVertexAI moved to langchain-google-vertexai.

This project never uses Vertex AI (the judge is a local Ollama model). This stub
exists only so imports of `langchain_community.chat_models.vertexai.ChatVertexAI`
that remain in some libraries do not fail. Instantiating it raises clearly.
"""


class ChatVertexAI:  # noqa: D401 - minimal stub
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "ChatVertexAI is a stub. Install langchain-google-vertexai to use it."
        )
'''

if target.exists():
    print(f"Stub already present at {target}")
else:
    target.write_text(stub, encoding="utf-8")
    print(f"Wrote compatibility stub to {target}")

print("Done. Try importing ragas now.")
