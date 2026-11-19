#!/usr/bin/env python
"""End-to-end smoke test with NO API keys and NO database.

Runs the real pipeline (local ONNX embedder + cross-encoder reranker, hybrid
retrieval, grounding gate, extractive answers) over the tiny built-in sample
corpus in a throwaway index. Exit code 0 means everything is wired correctly.

The first run downloads about 90 MB of model weights from Hugging Face.
"""
from __future__ import annotations

import os
import sys
import tempfile

# Isolate the run: its own index, no LLM, no tracing. Set before importing ragsys.
os.environ.update({
    "INDEX_DIR": tempfile.mkdtemp(prefix="ragsys-smoke-"),
    "VECTOR_BACKEND": "memory",
    "LLM_PROVIDER": "none",
    "TRACING_ENABLED": "false",
})


def main() -> int:
    from ragsys.agent import AgenticRAG
    from ragsys.ingest import ingest_path
    from ragsys.ingest.sources import sample_corpus

    corpus = sample_corpus(dest=os.path.join(os.environ["INDEX_DIR"], "docs"))[0].parent
    stats = ingest_path(corpus)
    print(f"[smoke] ingest: {stats}")
    assert stats["chunks"] > 0 and stats["vector_count"] == stats["bm25_count"]

    agent = AgenticRAG()

    grounded = agent.answer("What net worth is required for an individual investment adviser?")
    print(f"[smoke] grounded -> mode={grounded.mode} top_relevance={grounded.sources[0].relevance}")
    print(f"        {grounded.answer.splitlines()[-1][:110]}")
    assert not grounded.refused, "a question the corpus answers was refused"
    assert grounded.citations, "expected at least one citation"
    assert "five lakh" in " ".join(grounded.contexts), "the right passage was not retrieved"

    off_topic = agent.answer("What is the price of Bitcoin today?")
    print(f"[smoke] off-topic -> mode={off_topic.mode}")
    assert off_topic.refused, "an out-of-scope question was not refused"

    empty = agent.answer("   ")
    assert empty.refused and empty.answer == "Please provide a question."

    print("\n[smoke] PASS: pipeline wired end to end.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"[smoke] FAIL: {exc}")
        sys.exit(1)
