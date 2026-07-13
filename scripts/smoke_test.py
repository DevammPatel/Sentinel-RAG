#!/usr/bin/env python
"""End-to-end wiring smoke test with NO API keys and NO Postgres.

Uses the in-memory vector store, BM25, a fake embedder, and a stub LLM so we can
prove ingest -> hybrid retrieve -> rerank -> guardrail -> generate works before
you plug in real keys/services. Exit code 0 means the pipeline is wired.
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path


class FakeEmbedder:
    dim = 64

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def embed_one(self, text):
        return self._vec(text)

    def _vec(self, text):
        v = [0.0] * self.dim
        for w in text.lower().split():
            v[hash(w) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


def main() -> int:
    from ragsys.agent.orchestrator import AgenticRAG
    from ragsys.generation.generator import Generator
    from ragsys.ingest.sources import sample_corpus
    from ragsys.ingest.pipeline import ingest_path
    from ragsys.llm.provider import LLMResult
    from ragsys.retrieval.hybrid import HybridRetriever
    from ragsys.retrieval.rerank import Reranker
    from ragsys.store import BM25Store
    from ragsys.store.vector_store import InMemoryVectorStore

    tmp = Path(tempfile.mkdtemp())
    emb = FakeEmbedder()
    vs = InMemoryVectorStore(path=tmp / "vec.pkl")
    bm = BM25Store(path=tmp / "bm25.pkl")

    # 1. Ingest the offline sample corpus using the fake embedder.
    corpus_dir = sample_corpus(dest=tmp / "pdfs")[0].parent

    # monkeypatch the embedder used by the ingest pipeline
    import ragsys.ingest.pipeline as pipe
    pipe.get_embedder = lambda: emb  # type: ignore

    stats = ingest_path(corpus_dir, vector_store=vs, bm25_store=bm)
    print(f"[smoke] ingest: {stats}")
    assert stats["chunks"] > 0, "no chunks ingested"

    # 2. Build retriever with the fake embedder.
    retriever = HybridRetriever(vector_store=vs, bm25_store=bm)
    retriever.embedder = emb

    class NoopRerank(Reranker):
        def __init__(self):
            self._kind = "noop"
            self.model_name = "noop"

    # 3. Stub the LLM so no key is needed — echo the first citation.
    class StubGenerator(Generator):
        def __init__(self, provider=None):
            self.provider = provider

        def generate(self, question, contexts, injection_flagged=False):
            from ragsys.generation.generator import GenerationOutput, _format_context
            _, citations = _format_context(contexts)
            return GenerationOutput(
                answer=f"Based on the circulars {citations[0].marker}, "
                       f"the relevant provision addresses: {question}",
                citations=citations[:1],
                provider="stub",
                model="stub-model",
            )

    import ragsys.agent.orchestrator as orch
    orch.Generator = StubGenerator  # type: ignore

    agent = AgenticRAG(retriever=retriever, reranker=NoopRerank())

    # 4. Ask a grounded question.
    resp = agent.answer("What net worth is required for an individual investment adviser?")
    print(f"[smoke] answer: {resp.answer[:120]}")
    print(f"[smoke] refused={resp.refused} provider={resp.provider} "
          f"latency_ms={resp.latency_ms} contexts={len(resp.contexts)}")
    assert not resp.refused, "grounded question was wrongly refused"
    assert resp.citations, "expected at least one citation"

    # 5. Ask an out-of-scope question -> should refuse.
    resp2 = agent.answer("What is the current price of Bitcoin?")
    print(f"[smoke] out-of-scope refused={resp2.refused}")

    print("\n[smoke] PASS — pipeline wired end to end.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"[smoke] FAIL — {exc}")
        sys.exit(1)
