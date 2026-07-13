"""Agent-level tests that avoid any network / model calls.

We exercise query rewriting, the refusal path, and hybrid retrieval wiring using
the in-memory stores plus a tiny fake embedder — so these run in CI with no keys.
"""
from ragsys.agent.orchestrator import AgenticRAG, rewrite_query
from ragsys.ingest.chunker import chunk_document
from ragsys.models import Document
from ragsys.retrieval.hybrid import HybridRetriever
from ragsys.store import BM25Store
from ragsys.store.vector_store import InMemoryVectorStore


def test_rewrite_expands_known_acronyms():
    out = rewrite_query("What are the KYC updation rules?")
    assert "know your customer" in out.lower()


def test_rewrite_noop_when_no_acronym():
    q = "What net worth is required?"
    assert rewrite_query(q) == q


class _FakeEmbedder:
    """Hashing bag-of-words embedder — deterministic, no downloads."""

    dim = 32

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def embed_one(self, text):
        return self._vec(text)

    def _vec(self, text):
        import math
        v = [0.0] * self.dim
        for w in text.lower().split():
            v[hash(w) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


def _seed_stores(tmp_path):
    vs = InMemoryVectorStore(path=tmp_path / "vec.pkl")
    bm = BM25Store(path=tmp_path / "bm25.pkl")
    emb = _FakeEmbedder()
    doc = Document(
        doc_id="d1", title="KYC Direction", source="RBI",
        text=("1. Periodic KYC updation shall be carried out once in every ten "
              "years for low risk customers and every two years for high risk "
              "customers.\n\n2. Enhanced due diligence applies to politically "
              "exposed persons."),
    )
    chunks = chunk_document(doc, size=300, overlap=40)
    vs.upsert(chunks, emb.embed([c.text for c in chunks]))
    bm.build(chunks)
    return vs, bm, emb


def test_hybrid_retrieval_finds_relevant_chunk(tmp_path):
    vs, bm, emb = _seed_stores(tmp_path)
    retriever = HybridRetriever(vector_store=vs, bm25_store=bm)
    retriever.embedder = emb
    hits = retriever.retrieve("KYC updation for low risk customers", top_k=5)
    assert hits, "expected retrieval hits"
    assert any("ten years" in h.chunk.text for h in hits)


def test_agent_refuses_empty_question():
    agent = AgenticRAG(retriever=None) if False else None  # doc: see below
    # Build with in-memory stores so no model loads on construction is needed.
    from ragsys.retrieval.rerank import Reranker

    class _NoopRerank(Reranker):
        def __init__(self):
            self._kind = "noop"
            self.model_name = "noop"

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path
        vs, bm, emb = _seed_stores(Path(d))
        retriever = HybridRetriever(vector_store=vs, bm25_store=bm)
        retriever.embedder = emb
        agent = AgenticRAG(retriever=retriever, reranker=_NoopRerank())
        resp = agent.answer("   ")
        assert resp.refused is True
