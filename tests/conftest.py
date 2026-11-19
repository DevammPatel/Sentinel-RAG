"""Shared test setup.

The suite is fully offline: no model downloads, no network, no API keys. Real
models are replaced by a hashing embedder and a pass-through reranker.
"""
import math
import os
import tempfile
import zlib

# Must be set before `ragsys` is imported, because settings load at import time.
# Explicit values also stop a developer's local .env from leaking into tests.
os.environ.update({
    "TRACING_ENABLED": "false",
    "WARM_ON_STARTUP": "false",
    "LLM_PROVIDER": "none",
    "GROQ_API_KEY": "",
    "GEMINI_API_KEY": "",
    "OPENAI_API_KEY": "",
    "ANTHROPIC_API_KEY": "",
    "ADMIN_TOKEN": "",
    "VECTOR_BACKEND": "memory",
    "INDEX_DIR": tempfile.mkdtemp(prefix="ragsys-test-index-"),
    "MIN_CONTEXT_SCORE": "0.3",
})

import pytest  # noqa: E402

from ragsys.ingest.chunker import chunk_document  # noqa: E402
from ragsys.models import Document  # noqa: E402
from ragsys.retrieval.hybrid import HybridRetriever  # noqa: E402
from ragsys.retrieval.rerank import Reranker  # noqa: E402
from ragsys.store import BM25Store, InMemoryVectorStore  # noqa: E402


class FakeEmbedder:
    """Hashing bag-of-words embedder: deterministic, no downloads."""

    dim = 64

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def embed_one(self, text):
        return self._vec(text)

    def _vec(self, text):
        v = [0.0] * self.dim
        for w in text.lower().split():
            v[zlib.crc32(w.strip(".,?;:()").encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


class NoopReranker(Reranker):
    """Keeps fusion order and assigns no rerank score."""

    def __init__(self):
        self.model_name = "noop"
        self.kind = "noop"
        self._impl = None


KYC_DOC = Document(
    doc_id="rbi-kyc", title="KYC Direction", source="RBI",
    url="https://example.org/kyc", circular_no="RBI/2025-26/1",
    text=("1. Periodic KYC updation shall be carried out once in every ten years for low "
          "risk customers and every two years for high risk customers.\n\n"
          "2. Enhanced due diligence applies to politically exposed persons."),
)


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def stores(tmp_path, embedder):
    """An in-memory vector store + BM25 index seeded with one small document."""
    vs = InMemoryVectorStore(index_dir=tmp_path / "index")
    chunks = chunk_document(KYC_DOC, size=300, overlap=40)
    vs.upsert(chunks, embedder.embed([c.search_text for c in chunks]))
    return vs, BM25Store(vs.all_chunks())


@pytest.fixture
def retriever(stores, embedder):
    vs, bm = stores
    return HybridRetriever(vector_store=vs, bm25_store=bm, embedder=embedder)
