"""Ingestion orchestration: files -> Documents -> Chunks -> (dense + sparse) indexes."""
from __future__ import annotations

from pathlib import Path

from ragsys.ingest.chunker import chunk_document
from ragsys.ingest.loader import load_any
from ragsys.llm import get_embedder
from ragsys.models import Chunk, Document
from ragsys.store import BM25Store, get_vector_store
from ragsys.telemetry import SpanKind, span


def ingest_documents(
    docs: list[Document],
    vector_store=None,
    bm25_store: BM25Store | None = None,
) -> dict:
    vector_store = vector_store or get_vector_store()
    bm25_store = bm25_store or BM25Store()
    embedder = get_embedder()

    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))

    with span("ingest", kind=SpanKind.CHAIN,
              **{"ingest.docs": len(docs), "ingest.chunks": len(all_chunks)}):
        if all_chunks:
            vectors = embedder.embed([c.text for c in all_chunks])
            vector_store.upsert(all_chunks, vectors)
            bm25_store.build(all_chunks)

    return {
        "documents": len(docs),
        "chunks": len(all_chunks),
        "vector_count": vector_store.count(),
        "bm25_count": bm25_store.count(),
    }


def ingest_path(
    path: str | Path,
    vector_store=None,
    bm25_store: BM25Store | None = None,
) -> dict:
    """Ingest a single file or every supported file in a directory."""
    path = Path(path)
    files: list[Path]
    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"}
        )
    else:
        files = [path]
    docs = [load_any(f) for f in files]
    return ingest_documents(docs, vector_store, bm25_store)
