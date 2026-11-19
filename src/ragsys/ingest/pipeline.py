"""Ingestion orchestration: files -> Documents -> Chunks -> (dense + sparse) indexes."""
from __future__ import annotations

from pathlib import Path

from ragsys.ingest.chunker import chunk_document
from ragsys.ingest.loader import SUPPORTED_SUFFIXES, load_any, load_manifest
from ragsys.models import Chunk, Document
from ragsys.telemetry import SpanKind, span

_EMBED_BATCH = 64


def ingest_documents(
    docs: list[Document],
    vector_store=None,
    bm25_store=None,
    embedder=None,
    verbose: bool = False,
) -> dict:
    """Chunk, embed and index `docs`. Re-ingesting a document replaces it."""
    from ragsys.llm import get_embedder
    from ragsys.store import get_bm25_store, get_vector_store

    vector_store = vector_store or get_vector_store()
    bm25_store = bm25_store or get_bm25_store()
    embedder = embedder or get_embedder()

    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))

    with span("ingest", kind=SpanKind.CHAIN,
              **{"ingest.docs": len(docs), "ingest.chunks": len(all_chunks)}):
        vectors: list[list[float]] = []
        for start in range(0, len(all_chunks), _EMBED_BATCH):
            batch = all_chunks[start : start + _EMBED_BATCH]
            vectors.extend(embedder.embed([c.search_text for c in batch]))
            if verbose:
                print(f"[ingest] embedded {min(start + _EMBED_BATCH, len(all_chunks))}"
                      f"/{len(all_chunks)} chunks", flush=True)
        # Drop the previous version of each document first so a shorter re-ingest
        # doesn't leave orphaned trailing chunks behind.
        vector_store.delete_docs([d.doc_id for d in docs])
        if all_chunks:
            vector_store.upsert(all_chunks, vectors)
        bm25_store.rebuild(vector_store.all_chunks())

    return {
        "documents": len(docs),
        "chunks": len(all_chunks),
        "vector_count": vector_store.count(),
        "bm25_count": bm25_store.count(),
    }


def load_path(path: str | Path) -> list[Document]:
    """Load a single file or every supported file in a directory."""
    path = Path(path)
    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        manifest = load_manifest(path)
    else:
        files = [path]
        manifest = load_manifest(path.parent)
    docs = [load_any(f, info=manifest.get(f.name)) for f in files]
    return [d for d in docs if d.text.strip()]


def ingest_path(
    path: str | Path,
    vector_store=None,
    bm25_store=None,
    embedder=None,
    verbose: bool = False,
) -> dict:
    """Ingest a single file or every supported file in a directory."""
    return ingest_documents(load_path(path), vector_store, bm25_store, embedder, verbose)
