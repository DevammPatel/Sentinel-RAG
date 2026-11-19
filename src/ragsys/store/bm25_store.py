"""Sparse BM25 index (rank-bm25), held in memory.

The vector store is the single source of truth for chunk text, so this index is
never persisted: it is rebuilt from `vector_store.all_chunks()` on startup and
after every ingest. That removes a whole class of "the two indexes disagree"
bugs. For very large corpora you'd swap this for an Elasticsearch / OpenSearch
BM25 index behind the same `.search()` signature.
"""
from __future__ import annotations

import re
from functools import lru_cache

from ragsys.models import Chunk, RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Store:
    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._chunks: list[Chunk] = []
        self._bm25 = None
        if chunks:
            self.rebuild(chunks)

    def rebuild(self, chunks: list[Chunk]) -> None:
        """Replace the whole index with `chunks`."""
        from rank_bm25 import BM25Okapi

        self._chunks = list({c.chunk_id: c for c in chunks}.values())
        corpus = [_tokenize(c.search_text) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._bm25 is None or not self._chunks:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        out: list[RetrievedChunk] = []
        for idx in ranked:
            if scores[idx] <= 0:
                break
            out.append(
                RetrievedChunk(
                    chunk=self._chunks[idx],
                    score=float(scores[idx]),
                    sparse_rank=len(out) + 1,
                )
            )
        return out

    def count(self) -> int:
        return len(self._chunks)


@lru_cache
def get_bm25_store() -> BM25Store:
    """The process-wide BM25 index, built from the vector store's chunks."""
    from ragsys.store.vector_store import get_vector_store

    return BM25Store(get_vector_store().all_chunks())
