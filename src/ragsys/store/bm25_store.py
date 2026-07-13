"""Sparse BM25 store (rank-bm25), persisted to a pickle.

Kept separate from the dense store so the two can be rebuilt independently and
fused at query time. For very large corpora you'd swap this for an Elasticsearch
/ OpenSearch BM25 index behind the same `.search()` signature.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from ragsys.models import Chunk, RetrievedChunk

_DEFAULT_PATH = Path("data/index/bm25.pkl")
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Store:
    def __init__(self, path: str | Path = _DEFAULT_PATH) -> None:
        self.path = Path(path)
        self._chunks: list[Chunk] = []
        self._bm25 = None
        if self.path.exists():
            self.load()

    def build(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        # de-dup by chunk_id so re-ingestion doesn't double-count
        seen: dict[str, Chunk] = {c.chunk_id: c for c in self._chunks}
        for c in chunks:
            seen[c.chunk_id] = c
        self._chunks = list(seen.values())
        corpus = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self.save()

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._bm25 is None or not self._chunks:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        out: list[RetrievedChunk] = []
        for rank, idx in enumerate(ranked):
            if scores[idx] <= 0:
                continue
            out.append(
                RetrievedChunk(
                    chunk=self._chunks[idx],
                    score=float(scores[idx]),
                    sparse_rank=rank + 1,
                )
            )
        return out

    def count(self) -> int:
        return len(self._chunks)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as fh:
            pickle.dump({"chunks": [c.model_dump() for c in self._chunks]}, fh)

    def load(self) -> None:
        from rank_bm25 import BM25Okapi

        with self.path.open("rb") as fh:
            data = pickle.load(fh)
        self._chunks = [Chunk(**c) for c in data["chunks"]]
        corpus = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None
