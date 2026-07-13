"""Hybrid retrieval: dense (pgvector) + sparse (BM25) fused with Reciprocal Rank Fusion.

RRF is rank-based, so it needs no score normalisation across the two very
different score scales (cosine similarity vs BM25). score(d) = sum_r 1/(k + rank_r(d)).
"""
from __future__ import annotations

from collections import defaultdict

from ragsys.config import settings
from ragsys.llm import get_embedder
from ragsys.models import RetrievedChunk
from ragsys.store import BM25Store
from ragsys.store.vector_store import BaseVectorStore, get_vector_store
from ragsys.telemetry import SpanKind, span


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked lists of RetrievedChunk into one, by chunk_id."""
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            cid = item.chunk.chunk_id
            scores[cid] += 1.0 / (k + rank + 1)
            # keep the richest record for this chunk (merge rank info)
            if cid not in best:
                best[cid] = item.model_copy(deep=True)
            else:
                if item.dense_rank is not None:
                    best[cid].dense_rank = item.dense_rank
                if item.sparse_rank is not None:
                    best[cid].sparse_rank = item.sparse_rank
            # preserve the strongest raw semantic similarity across lists
            if item.semantic_score is not None:
                prev = best[cid].semantic_score
                best[cid].semantic_score = (
                    item.semantic_score if prev is None else max(prev, item.semantic_score)
                )

    fused: list[RetrievedChunk] = []
    for cid, score in scores.items():
        rc = best[cid]
        rc.score = score
        fused.append(rc)
    fused.sort(key=lambda r: -r.score)
    return fused[:top_k] if top_k else fused


class HybridRetriever:
    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        bm25_store: BM25Store | None = None,
        embedder=None,
    ) -> None:
        self.vector_store = vector_store or get_vector_store()
        self.bm25_store = bm25_store or BM25Store()
        # embedder is loaded lazily so tests/smoke can inject a fake one
        self._embedder = embedder

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @embedder.setter
    def embedder(self, value) -> None:
        self._embedder = value

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.retrieve_top_k
        rrf_k = rrf_k or settings.rrf_k

        with span("retrieve.hybrid", kind=SpanKind.RETRIEVER,
                  **{"retrieval.query": query, "retrieval.top_k": top_k}) as sp:
            qvec = self.embedder.embed_one(query)
            dense = self.vector_store.search(qvec, top_k)
            sparse = self.bm25_store.search(query, top_k)
            fused = reciprocal_rank_fusion([dense, sparse], k=rrf_k, top_k=top_k)
            sp.set("retrieval.dense_hits", len(dense))
            sp.set("retrieval.sparse_hits", len(sparse))
            sp.set("retrieval.fused_hits", len(fused))
            return fused
