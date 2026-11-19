"""Hybrid retrieval: dense (vector store) + sparse (BM25) fused with Reciprocal Rank Fusion.

RRF is rank-based, so it needs no score normalisation across the two very
different score scales (cosine similarity vs BM25). score(d) = sum_r 1/(k + rank_r(d)).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Literal

from ragsys.config import settings
from ragsys.models import RetrievedChunk
from ragsys.telemetry import SpanKind, span

RetrievalMode = Literal["hybrid", "dense", "sparse"]


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
    """Dense + sparse retrieval over the shared stores.

    Stores and the embedder are resolved lazily so tests can inject fakes and so
    constructing a retriever never loads a model by itself.
    """

    def __init__(self, vector_store=None, bm25_store=None, embedder=None) -> None:
        self._vector_store = vector_store
        self._bm25_store = bm25_store
        self._embedder = embedder

    @property
    def vector_store(self):
        if self._vector_store is None:
            from ragsys.store import get_vector_store

            self._vector_store = get_vector_store()
        return self._vector_store

    @property
    def bm25_store(self):
        if self._bm25_store is None:
            from ragsys.store import get_bm25_store

            self._bm25_store = get_bm25_store()
        return self._bm25_store

    @property
    def embedder(self):
        if self._embedder is None:
            from ragsys.llm import get_embedder

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
        mode: RetrievalMode = "hybrid",
    ) -> list[RetrievedChunk]:
        """`mode` exists for the eval ablation: "dense" or "sparse" skip fusion."""
        top_k = top_k or settings.retrieve_top_k
        rrf_k = rrf_k or settings.rrf_k

        with span("retrieve", kind=SpanKind.RETRIEVER,
                  **{"input.value": query, "retrieval.mode": mode,
                     "retrieval.top_k": top_k}) as sp:
            dense: list[RetrievedChunk] = []
            sparse: list[RetrievedChunk] = []
            if mode in ("hybrid", "dense"):
                dense = self.vector_store.search(self.embedder.embed_one(query), top_k)
            if mode in ("hybrid", "sparse"):
                sparse = self.bm25_store.search(query, top_k)
            lists = [lst for lst in (dense, sparse) if lst]
            fused = reciprocal_rank_fusion(lists, k=rrf_k, top_k=top_k) if lists else []
            sp.set("retrieval.dense_hits", len(dense))
            sp.set("retrieval.sparse_hits", len(sparse))
            sp.set("retrieval.fused_hits", len(fused))
            for i, rc in enumerate(fused[:5]):
                sp.set(f"retrieval.documents.{i}.document.id", rc.chunk.chunk_id)
                sp.set(f"retrieval.documents.{i}.document.score", rc.score)
                sp.set(f"retrieval.documents.{i}.document.content", rc.chunk.text[:500])
            return fused
