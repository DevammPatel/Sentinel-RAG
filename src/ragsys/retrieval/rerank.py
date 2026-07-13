"""Cross-encoder reranking.

Fusion gives good recall; a cross-encoder gives precision by scoring each
(query, chunk) pair jointly. We try the BGE reranker (FlagEmbedding), fall back
to a sentence-transformers CrossEncoder, and finally to a no-op that preserves
fusion order — so reranking never becomes a hard dependency for the demo.
"""
from __future__ import annotations

from functools import lru_cache

from ragsys.config import settings
from ragsys.models import RetrievedChunk
from ragsys.telemetry import SpanKind, span


class Reranker:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self._impl = None
        self._kind = "noop"
        self._load()

    def _load(self) -> None:
        try:
            from FlagEmbedding import FlagReranker

            self._impl = FlagReranker(self.model_name, use_fp16=False)
            self._kind = "flag"
            return
        except Exception:
            pass
        try:
            from sentence_transformers import CrossEncoder

            self._impl = CrossEncoder(self.model_name)
            self._kind = "cross-encoder"
        except Exception as exc:  # pragma: no cover
            print(f"[rerank] no reranker available ({exc}); using fusion order")
            self._kind = "noop"

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int | None = None
    ) -> list[RetrievedChunk]:
        top_n = top_n or settings.rerank_top_n
        if not candidates:
            return []
        with span("rerank", kind=SpanKind.RERANKER,
                  **{"reranker.model": self.model_name, "reranker.kind": self._kind,
                     "reranker.candidates": len(candidates), "reranker.top_n": top_n}):
            if self._kind == "noop":
                return candidates[:top_n]
            pairs = [[query, c.chunk.text] for c in candidates]
            scores = self._impl.compute_score(pairs) if self._kind == "flag" \
                else self._impl.predict(pairs)
            if not isinstance(scores, (list, tuple)):
                scores = [float(scores)]
            for c, s in zip(candidates, scores):
                c.rerank_score = float(s)
            ranked = sorted(candidates, key=lambda c: -(c.rerank_score or 0.0))
            return ranked[:top_n]


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()
