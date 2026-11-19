"""Cross-encoder reranking.

Fusion gives good recall; a cross-encoder gives precision by scoring each
(query, chunk) pair jointly. The model runs on ONNX Runtime via `fastembed`, so
it works on a small CPU box.

Cross-encoders emit raw logits whose range differs from model to model. We pass
them through a sigmoid so `rerank_score` is always in 0..1, which is what lets
the grounding gate use one threshold regardless of which reranker is configured.

If the reranker is disabled or cannot load, we keep fusion order and leave
`rerank_score` unset; the grounding gate then falls back to dense similarity.
"""
from __future__ import annotations

import math
from functools import lru_cache

from ragsys.config import settings
from ragsys.models import RetrievedChunk
from ragsys.telemetry import SpanKind, span


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class Reranker:
    def __init__(
        self,
        model_name: str | None = None,
        enabled: bool | None = None,
        onnx_file: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.reranker_model
        self.onnx_file = settings.reranker_onnx_file if onnx_file is None else onnx_file
        self._impl = None
        self.kind = "noop"
        if enabled if enabled is not None else settings.reranker_enabled:
            self._load()

    def _load(self) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            name = self.model_name
            if self.onnx_file:
                # Register this specific weights file under its own name.
                from fastembed.common.model_description import ModelSource

                name = f"{self.model_name}::{self.onnx_file}"
                known = {m["model"] for m in TextCrossEncoder.list_supported_models()}
                if name not in known:
                    TextCrossEncoder.add_custom_model(
                        model=name,
                        sources=ModelSource(hf=self.model_name),
                        model_file=self.onnx_file,
                    )
            self._impl = TextCrossEncoder(
                model_name=name,
                threads=settings.model_threads,
                enable_cpu_mem_arena=settings.onnx_mem_arena,
            )
            self.kind = "cross-encoder"
        except Exception as exc:  # pragma: no cover - depends on environment
            print(f"[rerank] reranker unavailable ({exc}); using fusion order")

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance of each text to the query, in 0..1."""
        raw = self._impl.rerank(query, texts, batch_size=settings.rerank_batch_size)
        return [sigmoid(float(s)) for s in raw]

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int | None = None
    ) -> list[RetrievedChunk]:
        top_n = top_n or settings.rerank_top_n
        if not candidates:
            return []
        with span("rerank", kind=SpanKind.RERANKER,
                  **{"reranker.model_name": self.model_name, "reranker.kind": self.kind,
                     "reranker.candidates": len(candidates), "reranker.top_n": top_n}) as sp:
            if self.kind == "noop":
                return candidates[:top_n]
            scores = self.score(query, [c.chunk.search_text for c in candidates])
            for c, s in zip(candidates, scores):
                c.rerank_score = s
            ranked = sorted(candidates, key=lambda c: -(c.rerank_score or 0.0))
            sp.set("reranker.top_score", ranked[0].rerank_score or 0.0)
            return ranked[:top_n]


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()
