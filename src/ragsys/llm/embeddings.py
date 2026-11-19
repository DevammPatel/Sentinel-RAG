"""Embeddings: a local ONNX model (default) or OpenAI, chosen by config.

The local path uses `fastembed`, which runs BGE models on ONNX Runtime. That
keeps the service small enough for free CPU hosts: no torch, ~70 MB of model
weights, and a few hundred MB of RAM.

Queries and passages are embedded differently on purpose: BGE models are trained
with an instruction prefix on the query side only, and `query_embed` applies it.
Both paths return L2-normalised vectors, so a dot product is cosine similarity.
"""
from __future__ import annotations

import math
from functools import lru_cache

from ragsys.config import settings
from ragsys.telemetry import SpanKind, span


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class Embedder:
    def __init__(self) -> None:
        self.provider = settings.embeddings_provider
        self._local = None
        self._openai = None
        if self.provider == "local":
            from fastembed import TextEmbedding

            self._local = TextEmbedding(
                model_name=settings.local_embedding_model,
                threads=settings.model_threads,
                enable_cpu_mem_arena=settings.onnx_mem_arena,
            )
            self.dim = settings.embedding_dim
        else:
            from openai import OpenAI

            self._openai = OpenAI(api_key=settings.openai_api_key)
            self.dim = settings.embedding_dim

    @property
    def model_name(self) -> str:
        return (
            settings.local_embedding_model
            if self.provider == "local"
            else settings.openai_embedding_model
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed passages (documents)."""
        if not texts:
            return []
        with span("embed.passages", kind=SpanKind.EMBEDDING,
                  **{"embedding.model_name": self.model_name, "embedding.n": len(texts)}):
            if self.provider == "local":
                return [_normalise(v.tolist()) for v in self._local.passage_embed(texts)]
            return self._openai_embed(texts)

    def embed_one(self, text: str) -> list[float]:
        """Embed a search query."""
        with span("embed.query", kind=SpanKind.EMBEDDING,
                  **{"embedding.model_name": self.model_name}):
            if self.provider == "local":
                return _normalise(next(iter(self._local.query_embed(text))).tolist())
            return self._openai_embed([text])[0]

    def _openai_embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._openai.embeddings.create(
            model=settings.openai_embedding_model, input=texts
        )
        return [_normalise(d.embedding) for d in resp.data]


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()
