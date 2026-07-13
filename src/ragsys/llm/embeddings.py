"""Embeddings — OpenAI or a local sentence-transformers model, chosen by config.

Both return plain `list[list[float]]`, so the vector store never cares which
backend produced them (as long as EMBEDDING_DIM matches the pgvector column).
"""
from __future__ import annotations

from functools import lru_cache

from ragsys.config import settings
from ragsys.telemetry import SpanKind, span


class Embedder:
    def __init__(self) -> None:
        self.provider = settings.embeddings_provider
        self._local = None
        self._openai = None
        if self.provider == "local":
            from sentence_transformers import SentenceTransformer

            self._local = SentenceTransformer(settings.local_embedding_model)
            self.dim = self._local.get_sentence_embedding_dimension()
        else:
            from openai import OpenAI

            self._openai = OpenAI(api_key=settings.openai_api_key)
            self.dim = settings.embedding_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with span("embed", kind=SpanKind.EMBEDDING,
                  **{"embedding.model": self._model_name(), "embedding.n": len(texts)}):
            if self.provider == "local":
                vecs = self._local.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False
                )
                return [v.tolist() for v in vecs]
            resp = self._openai.embeddings.create(
                model=settings.openai_embedding_model, input=texts
            )
            return [d.embedding for d in resp.data]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _model_name(self) -> str:
        return (
            settings.local_embedding_model
            if self.provider == "local"
            else settings.openai_embedding_model
        )


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()
