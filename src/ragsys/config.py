"""Central, typed configuration loaded from environment / .env.

Everything provider-related is switchable here so the rest of the code never
hardcodes a vendor. Import `settings` (a cached singleton) anywhere.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- LLM provider ----
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_model: str = "claude-sonnet-5"
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # ---- Embeddings ----
    embeddings_provider: Literal["openai", "local"] = "local"
    openai_embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # ---- Reranking ----
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int = 5

    # ---- Retrieval / chunking ----
    retrieve_top_k: int = 20
    rrf_k: int = 60
    chunk_size: int = 900
    chunk_overlap: int = 150

    # ---- Postgres / pgvector ----
    pghost: str = "localhost"
    pgport: int = 5432
    pgdatabase: str = "ragsys"
    pguser: str = "ragsys"
    pgpassword: str = "ragsys"

    # ---- Observability ----
    tracing_enabled: bool = True
    otel_exporter_otlp_endpoint: str = "http://localhost:6006/v1/traces"
    otel_service_name: str = "agentic-rag"
    otel_exporter_otlp_headers: str | None = None
    langsmith_api_key: str | None = None

    # ---- Guardrails ----
    min_context_score: float = 0.15
    max_input_chars: int = 4000

    @property
    def pg_dsn(self) -> str:
        return (
            f"host={self.pghost} port={self.pgport} dbname={self.pgdatabase} "
            f"user={self.pguser} password={self.pgpassword}"
        )

    @property
    def active_llm_model(self) -> str:
        return self.anthropic_model if self.llm_provider == "anthropic" else self.openai_model


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
