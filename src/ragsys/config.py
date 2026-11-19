"""Central, typed configuration loaded from environment / .env.

Everything provider-related is switchable here so the rest of the code never
hardcodes a vendor. Import `settings` (a cached singleton) anywhere.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# OpenAI-compatible providers: name -> default base URL.
OPENAI_COMPATIBLE_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openai": None,  # SDK default
}

LLMProviderName = Literal["groq", "gemini", "openai", "anthropic", "none"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- LLM provider ----
    # "groq" and "gemini" have free tiers; "openai" and "anthropic" are paid.
    # "none" (or a missing key) runs in extractive mode: no LLM call, the answer
    # is the best-matching passages quoted with citations.
    llm_provider: LLMProviderName = "groq"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.8-flash"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None  # any other OpenAI-compatible endpoint
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"
    anthropic_max_tokens: int = 16000
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 1
    # Use the LLM to rewrite a weak query before the second retrieval attempt.
    llm_query_rewrite: bool = True

    # ---- Embeddings ----
    # "local" runs a small ONNX model on CPU (no key, no torch).
    embeddings_provider: Literal["local", "openai"] = "local"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 384

    # ---- Reranking ----
    reranker_enabled: bool = True
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # Which ONNX file of that repo to load. The int8 build is about twice as fast
    # as full precision on 2 CPUs and scored the same or better on the golden set.
    # Leave empty to use the library's default file for the model.
    reranker_onnx_file: str = "onnx/model_int8.onnx"
    rerank_top_n: int = 5
    rerank_batch_size: int = 8          # smaller batches = lower peak memory

    # ---- ONNX Runtime ----
    # Threads per model. Unset = the CPUs this container is actually allowed
    # (see ragsys.runtime.available_cpus), not the host's core count.
    onnx_threads: int | None = None
    # The memory arena trades RAM for a little speed. Off keeps the process
    # small enough for 1 GB free tiers.
    onnx_mem_arena: bool = False

    # ---- Retrieval / chunking ----
    retrieve_top_k: int = 20
    rrf_k: int = 60
    chunk_size: int = 900
    chunk_overlap: int = 150

    # ---- Index storage ----
    # "memory": numpy cosine search over files in INDEX_DIR (zero infra).
    # "pgvector": Postgres + pgvector (docker-compose, or a hosted free tier).
    vector_backend: Literal["memory", "pgvector"] = "memory"
    index_dir: str = "data/index"
    database_url: str | None = None  # e.g. postgresql://user:pass@host/db?sslmode=require
    pghost: str = "localhost"
    pgport: int = 5432
    pgdatabase: str = "ragsys"
    pguser: str = "ragsys"
    pgpassword: str = "ragsys"

    # ---- Observability ----
    tracing_enabled: bool = False
    otel_exporter_otlp_endpoint: str = "http://localhost:6006/v1/traces"
    otel_service_name: str = "sentinel-rag"
    otel_exporter_otlp_headers: str | None = None
    langsmith_api_key: str | None = None

    # ---- Guardrails ----
    # Relevance floor on a 0..1 scale (reranker scores are squashed with a
    # sigmoid). Chosen with `python -m ragsys.eval.calibrate`. On the golden set
    # 0.4 separates every answerable question (lowest score 0.42) from every
    # out-of-scope one (highest 0.36). That gap is narrow and the set is small,
    # so treat it as a sensible default rather than a guarantee; the generator's
    # own "say you don't know" rule is the second line of defence.
    min_context_score: float = 0.4
    max_input_chars: int = 1000

    # ---- API hardening ----
    # Ingest endpoints only exist when ADMIN_TOKEN is set, and require it.
    admin_token: str | None = None
    allow_provider_override: bool = False
    rate_limit_per_minute: int = 8      # per client IP
    rate_limit_per_day: int = 150       # per client IP
    global_daily_limit: int = 1200      # whole service; protects the free LLM quota
    trust_proxy_headers: bool = False   # set true behind a reverse proxy (Cloud Run, Render...)
    proxy_hops: int = 1                 # trusted proxies in front of the app (2 behind a CDN + LB)
    cors_origins: str = ""              # comma-separated; empty = same-origin only
    warm_on_startup: bool = True        # load models + index before serving traffic

    # ---- derived ----
    @property
    def pg_dsn(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"host={self.pghost} port={self.pgport} dbname={self.pgdatabase} "
            f"user={self.pguser} password={self.pgpassword}"
        )

    def model_for(self, provider: str) -> str:
        return {
            "groq": self.groq_model,
            "gemini": self.gemini_model,
            "openai": self.openai_model,
            "anthropic": self.anthropic_model,
        }.get(provider, "extractive")

    def api_key_for(self, provider: str) -> str | None:
        return {
            "groq": self.groq_api_key,
            "gemini": self.gemini_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }.get(provider)

    def llm_available(self, provider: str | None = None) -> bool:
        """True when the provider is a real LLM and its key is configured."""
        provider = provider or self.llm_provider
        return provider != "none" and bool(self.api_key_for(provider))

    @property
    def active_llm_model(self) -> str:
        return self.model_for(self.llm_provider)

    @property
    def model_threads(self) -> int:
        from ragsys.runtime import available_cpus

        return self.onnx_threads or available_cpus()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
