"""Pydantic domain models shared across ingestion, retrieval, generation and API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source document (one SEBI/RBI circular)."""

    doc_id: str
    title: str
    source: str  # "SEBI" | "RBI"
    url: str | None = None
    circular_no: str | None = None
    issued_on: str | None = None  # ISO date string if parsed
    text: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable slice of a Document."""

    chunk_id: str
    doc_id: str
    title: str
    source: str
    url: str | None = None
    ordinal: int = 0
    text: str
    meta: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk returned by a retriever, with scores attached."""

    chunk: Chunk
    score: float = 0.0  # generic score; becomes the RRF fused score after fusion
    semantic_score: float | None = None  # raw dense cosine similarity (survives fusion)
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    marker: str  # e.g. "[1]"
    doc_id: str
    title: str
    source: str
    url: str | None = None


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    rerank_top_n: int | None = None
    provider: str | None = None  # override LLM provider per-request


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    refused: bool = False
    provider: str
    model: str
    latency_ms: float
    retrieval_ms: float
    generation_ms: float
    trace_id: str | None = None
