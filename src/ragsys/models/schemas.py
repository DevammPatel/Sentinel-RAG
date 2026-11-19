"""Pydantic domain models shared across ingestion, retrieval, generation and API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source document (one SEBI/RBI circular or direction)."""

    doc_id: str
    title: str
    source: str  # "SEBI" | "RBI"
    url: str | None = None
    circular_no: str | None = None
    issued_on: str | None = None  # date string as written in the document
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

    @property
    def search_text(self) -> str:
        """What gets embedded, BM25-indexed and reranked: the passage prefixed
        with its document title, so a clause still says which regulation it
        belongs to once it is cut out of the document."""
        return f"{self.title}\n{self.text}" if self.title else self.text


class RetrievedChunk(BaseModel):
    """A chunk returned by a retriever, with scores attached."""

    chunk: Chunk
    score: float = 0.0  # generic score; becomes the RRF fused score after fusion
    semantic_score: float | None = None  # raw dense cosine similarity (survives fusion)
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None  # cross-encoder relevance in 0..1


class Citation(BaseModel):
    marker: str  # e.g. "[1]"
    doc_id: str
    title: str
    source: str
    url: str | None = None
    circular_no: str | None = None


class SourcePassage(BaseModel):
    """One retrieved passage as shown to the user."""

    marker: str
    doc_id: str
    title: str
    source: str
    url: str | None = None
    circular_no: str | None = None
    text: str
    relevance: float  # 0..1, the signal the grounding gate used
    cited: bool = False


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    rerank_top_n: int | None = Field(default=None, ge=1, le=10)
    # Only honoured when ALLOW_PROVIDER_OVERRIDE=true.
    provider: Literal["groq", "gemini", "openai", "anthropic"] | None = None


class QueryResponse(BaseModel):
    answer: str
    # "generated": written by the LLM from the retrieved passages.
    # "extractive": no LLM (not configured, rate limited, timed out); the answer
    #               quotes the best passages directly.
    # "refused":   nothing relevant enough was retrieved, or the model declined.
    mode: Literal["generated", "extractive", "refused"] = "generated"
    notice: str | None = None  # human-readable reason when mode != "generated"
    refused: bool = False
    citations: list[Citation] = Field(default_factory=list)
    sources: list[SourcePassage] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    rewritten_query: str | None = None
    provider: str
    model: str
    latency_ms: float
    retrieval_ms: float
    generation_ms: float
    trace_id: str | None = None
