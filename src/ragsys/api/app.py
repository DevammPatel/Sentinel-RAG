"""FastAPI service exposing the agentic RAG pipeline.

Endpoints:
  GET  /health          liveness + index sizes + active provider
  POST /query           ask a question -> grounded, cited answer
  POST /ingest/sample   load the offline sample SEBI/RBI corpus (demo bootstrap)
  POST /ingest/path     ingest a server-side file or directory of PDFs/txt

Tracing is initialised on startup so every request produces a span tree.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ragsys.config import settings
from ragsys.models import QueryRequest, QueryResponse
from ragsys.telemetry import init_tracing


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_tracing()
    yield


app = FastAPI(
    title="Agentic RAG — SEBI/RBI Circulars",
    version="0.1.0",
    description="Hybrid retrieval + reranking + guardrailed, cited generation.",
    lifespan=lifespan,
)


class IngestPathRequest(BaseModel):
    path: str


@app.get("/health")
def health() -> dict:
    from ragsys.store import BM25Store
    from ragsys.store.vector_store import get_vector_store

    try:
        vc = get_vector_store().count()
    except Exception:
        vc = -1
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "llm_model": settings.active_llm_model,
        "embeddings_provider": settings.embeddings_provider,
        "vector_chunks": vc,
        "bm25_chunks": BM25Store().count(),
        "tracing_enabled": settings.tracing_enabled,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    from ragsys.agent import get_pipeline

    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    pipeline = get_pipeline()
    return pipeline.answer(
        req.question,
        top_k=req.top_k,
        rerank_top_n=req.rerank_top_n,
        provider=req.provider,
    )


@app.post("/ingest/sample")
def ingest_sample() -> dict:
    from ragsys.ingest import ingest_path
    from ragsys.ingest.sources import sample_corpus

    paths = sample_corpus()
    stats = ingest_path(paths[0].parent)
    return {"ingested_from": str(paths[0].parent), **stats}


@app.post("/ingest/path")
def ingest_from_path(req: IngestPathRequest) -> dict:
    from pathlib import Path

    from ragsys.ingest import ingest_path

    if not Path(req.path).exists():
        raise HTTPException(status_code=404, detail=f"path not found: {req.path}")
    return ingest_path(req.path)
