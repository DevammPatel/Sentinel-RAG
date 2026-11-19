"""FastAPI service exposing the agentic RAG pipeline.

Public:
  GET  /               the chat UI
  GET  /health         liveness + index size + active configuration
  GET  /meta           what the UI needs: corpus listing and example questions
  POST /query          ask a question -> grounded, cited answer (rate limited)

Admin (only when ADMIN_TOKEN is set; require `Authorization: Bearer <token>`):
  POST /ingest/sample  load the tiny offline sample corpus
  POST /ingest/path    ingest a file or folder under ./data
"""
from __future__ import annotations

import inspect
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ragsys import __version__
from ragsys.api.security import enforce_rate_limit, require_admin
from ragsys.config import settings
from ragsys.models import QueryRequest, QueryResponse
from ragsys.telemetry import init_tracing

log = logging.getLogger("uvicorn.error")  # shows up in the server log without extra config
STATIC_DIR = Path(__file__).parent / "static"
DATA_ROOT = Path("data").resolve()

EXAMPLE_QUESTIONS = [
    "I asked my bank to cancel my credit card. How quickly do they have to close it?",
    "What limits apply to a 'Small Account' opened at a bank?",
    "What are the deadlines for reporting a cyber incident to SEBI?",
    "For how long is an accredited investor certificate valid?",
    "What net worth does a company need to become a payment aggregator?",
    "What is the price of Bitcoin today?",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_tracing()
    if settings.warm_on_startup:
        # A brand-new Postgres is empty: fill it from the index shipped in the image.
        from ragsys.store import seed_from_index

        seeded = seed_from_index()
        if seeded:
            log.info("seeded %s chunks into the %s store", seeded, settings.vector_backend)
        # Load the embedding model, reranker and index now, so the first visitor
        # doesn't pay for it and /health only turns green once we can answer.
        from ragsys.agent import get_pipeline

        pipeline = get_pipeline()
        pipeline.retrieve("warm up")
        log.info("pipeline warmed: %s chunks indexed", pipeline.retriever.vector_store.count())
    yield


# Newer FastAPI releases emit their own HTTP spans as soon as a tracer provider
# exists. We trace the pipeline ourselves (one AGENT span per question, with the
# retrieval, rerank and LLM steps underneath), and an extra HTTP span per request
# and per health check only buries those. Older releases have no such option.
_extra: dict = {}
if "telemetry" in inspect.signature(FastAPI.__init__).parameters:
    _extra["telemetry"] = {"tracing": False, "metrics": False, "logs": False}

app = FastAPI(
    title="Sentinel-RAG: SEBI / RBI regulations",
    version=__version__,
    description="Hybrid retrieval + reranking + guardrailed, cited answers over "
                "official SEBI and RBI documents.",
    lifespan=lifespan,
    **_extra,
)

if settings.cors_origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Never leak a stack trace to a visitor; log it and return a clean 500."""
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal error. Please try again."})


_PROVIDER_NAMES = {"gemini": "Google Gemini", "groq": "Groq", "openai": "OpenAI", "anthropic": "Anthropic"}


class IngestPathRequest(BaseModel):
    path: str


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    from ragsys.store import get_bm25_store, get_vector_store

    try:
        store = get_vector_store()
        chunks, backend = store.count(), store.backend
    except Exception as exc:  # e.g. pgvector selected but unreachable
        log.warning("vector store unavailable: %s", exc)
        return JSONResponse(
            status_code=503, content={"status": "degraded", "detail": "vector store unavailable"}
        )
    return {
        "status": "ok",
        "version": __version__,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.active_llm_model,
        "llm_configured": settings.llm_available(),
        "answer_mode": "generated" if settings.llm_available() else "extractive",
        "embedding_model": settings.local_embedding_model
        if settings.embeddings_provider == "local" else settings.openai_embedding_model,
        "reranker_model": settings.reranker_model if settings.reranker_enabled else None,
        "vector_backend": backend,
        "vector_chunks": chunks,
        "bm25_chunks": get_bm25_store().count(),
        "tracing_enabled": settings.tracing_enabled,
    }


@app.get("/meta")
def meta() -> dict:
    """Corpus listing + example questions for the UI."""
    from ragsys.store import get_vector_store

    docs: dict[str, dict] = {}
    for chunk in get_vector_store().all_chunks():
        entry = docs.setdefault(chunk.doc_id, {
            "doc_id": chunk.doc_id, "title": chunk.title, "source": chunk.source,
            "url": chunk.url, "circular_no": chunk.meta.get("circular_no"), "chunks": 0,
        })
        entry["chunks"] += 1
    listing = sorted(docs.values(), key=lambda d: (d["source"], d["title"]))
    return {
        "documents": listing,
        "document_count": len(listing),
        "chunk_count": sum(d["chunks"] for d in listing),
        "examples": EXAMPLE_QUESTIONS,
        "llm_configured": settings.llm_available(),
        "llm_model": settings.active_llm_model if settings.llm_available() else None,
        "llm_provider_name": _PROVIDER_NAMES.get(settings.llm_provider) if settings.llm_available() else None,
        "max_question_chars": settings.max_input_chars,
    }


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(enforce_rate_limit)])
def query(req: QueryRequest) -> QueryResponse:
    from ragsys.agent import get_pipeline

    provider = req.provider if settings.allow_provider_override else None
    return get_pipeline().answer(
        req.question,
        top_k=req.top_k,
        rerank_top_n=req.rerank_top_n,
        provider=provider,
    )


@app.post("/ingest/sample", dependencies=[Depends(require_admin)])
def ingest_sample() -> dict:
    from ragsys.ingest import ingest_path
    from ragsys.ingest.sources import sample_corpus

    folder = sample_corpus()[0].parent
    return {"ingested_from": str(folder), **ingest_path(folder)}


@app.post("/ingest/path", dependencies=[Depends(require_admin)])
def ingest_from_path(req: IngestPathRequest) -> dict:
    from ragsys.ingest import ingest_path

    # Confine ingestion to ./data so the endpoint can't be pointed at arbitrary
    # files on the host, even by someone holding the admin token.
    target = Path(req.path).resolve()
    if not target.is_relative_to(DATA_ROOT):
        raise HTTPException(status_code=400, detail="path must be inside the data/ directory")
    if not target.exists():
        raise HTTPException(status_code=404, detail="path not found")
    return ingest_path(target)
