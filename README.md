# Agentic RAG — SEBI/RBI Circulars

A domain-specific, **production-shaped** Retrieval-Augmented Generation system over
Indian securities & banking regulation (SEBI / RBI circulars). It is not "a RAG
chatbot" — it ships the pieces most portfolio projects skip: **hybrid retrieval,
reranking, guardrails, an evaluation harness, and OpenTelemetry observability**,
behind a real FastAPI service that runs in Docker.

- **Provider-configurable generation** — Anthropic (Claude) or OpenAI (GPT), switchable by config or per request.
- **Hybrid retrieval** — BM25 (sparse) + pgvector (dense), fused with Reciprocal Rank Fusion.
- **Cross-encoder reranking** — BGE reranker for precision on top of high-recall fusion.
- **Agentic control flow** — the pipeline *decides*: query rewrite + re-retrieve when context is weak, and refuses rather than hallucinate.
- **Guardrails** — input validation, prompt-injection flagging, grounding threshold, PII redaction.
- **Evaluation** — RAGAS (faithfulness / answer-relevancy / context-precision) **plus** custom deterministic metrics (retrieval hit-rate, keyword recall, refusal accuracy, latency p95).
- **Observability** — OpenTelemetry + OpenInference spans that render in **Arize Phoenix or LangSmith** with a single endpoint change.
- **Runs with zero infra** — in-memory vector store + offline sample corpus + stubbable LLM, so `make smoke` proves the whole pipeline with no keys or Postgres.

---

## Architecture

```
                                 ┌─────────────────────────────────────────────┐
   PDFs / txt  ──►  Ingestion    │  loader → structure-aware chunker → embed    │
  (SEBI/RBI)        pipeline     │           │                    │             │
                                 │           ▼                    ▼             │
                                 │      BM25 store          pgvector store      │
                                 └─────────────────────────────────────────────┘
                                                     ▲
   HTTP  ──►  FastAPI /query                         │
                    │                                │
                    ▼                                │
        ┌───────────────────────── Agentic RAG loop ─────────────────────────┐
        │ 1. input guardrail (injection flag, length)                        │
        │ 2. hybrid retrieve  ──► BM25 + dense ──► Reciprocal Rank Fusion     │
        │ 3. cross-encoder rerank (BGE)                                       │
        │ 4. DECIDE grounded? ── no ──► rewrite query (expand acronyms) ──► 2 │
        │ 5. still ungrounded ──► REFUSE (no hallucination)                   │
        │ 6. generate cited answer (Claude | GPT)  ──► sanitize output        │
        └────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
        QueryResponse { answer, citations[], contexts[], latency breakdown, trace_id }

   Every stage emits an OpenInference span ──►  OTLP  ──►  Phoenix  /  LangSmith
```

Package layout (`src/ragsys/`):

| Module | Responsibility |
|---|---|
| `config.py` | One typed settings object; all provider switches live here. |
| `telemetry.py` | OTel/OpenInference tracer; Phoenix or LangSmith by endpoint. |
| `ingest/` | `sources` (download / sample corpus), `loader` (PDF parse + metadata), `chunker` (clause-aware), `pipeline` (orchestration). |
| `store/` | `vector_store` (pgvector **or** in-memory fallback), `bm25_store`. |
| `llm/` | `provider` (Claude/GPT abstraction), `embeddings` (OpenAI/local). |
| `retrieval/` | `hybrid` (RRF fusion), `rerank` (cross-encoder). |
| `generation/` | `guardrails`, `generator` (cited answers). |
| `agent/` | `orchestrator` — the decision loop. |
| `eval/` | `metrics` (custom), `run_eval` (RAGAS + custom runner). |
| `api/` | FastAPI app. |

---

## Quickstart

### Option A — zero infra (no keys, no Docker)

```bash
make install            # or: pip install -r requirements.txt && pip install -e .
make smoke              # ingest sample corpus → retrieve → rerank → (stub) generate
make test               # 15 unit tests, all offline
```

`make smoke` uses the in-memory vector store, a fake embedder and a stub LLM to
prove the pipeline is wired end to end.

### Option B — real answers (add a key)

```bash
cp .env.example .env
# set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY=...   (or openai + OPENAI_API_KEY)
# EMBEDDINGS_PROVIDER=local uses BGE locally and needs no embedding key

python scripts/ingest_cli.py --sample          # or --path data/pdfs  or --urls <pdf urls>
make serve                                      # http://localhost:8000/docs
```

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"What net worth is required for an individual investment adviser?"}' | jq
```

### Option C — full stack in Docker (API + pgvector + Phoenix)

```bash
cp .env.example .env      # add your API key
make up                   # builds API, starts pgvector + Phoenix
curl -X POST localhost:8000/ingest/sample
# Phoenix tracing UI:  http://localhost:6006
# API docs:            http://localhost:8000/docs
make down
```

---

## Retrieval design notes

- **Why hybrid?** BM25 nails exact regulatory tokens (circular numbers, "net worth",
  "Rupees fifty lakh"); dense retrieval handles paraphrase ("how much capital does an
  adviser need"). Fusing them beats either alone on regulatory Q&A.
- **Why RRF?** It fuses on *rank*, so it needs no score normalization between BM25's
  unbounded scores and cosine's `[-1, 1]`. `score(d) = Σ 1/(k + rank_r(d))`, `k=60`.
- **Why rerank?** Fusion maximizes recall; a cross-encoder scoring each `(query, chunk)`
  pair jointly restores precision so the LLM sees the 5 best passages, not 20 okay ones.
- **Grounding gate.** We threshold on the strongest available relevance signal
  (rerank score › dense cosine › fused score). Below `MIN_CONTEXT_SCORE` the agent
  rewrites the query and retries once, then refuses — that refusal path is what keeps
  faithfulness high and is explicitly measured in the eval suite.

## Observability

Spans use **OpenInference** semantic conventions (`LLM`, `RETRIEVER`, `RERANKER`,
`EMBEDDING`, `AGENT`, `GUARDRAIL`) so both backends render a proper trace tree.
Switch backend with one env var:

```bash
# Arize Phoenix (default, self-hosted via docker-compose)
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006/v1/traces

# LangSmith
OTEL_EXPORTER_OTLP_ENDPOINT=https://api.smith.langchain.com/otel/v1/traces
OTEL_EXPORTER_OTLP_HEADERS=x-api-key=<LANGSMITH_API_KEY>,Langsmith-Project=agentic-rag
```

If the OTel libs are absent or `TRACING_ENABLED=false`, all spans become no-ops.

## Evaluation

```bash
python -m ragsys.eval.run_eval --dataset data/golden/qa.jsonl --provider anthropic
```

Produces `data/eval_report.json` and a summary table:

| Metric | Source | Meaning |
|---|---|---|
| `faithfulness`, `answer_relevancy`, `context_precision` | RAGAS (LLM-judged) | answer grounded in context, on-topic, retrieved context precise |
| `context_hit_rate` | custom | did retrieval surface a passage covering the gold answer |
| `keyword_recall` | custom | lexical recall of the gold answer content words |
| `refusal_accuracy` | custom | refused exactly on the unanswerable rows |
| `latency_ms_mean / p95`, `retrieval_ms`, `generation_ms` | custom | performance budget |

RAGAS runs only if `ragas` + a judge key are available; otherwise it's skipped and the
deterministic metrics still run. Swap `--provider` to A/B Claude vs GPT on one dataset.

## Using real circulars

The offline `sample_corpus()` is paraphrased summaries so the demo is self-contained.
For real data, drop circular PDFs into `data/pdfs/` and run
`python scripts/ingest_cli.py --path data/pdfs`, or pass URLs you're permitted to fetch
with `--urls`. The loader auto-detects SEBI vs RBI and extracts circular numbers/dates.

## Testing

```bash
make test    # chunking, RRF fusion + metadata merge, guardrails, agent loop, hybrid retrieval
```

All tests are offline (no keys, no Postgres) using in-memory stores and a fake embedder.

## License / disclaimer

Educational portfolio project. The sample corpus is paraphrased and must not be relied
on as legal or compliance advice. Verify against the official SEBI/RBI circulars.
