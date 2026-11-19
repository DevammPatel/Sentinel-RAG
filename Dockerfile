# Sentinel-RAG API + chat UI.
#
# Small on purpose: no torch, models run on ONNX Runtime. The embedding and
# reranker weights and the prebuilt index are baked into the image, so a cold
# start downloads nothing and the container works with no outbound network
# access other than the LLM call.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/app/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PORT=7860

# Run as an unprivileged user (uid 1000 is also what Hugging Face Spaces expects).
RUN useradd --create-home --uid 1000 app && mkdir -p /app/models && chown app:app /app /app/models
WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
RUN pip install -r requirements.txt

COPY src ./src
RUN pip install --no-deps -e .

USER app

# Download the configured embedding + reranker models at build time.
RUN python -c "from ragsys.llm import get_embedder; from ragsys.retrieval import get_reranker; \
get_embedder().embed_one('warm'); r = get_reranker(); assert r.kind != 'noop', 'reranker failed to load'"

COPY --chown=app:app scripts ./scripts
COPY --chown=app:app data/index ./data/index
COPY --chown=app:app data/golden ./data/golden
COPY --chown=app:app data/sources.json ./data/sources.json

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)" || exit 1

# One worker: the rate limiter and the BM25 index live in process memory.
CMD ["sh", "-c", "exec uvicorn ragsys.api.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
