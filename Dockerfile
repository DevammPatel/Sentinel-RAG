FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# System deps for pypdf / sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY data ./data
RUN pip install -e .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "ragsys.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
