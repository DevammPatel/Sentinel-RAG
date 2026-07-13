import os

# Keep the test suite fully offline: no OTLP exporter, no network.
os.environ.setdefault("TRACING_ENABLED", "false")
os.environ.setdefault("EMBEDDINGS_PROVIDER", "local")
