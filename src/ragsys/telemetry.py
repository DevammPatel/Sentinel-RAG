"""OpenTelemetry / OpenInference tracing.

One tracer, two possible backends, chosen purely by the OTLP endpoint:

  * Arize Phoenix  -> http://localhost:6006/v1/traces   (run via docker-compose)
  * LangSmith      -> https://api.smith.langchain.com/otel/v1/traces
                      with OTEL_EXPORTER_OTLP_HEADERS="x-api-key=<key>,Langsmith-Project=..."

We emit spans using OpenInference semantic conventions so both backends render
them as proper LLM/RETRIEVER spans. If the OTEL libraries are not installed (or
TRACING_ENABLED=false) everything degrades to cheap no-ops so the app still runs.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from ragsys.config import settings

_TRACER = None
_INITIALISED = False


def _parse_headers(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    out: dict[str, str] = {}
    for pair in raw.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def init_tracing() -> None:
    """Idempotently configure the global tracer provider + OTLP exporter."""
    global _TRACER, _INITIALISED
    if _INITIALISED:
        return
    _INITIALISED = True

    if not settings.tracing_enabled:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception as exc:  # pragma: no cover - optional dep
        print(f"[telemetry] OTEL libs unavailable, tracing disabled: {exc}")
        return

    headers = _parse_headers(settings.otel_exporter_otlp_headers)
    if settings.langsmith_api_key and "x-api-key" not in headers:
        headers["x-api-key"] = settings.langsmith_api_key

    resource = Resource.create({
        "service.name": settings.otel_service_name,
        # Phoenix files traces under this project name instead of "default".
        "openinference.project.name": settings.otel_service_name,
    })
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(
        endpoint=settings.otel_exporter_otlp_endpoint,
        headers=headers or None,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer("ragsys")
    print(f"[telemetry] tracing -> {settings.otel_exporter_otlp_endpoint}")


def _tracer():
    if not _INITIALISED:
        init_tracing()
    return _TRACER


# OpenInference span-kind values (string constants avoid a hard import dependency).
class SpanKind:
    CHAIN = "CHAIN"
    RETRIEVER = "RETRIEVER"
    RERANKER = "RERANKER"
    LLM = "LLM"
    EMBEDDING = "EMBEDDING"
    AGENT = "AGENT"
    GUARDRAIL = "GUARDRAIL"


@contextlib.contextmanager
def span(name: str, kind: str = SpanKind.CHAIN, **attributes: Any) -> Iterator[Any]:
    """Context manager yielding a span-like object with `.set(k, v)`.

    No-ops cleanly when tracing is disabled.
    """
    tracer = _tracer()
    if tracer is None:
        yield _NoopSpan()
        return

    attrs = {"openinference.span.kind": kind}
    attrs.update({k: _coerce(v) for k, v in attributes.items()})
    with tracer.start_as_current_span(name, attributes=attrs) as otel_span:
        yield _SpanWrapper(otel_span)


def current_trace_id() -> str | None:
    tracer = _tracer()
    if tracer is None:
        return None
    try:
        from opentelemetry import trace

        ctx = trace.get_current_span().get_span_context()
        if ctx and ctx.trace_id:
            return format(ctx.trace_id, "032x")
    except Exception:
        return None
    return None


def _coerce(v: Any) -> Any:
    if isinstance(v, (str, bool, int, float)):
        return v
    return str(v)


class _SpanWrapper:
    def __init__(self, otel_span: Any) -> None:
        self._s = otel_span

    def set(self, key: str, value: Any) -> None:
        self._s.set_attribute(key, _coerce(value))


class _NoopSpan:
    def set(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        return None
