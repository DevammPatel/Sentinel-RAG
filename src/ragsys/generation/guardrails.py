"""Guardrails: input validation, prompt-injection screening, grounding checks.

These are intentionally lightweight and deterministic (regex + thresholds) so
they add no latency and are trivially unit-testable. In production you'd layer a
model-based moderation call on top, but the cheap checks catch the common cases.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ragsys.config import settings
from ragsys.models import RetrievedChunk
from ragsys.telemetry import SpanKind, span

# Common prompt-injection / jailbreak phrasing.
_INJECTION = re.compile(
    r"\b(ignore (all |the )?(previous|above|prior) instructions|"
    r"disregard (the )?(system|previous)|you are now|"
    r"reveal your (system )?prompt|act as (an? )?(dan|developer mode))\b",
    re.IGNORECASE,
)
_PII_HINT = re.compile(r"\b(\d{12}|\d{4}\s\d{4}\s\d{4})\b")  # naive Aadhaar-like


@dataclass
class InputGuardResult:
    ok: bool
    reason: str | None = None
    cleaned: str = ""
    injection_flagged: bool = False


def check_input(question: str) -> InputGuardResult:
    with span("guardrail.input", kind=SpanKind.GUARDRAIL) as sp:
        q = (question or "").strip()
        if not q:
            return InputGuardResult(ok=False, reason="empty_question")
        if len(q) > settings.max_input_chars:
            return InputGuardResult(ok=False, reason="too_long")
        injection = bool(_INJECTION.search(q))
        sp.set("guardrail.injection_flagged", injection)
        # We don't block on injection (the query might be legitimate), but we flag
        # it so the generator can harden the system prompt and the trace records it.
        return InputGuardResult(ok=True, cleaned=q, injection_flagged=injection)


def _relevance(c: RetrievedChunk) -> float:
    """Best available relevance signal, in priority order.

    A cross-encoder rerank score is the most trustworthy; failing that the raw
    dense cosine similarity (which survives fusion via `semantic_score`); the
    RRF fused score is only a last resort since its scale is tiny.
    """
    if c.rerank_score is not None:
        return c.rerank_score
    if c.semantic_score is not None:
        return c.semantic_score
    return c.score


def grounded_enough(contexts: list[RetrievedChunk]) -> bool:
    """True if at least one retrieved chunk clears the relevance floor."""
    if not contexts:
        return False
    return max(_relevance(c) for c in contexts) >= settings.min_context_score


def sanitize_output(text: str) -> str:
    """Redact obvious PII-looking numbers that may have leaked from context."""
    return _PII_HINT.sub("[REDACTED]", text or "")
