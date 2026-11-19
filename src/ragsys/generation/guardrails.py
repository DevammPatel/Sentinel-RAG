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
    r"\b(ignore (all |the |any )?(previous|above|prior|earlier) (instructions|rules|prompts?)|"
    r"disregard (the |all |your )?(system|previous|above)|you are now|"
    r"(reveal|show|print|repeat) (me )?(your |the )?(system )?(prompt|instructions)|"
    r"act as (an? )?(dan|developer mode)|jailbreak)\b",
    re.IGNORECASE,
)
# Indian identifiers that should never be echoed back: Aadhaar (12 digits, often
# grouped 4-4-4) and PAN (AAAAA9999A).
_PII = re.compile(r"\b(\d{4}[ -]?\d{4}[ -]?\d{4}|[A-Z]{5}\d{4}[A-Z])\b")


@dataclass
class InputGuardResult:
    ok: bool
    reason: str | None = None
    cleaned: str = ""
    injection_flagged: bool = False


def check_input(question: str) -> InputGuardResult:
    with span("guardrail.input", kind=SpanKind.GUARDRAIL) as sp:
        q = re.sub(r"\s+", " ", question or "").strip()
        if not q:
            sp.set("guardrail.blocked", "empty_question")
            return InputGuardResult(ok=False, reason="empty_question")
        if len(q) > settings.max_input_chars:
            sp.set("guardrail.blocked", "too_long")
            return InputGuardResult(ok=False, reason="too_long")
        injection = bool(_INJECTION.search(q))
        sp.set("guardrail.injection_flagged", injection)
        # We don't block on injection (the query might be legitimate), but we flag
        # it so the generator can harden the system prompt and the trace records it.
        return InputGuardResult(ok=True, cleaned=q, injection_flagged=injection)


def relevance(c: RetrievedChunk) -> float:
    """Best available relevance signal on a 0..1 scale, in priority order.

    A cross-encoder rerank score (already sigmoid-squashed) is the most
    trustworthy. Without a reranker we fall back to dense cosine similarity. A
    chunk that only BM25 found has neither, and scores 0: a lexical match on its
    own is not evidence that the passage answers the question.
    """
    if c.rerank_score is not None:
        return c.rerank_score
    if c.semantic_score is not None:
        return max(0.0, c.semantic_score)
    return 0.0


def top_relevance(contexts: list[RetrievedChunk]) -> float:
    return max((relevance(c) for c in contexts), default=0.0)


def grounded_enough(contexts: list[RetrievedChunk], threshold: float | None = None) -> bool:
    """True if at least one retrieved chunk clears the relevance floor."""
    floor = settings.min_context_score if threshold is None else threshold
    return bool(contexts) and top_relevance(contexts) >= floor


def sanitize_output(text: str) -> str:
    """Redact Aadhaar / PAN-shaped identifiers that may have leaked from context."""
    return _PII.sub("[REDACTED]", text or "")
