"""Custom, model-free metrics that complement RAGAS.

RAGAS metrics (faithfulness, answer/context relevancy) need an LLM judge and API
keys. These deterministic metrics run anywhere and cover the things RAGAS does
not: retrieval hit-rate against the expected source, cheap lexical recall of the
ground-truth answer, latency, and whether refusals happen on the right rows.
"""
from __future__ import annotations

import re
import statistics
from typing import Any

_STOP = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "is", "are", "be",
    "on", "at", "by", "as", "that", "this", "with", "per", "not", "no", "must",
    "shall", "any", "than", "which", "whichever", "from", "it", "its",
}
_WORD = re.compile(r"[a-z0-9]+")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 2}


def keyword_recall(answer: str, ground_truth: str) -> float:
    """Fraction of ground-truth content words present in the answer."""
    gt = _content_words(ground_truth)
    if not gt:
        return 0.0
    ans = _content_words(answer)
    return len(gt & ans) / len(gt)


def context_hit_rate(contexts: list[str], ground_truth: str, threshold: float = 0.4) -> float:
    """1.0 if any retrieved context lexically covers the ground truth answer."""
    gt = _content_words(ground_truth)
    if not gt:
        return 0.0
    for ctx in contexts:
        overlap = len(gt & _content_words(ctx)) / len(gt)
        if overlap >= threshold:
            return 1.0
    return 0.0


def refusal_correct(refused: bool, expected_source: str) -> bool:
    """A row with expected_source == NONE should be refused; others should not."""
    should_refuse = (expected_source or "").upper() == "NONE"
    return refused == should_refuse


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    def mean(key: str) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.mean(vals), 4) if vals else 0.0

    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    p95 = 0.0
    if latencies:
        s = sorted(latencies)
        p95 = round(s[min(len(s) - 1, int(0.95 * len(s)))], 2)

    return {
        "n": len(rows),
        "keyword_recall": mean("keyword_recall"),
        "context_hit_rate": mean("context_hit_rate"),
        "refusal_accuracy": round(
            sum(1 for r in rows if r.get("refusal_correct")) / len(rows), 4
        ) if rows else 0.0,
        "faithfulness": mean("faithfulness"),
        "answer_relevancy": mean("answer_relevancy"),
        "context_precision": mean("context_precision"),
        "latency_ms_mean": mean("latency_ms"),
        "latency_ms_p95": p95,
        "retrieval_ms_mean": mean("retrieval_ms"),
        "generation_ms_mean": mean("generation_ms"),
    }
