"""Deterministic, model-free metrics.

Retrieval is scored against an `evidence` quote: a short verbatim span from the
source document that contains the answer. A retrieved passage is a hit when it
contains that span. That is stricter and more honest than document-level
matching (a 400-chunk direction "matches" almost anything) and needs no LLM.
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


def normalise(text: str) -> str:
    """Lowercase, straighten quotes and collapse whitespace for span matching."""
    text = (text or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 2}


# ---- retrieval -------------------------------------------------------------
def evidence_rank(contexts: list[str], evidence: str | None) -> int | None:
    """1-based rank of the first passage containing the evidence span, else None."""
    if not evidence:
        return None
    needle = normalise(evidence)
    for rank, ctx in enumerate(contexts, start=1):
        if needle in normalise(ctx):
            return rank
    return None


def hit_at(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


# ---- answers ---------------------------------------------------------------
def keyword_recall(answer: str, ground_truth: str) -> float:
    """Fraction of ground-truth content words present in the answer."""
    gt = _content_words(ground_truth)
    if not gt:
        return 0.0
    return len(gt & _content_words(answer)) / len(gt)


def refusal_correct(refused: bool, answerable: bool) -> bool:
    """Unanswerable rows should be refused; answerable rows should not."""
    return refused == (not answerable)


def has_citation(answer: str) -> bool:
    return bool(re.search(r"\[\d+\]", answer or ""))


# ---- aggregation -----------------------------------------------------------
def mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return round(s[min(len(s) - 1, int(p * len(s)))], 2)


def refusal_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Refusal quality, treating "should refuse" as the positive class."""
    tp = sum(1 for r in rows if r["refused"] and not r["answerable"])
    fp = sum(1 for r in rows if r["refused"] and r["answerable"])
    fn = sum(1 for r in rows if not r["refused"] and not r["answerable"])
    correct = sum(1 for r in rows if refusal_correct(r["refused"], r["answerable"]))
    return {
        "refusal_accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "refusal_precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "refusal_recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "wrongly_refused": fp,
        "wrongly_answered": fn,
    }
