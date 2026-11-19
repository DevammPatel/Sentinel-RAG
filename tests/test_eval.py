import json
from pathlib import Path

import pytest

from ragsys.eval.calibrate import sweep
from ragsys.eval.metrics import (
    evidence_rank,
    has_citation,
    hit_at,
    keyword_recall,
    reciprocal_rank,
    refusal_correct,
    refusal_stats,
)
from ragsys.eval.run_eval import load_dataset

GOLDEN = Path(__file__).parent.parent / "data" / "golden" / "qa.jsonl"
INDEX = Path(__file__).parent.parent / "data" / "index" / "chunks.jsonl"


def test_evidence_rank_ignores_whitespace_case_and_quote_style():
    contexts = ["nothing here", "The merchant’s  KYC record\nfrom CKYCR shall be used."]
    assert evidence_rank(contexts, "merchant's kyc record from ckycr") == 2
    assert evidence_rank(contexts, "not present") is None
    assert evidence_rank(contexts, None) is None


def test_rank_metrics():
    assert (hit_at(1, 1), hit_at(3, 1), hit_at(3, 5), hit_at(None, 5)) == (1.0, 0.0, 1.0, 0.0)
    assert (reciprocal_rank(1), reciprocal_rank(4), reciprocal_rank(None)) == (1.0, 0.25, 0.0)


def test_keyword_recall_counts_content_words_only():
    assert keyword_recall("closed within seven working days", "Within seven working days.") == 1.0
    assert keyword_recall("no idea", "Within seven working days.") == 0.0
    assert keyword_recall("anything", "") == 0.0


def test_citation_detection():
    assert has_citation("Seven days [2].") and not has_citation("Seven days.")


def test_refusal_stats_treat_should_refuse_as_positive():
    rows = [
        {"refused": True, "answerable": False},   # correct refusal
        {"refused": False, "answerable": True},   # correct answer
        {"refused": True, "answerable": True},    # wrongly refused
        {"refused": False, "answerable": False},  # wrongly answered
    ]
    stats = refusal_stats(rows)
    assert stats["refusal_accuracy"] == 0.5
    assert (stats["refusal_precision"], stats["refusal_recall"]) == (0.5, 0.5)
    assert (stats["wrongly_refused"], stats["wrongly_answered"]) == (1, 1)
    assert refusal_correct(True, False) and not refusal_correct(True, True)


def test_threshold_sweep_finds_the_separating_range():
    scores = [(0.9, True), (0.8, True), (0.1, False), (0.05, False)]
    by_threshold = {r["threshold"]: r for r in sweep(scores)}
    assert by_threshold[0.5]["balanced_accuracy"] == 1.0
    assert by_threshold[0.05]["wrongly_answered"] == 2   # too low: nothing refused
    assert by_threshold[0.95]["wrongly_refused"] == 2    # too high: everything refused


def test_golden_set_is_well_formed():
    rows = load_dataset(GOLDEN)
    assert len(rows) >= 50
    assert len({r["id"] for r in rows}) == len(rows), "duplicate ids"
    assert len({r["question"] for r in rows}) == len(rows), "duplicate questions"
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    assert len(unanswerable) >= 10, "need enough negatives to measure refusals"
    for r in answerable:
        assert r["evidence"] and r["expected_doc_ids"] and r["ground_truth"]
    for r in unanswerable:
        assert r["evidence"] is None and r["expected_source"] == "NONE"


@pytest.mark.skipif(not INDEX.exists(), reason="index not built (run `make ingest`)")
def test_every_golden_evidence_quote_exists_in_the_indexed_corpus():
    """Guards the eval itself: a question whose evidence is not in the index can
    never be answered, and would silently drag every retrieval metric down."""
    chunks = [json.loads(line) for line in INDEX.read_text(encoding="utf-8").splitlines() if line]
    by_doc: dict[str, list[str]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c["text"])
    for r in load_dataset(GOLDEN):
        if not r["answerable"]:
            continue
        found = any(evidence_rank(by_doc.get(doc, []), r["evidence"]) for doc in r["expected_doc_ids"])
        assert found, f"{r['id']}: evidence not found in {r['expected_doc_ids']}"
