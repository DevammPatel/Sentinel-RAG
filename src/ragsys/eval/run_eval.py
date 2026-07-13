"""Eval harness runner.

Runs the full agentic pipeline over a golden JSONL dataset, computes:
  * custom deterministic metrics (always) — keyword recall, context hit-rate,
    refusal accuracy, and latency percentiles;
  * RAGAS metrics (faithfulness, answer_relevancy, context_precision) IF ragas
    and an LLM/embeddings key are available — otherwise skipped gracefully.

Writes a JSON report and prints a summary table. Use --provider to A/B the
Anthropic vs OpenAI backends on the same dataset.

  python -m ragsys.eval.run_eval --dataset data/golden/qa.jsonl --provider anthropic
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ragsys.eval.metrics import (
    aggregate,
    context_hit_rate,
    keyword_recall,
    refusal_correct,
)


def load_dataset(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def run_pipeline(dataset: list[dict], provider: str | None) -> list[dict]:
    from ragsys.agent import AgenticRAG

    pipeline = AgenticRAG(provider=provider)
    results = []
    for i, row in enumerate(dataset, 1):
        resp = pipeline.answer(row["question"], provider=provider)
        rec = {
            "question": row["question"],
            "ground_truth": row.get("ground_truth", ""),
            "expected_source": row.get("expected_source", ""),
            "answer": resp.answer,
            "contexts": resp.contexts,
            "refused": resp.refused,
            "latency_ms": resp.latency_ms,
            "retrieval_ms": resp.retrieval_ms,
            "generation_ms": resp.generation_ms,
        }
        rec["keyword_recall"] = keyword_recall(resp.answer, rec["ground_truth"])
        rec["context_hit_rate"] = context_hit_rate(resp.contexts, rec["ground_truth"])
        rec["refusal_correct"] = refusal_correct(resp.refused, rec["expected_source"])
        results.append(rec)
        print(f"  [{i}/{len(dataset)}] {row['question'][:60]!r} "
              f"recall={rec['keyword_recall']:.2f} refused={resp.refused}")
    return results


def run_ragas(results: list[dict]) -> None:
    """Attach RAGAS scores in-place. Skips silently if unavailable/no keys."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except Exception as exc:
        print(f"[eval] RAGAS not available ({exc}); skipping model-graded metrics.")
        return

    # RAGAS needs non-refused rows with contexts + a reference answer.
    gradable = [r for r in results if not r["refused"] and r["contexts"]]
    if not gradable:
        print("[eval] no gradable rows for RAGAS; skipping.")
        return
    ds = Dataset.from_dict({
        "question": [r["question"] for r in gradable],
        "answer": [r["answer"] for r in gradable],
        "contexts": [r["contexts"] for r in gradable],
        "reference": [r["ground_truth"] for r in gradable],
    })
    try:
        report = evaluate(
            ds, metrics=[faithfulness, answer_relevancy, context_precision]
        )
        df = report.to_pandas()
        for r, (_, row) in zip(gradable, df.iterrows()):
            r["faithfulness"] = float(row.get("faithfulness", float("nan")))
            r["answer_relevancy"] = float(row.get("answer_relevancy", float("nan")))
            r["context_precision"] = float(row.get("context_precision", float("nan")))
        print("[eval] RAGAS metrics computed.")
    except Exception as exc:
        print(f"[eval] RAGAS evaluation failed ({exc}); continuing with custom metrics.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the RAG eval harness.")
    ap.add_argument("--dataset", default="data/golden/qa.jsonl")
    ap.add_argument("--provider", default=None, choices=[None, "anthropic", "openai"])
    ap.add_argument("--no-ragas", action="store_true", help="skip RAGAS metrics")
    ap.add_argument("--out", default="data/eval_report.json")
    args = ap.parse_args()

    from ragsys.telemetry import init_tracing

    init_tracing()

    dataset = load_dataset(args.dataset)
    print(f"[eval] running {len(dataset)} questions "
          f"(provider={args.provider or 'default'})...")
    results = run_pipeline(dataset, args.provider)

    if not args.no_ragas:
        run_ragas(results)

    summary = aggregate(results)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider or "default",
        "dataset": args.dataset,
        "summary": summary,
        "rows": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== EVAL SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:>20}: {v}")
    print(f"\nFull report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
