"""Pick MIN_CONTEXT_SCORE from data instead of guessing.

For every golden question we record the best relevance score retrieval produced
(the same signal the grounding gate reads). A good threshold sits between the
answerable questions (which should score high) and the unanswerable ones (which
should score low). This sweeps thresholds and prints how each one would do.

  python -m ragsys.eval.calibrate
"""
from __future__ import annotations

import argparse
import sys

from ragsys.eval.metrics import refusal_stats
from ragsys.eval.run_eval import load_dataset


def sweep(scores: list[tuple[float, bool]]) -> list[dict]:
    """`scores` is (top_relevance, answerable) per question."""
    out = []
    for i in range(1, 20):
        threshold = round(i * 0.05, 2)
        rows = [{"refused": s < threshold, "answerable": a} for s, a in scores]
        stats = refusal_stats(rows)
        n_ans = sum(1 for _, a in scores if a) or 1
        n_un = sum(1 for _, a in scores if not a) or 1
        balanced = ((n_ans - stats["wrongly_refused"]) / n_ans
                    + (n_un - stats["wrongly_answered"]) / n_un) / 2
        out.append({"threshold": threshold, "balanced_accuracy": round(balanced, 4), **stats})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate the grounding threshold.")
    ap.add_argument("--dataset", default="data/golden/qa.jsonl")
    args = ap.parse_args()

    from ragsys.agent import AgenticRAG
    from ragsys.generation import top_relevance

    dataset = load_dataset(args.dataset)
    pipeline = AgenticRAG()
    scores = [(top_relevance(pipeline.retrieve(r["question"])), r["answerable"]) for r in dataset]

    ans = sorted(s for s, a in scores if a)
    un = sorted(s for s, a in scores if not a)
    print(f"answerable   (n={len(ans)}): min={ans[0]:.3f} p10={ans[len(ans) // 10]:.3f} "
          f"median={ans[len(ans) // 2]:.3f}")
    print(f"unanswerable (n={len(un)}): median={un[len(un) // 2]:.3f} max={un[-1]:.3f}\n")
    print(f"{'threshold':>9} {'balanced':>9} {'wrongly_refused':>16} {'wrongly_answered':>17}")
    results = sweep(scores)
    for r in results:
        print(f"{r['threshold']:>9} {r['balanced_accuracy']:>9} "
              f"{r['wrongly_refused']:>16} {r['wrongly_answered']:>17}")
    best = max(results, key=lambda r: r["balanced_accuracy"])
    print(f"\nbest balanced accuracy: {best['balanced_accuracy']} at {best['threshold']}")

    # A threshold in the middle of a long run of identical outcomes is more
    # trustworthy than one that wins by a single question on a small dataset.
    runs: list[list[dict]] = []
    for r in results:
        key = (r["wrongly_refused"], r["wrongly_answered"])
        if runs and (runs[-1][0]["wrongly_refused"], runs[-1][0]["wrongly_answered"]) == key:
            runs[-1].append(r)
        else:
            runs.append([r])
    plateau = max(runs, key=len)
    mid = plateau[len(plateau) // 2]
    print(f"widest stable range: {plateau[0]['threshold']} to {plateau[-1]['threshold']} "
          f"(balanced accuracy {mid['balanced_accuracy']}); midpoint {mid['threshold']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
