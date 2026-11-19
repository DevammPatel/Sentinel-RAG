"""Eval harness runner.

Two modes, both over the golden JSONL dataset:

  retrieval (default, no LLM, no key)
      An ablation across retrieval configurations: BM25 only, dense only,
      hybrid (RRF), and hybrid + cross-encoder rerank. Reports hit@1 / hit@5 /
      MRR against the evidence quotes, latency, and how well the grounding gate
      separates answerable from unanswerable questions.

  full (runs the whole agent, uses the configured LLM if there is one)
      Refusal accuracy, citation rate, keyword recall against the reference
      answer, latency split, and an LLM-judged faithfulness score when an LLM is
      available. Add --ragas to also compute RAGAS metrics.

  python -m ragsys.eval.run_eval                       # retrieval ablation
  python -m ragsys.eval.run_eval --mode full           # end to end
  python -m ragsys.eval.run_eval --mode full --provider gemini
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ragsys.eval.metrics import (
    evidence_rank,
    has_citation,
    hit_at,
    keyword_recall,
    mean,
    percentile,
    reciprocal_rank,
    refusal_stats,
)

# name -> (retrieval mode, rerank?)
CONFIGS: dict[str, tuple[str, bool]] = {
    "bm25_only": ("sparse", False),
    "dense_only": ("dense", False),
    "hybrid_rrf": ("hybrid", False),
    "bm25+rerank": ("sparse", True),
    "dense+rerank": ("dense", True),
    "hybrid_rrf+rerank": ("hybrid", True),
}

JUDGE_PROMPT = (
    "You are grading a question-answering system. You are given CONTEXT passages and "
    "an ANSWER. Decide whether every factual claim in the ANSWER is directly supported "
    "by the CONTEXT. Ignore citation markers like [1]. Reply with exactly one word: "
    "SUPPORTED or UNSUPPORTED."
)


def load_dataset(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            row.setdefault("answerable", (row.get("expected_source") or "").upper() != "NONE")
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# retrieval ablation                                                          #
# --------------------------------------------------------------------------- #
def run_retrieval(dataset: list[dict], top_n: int = 5) -> dict:
    from ragsys.agent import AgenticRAG
    from ragsys.config import settings
    from ragsys.generation import top_relevance

    pipeline = AgenticRAG()
    pipeline.retrieve("warm up")  # load models before timing anything
    answerable = [r for r in dataset if r["answerable"]]
    summary: dict[str, dict] = {}
    rows_out: list[dict] = []

    for name, (mode, rerank) in CONFIGS.items():
        ranks, latencies = [], []
        for row in answerable:
            t0 = time.perf_counter()
            contexts = pipeline.retrieve(row["question"], top_n=top_n, mode=mode, rerank=rerank)
            latencies.append((time.perf_counter() - t0) * 1000)
            rank = evidence_rank([c.chunk.text for c in contexts], row.get("evidence"))
            ranks.append(rank)
            rows_out.append({"config": name, "id": row.get("id"), "question": row["question"],
                             "evidence_rank": rank})
        summary[name] = {
            "hit@1": mean([hit_at(r, 1) for r in ranks]),
            f"hit@{top_n}": mean([hit_at(r, top_n) for r in ranks]),
            "mrr": mean([reciprocal_rank(r) for r in ranks]),
            "latency_ms_mean": mean(latencies),
            "latency_ms_p95": percentile(latencies, 0.95),
        }
        print(f"  {name:<20} " + "  ".join(f"{k}={v}" for k, v in summary[name].items()))
        # Break hit@1 down by question style: plain-language questions and
        # reference-number lookups stress different retrievers.
        by_style: dict[str, list[int | None]] = {}
        for row, rank in zip(answerable, ranks):
            by_style.setdefault(row.get("style", "direct"), []).append(rank)
        summary[name]["by_style"] = {
            style: {"n": len(rs), "hit@1": mean([hit_at(r, 1) for r in rs]),
                    f"hit@{top_n}": mean([hit_at(r, top_n) for r in rs])}
            for style, rs in sorted(by_style.items())
        }

    # Grounding gate: does the top relevance score separate the two classes?
    gate_rows = []
    for row in dataset:
        contexts = pipeline.retrieve(row["question"], top_n=top_n)
        score = top_relevance(contexts)
        gate_rows.append({"id": row.get("id"), "answerable": row["answerable"],
                          "top_relevance": round(score, 4),
                          "refused": score < settings.min_context_score})
    gate = {"threshold": settings.min_context_score, **refusal_stats(gate_rows)}
    print("  hit@1 by question style:")
    for name, stats in summary.items():
        print(f"    {name:<20} " + "  ".join(
            f"{style}={v['hit@1']} (n={v['n']})" for style, v in stats["by_style"].items()))
    print(f"  grounding gate @ {gate['threshold']}: " +
          "  ".join(f"{k}={v}" for k, v in gate.items() if k != "threshold"))

    return {"n_answerable": len(answerable), "n_total": len(dataset),
            "ablation": summary, "grounding_gate": gate,
            "rows": rows_out, "gate_rows": gate_rows}


# --------------------------------------------------------------------------- #
# full pipeline                                                               #
# --------------------------------------------------------------------------- #
def judge_faithfulness(answer: str, contexts: list[str], provider: str | None) -> float | None:
    """1.0 if an LLM judge finds the answer fully supported by the contexts."""
    from ragsys.llm import get_llm
    from ragsys.llm.provider import LLMError

    user = "CONTEXT:\n" + "\n\n".join(contexts) + f"\n\nANSWER:\n{answer}"
    try:
        verdict = get_llm(provider).complete(JUDGE_PROMPT, user, max_tokens=10).text.upper()
    except LLMError as exc:
        print(f"    [judge] skipped: {exc}")
        return None
    if "UNSUPPORTED" in verdict:
        return 0.0
    return 1.0 if "SUPPORTED" in verdict else None


def run_full(dataset: list[dict], provider: str | None, pause_s: float = 0.0) -> dict:
    from ragsys.agent import AgenticRAG
    from ragsys.config import settings

    provider = provider or settings.llm_provider
    llm_on = settings.llm_available(provider)
    pipeline = AgenticRAG(provider=provider)
    pipeline.retrieve("warm up")
    rows = []
    for i, row in enumerate(dataset, 1):
        resp = pipeline.answer(row["question"])
        rec = {
            "id": row.get("id"),
            "question": row["question"],
            "answerable": row["answerable"],
            "ground_truth": row.get("ground_truth", ""),
            "answer": resp.answer,
            "mode": resp.mode,
            "model": resp.model,
            "refused": resp.refused,
            "contexts": resp.contexts,
            "latency_ms": resp.latency_ms,
            "retrieval_ms": resp.retrieval_ms,
            "generation_ms": resp.generation_ms,
            "evidence_rank": evidence_rank(resp.contexts, row.get("evidence")),
            "keyword_recall": keyword_recall(resp.answer, row.get("ground_truth", "")),
            "cited": has_citation(resp.answer),
        }
        if llm_on and resp.mode == "generated":
            rec["faithfulness"] = judge_faithfulness(resp.answer, resp.contexts, provider)
        rows.append(rec)
        print(f"  [{i}/{len(dataset)}] mode={resp.mode:<10} refused={resp.refused!s:<5} "
              f"{row['question'][:60]!r}")
        if pause_s:
            time.sleep(pause_s)  # stay under free-tier rate limits

    answered = [r for r in rows if r["answerable"] and not r["refused"]]
    judged = [r["faithfulness"] for r in rows if r.get("faithfulness") is not None]
    latencies = [r["latency_ms"] for r in rows]
    summary = {
        "n": len(rows),
        "answer_modes": {m: sum(1 for r in rows if r["mode"] == m)
                         for m in ("generated", "extractive", "refused")},
        **refusal_stats(rows),
        "citation_rate": mean([1.0 if r["cited"] else 0.0 for r in answered]),
        "keyword_recall": mean([r["keyword_recall"] for r in answered]),
        "context_hit@5": mean([hit_at(r["evidence_rank"], 5) for r in rows if r["answerable"]]),
        "faithfulness_llm_judged": mean(judged) if judged else None,
        "latency_ms_mean": mean(latencies),
        "latency_ms_p95": percentile(latencies, 0.95),
        "retrieval_ms_mean": mean([r["retrieval_ms"] for r in rows]),
        "generation_ms_mean": mean([r["generation_ms"] for r in rows]),
    }
    return {"provider": provider if llm_on else "none (extractive)",
            "summary": summary, "rows": rows}


def run_ragas(rows: list[dict], provider: str | None) -> dict | None:
    """Optional RAGAS metrics, judged by the configured OpenAI-compatible LLM.

    Needs `pip install -r requirements-eval.txt` and an LLM key. RAGAS makes
    several LLM calls per row, so expect to hit free-tier rate limits on a large
    dataset; that is why it is opt-in.
    """
    try:
        from langchain_openai import ChatOpenAI
        from ragas import EvaluationDataset, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, LLMContextRecall
    except Exception as exc:
        print(f"[eval] RAGAS not available ({exc}); skipping.")
        return None

    from ragsys.config import OPENAI_COMPATIBLE_BASE_URLS, settings

    provider = provider or settings.llm_provider
    if provider == "anthropic" or not settings.llm_available(provider):
        print("[eval] RAGAS needs an OpenAI-compatible judge (groq, gemini or openai); skipping.")
        return None
    gradable = [r for r in rows if r["mode"] == "generated" and r["contexts"]]
    if not gradable:
        print("[eval] no generated answers to grade with RAGAS; skipping.")
        return None
    judge = LangchainLLMWrapper(ChatOpenAI(
        model=settings.model_for(provider),
        api_key=settings.api_key_for(provider),
        base_url=OPENAI_COMPATIBLE_BASE_URLS.get(provider) or settings.openai_base_url,
        temperature=0,
    ))
    dataset = EvaluationDataset.from_list([
        {"user_input": r["question"], "response": r["answer"],
         "retrieved_contexts": r["contexts"], "reference": r["ground_truth"]}
        for r in gradable
    ])
    try:
        result = evaluate(dataset=dataset, metrics=[Faithfulness(), LLMContextRecall()], llm=judge)
        df = result.to_pandas()
        return {col: round(float(df[col].mean()), 4)
                for col in df.columns if df[col].dtype.kind == "f"}
    except Exception as exc:
        print(f"[eval] RAGAS evaluation failed ({exc}); continuing without it.")
        return None


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Run the RAG eval harness.")
    ap.add_argument("--dataset", default="data/golden/qa.jsonl")
    ap.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    ap.add_argument("--provider", default=None,
                    choices=["groq", "gemini", "openai", "anthropic", "none"])
    ap.add_argument("--ragas", action="store_true", help="also compute RAGAS metrics (full mode)")
    ap.add_argument("--pause", type=float, default=0.0,
                    help="seconds to wait between questions in full mode (free-tier limits)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ragsys.config import settings
    from ragsys.telemetry import init_tracing

    init_tracing()
    dataset = load_dataset(args.dataset)
    print(f"[eval] {len(dataset)} questions, mode={args.mode}")

    if args.mode == "retrieval":
        body = run_retrieval(dataset)
    else:
        body = run_full(dataset, args.provider, pause_s=args.pause)
        if args.ragas:
            body["ragas"] = run_ragas(body["rows"], args.provider)
        print("\n=== SUMMARY ===")
        for k, v in body["summary"].items():
            print(f"  {k:>26}: {v}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": args.mode,
        "dataset": args.dataset,
        "embedding_model": settings.local_embedding_model,
        "reranker_model": settings.reranker_model,
        **body,
    }
    out = Path(args.out or f"data/eval/{args.mode}_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nFull report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
