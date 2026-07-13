"""The agentic RAG loop.

"Agentic" here means the pipeline makes decisions rather than running a fixed
chain:

  1. Guardrail the input (and flag possible prompt injection).
  2. Retrieve (hybrid) -> rerank.
  3. DECIDE: is the top context relevant enough to answer?
       * If not, REWRITE the query (expand acronyms / add regulator terms) and
         retrieve once more before giving up.
  4. If still ungrounded -> refuse (no hallucinated answer).
  5. Otherwise generate a cited answer, then sanitize the output.

Every stage is a span, so the whole decision tree shows up in Phoenix/LangSmith.
"""
from __future__ import annotations

import re
import time
from functools import lru_cache

from ragsys.config import settings
from ragsys.generation import Generator, check_input, grounded_enough, sanitize_output
from ragsys.models import QueryResponse, RetrievedChunk
from ragsys.retrieval import HybridRetriever, get_reranker
from ragsys.telemetry import SpanKind, current_trace_id, span

REFUSAL = "I don't have enough information in the provided circulars to answer that."

# Lightweight domain query-expansion: acronym -> expansion appended to the query.
_ACRONYMS = {
    r"\bkyc\b": "know your customer",
    r"\bria\b": "registered investment adviser",
    r"\brpt\b": "related party transaction",
    r"\bapr\b": "annual percentage rate",
    r"\baml\b": "anti money laundering",
    r"\baua\b": "assets under advice",
    r"\blodr\b": "listing obligations and disclosure requirements",
}


def rewrite_query(question: str) -> str:
    """Deterministic query expansion: append expanded regulator acronyms."""
    additions = []
    low = question.lower()
    for pat, expansion in _ACRONYMS.items():
        if re.search(pat, low) and expansion not in low:
            additions.append(expansion)
    if additions:
        return f"{question} ({'; '.join(additions)})"
    return question


class AgenticRAG:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker=None,
        provider: str | None = None,
    ) -> None:
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or get_reranker()
        self.default_provider = provider

    def _retrieve_and_rank(
        self, query: str, top_k: int | None, top_n: int | None
    ) -> list[RetrievedChunk]:
        candidates = self.retriever.retrieve(query, top_k=top_k)
        return self.reranker.rerank(query, candidates, top_n=top_n)

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        rerank_top_n: int | None = None,
        provider: str | None = None,
    ) -> QueryResponse:
        t0 = time.perf_counter()
        provider = provider or self.default_provider
        rerank_top_n = rerank_top_n or settings.rerank_top_n

        with span("agentic_rag", kind=SpanKind.AGENT,
                  **{"input.value": question, "llm.provider": provider or settings.llm_provider}) as root:
            guard = check_input(question)
            if not guard.ok:
                return self._refuse(question, provider, t0, 0.0, reason=guard.reason)

            # --- attempt 1 ---
            r0 = time.perf_counter()
            contexts = self._retrieve_and_rank(guard.cleaned, top_k, rerank_top_n)
            grounded = grounded_enough(contexts)
            root.set("agent.grounded_first_pass", grounded)

            # --- decide: rewrite + retry if ungrounded ---
            if not grounded:
                rewritten = rewrite_query(guard.cleaned)
                if rewritten != guard.cleaned:
                    root.set("agent.rewritten_query", rewritten)
                    contexts = self._retrieve_and_rank(rewritten, top_k, rerank_top_n)
                    grounded = grounded_enough(contexts)
                    root.set("agent.grounded_after_rewrite", grounded)
            retrieval_ms = (time.perf_counter() - r0) * 1000

            if not grounded:
                return self._refuse(question, provider, t0, retrieval_ms)

            # --- generate ---
            g0 = time.perf_counter()
            gen = Generator(provider).generate(
                guard.cleaned, contexts, injection_flagged=guard.injection_flagged
            )
            generation_ms = (time.perf_counter() - g0) * 1000

            answer = sanitize_output(gen.answer)
            refused = answer.strip().startswith("I don't have enough information")
            total_ms = (time.perf_counter() - t0) * 1000
            root.set("output.value", answer[:2000])
            root.set("agent.refused", refused)

            return QueryResponse(
                answer=answer,
                citations=[] if refused else gen.citations,
                contexts=[c.chunk.text for c in contexts],
                refused=refused,
                provider=gen.provider,
                model=gen.model,
                latency_ms=round(total_ms, 2),
                retrieval_ms=round(retrieval_ms, 2),
                generation_ms=round(generation_ms, 2),
                trace_id=current_trace_id(),
            )

    def _refuse(
        self, question: str, provider: str | None, t0: float,
        retrieval_ms: float, reason: str | None = None,
    ) -> QueryResponse:
        prov = provider or settings.llm_provider
        model = settings.anthropic_model if prov == "anthropic" else settings.openai_model
        msg = REFUSAL if reason != "empty_question" else "Please provide a question."
        return QueryResponse(
            answer=msg,
            citations=[],
            contexts=[],
            refused=True,
            provider=prov,
            model=model,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=0.0,
            trace_id=current_trace_id(),
        )


@lru_cache
def get_pipeline() -> AgenticRAG:
    return AgenticRAG()
