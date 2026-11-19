"""The agentic RAG loop.

"Agentic" here means the pipeline makes decisions rather than running a fixed
chain:

  1. Guardrail the input (and flag possible prompt injection).
  2. Retrieve (hybrid) -> rerank.
  3. DECIDE: is the top context relevant enough to answer?
       * If not, REWRITE the query and retrieve once more. The LLM does the
         rewrite when one is configured; otherwise a deterministic acronym
         expansion is used.
  4. If still ungrounded -> refuse (no hallucinated answer).
  5. Otherwise generate a cited answer. If the LLM is unavailable (no key, rate
     limit, timeout) DEGRADE to an extractive answer instead of failing.
  6. Sanitize the output.

Every stage is a span, so the whole decision tree shows up in Phoenix/LangSmith.
"""
from __future__ import annotations

import re
import time
from functools import lru_cache

from ragsys.config import settings
from ragsys.generation import (
    REFUSAL,
    Generator,
    build_citations,
    check_input,
    extractive_answer,
    grounded_enough,
    relevance,
    sanitize_output,
)
from ragsys.llm.provider import LLMError
from ragsys.models import QueryResponse, RetrievedChunk, SourcePassage
from ragsys.retrieval import HybridRetriever, get_reranker
from ragsys.telemetry import SpanKind, current_trace_id, span

# Deterministic fallback for query expansion: acronym -> expansion.
_ACRONYMS = {
    r"\bkyc\b": "know your customer",
    r"\bria\b": "registered investment adviser",
    r"\brpt\b": "related party transaction",
    r"\bapr\b": "annual percentage rate",
    r"\baml\b": "anti money laundering",
    r"\bfpi\b": "foreign portfolio investor",
    r"\breit\b": "real estate investment trust",
    r"\binvit\b": "infrastructure investment trust",
    r"\bppi\b": "prepaid payment instrument",
    r"\bpa\b": "payment aggregator",
    r"\bmii\b": "market infrastructure institution",
    r"\bisin\b": "international securities identification number",
    r"\bfatca\b": "foreign account tax compliance act",
    r"\bnbfc\b": "non-banking financial company",
    r"\blodr\b": "listing obligations and disclosure requirements",
}

_NOTICES = {
    "not_configured": "No LLM key is configured, so this answer quotes the source passages directly.",
    "rate_limited": "The LLM provider's free quota is exhausted right now, so this answer quotes the source passages directly.",
    "timeout": "The LLM provider timed out, so this answer quotes the source passages directly.",
    "refused": "The model declined to answer this question.",
    "error": "The LLM provider returned an error, so this answer quotes the source passages directly.",
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

    # -- retrieval ---------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        top_n: int | None = None,
        mode: str = "hybrid",
        rerank: bool = True,
    ) -> list[RetrievedChunk]:
        """Retrieve and (optionally) rerank. `mode` / `rerank` exist for the eval ablation."""
        candidates = self.retriever.retrieve(query, top_k=top_k, mode=mode)
        if rerank:
            return self.reranker.rerank(query, candidates, top_n=top_n)
        return candidates[: top_n or settings.rerank_top_n]

    def _rewrite(self, question: str, provider: str) -> str | None:
        """A better search query for `question`, or None if there isn't one."""
        if settings.llm_query_rewrite and settings.llm_available(provider):
            rewritten = Generator(provider).rewrite(question)
            if rewritten:
                return rewritten
        expanded = rewrite_query(question)
        return expanded if expanded != question else None

    # -- the loop ----------------------------------------------------------
    def answer(
        self,
        question: str,
        top_k: int | None = None,
        rerank_top_n: int | None = None,
        provider: str | None = None,
    ) -> QueryResponse:
        t0 = time.perf_counter()
        provider = provider or self.default_provider or settings.llm_provider
        rerank_top_n = rerank_top_n or settings.rerank_top_n

        with span("agentic_rag", kind=SpanKind.AGENT,
                  **{"input.value": question, "llm.provider": provider}) as root:
            guard = check_input(question)
            if not guard.ok:
                root.set("agent.refused", True)
                return self._refuse(provider, t0, 0.0, reason=guard.reason)

            # --- attempt 1 ---
            r0 = time.perf_counter()
            contexts = self.retrieve(guard.cleaned, top_k, rerank_top_n)
            grounded = grounded_enough(contexts)
            root.set("agent.grounded_first_pass", grounded)

            # --- decide: rewrite + retry if ungrounded ---
            rewritten = None
            if not grounded:
                rewritten = self._rewrite(guard.cleaned, provider)
                if rewritten:
                    root.set("agent.rewritten_query", rewritten)
                    contexts = self.retrieve(rewritten, top_k, rerank_top_n)
                    grounded = grounded_enough(contexts)
                    root.set("agent.grounded_after_rewrite", grounded)
            retrieval_ms = (time.perf_counter() - r0) * 1000

            if not grounded:
                root.set("agent.refused", True)
                root.set("output.value", REFUSAL)
                return self._refuse(provider, t0, retrieval_ms, rewritten=rewritten,
                                    contexts=contexts)

            # --- generate, degrading to extractive if the LLM is unavailable ---
            g0 = time.perf_counter()
            notice = None
            mode = "generated"
            try:
                if not settings.llm_available(provider):
                    raise LLMError("no LLM configured", kind="not_configured")
                gen = Generator(provider).generate(
                    guard.cleaned, contexts, injection_flagged=guard.injection_flagged
                )
            except LLMError as exc:
                root.set("agent.llm_error", exc.kind)
                if exc.kind == "refused":
                    return self._refuse(provider, t0, retrieval_ms, rewritten=rewritten,
                                        contexts=contexts, notice=_NOTICES["refused"])
                gen = extractive_answer(guard.cleaned, contexts)
                notice = _NOTICES.get(exc.kind, _NOTICES["error"])
                mode = "extractive"
            generation_ms = (time.perf_counter() - g0) * 1000

            answer = sanitize_output(gen.answer)
            refused = answer.strip().startswith("I don't have enough information")
            if refused:
                mode = "refused"
            cited = set() if refused else {c.marker for c in gen.citations}
            total_ms = (time.perf_counter() - t0) * 1000
            root.set("output.value", answer[:2000])
            root.set("agent.refused", refused)
            root.set("agent.mode", mode)

            return QueryResponse(
                answer=answer,
                mode=mode,
                notice=notice,
                refused=refused,
                citations=[] if refused else gen.citations,
                sources=_sources(contexts, cited),
                contexts=[c.chunk.text for c in contexts],
                rewritten_query=rewritten,
                provider=gen.provider,
                model=gen.model,
                latency_ms=round(total_ms, 2),
                retrieval_ms=round(retrieval_ms, 2),
                generation_ms=round(generation_ms, 2),
                trace_id=current_trace_id(),
            )

    def _refuse(
        self,
        provider: str,
        t0: float,
        retrieval_ms: float,
        reason: str | None = None,
        rewritten: str | None = None,
        contexts: list[RetrievedChunk] | None = None,
        notice: str | None = None,
    ) -> QueryResponse:
        if reason == "empty_question":
            msg = "Please provide a question."
        elif reason == "too_long":
            msg = f"Please keep the question under {settings.max_input_chars} characters."
        else:
            msg = REFUSAL
        return QueryResponse(
            answer=msg,
            mode="refused",
            notice=notice,
            refused=True,
            citations=[],
            # Show what was found even on a refusal: it explains *why* we refused.
            sources=_sources(contexts or [], set()),
            contexts=[],
            rewritten_query=rewritten,
            provider=provider,
            model=settings.model_for(provider),
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=0.0,
            trace_id=current_trace_id(),
        )


def _sources(contexts: list[RetrievedChunk], cited: set[str]) -> list[SourcePassage]:
    return [
        SourcePassage(
            marker=cit.marker,
            doc_id=cit.doc_id,
            title=cit.title,
            source=cit.source,
            url=cit.url,
            circular_no=cit.circular_no,
            text=rc.chunk.text,
            relevance=round(relevance(rc), 4),
            cited=cit.marker in cited,
        )
        for rc, cit in zip(contexts, build_citations(contexts))
    ]


@lru_cache
def get_pipeline() -> AgenticRAG:
    return AgenticRAG()
