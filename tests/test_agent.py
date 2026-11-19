"""Agent-loop tests. No network or model calls: real stores, fake embedder, stub LLM."""
import pytest
from conftest import NoopReranker

import ragsys.agent.orchestrator as orch
from ragsys.agent.orchestrator import AgenticRAG, rewrite_query
from ragsys.config import settings
from ragsys.generation import REFUSAL, GenerationOutput, build_citations
from ragsys.llm.provider import LLMError
from ragsys.models import Chunk, RetrievedChunk


class ScriptedReranker(NoopReranker):
    """Gives every candidate the relevance scripted for that query."""

    def __init__(self, scores: dict[str, float], default: float = 0.9):
        super().__init__()
        self.kind = "scripted"
        self.scores, self.default, self.queries = scores, default, []

    def rerank(self, query, candidates, top_n=None):
        self.queries.append(query)
        for c in candidates:
            c.rerank_score = self.scores.get(query, self.default)
        return candidates[: top_n or 5]


class StubGenerator:
    """Stands in for the LLM-backed Generator."""

    answer = "Low risk customers need KYC updation once in every ten years [1]."
    error: LLMError | None = None
    rewritten: str | None = None
    calls: list[str] = []

    def __init__(self, provider=None):
        self.provider = provider

    def generate(self, question, contexts, injection_flagged=False):
        type(self).calls.append(question)
        if self.error:
            raise self.error
        cits = build_citations(contexts)
        return GenerationOutput(answer=self.answer,
                                citations=[c for c in cits if c.marker in self.answer],
                                provider="groq", model="stub-model")

    def rewrite(self, question):
        return self.rewritten


@pytest.fixture
def llm(monkeypatch):
    """Pretend a Groq key is configured and route generation to the stub."""
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    StubGenerator.answer = "Low risk customers need KYC updation once in every ten years [1]."
    StubGenerator.error, StubGenerator.rewritten, StubGenerator.calls = None, None, []
    monkeypatch.setattr(orch, "Generator", StubGenerator)
    return StubGenerator


QUESTION = "How often is KYC updation needed for low risk customers?"


def test_rewrite_expands_known_acronyms():
    assert "know your customer" in rewrite_query("What are the KYC updation rules?").lower()


def test_rewrite_noop_when_no_acronym():
    q = "What net worth is required?"
    assert rewrite_query(q) == q


def test_hybrid_retrieval_finds_relevant_chunk(retriever):
    hits = retriever.retrieve("KYC updation for low risk customers", top_k=5)
    assert hits and "ten years" in hits[0].chunk.text


def test_empty_question_is_refused_without_retrieval(retriever):
    resp = AgenticRAG(retriever=retriever, reranker=NoopReranker()).answer("   ")
    assert resp.refused and resp.mode == "refused"
    assert resp.answer == "Please provide a question."
    assert resp.retrieval_ms == 0.0


def test_generated_answer_carries_citations_and_sources(retriever, llm):
    resp = AgenticRAG(retriever=retriever, reranker=ScriptedReranker({})).answer(QUESTION)
    assert resp.mode == "generated" and not resp.refused and resp.notice is None
    assert resp.answer == llm.answer
    assert (resp.provider, resp.model) == ("groq", "stub-model")
    assert [c.marker for c in resp.citations] == ["[1]"]
    assert resp.citations[0].url == "https://example.org/kyc"
    assert resp.citations[0].circular_no == "RBI/2025-26/1"
    assert resp.sources[0].cited is True and resp.sources[0].relevance == 0.9
    assert all(not s.cited for s in resp.sources[1:])
    assert resp.contexts == [s.text for s in resp.sources]


def test_no_llm_key_degrades_to_extractive_answer(retriever):
    # conftest configures LLM_PROVIDER=none: no key anywhere.
    resp = AgenticRAG(retriever=retriever, reranker=ScriptedReranker({})).answer(QUESTION)
    assert resp.mode == "extractive" and not resp.refused
    assert "ten years" in resp.answer and "[1]" in resp.answer
    assert "No LLM key" in resp.notice
    assert resp.citations and resp.model == "extractive"


@pytest.mark.parametrize("kind, phrase", [("rate_limited", "quota"), ("timeout", "timed out"),
                                          ("error", "returned an error")])
def test_llm_failure_degrades_to_extractive_instead_of_500(retriever, llm, kind, phrase):
    llm.error = LLMError("boom", kind=kind)
    resp = AgenticRAG(retriever=retriever, reranker=ScriptedReranker({})).answer(QUESTION)
    assert resp.mode == "extractive" and not resp.refused
    assert phrase in resp.notice
    assert "ten years" in resp.answer


def test_weak_retrieval_is_refused_and_llm_is_never_called(retriever, llm):
    weak = ScriptedReranker({}, default=0.02)
    resp = AgenticRAG(retriever=retriever, reranker=weak).answer(QUESTION)
    assert resp.refused and resp.mode == "refused" and resp.answer == REFUSAL
    assert resp.citations == [] and resp.contexts == []
    assert resp.sources, "the closest passages are still shown to explain the refusal"
    assert llm.calls == []


def test_llm_rewrite_rescues_a_weak_first_retrieval(retriever, llm):
    better = "periodic KYC updation frequency low risk customers"
    llm.rewritten = better
    rr = ScriptedReranker({better: 0.95}, default=0.05)
    resp = AgenticRAG(retriever=retriever, reranker=rr).answer("how often KYC redo?")
    assert rr.queries == ["how often KYC redo?", better]
    assert resp.rewritten_query == better
    assert resp.mode == "generated" and not resp.refused


def test_acronym_expansion_is_the_fallback_rewrite_without_an_llm(retriever):
    q = "KYC redo frequency?"
    expanded = rewrite_query(q)
    rr = ScriptedReranker({expanded: 0.9}, default=0.05)
    resp = AgenticRAG(retriever=retriever, reranker=rr).answer(q)
    assert rr.queries == [q, expanded]
    assert resp.rewritten_query == expanded and resp.mode == "extractive"


def test_model_refusal_text_is_reported_as_refused(retriever, llm):
    llm.answer = REFUSAL
    resp = AgenticRAG(retriever=retriever, reranker=ScriptedReranker({})).answer(QUESTION)
    assert resp.refused and resp.mode == "refused" and resp.citations == []


def test_provider_safety_refusal_is_a_refusal_not_an_extract(retriever, llm):
    llm.error = LLMError("declined", kind="refused")
    resp = AgenticRAG(retriever=retriever, reranker=ScriptedReranker({})).answer(QUESTION)
    assert resp.refused and resp.mode == "refused" and "declined" in resp.notice


def test_pii_in_model_output_is_redacted(retriever, llm):
    llm.answer = "The customer's Aadhaar 1234 5678 9012 was verified [1]."
    resp = AgenticRAG(retriever=retriever, reranker=ScriptedReranker({})).answer(QUESTION)
    assert "1234 5678 9012" not in resp.answer and "[REDACTED]" in resp.answer


def test_extractive_answer_quotes_the_sentence_matching_the_question():
    from ragsys.generation import extractive_answer

    chunk = Chunk(chunk_id="c::0", doc_id="c", title="Cards", source="RBI",
                  text="The issuer shall publish its fee schedule on the website. "
                       "Any request for closure of a credit card shall be honoured within "
                       "seven working days by the issuer.")
    out = extractive_answer("How fast must a credit card closure request be honoured?",
                            [RetrievedChunk(chunk=chunk, rerank_score=0.9)])
    assert "seven working days" in out.answer and "fee schedule" not in out.answer
    assert out.citations[0].marker == "[1]"
