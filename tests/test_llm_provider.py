"""Provider tests against an in-process fake HTTP transport: no network, no keys.

They pin down the two things that matter for a public demo: the request each
SDK actually sends, and that every failure surfaces as an `LLMError` with the
right `kind` (which is what lets the agent degrade instead of returning a 500).
"""
import json

import httpx
import pytest

from ragsys.config import OPENAI_COMPATIBLE_BASE_URLS, settings
from ragsys.llm.provider import LLMError, LLMProvider


def _openai_provider(monkeypatch, handler, name="groq"):
    from openai import OpenAI

    monkeypatch.setattr(settings, f"{name}_api_key", "test-key")
    provider = LLMProvider(name)
    provider._client = OpenAI(
        api_key="test-key", base_url=OPENAI_COMPATIBLE_BASE_URLS[name] or "https://api.openai.com/v1",
        max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return provider


def _chat_completion(text="Seven working days [1].", model="llama-3.3-70b-versatile"):
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 9, "total_tokens": 129},
    }


def test_groq_request_goes_to_groq_with_system_and_user_messages(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_chat_completion())

    result = _openai_provider(monkeypatch, handler).complete("SYSTEM RULES", "the question")
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == settings.groq_model
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "SYSTEM RULES"},
        {"role": "user", "content": "the question"},
    ]
    assert seen["body"]["temperature"] == 0.0
    assert (result.text, result.provider) == ("Seven working days [1].", "groq")
    assert (result.input_tokens, result.output_tokens) == (120, 9)


def test_gemini_uses_its_openai_compatible_endpoint(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json=_chat_completion(model="gemini"))

    _openai_provider(monkeypatch, handler, name="gemini").complete("s", "u")
    assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def test_max_tokens_override_is_sent(monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_chat_completion())

    _openai_provider(monkeypatch, handler).complete("s", "u", max_tokens=120)
    assert seen["body"]["max_tokens"] == 120


@pytest.mark.parametrize("status, kind", [(429, "rate_limited"), (500, "error"), (401, "error")])
def test_http_failures_become_llm_errors(monkeypatch, status, kind):
    def handler(request):
        return httpx.Response(status, json={"error": {"message": "nope", "type": "x"}})

    with pytest.raises(LLMError) as exc:
        _openai_provider(monkeypatch, handler).complete("s", "u")
    assert exc.value.kind == kind


def test_timeout_becomes_llm_error(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMError) as exc:
        _openai_provider(monkeypatch, handler).complete("s", "u")
    assert exc.value.kind == "timeout"


def test_connection_failure_becomes_llm_error(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(LLMError) as exc:
        _openai_provider(monkeypatch, handler).complete("s", "u")
    assert exc.value.kind == "error"


def test_missing_key_is_reported_as_not_configured():
    with pytest.raises(LLMError) as exc:
        LLMProvider("groq").complete("s", "u")  # conftest sets no keys
    assert exc.value.kind == "not_configured"


def test_unknown_provider_is_rejected():
    with pytest.raises(LLMError):
        LLMProvider("made-up")


# ---- Anthropic ------------------------------------------------------------
def _anthropic_provider(monkeypatch, handler):
    import anthropic
    import httpx2

    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    provider = LLMProvider("anthropic")
    provider._client = anthropic.Anthropic(
        api_key="test-key", max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return provider


def _message(stop_reason="end_turn", content=None):
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
        "content": content if content is not None else [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "Seven working days [1]."},
        ],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 200, "output_tokens": 12},
    }


def test_anthropic_request_has_no_sampling_params_and_skips_thinking_blocks(monkeypatch):
    import httpx2

    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=_message())

    result = _anthropic_provider(monkeypatch, handler).complete("SYSTEM RULES", "the question")
    assert seen["url"].endswith("/v1/messages")
    body = seen["body"]
    assert body["model"] == settings.anthropic_model
    assert body["system"] == "SYSTEM RULES"
    assert body["messages"] == [{"role": "user", "content": "the question"}]
    # Current Claude models reject sampling parameters with a 400.
    assert not {"temperature", "top_p", "top_k"} & body.keys()
    assert body["max_tokens"] >= 16000  # room for adaptive thinking
    assert result.text == "Seven working days [1]."
    assert (result.input_tokens, result.output_tokens) == (200, 12)


def test_anthropic_refusal_stop_reason_is_surfaced(monkeypatch):
    import httpx2

    def handler(request):
        return httpx2.Response(200, json=_message(stop_reason="refusal", content=[]))

    with pytest.raises(LLMError) as exc:
        _anthropic_provider(monkeypatch, handler).complete("s", "u")
    assert exc.value.kind == "refused"


def test_anthropic_rate_limit_becomes_llm_error(monkeypatch):
    import httpx2

    def handler(request):
        return httpx2.Response(429, json={"type": "error", "error": {
            "type": "rate_limit_error", "message": "slow down"}})

    with pytest.raises(LLMError) as exc:
        _anthropic_provider(monkeypatch, handler).complete("s", "u")
    assert exc.value.kind == "rate_limited"
