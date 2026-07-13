"""LLM provider abstraction: one interface, Anthropic or OpenAI behind it.

`get_llm()` returns the configured default; `get_llm("openai")` / `get_llm("anthropic")`
force a specific backend (used for per-request overrides and eval A/B runs).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ragsys.config import settings
from ragsys.telemetry import SpanKind, span


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMProvider:
    """Wraps a single vendor SDK behind a `.complete()` call."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.model = (
            settings.anthropic_model if provider == "anthropic" else settings.openai_model
        )
        self._client = None

    def _client_lazy(self):
        if self._client is not None:
            return self._client
        if self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        else:
            from openai import OpenAI

            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def complete(self, system: str, user: str) -> LLMResult:
        with span(
            "llm.complete",
            kind=SpanKind.LLM,
            **{"llm.provider": self.provider, "llm.model": self.model,
               "llm.temperature": settings.llm_temperature},
        ) as sp:
            if self.provider == "anthropic":
                result = self._complete_anthropic(system, user)
            else:
                result = self._complete_openai(system, user)
            sp.set("llm.output_text", result.text[:2000])
            if result.output_tokens is not None:
                sp.set("llm.token_count.completion", result.output_tokens)
            if result.input_tokens is not None:
                sp.set("llm.token_count.prompt", result.input_tokens)
            return result

    def _complete_anthropic(self, system: str, user: str) -> LLMResult:
        client = self._client_lazy()
        msg = client.messages.create(
            model=self.model,
            system=system,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return LLMResult(
            text=text.strip(),
            provider=self.provider,
            model=self.model,
            input_tokens=getattr(msg.usage, "input_tokens", None),
            output_tokens=getattr(msg.usage, "output_tokens", None),
        )

    def _complete_openai(self, system: str, user: str) -> LLMResult:
        client = self._client_lazy()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = resp.usage
        return LLMResult(
            text=(resp.choices[0].message.content or "").strip(),
            provider=self.provider,
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )


@lru_cache
def get_llm(provider: str | None = None) -> LLMProvider:
    return LLMProvider(provider or settings.llm_provider)
