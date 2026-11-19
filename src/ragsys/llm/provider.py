"""LLM provider abstraction: one `.complete()` interface, several backends.

  * groq / gemini / openai -> the OpenAI SDK pointed at each provider's
    OpenAI-compatible endpoint (Groq and Gemini both have free tiers).
  * anthropic              -> the Anthropic SDK.

`get_llm()` returns the configured default; `get_llm("gemini")` forces a
backend (used for eval A/B runs and, if enabled, per-request overrides).
Every failure is raised as `LLMError` so callers can degrade gracefully.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ragsys.config import OPENAI_COMPATIBLE_BASE_URLS, settings
from ragsys.telemetry import SpanKind, span

PROVIDERS = ("groq", "gemini", "openai", "anthropic")


class LLMError(RuntimeError):
    """The LLM call could not be completed (no key, rate limit, timeout, ...)."""

    def __init__(self, message: str, kind: str = "error") -> None:
        super().__init__(message)
        self.kind = kind  # "not_configured" | "rate_limited" | "timeout" | "refused" | "error"


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
        if provider not in PROVIDERS:
            raise LLMError(f"unknown LLM provider: {provider!r}", kind="not_configured")
        self.provider = provider
        self.model = settings.model_for(provider)
        self._client = None

    def _client_lazy(self):
        if self._client is not None:
            return self._client
        key = settings.api_key_for(self.provider)
        if not key:
            raise LLMError(
                f"no API key configured for provider {self.provider!r}", kind="not_configured"
            )
        if self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=key,
                timeout=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,
            )
        else:
            from openai import OpenAI

            base_url = OPENAI_COMPATIBLE_BASE_URLS.get(self.provider)
            if self.provider == "openai" and settings.openai_base_url:
                base_url = settings.openai_base_url
            self._client = OpenAI(
                api_key=key,
                base_url=base_url,
                timeout=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,
            )
        return self._client

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> LLMResult:
        with span(
            "llm.complete",
            kind=SpanKind.LLM,
            **{"llm.provider": self.provider, "llm.model_name": self.model},
        ) as sp:
            if self.provider == "anthropic":
                result = self._complete_anthropic(system, user, max_tokens)
            else:
                result = self._complete_openai_compatible(system, user, max_tokens)
            sp.set("output.value", result.text[:2000])
            if result.output_tokens is not None:
                sp.set("llm.token_count.completion", result.output_tokens)
            if result.input_tokens is not None:
                sp.set("llm.token_count.prompt", result.input_tokens)
            return result

    # -- Anthropic ---------------------------------------------------------
    def _complete_anthropic(self, system: str, user: str, max_tokens: int | None) -> LLMResult:
        import anthropic

        client = self._client_lazy()
        try:
            # No sampling parameters: current Claude models reject `temperature`.
            # The budget is generous because adaptive thinking shares max_tokens.
            msg = client.messages.create(
                model=self.model,
                system=system,
                max_tokens=max(max_tokens or 0, settings.anthropic_max_tokens),
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.RateLimitError as exc:
            raise LLMError(f"anthropic rate limit: {exc}", kind="rate_limited") from exc
        except anthropic.APITimeoutError as exc:
            raise LLMError("anthropic request timed out", kind="timeout") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"anthropic connection error: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"anthropic API error {exc.status_code}: {exc.message}") from exc

        if msg.stop_reason == "refusal":
            raise LLMError("the model declined to answer this request", kind="refused")
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResult(
            text=text.strip(),
            provider=self.provider,
            model=self.model,
            input_tokens=getattr(msg.usage, "input_tokens", None),
            output_tokens=getattr(msg.usage, "output_tokens", None),
        )

    # -- OpenAI-compatible (Groq, Gemini, OpenAI) ---------------------------
    def _complete_openai_compatible(
        self, system: str, user: str, max_tokens: int | None
    ) -> LLMResult:
        import openai

        client = self._client_lazy()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                temperature=settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except openai.RateLimitError as exc:
            raise LLMError(f"{self.provider} rate limit: {exc}", kind="rate_limited") from exc
        except openai.APITimeoutError as exc:
            raise LLMError(f"{self.provider} request timed out", kind="timeout") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"{self.provider} connection error: {exc}") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"{self.provider} API error {exc.status_code}: {exc.message}") from exc

        if not resp.choices:
            raise LLMError(f"{self.provider} returned no choices")
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
