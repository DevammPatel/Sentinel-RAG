"""LLM provider abstraction: one `.complete()` interface, several backends.

  * groq / gemini / openai -> the OpenAI SDK pointed at each provider's
    OpenAI-compatible endpoint (Groq and Gemini both have free tiers).
  * anthropic              -> the Anthropic SDK.

`get_llm()` returns the configured default; `get_llm("gemini")` forces a
backend (used for eval A/B runs and, if enabled, per-request overrides).
Every failure is raised as `LLMError` so callers can degrade gracefully.

Free tiers fail in ordinary ways: rate limits, 503 "overloaded", slow replies.
So a provider can have fallback models, tried in order, and a model that just
failed is skipped for LLM_COOLDOWN_S. Without the cool-down every visitor would
wait out the same timeout before the app gave up on a struggling model.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from functools import lru_cache

from ragsys.config import OPENAI_COMPATIBLE_BASE_URLS, settings
from ragsys.telemetry import SpanKind, span

PROVIDERS = ("groq", "gemini", "openai", "anthropic")


class LLMError(RuntimeError):
    """The LLM call could not be completed (no key, rate limit, timeout, ...)."""

    def __init__(self, message: str, kind: str = "error", status: int | None = None) -> None:
        super().__init__(message)
        # "not_configured" | "rate_limited" | "timeout" | "unavailable" | "refused" | "error"
        self.kind = kind
        self.status = status  # HTTP status, when there was one

    @property
    def transient(self) -> bool:
        """Worth trying another model, and worth cooling this one down."""
        if self.kind in ("rate_limited", "timeout", "unavailable"):
            return True
        if self.kind == "error":
            return self.status is None or self.status >= 500 or self.status == 404
        return False


# (provider, model) -> time.monotonic() until which the model is skipped.
_cooldown: dict[tuple[str, str], float] = {}
_cooldown_lock = threading.Lock()


def _cooling(provider: str, model: str) -> bool:
    with _cooldown_lock:
        return _cooldown.get((provider, model), 0.0) > time.monotonic()


def _cool_down(provider: str, model: str) -> None:
    with _cooldown_lock:
        _cooldown[(provider, model)] = time.monotonic() + settings.llm_cooldown_s


def reset_cooldowns() -> None:
    with _cooldown_lock:
        _cooldown.clear()


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
        self.models = [self.model] + [
            m for m in settings.fallback_models_for(provider) if m != self.model
        ]
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
        """Try each model in order, skipping ones that are cooling down."""
        if not settings.api_key_for(self.provider):
            raise LLMError(
                f"no API key configured for provider {self.provider!r}", kind="not_configured"
            )
        candidates = [m for m in self.models if not _cooling(self.provider, m)]
        if not candidates:
            raise LLMError(
                f"all {self.provider} models failed recently; retrying after the cool-down",
                kind="unavailable",
            )
        last: LLMError | None = None
        for model in candidates:
            try:
                return self._complete_one(model, system, user, max_tokens)
            except LLMError as exc:
                if not exc.transient:
                    raise
                _cool_down(self.provider, model)
                last = exc
        assert last is not None
        raise last

    def _complete_one(
        self, model: str, system: str, user: str, max_tokens: int | None
    ) -> LLMResult:
        with span(
            "llm.complete",
            kind=SpanKind.LLM,
            **{"llm.provider": self.provider, "llm.model_name": model},
        ) as sp:
            if self.provider == "anthropic":
                result = self._complete_anthropic(model, system, user, max_tokens)
            else:
                result = self._complete_openai_compatible(model, system, user, max_tokens)
            sp.set("output.value", result.text[:2000])
            if result.output_tokens is not None:
                sp.set("llm.token_count.completion", result.output_tokens)
            if result.input_tokens is not None:
                sp.set("llm.token_count.prompt", result.input_tokens)
            return result

    # -- Anthropic ---------------------------------------------------------
    def _complete_anthropic(
        self, model: str, system: str, user: str, max_tokens: int | None
    ) -> LLMResult:
        import anthropic

        client = self._client_lazy()
        try:
            # No sampling parameters: current Claude models reject `temperature`.
            # The budget is generous because adaptive thinking shares max_tokens.
            msg = client.messages.create(
                model=model,
                system=system,
                max_tokens=max(max_tokens or 0, settings.anthropic_max_tokens),
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.RateLimitError as exc:
            raise LLMError(f"anthropic rate limit: {exc}", kind="rate_limited", status=429) from exc
        except anthropic.APITimeoutError as exc:
            raise LLMError("anthropic request timed out", kind="timeout") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"anthropic connection error: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"anthropic API error {exc.status_code}: {exc.message}",
                           status=exc.status_code) from exc

        if msg.stop_reason == "refusal":
            raise LLMError("the model declined to answer this request", kind="refused")
        text = "".join(b.text for b in msg.content if b.type == "text")
        if not text.strip():
            raise LLMError(f"anthropic returned an empty answer (stop_reason={msg.stop_reason})")
        return LLMResult(
            text=text.strip(),
            provider=self.provider,
            model=model,
            input_tokens=getattr(msg.usage, "input_tokens", None),
            output_tokens=getattr(msg.usage, "output_tokens", None),
        )

    # -- OpenAI-compatible (Groq, Gemini, OpenAI) ---------------------------
    def _complete_openai_compatible(
        self, model: str, system: str, user: str, max_tokens: int | None
    ) -> LLMResult:
        import openai

        client = self._client_lazy()
        kwargs: dict = {}
        limit = max_tokens or settings.llm_max_tokens
        if self.provider == "gemini":
            limit = max(limit, settings.gemini_min_output_tokens)
            if settings.gemini_reasoning_effort:
                kwargs["reasoning_effort"] = settings.gemini_reasoning_effort
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=settings.llm_temperature,
                max_tokens=limit,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **kwargs,
            )
        except openai.RateLimitError as exc:
            raise LLMError(f"{self.provider} {model} rate limit: {exc}", kind="rate_limited",
                           status=429) from exc
        except openai.APITimeoutError as exc:
            raise LLMError(f"{self.provider} {model} timed out", kind="timeout") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"{self.provider} connection error: {exc}") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"{self.provider} {model} API error {exc.status_code}: {exc.message}",
                           status=exc.status_code) from exc

        if not resp.choices:
            raise LLMError(f"{self.provider} returned no choices")
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            # e.g. the whole output budget went on thinking. Treat it as a
            # failure so the agent falls back instead of showing a blank answer.
            reason = resp.choices[0].finish_reason
            raise LLMError(f"{self.provider} returned an empty answer (finish_reason={reason})")
        usage = resp.usage
        return LLMResult(
            text=text,
            provider=self.provider,
            model=model,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )


@lru_cache
def get_llm(provider: str | None = None) -> LLMProvider:
    return LLMProvider(provider or settings.llm_provider)
