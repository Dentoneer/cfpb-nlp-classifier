"""Gemini client abstraction: protocol, response type, fake for tests, google-genai impl."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    text: str
    input_tokens: int | None  # None -> provider gave no usage; runner records the estimate
    output_tokens: int | None  # candidates + thinking tokens
    logprobs: list | None = None  # not requested; always None


class RateLimited(Exception):
    """HTTP 429 from the provider. `daily=True` means a per-day quota, not a per-minute one."""

    def __init__(self, retry_after: float | None = None, daily: bool = False):
        super().__init__(f"rate limited (retry_after={retry_after}, daily={daily})")
        self.retry_after = retry_after
        self.daily = daily


class GeminiClient(Protocol):
    def generate(
        self,
        model: str,
        system: str,
        user: str,
        response_schema: dict,
        thinking: str,
        max_output_tokens: int,
    ) -> LLMResponse: ...


class FakeClient:
    """Test double: `responder(model, system, user, schema, thinking) -> LLMResponse`."""

    def __init__(self, responder: Callable[..., LLMResponse]):
        self.responder = responder
        self.calls: list[tuple] = []

    def generate(
        self,
        model: str,
        system: str,
        user: str,
        response_schema: dict,
        thinking: str,
        max_output_tokens: int,
    ) -> LLMResponse:
        self.calls.append((model, system, user, response_schema, thinking, max_output_tokens))
        return self.responder(model, system, user, response_schema, thinking)


_BUDGETS = {"none": 0, "minimal": 0, "low": 512, "medium": 2048, "high": 8192}
_LEVELS = {
    "none": "MINIMAL",
    "minimal": "MINIMAL",
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
}


class GenAIClient:
    """Thin wrapper over the google-genai SDK (exercised only by the slow test)."""

    def __init__(self, api_key: str):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._quota_logged = False

    def _thinking_config(self, model: str, thinking: str):
        from google.genai import types

        if model.startswith("gemini-2.5"):
            return types.ThinkingConfig(thinking_budget=_BUDGETS[thinking])
        return types.ThinkingConfig(thinking_level=_LEVELS[thinking])

    def generate(
        self,
        model: str,
        system: str,
        user: str,
        response_schema: dict,
        thinking: str,
        max_output_tokens: int,
    ) -> LLMResponse:
        from google.genai import errors, types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=response_schema,
            thinking_config=self._thinking_config(model, thinking),
            max_output_tokens=max_output_tokens,
        )
        try:
            resp = self._client.models.generate_content(model=model, contents=user, config=cfg)
        except errors.ClientError as e:
            if e.code == 429:
                raise self._rate_limited(e) from e
            raise
        u = resp.usage_metadata
        if u is None or u.prompt_token_count is None or u.candidates_token_count is None:
            return LLMResponse(resp.text or "", None, None)
        out = u.candidates_token_count + (u.thoughts_token_count or 0)
        return LLMResponse(resp.text or "", u.prompt_token_count, out)

    def _rate_limited(self, e) -> RateLimited:
        text = f"{e.message} {e.details}"
        daily = "PerDay" in text
        retry_after = None
        body = e.details.get("error", {}) if isinstance(e.details, dict) else {}
        for d in body.get("details", []):
            delay = d.get("retryDelay") if isinstance(d, dict) else None
            if isinstance(delay, str) and delay.endswith("s"):
                try:
                    retry_after = float(delay[:-1])
                except ValueError:
                    pass
        if not self._quota_logged:
            self._quota_logged = True
            print(f"[gemini] 429 quota info (daily={daily}): {text[:600]}")
        return RateLimited(retry_after, daily)
