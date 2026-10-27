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


class ServiceUnavailable(Exception):
    """Provider overloaded (HTTP 5xx) or timed out. Nothing is billed; safe to retry later."""


class RequestRejected(Exception):
    """Google refused the request (HTTP 4xx other than 429). Never billed; retrying won't help.

    `message` is already plain English and says what to do.
    """

    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


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


KEY_HELP = "Gemini rejected the API key. See README 'Key problems'."


def explain_rejection(e, model: str | None = None, thinking: str | None = None) -> str:
    """Plain-English reason (and what to do) for a google-genai ClientError other than 429."""
    code = getattr(e, "code", None)
    message = str(getattr(e, "message", None) or e)
    text = f"{message} {getattr(e, 'status', '')} {getattr(e, 'details', '')}".lower()
    if code == 404 or "no longer available" in text:
        return (
            f"Gemini model '{model}' is retired or unknown (Google says: {message[:160]}): model "
            "retired, remove it from the list."
        )
    if "api key" in text or "api_key" in text or code in (401, 403):
        return KEY_HELP
    if code == 400 and "thinking" in text:
        return (
            f"Model '{model}' does not accept thinking level '{thinking}': this model does not "
            "support that thinking level. Remove that level from the list (or pick another)."
        )
    return f"Google refused the request (HTTP {code}): {message[:300]}"


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
        import httpx

        try:
            resp = self._client.models.generate_content(model=model, contents=user, config=cfg)
        except errors.ClientError as e:
            if e.code == 429:
                raise self._rate_limited(e) from e
            raise RequestRejected(e.code, explain_rejection(e, model, thinking)) from e
        except errors.ServerError as e:  # 5xx, e.g. 503 "model is overloaded": not billed
            raise ServiceUnavailable(f"Gemini HTTP {e.code}: {e.message}") from e
        except httpx.TimeoutException as e:
            raise ServiceUnavailable(f"Gemini request timed out: {e}") from e
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
