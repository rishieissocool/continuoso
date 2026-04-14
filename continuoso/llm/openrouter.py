"""OpenRouter HTTP client. Single endpoint, many models."""
from __future__ import annotations

import time
from typing import Any

import requests

from .base import LLMClient, LLMError, LLMResponse

# Approximate per-million-token prices (USD). Updated periodically.
# Router uses these to pick the cheapest model within a tier.
# Free models all map to (0, 0); missing entries default to (0, 0) too.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    # Cheap paid
    "google/gemini-2.5-flash": (0.30, 2.50),
    "deepseek/deepseek-r1-0528": (0.45, 2.18),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    # Heavy fallback (paid)
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "openai/gpt-4o": (2.50, 10.00),
}


def estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    inp, out = PRICE_TABLE.get(model, (0.0, 0.0))
    return (in_tok / 1_000_000) * inp + (out_tok / 1_000_000) * out


class OpenRouterClient(LLMClient):
    provider = "openrouter"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        referer: str = "https://github.com/continuoso",
        title: str = "continuoso",
        timeout: int = 180,
    ) -> None:
        if not api_key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.referer = referer
        self.title = title
        self.timeout = timeout
        self.session = requests.Session()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
            "Content-Type": "application/json",
        }

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
        workdir: str | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.base_url}/chat/completions"
        start = time.monotonic()
        last_exc: Exception | None = None

        # Simple retry with backoff for transient 5xx / rate limits.
        for attempt in range(3):
            try:
                r = self.session.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    time.sleep(2 ** attempt)
                    last_exc = LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                    continue
                if r.status_code >= 400:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")
                data = r.json()
                break
            except requests.RequestException as e:
                last_exc = e
                time.sleep(2 ** attempt)
        else:
            raise LLMError(f"OpenRouter failed after retries: {last_exc}")

        latency_ms = int((time.monotonic() - start) * 1000)

        # Guard against models that return 200 with a non-standard body
        # (missing "choices", empty choices list, or an error object).
        if "error" in data:
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise LLMError(f"OpenRouter error response: {msg[:300]}")

        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise LLMError(
                f"OpenRouter response missing 'choices': {str(data)[:300]}"
            )

        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        if not text:
            raise LLMError(
                f"OpenRouter returned empty content: {str(choice)[:300]}"
            )

        usage = data.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens", 0))
        out_tok = int(usage.get("completion_tokens", 0))
        cost = estimate_cost(model, in_tok, out_tok)

        return LLMResponse(
            text=text,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
