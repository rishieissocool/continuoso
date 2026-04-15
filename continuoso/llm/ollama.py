"""Local Ollama — OpenAI-compatible `/v1/chat/completions` (default :11434).

No API key; cost is tracked as $0. Pull the model first: `ollama pull qwen2.5:7b`
"""
from __future__ import annotations

import logging
import time
from typing import Any

import requests

from .base import LLMClient, LLMError, LLMResponse
from .wait_heartbeat import wait_heartbeat

log = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/v1",
        timeout: int = 180,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

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
        data: dict[str, Any] | None = None

        log.info(
            "ollama: POST chat/completions model=%s (first reply can take minutes on CPU)…",
            model,
        )

        for attempt in range(4):
            try:
                with wait_heartbeat(f"ollama/{model}"):
                    r = self.session.post(
                        url,
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=self.timeout,
                    )
            except requests.RequestException as e:
                last_exc = e
                time.sleep(min(2 ** attempt, 8))
                continue

            if r.status_code == 429 or 500 <= r.status_code < 600:
                last_exc = LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(min(2 ** attempt, 8))
                continue

            if r.status_code == 400 and json_mode and "response_format" in payload:
                # Older Ollama builds may not support JSON mode via OpenAI compat.
                del payload["response_format"]
                continue

            if r.status_code >= 400:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")

            data = r.json()
            break
        else:
            raise LLMError(
                f"Ollama request failed: {last_exc or 'unknown'}"
            )

        assert data is not None
        latency_ms = int((time.monotonic() - start) * 1000)

        if "error" in data:
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise LLMError(f"Ollama error: {msg[:300]}")

        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise LLMError(
                f"Ollama response missing 'choices': {str(data)[:300]}"
            )

        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        if not text:
            raise LLMError(
                f"Ollama returned empty content: {str(choice)[:300]}"
            )

        usage = data.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens", 0))
        out_tok = int(usage.get("completion_tokens", 0))

        return LLMResponse(
            text=text,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=0.0,
            latency_ms=latency_ms,
        )
