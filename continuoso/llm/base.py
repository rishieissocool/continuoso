"""LLM client interface. Any provider implements this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class LLMError(Exception):
    """Raised on provider/network failures we can recover from by escalating."""


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0


class LLMClient(ABC):
    provider: str = "base"

    @abstractmethod
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
        ...
