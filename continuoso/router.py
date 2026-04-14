"""Model router.

Two APIs:

- `iter_selections(task_class, skip_tiers)` — yields Selections in preference
  order: best model in cheapest tier first, then remaining models in that tier,
  then next tier, etc. Callers iterate until one succeeds.

- `select(task_class)` — convenience wrapper that returns the first yield.

Selection scoring within a tier is a Laplace-smoothed success rate minus a
small cost penalty, so models with no history start at ~0.5 and successful
cheap models rise. Models that repeatedly 4xx/5xx will sink on success_rate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

from .config import RoutingConfig, BudgetsConfig, Tier
from .llm import ClaudeCodeClient, LLMClient, OllamaClient, OpenRouterClient
from .memory import Memory

log = logging.getLogger(__name__)


@dataclass
class Selection:
    tier: str
    provider: str
    model: str
    client: LLMClient
    is_fallback: bool = False


class Router:
    def __init__(
        self,
        routing: RoutingConfig,
        budgets: BudgetsConfig,
        memory: Memory,
        openrouter: OpenRouterClient | None,
        claude_code: ClaudeCodeClient | None,
        ollama: OllamaClient | None = None,
    ) -> None:
        self.routing = routing
        self.budgets = budgets
        self.memory = memory
        self.openrouter = openrouter
        self.claude_code = claude_code
        self.ollama = ollama

    # ---------- Public API ----------
    def iter_selections(
        self,
        task_class: str,
        skip_tiers: set[str] | None = None,
    ) -> Iterator[Selection]:
        """Yield every usable (tier, provider, model, client) for this task
        class in preference order — cheapest tier and best model first."""
        skip = skip_tiers or set()
        default_tier = self.routing.task_class_defaults.get(task_class, "cheap")
        order = self._ladder_from(default_tier)

        for tier_name in order:
            if tier_name in skip:
                continue
            if self._budget_exhausted(tier_name):
                log.info("tier %s budget exhausted; skipping", tier_name)
                continue
            tier = self.routing.tiers[tier_name]

            # Primary provider for this tier.
            primary_client = self._client_for(tier.provider)
            if primary_client and tier.models:
                for model_id in self._ranked_models(task_class, tier.provider, tier.models):
                    yield Selection(tier.name, tier.provider, model_id, primary_client)

            # Optional per-tier fallback (e.g. heavy -> openrouter).
            if tier.fallback_provider and tier.fallback_models:
                fb_client = self._client_for(tier.fallback_provider)
                if fb_client:
                    for model_id in self._ranked_models(
                        task_class, tier.fallback_provider, tier.fallback_models
                    ):
                        yield Selection(
                            tier.name, tier.fallback_provider, model_id,
                            fb_client, is_fallback=True,
                        )

    def select(self, task_class: str, skip_tiers: set[str] | None = None) -> Selection | None:
        """Return the top-preference Selection, or None if nothing is usable."""
        return next(self.iter_selections(task_class, skip_tiers), None)

    # ---------- Internals ----------
    def _ladder_from(self, default_tier: str) -> list[str]:
        order = self.routing.tier_order
        if default_tier not in order:
            return order
        idx = order.index(default_tier)
        return order[idx:]

    def _budget_exhausted(self, tier_name: str) -> bool:
        caps = self.budgets.tiers.get(tier_name)
        if not caps:
            return False
        tokens, cost = self.memory.get_usage_today(tier_name)
        if caps.max_tokens_per_day is not None and tokens >= caps.max_tokens_per_day:
            return True
        if caps.max_usd_per_day is not None and cost >= caps.max_usd_per_day:
            return True
        return False

    def _client_for(self, provider: str) -> LLMClient | None:
        if provider == "ollama":
            return self.ollama
        if provider == "openrouter":
            return self.openrouter
        if provider == "claude_code":
            return self.claude_code
        return None

    def _ranked_models(
        self,
        task_class: str,
        provider: str,
        models: list,
    ) -> list[str]:
        """Return all model ids, best-first by smoothed success-rate minus cost."""
        if not models:
            return []
        stats = {
            (s.provider, s.model): s
            for s in self.memory.router_stats(task_class)
        }
        scored: list[tuple[float, str]] = []
        for m in models:
            s = stats.get((provider, m.id))
            if s and s.attempts > 0:
                sr = (s.successes + 1) / (s.attempts + 2)
                cost_pen = min(s.avg_cost, 0.50)
            else:
                sr = 0.5
                cost_pen = 0.0
            score = sr - cost_pen * 0.2
            scored.append((score, m.id))
        # Stable sort so order in routing.yaml is a tiebreaker for untried models.
        scored.sort(key=lambda x: -x[0])
        return [mid for _, mid in scored]

    # ---------- Accounting hook ----------
    def record_usage(
        self,
        tier: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        self.memory.add_usage(
            tier=tier,
            tokens=input_tokens + output_tokens,
            cost_usd=cost_usd,
        )
