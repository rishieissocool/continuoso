"""Planner — turns the current Snapshot + goals into a task DAG.

Two LLM passes:
  1) Reflect: identify gaps (cheap tier — run on free/cheap model).
  2) Plan: pick the best gap and decompose into subtasks (heavy tier).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .config import AppConfig
from .llm.base import LLMClient, LLMError
from .memory import Memory
from .observer import Snapshot
from .prompts import PLAN_PROMPT, REFLECT_PROMPT, SYSTEM_BASE
from .router import Router, Selection

log = logging.getLogger(__name__)


@dataclass
class Subtask:
    id: str
    task_class: str
    instruction: str
    files: list[str]
    acceptance_criteria: list[str]


@dataclass
class Plan:
    iteration_goal: str
    chosen_gap_id: str
    subtasks: list[Subtask]
    gap_priority_weight: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class Planner:
    def __init__(
        self,
        cfg: AppConfig,
        router: Router,
        memory: Memory,
        iteration_id: int,
    ) -> None:
        self.cfg = cfg
        self.router = router
        self.memory = memory
        self.iteration_id = iteration_id

    def reflect(self, snapshot: Snapshot) -> list[dict]:
        prompt = REFLECT_PROMPT.format(
            goals=json.dumps(self.cfg.goals.raw, indent=2),
            state=snapshot.to_json(),
        )
        resp, sel = self._call_with_fallback("reflect_gaps", prompt)
        self._account(sel, resp, "reflect_gaps")
        data = _loads_robust(resp.text)
        gaps = data.get("gaps", []) if isinstance(data, dict) else []
        return self._rank_gaps(gaps)

    def plan(self, snapshot: Snapshot, gaps: list[dict]) -> Plan:
        prompt = PLAN_PROMPT.format(
            gaps=json.dumps(gaps[:10], indent=2),
            state=snapshot.to_json(),
            max_loc=self.cfg.budgets.max_loc_changed,
            max_files=self.cfg.budgets.max_files_changed,
        )
        resp, sel = self._call_with_fallback("plan_iteration", prompt)
        self._account(sel, resp, "plan_iteration")
        data = _loads_robust(resp.text)
        if not isinstance(data, dict) or "subtasks" not in data:
            raise LLMError(f"planner returned invalid JSON: {resp.text[:300]}")

        subtasks = [
            Subtask(
                id=str(s.get("id") or f"s{i+1}"),
                task_class=s.get("task_class", "edit_single_file"),
                instruction=s.get("instruction", ""),
                files=list(s.get("files", []) or []),
                acceptance_criteria=list(s.get("acceptance_criteria", []) or []),
            )
            for i, s in enumerate(data.get("subtasks", []))
        ]
        # Match gap weight for reporting.
        chosen = data.get("chosen_gap_id", "")
        weight = 0.0
        for g in gaps:
            if g.get("id") == chosen:
                weight = float(g.get("_weight", 0.0))
                break
        return Plan(
            iteration_goal=data.get("iteration_goal", ""),
            chosen_gap_id=chosen,
            subtasks=subtasks,
            gap_priority_weight=weight,
            raw=data,
        )

    def _call_with_fallback(self, task_class: str, prompt: str):
        """Try each model the router yields until one succeeds."""
        last_err: Exception | None = None
        for sel in self.router.iter_selections(task_class):
            try:
                resp = _call(sel, prompt, json_mode=True)
                return resp, sel
            except (LLMError, KeyError, IndexError, TypeError) as e:
                log.warning(
                    "%s: %s/%s failed (%s: %s), trying next model",
                    task_class, sel.provider, sel.model, type(e).__name__, e,
                )
                last_err = e
        raise LLMError(f"all models exhausted for {task_class}: {last_err}")

    def _rank_gaps(self, gaps: list[dict]) -> list[dict]:
        prio_weight = {
            p["id"]: float(p.get("weight", 0.5))
            for p in self.cfg.goals.priorities
        }
        scored: list[tuple[float, dict]] = []
        for g in gaps:
            pid = g.get("priority_id", "")
            w = prio_weight.get(pid, 0.3)
            est_loc = max(10, int(g.get("est_loc", 50)))
            feas = 1.0 / (1 + est_loc / 100)
            score = w * feas
            g["_weight"] = w
            g["_score"] = score
            scored.append((score, g))
        scored.sort(key=lambda x: -x[0])
        return [g for _, g in scored]

    def _account(self, sel: Selection, resp, task_class: str) -> None:
        self.router.record_usage(
            sel.tier, resp.input_tokens, resp.output_tokens, resp.cost_usd
        )
        from .memory import SubtaskRecord
        self.memory.record_subtask(
            SubtaskRecord(
                iteration_id=self.iteration_id,
                subtask_slug=task_class,
                task_class=task_class,
                provider=sel.provider,
                model=sel.model,
                success=True,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cost_usd=resp.cost_usd,
                latency_ms=resp.latency_ms,
            )
        )


# ---------- Helpers ----------
def _call(sel: Selection, user: str, *, json_mode: bool):
    return sel.client.complete(
        system=SYSTEM_BASE,
        user=user,
        model=sel.model,
        max_tokens=4096,
        temperature=0.2,
        json_mode=json_mode,
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _loads_robust(text: str) -> Any:
    """Tolerate models that wrap JSON in prose or code fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip triple-backtick fences.
    fenced = re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.MULTILINE)
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise LLMError(f"could not parse JSON from model output: {text[:300]}")
