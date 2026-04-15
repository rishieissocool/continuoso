"""Planner — turns the current Snapshot + goals into a task DAG.

Two LLM passes:
  1) Reflect: identify gaps (cheap tier — run on free/cheap model).
  2) Plan: pick the best gap and decompose into subtasks (heavy tier).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .config import AppConfig
from .llm.base import LLMClient, LLMError
from .memory import Memory
from .observer import Snapshot
from .llm_trace import log_llm_trace
from .json_compact import dumps_llm
from .json_parse import parse_llm_json
from .prompts import (
    PLAN_PROMPT,
    REFLECT_PROMPT,
    SYSTEM_BASE,
    TASK_CLASSES,
    format_session_focus,
)
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
            task_classes=TASK_CLASSES,
            session_focus=format_session_focus(self.cfg.session_focus),
            goals=dumps_llm(self.cfg.goals.raw),
            state=snapshot.to_json(),
        )
        t0 = time.monotonic()
        resp, sel = self._call_with_fallback("reflect_gaps", prompt)
        wall_ms = int((time.monotonic() - t0) * 1000)
        self._account(sel, resp, "reflect_gaps")
        log.info(
            "reflect_gaps: %s/%s — %dms wall, %d in / %d out tok",
            sel.provider, sel.model, wall_ms,
            resp.input_tokens, resp.output_tokens,
        )
        log_llm_trace(log, self.cfg, "reflect_gaps", resp.text)
        log.debug("reflect_gaps: raw model text (first 2k):\n%s", resp.text[:2000])
        data = _loads_robust(resp.text)
        gaps = data.get("gaps", []) if isinstance(data, dict) else []
        ranked = self._rank_gaps(gaps)
        if ranked:
            ids = ", ".join(str(g.get("id", "?")) for g in ranked[:8])
            log.info("reflect_gaps: %d gaps (ids: %s%s)", len(ranked), ids, " …" if len(ranked) > 8 else "")
        else:
            log.info("reflect_gaps: %d gaps after ranking", len(ranked))
        return ranked

    def plan(self, snapshot: Snapshot, gaps: list[dict]) -> Plan:
        prompt = PLAN_PROMPT.format(
            session_focus=format_session_focus(self.cfg.session_focus),
            gaps=dumps_llm(gaps[:10]),
            state=snapshot.to_json(),
            max_loc=self.cfg.budgets.max_loc_changed,
            max_files=self.cfg.budgets.max_files_changed,
        )
        t0 = time.monotonic()
        resp, sel = self._call_with_fallback("plan_iteration", prompt)
        wall_ms = int((time.monotonic() - t0) * 1000)
        self._account(sel, resp, "plan_iteration")
        log.info(
            "plan_iteration: %s/%s — %dms wall, %d in / %d out tok",
            sel.provider, sel.model, wall_ms,
            resp.input_tokens, resp.output_tokens,
        )
        log_llm_trace(log, self.cfg, "plan_iteration", resp.text)
        log.debug("plan_iteration: raw model text (first 2k):\n%s", resp.text[:2000])
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
        goal = data.get("iteration_goal", "")
        plan = Plan(
            iteration_goal=goal,
            chosen_gap_id=chosen,
            subtasks=subtasks,
            gap_priority_weight=weight,
            raw=data,
        )
        log.info(
            "plan_iteration: gap=%s subtasks=%d goal=%r",
            chosen, len(subtasks), (goal[:120] + "…") if len(goal) > 120 else goal,
        )
        return plan

    def _call_with_fallback(self, task_class: str, prompt: str):
        """Try each model the router yields until one succeeds."""
        last_err: Exception | None = None
        for sel in self.router.iter_selections(task_class):
            log.info(
                "%s: trying %s/%s (tier=%s)",
                task_class, sel.provider, sel.model, sel.tier,
            )
            try:
                resp = _call(
                    sel,
                    prompt,
                    json_mode=True,
                    workdir=str(self.cfg.project_dir),
                )
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
        focus = (self.cfg.session_focus or "").strip().lower()
        focus_tokens = {t for t in focus.replace(",", " ").split() if len(t) > 2} if focus else set()

        scored: list[tuple[float, dict]] = []
        for g in gaps:
            pid = g.get("priority_id", "")
            w = prio_weight.get(pid, 0.3)
            est_loc = max(10, int(g.get("est_loc", 50)))
            feas = 1.0 / (1 + est_loc / 100)
            score = w * feas
            if focus_tokens:
                blob = f"{g.get('title', '')} {g.get('rationale', '')}".lower()
                hits = sum(1 for t in focus_tokens if t in blob)
                score *= 1.0 + min(0.12, hits * 0.04)
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
def _call(
    sel: Selection,
    user: str,
    *,
    json_mode: bool,
    workdir: str | None = None,
):
    return sel.client.complete(
        system=SYSTEM_BASE,
        user=user,
        model=sel.model,
        max_tokens=4096,
        temperature=0.2,
        json_mode=json_mode,
        workdir=workdir,
    )


def _loads_robust(text: str) -> Any:
    """Tolerate models that wrap JSON in prose or code fences."""
    return parse_llm_json(text, context="model")
