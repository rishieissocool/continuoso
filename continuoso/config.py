"""Config loading — per-project state with bundled defaults.

Each project keeps its own state in `<project>/.continuoso/`:
  goals.yaml           user-edited; starter created by `continuoso init`
  routing.yaml         optional override; falls back to bundled default
  budgets.yaml         optional override
  dangerous_paths.yaml optional override
  memory.db            SQLite state (iterations, router stats, budgets)
  worktrees/           ephemeral git worktrees
  logs/features.md     append-only list of merged features (all sessions)
  logs/sessions/       one markdown file per `continuoso run` with that session's features
  Session focus        optional theme for a run (CLI prompt, --focus, or CONTINUOSO_SESSION_FOCUS)
  Parallel subtasks    CONTINUOSO_PARALLEL_WORKERS / --parallel — disjoint file sets run in parallel

Run `continuoso run /path/to/project` from anywhere.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

# Bundled defaults ship inside the continuoso package repo.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_CONFIG = PACKAGE_ROOT / "config"
STATE_DIRNAME = ".continuoso"


@dataclass
class ModelSpec:
    id: str
    context: int = 32000


@dataclass
class Tier:
    name: str
    provider: str
    models: list[ModelSpec]
    fallback_provider: str | None = None
    fallback_models: list[ModelSpec] = field(default_factory=list)


@dataclass
class RoutingConfig:
    success_threshold: float
    escalation_attempts: int
    tiers: dict[str, Tier]
    task_class_defaults: dict[str, str]
    # If set in routing.yaml, defines tier walk order (e.g. heavy first for Claude Code CLI).
    explicit_tier_order: list[str] | None = None

    @property
    def tier_order(self) -> list[str]:
        base = self.explicit_tier_order or ["local", "free", "cheap", "heavy"]
        return [t for t in base if t in self.tiers]


@dataclass
class BudgetCaps:
    max_tokens_per_day: int | None = None
    max_usd_per_day: float | None = None
    max_requests_per_minute: int | None = None


@dataclass
class BudgetsConfig:
    window_hours: int
    tiers: dict[str, BudgetCaps]
    max_files_changed: int
    max_loc_changed: int
    max_wall_seconds: int
    max_subtask_attempts: int


@dataclass
class GoalsConfig:
    product_name: str
    vision: str
    priorities: list[dict[str, Any]]
    non_goals: list[str]
    invariants: list[str]
    stack: dict[str, str]
    raw: dict[str, Any]


@dataclass
class DangerousPathsConfig:
    require_human_approval: list[str]
    forbidden: list[str]


@dataclass
class EnvConfig:
    openrouter_api_key: str | None
    openrouter_base_url: str
    openrouter_referer: str
    openrouter_title: str
    claude_code_bin: str
    log_level: str
    ollama_enabled: bool
    ollama_base_url: str
    ollama_timeout: int
    snapshot_run_tests: bool
    pytest_timeout: int
    iteration_delay_sec: float
    snapshot_max_files: int
    verbose_llm: bool
    verbose_llm_chars: int
    parallel_workers: int


def load_env() -> EnvConfig:
    _oe = os.environ.get("OLLAMA_ENABLED", "1").lower()
    ollama_enabled = _oe not in ("0", "false", "no", "off")
    return EnvConfig(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openrouter_base_url=os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        openrouter_referer=os.environ.get(
            "OPENROUTER_REFERER", "https://github.com/continuoso"
        ),
        openrouter_title=os.environ.get("OPENROUTER_TITLE", "continuoso"),
        claude_code_bin=os.environ.get("CLAUDE_CODE_BIN", "claude"),
        log_level=os.environ.get("CONTINUOSO_LOG_LEVEL", "INFO"),
        ollama_enabled=ollama_enabled,
        ollama_base_url=os.environ.get(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"
        ),
        ollama_timeout=int(os.environ.get("OLLAMA_TIMEOUT", "180")),
        snapshot_run_tests=(
            os.environ.get("CONTINUOSO_SNAPSHOT_TESTS", "1").lower()
            not in ("0", "false", "no", "off")
        ),
        pytest_timeout=int(os.environ.get("CONTINUOSO_PYTEST_TIMEOUT", "300")),
        iteration_delay_sec=float(
            os.environ.get("CONTINUOSO_ITERATION_DELAY_SEC", "5")
        ),
        snapshot_max_files=int(
            os.environ.get("CONTINUOSO_SNAPSHOT_MAX_FILES", "500")
        ),
        verbose_llm=os.environ.get(
            "CONTINUOSO_VERBOSE_LLM", "0"
        ).lower()
        in ("1", "true", "yes", "on"),
        verbose_llm_chars=int(
            os.environ.get("CONTINUOSO_VERBOSE_LLM_CHARS", "8000")
        ),
        parallel_workers=max(
            1,
            int(os.environ.get("CONTINUOSO_PARALLEL_WORKERS", "1")),
        ),
    )


def _resolve_config_file(project_dir: Path, filename: str) -> Path:
    """Prefer per-project override; fall back to bundled default."""
    override = project_dir / STATE_DIRNAME / filename
    if override.exists():
        return override
    return BUNDLED_CONFIG / filename


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_routing(project_dir: Path) -> RoutingConfig:
    raw = _read_yaml(_resolve_config_file(project_dir, "routing.yaml"))
    tiers: dict[str, Tier] = {}
    for name, body in raw["tiers"].items():
        models = [ModelSpec(**m) for m in body["models"]]
        fb = body.get("fallback")
        fb_provider = fb["provider"] if fb else None
        fb_models = [ModelSpec(**m) for m in fb["models"]] if fb else []
        tiers[name] = Tier(
            name=name,
            provider=body["provider"],
            models=models,
            fallback_provider=fb_provider,
            fallback_models=fb_models,
        )
    explicit = raw.get("tier_order")
    if explicit is not None and not isinstance(explicit, list):
        raise ValueError("routing.yaml: tier_order must be a list of tier names")
    explicit_tier_order: list[str] | None = None
    if explicit:
        explicit_tier_order = [str(x) for x in explicit]

    return RoutingConfig(
        success_threshold=float(raw["success_threshold"]),
        escalation_attempts=int(raw["escalation_attempts"]),
        tiers=tiers,
        task_class_defaults=raw["task_class_defaults"],
        explicit_tier_order=explicit_tier_order,
    )


def _apply_ollama_model_override(routing: RoutingConfig) -> None:
    """If OLLAMA_MODEL is set, use it for the `local` tier's first model."""
    mid = os.environ.get("OLLAMA_MODEL", "").strip()
    if not mid:
        return
    loc = routing.tiers.get("local")
    if not loc or not loc.models:
        return
    ctx = loc.models[0].context
    loc.models[0] = ModelSpec(id=mid, context=ctx)


def load_budgets(project_dir: Path) -> BudgetsConfig:
    raw = _read_yaml(_resolve_config_file(project_dir, "budgets.yaml"))
    tiers = {name: BudgetCaps(**body) for name, body in raw["tiers"].items()}
    per_it = raw["per_iteration"]
    return BudgetsConfig(
        window_hours=int(raw["window_hours"]),
        tiers=tiers,
        max_files_changed=int(per_it["max_files_changed"]),
        max_loc_changed=int(per_it["max_loc_changed"]),
        max_wall_seconds=int(per_it["max_wall_seconds"]),
        max_subtask_attempts=int(per_it["max_subtask_attempts"]),
    )


def load_goals(project_dir: Path, explicit: Path | None = None) -> GoalsConfig:
    path = explicit or _resolve_config_file(project_dir, "goals.yaml")
    raw = _read_yaml(path)
    product = raw.get("product", {})
    return GoalsConfig(
        product_name=product.get("name", project_dir.name or "target-app"),
        vision=product.get("vision", ""),
        priorities=raw.get("priorities", []),
        non_goals=raw.get("non_goals", []),
        invariants=raw.get("invariants", []),
        stack=raw.get("stack", {}),
        raw=raw,
    )


def load_dangerous_paths(project_dir: Path) -> DangerousPathsConfig:
    raw = _read_yaml(_resolve_config_file(project_dir, "dangerous_paths.yaml"))
    return DangerousPathsConfig(
        require_human_approval=raw.get("require_human_approval", []),
        forbidden=raw.get("forbidden", []),
    )


@dataclass
class AppConfig:
    project_dir: Path
    env: EnvConfig
    routing: RoutingConfig
    budgets: BudgetsConfig
    goals: GoalsConfig
    danger: DangerousPathsConfig
    # Optional theme for this `continuoso run` (CLI prompt or CONTINUOSO_SESSION_FOCUS).
    session_focus: str | None = None

    @property
    def state_dir(self) -> Path:
        return self.project_dir / STATE_DIRNAME

    @property
    def memory_db(self) -> Path:
        return self.state_dir / "memory.db"

    @property
    def worktrees_dir(self) -> Path:
        return self.state_dir / "worktrees"

    @classmethod
    def load(
        cls,
        project_dir: Path,
        goals_path: Path | None = None,
        *,
        session_focus: str | None = None,
    ) -> "AppConfig":
        project_dir = project_dir.resolve()
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / STATE_DIRNAME).mkdir(exist_ok=True)
        routing = load_routing(project_dir)
        _apply_ollama_model_override(routing)
        return cls(
            project_dir=project_dir,
            env=load_env(),
            routing=routing,
            budgets=load_budgets(project_dir),
            goals=load_goals(project_dir, goals_path),
            danger=load_dangerous_paths(project_dir),
            session_focus=session_focus,
        )


def scaffold_project(project_dir: Path) -> Path:
    """Create `<project>/.continuoso/goals.yaml` with a starter template.

    Returns the path to the created (or existing) goals file.
    """
    project_dir = project_dir.resolve()
    state = project_dir / STATE_DIRNAME
    state.mkdir(parents=True, exist_ok=True)
    goals = state / "goals.yaml"
    if goals.exists():
        return goals
    # Use bundled goals.yaml as the starter template, swapping in the project name.
    template = (BUNDLED_CONFIG / "goals.yaml").read_text(encoding="utf-8")
    template = template.replace('name: "target-app"', f'name: "{project_dir.name}"')
    goals.write_text(template, encoding="utf-8")
    return goals
