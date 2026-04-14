"""CLI entry point for continuoso.

Point at any project folder:
    continuoso init  [PATH]   # scaffold .continuoso/goals.yaml (PATH defaults to cwd)
    continuoso run   [PATH]
    continuoso status [PATH]
    continuoso rollback N [PATH]
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import AppConfig, scaffold_project
from .logging_setup import setup_logging
from .memory import Memory
from .orchestrator import Orchestrator
from .sandbox import ensure_repo, rollback_last_iteration

console = Console()
log = logging.getLogger(__name__)


def _path_arg() -> click.Argument:
    # Shared argument: optional project directory, defaults to cwd.
    return click.argument(
        "project",
        type=click.Path(file_okay=False, path_type=Path),
        required=False,
    )


def _resolve_project(project: Path | None) -> Path:
    return (project or Path.cwd()).resolve()


@click.group()
def cli() -> None:
    """continuoso — autonomous self-improving dev loop. Runs on any project folder."""


@cli.command()
@_path_arg()
def init(project: Path | None) -> None:
    """Initialize continuoso state in PROJECT (defaults to cwd).

    Creates `<project>/.continuoso/goals.yaml` and ensures a git repo exists.
    """
    setup_logging("INFO")
    project = _resolve_project(project)
    ensure_repo(project)
    goals = scaffold_project(project)
    console.print(f"[green]initialized[/green] {project}")
    console.print(f"edit your goals: [cyan]{goals}[/cyan]")


@cli.command()
@_path_arg()
@click.option("--goals", "goals_path", type=click.Path(exists=True, path_type=Path), default=None,
              help="Override goals.yaml path.")
@click.option("--max-iterations", type=int, default=None, help="Stop after N iterations.")
@click.option("--once", is_flag=True, help="Run exactly one iteration and exit.")
def run(
    project: Path | None,
    goals_path: Path | None,
    max_iterations: int | None,
    once: bool,
) -> None:
    """Start the continuous loop on PROJECT (defaults to cwd)."""
    project = _resolve_project(project)
    cfg = AppConfig.load(project, goals_path)
    setup_logging(cfg.env.log_level)

    if not cfg.env.openrouter_api_key:
        console.print(
            "[yellow]warning:[/yellow] OPENROUTER_API_KEY not set — "
            "only Claude Code CLI (heavy tier) will be usable"
        )

    console.rule(f"[bold cyan]continuoso[/bold cyan] project={project}")
    orch = Orchestrator(cfg)
    if once:
        res = orch.run_iteration()
        console.print(f"outcome: [bold]{res.outcome}[/bold]  score={res.score:.2f}")
        return
    try:
        orch.run_forever(max_iterations=max_iterations)
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted — state saved[/yellow]")
        sys.exit(130)


@cli.command()
@_path_arg()
def status(project: Path | None) -> None:
    """Show recent iterations and router stats for PROJECT."""
    setup_logging("WARNING")
    project = _resolve_project(project)
    cfg = AppConfig.load(project)
    db = cfg.memory_db
    if not db.exists():
        console.print(f"[dim]no memory db at {db} — run `continuoso run` first[/dim]")
        return
    mem = Memory(db)

    its = mem.last_iterations(15)
    t = Table(title=f"Recent iterations — {project.name}", show_lines=False)
    for col in ("id", "outcome", "score", "goal", "notes"):
        t.add_column(col)
    for row in its:
        t.add_row(
            str(row["id"]),
            str(row.get("outcome") or "-"),
            f"{row.get('score') or 0:.2f}",
            (row.get("goal") or "")[:60],
            (row.get("notes") or "")[:60],
        )
    console.print(t)

    cur = mem._conn.execute(
        """SELECT task_class, provider, model, attempts, successes,
                  ROUND(total_cost_usd, 4) AS cost
             FROM router_stats
             ORDER BY attempts DESC LIMIT 20"""
    )
    t2 = Table(title="Router stats")
    for col in ("task_class", "provider", "model", "attempts", "successes", "cost_usd"):
        t2.add_column(col)
    for r in cur.fetchall():
        t2.add_row(*[str(x) for x in r])
    console.print(t2)


@cli.command()
@click.argument("iteration_id", type=int)
@_path_arg()
def rollback(iteration_id: int, project: Path | None) -> None:
    """Revert the squash-merge commit tagged iter-ITERATION_ID."""
    project = _resolve_project(project)
    cfg = AppConfig.load(project)
    setup_logging(cfg.env.log_level)
    rollback_last_iteration(cfg.project_dir, iteration_id)
    console.print(f"[green]reverted iter-{iteration_id}[/green]")


if __name__ == "__main__":
    cli()
