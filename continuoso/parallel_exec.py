"""Partition subtasks into waves with disjoint file sets for parallel execution."""
from __future__ import annotations

from .planner import Subtask


def _norm_path(p: str) -> str:
    return p.replace("\\", "/").strip().lstrip("/")


def _file_set(st: Subtask) -> set[str]:
    return {_norm_path(f) for f in (st.files or []) if f}


def _overlaps_scope(a: Subtask, b: Subtask) -> bool:
    """True if subtasks must not run in parallel (same file or unknown scope)."""
    fa, fb = _file_set(a), _file_set(b)
    if not fa or not fb:
        return True
    return bool(fa & fb)


def partition_into_waves(subtasks: list[Subtask]) -> list[list[Subtask]]:
    """Greedy waves: within each wave, subtasks have pairwise disjoint `files`.

    Empty `files` is treated as unknown scope — never parallelized with others.

    Waves run sequentially (main updates between waves). Subtasks inside a wave
    may run in parallel (one worktree each).
    """
    if not subtasks:
        return []
    ordered = sorted(subtasks, key=lambda s: (-len(s.files or []), s.id))
    waves: list[list[Subtask]] = []
    for st in ordered:
        placed = False
        for wave in waves:
            if all(not _overlaps_scope(st, o) for o in wave):
                wave.append(st)
                placed = True
                break
        if not placed:
            waves.append([st])
    return waves
