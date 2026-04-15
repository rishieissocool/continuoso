"""SQLite-backed episodic memory + router statistics.

Tables:
  iterations         one row per loop iteration
  subtasks           one row per attempted subtask (may retry)
  router_stats       (task_class, provider, model) success/cost aggregates
  lessons            distilled textual lessons learned
  budget_usage       per-tier token/cost counters, bucketed by day
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    finished_at REAL,
    goal TEXT,
    chosen_gap_id TEXT,
    outcome TEXT,            -- merged | rolled_back | aborted
    score REAL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS subtasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration_id INTEGER NOT NULL,
    subtask_slug TEXT,
    task_class TEXT,
    provider TEXT,
    model TEXT,
    attempt INTEGER DEFAULT 1,
    success INTEGER,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    error TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY (iteration_id) REFERENCES iterations(id)
);

CREATE TABLE IF NOT EXISTS router_stats (
    task_class TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    successes INTEGER NOT NULL DEFAULT 0,
    total_cost_usd REAL NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    total_latency_ms INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (task_class, provider, model)
);

CREATE TABLE IF NOT EXISTS failure_fingerprints (
    fingerprint TEXT PRIMARY KEY,
    count INTEGER NOT NULL DEFAULT 1,
    last_seen REAL NOT NULL,
    quarantined INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    topic TEXT NOT NULL,
    body TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS budget_usage (
    day TEXT NOT NULL,        -- YYYY-MM-DD
    tier TEXT NOT NULL,
    tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (day, tier)
);
"""


@dataclass
class SubtaskRecord:
    iteration_id: int
    subtask_slug: str
    task_class: str
    provider: str
    model: str
    success: bool
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    attempt: int = 1
    error: str | None = None


@dataclass
class RouterStat:
    task_class: str
    provider: str
    model: str
    attempts: int
    successes: int
    total_cost_usd: float
    total_tokens: int
    total_latency_ms: int

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    @property
    def avg_cost(self) -> float:
        return self.total_cost_usd / self.attempts if self.attempts else 0.0


class Memory:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(db_path), isolation_level=None, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ---- Iterations ----
    def start_iteration(self, goal: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO iterations (started_at, goal) VALUES (?, ?)",
                (time.time(), goal),
            )
            return int(cur.lastrowid)

    def finish_iteration(
        self,
        iteration_id: int,
        *,
        outcome: str,
        score: float | None,
        chosen_gap_id: str | None,
        notes: str = "",
    ) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE iterations
                   SET finished_at=?, outcome=?, score=?, chosen_gap_id=?, notes=?
                   WHERE id=?""",
                (time.time(), outcome, score, chosen_gap_id, notes, iteration_id),
            )

    def last_iterations(self, n: int = 10) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM iterations ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- Subtasks ----
    def record_subtask(self, r: SubtaskRecord) -> None:
        with self._lock:
            self._record_subtask_unlocked(r)

    def _record_subtask_unlocked(self, r: SubtaskRecord) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO subtasks
                   (iteration_id, subtask_slug, task_class, provider, model,
                    attempt, success, input_tokens, output_tokens,
                    cost_usd, latency_ms, error, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    r.iteration_id,
                    r.subtask_slug,
                    r.task_class,
                    r.provider,
                    r.model,
                    r.attempt,
                    1 if r.success else 0,
                    r.input_tokens,
                    r.output_tokens,
                    r.cost_usd,
                    r.latency_ms,
                    r.error,
                    time.time(),
                ),
            )
            # Upsert router stat.
            c.execute(
                """INSERT INTO router_stats
                   (task_class, provider, model, attempts, successes,
                    total_cost_usd, total_tokens, total_latency_ms)
                   VALUES (?,?,?,1,?,?,?,?)
                   ON CONFLICT(task_class, provider, model) DO UPDATE SET
                     attempts = attempts + 1,
                     successes = successes + excluded.successes,
                     total_cost_usd = total_cost_usd + excluded.total_cost_usd,
                     total_tokens = total_tokens + excluded.total_tokens,
                     total_latency_ms = total_latency_ms + excluded.total_latency_ms
                """,
                (
                    r.task_class,
                    r.provider,
                    r.model,
                    1 if r.success else 0,
                    r.cost_usd,
                    r.input_tokens + r.output_tokens,
                    r.latency_ms,
                ),
            )

    def router_stats(self, task_class: str) -> list[RouterStat]:
        rows = self._conn.execute(
            "SELECT * FROM router_stats WHERE task_class=?", (task_class,)
        ).fetchall()
        return [
            RouterStat(
                task_class=r["task_class"],
                provider=r["provider"],
                model=r["model"],
                attempts=r["attempts"],
                successes=r["successes"],
                total_cost_usd=r["total_cost_usd"],
                total_tokens=r["total_tokens"],
                total_latency_ms=r["total_latency_ms"],
            )
            for r in rows
        ]

    # ---- Failure fingerprints ----
    def bump_fingerprint(self, fp: str) -> int:
        with self.tx() as c:
            c.execute(
                """INSERT INTO failure_fingerprints (fingerprint, count, last_seen)
                   VALUES (?, 1, ?)
                   ON CONFLICT(fingerprint) DO UPDATE SET
                     count = count + 1, last_seen = excluded.last_seen""",
                (fp, time.time()),
            )
            row = c.execute(
                "SELECT count FROM failure_fingerprints WHERE fingerprint=?", (fp,)
            ).fetchone()
            return int(row["count"]) if row else 1

    def quarantine(self, fp: str) -> None:
        self._conn.execute(
            "UPDATE failure_fingerprints SET quarantined=1 WHERE fingerprint=?",
            (fp,),
        )

    def is_quarantined(self, fp: str) -> bool:
        row = self._conn.execute(
            "SELECT quarantined FROM failure_fingerprints WHERE fingerprint=?",
            (fp,),
        ).fetchone()
        return bool(row and row["quarantined"])

    # ---- Budget ----
    def add_usage(self, tier: str, tokens: int, cost_usd: float) -> None:
        day = time.strftime("%Y-%m-%d")
        with self._lock:
            self._add_usage_unlocked(day, tier, tokens, cost_usd)

    def _add_usage_unlocked(
        self, day: str, tier: str, tokens: int, cost_usd: float
    ) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO budget_usage (day, tier, tokens, cost_usd)
                   VALUES (?,?,?,?)
                   ON CONFLICT(day, tier) DO UPDATE SET
                     tokens = tokens + excluded.tokens,
                     cost_usd = cost_usd + excluded.cost_usd""",
                (day, tier, tokens, cost_usd),
            )

    def get_usage_today(self, tier: str) -> tuple[int, float]:
        day = time.strftime("%Y-%m-%d")
        row = self._conn.execute(
            "SELECT tokens, cost_usd FROM budget_usage WHERE day=? AND tier=?",
            (day, tier),
        ).fetchone()
        if not row:
            return 0, 0.0
        return int(row["tokens"]), float(row["cost_usd"])

    # ---- Lessons ----
    def save_lesson(self, topic: str, body: str) -> None:
        self._conn.execute(
            "INSERT INTO lessons (created_at, topic, body) VALUES (?,?,?)",
            (time.time(), topic, body),
        )

    def recent_lessons(self, n: int = 10) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM lessons ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]
