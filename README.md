# continuoso

Autonomous self-improving development loop. Plans, builds, tests, and iterates on any project folder, routing work between Claude Code (heavy reasoning) and OpenRouter (cheap/free coding models) based on measured per-task success.

## How it works

```
OBSERVE → REFLECT → PLAN → DECOMPOSE → ROUTE → EXECUTE
   ↑                                              │
 COMMIT ← MERGE ← EVALUATE ← SANDBOX-RUN ←────────┘
   │         │
   └── LEARN ┘
```

- **Observer** snapshots the target repo (files, tests, coverage, churn).
- **Planner** scores gaps against the project's `goals.yaml` and emits a task DAG.
- **Router** picks the cheapest model with ≥0.8 historical success for the task class; escalates on failure.
- **Executor** runs each task inside a git worktree via Claude Code or an OpenRouter model.
- **Evaluator** gates merge on tests, lint, invariants, and size caps.
- **Memory** (SQLite) records every attempt for online learning.

## Install

```bash
git clone <this-repo> continuoso
cd continuoso
pip install -e .
cp .env.example .env      # add OPENROUTER_API_KEY
```

Optionally make sure `claude` (Claude Code CLI) is on your PATH — continuoso will auto-detect it and use it for the heavy tier.

## Run on any project

```bash
# from inside a project folder:
continuoso init
continuoso run

# or point at any folder from anywhere:
continuoso init /path/to/my-project
continuoso run  /path/to/my-project
continuoso status /path/to/my-project
continuoso rollback 42 /path/to/my-project
```

`init` creates:

```
<project>/
├── .git/                     # initialized if missing
└── .continuoso/
    ├── goals.yaml            # edit this to steer the loop
    ├── memory.db             # created on first run
    ├── worktrees/            # ephemeral per-iteration
    └── .gitignore            # keeps state out of your repo
```

Stop the loop anytime with Ctrl+C — SQLite state is persisted; resume with `continuoso run` again.

## Per-project config

Any file in `<project>/.continuoso/` overrides the bundled default:

- `goals.yaml` — product vision, priorities, invariants. **You will edit this.**
- `routing.yaml` — model tiers and per-task-class defaults.
- `budgets.yaml` — daily token/cost caps per tier + per-iteration size caps.
- `dangerous_paths.yaml` — paths requiring human approval.

Bundled defaults live in the `continuoso` install's `config/` folder.

## Model tiers

| Tier | Default models | Used for |
|---|---|---|
| `free` | deepseek-v3\:free, qwen-coder-32b\:free, llama-3.3-70b\:free, gemini-2.0-flash-exp\:free | observation summaries, lint fixes, doc generation |
| `cheap` | deepseek-v3, claude-haiku-4.5, gpt-4o-mini | single-file edits, test authoring |
| `heavy` | claude-sonnet-4.6, claude-opus-4.6 (via Claude Code CLI) | cross-file refactors, planning, debugging |
| `heavy` fallback (via OpenRouter) | claude-sonnet-4.5, gpt-4o, grok-code-fast-1 | used if Claude Code CLI unavailable |

The router keeps per-`(task_class, model)` success rates in SQLite. Starts at the cheapest tier whose history for that class meets the threshold, escalates on parse/LLM/apply failure, and records each outcome so future selection improves.

## Commands

```
continuoso init  [PATH]                  scaffold .continuoso/goals.yaml and git repo
continuoso run   [PATH] [--once] [--max-iterations N] [--goals PATH]
continuoso status [PATH]                 table of recent iterations + router stats
continuoso rollback N [PATH]             revert the squash-merge commit tagged iter-N
```

## Safeguards

- **Worktree isolation** — main is never written to directly.
- **Evaluator gate** — tests/invariants/size caps must all pass before merge.
- **Path guards** — `dangerous_paths.yaml` blocks edits to CI configs, migrations, secrets, etc.
- **Size caps** — default max 8 files and 400 LOC per iteration.
- **Failure fingerprints** — same task+error 3× → quarantined.
- **Progress invariant** — no merged iteration in last 10 triggers a diversify signal.
- **Daily budget caps** — per-tier token and USD limits in `budgets.yaml`.
