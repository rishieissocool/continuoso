"""All system prompts in one place. Kept short to save input tokens."""
from __future__ import annotations

# One dense block — repeated every LLM call.
SYSTEM_BASE = (
    "continuoso: autonomous engineer on a git worktree. "
    "Small steps; no invented paths; keep tests green. "
    "If snapshot.test_ok is false, fix tests before features. "
    "Return only JSON when the user prompt asks for JSON."
)

# Passed into REFLECT_PROMPT.format(task_classes=...)
TASK_CLASSES = (
    "edit_single_file,edit_cross_file,author_tests,fix_failing_tests,"
    "refactor,generate_docs,dependency_bump"
)


def format_session_focus(raw: str | None) -> str:
    """Compact line for prompts; empty → token-saving placeholder."""
    t = (raw or "").strip()
    return t if t else "(none)"


REFLECT_PROMPT = """Find gaps between goals and repo. JSON only.

Schema: {{"gaps":[{{"id":"","title":"","rationale":"","touches":[],"task_class":"","est_loc":0,"priority_id":""}}]}}
task_class: one of {task_classes}

Rules: If test_ok is false or test (pytest) shows failures, include ≥1 gap with task_class fix_failing_tests or author_tests among top items. Else correctness→tests→polish. Target repo only.

Session focus:{session_focus}
Prefer gaps matching focus; failing tests still win.

Goals:{goals}
State:{state}"""

PLAN_PROMPT = """One gap → 1–6 ordered subtasks. JSON only.

Schema: {{"iteration_goal":"","chosen_gap_id":"","subtasks":[{{"id":"","task_class":"","instruction":"","files":[],"acceptance_criteria":[]}}]}}

Rules: Mechanical acceptance_criteria. author_tests before/with code if tests missing; fix_failing_tests first if red. Cap ≤{max_loc} LOC, ≤{max_files} files.

Session focus:{session_focus}
Align iteration_goal and subtasks with focus when tests allow.

Gaps:{gaps}
State:{state}"""

EXECUTE_PROMPT = """One subtask; minimal diff; keep tests green.

Session focus:{session_focus}

{instruction}
Files:{files}
Criteria:{criteria}
{file_contents}

JSON: {{"changes":[{{"path":"","action":"create|modify|delete","content":""}}],"notes":""}}"""

COMMIT_MSG_PROMPT = """Conventional commit ≤72 chars (feat:/fix:/test:/chore: lowercase):

{diff}

One line only."""
