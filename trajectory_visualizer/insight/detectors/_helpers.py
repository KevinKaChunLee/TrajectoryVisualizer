"""Shared predicates for classifying canonical-action steps.

Detectors use these rather than hardcoding tool-name sets, so a new scaffold
with a new tool name is a single-file update here.
"""

from __future__ import annotations

from typing import Any

# Action-type constants match trajectory_visualizer/converge/canonical.py.
FILE_READ = "FILE_READ"
FILE_WRITE = "FILE_WRITE"
SEARCH = "SEARCH"
COMMAND = "COMMAND"
AGENT_SPAWN = "AGENT_SPAWN"
REASON = "REASON"


_PLANNING_TOOLS = frozenset({
    "todowrite", "taskcreate", "taskupdate", "tasklist", "plan", "update_plan",
})

_SHELL_TOOLS = frozenset({"bash", "shell", "execute_command", "terminal"})

_STRUCTURED_READ_TOOLS = frozenset({"read", "grep", "glob"})

# Validation-command substring detection (mirrors canonical._VALIDATION_COMMAND_PATTERNS).
_VALIDATION_SUBSTRINGS = (
    "pytest", "unittest", "tox", "nox", "go test",
    "cargo test", "npm test", "pnpm test", "yarn test", "jest", "vitest",
    "mvn test", "gradle test", "bazel test", "make test", "ctest", "ruff",
    "flake8", "pylint", "mypy", "eslint", "lint", "check", "verify",
)


def action_type(step: Any) -> str:
    """Return the canonical action_type of a step (CanonicalAction or dict)."""
    if hasattr(step, "action_type"):
        return str(step.action_type)
    if isinstance(step, dict):
        return str(step.get("action_type", ""))
    return ""


def target(step: Any) -> str:
    """Return the step's target path/query/command (empty when not set)."""
    if hasattr(step, "target"):
        return str(step.target)
    if isinstance(step, dict):
        return str(step.get("target", ""))
    return ""


def tool(step: Any) -> str:
    """Return the step's raw tool name (empty when not set)."""
    if hasattr(step, "tool"):
        return str(step.tool)
    if isinstance(step, dict):
        return str(step.get("tool", ""))
    return ""


def effect_label(step: Any) -> str:
    """Return the effect_label (survived/failed/reverted/justified/unknown)."""
    if hasattr(step, "effect_label"):
        return str(step.effect_label)
    if isinstance(step, dict):
        return str(step.get("effect_label", "unknown"))
    return "unknown"


def status(step: Any) -> str:
    """Return the raw tool status."""
    if hasattr(step, "status"):
        return str(step.status)
    if isinstance(step, dict):
        return str(step.get("status", ""))
    return ""


def phase_label(step: Any) -> str | None:
    """Return the semantic phase_label if present."""
    if hasattr(step, "phase_label"):
        v = step.phase_label
        return None if v is None else str(v)
    if isinstance(step, dict):
        v = step.get("phase_label")
        return None if v is None else str(v)
    return None


def action_label(step: Any) -> str | None:
    """Return the semantic action_label if present."""
    if hasattr(step, "action_label"):
        v = step.action_label
        return None if v is None else str(v)
    if isinstance(step, dict):
        v = step.get("action_label")
        return None if v is None else str(v)
    return None


def args(step: Any) -> dict:
    """Return the step's args dict."""
    if hasattr(step, "args"):
        return dict(step.args or {})
    if isinstance(step, dict):
        return dict(step.get("args") or {})
    return {}


def is_read(step: Any) -> bool:
    return action_type(step) == FILE_READ


def is_write(step: Any) -> bool:
    return action_type(step) == FILE_WRITE


def is_search(step: Any) -> bool:
    return action_type(step) == SEARCH


def is_command(step: Any) -> bool:
    return action_type(step) == COMMAND


def is_planning(step: Any) -> bool:
    """Planning/todo-tool calls (TodoWrite, etc.)."""
    return tool(step).lower() in _PLANNING_TOOLS


def is_validation_command(step: Any) -> bool:
    """True for COMMAND steps whose target looks like a validation invocation."""
    if not is_command(step):
        return False
    cmd = target(step).lower()
    return any(sub in cmd for sub in _VALIDATION_SUBSTRINGS)


def is_shell(step: Any) -> bool:
    """True for Bash/shell tool calls."""
    return tool(step).lower() in _SHELL_TOOLS


def is_structured_read(step: Any) -> bool:
    """True for Read/Grep/Glob calls (not shell-based reads)."""
    return tool(step).lower() in _STRUCTURED_READ_TOOLS


def is_failed(step: Any) -> bool:
    """True when effect_label == 'failed' OR status looks like an error."""
    if effect_label(step) == "failed":
        return True
    st = status(step).lower()
    return st in {"error", "failed", "failure", "timeout", "exception"}


def step_index(step: Any) -> int:
    """Return the 0-based step index."""
    if hasattr(step, "step_index"):
        return int(step.step_index)
    if isinstance(step, dict):
        return int(step.get("step_index", 0))
    return 0


def error_signature(step: Any) -> str:
    """Return a coarse error signature for grouping spiral-like recurrences.

    Shape: '<tool>:<first-line-of-error>' (lowercased, truncated).
    Empty string when the step has no error signal.
    """
    if not is_failed(step):
        return ""
    t = tool(step).lower()
    # Try to find a useful error text in args/status.
    err = status(step) or ""
    a = args(step)
    for key in ("stderr", "error", "message", "output"):
        v = a.get(key)
        if v:
            err = str(v)
            break
    first_line = err.strip().splitlines()[0] if err.strip() else ""
    return f"{t}:{first_line[:80].lower().strip()}"
