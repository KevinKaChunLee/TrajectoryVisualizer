"""Classify step failures as scaffold (system) vs agentic (tool) errors."""

from __future__ import annotations

from trajviz.tool_vocab import SYSTEM_TOOL_NAMES

_FAILURE_STATUSES = frozenset({"error", "failed", "failure", "cancelled", "timeout"})


def tool_call_failed(tc: dict) -> bool:
    """True when a tool call clearly failed (status or non-zero exit)."""
    status = str(tc.get("status") or "").lower()
    if status in _FAILURE_STATUSES:
        return True
    meta = tc.get("metadata")
    if isinstance(meta, dict) and meta.get("exit") not in (None, 0):
        return True
    return bool(tc.get("error") or tc.get("error_type"))


def is_system_tool_name(name: object) -> bool:
    return isinstance(name, str) and name in SYSTEM_TOOL_NAMES


def step_error_kind(step: dict) -> str | None:
    """Return ``\"system\"``, ``\"tool\"``, or ``None`` for a step.

    System errors (amber): failures of scaffold primitives (Grep, Read, Write,
    …) or a provider abort with ``finish == \"error\"``.

    Tool errors (red): failures of Bash (user scripts), Skill, Task, MCP, and
    other agentic / workflow-defined tools. When a step has both, tool wins so
    workflow failures stay visible.
    """
    failed = [tc for tc in step.get("tool_calls") or [] if tool_call_failed(tc)]
    if failed:
        if any(not is_system_tool_name(tc.get("tool_name")) for tc in failed):
            return "tool"
        return "system"

    finish = step.get("finish") or ""
    if isinstance(finish, str) and finish.strip().lower() == "error":
        return "system"
    # Fallback when tool_calls were not attached but the step was flagged.
    if (step.get("error_count") or 0) > 0:
        return "tool"
    return None
