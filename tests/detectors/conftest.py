"""Shared fixtures/helpers for detector tests."""

from __future__ import annotations

from trajectory_visualizer.converge.canonical import CanonicalAction


def step(
    i: int,
    action_type: str,
    *,
    target: str = "",
    tool: str = "",
    args: dict | None = None,
    status: str = "",
    effect_label: str = "unknown",
    phase_label: str | None = None,
    action_label: str | None = None,
) -> CanonicalAction:
    """Build a CanonicalAction with sensible defaults for tests."""
    return CanonicalAction(
        step_index=i,
        action_type=action_type,
        target=target,
        tool=tool,
        args=args or {},
        status=status,
        effect_label=effect_label,
        phase_label=phase_label,
        action_label=action_label,
    )
