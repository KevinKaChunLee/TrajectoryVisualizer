"""Premature implementation detector ([H])."""

from __future__ import annotations

from typing import Any

from trajectory_visualizer.core.detection import DetectorContext, PatternDetection


def detect(steps: list[Any], context: DetectorContext) -> list[PatternDetection]:
    """Premature implementation ([H]).

    Operational definition (appendix_catalog.tex, [H]):
    "First implement-phase step precedes any plan-phase step (depends on
    semantic labeler)."
    """
    if not context.labels:
        return []
    first_plan: int | None = None
    first_implement: int | None = None
    for i in sorted(context.labels.keys()):
        phase = context.labels[i].get("phase", "")
        if phase == "plan" and first_plan is None:
            first_plan = i
        if phase == "implement" and first_implement is None:
            first_implement = i
        if first_plan is not None and first_implement is not None:
            break
    if first_implement is None:
        return []
    if first_plan is not None and first_plan <= first_implement:
        return []
    return [
        PatternDetection(
            detector_id="premature-implementation",
            span=(0, first_implement),
            evidence={
                "first_implement_step": first_implement,
                "first_plan_step": first_plan,
            },
        )
    ]
