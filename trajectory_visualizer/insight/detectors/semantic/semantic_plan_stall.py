"""Semantic plan stall detector ([H])."""

from __future__ import annotations

from typing import Any

from trajectory_visualizer.core.detection import DetectorContext, PatternDetection


def detect(steps: list[Any], context: DetectorContext) -> list[PatternDetection]:
    """Semantic plan stall ([H]).

    Operational definition (appendix_catalog.tex, [H]):
    ">=5 plan-phase steps without any implement step (depends on semantic
    labeler)."
    """
    if not context.labels:
        return []
    min_plan = int(context.thresholds_for("semantic-plan-stall")["min_plan_steps"])

    plan_count = 0
    first_plan_idx: int | None = None
    for i in sorted(context.labels.keys()):
        phase = context.labels[i].get("phase", "")
        if phase == "plan":
            if first_plan_idx is None:
                first_plan_idx = i
            plan_count += 1
        elif phase == "implement":
            if plan_count >= min_plan and first_plan_idx is not None:
                return [
                    PatternDetection(
                        detector_id="semantic-plan-stall",
                        span=(first_plan_idx, i - 1),
                        evidence={"plan_phase_steps": plan_count},
                    )
                ]
            plan_count = 0
            first_plan_idx = None

    if plan_count >= min_plan and first_plan_idx is not None:
        return [
            PatternDetection(
                detector_id="semantic-plan-stall",
                span=(first_plan_idx, max(context.labels.keys())),
                evidence={"plan_phase_steps": plan_count, "terminal": True},
            )
        ]
    return []
