"""Debug without hypothesis detector ([H])."""

from __future__ import annotations

from typing import Any

from trajectory_visualizer.core.detection import DetectorContext, PatternDetection


_REPRODUCE_LABELS = frozenset({"debug_reproduce", "reproduce_failure", "run_test"})
_ROOT_CAUSE_LABELS = frozenset({"root_cause_analysis", "diagnose", "hypothesize"})


def detect(steps: list[Any], context: DetectorContext) -> list[PatternDetection]:
    """Debug without hypothesis ([H]).

    Operational definition (appendix_catalog.tex, [H]):
    ">=3 debug-reproduction steps without any root-cause-analysis step
    (depends on semantic labeler)."
    """
    if not context.labels:
        return []
    min_reproduce = int(context.thresholds_for("debug-wo-hypothesis")["min_reproduce_steps"])

    reproduce_steps: list[int] = []
    has_root_cause = False
    for i in sorted(context.labels.keys()):
        action = context.labels[i].get("action", "")
        phase = context.labels[i].get("phase", "")
        if phase != "debug":
            continue
        if action in _REPRODUCE_LABELS:
            reproduce_steps.append(i)
        if action in _ROOT_CAUSE_LABELS:
            has_root_cause = True

    if len(reproduce_steps) >= min_reproduce and not has_root_cause:
        return [
            PatternDetection(
                detector_id="debug-wo-hypothesis",
                span=(reproduce_steps[0], reproduce_steps[-1]),
                evidence={
                    "reproduce_steps": len(reproduce_steps),
                    "root_cause_observed": False,
                },
            )
        ]
    return []
