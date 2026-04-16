"""Phase 2: Plan detectors."""

from __future__ import annotations

from typing import Any

from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from . import _helpers as h


def _plan_items(step: Any) -> list[str]:
    """Extract the plan/todo items asserted by a planning step, as a list of
    stable string keys. Shape depends on the tool; we try common fields."""
    a = h.args(step)
    for key in ("todos", "items", "plan", "tasks"):
        v = a.get(key)
        if isinstance(v, list):
            return [str(_item_key(x)) for x in v]
    return []


def _item_key(x: Any) -> str:
    if isinstance(x, str):
        return x.strip().lower()
    if isinstance(x, dict):
        for k in ("id", "content", "text", "title", "description"):
            v = x.get(k)
            if isinstance(v, str):
                return v.strip().lower()
    return str(x)


def _is_implement(step: Any) -> bool:
    """Implement-phase proxy: structural (FILE_WRITE) or semantic (phase=implement)."""
    if h.is_write(step):
        return True
    p = h.phase_label(step)
    return p == "implement"


def detect_plan_stall(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Plan stall detector.

    Operational definition (appendix_catalog.tex, Phase 2):
    ">=5 planning/TodoWrite actions without any implement-phase step. Requires
    a structured todo tool."
    """
    thresh = int(context.thresholds_for("plan-stall")["min_plan_steps"])
    plan_count = 0
    first_plan_idx: int | None = None

    for i, step in enumerate(steps):
        if h.is_planning(step):
            if first_plan_idx is None:
                first_plan_idx = i
            plan_count += 1
        elif _is_implement(step):
            if plan_count >= thresh and first_plan_idx is not None:
                return [
                    PatternDetection(
                        detector_id="plan-stall",
                        span=(first_plan_idx, i - 1),
                        evidence={"plan_steps_before_implement": plan_count},
                    )
                ]
            plan_count = 0
            first_plan_idx = None

    # Trajectory ended without implement step after a stall.
    if plan_count >= thresh and first_plan_idx is not None:
        return [
            PatternDetection(
                detector_id="plan-stall",
                span=(first_plan_idx, len(steps) - 1),
                evidence={"plan_steps_before_implement": plan_count,
                          "terminal": True},
            )
        ]
    return []


def detect_plan_thrash(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Plan thrash detector.

    Operational definition (appendix_catalog.tex, Phase 2):
    "Repeated TodoWrite rewrites with high item-set turnover and no downstream
    execution."
    """
    thresholds = context.thresholds_for("plan-thrash")
    min_rewrites = int(thresholds["min_rewrites"])
    min_turnover = float(thresholds["min_item_turnover"])

    # Track the CURRENT plan block only. An implement step ends the block and
    # -- because the paper requires "no downstream execution" -- cancels any
    # would-be detection on that block. The fire only lands at trajectory end
    # if the trailing block is still a thrash.
    plan_snapshots: list[tuple[int, frozenset[str]]] = []
    first_plan_idx: int | None = None

    for i, step in enumerate(steps):
        if h.is_planning(step):
            if first_plan_idx is None:
                first_plan_idx = i
            items = frozenset(_plan_items(step))
            plan_snapshots.append((i, items))
        elif _is_implement(step):
            plan_snapshots = []
            first_plan_idx = None

    if len(plan_snapshots) < min_rewrites or first_plan_idx is None:
        return []

    turnover = _mean_pairwise_turnover(plan_snapshots)
    if turnover < min_turnover:
        return []
    return [
        PatternDetection(
            detector_id="plan-thrash",
            span=(first_plan_idx, plan_snapshots[-1][0]),
            evidence={
                "rewrites": len(plan_snapshots),
                "mean_turnover": round(turnover, 3),
                "no_execution": True,
            },
        )
    ]


def _mean_pairwise_turnover(snapshots: list[tuple[int, frozenset[str]]]) -> float:
    if len(snapshots) < 2:
        return 0.0
    ratios: list[float] = []
    for (_, a), (_, b) in zip(snapshots, snapshots[1:]):
        union = a | b
        if not union:
            continue
        sym_diff = a ^ b
        ratios.append(len(sym_diff) / len(union))
    return sum(ratios) / len(ratios) if ratios else 0.0


def detect_plan_less_execution(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Plan-less execution detector.

    Operational definition (appendix_catalog.tex, Phase 2):
    "Long trajectory with >=5 FILE_WRITE steps but zero TodoWrite calls. Fires
    only when the scaffold exposes a planning tool."
    """
    thresh = int(context.thresholds_for("plan-less-execution")["min_file_writes"])
    write_idxs = [i for i, s in enumerate(steps) if h.is_write(s)]
    plan_idxs = [i for i, s in enumerate(steps) if h.is_planning(s)]
    if len(write_idxs) >= thresh and not plan_idxs:
        return [
            PatternDetection(
                detector_id="plan-less-execution",
                span=(write_idxs[0], write_idxs[-1]),
                evidence={
                    "file_writes": len(write_idxs),
                    "planning_calls": 0,
                },
            )
        ]
    return []
