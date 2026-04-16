"""Tests for plan-phase detectors."""

from __future__ import annotations

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors.plan import (
    detect_plan_less_execution,
    detect_plan_stall,
    detect_plan_thrash,
)

from tests.detectors.conftest import step


def _plan(i: int, items: list[str]):
    return step(i, "COMMAND", tool="TodoWrite", args={"todos": items})


def _edit(i: int, tgt: str):
    return step(i, "FILE_WRITE", target=tgt, tool="Edit")


# ---------------------------------------------------------------------------
# plan-stall
# ---------------------------------------------------------------------------

def test_plan_stall_fires_on_five_plan_steps_then_implement() -> None:
    ctx = DetectorContext()
    steps = [_plan(i, [f"a{i}"]) for i in range(5)] + [_edit(5, "a.py")]
    hits = detect_plan_stall(steps, ctx)
    assert len(hits) == 1


def test_plan_stall_not_fires_on_four_plan_steps() -> None:
    ctx = DetectorContext()
    steps = [_plan(i, [f"a{i}"]) for i in range(4)] + [_edit(4, "a.py")]
    assert detect_plan_stall(steps, ctx) == []


# ---------------------------------------------------------------------------
# plan-thrash
# ---------------------------------------------------------------------------

def test_plan_thrash_fires_on_high_turnover_without_execution() -> None:
    ctx = DetectorContext()
    steps = [
        _plan(0, ["a", "b", "c"]),
        _plan(1, ["d", "e", "f"]),  # full turnover
        _plan(2, ["g", "h", "i"]),  # full turnover
    ]
    hits = detect_plan_thrash(steps, ctx)
    assert len(hits) == 1
    assert hits[0].evidence["rewrites"] == 3


def test_plan_thrash_not_fires_when_execution_happens() -> None:
    ctx = DetectorContext()
    steps = [
        _plan(0, ["a"]),
        _plan(1, ["b"]),
        _plan(2, ["c"]),
        _edit(3, "x.py"),
    ]
    assert detect_plan_thrash(steps, ctx) == []


def test_plan_thrash_fires_on_terminal_block_even_after_prior_writes() -> None:
    """Regression: prior `executed_after_plan=True` bug silenced all subsequent
    plan blocks. A thrash block with no downstream execution must still fire."""
    ctx = DetectorContext()
    steps = [
        _edit(0, "a.py"),
        _plan(1, ["a", "b", "c"]),
        _plan(2, ["d", "e", "f"]),
        _plan(3, ["g", "h", "i"]),
    ]
    hits = detect_plan_thrash(steps, ctx)
    assert len(hits) == 1, "terminal plan-thrash block must fire despite earlier write"


# ---------------------------------------------------------------------------
# plan-less-execution
# ---------------------------------------------------------------------------

def test_plan_less_execution_fires_on_five_writes_no_planning() -> None:
    ctx = DetectorContext()
    steps = [_edit(i, f"f{i}.py") for i in range(5)]
    hits = detect_plan_less_execution(steps, ctx)
    assert len(hits) == 1
    assert hits[0].evidence["file_writes"] == 5


def test_plan_less_execution_not_fires_when_planning_present() -> None:
    ctx = DetectorContext()
    steps = [_plan(0, ["a"])] + [_edit(i, f"f{i}.py") for i in range(1, 6)]
    assert detect_plan_less_execution(steps, ctx) == []
