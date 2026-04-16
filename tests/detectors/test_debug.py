"""Tests for debug-phase detectors."""

from __future__ import annotations

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors.debug import (
    detect_error_spiral,
    detect_recovery_free_retry,
)

from tests.detectors.conftest import step


def _failed_bash(i: int, cmd: str, err: str):
    return step(
        i,
        "COMMAND",
        target=cmd,
        tool="Bash",
        effect_label="failed",
        args={"stderr": err},
    )


# ---------------------------------------------------------------------------
# error-spiral
# ---------------------------------------------------------------------------

def test_error_spiral_fires_on_three_same_error_signatures() -> None:
    ctx = DetectorContext()
    steps = [
        _failed_bash(0, "pytest a", "AssertionError: x"),
        _failed_bash(1, "pytest a", "AssertionError: x"),
        _failed_bash(2, "pytest a", "AssertionError: x"),
    ]
    hits = detect_error_spiral(steps, ctx)
    assert len(hits) == 1
    assert hits[0].evidence["recurrences"] == 3


def test_error_spiral_not_fires_on_two() -> None:
    ctx = DetectorContext()
    steps = [
        _failed_bash(0, "pytest a", "AssertionError: x"),
        _failed_bash(1, "pytest a", "AssertionError: x"),
    ]
    assert detect_error_spiral(steps, ctx) == []


def test_error_spiral_resets_on_intervening_write() -> None:
    """Paper: 'no observable change in approach'. An intervening edit resets."""
    ctx = DetectorContext()
    steps = [
        _failed_bash(0, "pytest a", "AssertionError: x"),
        _failed_bash(1, "pytest a", "AssertionError: x"),
        step(2, "FILE_WRITE", target="a.py", tool="Edit"),
        _failed_bash(3, "pytest a", "AssertionError: x"),
    ]
    assert detect_error_spiral(steps, ctx) == []


# ---------------------------------------------------------------------------
# recovery-free-retry
# ---------------------------------------------------------------------------

def test_recovery_free_retry_fires_on_identical_retry() -> None:
    ctx = DetectorContext()
    a = step(0, "COMMAND", target="pytest a", tool="Bash",
             effect_label="failed", args={"x": 1})
    b = step(1, "COMMAND", target="pytest a", tool="Bash",
             effect_label="failed", args={"x": 1})
    hits = detect_recovery_free_retry([a, b], ctx)
    assert len(hits) == 1


def test_recovery_free_retry_not_fires_on_different_args() -> None:
    ctx = DetectorContext()
    a = step(0, "COMMAND", target="pytest a", tool="Bash",
             effect_label="failed", args={"x": 1})
    b = step(1, "COMMAND", target="pytest a", tool="Bash",
             effect_label="failed", args={"x": 2})
    assert detect_recovery_free_retry([a, b], ctx) == []
