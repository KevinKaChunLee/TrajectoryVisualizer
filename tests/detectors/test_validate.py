"""Tests for validate-phase detectors."""

from __future__ import annotations

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors.validate import (
    detect_late_validation,
    detect_test_retry_loop,
    detect_validation_avoidance,
)

from tests.detectors.conftest import step


def _write(i: int, tgt: str = "a.py"):
    return step(i, "FILE_WRITE", target=tgt, tool="Edit")


def _pytest(i: int, failed: bool = False, err: str = ""):
    return step(
        i, "COMMAND",
        target="pytest tests/",
        tool="Bash",
        effect_label="failed" if failed else "survived",
        args={"stderr": err} if failed else {},
    )


# ---------------------------------------------------------------------------
# late-validation
# ---------------------------------------------------------------------------

def test_late_validation_fires_after_ten_writes() -> None:
    ctx = DetectorContext()
    steps = [_write(i) for i in range(10)] + [_pytest(10)]
    hits = detect_late_validation(steps, ctx)
    assert len(hits) == 1


def test_late_validation_not_fires_if_validation_is_early() -> None:
    ctx = DetectorContext()
    steps = [_write(0), _pytest(1)] + [_write(i) for i in range(2, 12)]
    assert detect_late_validation(steps, ctx) == []


# ---------------------------------------------------------------------------
# validation-avoidance
# ---------------------------------------------------------------------------

def test_validation_avoidance_fires_on_no_validation_at_all() -> None:
    ctx = DetectorContext()
    steps = [_write(i) for i in range(3)]
    hits = detect_validation_avoidance(steps, ctx)
    assert len(hits) == 1
    assert hits[0].evidence["mode"] == "no-validation"


def test_validation_avoidance_fires_on_bad_ratio() -> None:
    ctx = DetectorContext()
    # 6 writes, 1 validation -> ratio 6 > 5.0
    steps = [_write(i) for i in range(6)] + [_pytest(6)]
    hits = detect_validation_avoidance(steps, ctx)
    assert len(hits) == 1
    assert hits[0].evidence["mode"] == "ratio"


def test_validation_avoidance_not_fires_on_balanced_ratio() -> None:
    ctx = DetectorContext()
    steps = [_write(0), _pytest(1), _write(2), _pytest(3)]
    assert detect_validation_avoidance(steps, ctx) == []


# ---------------------------------------------------------------------------
# test-retry-loop
# ---------------------------------------------------------------------------

def test_test_retry_loop_fires_on_two_retries_same_error_no_edit() -> None:
    ctx = DetectorContext()
    # 3 failed runs of the same command with same error, no edit in between.
    steps = [
        _pytest(0, failed=True, err="AssertionError: foo"),
        _pytest(1, failed=True, err="AssertionError: foo"),
        _pytest(2, failed=True, err="AssertionError: foo"),
    ]
    hits = detect_test_retry_loop(steps, ctx)
    assert len(hits) == 1
    assert hits[0].evidence["retries"] == 2


def test_test_retry_loop_not_fires_with_edit_between() -> None:
    ctx = DetectorContext()
    steps = [
        _pytest(0, failed=True, err="AssertionError: foo"),
        _write(1),
        _pytest(2, failed=True, err="AssertionError: foo"),
    ]
    assert detect_test_retry_loop(steps, ctx) == []
