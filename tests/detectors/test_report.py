"""Tests for report-phase detectors."""

from __future__ import annotations

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors.report import (
    detect_unsupported_completion_claim,
    detect_verification_skip,
)

from tests.detectors.conftest import step


# ---------------------------------------------------------------------------
# verification-skip
# ---------------------------------------------------------------------------

def test_verification_skip_fires_when_tail_has_no_validation_after_write() -> None:
    ctx = DetectorContext()
    steps = [
        step(0, "FILE_WRITE", target="a.py", tool="Edit"),
        step(1, "REASON", args={"text": "done"}),
        step(2, "REASON", args={"text": "handing off"}),
    ]
    hits = detect_verification_skip(steps, ctx)
    assert len(hits) == 1


def test_verification_skip_not_fires_with_validation_after_write() -> None:
    ctx = DetectorContext()
    steps = [
        step(0, "FILE_WRITE", target="a.py", tool="Edit"),
        step(1, "COMMAND", target="pytest", tool="Bash"),
    ]
    assert detect_verification_skip(steps, ctx) == []


def test_verification_skip_fires_when_validation_outside_tail_window() -> None:
    """Regression: paper restricts the check to the final N steps. A validation
    that ran far before the tail window must not credit the tail."""
    # Last write at index 1; validation at index 2; then 10 REASON steps so
    # the tail window (last 5) no longer contains the validation.
    ctx = DetectorContext()
    steps = [
        step(0, "FILE_WRITE", target="a.py", tool="Edit"),
        step(1, "FILE_WRITE", target="a.py", tool="Edit"),
        step(2, "COMMAND", target="pytest", tool="Bash"),
    ] + [step(i, "REASON") for i in range(3, 13)]
    hits = detect_verification_skip(steps, ctx)
    assert len(hits) == 1


# ---------------------------------------------------------------------------
# unsupported-completion-claim
# ---------------------------------------------------------------------------

def test_unsupported_completion_claim_fires_on_done_cue_without_passing_validation() -> None:
    ctx = DetectorContext()
    steps = [
        step(0, "FILE_WRITE", target="a.py", tool="Edit"),
        step(1, "COMMAND", target="pytest", tool="Bash", effect_label="failed"),
        step(2, "REASON", args={"text": "I have fixed the issue."}),
    ]
    hits = detect_unsupported_completion_claim(steps, ctx)
    assert len(hits) == 1
    assert hits[0].evidence["cue"] == "fixed"


def test_unsupported_completion_claim_not_fires_when_validation_passed() -> None:
    ctx = DetectorContext()
    steps = [
        step(0, "FILE_WRITE", target="a.py", tool="Edit"),
        step(1, "COMMAND", target="pytest", tool="Bash", effect_label="survived"),
        step(2, "REASON", args={"text": "I have fixed the issue."}),
    ]
    assert detect_unsupported_completion_claim(steps, ctx) == []
