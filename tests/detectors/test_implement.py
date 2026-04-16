"""Tests for implement-phase detectors."""

from __future__ import annotations

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors.implement import (
    detect_edit_thrash,
    detect_edit_without_inspection,
)

from tests.detectors.conftest import step


def _read(i: int, tgt: str):
    return step(i, "FILE_READ", target=tgt, tool="Read")


def _write(i: int, tgt: str, effect: str = "survived"):
    return step(i, "FILE_WRITE", target=tgt, tool="Edit", effect_label=effect)


# ---------------------------------------------------------------------------
# edit-without-inspection
# ---------------------------------------------------------------------------

def test_edit_without_inspection_fires_when_no_prior_read() -> None:
    ctx = DetectorContext()
    steps = [_write(0, "a.py")]
    hits = detect_edit_without_inspection(steps, ctx)
    assert len(hits) == 1
    assert hits[0].evidence["file"] == "a.py"


def test_edit_without_inspection_not_fires_when_read_first() -> None:
    ctx = DetectorContext()
    steps = [_read(0, "a.py"), _write(1, "a.py")]
    assert detect_edit_without_inspection(steps, ctx) == []


# ---------------------------------------------------------------------------
# edit-thrash
# ---------------------------------------------------------------------------

def test_edit_thrash_fires_on_three_writes_with_reverted() -> None:
    ctx = DetectorContext()
    steps = [
        _write(0, "a.py", effect="reverted"),
        _write(1, "a.py", effect="survived"),
        _write(2, "a.py", effect="survived"),
    ]
    hits = detect_edit_thrash(steps, ctx)
    assert len(hits) == 1
    assert hits[0].evidence["file"] == "a.py"


def test_edit_thrash_not_fires_without_revert() -> None:
    ctx = DetectorContext()
    steps = [_write(i, "a.py", effect="survived") for i in range(3)]
    assert detect_edit_thrash(steps, ctx) == []
