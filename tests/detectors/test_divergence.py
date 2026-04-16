"""Tests for cross-trajectory divergence detectors."""

from __future__ import annotations

from trajectory_visualizer.converge.detectors import (
    dead_end_exploration,
    iterative_refinement,
    off_anchor_exploration,
    ordering_inefficiency,
    rapid_rewrite,
    scope_drift,
)
from trajectory_visualizer.core.detection import DetectorContext

from tests.detectors.conftest import step


def _w(i: int, tgt: str, effect: str = "survived"):
    return step(i, "FILE_WRITE", target=tgt, tool="Edit", effect_label=effect)


def _r(i: int, tgt: str):
    return step(i, "FILE_READ", target=tgt, tool="Read")


# ---------------------------------------------------------------------------
# rapid-rewrite
# ---------------------------------------------------------------------------

def test_rapid_rewrite_fires_when_reference_did_not_rewrite() -> None:
    ctx = DetectorContext()
    compared = [_w(0, "a.py", effect="reverted"), _w(1, "a.py")]
    reference = [_w(0, "a.py")]
    hits = rapid_rewrite.detect(compared, reference, ctx)
    assert len(hits) == 1
    assert hits[0].tier == "high"


def test_rapid_rewrite_not_fires_when_reference_also_rewrote() -> None:
    ctx = DetectorContext()
    compared = [_w(0, "a.py"), _w(1, "a.py")]
    reference = [_w(0, "a.py"), _w(1, "a.py")]
    assert rapid_rewrite.detect(compared, reference, ctx) == []


# ---------------------------------------------------------------------------
# scope-drift
# ---------------------------------------------------------------------------

def test_scope_drift_fires_on_write_outside_reference_set() -> None:
    ctx = DetectorContext()
    compared = [_w(0, "a.py"), _w(1, "extra.py")]
    reference = [_w(0, "a.py")]
    hits = scope_drift.detect(compared, reference, ctx)
    assert any(h.evidence["file"] == "extra.py" for h in hits)


def test_scope_drift_not_fires_when_all_writes_in_anchor() -> None:
    ctx = DetectorContext()
    compared = [_w(0, "a.py"), _w(1, "b.py")]
    reference = [_w(0, "a.py"), _w(1, "b.py")]
    assert scope_drift.detect(compared, reference, ctx) == []


# ---------------------------------------------------------------------------
# off-anchor-exploration
# ---------------------------------------------------------------------------

def test_off_anchor_exploration_fires_on_high_off_anchor_ratio() -> None:
    ctx = DetectorContext(anchor_set=frozenset({"a.py"}))
    compared = [_r(0, "a.py"), _r(1, "x.py"), _r(2, "y.py"), _r(3, "z.py")]
    reference: list = []
    hits = off_anchor_exploration.detect(compared, reference, ctx)
    assert len(hits) == 1
    assert hits[0].evidence["off_anchor_count"] == 3


def test_off_anchor_exploration_not_fires_on_focused_reads() -> None:
    ctx = DetectorContext(anchor_set=frozenset({"a.py", "b.py"}))
    compared = [_r(0, "a.py"), _r(1, "b.py"), _r(2, "a.py")]
    reference: list = []
    assert off_anchor_exploration.detect(compared, reference, ctx) == []


# ---------------------------------------------------------------------------
# dead-end-exploration
# ---------------------------------------------------------------------------

def test_dead_end_exploration_fires_when_files_never_written() -> None:
    ctx = DetectorContext()
    compared = [_r(0, "x.py"), _r(1, "y.py"), _r(2, "z.py")]
    reference = [_w(0, "patch.py")]
    hits = dead_end_exploration.detect(compared, reference, ctx)
    assert len(hits) == 1


def test_dead_end_not_fires_when_later_write_to_same_file() -> None:
    ctx = DetectorContext()
    compared = [_r(0, "x.py"), _w(1, "x.py")]
    reference = [_w(0, "patch.py")]
    # x.py is written later in compared, so reading it is not a dead-end.
    assert dead_end_exploration.detect(compared, reference, ctx) == []


# ---------------------------------------------------------------------------
# ordering-inefficiency
# ---------------------------------------------------------------------------

def test_ordering_inefficiency_fires_on_inverted_order() -> None:
    ctx = DetectorContext()
    reference = [_w(0, "a.py"), _w(1, "b.py"), _w(2, "c.py"), _w(3, "d.py")]
    compared = [_w(0, "d.py"), _w(1, "c.py"), _w(2, "b.py"), _w(3, "a.py")]
    hits = ordering_inefficiency.detect(compared, reference, ctx)
    assert len(hits) == 1


def test_ordering_inefficiency_not_fires_on_same_order() -> None:
    ctx = DetectorContext()
    reference = [_w(0, "a.py"), _w(1, "b.py"), _w(2, "c.py")]
    compared = [_w(0, "a.py"), _w(1, "b.py"), _w(2, "c.py")]
    assert ordering_inefficiency.detect(compared, reference, ctx) == []


# ---------------------------------------------------------------------------
# iterative-refinement
# ---------------------------------------------------------------------------

def test_iterative_refinement_fires_on_distant_rewrite() -> None:
    ctx = DetectorContext()
    compared = [_w(0, "a.py")] + [step(i, "FILE_READ", target="b.py", tool="Read") for i in range(1, 6)] + [_w(6, "a.py")]
    hits = iterative_refinement.detect(compared, [], ctx)
    assert len(hits) == 1
    assert hits[0].tier == "low"


def test_iterative_refinement_not_fires_on_adjacent_rewrite() -> None:
    ctx = DetectorContext()
    compared = [_w(0, "a.py"), _w(1, "a.py")]
    assert iterative_refinement.detect(compared, [], ctx) == []
