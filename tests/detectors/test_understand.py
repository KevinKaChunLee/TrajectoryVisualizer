"""Tests for understand-phase detectors."""

from __future__ import annotations

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors.understand import (
    detect_empty_result_churn,
    detect_re_read_churn,
    detect_search_loop,
)

from tests.detectors.conftest import step


def _search(i: int, query: str, matches: list[str] | None = None, no_matches: bool = False):
    args = {"pattern": query}
    if no_matches:
        args["output"] = ""
    elif matches is not None:
        args["matches"] = matches
    return step(i, "SEARCH", target=query, tool="Grep", args=args)


# ---------------------------------------------------------------------------
# empty-result-churn
# ---------------------------------------------------------------------------

def test_empty_result_churn_fires_on_three_consecutive_empty_searches() -> None:
    ctx = DetectorContext()
    steps = [
        _search(0, "foo", no_matches=True),
        _search(1, "bar", no_matches=True),
        _search(2, "baz", no_matches=True),
    ]
    hits = detect_empty_result_churn(steps, ctx)
    assert len(hits) == 1
    assert hits[0].evidence["empty_searches"] == 3


def test_empty_result_churn_not_fires_on_two() -> None:
    ctx = DetectorContext()
    steps = [_search(0, "foo", no_matches=True), _search(1, "bar", no_matches=True)]
    assert detect_empty_result_churn(steps, ctx) == []


# ---------------------------------------------------------------------------
# search-loop
# ---------------------------------------------------------------------------

def test_search_loop_fires_on_four_consecutive_reads_and_searches() -> None:
    ctx = DetectorContext()
    steps = [
        _search(0, "foo"),
        step(1, "FILE_READ", target="a.py", tool="Read"),
        _search(2, "bar"),
        step(3, "FILE_READ", target="b.py", tool="Read"),
    ]
    hits = detect_search_loop(steps, ctx)
    assert len(hits) == 1
    assert hits[0].evidence["consecutive_read_search_steps"] == 4


def test_search_loop_broken_by_write() -> None:
    ctx = DetectorContext()
    steps = [
        _search(0, "foo"),
        step(1, "FILE_READ", target="a.py", tool="Read"),
        step(2, "FILE_WRITE", target="a.py", tool="Edit"),
        _search(3, "bar"),
    ]
    assert detect_search_loop(steps, ctx) == []


# ---------------------------------------------------------------------------
# re-read-churn
# ---------------------------------------------------------------------------

def test_re_read_churn_fires_on_three_reads_same_file_no_write() -> None:
    ctx = DetectorContext()
    steps = [
        step(0, "FILE_READ", target="a.py", tool="Read"),
        step(1, "FILE_READ", target="a.py", tool="Read"),
        step(2, "FILE_READ", target="a.py", tool="Read"),
    ]
    hits = detect_re_read_churn(steps, ctx)
    assert len(hits) == 1
    assert hits[0].evidence["file"] == "a.py"


def test_re_read_churn_not_fires_when_intervening_write() -> None:
    ctx = DetectorContext()
    steps = [
        step(0, "FILE_READ", target="a.py", tool="Read"),
        step(1, "FILE_READ", target="a.py", tool="Read"),
        step(2, "FILE_WRITE", target="a.py", tool="Edit"),
        step(3, "FILE_READ", target="a.py", tool="Read"),
    ]
    assert detect_re_read_churn(steps, ctx) == []
