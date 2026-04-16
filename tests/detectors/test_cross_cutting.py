"""Tests for cross-cutting detectors."""

from __future__ import annotations

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors.cross_cutting import (
    detect_redundant_search,
    detect_shell_over_tool,
    detect_tool_oscillation,
)

from tests.detectors.conftest import step


# ---------------------------------------------------------------------------
# redundant-search
# ---------------------------------------------------------------------------

def test_redundant_search_fires_on_duplicate_query_in_window() -> None:
    ctx = DetectorContext()
    steps = [
        step(0, "SEARCH", target="foo", tool="Grep", args={"pattern": "foo"}),
        step(1, "SEARCH", target="bar", tool="Grep", args={"pattern": "bar"}),
        step(2, "SEARCH", target="foo", tool="Grep", args={"pattern": "foo"}),
    ]
    hits = detect_redundant_search(steps, ctx)
    assert len(hits) == 1
    assert hits[0].evidence["query"] == "foo"


def test_redundant_search_not_fires_on_unique_queries() -> None:
    ctx = DetectorContext()
    steps = [
        step(i, "SEARCH", target=q, tool="Grep", args={"pattern": q})
        for i, q in enumerate(["a", "b", "c"])
    ]
    assert detect_redundant_search(steps, ctx) == []


# ---------------------------------------------------------------------------
# shell-over-tool
# ---------------------------------------------------------------------------

def test_shell_over_tool_fires_on_cat_when_read_available() -> None:
    ctx = DetectorContext(tool_exposure=frozenset({"Bash", "Read"}))
    steps = [step(0, "COMMAND", target="cat a.py", tool="Bash")]
    hits = detect_shell_over_tool(steps, ctx)
    assert len(hits) == 1


def test_shell_over_tool_not_fires_without_structured_read_exposure() -> None:
    ctx = DetectorContext(tool_exposure=frozenset({"Bash"}))
    steps = [step(0, "COMMAND", target="cat a.py", tool="Bash")]
    assert detect_shell_over_tool(steps, ctx) == []


# ---------------------------------------------------------------------------
# tool-oscillation
# ---------------------------------------------------------------------------

def test_tool_oscillation_fires_on_two_rwr_cycles() -> None:
    ctx = DetectorContext()
    steps = [
        step(0, "FILE_READ", target="a.py", tool="Read"),
        step(1, "FILE_WRITE", target="a.py", tool="Edit"),
        step(2, "FILE_READ", target="a.py", tool="Read"),
        step(3, "FILE_WRITE", target="a.py", tool="Edit"),
        step(4, "FILE_READ", target="a.py", tool="Read"),
    ]
    hits = detect_tool_oscillation(steps, ctx)
    assert len(hits) == 1
    assert hits[0].evidence["cycles"] >= 2


def test_tool_oscillation_not_fires_on_single_cycle() -> None:
    ctx = DetectorContext()
    steps = [
        step(0, "FILE_READ", target="a.py", tool="Read"),
        step(1, "FILE_WRITE", target="a.py", tool="Edit"),
        step(2, "FILE_READ", target="a.py", tool="Read"),
    ]
    assert detect_tool_oscillation(steps, ctx) == []
