"""Tests for intake-phase detectors."""

from __future__ import annotations

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors.intake import (
    detect_memory_bypass,
    detect_premature_code_action,
)

from tests.detectors.conftest import step


# ---------------------------------------------------------------------------
# memory-bypass
# ---------------------------------------------------------------------------

def test_memory_bypass_fires_when_memory_file_unread_before_first_write() -> None:
    ctx = DetectorContext(workspace_files=frozenset({"CLAUDE.md", "src/a.py"}))
    steps = [
        step(0, "FILE_READ", target="src/a.py", tool="Read"),
        step(1, "FILE_WRITE", target="src/a.py", tool="Edit"),
    ]
    hits = detect_memory_bypass(steps, ctx)
    assert len(hits) == 1
    assert hits[0].detector_id == "memory-bypass"
    assert "CLAUDE.md" in hits[0].evidence["memory_files_present"]


def test_memory_bypass_does_not_fire_when_memory_read_first() -> None:
    ctx = DetectorContext(workspace_files=frozenset({"CLAUDE.md", "src/a.py"}))
    steps = [
        step(0, "FILE_READ", target="CLAUDE.md", tool="Read"),
        step(1, "FILE_WRITE", target="src/a.py", tool="Edit"),
    ]
    assert detect_memory_bypass(steps, ctx) == []


def test_memory_bypass_not_fires_without_memory_file() -> None:
    ctx = DetectorContext(workspace_files=frozenset({"src/a.py"}))
    steps = [step(0, "FILE_WRITE", target="src/a.py", tool="Edit")]
    assert detect_memory_bypass(steps, ctx) == []


def test_memory_bypass_write_to_memory_file_is_not_a_code_action() -> None:
    """Regression: writing CLAUDE.md itself does not count as 'first code action'."""
    ctx = DetectorContext(workspace_files=frozenset({"CLAUDE.md", "src/a.py"}))
    steps = [
        step(0, "FILE_WRITE", target="CLAUDE.md", tool="Edit"),
        step(1, "FILE_READ", target="CLAUDE.md", tool="Read"),
        step(2, "FILE_WRITE", target="src/a.py", tool="Edit"),
    ]
    # Memory file was read before the first source write at step 2.
    assert detect_memory_bypass(steps, ctx) == []


# ---------------------------------------------------------------------------
# premature-code-action
# ---------------------------------------------------------------------------

def test_premature_code_action_fires_when_write_first() -> None:
    ctx = DetectorContext()
    steps = [step(0, "FILE_WRITE", target="src/a.py", tool="Edit")]
    hits = detect_premature_code_action(steps, ctx)
    assert len(hits) == 1
    assert hits[0].evidence["first_write_step"] == 0


def test_premature_code_action_not_fires_when_read_first() -> None:
    ctx = DetectorContext()
    steps = [
        step(0, "FILE_READ", target="src/a.py", tool="Read"),
        step(1, "FILE_WRITE", target="src/a.py", tool="Edit"),
    ]
    assert detect_premature_code_action(steps, ctx) == []
