"""Tests for PatternDetection and DetectorContext."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from trajectory_visualizer.core.detection import DetectorContext, PatternDetection


# ---------------------------------------------------------------------------
# PatternDetection is frozen
# ---------------------------------------------------------------------------

def test_pattern_detection_is_frozen() -> None:
    det = PatternDetection(detector_id="search-loop", span=(2, 6))
    with pytest.raises(FrozenInstanceError):
        det.detector_id = "other"  # type: ignore[misc]


def test_pattern_detection_defaults() -> None:
    det = PatternDetection(detector_id="search-loop", span=(2, 6))
    assert det.evidence == {}
    assert det.tier is None


# ---------------------------------------------------------------------------
# DetectorContext.thresholds_for — catalog defaults + overrides
# ---------------------------------------------------------------------------

def test_thresholds_for_uses_catalog_defaults() -> None:
    ctx = DetectorContext()
    t = ctx.thresholds_for("search-loop")
    assert t["min_consecutive_steps"] == 4


def test_thresholds_for_applies_overrides() -> None:
    ctx = DetectorContext(
        threshold_overrides={"search-loop": {"min_consecutive_steps": 3}}
    )
    t = ctx.thresholds_for("search-loop")
    assert t["min_consecutive_steps"] == 3


def test_thresholds_for_partial_override_preserves_other_defaults() -> None:
    ctx = DetectorContext(
        threshold_overrides={"edit-thrash": {"min_writes": 5}}
    )
    t = ctx.thresholds_for("edit-thrash")
    assert t["min_writes"] == 5
    assert t["window_steps"] == 10  # unchanged default


def test_thresholds_for_returns_read_only_mapping() -> None:
    ctx = DetectorContext()
    t = ctx.thresholds_for("search-loop")
    with pytest.raises(TypeError):
        t["min_consecutive_steps"] = 99  # type: ignore[index]


# ---------------------------------------------------------------------------
# DetectorContext.load_labels — sidecar fallback
# ---------------------------------------------------------------------------

def test_load_labels_missing_root_returns_empty() -> None:
    assert DetectorContext.load_labels("task-1", None) == {}


def test_load_labels_missing_file_returns_empty(tmp_path: Path) -> None:
    assert DetectorContext.load_labels("task-missing", tmp_path) == {}


def test_load_labels_malformed_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "task-x.json"
    path.write_text("not json")
    assert DetectorContext.load_labels("task-x", tmp_path) == {}


def test_load_labels_parses_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "task-ok.json"
    path.write_text(json.dumps({
        "0": {"phase": "understand", "action": "code_reading"},
        "1": {"phase": "implement", "action": "implement_fix"},
    }))
    labels = DetectorContext.load_labels("task-ok", tmp_path)
    assert labels[0]["phase"] == "understand"
    assert labels[1]["action"] == "implement_fix"


def test_load_labels_ignores_non_integer_keys(tmp_path: Path) -> None:
    path = tmp_path / "task-mixed.json"
    path.write_text(json.dumps({"0": {"phase": "understand"}, "abc": {"phase": "implement"}}))
    labels = DetectorContext.load_labels("task-mixed", tmp_path)
    assert 0 in labels
    assert "abc" not in labels  # non-integer key dropped


# ---------------------------------------------------------------------------
# DetectorContext.gating_satisfied — per-band behavior
# ---------------------------------------------------------------------------

def test_gating_satisfied_for_ungated_detector() -> None:
    ctx = DetectorContext()
    ok, reason = ctx.gating_satisfied("premature-code-action")
    assert ok is True
    assert reason is None


def test_config_gated_requires_memory_file() -> None:
    ctx = DetectorContext(workspace_files=frozenset({"src/a.py"}))
    ok, reason = ctx.gating_satisfied("memory-bypass")
    assert ok is False
    assert "config-gated" in (reason or "")


def test_config_gated_satisfied_with_claude_md() -> None:
    ctx = DetectorContext(workspace_files=frozenset({"CLAUDE.md", "src/a.py"}))
    ok, _ = ctx.gating_satisfied("memory-bypass")
    assert ok is True


def test_tool_gated_requires_planning_tool() -> None:
    ctx = DetectorContext(tool_exposure=frozenset({"Bash", "Read"}))
    ok, reason = ctx.gating_satisfied("plan-stall")
    assert ok is False
    assert "tool-gated" in (reason or "")


def test_tool_gated_satisfied_with_todowrite() -> None:
    ctx = DetectorContext(tool_exposure=frozenset({"TodoWrite", "Bash"}))
    ok, _ = ctx.gating_satisfied("plan-stall")
    assert ok is True


def test_capability_gated_requires_both_shell_and_structured_read() -> None:
    ctx_shell_only = DetectorContext(tool_exposure=frozenset({"Bash"}))
    ok, reason = ctx_shell_only.gating_satisfied("shell-over-tool")
    assert ok is False
    assert "capability-gated" in (reason or "")

    ctx_both = DetectorContext(tool_exposure=frozenset({"Bash", "Read"}))
    ok, _ = ctx_both.gating_satisfied("shell-over-tool")
    assert ok is True


def test_h_detector_requires_labels() -> None:
    ctx_unlabeled = DetectorContext()
    ok, reason = ctx_unlabeled.gating_satisfied("phase-oscillation")
    assert ok is False
    assert "requires_semantic_labels" in (reason or "")

    ctx_labeled = DetectorContext(labels={0: {"phase": "understand"}})
    ok, _ = ctx_labeled.gating_satisfied("phase-oscillation")
    assert ok is True


def test_weaker_gating_is_informational_only() -> None:
    """'weaker' tags should not block firing on their own."""
    # plan-less-execution is tool-gated + weaker. Expose a planning tool.
    ctx = DetectorContext(tool_exposure=frozenset({"TodoWrite"}))
    ok, _ = ctx.gating_satisfied("plan-less-execution")
    assert ok is True
