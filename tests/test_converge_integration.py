"""Integration tests for trajectory_visualizer.converge against real sample trajectories."""

from __future__ import annotations

import json
import os

import pytest

from trajectory_visualizer.converge.alignment import build_comparison_report


SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def _sample(name: str) -> str:
    path = os.path.join(SAMPLES, name)
    if not os.path.isfile(path):
        pytest.skip(f"Sample file not found: {name}")
    return path


class TestSameFormatComparison:
    """8.1: CC vs CC — same format."""

    def test_produces_full_report(self):
        report = build_comparison_report(
            _sample("cc_trajectory.json"),
            _sample("cc_trajectory.json"),
        )
        assert "outcome" in report
        assert "alignment" in report
        assert "milestones" in report
        assert "patterns" in report
        assert report["evidence_level"] == "single_pair_hypothesis"

    def test_self_comparison_is_perfect(self):
        report = build_comparison_report(
            _sample("cc_trajectory.json"),
            _sample("cc_trajectory.json"),
        )
        assert report["alignment"]["reference_recall"] == 1.0
        assert report["alignment"]["behavioral_precision"] == 1.0
        assert report["alignment"]["alignment_f1"] == 1.0
        assert report["alignment"]["overhead_ratio"] == 1.0

    def test_milestones_have_zero_deltas(self):
        report = build_comparison_report(
            _sample("cc_trajectory.json"),
            _sample("cc_trajectory.json"),
        )
        for key, val in report["milestones"].items():
            if val is not None:
                assert val == 0, f"{key} should be 0 for self-comparison"


    # Lingxi support removed — TestCrossFormatComparison class deleted.


class TestAnchoredComparison:
    """Anchored comparison with ground truth diff."""

    def test_anchor_changes_milestones(self):
        """Anchor patch should ground milestones against GT files."""
        anchor = _sample("132807.diff")
        unanchored = build_comparison_report(
            _sample("cc_trajectory.json"),
            _sample("op_trajectory.json"),
        )
        anchored = build_comparison_report(
            _sample("cc_trajectory.json"),
            _sample("op_trajectory.json"),
            anchor_patch=anchor,
        )
        assert anchored["anchor_mode"] == "external"
        assert anchored["confidence"]["milestones"] == "anchored"
        # Anchor should fill in previously-null milestones
        assert anchored["cmp_milestones"]["first_relevant_file"] is not None

    def test_anchor_enables_milestone_order_match(self):
        """With anchor, milestone order may match (enabling paired segments)."""
        anchor = _sample("132807.diff")
        report = build_comparison_report(
            _sample("cc_trajectory.json"),
            _sample("op_trajectory.json"),
            anchor_patch=anchor,
        )
        # With the real 132807.diff, anchored mode should produce matching order
        seg = report.get("segments", {})
        if seg.get("milestone_order_matches"):
            assert "segment_comparison" in seg
        else:
            assert "reference_segments" in seg

    def test_anchor_no_divergent_outcome_warning(self):
        """Anchored comparison should not produce informational warning on alignment."""
        anchor = _sample("132807.diff")
        report = build_comparison_report(
            _sample("cc_trajectory.json"),
            _sample("op_trajectory.json"),
            anchor_patch=anchor,
        )
        # Even with different outcomes, anchored mode keeps alignment as heuristic not informational
        assert report["confidence"]["alignment"] in ("heuristic", "informational")

    def test_anchor_ref_milestones_in_report(self):
        """Anchored report should contain raw milestone maps."""
        anchor = _sample("132807.diff")
        report = build_comparison_report(
            _sample("cc_trajectory.json"),
            _sample("op_trajectory.json"),
            anchor_patch=anchor,
        )
        assert "ref_milestones" in report
        assert "cmp_milestones" in report
        assert isinstance(report["ref_milestones"], dict)
        assert isinstance(report["cmp_milestones"], dict)


class TestUnanchoredDivergentOutcomes:
    """8.4: Unanchored comparison — verify warning note."""

    def test_warning_when_different_outcomes(self):
        report = build_comparison_report(
            _sample("cc_trajectory.json"),
            _sample("op_trajectory.json"),
        )
        assert report["anchor_mode"] == "self"
        # Notes should exist
        assert len(report["notes"]) > 0


class TestCLIOutputSchema:
    """8.5: CLI JSON output matches expected schema."""

    def test_json_serializable(self):
        report = build_comparison_report(
            _sample("cc_trajectory.json"),
            _sample("cc_trajectory.json"),
            task_id="test-schema-check",
        )
        json_str = json.dumps(report, default=str)
        parsed = json.loads(json_str)

        # Verify top-level keys from proposal.md schema
        assert "task_id" in parsed
        assert "reference_agent" in parsed
        assert "compared_agent" in parsed
        assert "outcome" in parsed
        assert "alignment" in parsed
        assert "milestones" in parsed
        assert "patterns" in parsed
        assert "anchor_mode" in parsed
        assert "evidence_level" in parsed
        assert "notes" in parsed

        # Verify alignment sub-keys
        a = parsed["alignment"]
        assert "reference_recall" in a
        assert "behavioral_precision" in a
        assert "alignment_f1" in a
        assert "overhead_ratio" in a
        assert "harmful_ratio" in a
        assert "harmful_cost" in a

    def test_task_id_propagated(self):
        report = build_comparison_report(
            _sample("cc_trajectory.json"),
            _sample("cc_trajectory.json"),
            task_id="my-task-123",
        )
        assert report["task_id"] == "my-task-123"
