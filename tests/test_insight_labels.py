"""Unit tests for trajectory_visualizer.insight.labels — label loading and aggregation."""

from __future__ import annotations

import os

import pytest

from trajectory_visualizer.insight.labels import aggregate_labels, load_labeled_json

_SAMPLE_LABELED = os.path.join(os.path.dirname(__file__), "..", "samples", "cc_trajectory_labeled.json")


# ---------------------------------------------------------------------------
# load_labeled_json
# ---------------------------------------------------------------------------


class TestLoadLabeledJson:
    @pytest.mark.skipif(not os.path.isfile(_SAMPLE_LABELED), reason="Labeled sample not present")
    def test_load_valid_file(self):
        data = load_labeled_json(_SAMPLE_LABELED)
        assert isinstance(data, dict)
        assert "steps" in data

    def test_load_nonexistent_file(self):
        with pytest.raises((FileNotFoundError, Exception)):
            load_labeled_json("/nonexistent/labeled.json")

    def test_load_invalid_json(self, tmp_path):
        bad = tmp_path / "bad_labeled.json"
        bad.write_text("not json")
        with pytest.raises(Exception):
            load_labeled_json(str(bad))


# ---------------------------------------------------------------------------
# aggregate_labels
# ---------------------------------------------------------------------------


class TestAggregateLabels:
    @pytest.mark.skipif(not os.path.isfile(_SAMPLE_LABELED), reason="Labeled sample not present")
    def test_aggregation_structure(self):
        data = load_labeled_json(_SAMPLE_LABELED)
        agg = aggregate_labels(data)
        assert "phase_counts" in agg
        assert "action_counts" in agg
        assert "steps" in agg
        assert isinstance(agg["phase_counts"], dict)
        assert isinstance(agg["action_counts"], dict)

    @pytest.mark.skipif(not os.path.isfile(_SAMPLE_LABELED), reason="Labeled sample not present")
    def test_step_count_at_least_labeled(self):
        data = load_labeled_json(_SAMPLE_LABELED)
        agg = aggregate_labels(data)
        # Aggregated steps may include user steps not in labeled data
        assert len(agg["steps"]) >= len(data["steps"])

    @pytest.mark.skipif(not os.path.isfile(_SAMPLE_LABELED), reason="Labeled sample not present")
    def test_phase_counts_sum(self):
        data = load_labeled_json(_SAMPLE_LABELED)
        agg = aggregate_labels(data)
        classified = sum(agg["phase_counts"].values())
        # Classified + unknown should equal total labeled steps
        assert classified + agg["unknown"] == len(data["steps"])

    def test_aggregate_minimal(self):
        data = {
            "steps": [
                {"index": 0, "role": "assistant", "phase": "implement", "action": "code_writing"},
                {"index": 1, "role": "assistant", "phase": "debug", "action": "error_analysis"},
            ]
        }
        agg = aggregate_labels(data)
        assert agg["phase_counts"]["implement"] == 1
        assert agg["phase_counts"]["debug"] == 1
        assert len(agg["steps"]) == 2
