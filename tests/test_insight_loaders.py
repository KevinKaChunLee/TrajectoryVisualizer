"""Unit tests for trajectory_visualizer.insight.loaders — format detection and trajectory loading."""

from __future__ import annotations

import os

import pytest

from trajectory_visualizer.insight.loaders import detect_format, load_trajectory, safe_get


# ---------------------------------------------------------------------------
# safe_get
# ---------------------------------------------------------------------------


class TestSafeGet:
    def test_single_key(self):
        assert safe_get({"a": 1}, "a") == 1

    def test_nested_keys(self):
        assert safe_get({"a": {"b": {"c": 3}}}, "a", "b", "c") == 3

    def test_missing_key_returns_default(self):
        assert safe_get({"a": 1}, "x", default="nope") == "nope"

    def test_none_dict_returns_default(self):
        assert safe_get(None, "a", default=0) == 0

    def test_empty_dict(self):
        assert safe_get({}, "a", "b", default=-1) == -1


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------


class TestDetectFormat:
    def test_ccsession_format(self):
        raw = {"format": "ccsession-trajectory"}
        assert detect_format(raw) == "ccsession"

    def test_opencode_format(self):
        raw = {"info": {}, "messages": []}
        assert detect_format(raw) == "opencode"

    def test_unknown_format(self):
        raw = {"something": "else"}
        fmt = detect_format(raw)
        assert fmt == "unknown"

    def test_codearts_format(self):
        raw = {"format": "codearts"}
        assert detect_format(raw) == "codearts"

    def test_codearts_does_not_match_opencode(self):
        """CodeArts format should not be confused with OpenCode."""
        raw = {"format": "codearts", "info": {}, "messages": []}
        assert detect_format(raw) == "codearts"

    def test_lingxi_format_no_longer_detected(self):
        raw = {"format": "lingxi"}
        assert detect_format(raw) == "unknown"

    def test_lingxi_keys_no_longer_detected(self):
        raw = {"workflow_id": "Lingxi Coder Workflow", "run_id": "abc-123"}
        assert detect_format(raw) == "unknown"

    def test_empty_dict(self):
        fmt = detect_format({})
        assert isinstance(fmt, str)


# ---------------------------------------------------------------------------
# load_trajectory
# ---------------------------------------------------------------------------

_SAMPLE_CC = os.path.join(os.path.dirname(__file__), "..", "samples", "cc_trajectory.json")
_SAMPLE_CC_LABELED = os.path.join(os.path.dirname(__file__), "..", "samples", "cc_trajectory_labeled.json")
_SAMPLE_CA = os.path.join(os.path.dirname(__file__), "..", "samples", "ca_trajectory.json")


class TestLoadTrajectory:
    @pytest.mark.skipif(not os.path.isfile(_SAMPLE_CC), reason="Large sample file not present")
    def test_load_cc_trajectory(self):
        result = load_trajectory(_SAMPLE_CC)
        assert isinstance(result, dict)
        assert "_error" not in result
        assert "trajectory" in result or "messages" in result

    def test_load_nonexistent_file(self):
        result = load_trajectory("/nonexistent/path/to/file.json")
        assert isinstance(result, dict)
        assert "_error" in result

    def test_load_invalid_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json at all")
        result = load_trajectory(str(bad_file))
        assert "_error" in result

    @pytest.mark.skipif(not os.path.isfile(_SAMPLE_CC_LABELED), reason="Labeled sample not present")
    def test_load_labeled_as_trajectory(self):
        """Labeled JSON has a different structure — loader should handle it gracefully."""
        result = load_trajectory(_SAMPLE_CC_LABELED)
        assert isinstance(result, dict)

    # -- CodeArts format tests --

    @pytest.mark.skipif(not os.path.isfile(_SAMPLE_CA), reason="CodeArts sample not present")
    def test_load_codearts_trajectory(self):
        result = load_trajectory(_SAMPLE_CA)
        assert isinstance(result, dict)
        assert "_error" not in result
        assert "trajectory" in result
        assert "metadata" in result
        assert result["metadata"]["generator_name"] == "codearts"

    @pytest.mark.skipif(not os.path.isfile(_SAMPLE_CA), reason="CodeArts sample not present")
    def test_codearts_metadata_fields(self):
        result = load_trajectory(_SAMPLE_CA)
        md = result["metadata"]
        assert md.get("session_id")
        assert md.get("agent")
        assert md.get("model")
        assert "timing" in result
        assert result["timing"].get("total_duration") is not None

    @pytest.mark.skipif(not os.path.isfile(_SAMPLE_CA), reason="CodeArts sample not present")
    def test_codearts_parse_steps(self):
        """Verify parse_steps works on CodeArts loaded trajectory."""
        from trajectory_visualizer.insight.parser import parse_steps
        result = load_trajectory(_SAMPLE_CA)
        steps = parse_steps(result)
        assert len(steps) > 0
        roles = {s["role"] for s in steps}
        assert "user" in roles
        assert "assistant" in roles

    # -- .log files are no longer supported --

    def test_load_log_file_rejected(self, tmp_path):
        """Any .log file should return an error (Lingxi support removed)."""
        log_file = tmp_path / "test.log"
        log_file.write_text("some content")
        result = load_trajectory(str(log_file))
        assert "_error" in result
