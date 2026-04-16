"""Tests for the comparison bridge module and CSS integration."""

import json
import os
import tempfile

import plotly.graph_objects as go
import pytest


class TestRunComparison:
    """Tests for the comparison bridge module."""

    def _make_trajectory(self, n_steps=5):
        """Create a minimal trajectory dict."""
        messages = []
        for i in range(n_steps):
            role = "user" if i % 2 == 0 else "assistant"
            msg = {
                "role": role,
                "content": f"Step {i} content",
                "tokens": {
                    "total": 100, "input": 60, "output": 30,
                    "reasoning": 10, "cache": {"read": 5},
                },
            }
            if role == "assistant":
                msg["finish"] = "stop" if i == n_steps - 1 else None
            messages.append(msg)
        return {"messages": messages}

    def _write_trajectory(self, traj_dict):
        """Write trajectory to a temp file and return path."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False)
        json.dump(traj_dict, tmp)
        tmp.close()
        return tmp.name

    def test_valid_comparison_returns_all_keys(self):
        from trajectory_visualizer.insight.comparison import run_comparison

        ref_raw = self._make_trajectory(6)
        cmp_raw = self._make_trajectory(6)

        result = run_comparison(ref_raw, cmp_raw)
        assert "report_html" in result
        assert "milestone_fig" in result
        assert "segment_fig" in result
        assert "waterfall_fig" in result
        assert "anchor_fig" in result
        assert isinstance(result["milestone_fig"], go.Figure)
        assert isinstance(result["report_html"], str)
        assert len(result["report_html"]) > 0

    def test_error_handling_returns_error_html(self):
        from trajectory_visualizer.insight.comparison import run_comparison

        # Pass an error-marked ref to trigger an error path
        result = run_comparison({"_error": "bad file"}, self._make_trajectory())
        assert "report_html" in result
        assert isinstance(result["report_html"], str)

    def test_dark_mode_flag(self):
        from trajectory_visualizer.insight.comparison import run_comparison

        ref_raw = self._make_trajectory(4)
        cmp_raw = self._make_trajectory(4)

        result = run_comparison(ref_raw, cmp_raw, dark=True)
        assert isinstance(result["milestone_fig"], go.Figure)


class TestConvergeCSSIntegration:
    """Verify Converge CSS is present in Insight's APP_CSS."""

    def test_converge_css_included(self):
        from trajectory_visualizer.insight.styles import APP_CSS
        # Converge uses .cvg- prefixed classes
        assert ".cvg-" in APP_CSS, "Converge CSS classes not found in Insight APP_CSS"

    def test_cvg_report_class_present(self):
        from trajectory_visualizer.insight.styles import APP_CSS
        assert ".cvg-report" in APP_CSS

    def test_cvg_metric_grid_present(self):
        from trajectory_visualizer.insight.styles import APP_CSS
        assert ".cvg-metric-grid" in APP_CSS


class TestStateRawPopulated:
    """Verify state_raw is in the output list of do_load."""

    def test_state_raw_in_source(self):
        """Verify that insight.py source references state_raw in all_outputs."""
        source = open("trajectory_visualizer/insight/insight.py").read()
        assert "state_raw = gr.State" in source, "state_raw State not declared in insight.py"
        assert "state_raw," in source, "state_raw not in outputs list"
        # Verify it appears in both the return tuple and all_outputs
        assert "# state_raw" in source, "state_raw not commented in return tuple"
