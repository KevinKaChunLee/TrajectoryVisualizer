"""Tests for new chart builders: tool heatmap, burn-down, error recovery, phase radar."""

import plotly.graph_objects as go
import pytest

from trajectory_visualizer.insight.charts import (
    build_tool_frequency_heatmap,
    build_token_burndown_chart,
    build_error_recovery_chart,
    build_phase_radar_chart,
)


def _make_step(role="assistant", tool_calls=None, error_count=0,
               input_tokens=100, output_tokens=50, cache_read_tokens=20,
               reasoning_tokens=10, tool_call_count=0, duration=1.0):
    """Build a minimal step dict for chart tests (matches parser output structure)."""
    total = input_tokens + output_tokens + cache_read_tokens + reasoning_tokens
    return {
        "role": role,
        "tool_calls": tool_calls or [],
        "error_count": error_count,
        "tokens": {
            "total": total,
            "input": input_tokens,
            "output": output_tokens,
            "cache_read": cache_read_tokens,
            "reasoning": reasoning_tokens,
        },
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "reasoning_tokens": reasoning_tokens,
        "tool_call_count": tool_call_count,
        "duration": duration,
    }


class TestToolFrequencyHeatmap:
    def test_returns_figure(self):
        steps = [
            _make_step(tool_calls=[{"tool_name": "read_file"}, {"tool_name": "write_file"}]),
            _make_step(tool_calls=[{"tool_name": "read_file"}]),
        ]
        fig = build_tool_frequency_heatmap(steps)
        assert isinstance(fig, go.Figure)

    def test_empty_steps(self):
        fig = build_tool_frequency_heatmap([])
        assert isinstance(fig, go.Figure)

    def test_no_tool_calls(self):
        steps = [_make_step(), _make_step()]
        fig = build_tool_frequency_heatmap(steps)
        assert isinstance(fig, go.Figure)

    def test_dark_mode(self):
        steps = [_make_step(tool_calls=[{"tool_name": "bash"}])]
        fig = build_tool_frequency_heatmap(steps, dark=True)
        assert isinstance(fig, go.Figure)


class TestTokenBurndownChart:
    def test_returns_figure(self):
        steps = [_make_step() for _ in range(5)]
        fig = build_token_burndown_chart(steps)
        assert isinstance(fig, go.Figure)

    def test_empty_steps(self):
        fig = build_token_burndown_chart([])
        assert isinstance(fig, go.Figure)

    def test_dark_mode(self):
        steps = [_make_step() for _ in range(3)]
        fig = build_token_burndown_chart(steps, dark=True)
        assert isinstance(fig, go.Figure)


class TestErrorRecoveryChart:
    def test_with_errors_and_recovery(self):
        steps = [
            _make_step(error_count=1),
            _make_step(error_count=0, tool_calls=[{"tool_name": "bash"}], tool_call_count=1),
            _make_step(error_count=0),
        ]
        fig = build_error_recovery_chart(steps)
        assert isinstance(fig, go.Figure)

    def test_no_errors(self):
        steps = [_make_step() for _ in range(3)]
        fig = build_error_recovery_chart(steps)
        assert isinstance(fig, go.Figure)

    def test_empty_steps(self):
        fig = build_error_recovery_chart([])
        assert isinstance(fig, go.Figure)


class TestPhaseRadarChart:
    def test_multiple_phases(self):
        steps = [_make_step() for _ in range(10)]
        phases = [
            {"name": "Boot", "start_idx": 0, "end_idx": 3},
            {"name": "Steady", "start_idx": 4, "end_idx": 7},
            {"name": "Closeout", "start_idx": 8, "end_idx": 9},
        ]
        fig = build_phase_radar_chart(steps, phases)
        assert isinstance(fig, go.Figure)

    def test_single_phase_falls_back_to_bar(self):
        steps = [_make_step() for _ in range(5)]
        phases = [{"name": "Full Run", "start_idx": 0, "end_idx": 4}]
        fig = build_phase_radar_chart(steps, phases)
        assert isinstance(fig, go.Figure)

    def test_empty_phases(self):
        steps = [_make_step()]
        fig = build_phase_radar_chart(steps, [])
        assert isinstance(fig, go.Figure)

    def test_dark_mode(self):
        steps = [_make_step() for _ in range(5)]
        phases = [{"name": "Boot", "start_idx": 0, "end_idx": 4}]
        fig = build_phase_radar_chart(steps, phases, dark=True)
        assert isinstance(fig, go.Figure)
