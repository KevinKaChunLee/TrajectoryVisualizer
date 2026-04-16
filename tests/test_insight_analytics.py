"""Unit tests for trajectory_visualizer.insight.analytics — step analytics, phase detection, insights."""

from __future__ import annotations

import pytest

from trajectory_visualizer.insight.analytics import build_sub_agent_tree, compute_step_analytics, detect_phases, generate_insights


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_step(index=0, role="assistant", total_tok=500, output_tok=50,
               duration=5.0, tool_calls=None, cache_read=400):
    return {
        "index": index,
        "role": role,
        "tokens": {
            "total": total_tok, "input": total_tok - output_tok - cache_read,
            "output": output_tok, "reasoning": 0,
            "cache_read": cache_read, "cache_write": 0,
        },
        "duration": duration,
        "parts": [],
        "tool_calls": tool_calls or [],
        "tool_call_count": len(tool_calls) if tool_calls else 0,
        "error_count": 0,
        "has_reasoning": False,
        "text_preview": "step text",
        "finish": "tool_use",
        "model_id": "m",
        "provider_id": "p",
        "time_created_ms": 1000000 + index * 10000,
        "time_completed_ms": 1000000 + index * 10000 + int(duration * 1000),
        "agent": "",
        "mode": "",
        "message_id": "",
        "id": f"msg_{index}",
        "parent_id": "",
        "session_id": "ses",
        "cwd": "",
        "root": "",
    }


# ---------------------------------------------------------------------------
# compute_step_analytics
# ---------------------------------------------------------------------------


class TestComputeStepAnalytics:
    def test_returns_list(self):
        steps = [_make_step(i) for i in range(5)]
        result = compute_step_analytics(steps)
        assert isinstance(result, list)
        assert len(result) == 5

    def test_analytics_keys(self):
        steps = [_make_step()]
        result = compute_step_analytics(steps)
        row = result[0]
        assert "index" in row
        assert "tok_total" in row
        assert "cache_ratio" in row

    def test_empty_steps(self):
        assert compute_step_analytics([]) == []


# ---------------------------------------------------------------------------
# detect_phases
# ---------------------------------------------------------------------------


class TestDetectPhases:
    def test_returns_phases(self):
        steps = [_make_step(i) for i in range(10)]
        analytics = compute_step_analytics(steps)
        phases = detect_phases(analytics)
        assert isinstance(phases, list)
        for phase in phases:
            assert "name" in phase
            assert "start_idx" in phase
            assert "end_idx" in phase

    def test_single_step(self):
        analytics = compute_step_analytics([_make_step()])
        phases = detect_phases(analytics)
        assert isinstance(phases, list)
        assert len(phases) >= 1

    def test_empty_analytics(self):
        phases = detect_phases([])
        assert isinstance(phases, list)


# ---------------------------------------------------------------------------
# generate_insights
# ---------------------------------------------------------------------------


class TestGenerateInsights:
    def test_returns_strings(self):
        steps = [_make_step(i) for i in range(5)]
        analytics = compute_step_analytics(steps)
        phases = detect_phases(analytics)
        insights = generate_insights(analytics, phases)
        assert isinstance(insights, list)
        for item in insights:
            assert isinstance(item, str)

    def test_empty_input(self):
        insights = generate_insights([], [])
        assert isinstance(insights, list)


# ---------------------------------------------------------------------------
# build_sub_agent_tree
# ---------------------------------------------------------------------------


class TestBuildSubAgentTree:
    def test_builds_tree_from_sub_agent_msg_list(self):
        steps = [
            {"id": "parent_1", "sub_agent_msg_list": ["child_a", "child_b"]},
            {"id": "child_a", "sub_agent_msg_list": []},
            {"id": "child_b", "sub_agent_msg_list": []},
        ]
        tree = build_sub_agent_tree(steps)
        assert "parent_1" in tree
        assert tree["parent_1"] == [1, 2]

    def test_empty_when_no_sub_agents(self):
        steps = [
            {"id": "msg_0", "sub_agent_msg_list": []},
            {"id": "msg_1", "sub_agent_msg_list": []},
        ]
        tree = build_sub_agent_tree(steps)
        assert tree == {}

    def test_handles_missing_child_ids(self):
        steps = [
            {"id": "parent", "sub_agent_msg_list": ["exists", "missing"]},
            {"id": "exists", "sub_agent_msg_list": []},
        ]
        tree = build_sub_agent_tree(steps)
        assert tree["parent"] == [1]  # only "exists" maps

    def test_empty_steps(self):
        assert build_sub_agent_tree([]) == {}
