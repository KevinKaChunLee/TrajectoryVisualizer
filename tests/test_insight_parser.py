"""Unit tests for trajectory_visualizer.insight.parser — step parsing and token inference."""

from __future__ import annotations

import os

import pytest

from trajectory_visualizer.insight.parser import infer_non_cache_input, parse_steps


# ---------------------------------------------------------------------------
# infer_non_cache_input
# ---------------------------------------------------------------------------


class TestInferNonCacheInput:
    def test_fresh_input_schema(self):
        """When total = input + output + reasoning + cache, input is already fresh."""
        result = infer_non_cache_input(
            total_tokens=1000,
            input_tokens=200,
            output_tokens=300,
            reasoning_tokens=0,
            cache_read_tokens=500,
        )
        assert result >= 0

    def test_cached_input_schema(self):
        """When total = input + output + reasoning, input includes cache."""
        result = infer_non_cache_input(
            total_tokens=500,
            input_tokens=200,
            output_tokens=300,
            reasoning_tokens=0,
            cache_read_tokens=100,
        )
        assert result >= 0

    def test_zero_tokens(self):
        result = infer_non_cache_input(0, 0, 0, 0, 0)
        assert result == 0

    def test_none_values_treated_as_zero(self):
        result = infer_non_cache_input(0, None, None, None, None)
        assert result == 0

    def test_returns_non_negative(self):
        """Edge case: should never return negative."""
        result = infer_non_cache_input(10, 5, 3, 0, 100)
        assert result >= 0


# ---------------------------------------------------------------------------
# parse_steps
# ---------------------------------------------------------------------------

_SAMPLE_CC = os.path.join(os.path.dirname(__file__), "..", "samples", "cc_trajectory.json")
_REQUIRED_STEP_KEYS = {"index", "role", "tokens", "parts", "tool_calls", "tool_call_count", "error_count"}


class TestParseSteps:
    @pytest.mark.skipif(not os.path.isfile(_SAMPLE_CC), reason="Large sample file not present")
    def test_parse_cc_trajectory(self):
        import json
        with open(_SAMPLE_CC) as f:
            raw = json.load(f)
        # Need to convert CC format first
        from trajectory_visualizer.insight.loaders import load_trajectory
        loaded = load_trajectory(_SAMPLE_CC)
        steps = parse_steps(loaded)
        assert len(steps) > 0
        for step in steps:
            assert _REQUIRED_STEP_KEYS.issubset(step.keys()), f"Missing keys in step: {_REQUIRED_STEP_KEYS - step.keys()}"

    def test_parse_empty_trajectory(self):
        steps = parse_steps({})
        assert steps == []

    def test_parse_trajectory_with_no_messages(self):
        steps = parse_steps({"trajectory": []})
        assert steps == []

    def test_parse_minimal_opencode_message(self):
        raw = {
            "trajectory": [
                {
                    "info": {"role": "user", "finish": ""},
                    "parts": [{"type": "text", "text": "hello"}],
                }
            ]
        }
        steps = parse_steps(raw)
        assert len(steps) == 1
        assert steps[0]["role"] == "user"
        assert steps[0]["index"] == 0

    def test_step_tokens_structure(self):
        raw = {
            "trajectory": [
                {
                    "info": {
                        "role": "assistant",
                        "tokens": {"total": 100, "input": 50, "output": 30,
                                   "reasoning": 0, "cache": {"read": 20, "write": 0}},
                        "finish": "stop",
                    },
                    "parts": [{"type": "text", "text": "response"}],
                }
            ]
        }
        steps = parse_steps(raw)
        tok = steps[0]["tokens"]
        assert tok["total"] == 100
        assert tok["input"] == 50
        assert tok["output"] == 30
        assert tok["cache_read"] == 20

    def test_codearts_fields_extracted(self):
        raw = {
            "trajectory": [
                {
                    "info": {
                        "role": "assistant", "finish": "tool_use",
                        "round": 7, "isSubAgent": True,
                        "subAgentMsgList": ["msg_a", "msg_b"],
                        "toolOutput": {"bash": {"content": "total 20..."}},
                        "outputText": "I'll explore this...",
                        "agentId": "CodeAgent",
                        "question": [{"type": "text", "content": "analyze this"}],
                        "sessionID": "ses_123",
                    },
                    "parts": [{"type": "text", "text": "hello"}],
                }
            ]
        }
        steps = parse_steps(raw)
        s = steps[0]
        assert s["round"] == 7
        assert s["is_sub_agent"] is True
        assert s["sub_agent_msg_list"] == ["msg_a", "msg_b"]
        assert s["tool_output"]["bash"]["content"] == "total 20..."
        assert s["output_text"] == "I'll explore this..."
        assert s["agent_id"] == "CodeAgent"
        assert len(s["question"]) == 1

    def test_codearts_fields_default_for_non_codearts(self):
        raw = {
            "trajectory": [
                {
                    "info": {"role": "user"},
                    "parts": [{"type": "text", "text": "hi"}],
                }
            ]
        }
        steps = parse_steps(raw)
        s = steps[0]
        assert s["round"] is None
        assert s["is_sub_agent"] is False
        assert s["sub_agent_msg_list"] == []
        assert s["tool_output"] is None
        assert s["output_text"] == ""
        assert s["agent_id"] == ""
        assert s["question"] == []
