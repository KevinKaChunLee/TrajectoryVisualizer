"""Unit tests for trajectory_visualizer.insight.judge."""

from __future__ import annotations

import pytest

from trajectory_visualizer.insight.judge import (
    build_judge_prompt,
    _parse_judge_response,
    invoke_judge,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SCORE_RESULT = {
    "composite_score": 52.0,
    "composite_verdict": "warn",
    "dimensions": {
        "targeting": {
            "score": 60.0, "verdict": "warn",
            "metrics": {"avg_steps_to_first_touch": 5.5, "exploration_ratio": 4.2},
        },
        "error_resilience": {
            "score": 45.0, "verdict": "warn",
            "metrics": {"failure_chain_count": 2, "tool_success_rate": 82.0},
        },
    },
    "profile": "default",
    "judge": None,
}

_CHAINS = [
    {"start": 5, "end": 7, "steps": [5, 6, 7]},
    {"start": 12, "end": 12, "steps": [12]},
]

_BOTTLENECKS = [
    {"step_idx": 8, "duration": 45.0, "explanation": "Step 8: 45.0s — 38.0s executing tools", "decomposition": {"inference_pct": 15.6}},
]


# ===========================================================================
# Prompt construction
# ===========================================================================

class TestBuildJudgePrompt:
    def test_contains_composite_score(self):
        prompt = build_judge_prompt(_SCORE_RESULT, _CHAINS, _BOTTLENECKS)
        assert "52.0" in prompt

    def test_contains_dimension_metrics(self):
        prompt = build_judge_prompt(_SCORE_RESULT, _CHAINS, _BOTTLENECKS)
        assert "exploration_ratio" in prompt
        assert "4.2" in prompt

    def test_contains_failure_chain_info(self):
        prompt = build_judge_prompt(_SCORE_RESULT, _CHAINS, _BOTTLENECKS)
        assert "Steps 5" in prompt

    def test_contains_bottleneck_info(self):
        prompt = build_judge_prompt(_SCORE_RESULT, _CHAINS, _BOTTLENECKS)
        assert "Step 8" in prompt

    def test_no_chains(self):
        prompt = build_judge_prompt(_SCORE_RESULT, None, None)
        assert "No failure chains" in prompt

    def test_asks_targeted_question(self):
        prompt = build_judge_prompt(_SCORE_RESULT)
        assert "acceptable" in prompt.lower() or "evaluate" in prompt.lower()
        # Should NOT use blanket correct/incorrect
        assert "exactly one word" not in prompt.lower()


# ===========================================================================
# Response parsing
# ===========================================================================

class TestParseJudgeResponse:
    def test_valid_json_block(self):
        raw = '```json\n{"verdict": "poor", "reasoning": "Too many retries", "flagged_steps": [5, 12]}\n```'
        result = _parse_judge_response(raw)
        assert result["verdict"] == "poor"
        assert result["reasoning"] == "Too many retries"
        assert result["flagged_steps"] == [5, 12]

    def test_bare_json(self):
        raw = '{"verdict": "acceptable", "reasoning": "Looks fine", "flagged_steps": []}'
        result = _parse_judge_response(raw)
        assert result["verdict"] == "acceptable"

    def test_json_with_surrounding_text(self):
        raw = 'Here is my analysis:\n{"verdict": "poor", "reasoning": "bad", "flagged_steps": [3]}\nDone.'
        result = _parse_judge_response(raw)
        assert result["verdict"] == "poor"
        assert result["flagged_steps"] == [3]

    def test_gibberish(self):
        result = _parse_judge_response("I don't understand the question")
        assert result["verdict"] == "uncertain"
        assert "Failed to parse" in result["reasoning"]

    def test_empty_response(self):
        result = _parse_judge_response("")
        assert result["verdict"] == "uncertain"

    def test_invalid_verdict(self):
        raw = '{"verdict": "correct", "reasoning": "ok", "flagged_steps": []}'
        result = _parse_judge_response(raw)
        assert result["verdict"] == "uncertain"  # "correct" not in valid set

    def test_missing_fields_defaults(self):
        raw = '{"verdict": "acceptable"}'
        result = _parse_judge_response(raw)
        assert result["verdict"] == "acceptable"
        assert result["reasoning"] == ""
        assert result["flagged_steps"] == []


# ===========================================================================
# Judge orchestrator
# ===========================================================================

class TestInvokeJudge:
    def test_skips_when_score_above_band(self):
        score = {**_SCORE_RESULT, "composite_score": 85.0}
        result = invoke_judge(score)
        assert result is None

    def test_skips_when_score_below_band(self):
        score = {**_SCORE_RESULT, "composite_score": 20.0}
        result = invoke_judge(score)
        assert result is None

    def test_skips_when_score_is_none(self):
        score = {**_SCORE_RESULT, "composite_score": None}
        result = invoke_judge(score)
        assert result is None

    def test_invokes_when_in_band(self):
        mock_response = '{"verdict": "poor", "reasoning": "bad trajectory", "flagged_steps": [5]}'
        result = invoke_judge(
            _SCORE_RESULT,
            llm_callable=lambda prompt: mock_response,
        )
        assert result is not None
        assert result["verdict"] == "poor"

    def test_handles_llm_error(self):
        def failing_llm(prompt):
            raise TimeoutError("LLM call timed out")

        result = invoke_judge(
            _SCORE_RESULT,
            llm_callable=failing_llm,
        )
        assert result["verdict"] == "uncertain"
        assert "timed out" in result["reasoning"]

    def test_custom_uncertain_band(self):
        score = {**_SCORE_RESULT, "composite_score": 52.0}
        # Narrow band that excludes 52
        result = invoke_judge(score, uncertain_band=(53, 60))
        assert result is None

    def test_prompt_contains_metrics(self):
        captured = []
        def capture_llm(prompt):
            captured.append(prompt)
            return '{"verdict": "acceptable", "reasoning": "ok", "flagged_steps": []}'

        invoke_judge(_SCORE_RESULT, _CHAINS, _BOTTLENECKS, llm_callable=capture_llm)
        assert len(captured) == 1
        assert "exploration_ratio" in captured[0]
        assert "Steps 5" in captured[0]
