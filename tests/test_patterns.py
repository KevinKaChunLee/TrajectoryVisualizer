"""Tests for pattern detection functions."""

import pytest

from trajectory_visualizer.insight.patterns import (
    build_structural_phase_segments,
    classify_structural_phase,
    detect_tool_sequences,
    detect_failure_patterns,
    detect_phase_anomalies,
)


def _make_step(index=0, role="assistant", tool_calls=None, error_count=0,
               tool_call_count=0, finish="tool_use", parts=None):
    return {
        "index": index,
        "role": role,
        "tool_calls": tool_calls or [],
        "error_count": error_count,
        "tool_call_count": tool_call_count,
        "finish": finish,
        "parts": parts or [],
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_tokens": 10,
        "reasoning_tokens": 5,
        "duration": 1.0,
    }


class TestDetectToolSequences:
    def test_finds_recurring_bigrams(self):
        tc_read = [{"tool_name": "read_file"}]
        tc_edit = [{"tool_name": "edit_file"}]
        # repeat read->edit 4 times
        steps = []
        for _ in range(4):
            steps.append(_make_step(tool_calls=tc_read, tool_call_count=1))
            steps.append(_make_step(tool_calls=tc_edit, tool_call_count=1))

        result = detect_tool_sequences(steps, min_freq=3)
        assert len(result) > 0
        # Should find the read->edit bigram
        found = any(
            s["sequence"] == ["read_file", "edit_file"]
            for s in result
        )
        assert found

    def test_no_recurring_sequences(self):
        steps = [
            _make_step(tool_calls=[{"tool_name": "a"}], tool_call_count=1),
            _make_step(tool_calls=[{"tool_name": "b"}], tool_call_count=1),
            _make_step(tool_calls=[{"tool_name": "c"}], tool_call_count=1),
        ]
        result = detect_tool_sequences(steps, min_freq=3)
        assert result == []

    def test_empty_steps(self):
        assert detect_tool_sequences([]) == []

    def test_no_tool_calls(self):
        steps = [_make_step() for _ in range(5)]
        assert detect_tool_sequences(steps) == []


class TestDetectFailurePatterns:
    def test_clusters_similar_errors(self):
        steps = [
            _make_step(error_count=1, tool_calls=[{"tool_name": "bash", "error": "file not found: a.py"}]),
            _make_step(tool_calls=[{"tool_name": "read_file"}], tool_call_count=1),
            _make_step(error_count=1, tool_calls=[{"tool_name": "bash", "error": "file not found: b.py"}]),
            _make_step(tool_calls=[{"tool_name": "read_file"}], tool_call_count=1),
        ]
        result = detect_failure_patterns(steps)
        assert isinstance(result, list)

    def test_no_errors(self):
        steps = [_make_step() for _ in range(3)]
        result = detect_failure_patterns(steps)
        assert result == []

    def test_empty_steps(self):
        assert detect_failure_patterns([]) == []


class TestDetectPhaseAnomalies:
    def test_backward_transition(self):
        phases = [
            {"name": "implement", "start_idx": 0, "end_idx": 4},
            {"name": "understand", "start_idx": 5, "end_idx": 8},
        ]
        steps = [_make_step() for _ in range(9)]
        result = detect_phase_anomalies(steps, phases)
        assert len(result) > 0
        assert result[0]["from_phase"] == "implement"
        assert result[0]["to_phase"] == "understand"

    def test_forward_only(self):
        phases = [
            {"name": "understand", "start_idx": 0, "end_idx": 3},
            {"name": "implement", "start_idx": 4, "end_idx": 8},
        ]
        steps = [_make_step() for _ in range(9)]
        result = detect_phase_anomalies(steps, phases)
        assert result == []

    def test_empty_phases(self):
        assert detect_phase_anomalies([], []) == []

    def test_single_phase(self):
        phases = [{"name": "implement", "start_idx": 0, "end_idx": 5}]
        steps = [_make_step() for _ in range(6)]
        assert detect_phase_anomalies(steps, phases) == []


class TestStructuralWorkflowPhases:
    def test_classify_understand_from_read(self):
        step = _make_step(index=0, tool_calls=[{"tool_name": "Read", "input": {"file_path": "a.py"}}])
        assert classify_structural_phase(step) == "understand"

    def test_classify_plan_from_todo_tool(self):
        step = _make_step(index=0, tool_calls=[{"tool_name": "TodoWrite", "input": {"todos": []}}])
        assert classify_structural_phase(step) == "plan"

    def test_classify_implement_from_edit(self):
        step = _make_step(index=0, tool_calls=[{"tool_name": "Edit", "input": {"file_path": "a.py"}}])
        assert classify_structural_phase(step) == "implement"

    def test_classify_validate_from_test_command(self):
        step = _make_step(index=0, tool_calls=[{
            "tool_name": "Bash",
            "input": {"command": "pytest tests/test_app.py"},
            "status": "success",
            "metadata": {},
        }])
        assert classify_structural_phase(step) == "validate"

    def test_classify_debug_from_failure(self):
        step = _make_step(index=0, tool_calls=[{
            "tool_name": "Bash",
            "input": {"command": "python app.py"},
            "status": "error",
            "metadata": {},
        }], error_count=1)
        assert classify_structural_phase(step) == "debug"

    def test_classify_report_from_terminal_text_step(self):
        step = _make_step(index=0, tool_calls=[], finish="stop")
        assert classify_structural_phase(step) == "report"

    def test_build_segments_and_detect_regression(self):
        steps = [
            _make_step(index=0, tool_calls=[{"tool_name": "Read", "input": {"file_path": "a.py"}}]),
            _make_step(index=1, tool_calls=[{"tool_name": "Edit", "input": {"file_path": "a.py"}}]),
            _make_step(index=2, tool_calls=[{"tool_name": "Read", "input": {"file_path": "b.py"}}]),
        ]
        phases = build_structural_phase_segments(steps)
        assert [p["name"] for p in phases] == ["understand", "implement", "understand"]
        anomalies = detect_phase_anomalies(steps, phases)
        assert len(anomalies) == 1
        assert anomalies[0]["from_phase"] == "implement"
        assert anomalies[0]["to_phase"] == "understand"
