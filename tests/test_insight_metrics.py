"""Unit tests for trajectory_visualizer.insight.metrics — metrics computation and health verdicts."""

from __future__ import annotations

import pytest

from trajectory_visualizer.insight.metrics import build_message_metrics, compute_health_verdict, compute_metrics


# ---------------------------------------------------------------------------
# Fixtures — minimal step data
# ---------------------------------------------------------------------------


def _make_step(role="assistant", total_tok=1000, output_tok=100, duration=5.0,
               tool_calls=None, finish="tool_use", cache_read=800):
    return {
        "index": 0,
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
        "text_preview": "test",
        "finish": finish,
        "model_id": "test-model",
        "provider_id": "test",
        "time_created_ms": 1000000,
        "time_completed_ms": 1000000 + int(duration * 1000),
        "agent": "",
        "mode": "",
        "message_id": "",
        "id": "msg_1",
        "parent_id": "",
        "session_id": "ses_1",
        "cwd": "",
        "root": "",
    }


def _make_tool_call(name="Bash", status="success"):
    return {
        "type": "tool_call",
        "tool_name": name,
        "tool_id": "tc_1",
        "status": status,
        "title": "",
        "input": {},
        "output": "",
        "error": None,
        "time_start": None,
        "time_end": None,
        "duration_ms": None,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# build_message_metrics
# ---------------------------------------------------------------------------


class TestBuildMessageMetrics:
    def test_returns_list(self):
        steps = [_make_step()]
        rows = build_message_metrics(steps)
        assert isinstance(rows, list)
        assert len(rows) == 1

    def test_row_has_expected_keys(self):
        rows = build_message_metrics([_make_step()])
        row = rows[0]
        assert "tokens_total" in row
        assert "duration" in row
        assert "role" in row

    def test_empty_steps(self):
        rows = build_message_metrics([])
        assert rows == []


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_basic_metrics(self):
        steps = [
            _make_step(tool_calls=[_make_tool_call()]),
            _make_step(role="user", total_tok=0, output_tok=0, cache_read=0, duration=0),
            _make_step(tool_calls=[_make_tool_call(), _make_tool_call("Read")]),
        ]
        rows = build_message_metrics(steps)
        m = compute_metrics(steps, {}, message_rows=rows)

        assert m["total_steps"] == 3
        assert m["assistant_steps"] == 2
        assert m["tool_call_count"] >= 3
        assert 0 <= m["tool_success_rate"] <= 100
        assert m["tokens"]["total"] >= 0
        assert isinstance(m["tokens_per_second"], (int, float))

    def test_metrics_with_empty_steps(self):
        m = compute_metrics([], {})
        assert m["total_steps"] == 0
        assert m["tool_call_count"] == 0

    def test_tool_breakdown(self):
        steps = [
            _make_step(tool_calls=[_make_tool_call("Bash"), _make_tool_call("Read")]),
        ]
        rows = build_message_metrics(steps)
        m = compute_metrics(steps, {}, message_rows=rows)
        assert "Bash" in m["tool_breakdown"]
        assert "Read" in m["tool_breakdown"]


# ---------------------------------------------------------------------------
# compute_health_verdict
# ---------------------------------------------------------------------------


class TestComputeHealthVerdict:
    def test_returns_list_of_dicts(self):
        steps = [_make_step(tool_calls=[_make_tool_call()])]
        rows = build_message_metrics(steps)
        m = compute_metrics(steps, {}, message_rows=rows)
        verdicts = compute_health_verdict(m, [])
        assert isinstance(verdicts, list)
        for v in verdicts:
            assert "metric" in v
            assert "status" in v
            assert v["status"] in ("good", "warn", "bad")
            assert "detail" in v

    def test_empty_metrics(self):
        m = compute_metrics([], {})
        verdicts = compute_health_verdict(m, [])
        assert isinstance(verdicts, list)
