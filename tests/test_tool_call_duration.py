"""tool_call_duration_ms prefers part wall-clock over execution-only stamps."""

from __future__ import annotations

import unittest

from trajviz.insight.metrics import (
    compute_metrics,
    tool_call_duration_ms,
    tool_call_stats_duration_ms,
)
from trajviz.insight.parser import parse_steps


class ToolCallDurationMsTests(unittest.TestCase):
    def test_prefers_part_created_updated_over_state_start_end(self):
        tc = {
            "time_created": 1_000,
            "time_updated": 1_500,
            "time_start": 1_400,
            "time_end": 1_420,
            "duration_ms": 20,
        }
        self.assertEqual(tool_call_duration_ms(tc), 500.0)

    def test_falls_back_to_execution_window(self):
        tc = {"time_start": 100, "time_end": 250}
        self.assertEqual(tool_call_duration_ms(tc), 150.0)

    def test_falls_back_to_duration_ms(self):
        self.assertEqual(tool_call_duration_ms({"duration_ms": 80}), 80.0)
        self.assertEqual(
            tool_call_duration_ms({"metadata": {"totalDurationMs": 90}}),
            90.0,
        )

    def test_parse_steps_keeps_part_and_state_times(self):
        raw = {
            "messages": [
                {
                    "info": {
                        "role": "assistant",
                        "time": {"created": 1, "completed": 10_000},
                        "tokens": {"input": 1, "output": 1, "total": 2},
                    },
                    "parts": [
                        {
                            "type": "tool",
                            "tool": "bash",
                            "callID": "c1",
                            "time": {"created": 1_000, "updated": 2_000},
                            "state": {
                                "status": "success",
                                "input": {"command": "ls"},
                                "output": "ok",
                                "time": {"start": 1_800, "end": 1_820},
                            },
                        }
                    ],
                }
            ]
        }
        steps = parse_steps(raw)
        tc = steps[0]["tool_calls"][0]
        self.assertEqual(tc["time_created"], 1_000)
        self.assertEqual(tc["time_updated"], 2_000)
        self.assertEqual(tc["time_start"], 1_800)
        self.assertEqual(tc["time_end"], 1_820)
        self.assertEqual(tool_call_duration_ms(tc), 1_000.0)

    def test_stats_duration_skips_spawn_tools(self):
        task = {"tool_name": "task", "duration_ms": 80_000}
        bash = {"tool_name": "bash", "duration_ms": 1_000}
        self.assertIsNone(tool_call_stats_duration_ms(task))
        self.assertEqual(tool_call_stats_duration_ms(bash), 1_000.0)
        self.assertEqual(tool_call_duration_ms(task), 80_000.0)


class ToolWaitShareSubagentTests(unittest.TestCase):
    def test_excludes_spawn_and_uses_wall_clock_denominator(self):
        """Spawn wait + summed step durations must not inflate Tool-wait %."""
        parent = {
            "index": 0,
            "role": "assistant",
            "duration": 100.0,
            "time_created_ms": 0,
            "time_completed_ms": 100_000,
            "parts": [],
            "tool_calls": [
                {"tool_name": "task", "status": "success", "duration_ms": 80_000},
                {"tool_name": "bash", "status": "success", "duration_ms": 5_000},
            ],
            "tool_call_count": 2,
            "tokens": {
                "total": 10, "input": 0, "output": 10,
                "reasoning": 0, "cache_read": 0, "cache_write": 0,
            },
        }
        child = {
            "index": 1,
            "role": "assistant",
            "duration": 80.0,
            "is_sub_agent": True,
            "time_created_ms": 10_000,
            "time_completed_ms": 90_000,
            "parts": [],
            "tool_calls": [
                {"tool_name": "read", "status": "success", "duration_ms": 10_000},
            ],
            "tool_call_count": 1,
            "tokens": {
                "total": 20, "input": 0, "output": 20,
                "reasoning": 0, "cache_read": 0, "cache_write": 0,
            },
        }
        # No loader timing: wall from timestamps = 100s (not sum 180s).
        metrics = compute_metrics([parent, child], {})
        # Non-spawn only: 5s (parent bash) + 10s (child read) = 15s
        self.assertEqual(metrics["wall_clock"], 100.0)
        self.assertEqual(metrics["tool_time_total"], 15.0)
        self.assertEqual(metrics["tool_wait_share"], 15.0)  # 15/100
        self.assertEqual(metrics["max_tool_duration"], 10.0)

    def test_loader_timing_preferred_over_step_span(self):
        steps = [{
            "index": 0,
            "role": "assistant",
            "duration": 50.0,
            "time_created_ms": 0,
            "time_completed_ms": 50_000,
            "parts": [],
            "tool_calls": [
                {"tool_name": "bash", "status": "success", "duration_ms": 10_000},
            ],
            "tool_call_count": 1,
            "tokens": {
                "total": 1, "input": 0, "output": 1,
                "reasoning": 0, "cache_read": 0, "cache_write": 0,
            },
        }]
        metrics = compute_metrics(steps, {"timing": {"total_duration": 40.0}})
        self.assertEqual(metrics["wall_clock"], 40.0)
        self.assertEqual(metrics["tool_wait_share"], 25.0)  # 10/40

if __name__ == "__main__":
    unittest.main()
