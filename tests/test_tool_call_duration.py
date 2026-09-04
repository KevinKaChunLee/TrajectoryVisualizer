"""tool_call_duration_ms prefers part wall-clock over execution-only stamps."""

from __future__ import annotations

import unittest

from trajviz.insight.metrics import tool_call_duration_ms
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


if __name__ == "__main__":
    unittest.main()
