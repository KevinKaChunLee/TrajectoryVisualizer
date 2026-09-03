"""Step Duration chart: system (scaffold) vs tool (agentic) error series."""

from __future__ import annotations

import unittest

from trajviz.insight.charts.usage import _duration_error_kind, build_duration_chart
from trajviz.insight.palette import DURATION_ERROR_COLORS


def _tc(name: str, *, status: str = "error", exit_code: int | None = None) -> dict:
    meta = {} if exit_code is None else {"exit": exit_code}
    return {"tool_name": name, "status": status, "metadata": meta}


def _step(
    *,
    duration: float = 1.0,
    error_count: int = 0,
    finish: str = "",
    tool_calls: list[dict] | None = None,
) -> dict:
    return {
        "duration": duration,
        "error_count": error_count,
        "finish": finish,
        "tool_calls": tool_calls or [],
        "tokens": {"total": 0},
    }


class DurationErrorKindTests(unittest.TestCase):
    def test_bash_grep_failures_are_system(self):
        self.assertEqual(
            _duration_error_kind(_step(tool_calls=[_tc("Bash")])),
            "system",
        )
        self.assertEqual(
            _duration_error_kind(_step(tool_calls=[_tc("Grep")])),
            "system",
        )
        self.assertEqual(
            _duration_error_kind(_step(tool_calls=[_tc("Read")])),
            "system",
        )

    def test_agentic_workflow_failures_are_tool(self):
        self.assertEqual(
            _duration_error_kind(_step(tool_calls=[_tc("skill")])),
            "tool",
        )
        self.assertEqual(
            _duration_error_kind(_step(tool_calls=[_tc("Task")])),
            "tool",
        )
        self.assertEqual(
            _duration_error_kind(_step(tool_calls=[_tc("mcp__jira__create")])),
            "tool",
        )
        self.assertEqual(
            _duration_error_kind(_step(tool_calls=[_tc("TodoWrite")])),
            "tool",
        )

    def test_mixed_failures_prefer_tool(self):
        self.assertEqual(
            _duration_error_kind(
                _step(tool_calls=[_tc("Bash"), _tc("skill")]),
            ),
            "tool",
        )

    def test_provider_abort_is_system(self):
        self.assertEqual(_duration_error_kind(_step(finish="error")), "system")

    def test_error_count_fallback_without_tool_calls(self):
        self.assertEqual(_duration_error_kind(_step(error_count=1)), "tool")

    def test_normal_when_neither(self):
        self.assertIsNone(
            _duration_error_kind(_step(finish="stop", tool_calls=[_tc("Bash", status="success")])),
        )


class DurationChartSeriesTests(unittest.TestCase):
    def test_splits_system_and_tool_into_separate_traces(self):
        steps = [
            _step(duration=1.0),
            _step(duration=2.0, tool_calls=[_tc("Bash")]),
            _step(duration=3.0, tool_calls=[_tc("skill")]),
        ]
        fig = build_duration_chart(steps)
        names = [t.name for t in fig.data if getattr(t, "name", None)]
        self.assertEqual(names, ["Normal", "System Error", "Tool Error"])

        by_name = {t.name: t for t in fig.data if getattr(t, "name", None)}
        self.assertEqual(list(by_name["Normal"].x), [0])
        self.assertEqual(list(by_name["System Error"].x), [1])
        self.assertEqual(list(by_name["Tool Error"].x), [2])
        self.assertEqual(
            by_name["System Error"].marker.color,
            DURATION_ERROR_COLORS["system"],
        )
        self.assertEqual(
            by_name["Tool Error"].marker.color,
            DURATION_ERROR_COLORS["tool"],
        )
        self.assertEqual(DURATION_ERROR_COLORS["system"], "#d97706")
        self.assertEqual(DURATION_ERROR_COLORS["tool"], "#dc2626")

    def test_omits_empty_error_series(self):
        fig = build_duration_chart([_step(duration=1.0, tool_calls=[_tc("Bash")])])
        names = [t.name for t in fig.data if getattr(t, "name", None)]
        self.assertEqual(names, ["Normal", "System Error"])


if __name__ == "__main__":
    unittest.main()
