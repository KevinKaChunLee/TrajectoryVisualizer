"""Tool Outcome Timeline: success/failure markers, colored by agent when multi-agent."""

from __future__ import annotations

import unittest

from trajviz.insight.charts.swimlanes import build_tool_outcome_timeline
from trajviz.insight.palette import SESSION_COLORS, TOOL_OUTCOME_COLORS


def _step(index: int, *, agent: str = "", tools: list[tuple[str, str]] | None = None) -> dict:
    tool_calls = [
        {"tool_name": name, "status": status}
        for name, status in (tools or [])
    ]
    return {
        "index": index,
        "role": "assistant",
        "agent": agent,
        "is_sub_agent": bool(agent),
        "session_id": agent or "main",
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "tokens": {"total": 0},
    }


class ToolOutcomeTimelineTests(unittest.TestCase):
    def test_single_agent_uses_outcome_colors(self):
        fig = build_tool_outcome_timeline([
            _step(0, tools=[("Read", "success"), ("Bash", "error")]),
        ])
        names = [t.name for t in fig.data if getattr(t, "name", None)]
        self.assertEqual(names, ["Success", "Failure"])
        self.assertEqual(fig.layout.title.text, "Tool Outcome Timeline")
        by_name = {t.name: t for t in fig.data}
        self.assertEqual(by_name["Success"].marker.color, TOOL_OUTCOME_COLORS["success"])
        self.assertEqual(by_name["Failure"].marker.color, TOOL_OUTCOME_COLORS["failure"])

    def test_multi_agent_colors_by_agent_and_shapes_by_outcome(self):
        steps = [
            _step(0, tools=[("Read", "success")]),
            _step(1, agent="explore", tools=[("Grep", "success"), ("Bash", "error")]),
        ]
        fig = build_tool_outcome_timeline(steps)
        self.assertEqual(fig.layout.title.text, "Tool Outcome Timeline by Agent")
        named = [
            t for t in fig.data
            if t.name and t.name not in ("Success (circle)", "Failure (x)")
        ]
        self.assertEqual({t.name for t in named}, {"main", "explore"})
        self.assertEqual(
            {t.marker.color for t in named},
            {SESSION_COLORS[0], SESSION_COLORS[1]},
        )
        self.assertEqual(
            [t.name for t in fig.data if t.name in ("Success (circle)", "Failure (x)")],
            ["Success (circle)", "Failure (x)"],
        )
        explore = next(t for t in named if t.name == "explore")
        symbols = list(explore.marker.symbol) if isinstance(explore.marker.symbol, (list, tuple)) else [explore.marker.symbol]
        self.assertIn("x", symbols)

    def test_only_agents_with_tools_count_as_multi(self):
        # Second agent exists in the trajectory but has no tool calls.
        steps = [
            _step(0, tools=[("Read", "success"), ("Bash", "error")]),
            _step(1, agent="explore", tools=[]),
        ]
        fig = build_tool_outcome_timeline(steps)
        self.assertEqual(fig.layout.title.text, "Tool Outcome Timeline")
        names = [t.name for t in fig.data]
        self.assertEqual(names, ["Success", "Failure"])

    def test_empty_trajectory(self):
        fig = build_tool_outcome_timeline([])
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 0)


if __name__ == "__main__":
    unittest.main()
