"""Focused regressions for follow-up findings from the Insight review."""

import unittest

from trajviz.insight import patterns
from trajviz.insight.metrics import compute_health_verdict, compute_metrics


def _assistant_step(index: int, output_tokens: int, duration: float | None) -> dict:
    return {
        "index": index,
        "role": "assistant",
        "duration": duration,
        "parts": [],
        "tool_calls": [],
        "tool_call_count": 0,
        "tokens": {
            "total": output_tokens,
            "input": 0,
            "output": output_tokens,
            "reasoning": 0,
            "cache_read": 0,
            "cache_write": 0,
        },
    }


def _bash_call(command: str, *, tool_name: str = "Bash") -> dict:
    return {
        "tool_name": tool_name,
        "input": {"command": command},
        "output": "",
        "status": "success",
    }


class OutputThroughputTests(unittest.TestCase):
    def test_partial_timing_uses_tokens_from_only_the_timed_steps(self):
        steps = [
            _assistant_step(0, output_tokens=100, duration=10.0),
            _assistant_step(1, output_tokens=900, duration=None),
        ]

        metrics = compute_metrics(steps, {})

        self.assertEqual(metrics["tokens"]["output"], 1000)
        self.assertEqual(metrics["output_tokens_per_sec"], 10.0)
        self.assertEqual(metrics["output_throughput_timed_tokens"], 100.0)
        self.assertEqual(metrics["output_throughput_timed_seconds"], 10.0)
        self.assertEqual(metrics["output_throughput_timed_steps"], 1)
        self.assertEqual(metrics["output_throughput_total_steps"], 2)
        self.assertEqual(metrics["output_throughput_coverage_pct"], 50.0)
        self.assertTrue(metrics["output_throughput_incomplete"])

        throughput = {
            verdict["metric"]: verdict
            for verdict in compute_health_verdict(metrics, [])
        }["Throughput"]
        self.assertEqual(throughput["status"], "bad")
        self.assertEqual(throughput["label"], "10.0 output tok/s")
        self.assertIn("1/2 assistant steps with timing", throughput["detail"])

    def test_complete_timing_reports_full_coverage(self):
        metrics = compute_metrics([
            _assistant_step(0, output_tokens=100, duration=10.0),
            _assistant_step(1, output_tokens=400, duration=20.0),
        ], {})

        self.assertEqual(metrics["output_tokens_per_sec"], 16.7)
        self.assertEqual(metrics["output_throughput_coverage_pct"], 100.0)
        self.assertFalse(metrics["output_throughput_incomplete"])

    def test_overview_labels_generation_rate_as_output_throughput(self):
        # Import here because this UI module loads optional Gradio dependencies.
        from trajviz.insight.presenters import build_overview_kpi_html
        from trajviz.insight.formatting import (
            format_banner_html,
            format_performance_md,
        )

        metrics = {
            "total_steps": 2,
            "assistant_steps": 2,
            "user_steps": 0,
            "p95_duration": 10.0,
            "tokens": {"total": 1000},
            "tokens_per_second": 100.0,
            "output_tokens_per_sec": 10.0,
            "output_throughput_timed_steps": 1,
            "output_throughput_total_steps": 2,
            "output_throughput_incomplete": True,
            "tool_success_rate": 0,
            "tool_call_count": 0,
        }

        html = build_overview_kpi_html(metrics, "10s")

        self.assertIn("10.0 output tok/s", html)
        self.assertIn("1/2 timed", html)
        self.assertNotIn("100.0 tok/s", html)

        banner = format_banner_html("trace.json", metrics, "10s")
        self.assertIn("10.0 output tok/s", banner)
        self.assertNotIn("100.0 tok/s", banner)

        computed_metrics = compute_metrics([
            _assistant_step(0, output_tokens=100, duration=10.0),
            _assistant_step(1, output_tokens=900, duration=None),
        ], {})
        performance = format_performance_md(computed_metrics, "10s")
        self.assertIn("Total processed tok/sec", performance)
        self.assertIn("Median processed tok/sec", performance)


class WrappedShellSearchTests(unittest.TestCase):
    def test_common_wrappers_are_recognized_as_searches(self):
        commands = [
            "cd src && /usr/bin/rg needle .",
            "env LANG=C command grep -R needle .",
            "git -C repo grep needle",
            "printf '%s\\0' src | xargs -0 rg needle",
            "bash -lc 'cd src && find . -name *.py'",
        ]

        for command in commands:
            with self.subTest(command=command):
                self.assertTrue(patterns._is_search_call(_bash_call(command)))

    def test_lowercase_bash_tool_name_is_recognized(self):
        self.assertTrue(patterns._is_search_call(
            _bash_call("FOO=1 ripgrep needle .", tool_name="bash")
        ))

    def test_commands_that_only_mention_search_tools_are_not_searches(self):
        commands = [
            "echo 'rg needle'",
            "python -c 'print(\"grep needle\")'",
            "printf '%s' 'find . -name *.py'",
        ]

        for command in commands:
            with self.subTest(command=command):
                self.assertFalse(patterns._is_search_call(_bash_call(command)))

    def test_malformed_shell_quoting_is_conservative(self):
        self.assertFalse(patterns._is_search_call(_bash_call("rg 'unterminated")))

    def test_wrapped_empty_searches_form_a_fruitless_streak(self):
        steps = [
            {"index": 0, "role": "assistant", "tool_calls": [
                _bash_call("cd src && rg first ."),
            ]},
            {"index": 1, "role": "assistant", "tool_calls": [
                _bash_call("env LANG=C grep -R second ."),
            ]},
            {"index": 2, "role": "assistant", "tool_calls": [
                _bash_call("git grep third"),
            ]},
        ]

        streaks = patterns.detect_fruitless_streaks(steps)

        self.assertEqual(len(streaks), 1)
        self.assertEqual(streaks[0]["length"], 3)


if __name__ == "__main__":
    unittest.main()
