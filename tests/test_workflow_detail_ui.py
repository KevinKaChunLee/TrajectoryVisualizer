import inspect
import unittest
from pathlib import Path


class WorkflowDetailUiTests(unittest.TestCase):
    @staticmethod
    def _metric_step(**overrides):
        step = {
            "index": 1,
            "role": "assistant",
            "parts": [],
            "tokens": {
                "total": 100,
                "input": 100,
                "output": 0,
                "reasoning": 0,
                "cache_read": 0,
                "cache_write": 0,
            },
            "duration": 2.0,
            "tool_calls": [],
            "tool_call_count": 0,
            "error_count": 0,
            "has_reasoning": False,
        }
        step.update(overrides)
        return step

    def test_selecting_workflow_card_resets_detail_panel_scroll(self):
        from trajectory_visualizer.insight import insight

        source = inspect.getsource(insight.build_ui)

        self.assertIn("detailPanel.scrollTop = 0", source)

    def test_detail_tabs_stay_visible_inside_scrollable_detail_panel(self):
        styles = Path("trajectory_visualizer/insight/styles.py").read_text()

        self.assertIn("position: sticky", styles[styles.index(".dp-tabs"):styles.index(".dp-tab {")])
        self.assertIn("top: 0", styles[styles.index(".dp-tabs"):styles.index(".dp-tab {")])

    def test_detail_tabs_do_not_use_fragile_inline_click_handler(self):
        from trajectory_visualizer.insight.rendering import format_step_detail

        html = format_step_detail({
            "index": 1,
            "role": "assistant",
            "parts": [{"type": "text", "content": "hello"}],
            "tokens": {
                "total": 0,
                "input": 0,
                "output": 0,
                "reasoning": 0,
                "cache_read": 0,
                "cache_write": 0,
            },
            "tool_calls": [],
            "tool_call_count": 0,
            "error_count": 0,
            "has_reasoning": False,
        })

        self.assertIn("class='dp-tabs'", html)
        self.assertNotIn("onclick=", html[html.index("class='dp-tabs'"):html.index("data-tab-content='content'")])

    def test_workflow_registers_delegated_detail_tab_handler(self):
        from trajectory_visualizer.insight import insight

        source = inspect.getsource(insight.build_ui)

        self.assertIn("__dpTabHandlerAttached", source)
        self.assertIn("document.addEventListener('click', function(e) {", source)
        self.assertIn("e.target.closest('.dp-tab')", source)

    def test_complete_metrics_render_the_whole_table_and_keep_real_zeroes(self):
        from trajectory_visualizer.insight.rendering import _format_metrics_tab

        html = _format_metrics_tab(self._metric_step())

        self.assertIn("<table class='dp-meta-table'>", html)
        self.assertIn("<td>Output Tokens</td><td>0</td>", html)
        self.assertIn("<td>Cache Ratio</td><td>0.0%</td>", html)
        self.assertNotIn("Metrics unavailable", html)

    def test_one_missing_metric_replaces_the_entire_table_with_a_message(self):
        from trajectory_visualizer.insight.rendering import _format_metrics_tab

        tokens = dict(self._metric_step()["tokens"])
        del tokens["cache_write"]
        html = _format_metrics_tab(self._metric_step(tokens=tokens))

        self.assertNotIn("<table", html)
        self.assertIn("Metrics unavailable", html)
        self.assertIn("Missing or unavailable: Cache Write", html)

    def test_missing_duration_keeps_token_table_with_na_derived_rows(self):
        # The final step of a Claude Code trajectory has genuine token data
        # but no recorded duration; the table must not be hidden for it.
        from trajectory_visualizer.insight.rendering import _format_metrics_tab

        html = _format_metrics_tab(self._metric_step(duration=None))

        self.assertIn("<table class='dp-meta-table'>", html)
        self.assertNotIn("Metrics unavailable", html)
        self.assertIn("<td>Total Tokens</td><td>100</td>", html)
        self.assertIn("<td>Duration</td><td>n/a</td>", html)
        self.assertIn("<td>Throughput</td><td>n/a</td>", html)
        self.assertIn("<td>Cache Ratio</td><td>0.0%</td>", html)

    def test_real_zero_duration_renders_as_zero_not_missing(self):
        from trajectory_visualizer.insight.rendering import _format_metrics_tab

        html = _format_metrics_tab(self._metric_step(duration=0.0))

        self.assertIn("<table class='dp-meta-table'>", html)
        self.assertNotIn("Metrics unavailable", html)
        self.assertIn("<td>Duration</td><td>0.0s</td>", html)
        self.assertIn("<td>Throughput</td><td>n/a</td>", html)

    def test_zero_total_tokens_shows_na_derived_rows_without_dividing(self):
        from trajectory_visualizer.insight.rendering import _format_metrics_tab

        tokens = {
            "total": 0, "input": 0, "output": 0,
            "reasoning": 0, "cache_read": 0, "cache_write": 0,
        }
        html = _format_metrics_tab(self._metric_step(tokens=tokens))

        self.assertIn("<table class='dp-meta-table'>", html)
        self.assertIn("<td>Total Tokens</td><td>0</td>", html)
        # 0 tokens over a real duration is a genuine 0 tok/s, but 0/0 cache
        # ratio is undefined and must degrade to n/a instead of dividing.
        self.assertIn("<td>Throughput</td><td>0 tok/s</td>", html)
        self.assertIn("<td>Cache Ratio</td><td>n/a</td>", html)

    def test_codearts_marks_placeholder_breakdown_and_cache_values_unavailable(self):
        from trajectory_visualizer.insight.parser import parse_steps
        from trajectory_visualizer.insight.rendering import _format_metrics_tab

        raw = {
            "trajectory": [{
                "info": {
                    "role": "assistant",
                    "tokens": {
                        "total": 100,
                        "input": 0,
                        "output": 0,
                        "reasoning": 0,
                        "cache": {"read": 0, "write": 0},
                    },
                    "time": {"created": 1000, "completed": 3000},
                },
                "parts": [],
                "_codearts_raw": {"total_tokens": 100},
            }],
        }

        step = parse_steps(raw)[0]
        html = _format_metrics_tab(step)

        self.assertNotIn("<table", html)
        self.assertIn("Metrics unavailable", html)
        self.assertIn("The trajectory does not provide complete per-step metrics.", html)
        self.assertNotIn("CodeArts", html)
        self.assertIn("Input Tokens", html)
        self.assertIn("Cache Write", html)


if __name__ == "__main__":
    unittest.main()
