import inspect
import unittest
from pathlib import Path


class WorkflowDetailUiTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
