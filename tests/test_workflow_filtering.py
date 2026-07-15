import inspect
import unittest


def _step(
    index,
    *,
    role="assistant",
    agent="",
    tool=False,
    error=False,
    reasoning=False,
    text="",
    tool_name="Read",
    tool_input=None,
):
    tool_calls = []
    if tool:
        tool_calls.append({
            "tool_name": tool_name,
            "input": tool_input or {},
        })
    return {
        "index": index,
        "role": role,
        "agent": agent,
        "is_sub_agent": bool(agent),
        "session_id": "",
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "error_count": int(error),
        "has_reasoning": reasoning,
        "text_preview": text,
        "duration": None,
        "parts": [],
        "tokens": {
            "total": 0,
            "input": 0,
            "output": 0,
            "reasoning": 0,
            "cache_read": 0,
            "cache_write": 0,
        },
    }


class WorkflowFilteringTests(unittest.TestCase):
    def setUp(self):
        self.steps = [
            _step(0, role="user", text="build the dashboard"),
            _step(1, tool=True, text="inspect files"),
            _step(2, agent="sub-agent", text="plain assistant response"),
            _step(3, agent="sub-agent", tool=True, error=True, text="command failed"),
            _step(4, agent="sub-agent", reasoning=True, text="consider options"),
        ]

    def test_roles_are_combined_with_or(self):
        from trajectory_visualizer.insight.insight import _filter_workflow_steps

        self.assertEqual(
            _filter_workflow_steps(self.steps, ["Assistant", "User", "All"]),
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            _filter_workflow_steps(self.steps, ["Assistant", "All"]),
            [1, 2, 3, 4],
        )
        self.assertEqual(
            _filter_workflow_steps(self.steps, ["User", "All"]),
            [0],
        )

    def test_features_are_combined_with_or_inside_selected_roles(self):
        from trajectory_visualizer.insight.insight import _filter_workflow_steps

        self.assertEqual(
            _filter_workflow_steps(
                self.steps,
                ["Assistant", "Tool Calls", "Reasoning"],
            ),
            [1, 3, 4],
        )

    def test_role_and_feature_groups_are_combined_with_and(self):
        from trajectory_visualizer.insight.insight import _filter_workflow_steps

        self.assertEqual(
            _filter_workflow_steps(self.steps, ["Assistant", "Tool Calls"]),
            [1, 3],
        )
        self.assertEqual(
            _filter_workflow_steps(self.steps, ["User", "Tool Calls"]),
            [],
        )

    def test_all_or_an_omitted_feature_selection_has_no_feature_predicate(self):
        from trajectory_visualizer.insight.insight import _filter_workflow_steps

        expected = [1, 2, 3, 4]
        self.assertEqual(
            _filter_workflow_steps(self.steps, ["Assistant", "All"]),
            expected,
        )
        self.assertEqual(
            _filter_workflow_steps(self.steps, ["Assistant"]),
            expected,
        )

    def test_no_selected_role_is_empty(self):
        from trajectory_visualizer.insight.insight import _filter_workflow_steps

        self.assertEqual(
            _filter_workflow_steps(self.steps, ["All", "Tool Calls"]),
            [],
        )

    def test_agent_tokens_do_not_apply_a_hidden_filter(self):
        from trajectory_visualizer.insight.insight import _filter_workflow_steps

        self.assertEqual(
            _filter_workflow_steps(
                self.steps,
                ["Assistant", "All", "agent:sub-agent"],
            ),
            [1, 2, 3, 4],
        )

    def test_keyword_is_anded_with_role_and_features(self):
        from trajectory_visualizer.insight.insight import _filter_workflow_steps

        self.steps[3]["tool_calls"][0] = {
            "tool_name": "Bash",
            "input": {"command": "pytest workflow"},
        }
        filters = ["Assistant", "Tool Calls"]

        self.assertEqual(_filter_workflow_steps(self.steps, filters, "pytest"), [3])
        self.assertEqual(_filter_workflow_steps(self.steps, filters, "inspect"), [1])

    def test_filtered_outputs_keep_cards_count_and_toc_in_sync(self):
        from trajectory_visualizer.insight.insight import _build_filtered_workflow_outputs

        workflow, count, toc = _build_filtered_workflow_outputs(
            self.steps,
            "Assistant,Errors",
            "",
        )

        self.assertIn("wf-card-3", workflow)
        self.assertNotIn("wf-card-1", workflow)
        self.assertIn("Showing 1 of 5 steps", count)
        self.assertIn("data-step-idx='3'", toc)
        self.assertNotIn("data-step-idx='1'", toc)

    def test_filter_chips_render_two_explicit_groups_without_agents(self):
        from trajectory_visualizer.insight.rendering import render_filter_chips

        chips = render_filter_chips()

        self.assertIn("data-filter-group-container='role'", chips)
        self.assertIn("data-filter-group-container='feature'", chips)
        self.assertIn("data-filter='All'", chips)
        self.assertIn("data-wf-action='reset-filters'", chips)
        self.assertIn("select at least one", chips)
        self.assertIn("match any selected", chips)
        self.assertNotIn("agent:", chips)
        self.assertNotIn("Clear all", chips)
        self.assertNotIn("onclick=", chips)

    def test_default_chips_select_both_roles_and_all_only(self):
        from trajectory_visualizer.insight.rendering import render_filter_chips

        chips = render_filter_chips()

        for name in ("Assistant", "User", "All"):
            start = chips.index(f"data-filter='{name}'")
            button_start = chips.rfind("<button", 0, start)
            self.assertIn("chip-active", chips[button_start:start])
        for name in ("Tool Calls", "Errors", "Reasoning"):
            start = chips.index(f"data-filter='{name}'")
            button_start = chips.rfind("<button", 0, start)
            self.assertNotIn("chip-active", chips[button_start:start])

    def test_chip_handler_enforces_required_roles_and_mutually_exclusive_all(self):
        from trajectory_visualizer.insight import insight

        source = inspect.getsource(insight.build_ui)

        self.assertIn("selectedRoles.length === 1", source)
        self.assertIn("chip.dataset.filter === 'All'", source)
        self.assertIn("selectedFeatures.length === 0", source)
        self.assertIn("data-wf-action=\"reset-filters\"", source)
        self.assertIn("c.dataset.filter === 'All'", source)
        self.assertIn("window.__updateWorkflowFilterQuery(root)", source)

    def test_chip_handler_updates_the_real_hidden_input_and_all_outputs(self):
        from trajectory_visualizer.insight import insight

        source = inspect.getsource(insight.build_ui)

        hidden_filter_source = source[source.index("wf_filter_hidden = gr.Textbox("):]
        hidden_filter_source = hidden_filter_source[:hidden_filter_source.index("wf_count_html")]
        self.assertIn("visible=True", hidden_filter_source)
        self.assertIn("hiddenEl.tagName === 'TEXTAREA'", source)
        self.assertIn("descriptor.set.call(hiddenEl, active.join(','))", source)
        self.assertIn("new InputEvent('input'", source)
        self.assertIn("new Event('change'", source)
        self.assertIn("wf_filter_hidden.change(", source)
        self.assertIn("outputs=[workflow_html, wf_count_html, toc_html]", source)

    def test_hidden_filter_bridge_is_visually_hidden_with_css(self):
        from pathlib import Path

        styles = Path("trajectory_visualizer/insight/styles.py").read_text()

        start = styles.index("#wf-filter-hidden")
        bridge_css = styles[start:styles.index(".filter-summary {", start)]
        self.assertIn("display: none !important", bridge_css)

    def test_hidden_selected_step_gets_a_clear_detail_message(self):
        from trajectory_visualizer.insight import insight

        source = inspect.getsource(insight.build_ui)

        self.assertIn("Selected step is hidden by the current filters", source)


if __name__ == "__main__":
    unittest.main()
