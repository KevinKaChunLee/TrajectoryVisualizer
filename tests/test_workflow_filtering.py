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
        self.all_agents = ["agent:", "agent:sub-agent"]

    def test_step_labels_are_combined_with_or(self):
        from trajectory_visualizer.insight.insight import _filter_workflow_steps

        filters = ["User", "Tool Calls", "Errors", "Reasoning", *self.all_agents]

        self.assertEqual(_filter_workflow_steps(self.steps, filters), [0, 1, 3, 4])

    def test_error_step_remains_visible_when_any_of_its_labels_is_selected(self):
        from trajectory_visualizer.insight.insight import _filter_workflow_steps

        self.assertEqual(
            _filter_workflow_steps(self.steps, ["Assistant", *self.all_agents]),
            [1, 2, 3, 4],
        )
        self.assertEqual(
            _filter_workflow_steps(self.steps, ["Tool Calls", *self.all_agents]),
            [1, 3],
        )
        self.assertEqual(
            _filter_workflow_steps(self.steps, ["Errors", *self.all_agents]),
            [3],
        )

    def test_agent_filter_is_anded_with_step_labels(self):
        from trajectory_visualizer.insight.insight import _filter_workflow_steps

        filters = ["Assistant", "User", "Tool Calls", "Errors", "Reasoning"]

        self.assertEqual(_filter_workflow_steps(self.steps, [*filters, "agent:"]), [0, 1])
        self.assertEqual(
            _filter_workflow_steps(self.steps, [*filters, "agent:sub-agent"]),
            [2, 3, 4],
        )

    def test_multi_agent_trace_with_no_agent_selected_is_empty(self):
        from trajectory_visualizer.insight.insight import _filter_workflow_steps

        self.assertEqual(_filter_workflow_steps(self.steps, ["Assistant", "User"]), [])

    def test_single_agent_trace_does_not_require_hidden_agent_filter(self):
        from trajectory_visualizer.insight.insight import _filter_workflow_steps

        main_steps = self.steps[:2]

        self.assertEqual(
            _filter_workflow_steps(main_steps, ["Assistant", "User"]),
            [0, 1],
        )

    def test_no_selected_step_label_is_empty(self):
        from trajectory_visualizer.insight.insight import _filter_workflow_steps

        self.assertEqual(_filter_workflow_steps(self.steps, self.all_agents), [])

    def test_keyword_is_anded_with_labels_and_agent(self):
        from trajectory_visualizer.insight.insight import _filter_workflow_steps

        self.steps[3]["tool_calls"][0] = {
            "tool_name": "Bash",
            "input": {"command": "pytest workflow"},
        }
        filters = ["Tool Calls", *self.all_agents]

        self.assertEqual(_filter_workflow_steps(self.steps, filters, "pytest"), [3])
        self.assertEqual(_filter_workflow_steps(self.steps, filters, "inspect"), [1])

    def test_filtered_outputs_keep_cards_count_and_toc_in_sync(self):
        from trajectory_visualizer.insight.insight import _build_filtered_workflow_outputs

        workflow, count, toc = _build_filtered_workflow_outputs(
            self.steps,
            "Errors,agent:,agent:sub-agent",
            "",
        )

        self.assertIn("wf-card-3", workflow)
        self.assertNotIn("wf-card-1", workflow)
        self.assertIn("Showing 1 of 5 steps", count)
        self.assertIn("data-step-idx='3'", toc)
        self.assertNotIn("data-step-idx='1'", toc)

    def test_filter_chips_use_delegated_show_all_action(self):
        from trajectory_visualizer.insight.rendering import render_filter_chips

        chips = render_filter_chips(agent_labels=[
            {"label": "main", "agent_id": ""},
            {"label": "sub", "agent_id": "sub-agent"},
        ])

        self.assertIn("data-wf-action='show-all'", chips)
        self.assertNotIn("onclick=", chips)
        self.assertIn("data-filter='agent:'", chips)
        self.assertIn("data-filter='agent:sub-agent'", chips)

    def test_chip_handler_updates_the_real_hidden_input_and_all_outputs(self):
        from trajectory_visualizer.insight import insight

        source = inspect.getsource(insight.build_ui)

        self.assertIn("hiddenEl.tagName === 'TEXTAREA'", source)
        self.assertIn("descriptor.set.call(hiddenEl, active.join(','))", source)
        self.assertIn("wf_filter_hidden.input(", source)
        self.assertIn("outputs=[workflow_html, wf_count_html, toc_html]", source)

    def test_hidden_selected_step_gets_a_clear_detail_message(self):
        from trajectory_visualizer.insight import insight

        source = inspect.getsource(insight.build_ui)

        self.assertIn("Selected step is hidden by the current filters", source)


if __name__ == "__main__":
    unittest.main()
