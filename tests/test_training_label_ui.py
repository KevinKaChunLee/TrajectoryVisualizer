import json
import tempfile
import unittest
from pathlib import Path


FIXTURE_DIR = Path(__file__).parent / "fixtures"
TRAINING_LABEL_FIXTURE = FIXTURE_DIR / "training_label_v2_minimal.json"
TRAINING_CONVERSATION_FIXTURE = FIXTURE_DIR / "training_conversation_minimal.json"


class TrainingLabelUiTests(unittest.TestCase):
    def test_training_v2_label_file_is_recognized_separately(self):
        from trajectory_visualizer.insight.insight import build_label_ui_payload

        payload = build_label_ui_payload(str(TRAINING_LABEL_FIXTURE))

        self.assertEqual(payload["kind"], "training_v2")
        self.assertIn("Training labels loaded", payload["badge_html"])
        self.assertIn("Quality", payload["status_html"])
        self.assertIn("Value", payload["status_html"])
        self.assertIn("Decision", payload["status_html"])

    def test_training_v2_label_summary_shows_quality_value_decision_counts(self):
        from trajectory_visualizer.insight.insight import build_label_ui_payload

        payload = build_label_ui_payload(str(TRAINING_LABEL_FIXTURE))

        self.assertIn("good: 1", payload["status_html"])
        self.assertIn("high: 1", payload["status_html"])
        self.assertIn("keep: 1", payload["status_html"])
        self.assertIn("tool_use_pattern: 1", payload["status_html"])
        self.assertGreater(len(payload["phase_count_fig"].data), 0)
        self.assertGreater(len(payload["action_count_fig"].data), 0)

    def test_training_v2_timeline_uses_tokens_instead_of_missing_duration(self):
        from trajectory_visualizer.insight.insight import build_label_ui_payload

        payload = build_label_ui_payload(str(TRAINING_LABEL_FIXTURE))

        self.assertEqual(list(payload["timeline_fig"].data[0].x), [1234])
        self.assertEqual(payload["timeline_fig"].layout.xaxis.title.text, "Estimated tokens")
        self.assertIn("Token Length", payload["timeline_fig"].layout.title.text)

    def test_training_v2_label_charts_use_compact_layout(self):
        from trajectory_visualizer.insight.insight import build_label_ui_payload

        payload = build_label_ui_payload(str(TRAINING_LABEL_FIXTURE))
        chart_keys = [
            "phase_count_fig",
            "action_count_fig",
            "phase_duration_fig",
            "action_duration_fig",
        ]

        for key in chart_keys:
            with self.subTest(chart=key):
                fig = payload[key]
                self.assertLessEqual(fig.layout.height, 300)
                self.assertLessEqual(fig.layout.margin.t, 52)
                self.assertLessEqual(fig.layout.margin.r, 32)
                self.assertEqual(fig.data[0].textposition, "inside")

        timeline = payload["timeline_fig"]
        self.assertLessEqual(timeline.layout.margin.r, 40)
        self.assertEqual(timeline.data[0].textposition, "inside")
        self.assertTrue(timeline.data[0].cliponaxis)

    def test_training_v2_token_timeline_keeps_bar_text_short(self):
        from trajectory_visualizer.insight.insight import build_label_ui_payload

        payload = build_label_ui_payload(str(TRAINING_LABEL_FIXTURE))
        timeline = payload["timeline_fig"]

        self.assertEqual(list(timeline.data[0].text), ["1,234 tok"])
        self.assertIn("implement_runtime_logic", timeline.layout.yaxis.ticktext[0])
        self.assertIn("tool_use_pattern", timeline.data[0].hovertext[0])

    def test_legacy_behavior_label_file_remains_legacy(self):
        from trajectory_visualizer.insight.insight import build_label_ui_payload

        legacy = {
            "trajectory_file": "/tmp/example.json",
            "taxonomy_version": "v1",
            "model": "test-model",
            "labeled_at": "2026-06-04T00:00:00+00:00",
            "steps": [
                {
                    "index": 1,
                    "role": "assistant",
                    "phase": "debug",
                    "action": "debug_hypothesis_test",
                    "duration_s": 2.5,
                    "tokens_total": 100,
                    "tool_calls": [],
                    "text_preview": "Test a hypothesis.",
                }
            ],
        }

        with tempfile.NamedTemporaryFile("w", suffix="_labeled.json", delete=False, encoding="utf-8") as f:
            json.dump(legacy, f)
            path = Path(f.name)

        try:
            payload = build_label_ui_payload(str(path))
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(payload["kind"], "legacy")
        self.assertIn("Labels loaded", payload["badge_html"])
        self.assertNotIn("Training labels loaded", payload["badge_html"])
        self.assertEqual(payload["phase_count_fig"].layout.title.text, "Step Count by Phase")
        self.assertEqual(list(payload["timeline_fig"].data[0].x), [2.5])
        self.assertEqual(payload["timeline_fig"].layout.xaxis.title.text, "Duration (s)")

    def test_behavior_v2_with_user_steps_is_not_misread_as_training_labels(self):
        from trajectory_visualizer.insight.insight import build_label_ui_payload

        behavior_v2 = {
            "schema_version": "trajectory_labels.v2",
            "trajectory_file": str(TRAINING_CONVERSATION_FIXTURE.resolve()),
            "taxonomy_version": "v1",
            "model": "test-model",
            "labeled_at": "2026-07-20T00:00:00+00:00",
            "defaults": {
                "user": {"phase": "user", "action": "user_prompt"},
                "other": {"phase": "unknown", "action": "unknown"},
            },
            "counts": {"total": 2, "assistant": 1, "user": 1},
            "steps": [
                {
                    "index": 1,
                    "raw_index": 1,
                    "role": "user",
                    "phase": "user",
                    "action": "user_prompt",
                    "label_source": "default",
                    "duration_s": 0,
                    "tokens_total": 0,
                    "text_preview": "Please inspect the project.",
                },
                {
                    "index": 2,
                    "raw_index": 2,
                    "role": "assistant",
                    "phase": "understand",
                    "action": "repository_exploration",
                    "label_source": "model",
                    "duration_s": 1.5,
                    "tokens_total": 100,
                    "text_preview": "I will inspect the repository.",
                },
            ],
        }

        with tempfile.NamedTemporaryFile("w", suffix="_labeled_v2.json", delete=False, encoding="utf-8") as f:
            json.dump(behavior_v2, f)
            path = Path(f.name)

        try:
            payload = build_label_ui_payload(str(path))
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(payload["kind"], "behavior_v2")
        self.assertIn("Labels loaded", payload["badge_html"])
        self.assertNotIn("Training labels loaded", payload["badge_html"])
        # The source trajectory also contains user step 1.  It must not be
        # appended a second time, while its other user step is still included.
        self.assertEqual(list(payload["timeline_fig"].layout.yaxis.ticktext), ["1", "2", "3"])
        self.assertEqual(
            [annotation.text for annotation in payload["timeline_fig"].layout.annotations],
            ["<i>user prompt</i>", "<i>user prompt</i>"],
        )
        self.assertNotIn(
            "Please inspect the project.",
            "".join(annotation.text for annotation in payload["timeline_fig"].layout.annotations),
        )

    def test_training_labels_can_be_attached_to_workflow_steps(self):
        from trajectory_visualizer.insight.insight import attach_training_labels_to_steps
        from trajectory_visualizer.insight.loaders import load_trajectory
        from trajectory_visualizer.insight.parser import parse_steps
        from trajectory_visualizer.insight.rendering import format_step_detail, render_workflow_html
        from trajectory_visualizer.insight.training_labels import load_training_labeled_json

        raw = load_trajectory(str(TRAINING_CONVERSATION_FIXTURE))
        steps = parse_steps(raw)
        first_assistant_idx = next(s["index"] for s in steps if s.get("role") == "assistant")
        labels = load_training_labeled_json(str(TRAINING_LABEL_FIXTURE))
        labels["steps"][0]["index"] = first_assistant_idx

        enriched = attach_training_labels_to_steps(steps, labels)
        assistant = next(s for s in enriched if s.get("index") == first_assistant_idx)

        self.assertEqual(assistant["training_label"]["quality"]["verdict"], "good")
        workflow_html = render_workflow_html(enriched)
        detail_html = format_step_detail(assistant)

        self.assertIn("Q good", workflow_html)
        self.assertIn("V high", workflow_html)
        self.assertIn("keep", workflow_html)
        self.assertIn("Training Labels", detail_html)
        self.assertIn("tool_use_pattern", detail_html)


if __name__ == "__main__":
    unittest.main()
