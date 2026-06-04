import json
import tempfile
import unittest
from pathlib import Path


FIXTURE_DIR = Path(__file__).parent / "fixtures"
TRAINING_LABEL_FIXTURE = FIXTURE_DIR / "training_label_v2_minimal.json"


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


if __name__ == "__main__":
    unittest.main()
