import json
import unittest
from pathlib import Path


FIXTURE_DIR = Path(__file__).parent / "fixtures"
MINIMAL_FIXTURE = FIXTURE_DIR / "training_label_v2_minimal.json"
REVIEW_FIXTURE = FIXTURE_DIR / "training_label_v2_with_review.json"


class TrainingLabelSchemaTests(unittest.TestCase):
    def test_minimal_fixture_uses_additive_v2_schema(self):
        from trajectory_visualizer.insight.training_labels import load_training_labeled_json

        data = load_training_labeled_json(str(MINIMAL_FIXTURE))
        step = data["steps"][0]

        self.assertEqual(data["schema_version"], "trajectory_labels.v2")
        self.assertEqual(data["taxonomy_version"], "v1")
        self.assertIn("steps", data)

        self.assertEqual(step["role"], "assistant")
        self.assertIn("phase", step)
        self.assertIn("action", step)

        self.assertIn("quality", step)
        self.assertEqual(set(step["quality"].keys()), {"verdict", "defect_flags", "confidence"})

        self.assertIn("value", step)
        self.assertEqual(set(step["value"].keys()), {"tier", "tags", "confidence"})

        self.assertIn("decision", step)
        self.assertEqual(
            set(step["decision"].keys()),
            {"label", "reasons", "matched_rules", "policy_version"},
        )

    def test_review_fixture_preserves_top_level_behavior_fields(self):
        from trajectory_visualizer.insight.training_labels import load_training_labeled_json

        data = load_training_labeled_json(str(REVIEW_FIXTURE))
        step = data["steps"][0]

        self.assertEqual(step["phase"], "debug")
        self.assertEqual(step["action"], "debug_hypothesis_test")
        self.assertNotIn("behavior", step)

    def test_schema_requires_steps_array(self):
        from trajectory_visualizer.insight.training_labels import load_training_labeled_json

        bad_path = FIXTURE_DIR / "training_label_v2_missing_steps.json"
        bad_path.write_text(
            json.dumps(
                {
                    "schema_version": "trajectory_labels.v2",
                    "taxonomy_version": "v1",
                    "labeled_at": "2026-06-04T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        try:
            with self.assertRaises(ValueError):
                load_training_labeled_json(str(bad_path))
        finally:
            bad_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
