import json
import tempfile
import unittest
from pathlib import Path


FIXTURE_DIR = Path(__file__).parent / "fixtures"
MINIMAL_FIXTURE = FIXTURE_DIR / "training_label_v2_minimal.json"
REVIEW_FIXTURE = FIXTURE_DIR / "training_label_v2_with_review.json"
TRAINING_CONVERSATION_FIXTURE = FIXTURE_DIR / "training_conversation_minimal.json"


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

    def test_schema_requires_phase_and_action_even_when_quality_value_exist(self):
        from trajectory_visualizer.insight.training_labels import load_training_labeled_json

        bad_path = FIXTURE_DIR / "training_label_v2_missing_behavior.json"
        bad_path.write_text(
            json.dumps(
                {
                    "schema_version": "trajectory_labels.v2",
                    "taxonomy_version": "v1",
                    "labeled_at": "2026-06-04T00:00:00+00:00",
                    "steps": [
                        {
                            "index": 1,
                            "role": "assistant",
                            "quality": {
                                "verdict": "good",
                                "defect_flags": [],
                                "confidence": "high",
                            },
                            "value": {
                                "tier": "high",
                                "tags": ["reasoning_pattern"],
                                "confidence": "medium",
                            },
                            "decision": {
                                "label": "keep",
                                "reasons": ["quality_good", "value_high"],
                                "matched_rules": ["keep_high_quality_agentic_turn"],
                                "policy_version": "keepdrop.v1",
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        try:
            with self.assertRaises(ValueError):
                load_training_labeled_json(str(bad_path))
        finally:
            bad_path.unlink(missing_ok=True)

    def test_aggregate_training_labels_summarizes_core_counts(self):
        from trajectory_visualizer.insight.training_labels import (
            aggregate_training_labels,
            load_training_labeled_json,
        )

        data = load_training_labeled_json(str(REVIEW_FIXTURE))
        agg = aggregate_training_labels(data)

        self.assertEqual(agg["total"], 1)
        self.assertEqual(agg["quality_verdict_counts"], {"flawed": 1})
        self.assertEqual(agg["quality_flag_counts"], {"incomplete": 1})
        self.assertEqual(agg["value_tier_counts"], {"high": 1})
        self.assertEqual(agg["value_tag_counts"], {"successful_recovery": 1})
        self.assertEqual(agg["decision_counts"], {"review": 1})
        self.assertEqual(agg["phase_counts"], {"debug": 1})
        self.assertEqual(agg["action_counts"], {"debug_hypothesis_test": 1})

    def test_training_labeler_writes_additive_v2_output(self):
        from scripts.training_labeler import label_training_trajectory
        from trajectory_visualizer.insight.training_labels import load_training_labeled_json

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            output_path = Path(f.name)

        try:
            label_training_trajectory(
                str(TRAINING_CONVERSATION_FIXTURE),
                str(output_path),
                behavior_labeler=lambda step, context: {
                    "phase": "implement",
                    "action": "implement_runtime_logic",
                },
                quality_labeler=lambda step, context: {
                    "verdict": "good",
                    "defect_flags": [],
                    "confidence": "high",
                    "decision": {"label": "drop"},
                },
                value_labeler=lambda step, context: {
                    "tier": "high",
                    "tags": ["tool_use_pattern"],
                    "confidence": "medium",
                    "decision": {"label": "drop"},
                },
                behavior_model="test-behavior",
                quality_model="test-quality",
                value_model="test-value",
            )

            data = load_training_labeled_json(str(output_path))
        finally:
            output_path.unlink(missing_ok=True)

        self.assertEqual(data["schema_version"], "trajectory_labels.v2")
        self.assertEqual(data["behavior_model"], "test-behavior")
        self.assertEqual(data["quality_model"], "test-quality")
        self.assertEqual(data["value_model"], "test-value")
        self.assertEqual(data["quality_label_version"], "quality.v1")
        self.assertEqual(data["value_label_version"], "value.v1")
        self.assertEqual(data["decision_policy_version"], "keepdrop.v1")

        self.assertEqual(len(data["steps"]), 2)
        step = data["steps"][0]
        self.assertEqual(step["phase"], "implement")
        self.assertEqual(step["action"], "implement_runtime_logic")
        self.assertNotIn("behavior", step)
        self.assertEqual(step["quality"], {
            "verdict": "good",
            "defect_flags": [],
            "confidence": "high",
        })
        self.assertEqual(step["value"], {
            "tier": "high",
            "tags": ["tool_use_pattern"],
            "confidence": "medium",
        })
        self.assertEqual(step["decision"]["label"], "keep")

    def test_training_labeler_derives_decision_instead_of_copying_model_output(self):
        from scripts.training_labeler import label_training_trajectory
        from trajectory_visualizer.insight.training_labels import load_training_labeled_json

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            output_path = Path(f.name)

        try:
            label_training_trajectory(
                str(TRAINING_CONVERSATION_FIXTURE),
                str(output_path),
                behavior_labeler=lambda step, context: {
                    "phase": "debug",
                    "action": "debug_hypothesis_test",
                },
                quality_labeler=lambda step, context: {
                    "verdict": "reject",
                    "defect_flags": [],
                    "confidence": "high",
                    "decision": {"label": "keep"},
                },
                value_labeler=lambda step, context: {
                    "tier": "high",
                    "tags": [],
                    "confidence": "high",
                    "decision": {"label": "keep"},
                },
            )

            data = load_training_labeled_json(str(output_path))
        finally:
            output_path.unlink(missing_ok=True)

        step = data["steps"][0]
        self.assertEqual(step["decision"]["label"], "drop")
        self.assertIn("drop_rejected_turn", step["decision"]["matched_rules"])


if __name__ == "__main__":
    unittest.main()
