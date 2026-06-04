import unittest


def _step(
    *,
    quality_verdict="good",
    defect_flags=None,
    quality_confidence="high",
    value_tier="high",
    value_tags=None,
    value_confidence="high",
):
    return {
        "quality": {
            "verdict": quality_verdict,
            "defect_flags": defect_flags or [],
            "confidence": quality_confidence,
        },
        "value": {
            "tier": value_tier,
            "tags": value_tags or [],
            "confidence": value_confidence,
        },
    }


class TrainingFilterRuleTests(unittest.TestCase):
    def test_incorrect_quality_drops_turn(self):
        from trajectory_visualizer.insight.training_filter_rules import derive_decision

        decision = derive_decision(_step(defect_flags=["incorrect"]))

        self.assertEqual(decision["label"], "drop")
        self.assertIn("quality_incorrect", decision["reasons"])
        self.assertIn("drop_incorrect_turn", decision["matched_rules"])
        self.assertEqual(decision["policy_version"], "keepdrop.v1")

    def test_good_quality_high_value_keeps_turn(self):
        from trajectory_visualizer.insight.training_filter_rules import derive_decision

        decision = derive_decision(_step(value_tier="high"))

        self.assertEqual(decision["label"], "keep")
        self.assertIn("quality_good", decision["reasons"])
        self.assertIn("value_high", decision["reasons"])
        self.assertIn("keep_high_quality_agentic_turn", decision["matched_rules"])

    def test_flawed_quality_high_value_requires_review(self):
        from trajectory_visualizer.insight.training_filter_rules import derive_decision

        decision = derive_decision(_step(quality_verdict="flawed", value_tier="high"))

        self.assertEqual(decision["label"], "review")
        self.assertIn("quality_value_conflict", decision["reasons"])
        self.assertIn("review_conflicting_quality_and_value", decision["matched_rules"])

    def test_low_confidence_requires_review(self):
        from trajectory_visualizer.insight.training_filter_rules import derive_decision

        decision = derive_decision(_step(quality_confidence="low", value_confidence="high"))

        self.assertEqual(decision["label"], "review")
        self.assertIn("low_confidence", decision["reasons"])
        self.assertIn("review_low_confidence", decision["matched_rules"])

    def test_negative_example_does_not_auto_keep_positive_sft(self):
        from trajectory_visualizer.insight.training_filter_rules import derive_decision

        decision = derive_decision(_step(value_tier="high", value_tags=["negative_example"]))

        self.assertEqual(decision["label"], "review")
        self.assertIn("negative_example", decision["reasons"])
        self.assertIn("review_negative_example", decision["matched_rules"])


if __name__ == "__main__":
    unittest.main()
