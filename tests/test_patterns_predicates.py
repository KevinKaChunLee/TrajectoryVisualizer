"""The pattern module's predicates must agree with each other and say what they mean.

Four defects, all of them in `trajviz/insight/patterns.py`, all verified against
the 2,500-trajectory corpus as moving no published number:

* The plan vocabulary was written twice. `_PLAN_TOOL_NAMES` matched exact
  spellings for the phase classifier, while `extract_plan_history` matched its
  own lowercase tuple, so `TodoUpdate` set the plan phase but produced no plan
  history and `todo_write` did the exact opposite. No corpus export emits either
  spelling (only `TodoWrite` and `todowrite` occur), so unifying them is
  behaviour-preserving on every published figure and only fixes the latent case.
* `detect_phase_anomalies` called a span fraction `confidence`: it is the share
  of the trajectory the regressed phase covers, so it *falls* as the evidence
  grows. Nothing reads the key, so the honest name is free.
* `patterns.compute_autonomy_ratio` was dead, and disagreed with the live
  definition in `metrics.py` (which divides by user+assistant turns, not by all
  steps — the corpus has a third `developer` role, so the two differ on codex).

The fourth, `_is_fruitless_step` treating a missing output as an empty one, is
characterised here rather than fixed: the fix would move `fruitless_streaks`
and the wasted-step total that the paper reports. See the comment at its
definition.
"""

from __future__ import annotations

import unittest

from trajviz.insight import patterns
from trajviz.insight.metrics import build_message_metrics, compute_metrics
from trajviz.insight.patterns import (
    classify_structural_phase,
    detect_fruitless_streaks,
    detect_phase_anomalies,
    extract_plan_history,
)


def _step(index: int, role: str = "assistant", calls: list[dict] | None = None, **extra) -> dict:
    """One parsed step, carrying only the keys the predicates under test read."""
    step = {
        "index": index,
        "role": role,
        "tool_calls": calls or [],
        "tool_call_count": len(calls or []),
        "tokens": {"total": 0, "input": 0, "output": 0,
                   "reasoning": 0, "cache_read": 0, "cache_write": 0},
        "parts": [],
        "finish": "stop",
    }
    step.update(extra)
    return step


def _plan_call(tool_name: str, *contents: str) -> dict:
    todos = [{"content": c, "status": "in_progress"} for c in contents]
    return {"tool_name": tool_name, "status": "success", "input": {"todos": todos}, "output": "ok"}


class PlanToolVocabularyIsShared(unittest.TestCase):
    """The phase classifier and extract_plan_history must accept the same names."""

    def test_todoupdate_produces_plan_history(self):
        """Set the plan phase but drop the snapshot: the original disagreement."""
        steps = [_step(0, calls=[_plan_call("TodoUpdate", "write the fix")])]
        self.assertEqual(classify_structural_phase(steps[0]), "plan")
        history = extract_plan_history(steps)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["items"], [{"content": "write the fix", "status": "in_progress"}])

    def test_snake_case_todo_write_sets_the_plan_phase(self):
        """The disagreement in the other direction: history but no plan phase."""
        steps = [_step(0, calls=[_plan_call("todo_write", "write the fix")])]
        self.assertEqual(classify_structural_phase(steps[0]), "plan")
        self.assertEqual(len(extract_plan_history(steps)), 1)

    def test_lowercase_spellings_are_recognised(self):
        """A future export may lowercase the name; one vocabulary, case-folded."""
        for name in ("todoupdate", "tasklist", "enterplanmode", "taskcreate"):
            with self.subTest(name=name):
                self.assertEqual(classify_structural_phase(_step(0, calls=[_plan_call(name)])), "plan")

    def test_plan_mode_tool_without_todos_adds_no_snapshot(self):
        """Guard against over-collection: unifying the names must not invent plans."""
        steps = [_step(0, calls=[{"tool_name": "EnterPlanMode", "status": "success", "input": {}}])]
        self.assertEqual(classify_structural_phase(steps[0]), "plan")
        self.assertEqual(extract_plan_history(steps), [])


class PhaseAnomalySpanIsNotConfidence(unittest.TestCase):
    def test_phase_anomaly_reports_span_not_confidence(self):
        steps = [_step(i) for i in range(10)]
        phases = [
            {"name": "validate", "start_idx": 0, "end_idx": 4},
            {"name": "implement", "start_idx": 5, "end_idx": 9},
        ]
        anomalies = detect_phase_anomalies(steps, phases)
        self.assertEqual(len(anomalies), 1)
        entry = anomalies[0]
        self.assertEqual(entry["span_fraction"], 0.5)
        self.assertNotIn("confidence", entry)
        # The prose already described it correctly; keep the two in step.
        self.assertIn("50.0% of trajectory", entry["explanation"])

    def test_span_fraction_grows_with_the_regression_not_with_certainty(self):
        """Pins why the old name was wrong: a longer regression scores higher."""
        short = detect_phase_anomalies(
            [_step(i) for i in range(10)],
            [{"name": "validate", "start_idx": 0, "end_idx": 7},
             {"name": "implement", "start_idx": 8, "end_idx": 9}],
        )
        self.assertEqual(short[0]["span_fraction"], 0.2)


class AutonomyRatioHasOneDefinition(unittest.TestCase):
    def test_patterns_no_longer_defines_autonomy_ratio(self):
        self.assertFalse(
            hasattr(patterns, "compute_autonomy_ratio"),
            "patterns.compute_autonomy_ratio is dead and uses a different denominator "
            "than the live metrics.py definition",
        )

    def test_live_definition_stays_a_fraction_with_a_third_role(self):
        """`developer` steps exist in the corpus; the live formula ignores them."""
        steps = [
            _step(0, role="user"),
            _step(1, role="assistant"),
            _step(2, role="developer"),
            _step(3, role="assistant"),
        ]
        ratio = compute_metrics(steps, {}, build_message_metrics(steps))["autonomy_ratio"]
        self.assertTrue(0.0 <= ratio <= 1.0)
        self.assertEqual(ratio, round(2 / 3, 4))


class FruitlessSearchTreatsMissingOutputAsEmpty(unittest.TestCase):
    """Characterisation, NOT a fix: `fruitless_streaks` is a published number.

    `_is_fruitless_step` cannot distinguish "the search returned nothing" from
    "the call never resolved, so nothing was recorded" — both read as empty.
    `parse_steps` always materialises `output`, so the only real exposure is an
    unresolved or failed search call (cursor's `status: unknown`, three opencode
    `status: error` steps). Tightening the predicate would shorten streaks and
    the wasted-step total the Overview reports, so it stays, pinned here.
    """

    def test_unresolved_search_calls_still_count_as_fruitless(self):
        steps = [
            _step(i, calls=[{"tool_name": "Grep", "status": "unknown", "input": {"pattern": "x"}}])
            for i in range(3)
        ]
        streaks = detect_fruitless_streaks(steps)
        self.assertEqual([s["length"] for s in streaks], [3])

    def test_failed_search_calls_still_count_as_fruitless(self):
        steps = [
            _step(i, calls=[{"tool_name": "Grep", "status": "error", "input": {"pattern": "x"}}])
            for i in range(3)
        ]
        self.assertEqual([s["length"] for s in detect_fruitless_streaks(steps)], [3])


if __name__ == "__main__":
    unittest.main()
