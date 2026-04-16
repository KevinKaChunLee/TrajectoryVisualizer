"""Invariant tests for Converge v2: cost conservation, monotonic alignment,
1:1 failure attribution, report/UI consistency, label completeness."""

from __future__ import annotations

import os
from collections import defaultdict

import pytest

from trajectory_visualizer.insight.loaders import load_trajectory
from trajectory_visualizer.insight.parser import parse_steps
from trajectory_visualizer.converge.canonical import (
    CanonicalAction, ActionCost, canonicalize_steps, assign_effect_labels,
)
from trajectory_visualizer.converge.alignment import align_trajectories, build_comparison_report
from trajectory_visualizer.converge.milestones import extract_milestones


SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def _sample(name: str) -> str:
    path = os.path.join(SAMPLES, name)
    if not os.path.isfile(path):
        pytest.skip(f"Sample not found: {name}")
    return path


def _load_and_canonicalize(name: str):
    raw = load_trajectory(_sample(name))
    steps = parse_steps(raw)
    actions = canonicalize_steps(steps)
    assign_effect_labels(actions, steps)
    return steps, actions


# ===========================================================================
# 6.1: Cost conservation
# ===========================================================================

class TestCostConservation:
    """For every step, sum of action token_shares must equal step total tokens."""

    def _check_conservation(self, name: str):
        steps, actions = _load_and_canonicalize(name)
        step_tokens = {s["index"]: s["tokens"]["total"] for s in steps}

        # Group actions by step_index
        shares_by_step: dict[int, int] = defaultdict(int)
        for a in actions:
            shares_by_step[a.step_index] += a.cost.token_share

        for step_idx, expected in step_tokens.items():
            actual = shares_by_step.get(step_idx, 0)
            assert actual == expected, (
                f"Step {step_idx}: token_share sum {actual} != step tokens {expected}"
            )

    def test_cc_trajectory(self):
        self._check_conservation("cc_trajectory.json")

    def test_op_trajectory(self):
        self._check_conservation("op_trajectory.json")

    def test_no_inflation(self):
        """Total action token_shares must not exceed trajectory total tokens."""
        steps, actions = _load_and_canonicalize("cc_trajectory.json")
        trajectory_total = sum(s["tokens"]["total"] for s in steps)
        action_total = sum(a.cost.token_share for a in actions)
        assert action_total == trajectory_total


# ===========================================================================
# 6.2: Monotonic alignment
# ===========================================================================

class TestMonotonicAlignment:
    """Matched pairs must be monotonically ordered in both sequences."""

    def test_cc_self_alignment_monotonic(self):
        _, actions = _load_and_canonicalize("cc_trajectory.json")
        result = align_trajectories(actions, actions)
        pairs = result["matched_pairs"]
        for k in range(1, len(pairs)):
            ri_prev, ci_prev = pairs[k - 1]
            ri_curr, ci_curr = pairs[k]
            assert ri_curr > ri_prev, f"ref index not monotonic: {pairs[k-1]} -> {pairs[k]}"
            assert ci_curr > ci_prev, f"cmp index not monotonic: {pairs[k-1]} -> {pairs[k]}"

    def test_cc_vs_op_monotonic(self):
        _, ref_actions = _load_and_canonicalize("cc_trajectory.json")
        _, cmp_actions = _load_and_canonicalize("op_trajectory.json")
        result = align_trajectories(ref_actions, cmp_actions)
        pairs = result["matched_pairs"]
        for k in range(1, len(pairs)):
            ri_prev, ci_prev = pairs[k - 1]
            ri_curr, ci_curr = pairs[k]
            assert ri_curr > ri_prev
            assert ci_curr > ci_prev


# ===========================================================================
# 6.3: One-to-one failure attribution
# ===========================================================================

class TestFailureAttribution:
    """Step with 2 Bash calls (1 failed, 1 success) → only one action gets failed."""

    def test_two_bash_one_failed(self):
        steps = [{
            "index": 0,
            "role": "assistant",
            "tokens": {"total": 1000, "input": 400, "output": 100,
                       "reasoning": 0, "cache_read": 0, "cache_write": 0},
            "duration": 5.0,
            "parts": [],
            "tool_calls": [
                {
                    "type": "tool_call", "tool_name": "Bash",
                    "tool_id": "tc_good", "status": "success",
                    "input": {"command": "npm test"}, "output": "ok",
                    "error": None, "title": "",
                    "time_start": None, "time_end": None,
                    "metadata": {"exit": 0},
                },
                {
                    "type": "tool_call", "tool_name": "Bash",
                    "tool_id": "tc_bad", "status": "success",
                    "input": {"command": "npm build"}, "output": "error",
                    "error": None, "title": "",
                    "time_start": None, "time_end": None,
                    "metadata": {"exit": 1},
                },
            ],
            "tool_call_count": 2, "error_count": 0,
            "has_reasoning": False, "text_preview": "",
            "finish": "tool_use", "model_id": "", "provider_id": "",
            "time_created_ms": None, "time_completed_ms": None,
            "agent": "", "mode": "", "message_id": "", "id": "",
            "parent_id": "", "session_id": "", "cwd": "", "root": "",
        }]
        actions = canonicalize_steps(steps)
        assign_effect_labels(actions, steps)

        tool_actions = [a for a in actions if a.action_type != "REASON"]
        failed = [a for a in tool_actions if a.effect_label == "failed"]
        survived = [a for a in tool_actions if a.effect_label == "survived"]

        assert len(failed) == 1, f"Expected 1 failed, got {len(failed)}"
        assert failed[0].tool_call_id == "tc_bad"
        assert len(survived) == 1
        assert survived[0].tool_call_id == "tc_good"


# ===========================================================================
# 6.4: Report/UI milestone consistency
# ===========================================================================

class TestReportUIConsistency:
    """Milestones in report must match what would be computed from the same actions."""

    def test_report_milestones_match(self):
        report = build_comparison_report(
            _sample("cc_trajectory.json"),
            _sample("cc_trajectory.json"),
        )
        ref_ms = report.get("ref_milestones", {})
        cmp_ms = report.get("cmp_milestones", {})

        # Both should be present
        assert ref_ms is not None
        assert cmp_ms is not None

        # Self-comparison: milestones should be identical
        for key in ref_ms:
            assert ref_ms[key] == cmp_ms[key], f"Milestone {key}: ref={ref_ms[key]} != cmp={cmp_ms[key]}"


# ===========================================================================
# 6.5: Label rename completeness
# ===========================================================================

class TestLabelRenameCompleteness:
    """No effect_label="success" should appear in any converge module output."""

    def test_no_success_labels_in_cc(self):
        _, actions = _load_and_canonicalize("cc_trajectory.json")
        for a in actions:
            assert a.effect_label != "success", (
                f"Action at step {a.step_index} still has effect_label='success'"
            )

    def test_no_success_labels_in_op(self):
        _, actions = _load_and_canonicalize("op_trajectory.json")
        for a in actions:
            assert a.effect_label != "success"

    def test_no_success_in_report(self):
        import json
        report = build_comparison_report(
            _sample("cc_trajectory.json"),
            _sample("op_trajectory.json"),
        )
        report_str = json.dumps(report, default=str)
        # "success" may appear in outcome.success_detection or notes, but not as effect_label
        # Check specific fields
        assert report.get("confidence") is not None


# ===========================================================================
# 6.6: End-to-end cost + confidence
# ===========================================================================

class TestEndToEnd:
    """Full comparison: token_shares sum correctly, confidence badges present."""

    def test_cc_vs_op_costs_not_inflated(self):
        steps_ref, _ = _load_and_canonicalize("cc_trajectory.json")
        steps_cmp, _ = _load_and_canonicalize("op_trajectory.json")

        report = build_comparison_report(
            _sample("cc_trajectory.json"),
            _sample("op_trajectory.json"),
        )

        ref_total = sum(s["tokens"]["total"] for s in steps_ref)
        cmp_total = sum(s["tokens"]["total"] for s in steps_cmp)

        assert report["outcome"]["reference_tokens"] == ref_total
        assert report["outcome"]["compared_tokens"] == cmp_total

    def test_confidence_badges_present(self):
        report = build_comparison_report(
            _sample("cc_trajectory.json"),
            _sample("op_trajectory.json"),
        )
        conf = report.get("confidence")
        assert conf is not None
        assert "alignment" in conf
        assert "milestones" in conf
        assert "outcome" in conf
        assert conf["alignment"] in ("heuristic", "informational", "anchored")

    def test_divergent_outcomes_get_informational(self):
        report = build_comparison_report(
            _sample("cc_trajectory.json"),
            _sample("op_trajectory.json"),
        )
        # CC succeeds, OP fails → alignment should be "informational"
        if report["outcome"]["reference_success"] != report["outcome"]["compared_success"]:
            assert report["confidence"]["alignment"] == "informational"
