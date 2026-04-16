"""Integration tests for trajectory scoring against real sample trajectories."""

from __future__ import annotations

import os

import pytest

from trajectory_visualizer.insight.loaders import load_trajectory
from trajectory_visualizer.insight.parser import parse_steps
from trajectory_visualizer.insight.metrics import compute_metrics, compute_agent_summary
from trajectory_visualizer.insight.analytics import compute_step_analytics
from trajectory_visualizer.insight.diagnostics import (
    extract_file_interactions,
    identify_target_files,
    compute_file_targeting_metrics,
    detect_failure_chains,
    compute_failure_chain_metrics,
    cluster_errors,
    compute_bottleneck_explanations,
)
from trajectory_visualizer.insight.scoring import compute_trajectory_score
from trajectory_visualizer.insight.scoring_config import get_profile
from trajectory_visualizer.insight.judge import build_judge_prompt, invoke_judge


SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def _load_and_analyze(filename: str):
    path = os.path.join(SAMPLES, filename)
    if not os.path.isfile(path):
        pytest.skip(f"Sample file not found: {filename}")
    raw = load_trajectory(path)
    steps = parse_steps(raw)
    metrics = compute_metrics(steps, raw)
    step_analytics = compute_step_analytics(steps)
    agent_summaries = compute_agent_summary(steps, raw)

    interactions = extract_file_interactions(steps)
    target_files = identify_target_files(steps)
    file_targeting = compute_file_targeting_metrics(interactions, target_files, len(steps))
    chains = detect_failure_chains(steps)
    chain_metrics = compute_failure_chain_metrics(
        chains, sum(1 for s in steps if s.get("role") == "assistant"))
    clusters = cluster_errors(steps)
    bottleneck_explanations = compute_bottleneck_explanations(steps, step_analytics)

    diagnostics = {
        "file_targeting": file_targeting,
        "chain_metrics": chain_metrics,
        "error_clusters": clusters,
        "failure_chains": chains,
        "bottleneck_explanations": bottleneck_explanations,
    }
    return steps, metrics, diagnostics


class TestClaudeCodeScoring:
    """6.1: End-to-end with Claude Code sample trajectory."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.steps, self.metrics, self.diagnostics = _load_and_analyze("cc_trajectory.json")

    def test_composite_score_computed(self):
        result = compute_trajectory_score(self.metrics, self.diagnostics)
        assert result["composite_score"] is not None
        assert 0 <= result["composite_score"] <= 100

    def test_all_four_dimensions(self):
        result = compute_trajectory_score(self.metrics, self.diagnostics)
        dims = result["dimensions"]
        assert "targeting" in dims
        assert "error_resilience" in dims
        assert "execution_efficiency" in dims
        assert "cost_efficiency" in dims

    def test_verdict_is_valid(self):
        result = compute_trajectory_score(self.metrics, self.diagnostics)
        assert result["composite_verdict"] in ("good", "warn", "bad", "n/a")

    def test_json_serializable(self):
        import json
        result = compute_trajectory_score(self.metrics, self.diagnostics)
        json_str = json.dumps(result)
        assert isinstance(json_str, str)


    # Lingxi support removed — TestLingxiGracefulDegradation class deleted.


class TestProfileComparison:
    """6.3: Verify strict < default < lenient scores for same trajectory."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.steps, self.metrics, self.diagnostics = _load_and_analyze("cc_trajectory.json")

    def test_strict_lower_than_default(self):
        default_result = compute_trajectory_score(self.metrics, self.diagnostics, profile="default")
        strict_result = compute_trajectory_score(self.metrics, self.diagnostics, profile="strict")
        # Strict should be <= default (tighter thresholds)
        if default_result["composite_score"] is not None and strict_result["composite_score"] is not None:
            assert strict_result["composite_score"] <= default_result["composite_score"] + 1  # allow rounding

    def test_lenient_higher_than_default(self):
        default_result = compute_trajectory_score(self.metrics, self.diagnostics, profile="default")
        lenient_result = compute_trajectory_score(self.metrics, self.diagnostics, profile="lenient")
        # Lenient should be >= default (looser thresholds)
        if default_result["composite_score"] is not None and lenient_result["composite_score"] is not None:
            assert lenient_result["composite_score"] >= default_result["composite_score"] - 1  # allow rounding


class TestJudgePromptWithRealData:
    """6.4: Judge prompt construction with real diagnostics data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.steps, self.metrics, self.diagnostics = _load_and_analyze("cc_trajectory.json")

    def test_prompt_contains_metric_values(self):
        result = compute_trajectory_score(self.metrics, self.diagnostics)
        chains = self.diagnostics.get("failure_chains", [])
        bottlenecks = self.diagnostics.get("bottleneck_explanations", [])
        prompt = build_judge_prompt(result, chains, bottlenecks)

        # Should contain composite score
        assert str(int(result["composite_score"])) in prompt or f"{result['composite_score']}" in prompt

        # Should contain dimension names
        assert "targeting" in prompt.lower() or "error_resilience" in prompt.lower()

    def test_judge_not_invoked_for_high_score(self):
        result = compute_trajectory_score(self.metrics, self.diagnostics)
        # If score is > 65, judge should not be invoked
        if result["composite_score"] and result["composite_score"] > 65:
            judge_result = invoke_judge(result)
            assert judge_result is None

    def test_judge_with_mock_llm(self):
        result = compute_trajectory_score(self.metrics, self.diagnostics)
        # Force invocation by using a wide uncertain band
        mock_response = '{"verdict": "acceptable", "reasoning": "Looks good", "flagged_steps": []}'
        judge_result = invoke_judge(
            result,
            self.diagnostics.get("failure_chains"),
            self.diagnostics.get("bottleneck_explanations"),
            llm_callable=lambda prompt: mock_response,
            uncertain_band=(0, 100),  # always invoke
        )
        assert judge_result is not None
        assert judge_result["verdict"] == "acceptable"
