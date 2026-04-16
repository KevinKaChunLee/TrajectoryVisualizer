"""Integration tests for diagnostics against real sample trajectories."""

from __future__ import annotations

import os

import pytest

from trajectory_visualizer.insight.loaders import load_trajectory
from trajectory_visualizer.insight.parser import parse_steps
from trajectory_visualizer.insight.metrics import compute_agent_summary
from trajectory_visualizer.insight.analytics import compute_step_analytics
from trajectory_visualizer.insight.diagnostics import (
    extract_file_interactions,
    identify_target_files,
    compute_file_targeting_metrics,
    detect_failure_chains,
    classify_chain_steps,
    link_chains_to_agents,
    compute_failure_chain_metrics,
    cluster_errors,
    annotate_clusters_with_agents,
    format_root_cause_summary,
    compute_bottleneck_explanations,
)
from trajectory_visualizer.insight.charts import build_file_interaction_chart
from trajectory_visualizer.insight.rendering import (
    build_failure_chain_strip_html,
    build_bottleneck_cards_html,
    build_root_cause_html,
)


SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def _load_and_parse(filename: str):
    path = os.path.join(SAMPLES, filename)
    if not os.path.isfile(path):
        pytest.skip(f"Sample file not found: {filename}")
    raw = load_trajectory(path)
    steps = parse_steps(raw)
    return raw, steps


class TestClaudeCodeTrajectory:
    """7.1: End-to-end with Claude Code sample."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.raw, self.steps = _load_and_parse("cc_trajectory.json")
        self.analytics = compute_step_analytics(self.steps)
        self.agent_summaries = compute_agent_summary(self.steps, self.raw)

    def test_file_interactions_extracted(self):
        interactions = extract_file_interactions(self.steps)
        assert len(interactions) > 0
        assert all(k in interactions[0] for k in ("step", "tool", "path", "type"))

    def test_target_files_identified(self):
        targets = identify_target_files(self.steps)
        # CC trajectory should have some patches or edits
        assert isinstance(targets, set)

    def test_file_targeting_metrics(self):
        interactions = extract_file_interactions(self.steps)
        targets = identify_target_files(self.steps)
        metrics = compute_file_targeting_metrics(interactions, targets, len(self.steps))
        assert "steps_to_first_touch" in metrics
        assert "exploration_ratio" in metrics
        assert "per_file_token_cost" in metrics

    def test_failure_chains(self):
        chains = detect_failure_chains(self.steps)
        assert isinstance(chains, list)
        for c in chains:
            assert "start" in c and "end" in c and "steps" in c
            # Classify
            classified = classify_chain_steps(c, self.steps)
            assert classified[0]["classification"] == "first_error"

    def test_failure_chain_metrics(self):
        chains = detect_failure_chains(self.steps)
        asst_count = sum(1 for s in self.steps if s.get("role") == "assistant")
        metrics = compute_failure_chain_metrics(chains, asst_count)
        assert metrics["total_chains"] >= 0
        assert 0 <= metrics["chain_step_pct"] <= 100

    def test_error_clusters(self):
        clusters = cluster_errors(self.steps)
        assert isinstance(clusters, list)
        if clusters:
            assert clusters[0]["count"] >= clusters[-1]["count"]
            summaries = format_root_cause_summary(clusters)
            assert len(summaries) == len(clusters)

    def test_bottleneck_explanations(self):
        results = compute_bottleneck_explanations(self.steps, self.analytics)
        assert len(results) <= 5
        for r in results:
            assert "explanation" in r
            assert "decomposition" in r
            d = r["decomposition"]
            # Percentages should roughly sum to 100
            total_pct = d["tool_pct"] + d["inference_pct"] + d["idle_pct"]
            assert 95 <= total_pct <= 105 or d["timing_incomplete"]

    def test_file_interaction_chart_renders(self):
        interactions = extract_file_interactions(self.steps)
        targets = identify_target_files(self.steps)
        fig = build_file_interaction_chart(interactions, targets)
        assert fig is not None

    def test_rendering_functions(self):
        chains = detect_failure_chains(self.steps)
        chain_html = build_failure_chain_strip_html(chains)
        assert isinstance(chain_html, str)

        explanations = compute_bottleneck_explanations(self.steps, self.analytics)
        bottleneck_html = build_bottleneck_cards_html(explanations)
        assert isinstance(bottleneck_html, str)

        clusters = cluster_errors(self.steps)
        rc_html = build_root_cause_html(clusters)
        assert isinstance(rc_html, str)


class TestCodeArtsTrajectory:
    """7.2: End-to-end with CodeArts — cross-agent failure chain linking."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.raw, self.steps = _load_and_parse("ca_trajectory.json")
        self.analytics = compute_step_analytics(self.steps)
        self.agent_summaries = compute_agent_summary(self.steps, self.raw)

    def test_cross_agent_chain_linking(self):
        chains = detect_failure_chains(self.steps)
        linked = link_chains_to_agents(chains, self.steps, self.agent_summaries)
        assert isinstance(linked, list)
        # Verify linking doesn't crash even if no cross-agent chains exist

    def test_cross_agent_error_clusters(self):
        clusters = cluster_errors(self.steps)
        annotated = annotate_clusters_with_agents(
            clusters, self.steps, self.agent_summaries)
        assert isinstance(annotated, list)

    def test_file_interactions(self):
        interactions = extract_file_interactions(self.steps)
        assert isinstance(interactions, list)


    # Lingxi support removed — TestLingxiTrajectory class deleted.
