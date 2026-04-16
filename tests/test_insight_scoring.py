"""Unit tests for trajectory_visualizer.insight.scoring and scoring_config."""

from __future__ import annotations

import json

import pytest

from trajectory_visualizer.insight.scoring_config import get_profile, merge_profile, PROFILES
from trajectory_visualizer.insight.scoring import (
    _score_metric,
    _score_dimension,
    _classify_verdict,
    compute_targeting_score,
    compute_error_resilience_score,
    compute_execution_efficiency_score,
    compute_cost_efficiency_score,
    compute_trajectory_score,
)


# ===========================================================================
# 1. Scoring Config
# ===========================================================================

class TestGetProfile:
    def test_default_profile(self):
        p = get_profile("default")
        assert "thresholds" in p and "weights" in p

    def test_strict_profile(self):
        p = get_profile("strict")
        # Strict exploration ratio good threshold is tighter
        assert p["thresholds"]["targeting"]["exploration_ratio"]["good"] < \
               get_profile("default")["thresholds"]["targeting"]["exploration_ratio"]["good"]

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError, match="Unknown scoring profile"):
            get_profile("nonexistent")

    def test_returns_deep_copy(self):
        p1 = get_profile("default")
        p2 = get_profile("default")
        p1["weights"]["targeting"] = 999
        assert p2["weights"]["targeting"] != 999


class TestMergeProfile:
    def test_override_weight(self):
        base = get_profile("default")
        result = merge_profile(base, {"weights": {"targeting": 0.50}})
        assert result["weights"]["targeting"] == 0.50
        assert result["weights"]["error_resilience"] == 0.25  # unchanged

    def test_override_threshold(self):
        base = get_profile("default")
        result = merge_profile(base, {
            "thresholds": {"targeting": {"exploration_ratio": {"good": 1.0, "warn": 2.0, "bad": 3.0}}}
        })
        assert result["thresholds"]["targeting"]["exploration_ratio"]["good"] == 1.0
        # Other dimensions unchanged
        assert "error_resilience" in result["thresholds"]

    def test_deep_merge_preserves_base(self):
        base = get_profile("default")
        original_good = base["thresholds"]["targeting"]["exploration_ratio"]["good"]
        merge_profile(base, {"thresholds": {"targeting": {"exploration_ratio": {"good": 999}}}})
        assert base["thresholds"]["targeting"]["exploration_ratio"]["good"] == original_good


# ===========================================================================
# 2. Scoring Engine
# ===========================================================================

class TestScoreMetric:
    def test_below_good_is_100(self):
        assert _score_metric(1.0, good=3.0, warn=8.0, bad=15.0) == 100.0

    def test_at_good_is_100(self):
        assert _score_metric(3.0, good=3.0, warn=8.0, bad=15.0) == 100.0

    def test_at_warn_is_50(self):
        assert _score_metric(8.0, good=3.0, warn=8.0, bad=15.0) == 50.0

    def test_at_bad_is_0(self):
        assert _score_metric(15.0, good=3.0, warn=8.0, bad=15.0) == 0.0

    def test_above_bad_is_0(self):
        assert _score_metric(100.0, good=3.0, warn=8.0, bad=15.0) == 0.0

    def test_between_good_and_warn(self):
        score = _score_metric(5.5, good=3.0, warn=8.0, bad=15.0)
        assert 50.0 < score < 100.0

    def test_between_warn_and_bad(self):
        score = _score_metric(11.5, good=3.0, warn=8.0, bad=15.0)
        assert 0.0 < score < 50.0

    def test_none_returns_none(self):
        assert _score_metric(None, good=3.0, warn=8.0, bad=15.0) is None

    def test_zero_value(self):
        assert _score_metric(0, good=3.0, warn=8.0, bad=15.0) == 100.0

    def test_inverted_above_good_is_100(self):
        assert _score_metric(98.0, good=95.0, warn=80.0, bad=60.0, invert=True) == 100.0

    def test_inverted_at_good_is_100(self):
        assert _score_metric(95.0, good=95.0, warn=80.0, bad=60.0, invert=True) == 100.0

    def test_inverted_at_warn_is_50(self):
        assert _score_metric(80.0, good=95.0, warn=80.0, bad=60.0, invert=True) == 50.0

    def test_inverted_at_bad_is_0(self):
        assert _score_metric(60.0, good=95.0, warn=80.0, bad=60.0, invert=True) == 0.0

    def test_inverted_below_bad_is_0(self):
        assert _score_metric(30.0, good=95.0, warn=80.0, bad=60.0, invert=True) == 0.0

    def test_inverted_between_warn_and_good(self):
        score = _score_metric(87.5, good=95.0, warn=80.0, bad=60.0, invert=True)
        assert 50.0 < score < 100.0


class TestScoreDimension:
    def test_weighted_min_aggregation(self):
        metrics = {"a": 1.0, "b": 10.0}
        thresholds = {
            "a": {"good": 3.0, "warn": 8.0, "bad": 15.0},
            "b": {"good": 3.0, "warn": 8.0, "bad": 15.0},
        }
        score, details = _score_dimension(metrics, thresholds)
        # "a" scores 100, "b" scores ~35.7
        # weighted_min = 0.6 * min(100, 35.7) + 0.4 * mean(100, 35.7)
        expected = round(0.6 * details["b"] + 0.4 * (details["a"] + details["b"]) / 2, 1)
        assert score == expected
        assert details["a"] == 100.0

    def test_pure_min_aggregation(self):
        metrics = {"a": 1.0, "b": 10.0}
        thresholds = {
            "a": {"good": 3.0, "warn": 8.0, "bad": 15.0},
            "b": {"good": 3.0, "warn": 8.0, "bad": 15.0},
        }
        score, details = _score_dimension(metrics, thresholds, aggregation="min")
        assert score == round(details["b"], 1)
        assert details["a"] == 100.0

    def test_all_none_returns_none(self):
        metrics = {"a": None, "b": None}
        thresholds = {
            "a": {"good": 3.0, "warn": 8.0, "bad": 15.0},
            "b": {"good": 3.0, "warn": 8.0, "bad": 15.0},
        }
        score, _ = _score_dimension(metrics, thresholds)
        assert score is None

    def test_partial_none(self):
        metrics = {"a": 1.0, "b": None}
        thresholds = {
            "a": {"good": 3.0, "warn": 8.0, "bad": 15.0},
            "b": {"good": 3.0, "warn": 8.0, "bad": 15.0},
        }
        score, details = _score_dimension(metrics, thresholds)
        assert score == 100.0  # only "a" is valid
        assert details["b"] is None


class TestClassifyVerdict:
    def test_good(self):
        assert _classify_verdict(78) == "good"

    def test_warn(self):
        assert _classify_verdict(55) == "warn"

    def test_bad(self):
        assert _classify_verdict(22) == "bad"

    def test_none(self):
        assert _classify_verdict(None) == "n/a"

    def test_boundary_good(self):
        assert _classify_verdict(70) == "good"

    def test_boundary_warn(self):
        assert _classify_verdict(40) == "warn"

    def test_boundary_bad(self):
        assert _classify_verdict(39.9) == "bad"


class TestComputeTargetingScore:
    def test_good_targeting(self):
        ft = {"avg_steps_to_first_touch": 2.0, "exploration_ratio": 1.5}
        result = compute_targeting_score(ft, get_profile("default")["thresholds"]["targeting"])
        assert result["score"] >= 80
        assert result["verdict"] == "good"

    def test_bad_targeting(self):
        ft = {"avg_steps_to_first_touch": 20.0, "exploration_ratio": 15.0}
        result = compute_targeting_score(ft, get_profile("default")["thresholds"]["targeting"])
        assert result["score"] <= 20
        assert result["verdict"] == "bad"

    def test_no_data(self):
        result = compute_targeting_score(None, {})
        assert result["score"] is None
        assert result["verdict"] == "n/a"


class TestComputeTrajectoryScore:
    def test_all_good(self):
        metrics = {
            "tool_success_rate": 98.0,
            "tool_fail": 0,
            "total_steps": 10,
            "patch_lines": 50,
            "non_cache_ratio": 15.0,
            "tokens_per_patch_line": 400,
            "cache_utilization_ratio": 0.7,
        }
        diagnostics = {
            "file_targeting": {
                "avg_steps_to_first_touch": 2.0,
                "exploration_ratio": 1.5,
            },
            "chain_metrics": {"total_chains": 0, "longest_chain": 0},
            "error_clusters": [],
            "bottleneck_explanations": [
                {"decomposition": {"inference_pct": 30.0}},
            ],
        }
        result = compute_trajectory_score(metrics, diagnostics)
        assert result["composite_score"] >= 70
        assert result["composite_verdict"] == "good"
        assert result["profile"] == "default"
        assert result["judge"] is None

    def test_all_bad(self):
        metrics = {
            "tool_success_rate": 40.0,
            "tool_fail": 15,
            "total_steps": 100,
            "patch_lines": 5,
            "non_cache_ratio": 90.0,
            "tokens_per_patch_line": 8000,
            "cache_utilization_ratio": 0.02,
        }
        diagnostics = {
            "file_targeting": {
                "avg_steps_to_first_touch": 20.0,
                "exploration_ratio": 15.0,
            },
            "chain_metrics": {"total_chains": 8, "longest_chain": 10},
            "error_clusters": [{}] * 8,
            "bottleneck_explanations": [
                {"decomposition": {"inference_pct": 95.0}},
            ],
        }
        result = compute_trajectory_score(metrics, diagnostics)
        assert result["composite_score"] <= 30
        assert result["composite_verdict"] == "bad"

    def test_missing_targeting(self):
        metrics = {"tool_success_rate": 98.0, "tool_fail": 0}
        diagnostics = {
            "file_targeting": None,
            "chain_metrics": {"total_chains": 0, "longest_chain": 0},
            "error_clusters": [],
            "bottleneck_explanations": [],
        }
        result = compute_trajectory_score(metrics, diagnostics)
        assert result["dimensions"]["targeting"]["score"] is None
        # Composite still computed from other dimensions
        assert result["composite_score"] is not None

    def test_custom_profile(self):
        metrics = {"tool_success_rate": 98.0, "tool_fail": 0}
        diagnostics = {
            "file_targeting": {"avg_steps_to_first_touch": 2.0, "exploration_ratio": 1.5},
            "chain_metrics": {"total_chains": 0, "longest_chain": 0},
            "error_clusters": [],
            "bottleneck_explanations": [],
        }
        result = compute_trajectory_score(metrics, diagnostics,
                                          profile={"weights": {"targeting": 0.50}})
        assert result["profile"] == "custom"

    def test_json_serializable(self):
        metrics = {"tool_success_rate": 98.0, "tool_fail": 0}
        diagnostics = {
            "file_targeting": {"avg_steps_to_first_touch": 2.0, "exploration_ratio": 1.5},
            "chain_metrics": {"total_chains": 0, "longest_chain": 0},
            "error_clusters": [],
            "bottleneck_explanations": [],
        }
        result = compute_trajectory_score(metrics, diagnostics)
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_no_dimensions_available(self):
        result = compute_trajectory_score({}, {})
        assert result["composite_score"] is not None or result["composite_verdict"] in ("n/a", "bad", "warn", "good")
