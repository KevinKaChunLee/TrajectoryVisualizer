"""Unit tests for score visualization (charts + rendering)."""

from __future__ import annotations

from trajectory_visualizer.insight.charts import build_score_gauge_chart
from trajectory_visualizer.insight.rendering import (
    build_dimension_cards_html,
    build_score_banner_badge_html,
    build_judge_result_html,
)


class TestBuildScoreGaugeChart:
    def test_renders_with_score(self):
        fig = build_score_gauge_chart(78.0, "good")
        assert fig is not None
        assert len(fig.data) > 0

    def test_renders_none_score(self):
        fig = build_score_gauge_chart(None, "n/a")
        assert fig is not None

    def test_renders_zero(self):
        fig = build_score_gauge_chart(0, "bad")
        assert fig is not None


class TestBuildDimensionCardsHtml:
    def test_renders_four_cards(self):
        dims = {
            "targeting": {"score": 85, "verdict": "good", "metrics": {"exploration_ratio": 1.5}},
            "error_resilience": {"score": 72, "verdict": "good", "metrics": {"tool_success_rate": 98.0}},
            "execution_efficiency": {"score": 68, "verdict": "warn", "metrics": {"steps_per_patch_line": 1.2}},
            "cost_efficiency": {"score": 82, "verdict": "good", "metrics": {"non_cache_ratio": 18.0}},
        }
        html = build_dimension_cards_html(dims)
        assert "score-dim-card" in html
        assert "Targeting" in html
        assert "Error Resilience" in html

    def test_na_dimension(self):
        dims = {
            "targeting": {"score": None, "verdict": "n/a", "metrics": {}},
        }
        html = build_dimension_cards_html(dims)
        assert "N/A" in html
        assert "insufficient data" in html

    def test_empty_dimensions(self):
        assert build_dimension_cards_html({}) == ""


class TestBuildScoreBannerBadgeHtml:
    def test_renders_badge(self):
        html = build_score_banner_badge_html(78.0, "good")
        assert "78" in html
        assert "Good" in html
        assert "score-banner-badge" in html

    def test_none_score(self):
        assert build_score_banner_badge_html(None, "n/a") == ""


class TestBuildJudgeResultHtml:
    def test_renders_result(self):
        result = {"verdict": "poor", "reasoning": "Too many retries", "flagged_steps": [5, 12]}
        html = build_judge_result_html(result)
        assert "judge-panel" in html
        assert "Poor" in html
        assert "Too many retries" in html
        assert "step 5" in html

    def test_none_result(self):
        assert build_judge_result_html(None) == ""

    def test_acceptable_verdict(self):
        result = {"verdict": "acceptable", "reasoning": "Looks fine", "flagged_steps": []}
        html = build_judge_result_html(result)
        assert "Acceptable" in html
