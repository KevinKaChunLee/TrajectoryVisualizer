"""Comprehensive tests for the Converge before/after intervention module."""

from __future__ import annotations

import json
import os

import pytest

from trajectory_visualizer.converge.batch import BatchResult
from trajectory_visualizer.converge.intervention import (
    METRIC_DIRECTIONS,
    PATTERN_DIRECTIONS,
    build_intervention_report,
    compute_metric_deltas,
    compute_pattern_deltas,
    detect_guardrail_regressions,
    generate_recommendation,
    pair_tasks,
    test_significance as run_significance_test,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic BatchResult factory
# ---------------------------------------------------------------------------

def _make_result(
    task_id: str,
    alignment: dict | None = None,
    patterns: list[dict] | None = None,
    anchor_analysis: dict | None = None,
    failed: bool = False,
) -> BatchResult:
    """Build a minimal BatchResult with a synthetic report dict."""
    if failed:
        return BatchResult(task_id=task_id, report=None, error="synthetic failure")
    report: dict = {}
    if alignment is not None:
        report["alignment"] = alignment
    if patterns is not None:
        report["patterns"] = patterns
    if anchor_analysis is not None:
        report["anchor_analysis"] = anchor_analysis
    return BatchResult(task_id=task_id, report=report)


# =========================================================================
# 1. Task pairing
# =========================================================================

class TestPairTasks:
    def test_full_overlap(self):
        before = [_make_result("t1"), _make_result("t2"), _make_result("t3")]
        after = [_make_result("t1"), _make_result("t2"), _make_result("t3")]
        paired, before_only, after_only = pair_tasks(before, after)
        assert len(paired) == 3
        assert len(before_only) == 0
        assert len(after_only) == 0
        # Verify pairing correctness
        paired_ids = [(b.task_id, a.task_id) for b, a in paired]
        for b_id, a_id in paired_ids:
            assert b_id == a_id

    def test_partial_overlap(self):
        before = [_make_result("t1"), _make_result("t2"), _make_result("t3")]
        after = [_make_result("t2"), _make_result("t3"), _make_result("t4")]
        paired, before_only, after_only = pair_tasks(before, after)
        assert len(paired) == 2
        assert len(before_only) == 1
        assert len(after_only) == 1
        assert before_only[0].task_id == "t1"
        assert after_only[0].task_id == "t4"

    def test_zero_overlap(self):
        before = [_make_result("t1"), _make_result("t2")]
        after = [_make_result("t3"), _make_result("t4")]
        paired, before_only, after_only = pair_tasks(before, after)
        assert len(paired) == 0
        assert len(before_only) == 2
        assert len(after_only) == 2

    def test_failed_tasks_excluded(self):
        """Tasks with report=None (failed) should be excluded from pairing."""
        before = [
            _make_result("t1"),
            _make_result("t2", failed=True),
            _make_result("t3"),
        ]
        after = [
            _make_result("t1"),
            _make_result("t2"),  # after succeeded, but before failed
            _make_result("t3"),
        ]
        paired, before_only, after_only = pair_tasks(before, after)
        # t2 excluded because before version failed
        assert len(paired) == 2
        paired_ids = {b.task_id for b, _ in paired}
        assert "t2" not in paired_ids
        assert len(after_only) == 1
        assert after_only[0].task_id == "t2"


# =========================================================================
# 2. Delta computation (compute_metric_deltas)
# =========================================================================

class TestComputeMetricDeltas:
    def test_improved_recall_higher_is_better(self):
        """reference_recall going up should be direction='improved'."""
        before = [_make_result("t1", alignment={"reference_recall": 0.06})]
        after = [_make_result("t1", alignment={"reference_recall": 0.12})]
        paired, _, _ = pair_tasks(before, after)
        deltas = compute_metric_deltas(paired)
        assert "reference_recall" in deltas
        d = deltas["reference_recall"]
        assert d["direction"] == "improved"
        assert d["delta"] == pytest.approx(0.06, abs=0.001)

    def test_regressed_harmful_lower_is_better(self):
        """harmful_ratio going up should be direction='regressed'."""
        before = [_make_result("t1", alignment={"harmful_ratio": 0.20})]
        after = [_make_result("t1", alignment={"harmful_ratio": 0.35})]
        paired, _, _ = pair_tasks(before, after)
        deltas = compute_metric_deltas(paired)
        assert "harmful_ratio" in deltas
        d = deltas["harmful_ratio"]
        assert d["direction"] == "regressed"

    def test_unchanged_metric(self):
        """Identical values should yield direction='unchanged'."""
        before = [_make_result("t1", alignment={"alignment_f1": 0.50})]
        after = [_make_result("t1", alignment={"alignment_f1": 0.50})]
        paired, _, _ = pair_tasks(before, after)
        deltas = compute_metric_deltas(paired)
        assert deltas["alignment_f1"]["direction"] == "unchanged"

    def test_missing_metric_values_excluded(self):
        """If a metric is missing from a report, that pair is skipped for that metric."""
        # t1 has recall, t2 does not
        before = [
            _make_result("t1", alignment={"reference_recall": 0.10}),
            _make_result("t2", alignment={}),
        ]
        after = [
            _make_result("t1", alignment={"reference_recall": 0.20}),
            _make_result("t2", alignment={}),
        ]
        paired, _, _ = pair_tasks(before, after)
        deltas = compute_metric_deltas(paired)
        # reference_recall should only count t1
        if "reference_recall" in deltas:
            assert len(deltas["reference_recall"]["before_values"]) == 1

    def test_multiple_pairs_averaged(self):
        before = [
            _make_result("t1", alignment={"reference_recall": 0.10}),
            _make_result("t2", alignment={"reference_recall": 0.20}),
        ]
        after = [
            _make_result("t1", alignment={"reference_recall": 0.30}),
            _make_result("t2", alignment={"reference_recall": 0.40}),
        ]
        paired, _, _ = pair_tasks(before, after)
        deltas = compute_metric_deltas(paired)
        d = deltas["reference_recall"]
        assert d["before_mean"] == pytest.approx(0.15, abs=0.001)
        assert d["after_mean"] == pytest.approx(0.35, abs=0.001)
        assert d["direction"] == "improved"


# =========================================================================
# 3. Pattern deltas (compute_pattern_deltas)
# =========================================================================

class TestComputePatternDeltas:
    def test_pattern_frequency_decrease_improved(self):
        """write_retry in 80% before, 30% after -> improved (lower is better)."""
        # 10 pairs: 8 have write_retry before, 3 after
        before = []
        after = []
        for i in range(10):
            b_patterns = [{"type": "write_retry"}] if i < 8 else []
            a_patterns = [{"type": "write_retry"}] if i < 3 else []
            before.append(_make_result(f"t{i}", patterns=b_patterns))
            after.append(_make_result(f"t{i}", patterns=a_patterns))
        paired, _, _ = pair_tasks(before, after)
        deltas = compute_pattern_deltas(paired)
        assert "write_retry" in deltas
        d = deltas["write_retry"]
        assert d["before_frequency"] == pytest.approx(0.8, abs=0.01)
        assert d["after_frequency"] == pytest.approx(0.3, abs=0.01)
        assert d["direction"] == "improved"

    def test_new_pattern_only_in_after_regressed(self):
        """Pattern appearing only in after should be regressed (lower is better default)."""
        before = [_make_result(f"t{i}", patterns=[]) for i in range(5)]
        after = [
            _make_result(f"t{i}", patterns=[{"type": "dead_end_branch"}] if i < 4 else [])
            for i in range(5)
        ]
        paired, _, _ = pair_tasks(before, after)
        deltas = compute_pattern_deltas(paired)
        assert "dead_end_branch" in deltas
        d = deltas["dead_end_branch"]
        assert d["before_frequency"] == 0.0
        assert d["after_frequency"] == pytest.approx(0.8, abs=0.01)
        assert d["direction"] == "regressed"

    def test_no_patterns_no_deltas(self):
        before = [_make_result("t1", patterns=[])]
        after = [_make_result("t1", patterns=[])]
        paired, _, _ = pair_tasks(before, after)
        deltas = compute_pattern_deltas(paired)
        assert deltas == {}


# =========================================================================
# 4. Statistical testing (test_significance)
# =========================================================================

class TestSignificance:
    def test_significant_difference(self):
        before = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        after = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = run_significance_test(before, after)
        assert result["p_value"] is not None
        assert result["p_value"] < 0.05
        assert result["significant"] == True
        assert result["test_method"] in ("wilcoxon", "sign_test")

    def test_not_significant_similar_data(self):
        before = [5.0, 5.1, 4.9, 5.0, 5.2, 4.8]
        after = [5.1, 5.0, 5.0, 4.9, 5.1, 4.9]
        result = run_significance_test(before, after)
        assert result["significant"] == False

    def test_small_sample_warning(self):
        before = [1.0, 2.0, 3.0]
        after = [4.0, 5.0, 6.0]
        result = run_significance_test(before, after)
        assert result["warning"] is not None
        assert "fewer than 6" in result["warning"]

    def test_all_equal_values(self):
        before = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        after = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        result = run_significance_test(before, after)
        assert result["p_value"] == 1.0
        assert result["significant"] == False

    def test_single_observation(self):
        result = run_significance_test([1.0], [2.0])
        assert result["p_value"] is None
        assert result["significant"] == False
        assert result["test_method"] == "none"

    def test_empty_lists(self):
        result = run_significance_test([], [])
        assert result["significant"] == False


# =========================================================================
# 5. Guardrail detection (detect_guardrail_regressions)
# =========================================================================

class TestGuardrailDetection:
    def test_precision_recall_tradeoff(self):
        """Precision improves, recall regresses by >10% -> warning."""
        deltas = {
            "behavioral_precision": {
                "before_mean": 0.50,
                "after_mean": 0.70,
                "delta": 0.20,
                "direction": "improved",
                "before_values": [0.45, 0.50, 0.55, 0.50, 0.50, 0.50],
                "after_values": [0.65, 0.70, 0.75, 0.70, 0.70, 0.70],
            },
            "reference_recall": {
                "before_mean": 0.60,
                "after_mean": 0.48,
                "delta": -0.12,
                "direction": "regressed",
                "before_values": [0.55, 0.60, 0.65, 0.60, 0.60, 0.60],
                "after_values": [0.43, 0.48, 0.53, 0.48, 0.48, 0.48],
            },
        }
        warnings = detect_guardrail_regressions(deltas)
        assert len(warnings) >= 1
        regressed_metrics = [w["metric"] for w in warnings]
        assert "reference_recall" in regressed_metrics

    def test_clean_improvement_no_warnings(self):
        """One metric improves, nothing regresses -> no warnings."""
        deltas = {
            "reference_recall": {
                "before_mean": 0.50,
                "after_mean": 0.60,
                "delta": 0.10,
                "direction": "improved",
                "before_values": [0.50],
                "after_values": [0.60],
            },
            "behavioral_precision": {
                "before_mean": 0.70,
                "after_mean": 0.70,
                "delta": 0.0,
                "direction": "unchanged",
                "before_values": [0.70],
                "after_values": [0.70],
            },
        }
        warnings = detect_guardrail_regressions(deltas)
        assert len(warnings) == 0

    def test_sub_threshold_regression_no_warning(self):
        """Metric worsens by less than 10% relative -> no warning (with small sample)."""
        deltas = {
            "reference_recall": {
                "before_mean": 0.50,
                "after_mean": 0.55,
                "delta": 0.05,
                "direction": "improved",
                "before_values": [0.50],
                "after_values": [0.55],
            },
            "harmful_ratio": {
                "before_mean": 0.50,
                "after_mean": 0.52,
                "delta": 0.02,
                "direction": "regressed",
                # 0.02 / 0.50 = 4% relative — below threshold
                "before_values": [0.50],
                "after_values": [0.52],
            },
        }
        warnings = detect_guardrail_regressions(deltas)
        assert len(warnings) == 0

    def test_custom_threshold(self):
        """With a very low threshold (1%), even small regressions trigger warnings."""
        deltas = {
            "reference_recall": {
                "before_mean": 0.50,
                "after_mean": 0.60,
                "delta": 0.10,
                "direction": "improved",
                "before_values": [0.50],
                "after_values": [0.60],
            },
            "harmful_ratio": {
                "before_mean": 0.50,
                "after_mean": 0.52,
                "delta": 0.02,
                "direction": "regressed",
                "before_values": [0.50],
                "after_values": [0.52],
            },
        }
        # 4% relative regression exceeds the 1% custom threshold
        warnings = detect_guardrail_regressions(deltas, relative_threshold=0.01)
        assert len(warnings) >= 1

    def test_no_improved_metrics_no_warnings(self):
        """No improved metrics at all -> no guardrail warnings (nothing to tradeoff against)."""
        deltas = {
            "harmful_ratio": {
                "before_mean": 0.20,
                "after_mean": 0.30,
                "delta": 0.10,
                "direction": "regressed",
                "before_values": [0.20],
                "after_values": [0.30],
            },
        }
        warnings = detect_guardrail_regressions(deltas)
        assert len(warnings) == 0


# =========================================================================
# 6. Recommendation (generate_recommendation)
# =========================================================================

class TestGenerateRecommendation:
    def test_clean_improvement(self):
        deltas = {
            "reference_recall": {"direction": "improved"},
        }
        rec = generate_recommendation(deltas, guardrails=[])
        assert "improved" in rec.lower()
        assert "reference_recall" in rec

    def test_tradeoff_detected(self):
        deltas = {
            "behavioral_precision": {"direction": "improved"},
            "reference_recall": {"direction": "regressed"},
        }
        guardrails = [{"metric": "reference_recall"}]
        rec = generate_recommendation(deltas, guardrails)
        assert "regressed" in rec.lower() or "tradeoff" in rec.lower()
        assert "reference_recall" in rec

    def test_no_changes(self):
        deltas = {
            "reference_recall": {"direction": "unchanged"},
            "behavioral_precision": {"direction": "unchanged"},
        }
        rec = generate_recommendation(deltas, guardrails=[])
        assert "no significant" in rec.lower() or "no changes" in rec.lower()

    def test_regression_only(self):
        deltas = {
            "reference_recall": {"direction": "regressed"},
        }
        rec = generate_recommendation(deltas, guardrails=[])
        assert "regressed" in rec.lower()


# =========================================================================
# 7. Intervention report (build_intervention_report)
# =========================================================================

class TestBuildInterventionReport:
    def _build_sample_report(self):
        before = [
            _make_result("t1", alignment={"reference_recall": 0.10, "behavioral_precision": 0.60}),
            _make_result("t2", alignment={"reference_recall": 0.20, "behavioral_precision": 0.70}),
        ]
        after = [
            _make_result("t1", alignment={"reference_recall": 0.30, "behavioral_precision": 0.80}),
            _make_result("t2", alignment={"reference_recall": 0.40, "behavioral_precision": 0.90}),
        ]
        paired, before_only, after_only = pair_tasks(before, after)
        metric_deltas = compute_metric_deltas(paired)
        pattern_deltas = compute_pattern_deltas(paired)
        guardrails = detect_guardrail_regressions(metric_deltas)
        report = build_intervention_report(
            before_manifest="before.json",
            after_manifest="after.json",
            paired=paired,
            before_only=before_only,
            after_only=after_only,
            metric_deltas=metric_deltas,
            pattern_deltas=pattern_deltas,
            guardrails=guardrails,
        )
        return report

    def test_report_structure(self):
        report = self._build_sample_report()
        expected_keys = {
            "before_manifest",
            "after_manifest",
            "paired_tasks",
            "unpaired_before",
            "unpaired_after",
            "intervention_effect",
            "guardrail_warnings",
            "pattern_deltas",
            "recommendation",
        }
        assert expected_keys.issubset(set(report.keys()))
        assert report["paired_tasks"] == 2
        assert report["unpaired_before"] == 0
        assert report["unpaired_after"] == 0
        assert isinstance(report["recommendation"], str)

    def test_json_serializable(self):
        report = self._build_sample_report()

        # Custom encoder to handle numpy types from scipy
        class _NumpySafeEncoder(json.JSONEncoder):
            def default(self, obj):
                try:
                    import numpy as np
                    if isinstance(obj, (np.bool_,)):
                        return bool(obj)
                    if isinstance(obj, (np.integer,)):
                        return int(obj)
                    if isinstance(obj, (np.floating,)):
                        return float(obj)
                except ImportError:
                    pass
                return super().default(obj)

        serialized = json.dumps(report, cls=_NumpySafeEncoder)
        deserialized = json.loads(serialized)
        assert deserialized["paired_tasks"] == report["paired_tasks"]

    def test_intervention_effect_contains_p_value(self):
        report = self._build_sample_report()
        for metric_name, details in report["intervention_effect"].items():
            assert "p_value" in details
            assert "significant" in details
            assert "direction" in details


# =========================================================================
# 8. Integration: self-anchored before==after => zero deltas, no guardrails
# =========================================================================

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "samples")
CC_TRAJECTORY = os.path.join(SAMPLES_DIR, "cc_trajectory.json")
OP_TRAJECTORY = os.path.join(SAMPLES_DIR, "op_trajectory.json")


def _load_sample_report(ref_path: str, cmp_path: str) -> dict | None:
    """Build a comparison report from sample trajectories if they exist."""
    if not os.path.isfile(ref_path) or not os.path.isfile(cmp_path):
        return None
    try:
        from trajectory_visualizer.converge.alignment import build_comparison_report
        return build_comparison_report(ref_file=ref_path, cmp_file=cmp_path)
    except Exception:
        return None


@pytest.mark.skipif(
    not os.path.isfile(CC_TRAJECTORY) or not os.path.isfile(OP_TRAJECTORY),
    reason="Sample trajectory files not found",
)
class TestIntegrationSelfAnchored:
    """Use the same pair for both before and after to confirm all deltas are zero."""

    def test_identical_before_after_zero_deltas(self):
        report = _load_sample_report(CC_TRAJECTORY, OP_TRAJECTORY)
        if report is None:
            pytest.skip("Could not build comparison report from samples")

        # Use the same report as both before and after
        result_before = BatchResult(task_id="sample_task", report=report)
        result_after = BatchResult(task_id="sample_task", report=report)

        paired, before_only, after_only = pair_tasks([result_before], [result_after])
        assert len(paired) == 1
        assert len(before_only) == 0
        assert len(after_only) == 0

        metric_deltas = compute_metric_deltas(paired)
        for metric_name, d in metric_deltas.items():
            assert d["direction"] == "unchanged", (
                f"Expected unchanged for {metric_name}, got {d['direction']} "
                f"(before={d['before_mean']}, after={d['after_mean']})"
            )
            assert d["delta"] == pytest.approx(0.0, abs=0.001)

        pattern_deltas = compute_pattern_deltas(paired)
        for ptype, d in pattern_deltas.items():
            assert d["direction"] == "unchanged", (
                f"Expected unchanged for pattern {ptype}, got {d['direction']}"
            )
            assert d["delta"] == pytest.approx(0.0, abs=0.01)

        guardrails = detect_guardrail_regressions(metric_deltas)
        assert len(guardrails) == 0, f"Expected no guardrails, got {guardrails}"

    def test_full_report_self_anchored(self):
        """Build the full intervention report and verify structure."""
        report = _load_sample_report(CC_TRAJECTORY, OP_TRAJECTORY)
        if report is None:
            pytest.skip("Could not build comparison report from samples")

        result_before = BatchResult(task_id="sample_task", report=report)
        result_after = BatchResult(task_id="sample_task", report=report)

        paired, before_only, after_only = pair_tasks([result_before], [result_after])
        metric_deltas = compute_metric_deltas(paired)
        pattern_deltas = compute_pattern_deltas(paired)
        guardrails = detect_guardrail_regressions(metric_deltas)

        full_report = build_intervention_report(
            before_manifest="self_anchored_before.json",
            after_manifest="self_anchored_after.json",
            paired=paired,
            before_only=before_only,
            after_only=after_only,
            metric_deltas=metric_deltas,
            pattern_deltas=pattern_deltas,
            guardrails=guardrails,
        )

        assert full_report["paired_tasks"] == 1
        assert full_report["unpaired_before"] == 0
        assert full_report["unpaired_after"] == 0
        assert len(full_report["guardrail_warnings"]) == 0
        assert "no significant" in full_report["recommendation"].lower()

        # Verify JSON-serializable
        json.dumps(full_report)
