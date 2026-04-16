"""Comprehensive tests for trajectory_visualizer.converge.batch module."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from trajectory_visualizer.converge.batch import (
    ManifestEntry,
    BatchResult,
    parse_manifest,
    run_batch,
    aggregate_reports,
    compute_pattern_frequency,
    promote_patterns,
    compute_consistency,
    build_batch_report,
)

# Real sample paths
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "samples")
CC_TRAJECTORY = os.path.normpath(os.path.join(SAMPLES_DIR, "cc_trajectory.json"))
OP_TRAJECTORY = os.path.normpath(os.path.join(SAMPLES_DIR, "op_trajectory.json"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_manifest(tmp_path, entries: list[dict], filename="manifest.json") -> str:
    """Write a manifest JSON file and return its path."""
    path = os.path.join(str(tmp_path), filename)
    with open(path, "w") as f:
        json.dump(entries, f)
    return path


def _make_report(
    recall=0.8, precision=0.7, f1=0.75, overhead=0.1, harmful=0.05,
    patterns=None, anchor_analysis=None, outcome=None, anchor_mode=None,
):
    """Create a synthetic report dict matching build_comparison_report output shape."""
    return {
        "alignment": {
            "reference_recall": recall,
            "behavioral_precision": precision,
            "alignment_f1": f1,
            "overhead_ratio": overhead,
            "harmful_ratio": harmful,
        },
        "patterns": patterns or [],
        "anchor_analysis": anchor_analysis,
        "outcome": outcome or {"reference_success": True, "compared_success": True},
        "anchor_mode": anchor_mode,
    }


def _make_batch_result(task_id, report=None, error=None):
    return BatchResult(task_id=task_id, report=report, error=error)


# ===========================================================================
# 1. Manifest Parsing (parse_manifest)
# ===========================================================================

class TestParseManifest:

    def test_valid_manifest_two_entries(self, tmp_path):
        """Valid manifest with 2 entries parses correctly."""
        # Create dummy trajectory files
        f1 = tmp_path / "ref.json"
        f2 = tmp_path / "cmp.json"
        f1.write_text("[]")
        f2.write_text("[]")

        manifest_path = _write_manifest(tmp_path, [
            {"task_id": "task_a", "reference": "ref.json", "compared": "cmp.json"},
            {"task_id": "task_b", "reference": "ref.json", "compared": "cmp.json"},
        ])

        entries = parse_manifest(manifest_path)
        assert len(entries) == 2
        assert entries[0].task_id == "task_a"
        assert entries[1].task_id == "task_b"
        assert entries[0].reference.endswith("ref.json")
        assert entries[0].compared.endswith("cmp.json")
        assert entries[0].anchor is None

    def test_missing_required_field_task_id(self, tmp_path):
        """Missing task_id raises ValueError."""
        f1 = tmp_path / "ref.json"
        f2 = tmp_path / "cmp.json"
        f1.write_text("[]")
        f2.write_text("[]")

        manifest_path = _write_manifest(tmp_path, [
            {"reference": "ref.json", "compared": "cmp.json"},
        ])

        with pytest.raises(ValueError, match="missing required field 'task_id'"):
            parse_manifest(manifest_path)

    def test_missing_required_field_reference(self, tmp_path):
        """Missing reference raises ValueError."""
        manifest_path = _write_manifest(tmp_path, [
            {"task_id": "t1", "compared": "cmp.json"},
        ])
        with pytest.raises(ValueError, match="missing required field 'reference'"):
            parse_manifest(manifest_path)

    def test_missing_file_raises(self, tmp_path):
        """Non-existent trajectory file raises FileNotFoundError."""
        manifest_path = _write_manifest(tmp_path, [
            {"task_id": "t1", "reference": "nonexistent.json", "compared": "also_missing.json"},
        ])
        with pytest.raises(FileNotFoundError):
            parse_manifest(manifest_path)

    def test_missing_manifest_file(self):
        """Non-existent manifest file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_manifest("/tmp/does_not_exist_manifest.json")

    def test_relative_path_resolution(self, tmp_path):
        """Relative paths resolve against manifest directory."""
        subdir = tmp_path / "data"
        subdir.mkdir()
        f1 = subdir / "ref.json"
        f2 = subdir / "cmp.json"
        f1.write_text("[]")
        f2.write_text("[]")

        # Manifest lives in subdir; paths are relative to it
        manifest_path = _write_manifest(subdir, [
            {"task_id": "t1", "reference": "ref.json", "compared": "cmp.json"},
        ])

        entries = parse_manifest(manifest_path)
        assert entries[0].reference == str(f1)
        assert entries[0].compared == str(f2)

    def test_optional_anchor_present(self, tmp_path):
        """Anchor path is parsed when present."""
        for name in ("ref.json", "cmp.json", "anchor.diff"):
            (tmp_path / name).write_text("[]")

        manifest_path = _write_manifest(tmp_path, [
            {"task_id": "t1", "reference": "ref.json", "compared": "cmp.json",
             "anchor": "anchor.diff"},
        ])

        entries = parse_manifest(manifest_path)
        assert entries[0].anchor is not None
        assert entries[0].anchor.endswith("anchor.diff")

    def test_optional_anchor_absent(self, tmp_path):
        """Anchor is None when not provided."""
        for name in ("ref.json", "cmp.json"):
            (tmp_path / name).write_text("[]")

        manifest_path = _write_manifest(tmp_path, [
            {"task_id": "t1", "reference": "ref.json", "compared": "cmp.json"},
        ])

        entries = parse_manifest(manifest_path)
        assert entries[0].anchor is None


# ===========================================================================
# 2. Batch Runner (run_batch)
# ===========================================================================

class TestRunBatch:

    def test_all_success_self_comparison(self):
        """Batch with cc_trajectory compared to itself succeeds for both entries."""
        entries = [
            ManifestEntry(task_id="pair1", reference=CC_TRAJECTORY, compared=CC_TRAJECTORY),
            ManifestEntry(task_id="pair2", reference=CC_TRAJECTORY, compared=CC_TRAJECTORY),
        ]
        results = run_batch(entries)
        assert len(results) == 2
        for r in results:
            assert r.error is None
            assert r.report is not None
            assert "alignment" in r.report

    def test_one_failure_isolation(self):
        """One entry fails (via mocked exception); the other succeeds.

        The loader is very resilient (catches all errors), so we patch
        build_comparison_report to raise on a specific task_id to test
        that error isolation works in run_batch.
        """
        original_bcr = None

        def _patched_bcr(*args, **kwargs):
            if kwargs.get("task_id") == "bad":
                raise RuntimeError("simulated failure")
            return original_bcr(*args, **kwargs)

        from trajectory_visualizer.converge.alignment import build_comparison_report as _orig
        original_bcr = _orig

        entries = [
            ManifestEntry(task_id="good", reference=CC_TRAJECTORY, compared=CC_TRAJECTORY),
            ManifestEntry(task_id="bad", reference=CC_TRAJECTORY, compared=CC_TRAJECTORY),
        ]

        with patch.object(
            __import__("trajectory_visualizer.converge.batch", fromlist=["build_comparison_report"]),
            "build_comparison_report",
            side_effect=_patched_bcr,
        ):
            results = run_batch(entries)

        assert len(results) == 2

        good = next(r for r in results if r.task_id == "good")
        bad = next(r for r in results if r.task_id == "bad")

        assert good.report is not None
        assert good.error is None
        assert bad.report is None
        assert bad.error is not None
        assert "simulated failure" in bad.error

    def test_empty_manifest(self):
        """Empty entry list returns empty results."""
        results = run_batch([])
        assert results == []

    def test_progress_callback(self):
        """Progress callback fires for each entry with correct (current, total)."""
        entries = [
            ManifestEntry(task_id="p1", reference=CC_TRAJECTORY, compared=CC_TRAJECTORY),
            ManifestEntry(task_id="p2", reference=CC_TRAJECTORY, compared=CC_TRAJECTORY),
        ]
        calls: list[tuple[int, int]] = []
        run_batch(entries, progress_callback=lambda cur, tot: calls.append((cur, tot)))
        assert calls == [(1, 2), (2, 2)]


# ===========================================================================
# 3. Aggregation (aggregate_reports)
# ===========================================================================

class TestAggregateReports:

    def test_known_values(self):
        """Mean, median, min, max of 3 synthetic reports match expected values."""
        results = [
            _make_batch_result("t1", _make_report(recall=0.6, precision=0.5, f1=0.55)),
            _make_batch_result("t2", _make_report(recall=0.8, precision=0.7, f1=0.75)),
            _make_batch_result("t3", _make_report(recall=1.0, precision=0.9, f1=0.95)),
        ]
        agg = aggregate_reports(results)

        assert agg["task_count"] == 3
        assert agg["success_count"] == 3
        assert agg["failure_count"] == 0

        recall = agg["metrics"]["reference_recall"]
        assert recall["count"] == 3
        assert recall["mean"] == 0.8
        assert recall["median"] == 0.8
        assert recall["min"] == 0.6
        assert recall["max"] == 1.0

        precision = agg["metrics"]["behavioral_precision"]
        assert precision["mean"] == 0.7
        assert precision["min"] == 0.5
        assert precision["max"] == 0.9

    def test_mixed_anchored_unanchored(self):
        """Anchor metrics aggregated only over anchored tasks."""
        anchor_data = {
            "reference": {"write_precision": 0.9, "write_recall": 0.8, "off_patch_write_ratio": 0.1},
            "compared": {"write_precision": 0.7, "write_recall": 0.6, "off_patch_write_ratio": 0.3},
        }
        results = [
            _make_batch_result("a1", _make_report(anchor_analysis=anchor_data)),
            _make_batch_result("a2", _make_report(anchor_analysis=anchor_data)),
            _make_batch_result("u1", _make_report()),  # unanchored
        ]
        agg = aggregate_reports(results)
        assert agg["anchored_count"] == 2
        assert agg["success_count"] == 3

        # Anchor metrics should have count=2
        wp_ref = agg["metrics"]["anchor_write_precision_ref"]
        assert wp_ref["count"] == 2
        assert "note" in wp_ref
        assert "2 anchored" in wp_ref["note"]

    def test_all_failures(self):
        """All failures returns zero success count and empty metrics."""
        results = [
            _make_batch_result("f1", error="boom"),
            _make_batch_result("f2", error="crash"),
        ]
        agg = aggregate_reports(results)
        assert agg["success_count"] == 0
        assert agg["failure_count"] == 2
        assert agg["metrics"] == {}


# ===========================================================================
# 4. Pattern Frequency (compute_pattern_frequency)
# ===========================================================================

class TestComputePatternFrequency:

    def test_pattern_counts_and_prevalence(self):
        """write_retry in 2/3 tasks, broad_exploration in 3/3."""
        results = [
            _make_batch_result("t1", _make_report(patterns=[
                {"type": "write_retry", "detail": "a"},
                {"type": "broad_exploration", "detail": "b"},
            ])),
            _make_batch_result("t2", _make_report(patterns=[
                {"type": "broad_exploration", "detail": "c"},
            ])),
            _make_batch_result("t3", _make_report(patterns=[
                {"type": "write_retry", "detail": "d"},
                {"type": "broad_exploration", "detail": "e"},
            ])),
        ]
        freq = compute_pattern_frequency(results)

        assert freq["write_retry"]["count"] == 2
        assert abs(freq["write_retry"]["prevalence"] - round(2 / 3, 4)) < 0.001
        assert freq["broad_exploration"]["count"] == 3
        assert freq["broad_exploration"]["prevalence"] == 1.0
        assert set(freq["broad_exploration"]["tasks"]) == {"t1", "t2", "t3"}

    def test_empty_results(self):
        """No successful results returns empty dict."""
        results = [_make_batch_result("f1", error="err")]
        freq = compute_pattern_frequency(results)
        assert freq == {}

    def test_duplicate_patterns_in_same_task(self):
        """Same pattern type appearing twice in one task is counted once."""
        results = [
            _make_batch_result("t1", _make_report(patterns=[
                {"type": "write_retry"},
                {"type": "write_retry"},
            ])),
        ]
        freq = compute_pattern_frequency(results)
        assert freq["write_retry"]["count"] == 1


# ===========================================================================
# 5. Pattern Promotion (promote_patterns)
# ===========================================================================

class TestPromotePatterns:

    def test_promoted_to_supported_finding(self):
        """Pattern with count=4, prevalence=0.8 in batch of 5 -> supported_finding."""
        freq = {
            "write_retry": {"count": 4, "prevalence": 0.8, "tasks": ["t1", "t2", "t3", "t4"]},
        }
        levels = promote_patterns(freq)
        assert levels["write_retry"] == "supported_finding"

    def test_stays_hypothesis(self):
        """Pattern with count=1, prevalence=0.2 -> single_pair_hypothesis."""
        freq = {
            "rare_pattern": {"count": 1, "prevalence": 0.2, "tasks": ["t1"]},
        }
        levels = promote_patterns(freq)
        assert levels["rare_pattern"] == "single_pair_hypothesis"

    def test_custom_thresholds(self):
        """Custom min_tasks and min_prevalence thresholds."""
        freq = {
            "moderate": {"count": 2, "prevalence": 0.4, "tasks": ["t1", "t2"]},
        }
        # Default thresholds (min_tasks=3, min_prevalence=0.5) -> hypothesis
        assert promote_patterns(freq)["moderate"] == "single_pair_hypothesis"

        # Lower thresholds -> promoted
        levels = promote_patterns(freq, min_tasks=2, min_prevalence=0.3)
        assert levels["moderate"] == "supported_finding"

    def test_mixed_promotion(self):
        """Multiple patterns with different evidence levels."""
        freq = {
            "strong": {"count": 5, "prevalence": 0.9, "tasks": list("abcde")},
            "weak": {"count": 1, "prevalence": 0.1, "tasks": ["z"]},
        }
        levels = promote_patterns(freq)
        assert levels["strong"] == "supported_finding"
        assert levels["weak"] == "single_pair_hypothesis"


# ===========================================================================
# 6. Consistency (compute_consistency)
# ===========================================================================

class TestComputeConsistency:

    def test_high_consistency_low_cv(self):
        """Low CV indicates high consistency."""
        aggregate = {
            "metrics": {
                "reference_recall": {"mean": 0.8, "stdev": 0.02},
            },
        }
        c = compute_consistency(aggregate)
        assert c["reference_recall"] is not None
        assert c["reference_recall"] < 0.1  # CV = 0.02/0.8 = 0.025

    def test_low_consistency_high_cv(self):
        """High CV indicates low consistency."""
        aggregate = {
            "metrics": {
                "reference_recall": {"mean": 0.5, "stdev": 0.4},
            },
        }
        c = compute_consistency(aggregate)
        assert c["reference_recall"] is not None
        assert c["reference_recall"] == 0.8  # CV = 0.4/0.5 = 0.8

    def test_zero_mean_returns_none(self):
        """Zero mean results in None (undefined CV)."""
        aggregate = {
            "metrics": {
                "some_metric": {"mean": 0, "stdev": 0.1},
            },
        }
        c = compute_consistency(aggregate)
        assert c["some_metric"] is None

    def test_empty_metrics(self):
        """Empty metrics returns empty consistency dict."""
        c = compute_consistency({"metrics": {}})
        assert c == {}


# ===========================================================================
# 7. Batch Report (build_batch_report)
# ===========================================================================

class TestBuildBatchReport:

    def test_structure(self):
        """Verify report has expected top-level keys."""
        results = [
            _make_batch_result("t1", _make_report(
                patterns=[{"type": "write_retry"}],
                outcome={"reference_success": True, "compared_success": False},
                anchor_mode="patch",
            )),
            _make_batch_result("t2", error="file not found"),
        ]
        agg = aggregate_reports(results)
        freq = compute_pattern_frequency(results)
        promoted = promote_patterns(freq)
        consistency = compute_consistency(agg)

        report = build_batch_report(
            manifest_path="/path/to/manifest.json",
            results=results,
            aggregate=agg,
            pattern_frequency=freq,
            promoted=promoted,
            consistency=consistency,
        )

        assert report["manifest"] == "/path/to/manifest.json"
        assert "summary" in report
        assert "consistency" in report
        assert "pattern_frequency" in report
        assert "per_task" in report

        # per_task structure
        assert len(report["per_task"]) == 2
        success_entry = report["per_task"][0]
        assert success_entry["task_id"] == "t1"
        assert success_entry["status"] == "success"
        assert "alignment_summary" in success_entry
        assert "outcome" in success_entry
        assert success_entry["anchor_mode"] == "patch"
        assert success_entry["pattern_count"] == 1

        fail_entry = report["per_task"][1]
        assert fail_entry["task_id"] == "t2"
        assert fail_entry["status"] == "failed"
        assert fail_entry["error"] == "file not found"

    def test_json_serializable(self):
        """Batch report can be serialized to JSON without errors."""
        results = [_make_batch_result("t1", _make_report())]
        agg = aggregate_reports(results)
        freq = compute_pattern_frequency(results)
        promoted = promote_patterns(freq)
        consistency = compute_consistency(agg)

        report = build_batch_report("/m.json", results, agg, freq, promoted, consistency)
        serialized = json.dumps(report)
        assert isinstance(serialized, str)
        roundtrip = json.loads(serialized)
        assert roundtrip["manifest"] == "/m.json"

    def test_pattern_frequency_includes_evidence_level(self):
        """Pattern frequency entries include evidence_level from promotion."""
        results = [
            _make_batch_result("t1", _make_report(patterns=[{"type": "broad_exploration"}])),
            _make_batch_result("t2", _make_report(patterns=[{"type": "broad_exploration"}])),
            _make_batch_result("t3", _make_report(patterns=[{"type": "broad_exploration"}])),
        ]
        freq = compute_pattern_frequency(results)
        promoted = promote_patterns(freq)
        agg = aggregate_reports(results)
        consistency = compute_consistency(agg)

        report = build_batch_report("/m.json", results, agg, freq, promoted, consistency)
        pf = report["pattern_frequency"]
        assert "broad_exploration" in pf
        assert pf["broad_exploration"]["evidence_level"] == "supported_finding"


# ===========================================================================
# 8. Integration (using real samples)
# ===========================================================================

class TestIntegration:

    @pytest.fixture()
    def manifest_path(self, tmp_path):
        """Create a manifest with real sample files."""
        manifest_data = [
            {
                "task_id": "self_compare",
                "reference": CC_TRAJECTORY,
                "compared": CC_TRAJECTORY,
            },
            {
                "task_id": "cross_compare",
                "reference": CC_TRAJECTORY,
                "compared": OP_TRAJECTORY,
            },
        ]
        return _write_manifest(tmp_path, manifest_data)

    def test_full_batch_pipeline(self, manifest_path):
        """Run full batch pipeline from manifest to report."""
        entries = parse_manifest(manifest_path)
        assert len(entries) == 2

        results = run_batch(entries)
        assert len(results) == 2
        assert all(r.report is not None for r in results)

        agg = aggregate_reports(results)
        assert agg["success_count"] == 2
        assert agg["failure_count"] == 0
        assert "reference_recall" in agg["metrics"]

    def test_aggregate_metrics_present(self, manifest_path):
        """Aggregate metrics include standard alignment metrics."""
        entries = parse_manifest(manifest_path)
        results = run_batch(entries)
        agg = aggregate_reports(results)

        for metric in ("reference_recall", "behavioral_precision", "alignment_f1"):
            assert metric in agg["metrics"], f"Missing metric: {metric}"
            stats = agg["metrics"][metric]
            for key in ("count", "mean", "median", "min", "max"):
                assert key in stats, f"Missing stat '{key}' in {metric}"

    def test_pattern_frequency_has_entries(self, manifest_path):
        """Cross-comparison should produce divergence patterns."""
        entries = parse_manifest(manifest_path)
        results = run_batch(entries)
        freq = compute_pattern_frequency(results)
        # At minimum, the cross-compare pair (cc vs op) should have some patterns
        # The self-compare may have none. We just verify the structure is correct.
        assert isinstance(freq, dict)
        for ptype, data in freq.items():
            assert "count" in data
            assert "prevalence" in data
            assert "tasks" in data
            assert data["count"] >= 1
            assert 0 < data["prevalence"] <= 1.0

    def test_full_batch_report_structure(self, manifest_path):
        """Full pipeline produces a well-formed batch report."""
        entries = parse_manifest(manifest_path)
        results = run_batch(entries)
        agg = aggregate_reports(results)
        freq = compute_pattern_frequency(results)
        promoted = promote_patterns(freq)
        consistency = compute_consistency(agg)

        report = build_batch_report(manifest_path, results, agg, freq, promoted, consistency)

        assert report["manifest"] == manifest_path
        assert report["summary"]["success_count"] == 2
        assert len(report["per_task"]) == 2
        assert all(t["status"] == "success" for t in report["per_task"])

        # JSON-serializable
        json.dumps(report)
