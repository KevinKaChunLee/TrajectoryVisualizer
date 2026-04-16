"""Integration tests for anchor file analysis against real sample trajectories."""

from __future__ import annotations

import json
import os

import pytest

from trajectory_visualizer.converge.alignment import build_comparison_report
from trajectory_visualizer.converge.anchor import classify_anchor_files


SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def _sample(name: str) -> str:
    path = os.path.join(SAMPLES, name)
    if not os.path.isfile(path):
        pytest.skip(f"Sample not found: {name}")
    return path


class TestAnchoredFileAnalysis:
    """6.1: CC vs OP with 132807.diff — verify file classes and precision/recall."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.report = build_comparison_report(
            _sample("cc_trajectory.json"),
            _sample("op_trajectory.json"),
            anchor_patch=_sample("132807.diff"),
            task_id="anchor-integration",
        )

    def test_anchor_analysis_present(self):
        aa = self.report.get("anchor_analysis")
        assert aa is not None, "anchor_analysis should be present in anchored report"

    def test_file_classes_sum_to_total(self):
        aa = self.report["anchor_analysis"]
        total = aa["total_anchor_files"]
        class_sum = sum(aa["file_classes"].values())
        assert class_sum == total, f"Class counts {class_sum} != total {total}"

    def test_total_anchor_files(self):
        aa = self.report["anchor_analysis"]
        assert aa["total_anchor_files"] == 43

    def test_cc_write_precision(self):
        """CC should have ~81% anchor-write precision (13/16)."""
        ref = self.report["anchor_analysis"]["reference"]
        assert ref["write_precision"] is not None
        assert 0.75 <= ref["write_precision"] <= 0.90

    def test_op_write_precision(self):
        """OP should have ~55% anchor-write precision (11/20)."""
        cmp = self.report["anchor_analysis"]["compared"]
        assert cmp["write_precision"] is not None
        assert 0.45 <= cmp["write_precision"] <= 0.65

    def test_op_off_patch_ratio(self):
        """OP should have ~45% off-patch write ratio."""
        cmp = self.report["anchor_analysis"]["compared"]
        assert cmp["off_patch_write_ratio"] is not None
        assert 0.35 <= cmp["off_patch_write_ratio"] <= 0.55

    def test_cc_off_patch_ratio(self):
        """CC should have lower off-patch ratio than OP."""
        ref = self.report["anchor_analysis"]["reference"]
        cmp = self.report["anchor_analysis"]["compared"]
        assert ref["off_patch_write_ratio"] < cmp["off_patch_write_ratio"]

    def test_per_class_recall_present(self):
        ref = self.report["anchor_analysis"]["reference"]
        assert "write_recall_by_class" in ref
        assert "source" in ref["write_recall_by_class"]

    def test_timing_metrics(self):
        ref = self.report["anchor_analysis"]["reference"]
        cmp = self.report["anchor_analysis"]["compared"]
        # Both should have found anchor files
        assert ref["time_to_first_anchor_read"] is not None
        assert cmp["time_to_first_anchor_read"] is not None

    def test_file_class_categories(self):
        """Verify all 5 categories are represented in the ground truth."""
        aa = self.report["anchor_analysis"]
        classes = aa["file_classes"]
        assert classes.get("source", 0) > 0
        assert classes.get("generated", 0) > 0
        assert classes.get("test", 0) > 0
        # fixture and spec may be 0 depending on classification — at least present
        assert "fixture" in classes
        assert "spec" in classes

    def test_json_serializable(self):
        json_str = json.dumps(self.report["anchor_analysis"], default=str)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "total_anchor_files" in parsed


class TestSelfAnchoredNoAnalysis:
    """6.2: Self-anchored comparison — anchor_analysis should be null."""

    def test_no_anchor_analysis(self):
        report = build_comparison_report(
            _sample("cc_trajectory.json"),
            _sample("op_trajectory.json"),
        )
        assert report.get("anchor_analysis") is None


class TestFileClassificationOnRealPatch:
    """3.3: Verify classification of 132807.diff files matches expectations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import re
        with open(_sample("132807.diff")) as f:
            content = f.read()
        self.anchor_files = set(
            re.findall(r'^(?:\+\+\+|\-\-\-) [ab]/(.+)$', content, re.MULTILINE)
        )

    def test_classification_counts(self):
        _, counts = classify_anchor_files(self.anchor_files)
        # Verify reasonable distribution
        assert counts["source"] > 0
        assert counts["generated"] > 0
        total = sum(counts.values())
        assert total == len(self.anchor_files)

    def test_specific_classifications(self):
        file_to_class, _ = classify_anchor_files(self.anchor_files)
        # These should be deterministic
        assert file_to_class.get("pkg/kubelet/kubelet_pods.go") == "source"
        assert file_to_class.get("pkg/apis/core/validation/validation_test.go") == "test"
        assert file_to_class.get("staging/src/k8s.io/api/core/v1/zz_generated.deepcopy.go") == "generated"
        assert file_to_class.get("api/openapi-spec/swagger.json") == "spec"

    def test_test_fixtures(self):
        file_to_class, _ = classify_anchor_files(self.anchor_files)
        # testdata files should be fixtures
        testdata_files = [f for f in self.anchor_files if "testdata" in f]
        for f in testdata_files:
            assert file_to_class[f] == "fixture", f"{f} should be fixture"

    def test_proto_files(self):
        file_to_class, _ = classify_anchor_files(self.anchor_files)
        # .proto files should be spec
        proto_files = [f for f in self.anchor_files if f.endswith(".proto")]
        for f in proto_files:
            assert file_to_class[f] == "spec", f"{f} should be spec"
