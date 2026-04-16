"""Unit tests for trajectory_visualizer.converge.anchor module (file classification, anchor metrics, orchestrator)."""

from __future__ import annotations

import pytest

from trajectory_visualizer.converge.canonical import CanonicalAction, ActionCost
from trajectory_visualizer.converge.anchor import (
    classify_file,
    classify_anchor_files,
    compute_anchor_metrics,
    compute_anchor_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(step_index=0, action_type="FILE_READ", target="/src/app.ts",
            tool="Read", status="success", tokens=500, latency_ms=1000,
            effect_label="unknown", effect_detail=None, args=None,
            token_share=None):
    """Create a CanonicalAction for testing.

    token_share defaults to tokens (as if the action is the only one in its step).
    """
    return CanonicalAction(
        step_index=step_index,
        action_type=action_type,
        target=target,
        tool=tool,
        args=args or {},
        status=status,
        cost=ActionCost(tokens=tokens, latency_ms=latency_ms,
                        token_share=token_share if token_share is not None else tokens),
        effect_label=effect_label,
        effect_detail=effect_detail or {},
    )


# ---------------------------------------------------------------------------
# 1. File classification — classify_file
# ---------------------------------------------------------------------------

class TestClassifyFile:
    """Tests for classify_file."""

    def test_go_test_file(self):
        assert classify_file("pkg/handler/handler_test.go") == "test"

    def test_generated_pb_go(self):
        assert classify_file("api/v1/service.pb.go") == "generated"

    def test_testdata_fixture(self):
        assert classify_file("pkg/handler/testdata/input.json") == "fixture"

    def test_openapi_spec_swagger(self):
        assert classify_file("docs/swagger.json") == "spec"

    def test_regular_source_go(self):
        assert classify_file("pkg/handler/handler.go") == "source"

    def test_proto_file(self):
        assert classify_file("api/v1/service.proto") == "spec"

    def test_zz_generated_deepcopy(self):
        assert classify_file("api/v1/zz_generated.deepcopy.go") == "generated"

    def test_after_roundtrip_fixture(self):
        assert classify_file("testdata/after_roundtrip_v2.json") == "fixture"

    def test_after_roundtrip_nested(self):
        assert classify_file("pkg/codec/testdata/after_roundtrip.yaml") == "fixture"

    def test_python_test_file(self):
        assert classify_file("tests/test_handler.py") == "test"

    def test_python_test_suffix(self):
        assert classify_file("tests/handler_test.py") == "test"

    def test_ts_spec_file(self):
        assert classify_file("src/components/button_spec.ts") == "test"

    def test_fixtures_directory(self):
        assert classify_file("tests/fixtures/sample.json") == "fixture"

    def test_openapi_spec_directory(self):
        assert classify_file("api/openapi-spec/v1.yaml") == "spec"

    def test_custom_override_takes_priority(self):
        """Custom rule should override default rules."""
        custom = [("**/handler.go", "custom_category")]
        assert classify_file("pkg/handler/handler.go", custom_rules=custom) == "custom_category"

    def test_custom_override_does_not_affect_others(self):
        """Custom rule for one file should not affect unrelated files."""
        custom = [("**/handler.go", "custom_category")]
        assert classify_file("pkg/handler/handler_test.go", custom_rules=custom) == "test"

    def test_custom_override_before_default(self):
        """Custom rule that matches a test file should win over default test rule."""
        custom = [("**/*_test.go", "important_test")]
        assert classify_file("pkg/handler_test.go", custom_rules=custom) == "important_test"

    def test_backslash_normalization(self):
        """Windows-style backslashes should be normalized."""
        assert classify_file("pkg\\handler\\handler_test.go") == "test"

    def test_compatibility_lifecycle_fixture(self):
        assert classify_file("some/compatibility_lifecycle/v1.json") == "fixture"

    def test_e2e_test_directory(self):
        # Pattern is **/test/e2e*/** — needs nested path for fnmatch ** to match
        assert classify_file("repo/test/e2e-suite/run.go") == "test"

    def test_integration_test_directory(self):
        assert classify_file("test/integration/handler_test.go") == "test"


# ---------------------------------------------------------------------------
# 1b. File classification — classify_anchor_files
# ---------------------------------------------------------------------------

class TestClassifyAnchorFiles:
    """Tests for classify_anchor_files."""

    def test_correct_counts(self):
        anchor = {
            "pkg/handler/handler.go",
            "pkg/handler/handler_test.go",
            "api/v1/service.pb.go",
            "pkg/testdata/input.json",
            "api/v1/service.proto",
        }
        file_to_class, counts = classify_anchor_files(anchor)
        assert file_to_class["pkg/handler/handler.go"] == "source"
        assert file_to_class["pkg/handler/handler_test.go"] == "test"
        assert file_to_class["api/v1/service.pb.go"] == "generated"
        assert file_to_class["pkg/testdata/input.json"] == "fixture"
        assert file_to_class["api/v1/service.proto"] == "spec"
        assert counts["source"] == 1
        assert counts["test"] == 1
        assert counts["generated"] == 1
        assert counts["fixture"] == 1
        assert counts["spec"] == 1

    def test_all_source(self):
        anchor = {"a.go", "b.go", "c.go"}
        _, counts = classify_anchor_files(anchor)
        assert counts["source"] == 3
        assert counts["test"] == 0
        assert counts["generated"] == 0

    def test_empty(self):
        file_to_class, counts = classify_anchor_files(set())
        assert file_to_class == {}
        assert counts["source"] == 0

    def test_with_custom_rules(self):
        anchor = {"src/special.go", "pkg/handler.go"}
        custom = [("**/special.go", "special")]
        file_to_class, counts = classify_anchor_files(anchor, custom_rules=custom)
        assert file_to_class["src/special.go"] == "special"
        assert file_to_class["pkg/handler.go"] == "source"
        assert counts.get("special", 0) == 1
        assert counts["source"] == 1

    def test_counts_sum_to_total(self):
        anchor = {
            "a.go", "b_test.go", "c.pb.go", "testdata/d.json",
            "e.proto", "f.go", "g_test.go",
        }
        _, counts = classify_anchor_files(anchor)
        assert sum(counts.values()) == len(anchor)


# ---------------------------------------------------------------------------
# 2. Anchor write metrics — compute_anchor_metrics
# ---------------------------------------------------------------------------

class TestComputeAnchorMetrics:
    """Tests for compute_anchor_metrics."""

    def test_precision_with_mixed_writes(self):
        """Some writes in anchor, some not => precision between 0 and 1."""
        anchor = {"src/handler.go", "src/handler_test.go"}
        file_to_class = {"src/handler.go": "source", "src/handler_test.go": "test"}
        actions = [
            _action(step_index=0, action_type="FILE_WRITE", target="src/handler.go"),
            _action(step_index=1, action_type="FILE_WRITE", target="src/handler_test.go"),
            _action(step_index=2, action_type="FILE_WRITE", target="src/unrelated.go"),
        ]
        m = compute_anchor_metrics(actions, anchor, file_to_class)
        # 2 anchor writes out of 3 total writes
        assert m["write_precision"] == round(2 / 3, 4)
        assert m["files_written"] == 3
        assert m["anchor_files_written"] == 2

    def test_recall_per_class(self):
        """Write recall should break down by file class."""
        anchor = {
            "src/handler.go",
            "src/handler_test.go",
            "src/util.go",
        }
        file_to_class = {
            "src/handler.go": "source",
            "src/handler_test.go": "test",
            "src/util.go": "source",
        }
        actions = [
            _action(step_index=0, action_type="FILE_WRITE", target="src/handler.go"),
            # test file not written, util.go not written
        ]
        m = compute_anchor_metrics(actions, anchor, file_to_class)
        # Overall recall: 1 out of 3
        assert m["write_recall"] == round(1 / 3, 4)
        # Source recall: 1 written out of 2 source anchor files
        assert m["write_recall_by_class"]["source"] == round(1 / 2, 4)
        # Test recall: 0 out of 1
        assert m["write_recall_by_class"]["test"] == 0.0

    def test_off_patch_write_ratio(self):
        """Off-patch ratio = (total - anchor) / total."""
        anchor = {"src/handler.go"}
        file_to_class = {"src/handler.go": "source"}
        actions = [
            _action(step_index=0, action_type="FILE_WRITE", target="src/handler.go"),
            _action(step_index=1, action_type="FILE_WRITE", target="src/extra1.go"),
            _action(step_index=2, action_type="FILE_WRITE", target="src/extra2.go"),
        ]
        m = compute_anchor_metrics(actions, anchor, file_to_class)
        # 2 off-patch out of 3 total
        assert m["off_patch_write_ratio"] == round(2 / 3, 4)

    def test_time_to_first_anchor_read(self):
        anchor = {"src/handler.go"}
        file_to_class = {"src/handler.go": "source"}
        actions = [
            _action(step_index=0, action_type="FILE_READ", target="src/unrelated.go"),
            _action(step_index=1, action_type="FILE_READ", target="src/handler.go"),
            _action(step_index=2, action_type="FILE_READ", target="src/handler.go"),
        ]
        m = compute_anchor_metrics(actions, anchor, file_to_class)
        assert m["time_to_first_anchor_read"] == 1

    def test_time_to_first_anchor_write(self):
        anchor = {"src/handler.go"}
        file_to_class = {"src/handler.go": "source"}
        actions = [
            _action(step_index=0, action_type="FILE_WRITE", target="src/unrelated.go"),
            _action(step_index=1, action_type="FILE_WRITE", target="src/other.go"),
            _action(step_index=2, action_type="FILE_WRITE", target="src/handler.go"),
        ]
        m = compute_anchor_metrics(actions, anchor, file_to_class)
        assert m["time_to_first_anchor_write"] == 2

    def test_no_writes_null_precision(self):
        """When no writes at all, precision should be None."""
        anchor = {"src/handler.go"}
        file_to_class = {"src/handler.go": "source"}
        actions = [
            _action(step_index=0, action_type="FILE_READ", target="src/handler.go"),
        ]
        m = compute_anchor_metrics(actions, anchor, file_to_class)
        assert m["write_precision"] is None
        assert m["off_patch_write_ratio"] is None
        assert m["files_written"] == 0

    def test_all_writes_outside_anchor(self):
        """All writes outside anchor => precision=0."""
        anchor = {"src/handler.go"}
        file_to_class = {"src/handler.go": "source"}
        actions = [
            _action(step_index=0, action_type="FILE_WRITE", target="src/foo.go"),
            _action(step_index=1, action_type="FILE_WRITE", target="src/bar.go"),
        ]
        m = compute_anchor_metrics(actions, anchor, file_to_class)
        assert m["write_precision"] == 0.0
        assert m["anchor_files_written"] == 0
        assert m["write_recall"] == 0.0
        assert m["off_patch_write_ratio"] == 1.0

    def test_all_anchor_files_written(self):
        """Perfect recall and precision when exactly the anchor files are written."""
        anchor = {"src/handler.go", "src/handler_test.go"}
        file_to_class = {"src/handler.go": "source", "src/handler_test.go": "test"}
        actions = [
            _action(step_index=0, action_type="FILE_WRITE", target="src/handler.go"),
            _action(step_index=1, action_type="FILE_WRITE", target="src/handler_test.go"),
        ]
        m = compute_anchor_metrics(actions, anchor, file_to_class)
        assert m["write_precision"] == 1.0
        assert m["write_recall"] == 1.0
        assert m["off_patch_write_ratio"] == 0.0

    def test_reason_actions_ignored(self):
        """REASON actions should be skipped entirely."""
        anchor = {"src/handler.go"}
        file_to_class = {"src/handler.go": "source"}
        actions = [
            _action(step_index=0, action_type="REASON", target=""),
            _action(step_index=1, action_type="FILE_WRITE", target="src/handler.go"),
        ]
        m = compute_anchor_metrics(actions, anchor, file_to_class)
        assert m["files_written"] == 1
        assert m["write_precision"] == 1.0
        assert m["time_to_first_anchor_write"] == 1

    def test_no_first_read_when_no_anchor_read(self):
        """time_to_first_anchor_read is None when anchor never read."""
        anchor = {"src/handler.go"}
        file_to_class = {"src/handler.go": "source"}
        actions = [
            _action(step_index=0, action_type="FILE_READ", target="src/other.go"),
        ]
        m = compute_anchor_metrics(actions, anchor, file_to_class)
        assert m["time_to_first_anchor_read"] is None

    def test_no_first_write_when_no_anchor_write(self):
        """time_to_first_anchor_write is None when anchor never written."""
        anchor = {"src/handler.go"}
        file_to_class = {"src/handler.go": "source"}
        actions = [
            _action(step_index=0, action_type="FILE_WRITE", target="src/other.go"),
        ]
        m = compute_anchor_metrics(actions, anchor, file_to_class)
        assert m["time_to_first_anchor_write"] is None

    def test_duplicate_write_same_file(self):
        """Writing the same anchor file twice should count as one unique file written."""
        anchor = {"src/handler.go"}
        file_to_class = {"src/handler.go": "source"}
        actions = [
            _action(step_index=0, action_type="FILE_WRITE", target="src/handler.go"),
            _action(step_index=1, action_type="FILE_WRITE", target="src/handler.go"),
        ]
        m = compute_anchor_metrics(actions, anchor, file_to_class)
        # files_written uses a set, so same target counts once
        assert m["files_written"] == 1
        assert m["anchor_files_written"] == 1
        assert m["write_precision"] == 1.0

    def test_empty_actions(self):
        """Empty action list should produce null metrics."""
        anchor = {"src/handler.go"}
        file_to_class = {"src/handler.go": "source"}
        m = compute_anchor_metrics([], anchor, file_to_class)
        assert m["write_precision"] is None
        assert m["write_recall"] == 0.0
        assert m["files_written"] == 0
        assert m["time_to_first_anchor_read"] is None
        assert m["time_to_first_anchor_write"] is None


# ---------------------------------------------------------------------------
# 3. Anchor analysis orchestrator — compute_anchor_analysis
# ---------------------------------------------------------------------------

class TestComputeAnchorAnalysis:
    """Tests for compute_anchor_analysis."""

    def test_returns_none_for_empty_anchor(self):
        ref = [_action(step_index=0, action_type="FILE_WRITE", target="src/a.go")]
        cmp = [_action(step_index=0, action_type="FILE_WRITE", target="src/a.go")]
        assert compute_anchor_analysis(ref, cmp, set()) is None

    def test_returns_none_for_none_anchor(self):
        ref = [_action(step_index=0, action_type="FILE_WRITE", target="src/a.go")]
        cmp = [_action(step_index=0, action_type="FILE_WRITE", target="src/a.go")]
        # The function checks `if not anchor_files`, None is falsy
        assert compute_anchor_analysis(ref, cmp, None) is None

    def test_full_analysis_structure(self):
        """Returns full dict with reference + compared metrics when anchor provided."""
        anchor = {"src/handler.go", "src/handler_test.go"}
        ref_actions = [
            _action(step_index=0, action_type="FILE_READ", target="src/handler.go"),
            _action(step_index=1, action_type="FILE_WRITE", target="src/handler.go"),
            _action(step_index=2, action_type="FILE_WRITE", target="src/handler_test.go"),
        ]
        cmp_actions = [
            _action(step_index=0, action_type="FILE_WRITE", target="src/handler.go"),
            _action(step_index=1, action_type="FILE_WRITE", target="src/extra.go"),
        ]
        result = compute_anchor_analysis(ref_actions, cmp_actions, anchor)

        assert result is not None
        assert result["total_anchor_files"] == 2
        assert "file_classes" in result
        assert "reference" in result
        assert "compared" in result

        # Reference: wrote both anchor files, no extras
        ref_m = result["reference"]
        assert ref_m["write_precision"] == 1.0
        assert ref_m["write_recall"] == 1.0
        assert ref_m["time_to_first_anchor_read"] == 0
        assert ref_m["time_to_first_anchor_write"] == 1

        # Compared: wrote 1 anchor + 1 extra
        cmp_m = result["compared"]
        assert cmp_m["write_precision"] == 0.5
        assert cmp_m["write_recall"] == 0.5
        assert cmp_m["anchor_files_written"] == 1

    def test_file_classes_counts_sum(self):
        """file_classes counts should sum to total_anchor_files."""
        anchor = {
            "src/handler.go",
            "src/handler_test.go",
            "api/v1/service.pb.go",
            "testdata/fixture.json",
            "api/service.proto",
        }
        ref_actions = []
        cmp_actions = []
        result = compute_anchor_analysis(ref_actions, cmp_actions, anchor)

        assert result is not None
        assert sum(result["file_classes"].values()) == result["total_anchor_files"]

    def test_custom_rules_passed_through(self):
        """Custom rules should be forwarded to classification."""
        anchor = {"src/special.go"}
        custom = [("**/special.go", "generated")]  # override: classify as generated
        result = compute_anchor_analysis([], [], anchor, custom_rules=custom)
        assert result is not None
        assert result["file_classes"].get("generated", 0) == 1
        assert result["file_classes"].get("source", 0) == 0  # not source despite .go

    def test_both_trajectories_empty(self):
        """Both empty action lists should still produce a valid result."""
        anchor = {"src/handler.go"}
        result = compute_anchor_analysis([], [], anchor)
        assert result is not None
        assert result["total_anchor_files"] == 1
        assert result["reference"]["write_precision"] is None
        assert result["compared"]["write_precision"] is None
