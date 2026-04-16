"""Unit tests for trajectory_visualizer.insight.diagnostics."""

from __future__ import annotations

import pytest

from trajectory_visualizer.insight.diagnostics import (
    extract_file_interactions,
    identify_target_files,
    compute_file_targeting_metrics,
    detect_failure_chains,
    classify_chain_steps,
    compute_failure_chain_metrics,
    link_chains_to_agents,
    cluster_errors,
    annotate_clusters_with_agents,
    format_root_cause_summary,
    decompose_hotspot_duration,
    explain_hotspot,
    compute_bottleneck_explanations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_step(index=0, role="assistant", total_tok=500, output_tok=50,
               duration=5.0, tool_calls=None, parts=None, error_count=0,
               cache_read=400, agent=""):
    tc = tool_calls or []
    return {
        "index": index,
        "role": role,
        "tokens": {
            "total": total_tok,
            "input": total_tok - output_tok - cache_read,
            "output": output_tok,
            "reasoning": 0,
            "cache_read": cache_read,
            "cache_write": 0,
        },
        "duration": duration,
        "parts": parts or [],
        "tool_calls": tc,
        "tool_call_count": len(tc),
        "error_count": error_count,
        "has_reasoning": False,
        "text_preview": "",
        "finish": "tool_use",
        "model_id": "m",
        "provider_id": "p",
        "time_created_ms": 1000000 + index * 10000,
        "time_completed_ms": 1000000 + index * 10000 + int(duration * 1000),
        "agent": agent,
        "mode": "",
        "message_id": "",
        "id": f"msg_{index}",
        "parent_id": "",
        "session_id": "ses",
        "cwd": "",
        "root": "",
    }


def _tc(tool_name, inp=None, status="success", error=None,
        time_start=None, time_end=None, metadata=None):
    return {
        "type": "tool_call",
        "tool_name": tool_name,
        "tool_id": f"tc_{tool_name}",
        "status": status,
        "title": "",
        "input": inp or {},
        "output": "",
        "error": error,
        "time_start": time_start,
        "time_end": time_end,
        "metadata": metadata or {},
    }


# ===========================================================================
# 1. File Interaction Analysis
# ===========================================================================

class TestExtractFileInteractions:
    def test_read_tool(self):
        steps = [_make_step(0, tool_calls=[
            _tc("Read", {"file_path": "/src/app.ts"}),
        ])]
        result = extract_file_interactions(steps)
        assert len(result) == 1
        assert result[0]["path"] == "/src/app.ts"
        assert result[0]["type"] == "read"
        assert result[0]["step"] == 0

    def test_edit_tool(self):
        steps = [_make_step(0, tool_calls=[
            _tc("Edit", {"file_path": "/src/db.ts"}),
        ])]
        result = extract_file_interactions(steps)
        assert len(result) == 1
        assert result[0]["type"] == "write"

    def test_glob_tool(self):
        steps = [_make_step(0, tool_calls=[
            _tc("Glob", {"pattern": "*.ts", "path": "/src"}),
        ])]
        result = extract_file_interactions(steps)
        assert len(result) == 2
        assert all(r["type"] == "search" for r in result)

    def test_grep_tool(self):
        steps = [_make_step(0, tool_calls=[
            _tc("Grep", {"pattern": "TODO", "path": "/src/utils"}),
        ])]
        result = extract_file_interactions(steps)
        # Grep extracts path (directory) but not pattern (text regex)
        assert len(result) == 1
        assert result[0]["path"] == "/src/utils"

    def test_bash_heuristic(self):
        steps = [_make_step(0, tool_calls=[
            _tc("Bash", {"command": "cat /src/config.ts | grep foo"}),
        ])]
        result = extract_file_interactions(steps)
        assert any(r["path"] == "/src/config.ts" for r in result)

    def test_mixed_tools(self):
        steps = [_make_step(0, tool_calls=[
            _tc("Read", {"file_path": "/a.ts"}),
            _tc("Edit", {"file_path": "/b.ts"}),
        ])]
        result = extract_file_interactions(steps)
        assert len(result) == 2

    def test_empty_steps(self):
        assert extract_file_interactions([]) == []

    def test_no_file_tool(self):
        steps = [_make_step(0, tool_calls=[
            _tc("SomeOtherTool", {"data": "value"}),
        ])]
        result = extract_file_interactions(steps)
        assert len(result) == 0


class TestIdentifyTargetFiles:
    def test_from_patch_parts(self):
        steps = [_make_step(0, parts=[
            {"type": "patch", "files": ["src/app.ts", "src/db.ts"], "hash": "", "id": ""},
        ])]
        targets = identify_target_files(steps)
        assert targets == {"src/app.ts", "src/db.ts"}

    def test_from_edit_tool(self):
        steps = [_make_step(0, tool_calls=[
            _tc("Edit", {"file_path": "/src/config.ts"}, status="success"),
        ])]
        targets = identify_target_files(steps)
        assert "/src/config.ts" in targets

    def test_excludes_failed_edit(self):
        steps = [_make_step(0, tool_calls=[
            _tc("Edit", {"file_path": "/src/fail.ts"}, status="error"),
        ])]
        targets = identify_target_files(steps)
        assert len(targets) == 0

    def test_union_of_patch_and_edit(self):
        steps = [
            _make_step(0, parts=[
                {"type": "patch", "files": ["a.ts"], "hash": "", "id": ""},
            ]),
            _make_step(1, tool_calls=[
                _tc("Write", {"file_path": "b.ts"}, status="success"),
            ]),
        ]
        targets = identify_target_files(steps)
        assert "a.ts" in targets and "b.ts" in targets


class TestComputeFileTargetingMetrics:
    def test_basic(self):
        interactions = [
            {"step": 0, "tool": "Read", "path": "/src/x.ts", "type": "read", "tokens": 100},
            {"step": 2, "tool": "Edit", "path": "/src/x.ts", "type": "write", "tokens": 200},
            {"step": 1, "tool": "Read", "path": "/src/y.ts", "type": "read", "tokens": 150},
        ]
        targets = {"/src/x.ts"}
        result = compute_file_targeting_metrics(interactions, targets, 10)
        assert result["steps_to_first_touch"]["/src/x.ts"]["absolute"] == 0
        assert result["exploration_ratio"] == 2.0  # 2 unique read/search files / 1 target

    def test_empty(self):
        result = compute_file_targeting_metrics([], set(), 0)
        assert result["avg_steps_to_first_touch"] is None

    def test_per_file_token_cost_split(self):
        interactions = [
            {"step": 0, "tool": "Grep", "path": "/a.ts", "type": "search", "tokens": 1000},
            {"step": 0, "tool": "Grep", "path": "/b.ts", "type": "search", "tokens": 1000},
        ]
        result = compute_file_targeting_metrics(interactions, {"/a.ts"}, 5)
        costs = result["per_file_token_cost"]
        assert costs.get("/a.ts", 0) == 500
        assert costs.get("/b.ts", 0) == 500


# ===========================================================================
# 2. Failure Chain Analysis
# ===========================================================================

class TestDetectFailureChains:
    def test_single_error_step(self):
        steps = [
            _make_step(0, error_count=0),
            _make_step(1, error_count=1, tool_calls=[_tc("Bash", status="error")]),
            _make_step(2, error_count=0),
        ]
        chains = detect_failure_chains(steps)
        assert len(chains) == 1
        assert chains[0]["steps"] == [1]

    def test_consecutive_errors(self):
        steps = [
            _make_step(0, error_count=1, tool_calls=[_tc("Bash", status="error")]),
            _make_step(1, error_count=1, tool_calls=[_tc("Bash", status="error")]),
            _make_step(2, error_count=1, tool_calls=[_tc("Bash", status="error")]),
            _make_step(3, error_count=0),
        ]
        chains = detect_failure_chains(steps)
        assert len(chains) == 1
        assert chains[0]["steps"] == [0, 1, 2]

    def test_non_adjacent_errors_separate(self):
        steps = [
            _make_step(0, error_count=1, tool_calls=[_tc("Bash", status="error")]),
            _make_step(1, error_count=0),
            _make_step(2, error_count=1, tool_calls=[_tc("Bash", status="error")]),
        ]
        chains = detect_failure_chains(steps)
        assert len(chains) == 2

    def test_user_steps_ignored(self):
        steps = [
            _make_step(0, error_count=1, tool_calls=[_tc("Bash", status="error")]),
            _make_step(1, role="user"),
            _make_step(2, error_count=1, tool_calls=[_tc("Bash", status="error")]),
        ]
        chains = detect_failure_chains(steps)
        # User step doesn't break the chain, but steps 0 and 2 aren't
        # truly consecutive assistant steps with an error-free assistant between them.
        # Since user steps are skipped entirely, 0 and 2 are consecutive assistant steps.
        assert len(chains) == 1
        assert chains[0]["steps"] == [0, 2]

    def test_no_errors(self):
        steps = [_make_step(0), _make_step(1)]
        assert detect_failure_chains(steps) == []

    def test_exit_code_error(self):
        steps = [_make_step(0, tool_calls=[
            _tc("Bash", {"command": "npm build"}, status="success", metadata={"exit": 1}),
        ])]
        chains = detect_failure_chains(steps)
        assert len(chains) == 1


class TestClassifyChainSteps:
    def test_recovery_attempt(self):
        steps = [
            _make_step(0, error_count=1, tool_calls=[
                _tc("Edit", {"file_path": "/src/app.ts"}, status="error"),
            ]),
            _make_step(1, error_count=1, tool_calls=[
                _tc("Edit", {"file_path": "/src/app.ts"}, status="error"),
            ]),
        ]
        chain = {"start": 0, "end": 1, "steps": [0, 1]}
        result = classify_chain_steps(chain, steps)
        assert result[0]["classification"] == "first_error"
        assert result[1]["classification"] == "recovery_attempt"

    def test_cascade(self):
        steps = [
            _make_step(0, error_count=1, tool_calls=[
                _tc("Bash", {"command": "npm build"}, status="error"),
            ]),
            _make_step(1, error_count=1, tool_calls=[
                _tc("Read", {"file_path": "/config.json"}, status="error"),
            ]),
        ]
        chain = {"start": 0, "end": 1, "steps": [0, 1]}
        result = classify_chain_steps(chain, steps)
        assert result[1]["classification"] == "cascade"


class TestComputeFailureChainMetrics:
    def test_basic(self):
        chains = [
            {"start": 0, "end": 2, "steps": [0, 1, 2]},
            {"start": 5, "end": 5, "steps": [5]},
        ]
        result = compute_failure_chain_metrics(chains, 20)
        assert result["total_chains"] == 2
        assert result["total_chain_steps"] == 4
        assert result["longest_chain"] == 3
        assert result["chain_step_pct"] == 20.0

    def test_empty(self):
        result = compute_failure_chain_metrics([], 10)
        assert result["total_chains"] == 0


# ===========================================================================
# 3. Root-Cause Attribution
# ===========================================================================

class TestClusterErrors:
    def test_same_pattern_clusters(self):
        steps = [
            _make_step(0, tool_calls=[_tc("Bash", status="error", error="Module not found: xyz")]),
            _make_step(1, tool_calls=[_tc("Bash", status="error", error="Module not found: xyz")]),
            _make_step(2, tool_calls=[_tc("Bash", status="error", error="Module not found: xyz")]),
        ]
        clusters = cluster_errors(steps)
        assert len(clusters) == 1
        assert clusters[0]["count"] == 3
        assert clusters[0]["tool"] == "Bash"

    def test_different_patterns(self):
        steps = [
            _make_step(0, tool_calls=[_tc("Edit", status="error", error="File not found")]),
            _make_step(1, tool_calls=[_tc("Bash", status="success", metadata={"exit": 2})]),
        ]
        clusters = cluster_errors(steps)
        assert len(clusters) == 2

    def test_sorted_by_count(self):
        steps = [
            _make_step(0, tool_calls=[_tc("Bash", status="error", error="A")]),
            _make_step(1, tool_calls=[_tc("Edit", status="error", error="B")]),
            _make_step(2, tool_calls=[_tc("Edit", status="error", error="B")]),
        ]
        clusters = cluster_errors(steps)
        assert clusters[0]["tool"] == "Edit"
        assert clusters[0]["count"] == 2

    def test_no_errors(self):
        steps = [_make_step(0, tool_calls=[_tc("Read", status="success")])]
        assert cluster_errors(steps) == []


class TestFormatRootCauseSummary:
    def test_basic(self):
        clusters = [
            {"tool": "Bash", "pattern": "Permission denied", "count": 3,
             "steps": [5, 6, 8], "first_step": 5, "last_step": 8},
        ]
        result = format_root_cause_summary(clusters)
        assert len(result) == 1
        assert "3x Bash" in result[0]
        assert "Permission denied" in result[0]

    def test_cross_agent(self):
        clusters = [
            {"tool": "Bash", "pattern": "err", "count": 1,
             "steps": [3], "first_step": 3, "last_step": 3,
             "parent_agent": "main", "parent_step": 1},
        ]
        result = format_root_cause_summary(clusters)
        assert "traced to main step 1" in result[0]


# ===========================================================================
# 4. Bottleneck Explanation
# ===========================================================================

class TestDecomposeHotspotDuration:
    def test_tool_dominated(self):
        step = _make_step(0, duration=45.0, tool_calls=[
            _tc("Bash", {"command": "npm run build"},
                 time_start=1000, time_end=39000),
        ])
        result = decompose_hotspot_duration(step, idle_gap=0)
        assert result["tool_s"] == 38.0
        assert result["inference_s"] == 7.0
        assert result["tool_pct"] > 80

    def test_inference_dominated(self):
        step = _make_step(0, duration=30.0, tool_calls=[
            _tc("Read", {"file_path": "/a.ts"}, time_start=1000, time_end=3000),
        ])
        result = decompose_hotspot_duration(step, idle_gap=1.0)
        assert result["inference_s"] == 27.0
        assert result["inference_pct"] == 90.0

    def test_missing_timing(self):
        step = _make_step(0, duration=20.0, tool_calls=[
            _tc("Read", {"file_path": "/a.ts"}),  # no timing
        ])
        result = decompose_hotspot_duration(step, idle_gap=0)
        assert result["timing_incomplete"] is True
        # All time attributed to inference when no tool timing
        assert result["inference_s"] == 20.0

    def test_tool_time_capped(self):
        # Tool time exceeds step duration (parallel tools)
        step = _make_step(0, duration=10.0, tool_calls=[
            _tc("Bash", time_start=0, time_end=8000),
            _tc("Read", time_start=0, time_end=7000),
        ])
        result = decompose_hotspot_duration(step, idle_gap=0)
        assert result["tool_s"] == 10.0  # capped at duration
        assert result["inference_s"] == 0

    def test_zero_duration(self):
        step = _make_step(0, duration=0)
        result = decompose_hotspot_duration(step)
        assert result["timing_incomplete"] is True


class TestExplainHotspot:
    def test_tool_heavy(self):
        step = _make_step(0, duration=45.0, total_tok=10000)
        decomp = {
            "tool_s": 38.0, "inference_s": 7.0, "idle_s": 0,
            "tool_pct": 84.4, "inference_pct": 15.6, "idle_pct": 0,
            "timing_incomplete": False,
            "dominant_tool": {"name": "Bash", "target": "npm run build", "duration_s": 35.0},
        }
        result = explain_hotspot(step, decomp)
        assert "Step 0" in result
        assert "Bash" in result
        assert "38.0s" in result

    def test_inference_heavy(self):
        step = _make_step(0, duration=30.0, total_tok=42000)
        decomp = {
            "tool_s": 2.0, "inference_s": 27.0, "idle_s": 1.0,
            "tool_pct": 6.7, "inference_pct": 90.0, "idle_pct": 3.3,
            "timing_incomplete": False, "dominant_tool": None,
        }
        result = explain_hotspot(step, decomp)
        assert "inference" in result.lower()
        assert "42,000" in result


class TestComputeBottleneckExplanations:
    def test_returns_top_n(self):
        steps = [_make_step(i, duration=float(10 - i)) for i in range(10)]
        from trajectory_visualizer.insight.analytics import compute_step_analytics
        analytics = compute_step_analytics(steps)
        result = compute_bottleneck_explanations(steps, analytics, n=3)
        assert len(result) == 3
        # Longest first
        assert result[0]["step_idx"] == 0

    def test_empty(self):
        result = compute_bottleneck_explanations([], [], n=5)
        assert result == []
