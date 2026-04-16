"""Unit tests for trajectory_visualizer.converge module (canonical, alignment, milestones, divergence)."""

from __future__ import annotations

import pytest

from trajectory_visualizer.converge.canonical import (
    CanonicalAction,
    ActionCost,
    canonicalize_steps,
    assign_effect_labels,
    compute_action_cost,
    semantic_equivalent,
    reduce_composite_command,
    DEFAULT_TOKEN_RATE,
)
from trajectory_visualizer.converge.alignment import (
    align_trajectories,
    compute_alignment_metrics,
    compute_harmful_divergence,
)
from trajectory_visualizer.converge.milestones import (
    extract_milestones,
    compute_milestone_deltas,
    segment_by_milestones,
    compare_segments,
)
from trajectory_visualizer.converge.divergence import classify_divergences


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


# ===========================================================================
# 1. Canonical Actions (trajectory_visualizer.converge.canonical)
# ===========================================================================

class TestCanonicalizeSteps:
    """Tests for canonicalize_steps type mapping."""

    def test_read_maps_to_file_read(self):
        steps = [_make_step(0, tool_calls=[
            _tc("Read", {"file_path": "/src/app.ts"}),
        ])]
        actions = canonicalize_steps(steps)
        tool_actions = [a for a in actions if a.action_type != "REASON"]
        assert len(tool_actions) == 1
        assert tool_actions[0].action_type == "FILE_READ"
        assert tool_actions[0].target == "/src/app.ts"

    def test_edit_maps_to_file_write(self):
        steps = [_make_step(0, tool_calls=[
            _tc("Edit", {"file_path": "/src/db.ts"}),
        ])]
        actions = canonicalize_steps(steps)
        tool_actions = [a for a in actions if a.action_type != "REASON"]
        assert len(tool_actions) == 1
        assert tool_actions[0].action_type == "FILE_WRITE"
        assert tool_actions[0].target == "/src/db.ts"

    def test_write_maps_to_file_write(self):
        steps = [_make_step(0, tool_calls=[
            _tc("Write", {"file_path": "/src/new.ts"}),
        ])]
        actions = canonicalize_steps(steps)
        tool_actions = [a for a in actions if a.action_type != "REASON"]
        assert tool_actions[0].action_type == "FILE_WRITE"

    def test_grep_maps_to_search(self):
        steps = [_make_step(0, tool_calls=[
            _tc("Grep", {"pattern": "TODO", "path": "/src"}),
        ])]
        actions = canonicalize_steps(steps)
        tool_actions = [a for a in actions if a.action_type != "REASON"]
        assert tool_actions[0].action_type == "SEARCH"
        assert "TODO" in tool_actions[0].target

    def test_glob_maps_to_search(self):
        steps = [_make_step(0, tool_calls=[
            _tc("Glob", {"pattern": "*.ts", "path": "/src"}),
        ])]
        actions = canonicalize_steps(steps)
        tool_actions = [a for a in actions if a.action_type != "REASON"]
        assert tool_actions[0].action_type == "SEARCH"

    def test_bash_maps_to_command(self):
        steps = [_make_step(0, tool_calls=[
            _tc("Bash", {"command": "npm run build"}),
        ])]
        actions = canonicalize_steps(steps)
        tool_actions = [a for a in actions if a.action_type != "REASON"]
        assert tool_actions[0].action_type == "COMMAND"
        assert tool_actions[0].target == "npm"

    def test_text_part_maps_to_reason(self):
        steps = [_make_step(0, parts=[{"type": "text", "content": "thinking..."}])]
        actions = canonicalize_steps(steps)
        reason_actions = [a for a in actions if a.action_type == "REASON"]
        assert len(reason_actions) == 1
        assert reason_actions[0].tool == "text"

    def test_reasoning_part_maps_to_reason(self):
        steps = [_make_step(0, parts=[{"type": "reasoning", "content": "deep thought"}])]
        actions = canonicalize_steps(steps)
        reason_actions = [a for a in actions if a.action_type == "REASON"]
        assert len(reason_actions) == 1

    def test_only_one_reason_per_step(self):
        steps = [_make_step(0, parts=[
            {"type": "text", "content": "a"},
            {"type": "reasoning", "content": "b"},
        ])]
        actions = canonicalize_steps(steps)
        reason_actions = [a for a in actions if a.action_type == "REASON"]
        assert len(reason_actions) == 1

    def test_agent_spawn(self):
        steps = [_make_step(0, tool_calls=[
            _tc("Agent", {"description": "Do something special"}),
        ])]
        actions = canonicalize_steps(steps)
        tool_actions = [a for a in actions if a.action_type != "REASON"]
        assert tool_actions[0].action_type == "AGENT_SPAWN"

    def test_unknown_tool_maps_to_command(self):
        steps = [_make_step(0, tool_calls=[
            _tc("CustomTool", {"data": "value"}),
        ])]
        actions = canonicalize_steps(steps)
        tool_actions = [a for a in actions if a.action_type != "REASON"]
        assert tool_actions[0].action_type == "COMMAND"
        assert tool_actions[0].target == "CustomTool"

    def test_latency_from_tool_timing(self):
        steps = [_make_step(0, tool_calls=[
            _tc("Read", {"file_path": "/a.ts"}, time_start=1000, time_end=3000),
        ])]
        actions = canonicalize_steps(steps)
        tool_actions = [a for a in actions if a.action_type != "REASON"]
        assert tool_actions[0].cost.latency_ms == 2000

    def test_empty_steps(self):
        assert canonicalize_steps([]) == []

    def test_search_target_includes_scope(self):
        steps = [_make_step(0, tool_calls=[
            _tc("Grep", {"pattern": "foo", "path": "/src/utils"}),
        ])]
        actions = canonicalize_steps(steps)
        tool_actions = [a for a in actions if a.action_type != "REASON"]
        assert "@" in tool_actions[0].target
        assert "foo" in tool_actions[0].target


class TestAssignEffectLabels:
    """Tests for assign_effect_labels."""

    def test_reverted_writes(self):
        """Two writes to the same file: first is reverted, second survives."""
        actions = [
            _action(0, "FILE_WRITE", "/src/app.ts", "Edit", "success"),
            _action(1, "FILE_WRITE", "/src/app.ts", "Edit", "success"),
        ]
        steps = [_make_step(0), _make_step(1)]
        assign_effect_labels(actions, steps, anchor_files={"/src/app.ts"})
        assert actions[0].effect_label == "reverted"
        assert actions[1].effect_label == "survived"

    def test_failed_commands(self):
        """Tool with status=error gets failed label."""
        actions = [
            _action(0, "COMMAND", "npm", "Bash", "error"),
        ]
        steps = [_make_step(0, tool_calls=[
            _tc("Bash", {"command": "npm build"}, status="error"),
        ])]
        assign_effect_labels(actions, steps, anchor_files=set())
        assert actions[0].effect_label == "failed"

    def test_failed_via_exit_code(self):
        """Non-zero exit code marks action as failed."""
        actions = [
            _action(0, "COMMAND", "npm", "Bash", "success"),
        ]
        steps = [_make_step(0, tool_calls=[
            _tc("Bash", {"command": "npm build"}, status="success",
                 metadata={"exit": 1}),
        ])]
        assign_effect_labels(actions, steps, anchor_files=set())
        assert actions[0].effect_label == "failed"

    def test_justified_read_target_in_patch(self):
        """Read of a file that appears in anchor files gets justified."""
        actions = [
            _action(0, "FILE_READ", "/src/app.ts", "Read", "success"),
        ]
        steps = [_make_step(0, tool_calls=[
            _tc("Read", {"file_path": "/src/app.ts"}),
        ])]
        assign_effect_labels(actions, steps, anchor_files={"/src/app.ts"})
        assert actions[0].effect_label == "justified"

    def test_justified_read_test_file(self):
        """Read of a test file is justified."""
        actions = [
            _action(0, "FILE_READ", "/src/tests/test_app.ts", "Read", "success"),
        ]
        steps = [_make_step(0, tool_calls=[
            _tc("Read", {"file_path": "/src/tests/test_app.ts"}),
        ])]
        assign_effect_labels(actions, steps, anchor_files=set())
        assert actions[0].effect_label == "justified"

    def test_unknown_read(self):
        """Read of an unrelated file stays unknown."""
        actions = [
            _action(0, "FILE_READ", "/docs/readme.txt", "Read", "success"),
        ]
        steps = [_make_step(0, tool_calls=[
            _tc("Read", {"file_path": "/docs/readme.txt"}),
        ])]
        assign_effect_labels(actions, steps, anchor_files={"/src/app.ts"})
        assert actions[0].effect_label == "unknown"

    def test_reason_stays_unknown(self):
        """REASON actions always get unknown label."""
        actions = [
            _action(0, "REASON", "", "text", "success"),
        ]
        steps = [_make_step(0)]
        assign_effect_labels(actions, steps, anchor_files=set())
        assert actions[0].effect_label == "unknown"

    def test_search_stays_unknown(self):
        actions = [
            _action(0, "SEARCH", "TODO@/src", "Grep", "success"),
        ]
        steps = [_make_step(0, tool_calls=[
            _tc("Grep", {"pattern": "TODO", "path": "/src"}),
        ])]
        assign_effect_labels(actions, steps, anchor_files=set())
        assert actions[0].effect_label == "unknown"

    def test_successful_command(self):
        """Successful command gets survived label."""
        actions = [
            _action(0, "COMMAND", "npm", "Bash", "success"),
        ]
        steps = [_make_step(0, tool_calls=[
            _tc("Bash", {"command": "npm build"}, status="success"),
        ])]
        assign_effect_labels(actions, steps, anchor_files=set())
        assert actions[0].effect_label == "survived"

    def test_validation_command_is_justified(self):
        """Validation commands are treated as justified observable work."""
        actions = [
            _action(0, "COMMAND", "pytest", "Bash", "success"),
        ]
        steps = [_make_step(0, tool_calls=[
            _tc("Bash", {"command": "pytest tests/test_app.py"}, status="success"),
        ])]
        assign_effect_labels(actions, steps, anchor_files=set())
        assert actions[0].effect_label == "justified"

    def test_single_write_is_success(self):
        """A lone write to a file is considered survived (not reverted)."""
        actions = [
            _action(0, "FILE_WRITE", "/src/app.ts", "Edit", "success"),
        ]
        steps = [_make_step(0)]
        assign_effect_labels(actions, steps, anchor_files={"/src/app.ts"})
        assert actions[0].effect_label == "survived"

    def test_three_writes_to_same_file(self):
        """Only the last write survives; the first two are reverted."""
        actions = [
            _action(0, "FILE_WRITE", "/src/app.ts", "Edit", "success"),
            _action(1, "FILE_WRITE", "/src/app.ts", "Edit", "success"),
            _action(2, "FILE_WRITE", "/src/app.ts", "Edit", "success"),
        ]
        steps = [_make_step(0), _make_step(1), _make_step(2)]
        assign_effect_labels(actions, steps, anchor_files={"/src/app.ts"})
        assert actions[0].effect_label == "reverted"
        assert actions[1].effect_label == "reverted"
        assert actions[2].effect_label == "survived"


class TestComputeActionCost:
    """Tests for compute_action_cost."""

    def test_default_rate(self):
        a = _action(tokens=1000, latency_ms=2000)
        # cost = 1000 + (2000/1000 * 50) = 1000 + 100 = 1100
        assert compute_action_cost(a) == 1100.0

    def test_custom_rate(self):
        a = _action(tokens=500, latency_ms=3000)
        # cost = 500 + (3000/1000 * 100) = 500 + 300 = 800
        assert compute_action_cost(a, token_rate=100.0) == 800.0

    def test_zero_latency(self):
        a = _action(tokens=200, latency_ms=0)
        assert compute_action_cost(a) == 200.0

    def test_zero_tokens(self):
        a = _action(tokens=0, latency_ms=1000)
        assert compute_action_cost(a) == 50.0  # 0 + 1 * 50


class TestSemanticEquivalent:
    """Tests for semantic_equivalent."""

    def test_same_type_and_target_matches(self):
        a = _action(action_type="FILE_READ", target="/src/app.ts", effect_label="survived")
        b = _action(action_type="FILE_READ", target="/src/app.ts", effect_label="survived")
        assert semantic_equivalent(a, b) is True

    def test_different_target_no_match(self):
        a = _action(action_type="FILE_READ", target="/src/app.ts", effect_label="survived")
        b = _action(action_type="FILE_READ", target="/src/db.ts", effect_label="survived")
        assert semantic_equivalent(a, b) is False

    def test_different_type_no_match(self):
        a = _action(action_type="FILE_READ", target="/src/app.ts", effect_label="survived")
        b = _action(action_type="FILE_WRITE", target="/src/app.ts", effect_label="survived")
        assert semantic_equivalent(a, b) is False

    def test_effect_label_survived_justified_compatible(self):
        a = _action(action_type="FILE_READ", target="/src/app.ts", effect_label="survived")
        b = _action(action_type="FILE_READ", target="/src/app.ts", effect_label="justified")
        assert semantic_equivalent(a, b) is True

    def test_effect_label_reverted_survived_incompatible(self):
        a = _action(action_type="FILE_WRITE", target="/src/app.ts", effect_label="reverted")
        b = _action(action_type="FILE_WRITE", target="/src/app.ts", effect_label="survived")
        assert semantic_equivalent(a, b) is False

    def test_effect_label_unknown_compatible_with_any(self):
        a = _action(action_type="FILE_READ", target="/src/app.ts", effect_label="unknown")
        b = _action(action_type="FILE_READ", target="/src/app.ts", effect_label="survived")
        assert semantic_equivalent(a, b) is True

    def test_effect_label_unknown_with_failed(self):
        a = _action(action_type="COMMAND", target="npm", effect_label="unknown")
        b = _action(action_type="COMMAND", target="npm", effect_label="failed")
        assert semantic_equivalent(a, b) is True

    def test_command_base_command_match(self):
        a = _action(action_type="COMMAND", target="npm", effect_label="survived")
        b = _action(action_type="COMMAND", target="npm", effect_label="survived")
        assert semantic_equivalent(a, b) is True

    def test_reason_always_matches_reason(self):
        a = _action(action_type="REASON", target="", effect_label="unknown")
        b = _action(action_type="REASON", target="", effect_label="unknown")
        assert semantic_equivalent(a, b) is True

    def test_fuzzy_commands_bash_grep_matches_search(self):
        a = _action(action_type="COMMAND", target="cat", effect_label="unknown",
                     args={"command": "cat /src/app.ts | grep pattern"})
        b = _action(action_type="SEARCH", target="/src/app.ts", effect_label="unknown")
        assert semantic_equivalent(a, b, fuzzy_commands=True) is True

    def test_fuzzy_commands_disabled_no_cross_type(self):
        a = _action(action_type="COMMAND", target="cat", effect_label="unknown",
                     args={"command": "cat /src/app.ts | grep pattern"})
        b = _action(action_type="SEARCH", target="/src/app.ts", effect_label="unknown")
        assert semantic_equivalent(a, b, fuzzy_commands=False) is False

    def test_agent_spawn_matches_any_spawn(self):
        a = _action(action_type="AGENT_SPAWN", target="task A", effect_label="unknown")
        b = _action(action_type="AGENT_SPAWN", target="task B", effect_label="unknown")
        assert semantic_equivalent(a, b) is True


class TestReduceCompositeCommand:
    """Tests for reduce_composite_command."""

    def test_cat_grep_reduces_to_search(self):
        a = _action(action_type="COMMAND", target="cat",
                     args={"command": "cat /src/file.ts | grep pattern"})
        result = reduce_composite_command(a)
        assert result is not None
        assert result.action_type == "SEARCH"

    def test_ambiguous_stays_none(self):
        a = _action(action_type="COMMAND", target="npm",
                     args={"command": "npm run build"})
        result = reduce_composite_command(a)
        assert result is None

    def test_non_command_returns_none(self):
        a = _action(action_type="FILE_READ", target="/src/app.ts")
        result = reduce_composite_command(a)
        assert result is None

    def test_empty_command_returns_none(self):
        a = _action(action_type="COMMAND", target="", args={"command": ""})
        result = reduce_composite_command(a)
        assert result is None

    def test_sed_reduces_to_file_write(self):
        a = _action(action_type="COMMAND", target="cat",
                     args={"command": "cat /src/data.json | sed 's/old/new/'"})
        result = reduce_composite_command(a)
        assert result is not None
        assert result.action_type == "FILE_WRITE"

    def test_cat_alone_reduces_to_file_read(self):
        a = _action(action_type="COMMAND", target="cat",
                     args={"command": "cat /src/config.ts"})
        result = reduce_composite_command(a)
        assert result is not None
        assert result.action_type == "FILE_READ"


# ===========================================================================
# 2. Alignment (trajectory_visualizer.converge.alignment)
# ===========================================================================

class TestAlignTrajectories:
    """Tests for align_trajectories."""

    def test_perfect_match(self):
        ref = [
            _action(0, "FILE_READ", "/a.ts", effect_label="justified"),
            _action(1, "FILE_WRITE", "/a.ts", effect_label="survived"),
        ]
        cmp = [
            _action(0, "FILE_READ", "/a.ts", effect_label="justified"),
            _action(1, "FILE_WRITE", "/a.ts", effect_label="survived"),
        ]
        result = align_trajectories(ref, cmp)
        assert len(result["matched_pairs"]) == 2
        assert result["unrecovered"] == []
        assert result["extra"] == []

    def test_partial_match(self):
        ref = [
            _action(0, "FILE_READ", "/a.ts", effect_label="justified"),
            _action(1, "FILE_WRITE", "/a.ts", effect_label="survived"),
        ]
        cmp = [
            _action(0, "FILE_READ", "/a.ts", effect_label="justified"),
            _action(1, "SEARCH", "TODO@/src", effect_label="unknown"),
            _action(2, "FILE_WRITE", "/a.ts", effect_label="survived"),
        ]
        result = align_trajectories(ref, cmp)
        assert len(result["matched_pairs"]) == 2
        assert result["unrecovered"] == []
        assert result["extra"] == [1]  # the extra SEARCH

    def test_effect_incompatible_rejection(self):
        """reverted vs success actions should not match."""
        ref = [
            _action(0, "FILE_WRITE", "/a.ts", effect_label="survived"),
        ]
        cmp = [
            _action(0, "FILE_WRITE", "/a.ts", effect_label="reverted"),
        ]
        result = align_trajectories(ref, cmp)
        assert len(result["matched_pairs"]) == 0
        assert result["unrecovered"] == [0]
        assert result["extra"] == [0]

    def test_one_to_many_single_match(self):
        """When multiple compared actions could match, only one is matched."""
        ref = [
            _action(0, "FILE_READ", "/a.ts", effect_label="unknown"),
        ]
        cmp = [
            _action(0, "FILE_READ", "/a.ts", effect_label="unknown"),
            _action(1, "FILE_READ", "/a.ts", effect_label="unknown"),
        ]
        result = align_trajectories(ref, cmp)
        assert len(result["matched_pairs"]) == 1
        assert result["matched_pairs"][0][0] == 0  # ref index is 0
        assert len(result["extra"]) == 1  # one cmp action is extra

    def test_reason_excluded(self):
        """REASON actions should not participate in alignment."""
        ref = [
            _action(0, "REASON", "", effect_label="unknown"),
            _action(1, "FILE_READ", "/a.ts", effect_label="justified"),
        ]
        cmp = [
            _action(0, "REASON", "", effect_label="unknown"),
            _action(1, "FILE_READ", "/a.ts", effect_label="justified"),
        ]
        result = align_trajectories(ref, cmp)
        # Only the FILE_READ should be matched
        assert len(result["matched_pairs"]) == 1
        assert result["matched_pairs"][0] == (1, 1)

    def test_empty_sequences(self):
        result = align_trajectories([], [])
        assert result["matched_pairs"] == []
        assert result["unrecovered"] == []
        assert result["extra"] == []

    def test_no_match_possible(self):
        ref = [_action(0, "FILE_READ", "/a.ts", effect_label="unknown")]
        cmp = [_action(0, "FILE_WRITE", "/b.ts", effect_label="survived")]
        result = align_trajectories(ref, cmp)
        assert len(result["matched_pairs"]) == 0
        assert result["unrecovered"] == [0]
        assert result["extra"] == [0]


class TestComputeAlignmentMetrics:
    """Tests for compute_alignment_metrics."""

    def test_perfect_alignment(self):
        ref = [_action(0, "FILE_READ", "/a.ts", effect_label="unknown")]
        cmp = [_action(0, "FILE_READ", "/a.ts", effect_label="unknown")]
        alignment = {"matched_pairs": [(0, 0)], "unrecovered": [], "extra": []}
        metrics = compute_alignment_metrics(alignment, ref, cmp)
        assert metrics["reference_recall"] == 1.0
        assert metrics["behavioral_precision"] == 1.0
        assert metrics["alignment_f1"] == 1.0

    def test_precision_and_recall_bounded(self):
        ref = [_action(0, "FILE_READ", "/a.ts", effect_label="unknown", tokens=100)]
        cmp = [
            _action(0, "FILE_READ", "/a.ts", effect_label="unknown", tokens=100),
            _action(1, "FILE_READ", "/b.ts", effect_label="unknown", tokens=100),
        ]
        alignment = {"matched_pairs": [(0, 0)], "unrecovered": [], "extra": [1]}
        metrics = compute_alignment_metrics(alignment, ref, cmp)
        assert 0 <= metrics["reference_recall"] <= 1.0
        assert 0 <= metrics["behavioral_precision"] <= 1.0

    def test_empty_sequences(self):
        alignment = {"matched_pairs": [], "unrecovered": [], "extra": []}
        metrics = compute_alignment_metrics(alignment, [], [])
        assert metrics["reference_recall"] == 0.0
        assert metrics["behavioral_precision"] == 0.0
        assert metrics["alignment_f1"] == 0.0

    def test_overhead_ratio(self):
        ref = [_action(0, "FILE_READ", "/a.ts", tokens=500, latency_ms=1000)]
        cmp = [
            _action(0, "FILE_READ", "/a.ts", tokens=500, latency_ms=1000),
            _action(1, "FILE_READ", "/b.ts", tokens=500, latency_ms=1000),
        ]
        alignment = {"matched_pairs": [(0, 0)], "unrecovered": [], "extra": [1]}
        metrics = compute_alignment_metrics(alignment, ref, cmp)
        # compared has 2x the cost of reference
        assert metrics["overhead_ratio"] == 2.0


class TestComputeHarmfulDivergence:
    """Tests for compute_harmful_divergence."""

    def test_failed_and_reverted_extras_counted(self):
        compared = [
            _action(0, "FILE_READ", "/a.ts", effect_label="unknown", tokens=100),
            _action(1, "COMMAND", "npm", effect_label="failed", tokens=200, latency_ms=500),
            _action(2, "FILE_WRITE", "/a.ts", effect_label="reverted", tokens=300, latency_ms=1000),
        ]
        extra_indices = [1, 2]
        result = compute_harmful_divergence(extra_indices, compared)
        assert result["harmful_cost"]["tokens"] == 500  # 200 + 300
        assert result["harmful_cost"]["latency_ms"] == 1500  # 500 + 1000

    def test_unknown_extras_not_counted(self):
        compared = [
            _action(0, "FILE_READ", "/a.ts", effect_label="unknown", tokens=100),
            _action(1, "SEARCH", "TODO", effect_label="unknown", tokens=200),
        ]
        extra_indices = [0, 1]
        result = compute_harmful_divergence(extra_indices, compared)
        assert result["harmful_cost"]["tokens"] == 0
        assert result["harmful_cost"]["latency_ms"] == 0

    def test_harmful_ratio_bounded(self):
        compared = [
            _action(0, "COMMAND", "npm", effect_label="failed", tokens=1000, latency_ms=0),
        ]
        extra_indices = [0]
        result = compute_harmful_divergence(extra_indices, compared)
        assert 0 <= result["harmful_ratio"] <= 1.0

    def test_empty_extras(self):
        compared = [_action(0, "FILE_READ", "/a.ts", effect_label="unknown")]
        result = compute_harmful_divergence([], compared)
        assert result["harmful_cost"]["tokens"] == 0

    def test_no_compared_actions(self):
        result = compute_harmful_divergence([], [])
        assert result["harmful_ratio"] == 0.0


# ===========================================================================
# 3. Milestones (trajectory_visualizer.converge.milestones)
# ===========================================================================

class TestExtractMilestones:
    """Tests for extract_milestones."""

    def test_all_milestones_present(self):
        actions = [
            _action(0, "FILE_READ", "/src/app.ts", effect_label="justified"),
            _action(1, "FILE_WRITE", "/src/app.ts", effect_label="reverted"),
            _action(2, "FILE_WRITE", "/src/app.ts", effect_label="survived"),
            _action(3, "COMMAND", "test", effect_label="survived"),
        ]
        m = extract_milestones(actions)
        assert m["first_relevant_file"] == 0
        assert m["first_edit"] == 1
        assert m["first_surviving_edit"] == 2
        assert m["first_passing_validation"] == 3
        assert m["final_patch"] == 2

    def test_absent_milestones_for_read_only_task(self):
        """A task with only reads should have no edit milestones."""
        actions = [
            _action(0, "FILE_READ", "/src/app.ts", effect_label="justified"),
            _action(1, "SEARCH", "TODO@/src", effect_label="unknown"),
        ]
        m = extract_milestones(actions)
        assert m["first_relevant_file"] == 0
        assert m["first_edit"] is None
        assert m["first_surviving_edit"] is None
        assert m["first_passing_validation"] is None
        assert m["final_patch"] is None

    def test_validation_pattern_matching(self):
        """build/lint/test/check/verify should match validation patterns."""
        for cmd in ["build", "lint", "test", "check", "verify"]:
            actions = [
                _action(0, "COMMAND", cmd, effect_label="survived"),
            ]
            m = extract_milestones(actions)
            assert m["first_passing_validation"] == 0, f"{cmd} should match validation"

    def test_validation_non_matching(self):
        actions = [
            _action(0, "COMMAND", "echo", effect_label="survived"),
        ]
        m = extract_milestones(actions)
        assert m["first_passing_validation"] is None

    def test_first_surviving_edit_requires_success_label(self):
        """first_surviving_edit requires effect_label=success, not reverted."""
        actions = [
            _action(0, "FILE_WRITE", "/src/app.ts", effect_label="reverted"),
            _action(1, "FILE_WRITE", "/src/app.ts", effect_label="survived"),
        ]
        m = extract_milestones(actions)
        assert m["first_edit"] == 0
        assert m["first_surviving_edit"] == 1

    def test_reason_actions_skipped(self):
        actions = [
            _action(0, "REASON", "", effect_label="unknown"),
            _action(1, "FILE_READ", "/src/app.ts", effect_label="justified"),
        ]
        m = extract_milestones(actions)
        assert m["first_relevant_file"] == 1

    def test_empty_actions(self):
        m = extract_milestones([])
        assert all(v is None for v in m.values())

    def test_first_relevant_file_requires_justified_or_success(self):
        """An unknown-labeled read does not count as first_relevant_file."""
        actions = [
            _action(0, "FILE_READ", "/docs/readme.txt", effect_label="unknown"),
            _action(1, "FILE_READ", "/src/app.ts", effect_label="justified"),
        ]
        m = extract_milestones(actions)
        assert m["first_relevant_file"] == 1


class TestComputeMilestoneDeltas:
    """Tests for compute_milestone_deltas."""

    def test_both_present_gives_delta(self):
        ref = {"first_edit": 3, "first_relevant_file": 1}
        cmp = {"first_edit": 5, "first_relevant_file": 2}
        deltas = compute_milestone_deltas(ref, cmp)
        assert deltas["first_edit_delta"] == 2
        assert deltas["first_relevant_file_delta"] == 1

    def test_one_absent_gives_null(self):
        ref = {"first_edit": 3, "first_passing_validation": None}
        cmp = {"first_edit": None, "first_passing_validation": 5}
        deltas = compute_milestone_deltas(ref, cmp)
        assert deltas["first_edit_delta"] is None
        assert deltas["first_passing_validation_delta"] is None

    def test_negative_delta(self):
        ref = {"first_edit": 5}
        cmp = {"first_edit": 2}
        deltas = compute_milestone_deltas(ref, cmp)
        assert deltas["first_edit_delta"] == -3


class TestSegmentByMilestones:
    """Tests for segment_by_milestones."""

    def test_correct_partitioning(self):
        actions = [
            _action(0, "FILE_READ", "/a.ts", effect_label="justified"),
            _action(1, "FILE_WRITE", "/a.ts", effect_label="survived"),
            _action(2, "COMMAND", "test", effect_label="survived"),
        ]
        milestones = {"first_relevant_file": 0, "first_edit": 1, "final_patch": 1,
                       "first_surviving_edit": 1, "first_passing_validation": 2}
        segments = segment_by_milestones(actions, milestones)
        assert len(segments) > 1
        # Each segment should have a label with →
        for seg in segments:
            assert "→" in seg["label"]

    def test_no_milestones(self):
        """All milestones None → single start→end segment."""
        actions = [
            _action(0, "FILE_READ", "/a.ts", effect_label="unknown"),
        ]
        milestones = {"first_relevant_file": None, "first_edit": None,
                       "first_surviving_edit": None, "first_passing_validation": None,
                       "final_patch": None}
        segments = segment_by_milestones(actions, milestones)
        assert len(segments) == 1
        assert "start" in segments[0]["label"]
        assert "end" in segments[0]["label"]

    def test_empty_actions(self):
        milestones = {"first_relevant_file": None, "first_edit": None,
                       "first_surviving_edit": None, "first_passing_validation": None,
                       "final_patch": None}
        segments = segment_by_milestones([], milestones)
        assert len(segments) == 1


class TestCompareSegments:
    """Tests for compare_segments."""

    def test_same_order_paired(self):
        ref_actions = [
            _action(0, "FILE_READ", "/a.ts", effect_label="justified"),
            _action(1, "FILE_WRITE", "/a.ts", effect_label="survived"),
        ]
        cmp_actions = [
            _action(0, "FILE_READ", "/a.ts", effect_label="justified"),
            _action(1, "FILE_WRITE", "/a.ts", effect_label="survived"),
        ]
        milestones = {"first_relevant_file": 0, "first_edit": 1,
                       "first_surviving_edit": 1, "first_passing_validation": None,
                       "final_patch": 1}
        ref_segs = segment_by_milestones(ref_actions, milestones)
        cmp_segs = segment_by_milestones(cmp_actions, milestones)
        result = compare_segments(ref_segs, cmp_segs, milestones, milestones,
                                   ref_actions, cmp_actions)
        assert result["milestone_order_matches"] is True
        assert "segment_comparison" in result

    def test_different_order_separate(self):
        ref_actions = [
            _action(0, "FILE_READ", "/a.ts", effect_label="justified"),
            _action(1, "FILE_WRITE", "/a.ts", effect_label="survived"),
        ]
        cmp_actions = [
            _action(0, "FILE_WRITE", "/a.ts", effect_label="survived"),
            _action(1, "FILE_READ", "/a.ts", effect_label="justified"),
        ]
        ref_milestones = {"first_relevant_file": 0, "first_edit": 1,
                           "first_surviving_edit": 1, "first_passing_validation": None,
                           "final_patch": 1}
        cmp_milestones = {"first_relevant_file": 1, "first_edit": 0,
                           "first_surviving_edit": 0, "first_passing_validation": None,
                           "final_patch": 0}
        ref_segs = segment_by_milestones(ref_actions, ref_milestones)
        cmp_segs = segment_by_milestones(cmp_actions, cmp_milestones)
        result = compare_segments(ref_segs, cmp_segs, ref_milestones, cmp_milestones,
                                   ref_actions, cmp_actions)
        assert result["milestone_order_matches"] is False
        assert "reference_segments" in result
        assert "compared_segments" in result


# ===========================================================================
# 4. Divergence (trajectory_visualizer.converge.divergence)
# ===========================================================================

class TestClassifyDivergences:
    """Tests for classify_divergences."""

    def test_write_retry_split(self):
        """Reverted FILE_WRITE should classify as write_retry subtype with confidence."""
        extra = [
            _action(0, "FILE_WRITE", "/a.ts", effect_label="reverted", tokens=100),
        ]
        matched = [
            _action(1, "FILE_WRITE", "/a.ts", effect_label="survived"),
        ]
        all_compared = extra + matched
        patterns = classify_divergences(extra, matched, all_compared)
        types = [p["type"] for p in patterns]
        assert "reverted_and_rewritten" in types or "iterative_refinement" in types
        wr = [p for p in patterns if p.get("parent_type") == "write_retry"][0]
        assert wr["evidence_level"] == "single_pair_hypothesis"
        assert "confidence" in wr

    def test_error_recovery_overhead(self):
        """Failed action followed by retry with same type+target."""
        extra = [
            _action(0, "COMMAND", "npm", effect_label="failed", tokens=200),
            _action(1, "COMMAND", "npm", effect_label="survived", tokens=200),
        ]
        matched = []
        all_compared = extra
        patterns = classify_divergences(extra, matched, all_compared)
        types = [p["type"] for p in patterns]
        assert "error_recovery_overhead" in types
        ero = [p for p in patterns if p["type"] == "error_recovery_overhead"][0]
        assert ero["evidence_level"] == "single_pair_hypothesis"
        assert len(ero["steps"]) == 2

    def test_premature_validation_on_write_run(self):
        """Validation command before first write in a write-run."""
        extra = [
            _action(0, "COMMAND", "test", effect_label="survived", tokens=100),
        ]
        matched = []
        all_compared = [
            _action(0, "COMMAND", "test", effect_label="survived", tokens=100),
            _action(1, "FILE_WRITE", "/a.ts", effect_label="survived"),
        ]
        patterns = classify_divergences(extra, matched, all_compared)
        types = [p["type"] for p in patterns]
        assert "premature_validation" in types

    def test_premature_validation_skipped_on_non_edit(self):
        """No premature_validation when there are no writes at all."""
        extra = [
            _action(0, "COMMAND", "test", effect_label="survived", tokens=100),
        ]
        matched = []
        all_compared = [
            _action(0, "COMMAND", "test", effect_label="survived"),
            _action(1, "FILE_READ", "/a.ts", effect_label="unknown"),
        ]
        patterns = classify_divergences(extra, matched, all_compared)
        types = [p["type"] for p in patterns]
        assert "premature_validation" not in types

    def test_broad_exploration(self):
        """Unmatched FILE_READ with unknown label and dead-end target."""
        extra = [
            _action(0, "FILE_READ", "/random/file.ts", effect_label="unknown", tokens=100),
        ]
        matched = [
            _action(1, "FILE_WRITE", "/a.ts", effect_label="survived"),
        ]
        all_compared = extra + matched
        patterns = classify_divergences(extra, matched, all_compared)
        types = [p["type"] for p in patterns]
        assert "broad_exploration" in types

    def test_redundant_search_repeats_matched(self):
        """Extra SEARCH that repeats a matched search pattern."""
        extra = [
            _action(2, "SEARCH", "TODO@/src", effect_label="unknown", tokens=100),
        ]
        matched = [
            _action(0, "SEARCH", "TODO@/src", effect_label="unknown"),
        ]
        all_compared = [
            _action(0, "SEARCH", "TODO@/src", effect_label="unknown"),
            _action(2, "SEARCH", "TODO@/src", effect_label="unknown", tokens=100),
        ]
        patterns = classify_divergences(extra, matched, all_compared)
        types = [p["type"] for p in patterns]
        assert "redundant_search" in types

    def test_redundant_search_multiple_extras(self):
        """Multiple extra SEARCH with same pattern — second one is redundant."""
        extra = [
            _action(1, "SEARCH", "TODO@/src", effect_label="unknown", tokens=100),
            _action(3, "SEARCH", "TODO@/src", effect_label="unknown", tokens=100),
        ]
        matched = []
        all_compared = extra[:]
        patterns = classify_divergences(extra, matched, all_compared)
        types = [p["type"] for p in patterns]
        assert "redundant_search" in types

    def test_dead_end_branch(self):
        """Sequence of 2+ consecutive unclassified FILE_READ/SEARCH with unknown labels.

        To reach the dead_end_branch classifier, actions must not be caught by
        earlier classifiers (broad_exploration catches unknown reads whose target
        is NOT in matched_targets, and redundant_search catches repeated SEARCH
        targets). So we use unique search patterns not already seen in
        all_compared and reads whose targets appear in matched_targets.
        """
        # Use FILE_READ targets that ARE in matched_targets so broad_exploration
        # skips them, and unique SEARCH targets not already in all_compared so
        # redundant_search skips them.
        extra = [
            _action(0, "FILE_READ", "/a.ts", effect_label="unknown", tokens=100),
            _action(1, "FILE_READ", "/a.ts", effect_label="unknown", tokens=100),
            _action(2, "FILE_READ", "/a.ts", effect_label="unknown", tokens=100),
        ]
        # matched targets include /a.ts so broad_exploration won't fire
        matched = [
            _action(10, "FILE_WRITE", "/a.ts", effect_label="survived"),
        ]
        all_compared = extra + matched
        patterns = classify_divergences(extra, matched, all_compared)
        types = [p["type"] for p in patterns]
        assert "dead_end_branch" in types
        deb = [p for p in patterns if p["type"] == "dead_end_branch"][0]
        assert deb["evidence_level"] == "single_pair_hypothesis"
        assert len(deb["steps"]) >= 2

    def test_evidence_level_on_all_patterns(self):
        """All returned patterns must have evidence_level=single_pair_hypothesis."""
        extra = [
            _action(0, "FILE_WRITE", "/a.ts", effect_label="reverted", tokens=100),
            _action(1, "COMMAND", "npm", effect_label="failed", tokens=200),
            _action(2, "COMMAND", "npm", effect_label="survived", tokens=200),
            _action(3, "FILE_READ", "/x.ts", effect_label="unknown", tokens=100),
        ]
        matched = [
            _action(10, "FILE_WRITE", "/a.ts", effect_label="survived"),
        ]
        all_compared = extra + matched
        patterns = classify_divergences(extra, matched, all_compared)
        for p in patterns:
            assert p["evidence_level"] == "single_pair_hypothesis"

    def test_empty_extras(self):
        patterns = classify_divergences([], [], [])
        assert patterns == []

    def test_write_retry_does_not_classify_success_writes(self):
        """Only reverted writes should be classified as write_retry."""
        extra = [
            _action(0, "FILE_WRITE", "/a.ts", effect_label="survived", tokens=100),
        ]
        matched = []
        all_compared = extra
        patterns = classify_divergences(extra, matched, all_compared)
        types = [p["type"] for p in patterns]
        assert "write_retry" not in types
        parent_types = [p.get("parent_type") for p in patterns]
        assert "write_retry" not in parent_types
