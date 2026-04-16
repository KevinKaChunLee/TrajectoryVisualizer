"""Tests for [H] semantic detectors."""

from __future__ import annotations

from trajectory_visualizer.core.detection import DetectorContext
from trajectory_visualizer.insight.detectors.semantic import (
    debug_wo_hypothesis,
    memory_contamination,
    phase_oscillation,
    premature_implementation,
    prompt_skim,
    semantic_fruitless_exploration,
    semantic_plan_stall,
)

from tests.detectors.conftest import step


def _ctx(labels: dict[int, dict[str, str]], **kw) -> DetectorContext:
    return DetectorContext(labels=labels, **kw)


# ---------------------------------------------------------------------------
# All [H] detectors return empty when labels absent
# ---------------------------------------------------------------------------

def test_all_h_detectors_return_empty_without_labels() -> None:
    empty_ctx = DetectorContext()
    steps: list = []
    for mod in (
        phase_oscillation,
        premature_implementation,
        semantic_fruitless_exploration,
        semantic_plan_stall,
        debug_wo_hypothesis,
        prompt_skim,
        memory_contamination,
    ):
        assert mod.detect(steps, empty_ctx) == []


# ---------------------------------------------------------------------------
# phase-oscillation
# ---------------------------------------------------------------------------

def test_phase_oscillation_fires_on_three_transitions_same_pair() -> None:
    labels = {
        0: {"phase": "understand"},
        1: {"phase": "implement"},
        2: {"phase": "understand"},
        3: {"phase": "implement"},
        4: {"phase": "understand"},
        5: {"phase": "implement"},
    }
    ctx = _ctx(labels)
    steps = [step(i, "REASON") for i in range(6)]
    hits = phase_oscillation.detect(steps, ctx)
    assert len(hits) == 1
    assert hits[0].evidence["transitions_in_window"] >= 3


# ---------------------------------------------------------------------------
# premature-implementation
# ---------------------------------------------------------------------------

def test_premature_implementation_fires_when_implement_first() -> None:
    labels = {0: {"phase": "implement"}, 1: {"phase": "plan"}}
    ctx = _ctx(labels)
    steps = [step(i, "REASON") for i in range(2)]
    hits = premature_implementation.detect(steps, ctx)
    assert len(hits) == 1


def test_premature_implementation_not_fires_when_plan_first() -> None:
    labels = {0: {"phase": "plan"}, 1: {"phase": "implement"}}
    ctx = _ctx(labels)
    steps = [step(i, "REASON") for i in range(2)]
    assert premature_implementation.detect(steps, ctx) == []


# ---------------------------------------------------------------------------
# semantic-fruitless-exploration
# ---------------------------------------------------------------------------

def test_semantic_fruitless_exploration_fires_on_unused_code_reads() -> None:
    labels = {i: {"phase": "understand", "action": "code_reading"} for i in range(5)}
    labels[5] = {"phase": "implement", "action": "implement_fix"}
    ctx = _ctx(labels)
    steps = [
        step(0, "FILE_READ", target="a.py", tool="Read"),
        step(1, "FILE_READ", target="b.py", tool="Read"),
        step(2, "FILE_READ", target="c.py", tool="Read"),
        step(3, "FILE_READ", target="d.py", tool="Read"),
        step(4, "FILE_READ", target="e.py", tool="Read"),
        step(5, "FILE_WRITE", target="z.py", tool="Edit"),
    ]
    hits = semantic_fruitless_exploration.detect(steps, ctx)
    assert len(hits) == 1
    assert len(hits[0].evidence["unused_files"]) >= 4


# ---------------------------------------------------------------------------
# semantic-plan-stall
# ---------------------------------------------------------------------------

def test_semantic_plan_stall_fires() -> None:
    labels = {i: {"phase": "plan"} for i in range(5)}
    ctx = _ctx(labels)
    steps = [step(i, "REASON") for i in range(5)]
    hits = semantic_plan_stall.detect(steps, ctx)
    assert len(hits) == 1


# ---------------------------------------------------------------------------
# debug-wo-hypothesis
# ---------------------------------------------------------------------------

def test_debug_wo_hypothesis_fires_on_three_reproduces_no_rca() -> None:
    labels = {
        0: {"phase": "debug", "action": "debug_reproduce"},
        1: {"phase": "debug", "action": "debug_reproduce"},
        2: {"phase": "debug", "action": "debug_reproduce"},
    }
    ctx = _ctx(labels)
    steps = [step(i, "COMMAND") for i in range(3)]
    hits = debug_wo_hypothesis.detect(steps, ctx)
    assert len(hits) == 1


def test_debug_wo_hypothesis_not_fires_with_root_cause() -> None:
    labels = {
        0: {"phase": "debug", "action": "debug_reproduce"},
        1: {"phase": "debug", "action": "root_cause_analysis"},
        2: {"phase": "debug", "action": "debug_reproduce"},
        3: {"phase": "debug", "action": "debug_reproduce"},
    }
    ctx = _ctx(labels)
    steps = [step(i, "COMMAND") for i in range(4)]
    assert debug_wo_hypothesis.detect(steps, ctx) == []


# ---------------------------------------------------------------------------
# prompt-skim
# ---------------------------------------------------------------------------

def test_prompt_skim_fires_when_prompt_never_rereferenced() -> None:
    labels = {0: {"phase": "understand"}, 1: {"phase": "implement"}}
    ctx = _ctx(labels)
    steps = [step(0, "REASON"), step(1, "FILE_WRITE", target="a.py", tool="Edit")]
    hits = prompt_skim.detect(steps, ctx)
    assert len(hits) == 1


def test_prompt_skim_not_fires_when_reread_seen() -> None:
    labels = {
        0: {"phase": "understand"},
        1: {"phase": "understand", "action": "reread_prompt"},
    }
    ctx = _ctx(labels)
    steps = [step(0, "REASON"), step(1, "REASON")]
    assert prompt_skim.detect(steps, ctx) == []


# ---------------------------------------------------------------------------
# memory-contamination
# ---------------------------------------------------------------------------

def test_memory_contamination_fires_when_labeled() -> None:
    labels = {0: {"action": "memory_contamination"}}
    ctx = _ctx(labels)
    steps = [step(0, "FILE_WRITE", target="CLAUDE.md", tool="Edit")]
    hits = memory_contamination.detect(steps, ctx)
    assert len(hits) == 1


def test_memory_contamination_not_fires_for_benign_write() -> None:
    labels = {0: {"action": "documentation_update"}}
    ctx = _ctx(labels)
    steps = [step(0, "FILE_WRITE", target="CLAUDE.md", tool="Edit")]
    assert memory_contamination.detect(steps, ctx) == []
