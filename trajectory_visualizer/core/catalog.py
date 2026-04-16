"""Pattern catalog: single source of truth for all anti-pattern and divergence-pattern definitions.

Every record is a frozen dataclass. Downstream consumers (insight detectors, converge
detectors, batch runner, paper-table generators) import from this module rather than
hardcoding detector IDs, thresholds, or bands. Renaming a detector is a single-file change.

Record counts (locked to the paper's appendix catalog and Table 4):
    20 structural deterministic anti-patterns ([S])
     7 intent-dependent hypothesis anti-patterns ([H])
     6 cross-trajectory divergence patterns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


Band = str  # "[S]" | "[H]" | "divergence"
Phase = str  # "intake" | "understand" | "plan" | "implement" | "validate" | "debug" | "report" | "cross-cutting" | None
Tier = str  # "high" | "med" | "low" (divergence only)
Gating = str  # "config-gated" | "tool-gated" | "capability-gated" | "weaker"


@dataclass(frozen=True)
class PatternRecord:
    """Immutable metadata for one pattern detector.

    Fields:
      id: stable kebab-case identifier used everywhere downstream.
      name: human-readable label matching the paper's table row.
      band: "[S]" | "[H]" | "divergence".
      phase: lifecycle phase for anti-patterns; None for divergence.
      gating: tuple of gating tags (empty when not gated).
      requires_semantic_labels: True for [H] detectors that need the labeler.
      thresholds: default threshold map (CLI/tests may override).
      tier: divergence only — "high" | "med" | "low".
      required_signals: divergence only — signals the detector consumes (Table 4 column).
      description: paper's operational definition, copied verbatim.
      legacy_aliases: old detector names in prior code (for migration).
    """

    id: str
    name: str
    band: Band
    description: str
    phase: Phase | None = None
    gating: tuple[Gating, ...] = ()
    requires_semantic_labels: bool = False
    thresholds: Mapping[str, int | float] = field(default_factory=lambda: MappingProxyType({}))
    tier: Tier | None = None
    required_signals: tuple[str, ...] = ()
    legacy_aliases: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# [S] Structural deterministic anti-patterns (20)
# Definitions copied from latex_draft/offline/sections/appendix_catalog.tex.
# ---------------------------------------------------------------------------

_S_PATTERNS: tuple[PatternRecord, ...] = (
    # Phase 0: Intake
    PatternRecord(
        id="memory-bypass",
        name="Memory bypass",
        band="[S]",
        phase="intake",
        gating=("config-gated",),
        thresholds=MappingProxyType({}),
        description=(
            "A designated memory/instruction file (e.g., CLAUDE.md) exists in the "
            "workspace and is never read before the first code action."
        ),
    ),
    PatternRecord(
        id="premature-code-action",
        name="Premature code action",
        band="[S]",
        phase="intake",
        description=(
            "The first source-code FILE_WRITE occurs before any repository FILE_READ "
            "or SEARCH."
        ),
    ),
    # Phase 1: Understand
    PatternRecord(
        id="empty-result-churn",
        name="Empty-result churn",
        band="[S]",
        phase="understand",
        thresholds=MappingProxyType({"min_consecutive_empty": 3}),
        description=(
            ">=3 consecutive SEARCH steps return zero matches."
        ),
    ),
    PatternRecord(
        id="search-loop",
        name="Search loop",
        band="[S]",
        phase="understand",
        thresholds=MappingProxyType({"min_consecutive_steps": 4}),
        description=(
            ">=4 consecutive SEARCH/FILE_READ steps with no FILE_WRITE or validation "
            "COMMAND in between."
        ),
        legacy_aliases=("fruitless_streak", "search_exploration_loop"),
    ),
    PatternRecord(
        id="re-read-churn",
        name="Re-read churn",
        band="[S]",
        phase="understand",
        thresholds=MappingProxyType({"min_reads": 3, "window_steps": 10}),
        description=(
            "Same file, or overlapping line range, is read >=3 times within a short "
            "window with no intervening write to that file."
        ),
    ),
    # Phase 2: Plan
    PatternRecord(
        id="plan-stall",
        name="Plan stall",
        band="[S]",
        phase="plan",
        gating=("tool-gated",),
        thresholds=MappingProxyType({"min_plan_steps": 5}),
        description=(
            ">=5 planning/TodoWrite actions without any implement-phase step. "
            "Requires a structured todo tool."
        ),
    ),
    PatternRecord(
        id="plan-thrash",
        name="Plan thrash",
        band="[S]",
        phase="plan",
        gating=("tool-gated",),
        thresholds=MappingProxyType({"min_rewrites": 3, "min_item_turnover": 0.5}),
        description=(
            "Repeated TodoWrite rewrites with high item-set turnover and no "
            "downstream execution."
        ),
    ),
    PatternRecord(
        id="plan-less-execution",
        name="Plan-less execution",
        band="[S]",
        phase="plan",
        gating=("tool-gated", "weaker"),
        thresholds=MappingProxyType({"min_file_writes": 5}),
        description=(
            "Long trajectory with >=5 FILE_WRITE steps but zero TodoWrite calls. "
            "Fires only when the scaffold exposes a planning tool."
        ),
    ),
    # Phase 3: Implement
    PatternRecord(
        id="edit-without-inspection",
        name="Edit without inspection",
        band="[S]",
        phase="implement",
        description=(
            "First FILE_WRITE to a file has no prior FILE_READ or SEARCH hit on that "
            "file."
        ),
    ),
    PatternRecord(
        id="edit-thrash",
        name="Edit thrash",
        band="[S]",
        phase="implement",
        thresholds=MappingProxyType({"min_writes": 3, "window_steps": 10}),
        description=(
            "Same file is written >=3 times within a short window with oscillating "
            "(non-monotonic) changes."
        ),
    ),
    # Phase 4: Validate
    PatternRecord(
        id="late-validation",
        name="Late validation",
        band="[S]",
        phase="validate",
        thresholds=MappingProxyType({"min_implement_steps_before_validate": 10}),
        description=(
            "No validation COMMAND fires until after >=N implement steps, with no "
            "incremental checks in between."
        ),
    ),
    PatternRecord(
        id="validation-avoidance",
        name="Validation avoidance",
        band="[S]",
        phase="validate",
        thresholds=MappingProxyType({"implement_to_validate_ratio": 5.0}),
        description=(
            "Implement:validate step ratio > 5:1, or the run ends after a long edit "
            "streak with no validation. Uses structural phase detection only "
            "(no semantic labels)."
        ),
    ),
    PatternRecord(
        id="test-retry-loop",
        name="Test retry loop",
        band="[S]",
        phase="validate",
        thresholds=MappingProxyType({"min_retries": 2}),
        description=(
            "The same validation COMMAND with the same failure signature is rerun "
            "without a relevant intervening edit."
        ),
    ),
    # Phase 5: Debug / Recover
    PatternRecord(
        id="error-spiral",
        name="Error spiral",
        band="[S]",
        phase="debug",
        thresholds=MappingProxyType({"min_recurrences": 3}),
        description=(
            "Same (tool, error_signature) pair recurs >=3 times with no observable "
            "change in approach."
        ),
        legacy_aliases=("error_recurrence",),
    ),
    PatternRecord(
        id="recovery-free-retry",
        name="Recovery-free retry",
        band="[S]",
        phase="debug",
        description=(
            "A failed action is immediately retried with no intervening inspection, "
            "edit, or parameter change."
        ),
    ),
    # Phase 6: Report / Complete
    PatternRecord(
        id="verification-skip",
        name="Verification skip",
        band="[S]",
        phase="report",
        thresholds=MappingProxyType({"tail_window_steps": 5}),
        description=(
            "The final 5 steps before session end contain no validation COMMAND "
            "after the last source FILE_WRITE."
        ),
    ),
    PatternRecord(
        id="unsupported-completion-claim",
        name="Unsupported completion claim",
        band="[S]",
        phase="report",
        description=(
            "Final assistant message contains an explicit completion cue (fixed, "
            "done, resolved) but no successful validation occurred after the last "
            "relevant edit. Narrow text cue + structural validation check."
        ),
    ),
    # Cross-cutting
    PatternRecord(
        id="redundant-search",
        name="Redundant search",
        band="[S]",
        phase="cross-cutting",
        thresholds=MappingProxyType({"window_steps": 10, "min_duplicates": 2}),
        description=(
            "Repeated nearly identical SEARCH query within a short window."
        ),
    ),
    PatternRecord(
        id="shell-over-tool",
        name="Shell-over-tool",
        band="[S]",
        phase="cross-cutting",
        gating=("capability-gated",),
        description=(
            "A general-purpose shell (e.g., Bash) is used for read/search when the "
            "same session exposes a dedicated structured tool (e.g., Read, Grep). "
            "Fires only when both capabilities are demonstrably available."
        ),
        legacy_aliases=("tool_selection_antipattern",),
    ),
    PatternRecord(
        id="tool-oscillation",
        name="Tool oscillation",
        band="[S]",
        phase="cross-cutting",
        thresholds=MappingProxyType({"min_cycles": 2}),
        description=(
            "Repeated FILE_READ -> FILE_WRITE -> FILE_READ loops on the same "
            "file/range with no progress."
        ),
    ),
)


# ---------------------------------------------------------------------------
# [H] Intent-dependent hypotheses (7)
# All require semantic labels from the post-hoc labeler (or human annotation).
# ---------------------------------------------------------------------------

_H_PATTERNS: tuple[PatternRecord, ...] = (
    PatternRecord(
        id="phase-oscillation",
        name="Phase oscillation",
        band="[H]",
        phase="cross-cutting",
        requires_semantic_labels=True,
        thresholds=MappingProxyType({"min_transitions": 3, "window_steps": 6}),
        description=(
            ">=3 transitions between the same two phases within a 6-step window "
            "(depends on semantic labeler)."
        ),
    ),
    PatternRecord(
        id="premature-implementation",
        name="Premature implementation",
        band="[H]",
        phase="implement",
        requires_semantic_labels=True,
        description=(
            "First implement-phase step precedes any plan-phase step (depends on "
            "semantic labeler)."
        ),
    ),
    PatternRecord(
        id="semantic-fruitless-exploration",
        name="Semantic fruitless exploration",
        band="[H]",
        phase="understand",
        requires_semantic_labels=True,
        thresholds=MappingProxyType({"min_code_reads": 5, "min_unused_files": 4}),
        description=(
            ">=5 code-reading steps where >=4 files never appear in subsequent "
            "implement steps (depends on semantic labeler)."
        ),
        legacy_aliases=("semantic_fruitless_streak",),
    ),
    PatternRecord(
        id="semantic-plan-stall",
        name="Semantic plan stall",
        band="[H]",
        phase="plan",
        requires_semantic_labels=True,
        thresholds=MappingProxyType({"min_plan_steps": 5}),
        description=(
            ">=5 plan-phase steps without any implement step (depends on semantic "
            "labeler)."
        ),
    ),
    PatternRecord(
        id="debug-wo-hypothesis",
        name="Debug without hypothesis",
        band="[H]",
        phase="debug",
        requires_semantic_labels=True,
        thresholds=MappingProxyType({"min_reproduce_steps": 3}),
        description=(
            ">=3 debug-reproduction steps without any root-cause-analysis step "
            "(depends on semantic labeler)."
        ),
    ),
    PatternRecord(
        id="prompt-skim",
        name="Prompt skim",
        band="[H]",
        phase="intake",
        requires_semantic_labels=True,
        gating=("weaker",),
        description=(
            "Agent never re-references the user prompt after the first turn. "
            "Weak textual proxy."
        ),
    ),
    PatternRecord(
        id="memory-contamination",
        name="Memory contamination",
        band="[H]",
        phase="report",
        requires_semantic_labels=True,
        description=(
            "At session end, the agent writes incorrect or outdated information "
            "into a persistent memory file. Requires LLM-judge or human annotation."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Divergence patterns (6) — cross-trajectory (require a reference)
# Definitions from latex_draft/offline/sections/approach.tex, Table 4 (tab:divergence).
# ---------------------------------------------------------------------------

_DIVERGENCE_PATTERNS: tuple[PatternRecord, ...] = (
    PatternRecord(
        id="rapid-rewrite",
        name="Rapid rewrite",
        band="divergence",
        tier="high",
        thresholds=MappingProxyType({"max_step_gap": 3}),
        required_signals=("action types", "target overlap", "reference match"),
        description=(
            "WRITE to file f followed by a second WRITE to f within <=3 steps that "
            "overwrites or reverses it; divergence iff reference did not rewrite."
        ),
        legacy_aliases=("reverted_and_rewritten",),
    ),
    PatternRecord(
        id="scope-drift",
        name="Scope drift",
        band="divergence",
        tier="high",
        required_signals=("action targets", "anchor set"),
        description=(
            "Compared WRITEs target files outside the reference's write set "
            "(benchmark-informed variant: outside the ground-truth changed-file set)."
        ),
        legacy_aliases=("scope_creep",),
    ),
    PatternRecord(
        id="off-anchor-exploration",
        name="Off-anchor exploration",
        band="divergence",
        tier="med",
        thresholds=MappingProxyType({"min_off_anchor_ratio": 0.5}),
        required_signals=("action targets", "anchor set"),
        description=(
            "Compared-trajectory READ/SEARCH targets a large share of files "
            "outside the anchor set (reference's read set, or -- when available "
            "-- the ground-truth changed-file set)."
        ),
        legacy_aliases=("broad_exploration",),
    ),
    PatternRecord(
        id="dead-end-exploration",
        name="Dead-end exploration",
        band="divergence",
        tier="med",
        required_signals=("alignment matches", "subsequent writes"),
        description=(
            "Exploration span touches files that are never subsequently written "
            "or matched to reference-critical files."
        ),
    ),
    PatternRecord(
        id="ordering-inefficiency",
        name="Ordering inefficiency",
        band="divergence",
        tier="med",
        required_signals=("alignment order",),
        description=(
            "Matched actions appear in a substantially less efficient order than "
            "the reference (longer minimum edit sequence)."
        ),
    ),
    PatternRecord(
        id="iterative-refinement",
        name="Iterative refinement",
        band="divergence",
        tier="low",
        thresholds=MappingProxyType({"min_step_gap": 3}),
        required_signals=("action types", "target overlap", "step gap"),
        description=(
            "WRITE to f overwritten by a later WRITE to f after >3 steps (neutral "
            "rewrite, low-confidence divergence)."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Combined catalog + helpers
# ---------------------------------------------------------------------------

ALL_PATTERNS: tuple[PatternRecord, ...] = _S_PATTERNS + _H_PATTERNS + _DIVERGENCE_PATTERNS


def by_id(pattern_id: str) -> PatternRecord:
    """Return the record with the given id. Raises KeyError if unknown."""
    for p in ALL_PATTERNS:
        if p.id == pattern_id:
            return p
        if pattern_id in p.legacy_aliases:
            return p
    raise KeyError(f"Unknown pattern id: {pattern_id!r}")


def by_band(band: Band) -> tuple[PatternRecord, ...]:
    """Return all records with the given band ('[S]' | '[H]' | 'divergence')."""
    return tuple(p for p in ALL_PATTERNS if p.band == band)


def by_phase(phase: Phase | None) -> tuple[PatternRecord, ...]:
    """Return all records with the given phase (None returns divergence records)."""
    return tuple(p for p in ALL_PATTERNS if p.phase == phase)


def thresholds_for(pattern_id: str) -> Mapping[str, int | float]:
    """Return the default thresholds for a pattern. Callers may layer overrides on top."""
    return by_id(pattern_id).thresholds
