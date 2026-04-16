"""Single-trajectory anti-pattern detectors.

Each detector is a pure function `detect_<id>(steps, *, context) -> list[PatternDetection]`.
The `DETECTOR_REGISTRY` dict maps every `[S]` and `[H]` catalog id to its function.
"""

from __future__ import annotations

from typing import Any, Callable

from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from . import cross_cutting, debug, implement, intake, plan, report, understand, validate
from .semantic import (
    debug_wo_hypothesis,
    memory_contamination,
    phase_oscillation,
    premature_implementation,
    prompt_skim,
    semantic_fruitless_exploration,
    semantic_plan_stall,
)


Detector = Callable[[list[Any], DetectorContext], list[PatternDetection]]


DETECTOR_REGISTRY: dict[str, Detector] = {
    # [S] Phase 0
    "memory-bypass": intake.detect_memory_bypass,
    "premature-code-action": intake.detect_premature_code_action,
    # [S] Phase 1
    "empty-result-churn": understand.detect_empty_result_churn,
    "search-loop": understand.detect_search_loop,
    "re-read-churn": understand.detect_re_read_churn,
    # [S] Phase 2
    "plan-stall": plan.detect_plan_stall,
    "plan-thrash": plan.detect_plan_thrash,
    "plan-less-execution": plan.detect_plan_less_execution,
    # [S] Phase 3
    "edit-without-inspection": implement.detect_edit_without_inspection,
    "edit-thrash": implement.detect_edit_thrash,
    # [S] Phase 4
    "late-validation": validate.detect_late_validation,
    "validation-avoidance": validate.detect_validation_avoidance,
    "test-retry-loop": validate.detect_test_retry_loop,
    # [S] Phase 5
    "error-spiral": debug.detect_error_spiral,
    "recovery-free-retry": debug.detect_recovery_free_retry,
    # [S] Phase 6
    "verification-skip": report.detect_verification_skip,
    "unsupported-completion-claim": report.detect_unsupported_completion_claim,
    # [S] Cross-cutting
    "redundant-search": cross_cutting.detect_redundant_search,
    "shell-over-tool": cross_cutting.detect_shell_over_tool,
    "tool-oscillation": cross_cutting.detect_tool_oscillation,
    # [H] Intent-dependent
    "phase-oscillation": phase_oscillation.detect,
    "premature-implementation": premature_implementation.detect,
    "semantic-fruitless-exploration": semantic_fruitless_exploration.detect,
    "semantic-plan-stall": semantic_plan_stall.detect,
    "debug-wo-hypothesis": debug_wo_hypothesis.detect,
    "prompt-skim": prompt_skim.detect,
    "memory-contamination": memory_contamination.detect,
}


__all__ = ["DETECTOR_REGISTRY", "Detector"]
