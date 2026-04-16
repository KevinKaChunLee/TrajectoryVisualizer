"""Phase 6: Report / Complete detectors."""

from __future__ import annotations

import re
from typing import Any

from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from . import _helpers as h


# Regex matches whole-word "fixed", "done", "resolved", "implemented",
# "completed" anywhere in the text (case-insensitive).
_COMPLETION_CUE_RE = re.compile(
    r"\b(fixed|done|resolved|implemented|completed)\b",
    re.IGNORECASE,
)


def detect_verification_skip(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Verification-skip detector.

    Operational definition (appendix_catalog.tex, Phase 6):
    "The final 5 steps before session end contain no validation COMMAND after
    the last source FILE_WRITE."
    """
    window = int(context.thresholds_for("verification-skip")["tail_window_steps"])
    # Find last source FILE_WRITE.
    last_write = -1
    for i in range(len(steps) - 1, -1, -1):
        if h.is_write(steps[i]):
            last_write = i
            break
    if last_write == -1:
        return []  # no writes at all; nothing to verify

    tail_start = len(steps) - window
    # The tail starts at max(last_write+1, tail_start).
    check_from = max(last_write + 1, tail_start)
    if check_from >= len(steps):
        # Last write happened after the tail start — definitely no validation post-write.
        return [
            PatternDetection(
                detector_id="verification-skip",
                span=(last_write, len(steps) - 1),
                evidence={
                    "last_write_step": last_write,
                    "validation_in_tail": False,
                },
            )
        ]

    # Per paper: "final 5 steps ... contain no validation COMMAND after the
    # last source FILE_WRITE." The scan window is the intersection of the
    # tail window and the post-last-write region.
    any_validation = any(h.is_validation_command(s) for s in steps[check_from:])
    if any_validation:
        return []
    return [
        PatternDetection(
            detector_id="verification-skip",
            span=(last_write, len(steps) - 1),
            evidence={
                "last_write_step": last_write,
                "validation_in_tail": False,
            },
        )
    ]


def detect_unsupported_completion_claim(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Unsupported-completion-claim detector.

    Operational definition (appendix_catalog.tex, Phase 6):
    "Final assistant message contains an explicit completion cue (fixed, done,
    resolved) but no successful validation occurred after the last relevant
    edit. Narrow text cue + structural validation check."
    """
    # Find last write.
    last_write = -1
    for i in range(len(steps) - 1, -1, -1):
        if h.is_write(steps[i]):
            last_write = i
            break
    # Scan post-last-write for a successful validation.
    has_passing_validation = any(
        h.is_validation_command(s) and not h.is_failed(s)
        for s in (steps[last_write + 1 :] if last_write >= 0 else steps)
    )
    if has_passing_validation:
        return []

    # Inspect the final REASON/assistant text for a completion cue.
    final_text = _final_assistant_text(steps)
    if not final_text:
        return []
    m = _COMPLETION_CUE_RE.search(final_text)
    if not m:
        return []
    return [
        PatternDetection(
            detector_id="unsupported-completion-claim",
            span=(max(last_write, 0), len(steps) - 1),
            evidence={
                "cue": m.group(0).lower(),
                "last_write_step": last_write,
                "validation_after_last_write_passed": has_passing_validation,
            },
        )
    ]


def _final_assistant_text(steps: list[Any]) -> str:
    """Return the text of the final REASON/assistant step, or empty if absent."""
    for step in reversed(steps):
        if h.action_type(step) == "REASON":
            a = h.args(step)
            for key in ("text", "content", "message"):
                v = a.get(key)
                if isinstance(v, str) and v.strip():
                    return v
            if h.target(step):
                return h.target(step)
    return ""
