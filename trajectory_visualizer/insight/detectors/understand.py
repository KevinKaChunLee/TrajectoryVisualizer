"""Phase 1: Understand detectors."""

from __future__ import annotations

from typing import Any

from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from . import _helpers as h


def detect_empty_result_churn(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Empty-result churn detector.

    Operational definition (appendix_catalog.tex, Phase 1):
    ">=3 consecutive SEARCH steps return zero matches."
    """
    thresh = int(context.thresholds_for("empty-result-churn")["min_consecutive_empty"])
    detections: list[PatternDetection] = []
    streak_start: int | None = None
    streak_len = 0

    for i, step in enumerate(steps):
        if h.is_search(step) and _search_returned_empty(step):
            if streak_start is None:
                streak_start = i
            streak_len += 1
            continue
        if streak_len >= thresh and streak_start is not None:
            detections.append(
                PatternDetection(
                    detector_id="empty-result-churn",
                    span=(streak_start, i - 1),
                    evidence={"empty_searches": streak_len},
                )
            )
        streak_start = None
        streak_len = 0

    if streak_len >= thresh and streak_start is not None:
        detections.append(
            PatternDetection(
                detector_id="empty-result-churn",
                span=(streak_start, len(steps) - 1),
                evidence={"empty_searches": streak_len},
            )
        )
    return detections


def _search_returned_empty(step: Any) -> bool:
    """Heuristic: a search returned empty if output is empty or marked 'no matches'."""
    if h.is_failed(step):
        return False  # a failed search is a different pattern
    a = h.args(step)
    for key in ("output", "result", "matches"):
        v = a.get(key)
        if v is None:
            continue
        if isinstance(v, (list, tuple, dict, set)) and len(v) == 0:
            return True
        if isinstance(v, str):
            s = v.strip().lower()
            if s == "" or "no matches" in s or "no results" in s or "0 matches" in s:
                return True
    # Also look for an empty-string status field.
    return h.status(step).lower() in {"no_matches", "empty", "no_match"}


def detect_search_loop(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Search loop detector.

    Operational definition (appendix_catalog.tex, Phase 1):
    ">=4 consecutive SEARCH/FILE_READ steps with no FILE_WRITE or validation
    COMMAND in between."
    """
    thresh = int(context.thresholds_for("search-loop")["min_consecutive_steps"])
    detections: list[PatternDetection] = []
    streak_start: int | None = None
    streak_len = 0

    def _flush(end_exclusive: int) -> None:
        nonlocal streak_start, streak_len
        if streak_len >= thresh and streak_start is not None:
            detections.append(
                PatternDetection(
                    detector_id="search-loop",
                    span=(streak_start, end_exclusive - 1),
                    evidence={"consecutive_read_search_steps": streak_len},
                )
            )
        streak_start = None
        streak_len = 0

    for i, step in enumerate(steps):
        if h.is_search(step) or h.is_read(step):
            if streak_start is None:
                streak_start = i
            streak_len += 1
        elif h.is_write(step) or h.is_validation_command(step):
            _flush(i)
        else:
            # non-read/search/write/validation (e.g., REASON, non-validation command)
            # doesn't break or extend the streak; we treat it as neutral.
            continue

    _flush(len(steps))
    return detections


def detect_re_read_churn(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Re-read churn detector.

    Operational definition (appendix_catalog.tex, Phase 1):
    "Same file, or overlapping line range, is read >=3 times within a short
    window with no intervening write to that file."
    """
    thresholds = context.thresholds_for("re-read-churn")
    min_reads = int(thresholds["min_reads"])
    window = int(thresholds["window_steps"])

    detections: list[PatternDetection] = []
    seen: dict[str, list[int]] = {}

    for i, step in enumerate(steps):
        if h.is_write(step):
            # Writing to this file resets its read history.
            seen.pop(h.target(step), None)
            continue
        if not h.is_read(step):
            continue
        tgt = h.target(step)
        if not tgt:
            continue
        reads = seen.setdefault(tgt, [])
        reads.append(i)
        # Drop reads outside the window.
        while reads and i - reads[0] > window:
            reads.pop(0)
        if len(reads) >= min_reads:
            detections.append(
                PatternDetection(
                    detector_id="re-read-churn",
                    span=(reads[0], i),
                    evidence={"file": tgt, "read_steps": list(reads)},
                )
            )
            # Anchor the next window on the current step so continuous-burst
            # rereads beyond the threshold are not lost.
            seen[tgt] = [i]
    return detections
