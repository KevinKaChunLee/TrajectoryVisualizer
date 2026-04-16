"""Phase 3: Implement detectors."""

from __future__ import annotations

from typing import Any

from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from . import _helpers as h


def detect_edit_without_inspection(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Edit-without-inspection detector.

    Operational definition (appendix_catalog.tex, Phase 3):
    "First FILE_WRITE to a file has no prior FILE_READ or SEARCH hit on that
    file."
    """
    detections: list[PatternDetection] = []
    read_targets: set[str] = set()
    search_hits: set[str] = set()
    first_write_seen: set[str] = set()

    for i, step in enumerate(steps):
        tgt = h.target(step)
        if h.is_read(step) and tgt:
            read_targets.add(tgt)
        elif h.is_search(step):
            # A search step's matches (when available) expose file paths.
            for match in _search_match_paths(step):
                search_hits.add(match)
        elif h.is_write(step) and tgt:
            if tgt in first_write_seen:
                continue
            first_write_seen.add(tgt)
            if tgt in read_targets or tgt in search_hits:
                continue
            detections.append(
                PatternDetection(
                    detector_id="edit-without-inspection",
                    span=(i, i),
                    evidence={"file": tgt, "inspected_before": False},
                )
            )
    return detections


def _search_match_paths(step: Any) -> list[str]:
    a = h.args(step)
    for key in ("matches", "output", "result"):
        v = a.get(key)
        if isinstance(v, list):
            return [str(x) for x in v if isinstance(x, str)]
    return []


def detect_edit_thrash(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Edit-thrash detector.

    Operational definition (appendix_catalog.tex, Phase 3):
    "Same file is written >=3 times within a short window with oscillating
    (non-monotonic) changes."

    Implementation choice: we treat "oscillating" as "at least one write in the
    window has effect_label == 'reverted'". This is the available observable
    signal; a stricter diff-size monotonicity check would require raw diffs we
    don't persist.
    """
    thresholds = context.thresholds_for("edit-thrash")
    min_writes = int(thresholds["min_writes"])
    window = int(thresholds["window_steps"])

    detections: list[PatternDetection] = []
    writes_by_file: dict[str, list[int]] = {}

    for i, step in enumerate(steps):
        if not h.is_write(step):
            continue
        tgt = h.target(step)
        if not tgt:
            continue
        xs = writes_by_file.setdefault(tgt, [])
        xs.append(i)
        while xs and i - xs[0] > window:
            xs.pop(0)
        if len(xs) >= min_writes:
            window_steps = [steps[j] for j in xs]
            oscillating = any(h.effect_label(s) == "reverted" for s in window_steps)
            if oscillating:
                detections.append(
                    PatternDetection(
                        detector_id="edit-thrash",
                        span=(xs[0], i),
                        evidence={
                            "file": tgt,
                            "write_steps": list(xs),
                            "any_reverted": True,
                        },
                    )
                )
                writes_by_file[tgt] = []  # avoid overlapping duplicates
    return detections
