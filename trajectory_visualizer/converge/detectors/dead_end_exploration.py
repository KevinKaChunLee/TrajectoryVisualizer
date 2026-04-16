"""Dead-end exploration divergence detector (tier MED)."""

from __future__ import annotations

from typing import Any

from trajectory_visualizer.core.catalog import by_id
from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from trajectory_visualizer.insight.detectors import _helpers as h


def detect(
    compared: list[Any], reference: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Dead-end exploration ([MED]).

    Operational definition (approach.tex Table 4):
    "Exploration span touches files that are never subsequently written or
    matched to reference-critical files."
    """
    tier = by_id("dead-end-exploration").tier

    reference_critical_files: set[str] = set()
    for s in reference:
        if h.is_write(s) and h.target(s):
            reference_critical_files.add(h.target(s))

    subsequent_write_targets: dict[int, set[str]] = {}
    acc: set[str] = set()
    for i in range(len(compared) - 1, -1, -1):
        subsequent_write_targets[i] = set(acc)
        if h.is_write(compared[i]) and h.target(compared[i]):
            acc.add(h.target(compared[i]))

    detections: list[PatternDetection] = []
    span_start: int | None = None
    dead_files: set[str] = set()
    for i, step in enumerate(compared):
        is_exploration = h.is_read(step) or h.is_search(step)
        if is_exploration:
            tgt = h.target(step)
            if not tgt:
                continue
            if tgt in reference_critical_files:
                _flush(detections, span_start, i - 1, dead_files, tier)
                span_start = None
                dead_files = set()
                continue
            if tgt in subsequent_write_targets.get(i, set()):
                _flush(detections, span_start, i - 1, dead_files, tier)
                span_start = None
                dead_files = set()
                continue
            if span_start is None:
                span_start = i
            dead_files.add(tgt)
        else:
            # A write / command / reasoning ends any open span.
            _flush(detections, span_start, i - 1, dead_files, tier)
            span_start = None
            dead_files = set()

    _flush(detections, span_start, len(compared) - 1, dead_files, tier)
    return detections


def _flush(
    out: list[PatternDetection],
    start: int | None,
    end: int,
    files: set[str],
    tier: str | None,
) -> None:
    if start is None or not files:
        return
    out.append(
        PatternDetection(
            detector_id="dead-end-exploration",
            span=(start, end),
            evidence={
                "dead_end_files": sorted(files),
                "count": len(files),
            },
            tier=tier,
        )
    )
