"""Iterative refinement divergence detector (tier LOW, neutral-valence)."""

from __future__ import annotations

from typing import Any

from trajectory_visualizer.core.catalog import by_id
from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from trajectory_visualizer.insight.detectors import _helpers as h


def detect(
    compared: list[Any], reference: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Iterative refinement ([LOW]).

    Operational definition (approach.tex Table 4):
    "WRITE to f overwritten by a later WRITE to f after >3 steps (neutral
    rewrite, low-confidence divergence)."
    """
    tier = by_id("iterative-refinement").tier
    min_gap = int(context.thresholds_for("iterative-refinement")["min_step_gap"])

    detections: list[PatternDetection] = []
    last_write: dict[str, int] = {}

    for i, step in enumerate(compared):
        if not h.is_write(step):
            continue
        tgt = h.target(step)
        if not tgt:
            continue
        prev = last_write.get(tgt)
        if prev is not None and (i - prev) > min_gap:
            detections.append(
                PatternDetection(
                    detector_id="iterative-refinement",
                    span=(prev, i),
                    evidence={
                        "file": tgt,
                        "first_write_step": prev,
                        "second_write_step": i,
                        "gap": i - prev,
                    },
                    tier=tier,
                )
            )
        last_write[tgt] = i
    return detections
