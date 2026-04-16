"""Scope drift divergence detector (tier HIGH)."""

from __future__ import annotations

from typing import Any

from trajectory_visualizer.core.catalog import by_id
from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from trajectory_visualizer.insight.detectors import _helpers as h


def detect(
    compared: list[Any], reference: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Scope drift ([HIGH]).

    Operational definition (approach.tex Table 4):
    "Compared WRITEs target files outside the reference's write set
    (benchmark-informed variant: outside the ground-truth changed-file set)."
    """
    tier = by_id("scope-drift").tier
    # Prefer anchor_set (ground-truth) when available; else use reference writes.
    anchor = set(context.anchor_set) if context.anchor_set else _reference_write_set(reference)
    if not anchor:
        return []

    detections: list[PatternDetection] = []
    for i, step in enumerate(compared):
        if not h.is_write(step):
            continue
        tgt = h.target(step)
        if not tgt:
            continue
        if tgt not in anchor:
            detections.append(
                PatternDetection(
                    detector_id="scope-drift",
                    span=(i, i),
                    evidence={
                        "file": tgt,
                        "anchor_size": len(anchor),
                        "source": "ground_truth" if context.anchor_set else "reference",
                    },
                    tier=tier,
                )
            )
    return detections


def _reference_write_set(steps: list[Any]) -> set[str]:
    return {h.target(s) for s in steps if h.is_write(s) and h.target(s)}
