"""Off-anchor exploration divergence detector (tier MED)."""

from __future__ import annotations

from typing import Any

from trajectory_visualizer.core.catalog import by_id
from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from trajectory_visualizer.insight.detectors import _helpers as h


def detect(
    compared: list[Any], reference: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Off-anchor exploration ([MED]).

    Operational definition (approach.tex Table 4):
    "Compared-trajectory READ/SEARCH targets a large share of files outside
    the anchor set (reference's read set, or -- when available -- the
    ground-truth changed-file set)."
    """
    tier = by_id("off-anchor-exploration").tier
    ratio_thresh = float(context.thresholds_for("off-anchor-exploration")["min_off_anchor_ratio"])

    anchor = set(context.anchor_set) if context.anchor_set else _reference_read_and_write_set(reference)
    if not anchor:
        return []

    off_anchor_steps: list[int] = []
    on_anchor_steps: list[int] = []
    for i, step in enumerate(compared):
        if not (h.is_read(step) or h.is_search(step)):
            continue
        tgt = h.target(step)
        if not tgt:
            continue
        if tgt in anchor:
            on_anchor_steps.append(i)
        else:
            off_anchor_steps.append(i)

    total = len(off_anchor_steps) + len(on_anchor_steps)
    if total == 0:
        return []
    ratio = len(off_anchor_steps) / total
    if ratio < ratio_thresh:
        return []
    return [
        PatternDetection(
            detector_id="off-anchor-exploration",
            span=(
                off_anchor_steps[0] if off_anchor_steps else 0,
                off_anchor_steps[-1] if off_anchor_steps else len(compared) - 1,
            ),
            evidence={
                "off_anchor_count": len(off_anchor_steps),
                "on_anchor_count": len(on_anchor_steps),
                "ratio": round(ratio, 3),
                "anchor_source": "ground_truth" if context.anchor_set else "reference",
            },
            tier=tier,
        )
    ]


def _reference_read_and_write_set(steps: list[Any]) -> set[str]:
    out: set[str] = set()
    for s in steps:
        if (h.is_read(s) or h.is_write(s)) and h.target(s):
            out.add(h.target(s))
    return out
