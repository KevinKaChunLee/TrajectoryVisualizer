"""Rapid rewrite divergence detector (tier HIGH)."""

from __future__ import annotations

from typing import Any

from trajectory_visualizer.core.catalog import by_id
from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from trajectory_visualizer.insight.detectors import _helpers as h


def detect(
    compared: list[Any], reference: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Rapid rewrite ([HIGH]).

    Operational definition (approach.tex Table 4):
    "WRITE to file f followed by a second WRITE to f within <=3 steps that
    overwrites or reverses it; divergence iff reference did not rewrite."
    """
    tier = by_id("rapid-rewrite").tier
    max_gap = int(context.thresholds_for("rapid-rewrite")["max_step_gap"])

    reference_rewrites = _files_rewritten_within(reference, max_gap)

    detections: list[PatternDetection] = []
    writes_by_file: dict[str, list[int]] = {}

    for i, step in enumerate(compared):
        if not h.is_write(step):
            continue
        tgt = h.target(step)
        if not tgt:
            continue
        prev = writes_by_file.get(tgt)
        if prev is not None and (i - prev[-1]) <= max_gap:
            if tgt not in reference_rewrites:
                detections.append(
                    PatternDetection(
                        detector_id="rapid-rewrite",
                        span=(prev[-1], i),
                        evidence={
                            "file": tgt,
                            "first_write_step": prev[-1],
                            "second_write_step": i,
                            "gap": i - prev[-1],
                            "reference_had_rewrite": False,
                        },
                        tier=tier,
                    )
                )
        writes_by_file.setdefault(tgt, []).append(i)
    return detections


def _files_rewritten_within(steps: list[Any], max_gap: int) -> set[str]:
    """Files that the reference trajectory rewrites within `max_gap` steps."""
    out: set[str] = set()
    seen: dict[str, int] = {}
    for i, step in enumerate(steps):
        if not h.is_write(step):
            continue
        tgt = h.target(step)
        if not tgt:
            continue
        if tgt in seen and (i - seen[tgt]) <= max_gap:
            out.add(tgt)
        seen[tgt] = i
    return out
