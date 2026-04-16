"""Semantic fruitless exploration detector ([H])."""

from __future__ import annotations

from typing import Any

from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from trajectory_visualizer.insight.detectors import _helpers as h


def detect(steps: list[Any], context: DetectorContext) -> list[PatternDetection]:
    """Semantic fruitless exploration ([H]).

    Operational definition (appendix_catalog.tex, [H]):
    ">=5 code-reading steps where >=4 files never appear in subsequent
    implement steps (depends on semantic labeler)."
    """
    if not context.labels:
        return []
    thresholds = context.thresholds_for("semantic-fruitless-exploration")
    min_code_reads = int(thresholds["min_code_reads"])
    min_unused = int(thresholds["min_unused_files"])

    # [H] detector: only count steps the labeler has explicitly labeled as
    # understand-phase reads or code-reading actions. Unlabeled steps are
    # not credited (the [H] band requires semantic grounding).
    code_reads: list[tuple[int, str]] = []
    implement_targets: set[str] = set()
    for i, step in enumerate(steps):
        lbl = context.labels.get(i, {})
        action = lbl.get("action", "")
        phase = lbl.get("phase", "")
        if action == "code_reading" or (h.is_read(step) and phase == "understand"):
            tgt = h.target(step)
            if tgt:
                code_reads.append((i, tgt))
        if phase == "implement" and h.is_write(step):
            tgt = h.target(step)
            if tgt:
                implement_targets.add(tgt)

    if len(code_reads) < min_code_reads:
        return []
    unused = {t for _, t in code_reads if t not in implement_targets}
    if len(unused) < min_unused:
        return []
    first = code_reads[0][0]
    last = code_reads[-1][0]
    return [
        PatternDetection(
            detector_id="semantic-fruitless-exploration",
            span=(first, last),
            evidence={
                "code_read_steps": len(code_reads),
                "unused_files": sorted(unused),
                "implement_targets": sorted(implement_targets),
            },
        )
    ]
