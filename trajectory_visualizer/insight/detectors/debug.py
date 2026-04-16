"""Phase 5: Debug / Recover detectors."""

from __future__ import annotations

from typing import Any

from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from . import _helpers as h


def detect_error_spiral(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Error-spiral detector.

    Operational definition (appendix_catalog.tex, Phase 5):
    "Same (tool, error_signature) pair recurs >=3 times with no observable
    change in approach."
    """
    min_recurrences = int(context.thresholds_for("error-spiral")["min_recurrences"])

    # Per-signature recurrence tracker; reset on any intervening write
    # (an edit is the observable "change in approach" the paper calls out).
    occurrences: dict[str, list[int]] = {}
    reported: set[str] = set()
    detections: list[PatternDetection] = []

    for i, step in enumerate(steps):
        if h.is_write(step):
            occurrences.clear()
            continue
        sig = h.error_signature(step)
        if not sig:
            continue
        idxs = occurrences.setdefault(sig, [])
        idxs.append(i)
        if len(idxs) >= min_recurrences and sig not in reported:
            detections.append(
                PatternDetection(
                    detector_id="error-spiral",
                    span=(idxs[0], idxs[-1]),
                    evidence={
                        "error_signature": sig,
                        "recurrences": len(idxs),
                        "steps": list(idxs),
                    },
                )
            )
            reported.add(sig)
    return detections


def detect_recovery_free_retry(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Recovery-free retry detector.

    Operational definition (appendix_catalog.tex, Phase 5):
    "A failed action is immediately retried with no intervening inspection,
    edit, or parameter change."
    """
    detections: list[PatternDetection] = []
    for i in range(len(steps) - 1):
        a = steps[i]
        b = steps[i + 1]
        if not h.is_failed(a):
            continue
        if h.tool(a) != h.tool(b):
            continue
        # Same tool; check args equality.
        if h.args(a) != h.args(b):
            continue
        # No intervening inspect/edit because i+1 follows i.
        detections.append(
            PatternDetection(
                detector_id="recovery-free-retry",
                span=(i, i + 1),
                evidence={
                    "tool": h.tool(a),
                    "target": h.target(a),
                    "error_signature": h.error_signature(a),
                },
            )
        )
    return detections
