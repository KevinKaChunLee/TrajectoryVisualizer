"""Ordering inefficiency divergence detector (tier MED)."""

from __future__ import annotations

from typing import Any

from trajectory_visualizer.core.catalog import by_id
from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from trajectory_visualizer.insight.detectors import _helpers as h


def detect(
    compared: list[Any], reference: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Ordering inefficiency ([MED]).

    Operational definition (approach.tex Table 4):
    "Matched actions appear in a substantially less efficient order than the
    reference (longer minimum edit sequence)."

    Implementation: build an action-signature sequence for each trajectory,
    find the common subset, and compare the number of transpositions needed
    to convert compared's order into reference's order. Fires when the
    transposition count exceeds 30% of the common-subset size.
    """
    tier = by_id("ordering-inefficiency").tier

    compared_sigs = [_action_signature(s) for s in compared]
    reference_sigs = [_action_signature(s) for s in reference]
    compared_sigs = [s for s in compared_sigs if s]
    reference_sigs = [s for s in reference_sigs if s]

    common = list(dict.fromkeys(s for s in compared_sigs if s in reference_sigs))
    if len(common) < 3:
        return []

    # Extract first occurrence order of each common sig in each trajectory.
    ref_order = _first_occurrence_order(reference_sigs, common)
    cmp_order = _first_occurrence_order(compared_sigs, common)
    if not ref_order or not cmp_order:
        return []
    # Map compared order to reference rank, then count inversions.
    rank = {sig: idx for idx, sig in enumerate(ref_order)}
    ranks_in_cmp = [rank[s] for s in cmp_order if s in rank]
    inversions = _count_inversions(ranks_in_cmp)
    threshold = max(1, int(0.3 * len(ranks_in_cmp) * (len(ranks_in_cmp) - 1) / 2))
    if inversions >= threshold:
        return [
            PatternDetection(
                detector_id="ordering-inefficiency",
                span=(0, len(compared) - 1),
                evidence={
                    "common_actions": len(common),
                    "inversions": inversions,
                    "threshold": threshold,
                },
                tier=tier,
            )
        ]
    return []


def _action_signature(step: Any) -> str:
    at = h.action_type(step)
    tg = h.target(step)
    if not at or not tg:
        return ""
    return f"{at}:{tg}"


def _first_occurrence_order(sigs: list[str], keep: list[str]) -> list[str]:
    keep_set = set(keep)
    seen: set[str] = set()
    out: list[str] = []
    for s in sigs:
        if s in keep_set and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _count_inversions(xs: list[int]) -> int:
    # O(n^2) is fine for small common-subset sizes (typically <30).
    inv = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[i] > xs[j]:
                inv += 1
    return inv
