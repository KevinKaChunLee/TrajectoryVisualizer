"""Phase oscillation detector ([H])."""

from __future__ import annotations

from collections import deque
from typing import Any

from trajectory_visualizer.core.detection import DetectorContext, PatternDetection


def detect(steps: list[Any], context: DetectorContext) -> list[PatternDetection]:
    """Phase oscillation ([H]).

    Operational definition (appendix_catalog.tex, [H]):
    ">=3 transitions between the same two phases within a 6-step window
    (depends on semantic labeler)."
    """
    if not context.labels:
        return []
    thresholds = context.thresholds_for("phase-oscillation")
    min_transitions = int(thresholds["min_transitions"])
    window = int(thresholds["window_steps"])

    # Transitions = boundaries where label[i] != label[i+1].
    # We detect transitions between the same two-phase pair recurring >=N times
    # within any sliding window of `window` consecutive steps (with labels).
    detections: list[PatternDetection] = []
    indices = sorted(context.labels.keys())
    phases = [context.labels[i].get("phase", "") for i in indices]

    # Slide a window of `window` step indices.
    for start in range(len(indices) - window + 1):
        window_indices = indices[start : start + window]
        window_phases = [context.labels[i].get("phase", "") for i in window_indices]
        transitions: list[tuple[str, str]] = []
        for a, b in zip(window_phases, window_phases[1:]):
            if a and b and a != b:
                transitions.append((a, b))
        # Count back-and-forth between the same pair.
        pair_counts: dict[frozenset[str], int] = {}
        for pair in transitions:
            pair_counts[frozenset(pair)] = pair_counts.get(frozenset(pair), 0) + 1
        hot = [(p, c) for p, c in pair_counts.items() if c >= min_transitions]
        if hot:
            pair, count = max(hot, key=lambda x: x[1])
            detections.append(
                PatternDetection(
                    detector_id="phase-oscillation",
                    span=(window_indices[0], window_indices[-1]),
                    evidence={
                        "phase_pair": sorted(pair),
                        "transitions_in_window": count,
                        "window_steps": window,
                    },
                )
            )
            break  # first hit is enough; runner aggregates per-trajectory
    return detections
