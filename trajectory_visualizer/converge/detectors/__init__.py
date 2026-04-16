"""Cross-trajectory divergence detectors. Each takes a compared trajectory and a reference."""

from __future__ import annotations

from typing import Any, Callable

from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from . import (
    dead_end_exploration,
    iterative_refinement,
    off_anchor_exploration,
    ordering_inefficiency,
    rapid_rewrite,
    scope_drift,
)


DivergenceDetector = Callable[[list[Any], list[Any], DetectorContext], list[PatternDetection]]


DIVERGENCE_REGISTRY: dict[str, DivergenceDetector] = {
    "rapid-rewrite": rapid_rewrite.detect,
    "scope-drift": scope_drift.detect,
    "off-anchor-exploration": off_anchor_exploration.detect,
    "dead-end-exploration": dead_end_exploration.detect,
    "ordering-inefficiency": ordering_inefficiency.detect,
    "iterative-refinement": iterative_refinement.detect,
}


__all__ = ["DIVERGENCE_REGISTRY", "DivergenceDetector"]
