"""Memory contamination detector ([H], LLM-judge or human annotation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from trajectory_visualizer.insight.detectors import _helpers as h
from trajectory_visualizer.insight.detectors.intake import _MEMORY_FILE_NAMES


def detect(steps: list[Any], context: DetectorContext) -> list[PatternDetection]:
    """Memory contamination ([H]).

    Operational definition (appendix_catalog.tex, [H]):
    "At session end, the agent writes incorrect or outdated information into
    a persistent memory file. Requires LLM-judge or human annotation."

    Structural pre-condition: a FILE_WRITE to a memory-file target near the end
    of the session (in the last 10 steps). Whether the content is 'incorrect
    or outdated' requires labels with `action == 'memory_contamination'` or
    `judge_memory_write == 'incorrect'`.
    """
    if not context.labels:
        return []

    # Find writes to memory files in the tail of the trajectory.
    tail_start = max(0, len(steps) - 10)
    for i in range(tail_start, len(steps)):
        step = steps[i]
        if not h.is_write(step):
            continue
        tgt = h.target(step)
        if Path(tgt).name.lower() not in _MEMORY_FILE_NAMES:
            continue
        lbl = context.labels.get(i, {})
        if (
            lbl.get("action") == "memory_contamination"
            or lbl.get("judge_memory_write") == "incorrect"
        ):
            return [
                PatternDetection(
                    detector_id="memory-contamination",
                    span=(i, i),
                    evidence={"memory_file": tgt, "judge_label": lbl},
                )
            ]
    return []
