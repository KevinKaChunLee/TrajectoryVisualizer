"""Phase 0: Intake detectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from . import _helpers as h


_MEMORY_FILE_NAMES = frozenset({
    "claude.md",
    "agents.md",
    "gemini.md",
    "codex.md",
    "opencode.md",
    ".cursorrules",
    ".clinerules",
})


def _memory_files_in_workspace(workspace_files: frozenset[str]) -> list[str]:
    return [f for f in workspace_files if Path(f).name.lower() in _MEMORY_FILE_NAMES]


def detect_memory_bypass(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Memory bypass detector.

    Operational definition (appendix_catalog.tex, Phase 0):
    "A designated memory/instruction file (e.g., CLAUDE.md) exists in the
    workspace and is never read before the first code action."
    """
    memory_files = _memory_files_in_workspace(context.workspace_files)
    if not memory_files:
        return []  # precondition (config-gated) handled by runner; defensive here.

    memory_names_lower = {Path(f).name.lower() for f in memory_files}
    first_write_idx: int | None = None
    memory_read_before_write = False

    for i, step in enumerate(steps):
        if h.is_read(step):
            name = Path(h.target(step)).name.lower()
            if name in memory_names_lower:
                memory_read_before_write = True
        if h.is_write(step):
            # Paper says "first source-code FILE_WRITE": skip writes to the
            # memory file itself (updating CLAUDE.md is not a code action).
            target_name = Path(h.target(step)).name.lower()
            if target_name in memory_names_lower:
                continue
            if first_write_idx is None:
                first_write_idx = i
                break

    if first_write_idx is None:
        return []  # no code action in this run; not a bypass

    if memory_read_before_write:
        return []

    return [
        PatternDetection(
            detector_id="memory-bypass",
            span=(0, first_write_idx),
            evidence={
                "memory_files_present": sorted(memory_files),
                "first_write_step": first_write_idx,
                "first_write_target": h.target(steps[first_write_idx]),
            },
        )
    ]


def detect_premature_code_action(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Premature code action detector.

    Operational definition (appendix_catalog.tex, Phase 0):
    "The first source-code FILE_WRITE occurs before any repository FILE_READ
    or SEARCH."
    """
    for i, step in enumerate(steps):
        if h.is_read(step) or h.is_search(step):
            return []  # a read/search happened before any write
        if h.is_write(step):
            return [
                PatternDetection(
                    detector_id="premature-code-action",
                    span=(0, i),
                    evidence={
                        "first_write_step": i,
                        "first_write_target": h.target(step),
                    },
                )
            ]
    return []
