"""Cross-cutting detectors (span multiple lifecycle phases)."""

from __future__ import annotations

import re
from typing import Any

from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from . import _helpers as h


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_search_query(step: Any) -> str:
    """Canonical form of a search query for near-duplicate comparison."""
    a = h.args(step)
    q = a.get("pattern") or a.get("query") or a.get("regex") or h.target(step) or ""
    return _WHITESPACE_RE.sub(" ", str(q).strip().lower())


def detect_redundant_search(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Redundant-search detector.

    Operational definition (appendix_catalog.tex, Cross-cutting):
    "Repeated nearly identical SEARCH query within a short window."
    """
    thresholds = context.thresholds_for("redundant-search")
    window = int(thresholds["window_steps"])
    min_dupes = int(thresholds["min_duplicates"])

    detections: list[PatternDetection] = []
    queries: dict[str, list[int]] = {}

    for i, step in enumerate(steps):
        if not h.is_search(step):
            continue
        q = _normalize_search_query(step)
        if not q:
            continue
        hist = queries.setdefault(q, [])
        hist.append(i)
        while hist and i - hist[0] > window:
            hist.pop(0)
        if len(hist) >= min_dupes:
            detections.append(
                PatternDetection(
                    detector_id="redundant-search",
                    span=(hist[0], i),
                    evidence={"query": q, "steps": list(hist)},
                )
            )
            queries[q] = []
    return detections


def detect_shell_over_tool(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Shell-over-tool detector.

    Operational definition (appendix_catalog.tex, Cross-cutting):
    "A general-purpose shell (e.g., Bash) is used for read/search when the
    same session exposes a dedicated structured tool (e.g., Read, Grep). Fires
    only when both capabilities are demonstrably available."
    """
    # Capability gating enforced by runner; do a defensive check too.
    if not (
        any(t.lower() in {"bash", "shell"} for t in context.tool_exposure)
        and any(t.lower() in {"read", "grep", "glob"} for t in context.tool_exposure)
    ):
        return []

    detections: list[PatternDetection] = []
    for i, step in enumerate(steps):
        if not h.is_shell(step):
            continue
        cmd = h.target(step).lower().strip()
        if _shell_does_read_or_search(cmd):
            detections.append(
                PatternDetection(
                    detector_id="shell-over-tool",
                    span=(i, i),
                    evidence={"command": h.target(step)},
                )
            )
    return detections


def _shell_does_read_or_search(cmd: str) -> bool:
    """True when the shell command performs a read/search that a structured tool covers."""
    first_word = cmd.split()[0] if cmd.split() else ""
    return first_word in {"cat", "head", "tail", "less", "more", "grep", "rg",
                          "ag", "find", "fgrep", "egrep"}


def detect_tool_oscillation(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Tool-oscillation detector.

    Operational definition (appendix_catalog.tex, Cross-cutting):
    "Repeated FILE_READ -> FILE_WRITE -> FILE_READ loops on the same file/range
    with no progress."
    """
    min_cycles = int(context.thresholds_for("tool-oscillation")["min_cycles"])

    # Track per-file read/write history.
    history: dict[str, list[tuple[int, str]]] = {}  # file -> [(step_idx, 'R'|'W')]

    for i, step in enumerate(steps):
        if h.is_read(step):
            tgt = h.target(step)
            if tgt:
                history.setdefault(tgt, []).append((i, "R"))
        elif h.is_write(step):
            tgt = h.target(step)
            if tgt:
                history.setdefault(tgt, []).append((i, "W"))

    detections: list[PatternDetection] = []
    for tgt, hist in history.items():
        # Count R-W-R cycles.
        cycles = 0
        cycle_spans: list[tuple[int, int]] = []
        j = 0
        while j + 2 < len(hist):
            a, b, c = hist[j], hist[j + 1], hist[j + 2]
            if a[1] == "R" and b[1] == "W" and c[1] == "R":
                cycles += 1
                cycle_spans.append((a[0], c[0]))
                j += 2  # advance past this cycle (R becomes start of next)
                continue
            j += 1
        if cycles >= min_cycles:
            start = cycle_spans[0][0]
            end = cycle_spans[-1][1]
            detections.append(
                PatternDetection(
                    detector_id="tool-oscillation",
                    span=(start, end),
                    evidence={"file": tgt, "cycles": cycles},
                )
            )
    return detections
