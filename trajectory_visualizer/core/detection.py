"""Detection data types and detector context.

Every anti-pattern and divergence detector returns `list[PatternDetection]` and
receives a `DetectorContext` with workspace state, tool affordances, semantic
labels, and overridable thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from trajectory_visualizer.core.catalog import by_id, thresholds_for


# ---------------------------------------------------------------------------
# PatternDetection: one per detector firing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatternDetection:
    """A single firing of a pattern detector.

    Fields:
      detector_id: stable kebab-case id from the catalog.
      span: (start_step_index, end_step_index), inclusive.
      evidence: dict of triggering observations. Shape depends on the detector.
      tier: copied from the catalog for divergence detectors; None otherwise.
    """

    detector_id: str
    span: tuple[int, int]
    evidence: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    tier: str | None = None


# ---------------------------------------------------------------------------
# DetectorContext: preconditions + overrides for one trajectory
# ---------------------------------------------------------------------------

@dataclass
class DetectorContext:
    """Per-trajectory context passed to every detector.

    - workspace_files: paths present in the task's workspace (lowercased, relative).
      Used by `config-gated` detectors to check for memory/instruction files.
    - tool_exposure: set of tool names the agent had access to during the run.
      Used by `tool-gated` and `capability-gated` detectors.
    - labels: per-step semantic labels (phase/action) from the labeler, keyed by
      step_index. Empty dict when labels are absent.
    - threshold_overrides: caller-supplied overrides, keyed by detector_id, then
      by threshold name. `thresholds_for` merges catalog defaults with these.
    - task_id: optional identifier for logging / label-sidecar lookup.
    - reference_trajectory: optional list of CanonicalAction for divergence
      detectors. None for single-trajectory detectors.
    - anchor_set: optional set of file paths used by anchor-dependent divergence
      detectors (reference's write set, or ground-truth changed files when
      available).
    """

    workspace_files: frozenset[str] = field(default_factory=frozenset)
    tool_exposure: frozenset[str] = field(default_factory=frozenset)
    labels: Mapping[int, Mapping[str, str]] = field(default_factory=dict)
    threshold_overrides: Mapping[str, Mapping[str, int | float]] = field(default_factory=dict)
    task_id: str | None = None
    reference_trajectory: list[Any] | None = None
    anchor_set: frozenset[str] = field(default_factory=frozenset)

    # -------------------------------------------------------------------
    # Threshold lookup (catalog defaults + per-detector overrides)
    # -------------------------------------------------------------------

    def thresholds_for(self, detector_id: str) -> Mapping[str, int | float]:
        """Return thresholds for a detector: catalog defaults overlaid with overrides."""
        defaults = dict(thresholds_for(detector_id))
        overrides = self.threshold_overrides.get(detector_id, {})
        defaults.update(overrides)
        return MappingProxyType(defaults)

    # -------------------------------------------------------------------
    # Label sidecar loading
    # -------------------------------------------------------------------

    @staticmethod
    def load_labels(task_id: str, labels_root: Path | str | None = None) -> dict[int, dict[str, str]]:
        """Load step-level labels from a sidecar JSON file.

        Looks for `{labels_root}/{task_id}.json`. If the file is missing or
        malformed, returns an empty dict (treated as "labels absent").

        The sidecar format: `{"<step_index>": {"phase": "...", "action": "..."}, ...}`.
        """
        if labels_root is None:
            return {}
        path = Path(labels_root) / f"{task_id}.json"
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[int, dict[str, str]] = {}
        for k, v in raw.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, dict):
                out[idx] = {str(kk): str(vv) for kk, vv in v.items()}
        return out

    # -------------------------------------------------------------------
    # Gating evaluation
    # -------------------------------------------------------------------

    def gating_satisfied(self, detector_id: str) -> tuple[bool, str | None]:
        """Check whether a detector's gating preconditions hold.

        Returns (True, None) when the detector may fire, or (False, reason) when
        a precondition is unmet. The runner records the reason so RQ analysis
        can distinguish "did not fire" from "could not fire".
        """
        rec = by_id(detector_id)
        # No gating = always satisfied.
        if not rec.gating and not rec.requires_semantic_labels:
            return True, None

        if rec.requires_semantic_labels and not self.labels:
            return False, "requires_semantic_labels"

        for tag in rec.gating:
            if tag == "config-gated":
                if not _has_memory_file(self.workspace_files):
                    return False, "config-gated: no memory file in workspace"
            elif tag == "tool-gated":
                if not _has_planning_tool(self.tool_exposure):
                    return False, "tool-gated: no planning tool exposed"
            elif tag == "capability-gated":
                if not _has_shell_and_structured_read(self.tool_exposure):
                    return False, "capability-gated: shell and structured-read not both exposed"
            elif tag == "weaker":
                # Informational only; does not block firing.
                continue
        return True, None


# ---------------------------------------------------------------------------
# Gating helpers (module-level so tests can exercise them directly)
# ---------------------------------------------------------------------------

_MEMORY_FILE_NAMES = frozenset({
    "claude.md",
    "agents.md",
    "gemini.md",
    "codex.md",
    "opencode.md",
    ".cursorrules",
    ".clinerules",
})

_PLANNING_TOOLS = frozenset({
    "todowrite", "taskcreate", "taskupdate", "tasklist", "plan", "update_plan",
})

_SHELL_TOOLS = frozenset({"bash", "shell", "execute_command", "terminal"})
_STRUCTURED_READ_TOOLS = frozenset({"read", "grep", "glob"})


def _has_memory_file(workspace_files: frozenset[str]) -> bool:
    return any(Path(f).name.lower() in _MEMORY_FILE_NAMES for f in workspace_files)


def _has_planning_tool(tool_exposure: frozenset[str]) -> bool:
    lowered = {t.lower() for t in tool_exposure}
    return bool(lowered & _PLANNING_TOOLS)


def _has_shell_and_structured_read(tool_exposure: frozenset[str]) -> bool:
    lowered = {t.lower() for t in tool_exposure}
    return bool(lowered & _SHELL_TOOLS) and bool(lowered & _STRUCTURED_READ_TOOLS)
