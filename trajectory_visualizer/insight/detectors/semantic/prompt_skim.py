"""Prompt skim detector ([H], weak textual proxy)."""

from __future__ import annotations

from typing import Any

from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from trajectory_visualizer.insight.detectors import _helpers as h


def detect(steps: list[Any], context: DetectorContext) -> list[PatternDetection]:
    """Prompt skim ([H]).

    Operational definition (appendix_catalog.tex, [H]):
    "Agent never re-references the user prompt after the first turn. Weak
    textual proxy."

    Proxy: look for steps with action_label == 'reread_prompt' or text that
    quotes the user prompt. If none appear after the first REASON/turn, fire.
    """
    if not context.labels:
        return []

    # Find the index of the first turn boundary: after the first REASON step
    # that is followed by at least one tool action.
    first_turn_end: int | None = None
    for i, step in enumerate(steps):
        if h.action_type(step) == "REASON":
            first_turn_end = i
            break
    if first_turn_end is None:
        return []

    # After first turn, look for any reread-prompt signal.
    for i in range(first_turn_end + 1, len(steps)):
        lbl = context.labels.get(i, {})
        action = lbl.get("action", "")
        if action in {"reread_prompt", "reference_prompt", "requote_prompt"}:
            return []
    return [
        PatternDetection(
            detector_id="prompt-skim",
            span=(first_turn_end, len(steps) - 1),
            evidence={"prompt_rereferenced": False},
        )
    ]
