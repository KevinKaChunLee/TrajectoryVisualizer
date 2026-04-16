"""Phase 4: Validate detectors."""

from __future__ import annotations

from typing import Any

from trajectory_visualizer.core.detection import DetectorContext, PatternDetection

from . import _helpers as h


def detect_late_validation(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Late-validation detector.

    Operational definition (appendix_catalog.tex, Phase 4):
    "No validation COMMAND fires until after >=N implement steps, with no
    incremental checks in between."
    """
    thresh = int(context.thresholds_for("late-validation")["min_implement_steps_before_validate"])

    implement_count = 0
    first_implement_idx: int | None = None
    for i, step in enumerate(steps):
        if h.is_write(step):
            if first_implement_idx is None:
                first_implement_idx = i
            implement_count += 1
            continue
        if h.is_validation_command(step):
            if implement_count >= thresh and first_implement_idx is not None:
                return [
                    PatternDetection(
                        detector_id="late-validation",
                        span=(first_implement_idx, i),
                        evidence={
                            "implement_steps_before_first_validation": implement_count,
                            "first_validation_step": i,
                        },
                    )
                ]
            return []  # a validation fired early enough

    # No validation command ever fired.
    if implement_count >= thresh and first_implement_idx is not None:
        return [
            PatternDetection(
                detector_id="late-validation",
                span=(first_implement_idx, len(steps) - 1),
                evidence={
                    "implement_steps_before_first_validation": implement_count,
                    "no_validation_ever": True,
                },
            )
        ]
    return []


def detect_validation_avoidance(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Validation-avoidance detector.

    Operational definition (appendix_catalog.tex, Phase 4):
    "Implement:validate step ratio > 5:1, or the run ends after a long edit
    streak with no validation. Uses structural phase detection only."
    """
    ratio_thresh = float(context.thresholds_for("validation-avoidance")["implement_to_validate_ratio"])

    implement_count = sum(1 for s in steps if h.is_write(s))
    validate_count = sum(1 for s in steps if h.is_validation_command(s))

    if implement_count == 0:
        return []

    # Ratio-based condition.
    if validate_count == 0:
        return [
            PatternDetection(
                detector_id="validation-avoidance",
                span=(0, len(steps) - 1),
                evidence={
                    "implement_count": implement_count,
                    "validate_count": 0,
                    "mode": "no-validation",
                },
            )
        ]

    ratio = implement_count / validate_count
    if ratio > ratio_thresh:
        return [
            PatternDetection(
                detector_id="validation-avoidance",
                span=(0, len(steps) - 1),
                evidence={
                    "implement_count": implement_count,
                    "validate_count": validate_count,
                    "ratio": round(ratio, 2),
                    "mode": "ratio",
                },
            )
        ]
    return []


def detect_test_retry_loop(
    steps: list[Any], context: DetectorContext
) -> list[PatternDetection]:
    """Test-retry-loop detector.

    Operational definition (appendix_catalog.tex, Phase 4):
    "The same validation COMMAND with the same failure signature is rerun
    without a relevant intervening edit."
    """
    min_retries = int(context.thresholds_for("test-retry-loop")["min_retries"])

    detections: list[PatternDetection] = []
    # Per (command, error_signature): (first_step, total_count_since_last_edit)
    tracker: dict[tuple[str, str], list[int]] = {}
    reported: set[tuple[str, str]] = set()

    for i, step in enumerate(steps):
        if h.is_write(step):
            # An edit resets retry counters for every tracked key.
            tracker.clear()
            continue
        if not (h.is_validation_command(step) and h.is_failed(step)):
            continue
        key = (h.target(step).strip(), h.error_signature(step))
        hist = tracker.setdefault(key, [])
        hist.append(i)
        retries = len(hist) - 1  # first failure is not a retry
        if retries >= min_retries and key not in reported:
            detections.append(
                PatternDetection(
                    detector_id="test-retry-loop",
                    span=(hist[0], i),
                    evidence={
                        "command": key[0],
                        "error_signature": key[1],
                        "retries": retries,
                    },
                )
            )
            reported.add(key)
    return detections
