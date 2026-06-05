"""Training label v2 loading and aggregation."""

from __future__ import annotations

import json
from collections import Counter


_QUALITY_VERDICTS = {"good", "usable", "flawed", "reject"}
_QUALITY_CONFIDENCE = {"high", "medium", "low"}
_VALUE_TIERS = {"high", "medium", "low", "none"}
_VALUE_CONFIDENCE = {"high", "medium", "low"}
_DECISION_LABELS = {"keep", "drop", "review"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_string_enum(value: object, allowed: set[str], message: str) -> str:
    _require(isinstance(value, str), message)
    _require(value in allowed, message)
    return value


def load_training_labeled_json(path: str) -> dict:
    """Load and validate a training `trajectory_labels.v2` file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    _require(isinstance(data, dict), "Training labeled JSON must be an object")
    _require(data.get("schema_version") == "trajectory_labels.v2", "Training labeled JSON missing supported 'schema_version'")
    _require(isinstance(data.get("taxonomy_version"), str) and data.get("taxonomy_version"), "Training labeled JSON missing 'taxonomy_version'")
    _require(isinstance(data.get("labeled_at"), str) and data.get("labeled_at"), "Training labeled JSON missing 'labeled_at'")
    steps = data.get("steps")
    _require(isinstance(steps, list), "Training labeled JSON missing 'steps' array")

    for i, step in enumerate(steps):
        _validate_training_step(step, i)

    return data


def _validate_training_step(step: object, idx: int) -> None:
    _require(isinstance(step, dict), f"Training labeled step {idx} must be an object")
    _require("index" in step, f"Training labeled step {idx} missing 'index'")
    _require(step.get("role") == "assistant", f"Training labeled step {idx} must have role='assistant'")
    _require(isinstance(step.get("phase"), str) and step.get("phase"), f"Training labeled step {idx} missing 'phase'")
    _require(isinstance(step.get("action"), str) and step.get("action"), f"Training labeled step {idx} missing 'action'")

    quality = step.get("quality")
    _require(isinstance(quality, dict), f"Training labeled step {idx} missing 'quality'")
    _validate_string_enum(quality.get("verdict"), _QUALITY_VERDICTS, f"Training labeled step {idx} has invalid quality verdict")
    _require(isinstance(quality.get("defect_flags"), list), f"Training labeled step {idx} missing quality defect flags")
    _validate_string_enum(quality.get("confidence"), _QUALITY_CONFIDENCE, f"Training labeled step {idx} has invalid quality confidence")

    value = step.get("value")
    _require(isinstance(value, dict), f"Training labeled step {idx} missing 'value'")
    _validate_string_enum(value.get("tier"), _VALUE_TIERS, f"Training labeled step {idx} has invalid value tier")
    _require(isinstance(value.get("tags"), list), f"Training labeled step {idx} missing value tags")
    _validate_string_enum(value.get("confidence"), _VALUE_CONFIDENCE, f"Training labeled step {idx} has invalid value confidence")

    decision = step.get("decision")
    _require(isinstance(decision, dict), f"Training labeled step {idx} missing 'decision'")
    _validate_string_enum(decision.get("label"), _DECISION_LABELS, f"Training labeled step {idx} has invalid decision label")
    _require(isinstance(decision.get("reasons"), list), f"Training labeled step {idx} missing decision reasons")
    _require(isinstance(decision.get("matched_rules"), list), f"Training labeled step {idx} missing decision matched rules")
    _require(isinstance(decision.get("policy_version"), str) and decision.get("policy_version"), f"Training labeled step {idx} missing decision policy version")


def aggregate_training_labels(data: dict) -> dict:
    """Aggregate counts for training label v2 files."""
    steps = data.get("steps", [])
    quality_verdicts = Counter()
    quality_flags = Counter()
    value_tiers = Counter()
    value_tags = Counter()
    decision_counts = Counter()
    phase_counts = Counter()
    action_counts = Counter()

    for step in steps:
        quality = step.get("quality", {})
        value = step.get("value", {})
        decision = step.get("decision", {})

        quality_verdicts[quality.get("verdict", "unknown")] += 1
        for flag in quality.get("defect_flags", []):
            quality_flags[flag] += 1

        value_tiers[value.get("tier", "unknown")] += 1
        for tag in value.get("tags", []):
            value_tags[tag] += 1

        decision_counts[decision.get("label", "unknown")] += 1
        phase_counts[step.get("phase", "unknown")] += 1
        action_counts[step.get("action", "unknown")] += 1

    return {
        "total": len(steps),
        "schema_version": data.get("schema_version", ""),
        "taxonomy_version": data.get("taxonomy_version", ""),
        "quality_verdict_counts": dict(quality_verdicts),
        "quality_flag_counts": dict(quality_flags),
        "value_tier_counts": dict(value_tiers),
        "value_tag_counts": dict(value_tags),
        "decision_counts": dict(decision_counts),
        "phase_counts": dict(phase_counts),
        "action_counts": dict(action_counts),
        "steps": steps,
    }
