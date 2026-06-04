"""Deterministic keep/drop/review rules for training label v2."""

from __future__ import annotations


POLICY_VERSION = "keepdrop.v1"


def derive_decision(step_labels: dict) -> dict:
    """Derive a filtering decision from quality and value labels."""
    quality = step_labels.get("quality", {})
    value = step_labels.get("value", {})

    verdict = quality.get("verdict", "")
    defect_flags = quality.get("defect_flags", [])
    quality_confidence = quality.get("confidence", "")
    tier = value.get("tier", "")
    value_tags = value.get("tags", [])
    value_confidence = value.get("confidence", "")

    reasons: list[str] = []
    matched_rules: list[str] = []

    if "incorrect" in defect_flags:
        reasons.append("quality_incorrect")
        matched_rules.append("drop_incorrect_turn")
        return _decision("drop", reasons, matched_rules)

    if verdict == "reject":
        reasons.append("quality_reject")
        matched_rules.append("drop_rejected_turn")
        return _decision("drop", reasons, matched_rules)

    if tier == "none":
        reasons.append("value_none")
        matched_rules.append("drop_no_training_value")
        return _decision("drop", reasons, matched_rules)

    if quality_confidence == "low" or value_confidence == "low":
        reasons.append("low_confidence")
        matched_rules.append("review_low_confidence")
        return _decision("review", reasons, matched_rules)

    if "negative_example" in value_tags:
        reasons.append("negative_example")
        matched_rules.append("review_negative_example")
        return _decision("review", reasons, matched_rules)

    if verdict == "flawed" and tier in {"high", "medium"}:
        reasons.append("quality_value_conflict")
        matched_rules.append("review_conflicting_quality_and_value")
        return _decision("review", reasons, matched_rules)

    if verdict in {"good", "usable"} and tier in {"high", "medium"}:
        reasons.append(f"quality_{verdict}")
        reasons.append(f"value_{tier}")
        matched_rules.append("keep_high_quality_agentic_turn")
        return _decision("keep", reasons, matched_rules)

    reasons.append(f"quality_{verdict or 'unknown'}")
    reasons.append(f"value_{tier or 'unknown'}")
    matched_rules.append("review_default")
    return _decision("review", reasons, matched_rules)


def _decision(label: str, reasons: list[str], matched_rules: list[str]) -> dict:
    return {
        "label": label,
        "reasons": reasons,
        "matched_rules": matched_rules,
        "policy_version": POLICY_VERSION,
    }
