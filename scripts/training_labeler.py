"""
Training labeler v2 — behavior, quality, value, and rule-derived decisions.

Reads a training conversation JSON, extracts assistant turns, preserves the
existing behavior `phase`/`action` fields, adds nested quality/value labels,
and derives the final keep/drop/review decision with deterministic rules.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import step_labeler
from trajectory_visualizer.insight.training_filter_rules import (
    POLICY_VERSION,
    derive_decision,
)


QUALITY_LABEL_VERSION = "quality.v1"
VALUE_LABEL_VERSION = "value.v1"

_QUALITY_VERDICTS = {"good", "usable", "flawed", "reject"}
_QUALITY_DEFECT_FLAGS = {
    "incorrect",
    "unsupported_claim",
    "instruction_violation",
    "incomplete",
    "oververbose_noise",
    "unsafe_or_sensitive",
    "format_broken",
    "context_misread",
}
_QUALITY_CONFIDENCE = {"high", "medium", "low"}
_VALUE_TIERS = {"high", "medium", "low", "none"}
_VALUE_TAGS = {
    "new_evidence_introduced",
    "strategy_pivot",
    "successful_recovery",
    "high_skill_operation",
    "verification_anchor",
    "reasoning_pattern",
    "tool_use_pattern",
    "negative_example",
}
_VALUE_CONFIDENCE = {"high", "medium", "low"}

Labeler = Callable[[dict, dict], dict]


def label_training_trajectory(
    trajectory_path: str,
    output_path: str,
    *,
    behavior_labeler: Labeler | None = None,
    quality_labeler: Labeler | None = None,
    value_labeler: Labeler | None = None,
    base_url: str = "",
    api_key: str = "",
    behavior_model: str = "",
    quality_model: str = "",
    value_model: str = "",
    provider: str = "openai",
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_content_chars: int = 8000,
    delay: float = 0.0,
    taxonomy_path: str | None = None,
    rubric_path: str | None = None,
) -> dict:
    """Label a training conversation and write a `trajectory_labels.v2` JSON."""
    taxonomy_path = taxonomy_path or str(Path(__file__).resolve().parent / "TAXONOMY_REFERENCE.md")
    rubric_path = rubric_path or str(Path(__file__).resolve().parent / "TRAINING_LABEL_REFERENCE.md")

    taxonomy_mapping, taxonomy_version = step_labeler.load_taxonomy(taxonomy_path)
    valid_phases, valid_actions, action_to_phase = step_labeler._build_valid_sets(taxonomy_mapping)
    taxonomy_text = Path(taxonomy_path).read_text(encoding="utf-8")
    rubric_text = Path(rubric_path).read_text(encoding="utf-8")

    all_steps = load_training_steps(trajectory_path)
    assistant_steps = [s for s in all_steps if s.get("role") == "assistant"]
    if not assistant_steps:
        raise ValueError("No assistant turns found in training conversation")

    behavior_labeler = behavior_labeler or _build_behavior_labeler(
        base_url=base_url,
        api_key=api_key,
        model=behavior_model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        max_content_chars=max_content_chars,
        taxonomy_text=taxonomy_text,
        valid_phases=valid_phases,
        valid_actions=valid_actions,
        action_to_phase=action_to_phase,
    )
    quality_labeler = quality_labeler or _build_quality_labeler(
        base_url=base_url,
        api_key=api_key,
        model=quality_model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        max_content_chars=max_content_chars,
        rubric_text=rubric_text,
    )
    value_labeler = value_labeler or _build_value_labeler(
        base_url=base_url,
        api_key=api_key,
        model=value_model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        max_content_chars=max_content_chars,
        rubric_text=rubric_text,
    )

    labeled_steps: list[dict] = []
    for position, step in enumerate(assistant_steps):
        context = _build_turn_context(all_steps, step)
        behavior = _normalize_behavior_label(
            behavior_labeler(step, context),
            valid_phases,
            valid_actions,
            action_to_phase,
        )
        context["behavior"] = behavior

        quality = _normalize_quality_label(quality_labeler(step, context))
        context["quality"] = quality

        value = _normalize_value_label(value_labeler(step, context))
        decision = derive_decision({"quality": quality, "value": value})

        labeled_steps.append(_build_labeled_step(step, behavior, quality, value, decision))

        if delay > 0 and position < len(assistant_steps) - 1:
            time.sleep(delay)

    output = {
        "schema_version": "trajectory_labels.v2",
        "trajectory_file": os.path.abspath(trajectory_path),
        "taxonomy_version": taxonomy_version,
        "labeled_at": datetime.now(timezone.utc).isoformat(),
        "behavior_model": behavior_model or "injected",
        "quality_model": quality_model or "injected",
        "value_model": value_model or "injected",
        "quality_label_version": QUALITY_LABEL_VERSION,
        "value_label_version": VALUE_LABEL_VERSION,
        "decision_policy_version": POLICY_VERSION,
        "steps": labeled_steps,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output


def load_training_steps(trajectory_path: str) -> list[dict]:
    """Load a training conversation and return normalized steps."""
    from trajectory_visualizer.insight.loaders import load_trajectory
    from trajectory_visualizer.insight.parser import parse_steps

    raw = load_trajectory(trajectory_path)
    if "_error" in raw:
        raise ValueError(f"Failed to load training conversation: {raw['_error']}")
    if raw.get("_analysis_profile") != "training":
        raise ValueError("Input is not a supported training conversation JSON")
    return parse_steps(raw)


def _build_behavior_labeler(
    *,
    base_url: str,
    api_key: str,
    model: str,
    provider: str,
    temperature: float | None,
    max_tokens: int | None,
    max_content_chars: int,
    taxonomy_text: str,
    valid_phases: set[str],
    valid_actions: set[str],
    action_to_phase: dict[str, str],
) -> Labeler:
    _require_llm_config(base_url, api_key, model, "behavior")
    system_prompt = step_labeler.build_system_prompt(taxonomy_text)

    def label(step: dict, context: dict) -> dict:
        del context
        response = step_labeler.call_llm(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_message=step_labeler.build_step_message(step, max_chars=max_content_chars),
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return step_labeler.parse_label_response(
            response,
            valid_phases,
            valid_actions,
            action_to_phase,
        )

    return label


def _build_quality_labeler(
    *,
    base_url: str,
    api_key: str,
    model: str,
    provider: str,
    temperature: float | None,
    max_tokens: int | None,
    max_content_chars: int,
    rubric_text: str,
) -> Labeler:
    _require_llm_config(base_url, api_key, model, "quality")
    system_prompt = build_quality_system_prompt(rubric_text)

    def label(step: dict, context: dict) -> dict:
        response = step_labeler.call_llm(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_message=build_training_turn_message(step, context, max_chars=max_content_chars),
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return parse_json_object(response)

    return label


def _build_value_labeler(
    *,
    base_url: str,
    api_key: str,
    model: str,
    provider: str,
    temperature: float | None,
    max_tokens: int | None,
    max_content_chars: int,
    rubric_text: str,
) -> Labeler:
    _require_llm_config(base_url, api_key, model, "value")
    system_prompt = build_value_system_prompt(rubric_text)

    def label(step: dict, context: dict) -> dict:
        response = step_labeler.call_llm(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_message=build_training_turn_message(step, context, max_chars=max_content_chars),
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return parse_json_object(response)

    return label


def build_quality_system_prompt(rubric_text: str) -> str:
    return f"""You are labeling assistant turns for SFT training data quality.

Use the rubric below and respond with ONLY a JSON object:
{{"verdict": "good|usable|flawed|reject", "defect_flags": [], "confidence": "high|medium|low"}}

Do not include a keep/drop/review decision.

## Rubric

{rubric_text}"""


def build_value_system_prompt(rubric_text: str) -> str:
    return f"""You are labeling assistant turns for marginal training value.

Use the rubric below and respond with ONLY a JSON object:
{{"tier": "high|medium|low|none", "tags": [], "confidence": "high|medium|low"}}

Value is marginal teaching contribution, not local answer quality. Do not
include a keep/drop/review decision.

## Rubric

{rubric_text}"""


def build_training_turn_message(step: dict, context: dict, max_chars: int = 8000) -> str:
    """Build a compact turn/context prompt for quality and value passes."""
    parts = []
    previous_user = context.get("previous_user")
    next_steps = context.get("next_steps", [])
    if previous_user:
        parts.append("Previous user turn:\n" + _step_text(previous_user))
    parts.append("Current assistant turn:\n" + step_labeler.build_step_message(step, max_chars=max_chars))
    if context.get("behavior"):
        behavior = context["behavior"]
        parts.append(f"Behavior label: {behavior.get('phase')}/{behavior.get('action')}")
    if context.get("quality"):
        parts.append("Quality label:\n" + json.dumps(context["quality"], ensure_ascii=False))
    if next_steps:
        next_preview = "\n\n".join(_step_text(s) for s in next_steps[:4])
        parts.append("Following context:\n" + next_preview)

    content = "\n\n".join(parts)
    if len(content) > max_chars:
        return content[:max_chars]
    return content


def parse_json_object(text: str) -> dict:
    """Parse a model response that should contain a single JSON object."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def _build_turn_context(all_steps: list[dict], step: dict) -> dict:
    index = step.get("index")
    position = next((i for i, candidate in enumerate(all_steps) if candidate.get("index") == index), -1)
    previous_user = None
    if position >= 0:
        for candidate in reversed(all_steps[:position]):
            if candidate.get("role") == "user":
                previous_user = candidate
                break
    next_steps = all_steps[position + 1:position + 5] if position >= 0 else []
    return {
        "all_steps": all_steps,
        "previous_user": previous_user,
        "next_steps": next_steps,
    }


def _normalize_behavior_label(
    label: dict,
    valid_phases: set[str],
    valid_actions: set[str],
    action_to_phase: dict[str, str],
) -> dict:
    phase = str(label.get("phase", "unknown")).strip().lower()
    action = str(label.get("action", "unknown")).strip().lower()
    if action in valid_actions:
        phase = action_to_phase.get(action, phase)
    elif action != "unknown":
        action = "unknown"
    if phase not in valid_phases and phase != "unknown":
        phase = "unknown"
    return {"phase": phase, "action": action}


def _normalize_quality_label(label: dict) -> dict:
    verdict = str(label.get("verdict", "")).strip().lower()
    if verdict not in _QUALITY_VERDICTS:
        verdict = "reject"
    confidence = str(label.get("confidence", "")).strip().lower()
    if confidence not in _QUALITY_CONFIDENCE:
        confidence = "low"
    flags = [
        str(flag).strip().lower()
        for flag in label.get("defect_flags", [])
        if str(flag).strip().lower() in _QUALITY_DEFECT_FLAGS
    ]
    return {
        "verdict": verdict,
        "defect_flags": flags,
        "confidence": confidence,
    }


def _normalize_value_label(label: dict) -> dict:
    tier = str(label.get("tier", "")).strip().lower()
    if tier not in _VALUE_TIERS:
        tier = "none"
    confidence = str(label.get("confidence", "")).strip().lower()
    if confidence not in _VALUE_CONFIDENCE:
        confidence = "low"
    tags = [
        str(tag).strip().lower()
        for tag in label.get("tags", [])
        if str(tag).strip().lower() in _VALUE_TAGS
    ]
    return {
        "tier": tier,
        "tags": tags,
        "confidence": confidence,
    }


def _build_labeled_step(
    step: dict,
    behavior: dict,
    quality: dict,
    value: dict,
    decision: dict,
) -> dict:
    tokens = step.get("tokens", {})
    token_total = tokens.get("total") or tokens.get("estimated_total", 0) or 0
    tool_names = [tc.get("tool_name", "?") for tc in step.get("tool_calls", [])]
    return {
        "index": step.get("index"),
        "role": "assistant",
        "phase": behavior["phase"],
        "action": behavior["action"],
        "time_created_ms": step.get("time_created_ms"),
        "time_completed_ms": step.get("time_completed_ms"),
        "duration_s": step.get("duration"),
        "tokens_total": token_total,
        "tool_calls": tool_names,
        "finish": step.get("finish", ""),
        "agent": step.get("agent", "") or step.get("agent_id", ""),
        "model_id": step.get("model_id", ""),
        "text_preview": (step.get("text_preview", "") or "")[:200],
        "quality": quality,
        "value": value,
        "decision": decision,
    }


def _step_text(step: dict) -> str:
    role = step.get("role", "?")
    text = step.get("text_preview") or step.get("output_text") or ""
    return f"{role}: {text[:800]}"


def _require_llm_config(base_url: str, api_key: str, model: str, pass_name: str) -> None:
    if not base_url:
        raise ValueError(f"{pass_name} labeler requires base_url")
    if not api_key:
        raise ValueError(f"{pass_name} labeler requires api_key")
    if not model:
        raise ValueError(f"{pass_name} labeler requires model")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label training conversation turns with v2 quality/value decisions.",
    )
    parser.add_argument("input", help="Path to training conversation JSON")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON path (default: <input>_training_labeled.json)")
    parser.add_argument("--base-url", default=None,
                        help="LLM API base URL (overrides LABEL_BASE_URL env)")
    parser.add_argument("--api-key", default=None,
                        help="LLM API key (overrides LABEL_API_KEY env)")
    parser.add_argument("--provider", default=None,
                        help="LLM provider: openai or anthropic")
    parser.add_argument("--behavior-model", default=None,
                        help="Behavior pass model")
    parser.add_argument("--quality-model", default=None,
                        help="Quality pass model")
    parser.add_argument("--value-model", default=None,
                        help="Value pass model")
    parser.add_argument("--model", default=None,
                        help="Fallback model for all passes")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-content-chars", type=int, default=8000)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--taxonomy", default=None)
    parser.add_argument("--rubric", default=None)
    args = parser.parse_args()

    fallback_model = args.model or os.getenv("LABEL_MODEL", "")
    output_path = args.output
    if output_path is None:
        inp = Path(args.input)
        output_path = str(inp.parent / f"{inp.stem}_training_labeled.json")

    label_training_trajectory(
        trajectory_path=args.input,
        output_path=output_path,
        base_url=args.base_url or os.getenv("LABEL_BASE_URL", ""),
        api_key=args.api_key or os.getenv("LABEL_API_KEY", ""),
        provider=args.provider or os.getenv("LABEL_PROVIDER", "openai"),
        behavior_model=args.behavior_model or os.getenv("BEHAVIOR_LABEL_MODEL", fallback_model),
        quality_model=args.quality_model or os.getenv("QUALITY_LABEL_MODEL", fallback_model),
        value_model=args.value_model or os.getenv("VALUE_LABEL_MODEL", fallback_model),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_content_chars=args.max_content_chars,
        delay=args.delay,
        taxonomy_path=args.taxonomy,
        rubric_path=args.rubric,
    )


if __name__ == "__main__":
    main()
