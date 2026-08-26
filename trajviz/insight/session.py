"""Gradio-free load pipeline: ingest a trajectory into a LoadedSession DTO."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from .analytics import compute_step_analytics
from .diagnostics import (
    PRESSURE_ALL_AGENTS,
    annotate_clusters_with_agents,
    cluster_errors,
    compute_bottleneck_explanations,
    compute_failure_chain_metrics,
    context_pressure_series,
    detect_failure_chains,
    extract_file_interactions,
    identify_target_files,
    link_chains_to_agents,
    pressure_agent_choices,
)
from .loaders import check_format_selection, detect_format, load_trajectory
from .formatting import wall_clock_fmt
from .metrics import (
    build_message_metrics,
    compute_agent_summary,
    compute_health_verdict,
    compute_metrics,
    validate_token_integrity,
)
from .parser import parse_steps
from .patterns import (
    compute_plan_metrics,
    detect_failure_patterns,
    detect_fruitless_streaks,
    detect_tool_selection_antipatterns,
    detect_tool_sequences,
    extract_plan_history,
)

# Maximum steps to process — keeps rendering, metrics, and charts bounded.
MAX_STEPS = 2000

LoadErrorCode = Literal["not_found", "mismatch", "unknown", "read_error"]

_NOISY_ROOT_CAUSE_FORMATS = frozenset({"opencode", "codearts", "codex"})


@dataclass(frozen=True)
class LoadError:
    """Failed ingest. Messages are plain text — the UI packer wraps HTML."""

    code: LoadErrorCode
    message: str
    selected: str | None = None
    detected: str | None = None


@dataclass
class LoadedSession:
    """Domain snapshot of a successfully loaded trajectory (no HTML / Plotly / Gradio)."""

    path: str
    raw: dict
    steps: list[dict]
    steps_total: int
    format: str
    token_warnings: list[str]
    message_rows: list[dict]
    metrics: dict
    verdicts: list[dict]
    agent_summaries: list[dict]
    step_analytics: list[dict]
    wall_clock: str
    anomalies: list[dict]
    tool_sequences: list[dict]
    failure_patterns: list[dict]
    file_interactions: list[dict]
    target_files: set[str]
    failure_chains: list
    chain_metrics: dict
    clusters: list
    bottleneck_explanations: list
    pressure_series: dict
    pressure_choices: list
    show_root_cause: bool
    plan_history: list
    plan_metrics: dict
    fruitless_streaks: list
    tool_selection: list
    truncated: bool = False


def compute_anomalies(message_rows: list[dict]) -> list[dict]:
    """Return a list of anomaly dicts (type, step_idx, value) from message rows."""
    anomalies: list[dict] = []
    if not message_rows:
        return anomalies

    with_dur = [r for r in message_rows if r.get("duration") is not None]
    if with_dur:
        longest = max(with_dur, key=lambda r: r["duration"])
        anomalies.append(
            {
                "type": "Slowest",
                "step_idx": longest["index"],
                "value": f"{longest['duration']:.1f}s",
            }
        )

    highest_tok = max(message_rows, key=lambda r: r["tokens_total"])
    if highest_tok["tokens_total"] > 0:
        anomalies.append(
            {
                "type": "Most Tokens",
                "step_idx": highest_tok["index"],
                "value": f"{highest_tok['tokens_total']:,} tok",
            }
        )

    asst_with_tok = [r for r in message_rows if r.get("role") == "assistant" and r["tokens_total"] > 0]
    if asst_with_tok:
        lowest_cache = min(asst_with_tok, key=lambda r: r["cache_ratio"])
        anomalies.append(
            {
                "type": "Lowest Cache",
                "step_idx": lowest_cache["index"],
                "value": f"{lowest_cache['cache_ratio'] * 100:.1f}%",
            }
        )

    with_tools = [r for r in message_rows if r["tool_calls"] > 0]
    if with_tools:
        most_tools = max(with_tools, key=lambda r: r["tool_calls"])
        anomalies.append(
            {
                "type": "Most Tools",
                "step_idx": most_tools["index"],
                "value": f"{most_tools['tool_calls']} calls",
            }
        )

    error_steps = [r for r in message_rows if r.get("error_count", 0) > 0]
    if error_steps:
        anomalies.append(
            {
                "type": "Errors",
                "step_idx": error_steps[0]["index"],
                "value": f"{len(error_steps)} step(s)",
            }
        )

    return anomalies[:5]


def _path_exists(file_path: str) -> bool:
    return os.path.isfile(file_path) or (
        os.path.isdir(file_path) and os.path.isfile(os.path.join(file_path, "session.jsonl"))
    )


def load_session(path: str, format_hint: str = "") -> LoadedSession | LoadError:
    """Load and analyze a trajectory file. Does not build HTML, Plotly, or Gradio updates."""
    if not path or not _path_exists(path):
        return LoadError(code="not_found", message="No file selected or file not found.")

    hint = format_hint or None
    raw = load_trajectory(path, format_hint=hint)
    if raw.get("_error_code") == "mismatch":
        return LoadError(
            code="mismatch",
            message=str(raw.get("_error") or "Format mismatch."),
            selected=raw.get("_selected") or format_hint or None,
            detected=raw.get("_detected") or None,
        )
    if "_error" in raw:
        return LoadError(code="read_error", message=str(raw["_error"]))

    detected = detect_format(raw)
    gate = check_format_selection(detected, format_hint)
    if gate == "unknown":
        return LoadError(
            code="unknown",
            message=(
                "Could not detect trajectory format. "
                "Select a format from the dropdown and try again."
            ),
        )
    if gate == "mismatch":
        return LoadError(
            code="mismatch",
            message=(
                f"Format mismatch: selected {format_hint} but file detected as {detected}."
            ),
            selected=format_hint or None,
            detected=detected,
        )

    steps = parse_steps(raw)
    steps_total = len(steps)
    truncated = False
    if steps_total > MAX_STEPS:
        steps = steps[:MAX_STEPS]
        truncated = True

    token_warnings = validate_token_integrity(steps)
    message_rows = build_message_metrics(steps)
    metrics = compute_metrics(steps, raw, message_rows=message_rows)
    _, wfmt = wall_clock_fmt(metrics)

    step_analytics = compute_step_analytics(steps)
    verdicts = compute_health_verdict(metrics, step_analytics if steps else [])
    agent_summaries = compute_agent_summary(steps, raw)

    interactions = extract_file_interactions(steps)
    target_files = identify_target_files(steps)
    chains = detect_failure_chains(steps)
    chains = link_chains_to_agents(chains, steps, agent_summaries)
    chain_metrics = compute_failure_chain_metrics(
        chains, sum(1 for s in steps if s.get("role") == "assistant")
    )
    clusters = cluster_errors(steps)
    clusters = annotate_clusters_with_agents(clusters, steps, agent_summaries)
    bottleneck_explanations = compute_bottleneck_explanations(steps, step_analytics)
    pressure_series = context_pressure_series(
        steps,
        agent_key=PRESSURE_ALL_AGENTS,
        raw=raw,
    )
    pressure_choices = pressure_agent_choices(steps)
    plan_history = extract_plan_history(steps)
    plan_metrics = compute_plan_metrics(plan_history)

    return LoadedSession(
        path=path,
        raw=raw,
        steps=steps,
        steps_total=steps_total,
        format=detected,
        token_warnings=list(token_warnings),
        message_rows=message_rows,
        metrics=metrics,
        verdicts=verdicts,
        agent_summaries=agent_summaries,
        step_analytics=step_analytics,
        wall_clock=wfmt,
        anomalies=compute_anomalies(message_rows),
        tool_sequences=detect_tool_sequences(steps),
        failure_patterns=detect_failure_patterns(steps),
        file_interactions=interactions,
        target_files=target_files,
        failure_chains=chains,
        chain_metrics=chain_metrics,
        clusters=clusters,
        bottleneck_explanations=bottleneck_explanations,
        pressure_series=pressure_series,
        pressure_choices=pressure_choices,
        show_root_cause=detected not in _NOISY_ROOT_CAUSE_FORMATS,
        plan_history=plan_history,
        plan_metrics=plan_metrics,
        fruitless_streaks=detect_fruitless_streaks(steps),
        tool_selection=detect_tool_selection_antipatterns(steps),
        truncated=truncated,
    )
