"""Gradio UI for TrajViz."""

import base64
import html
import json
import os

import gradio as gr
import plotly.graph_objects as go

from .parser import (
    load_trajectory,
    parse_steps,
    build_message_metrics,
    compute_metrics,
    _build_hotspots_md,
    _build_per_message_md,
    compute_health_verdict,
    validate_token_integrity,
    format_performance_md,
    format_behavioral_md,
    extract_agent_info,
    wall_clock_fmt,
    format_banner_html,
    compute_agent_summary,
)
from .analytics import compute_step_analytics
from .diagnostics import (
    extract_file_interactions,
    identify_target_files,
    detect_failure_chains,
    link_chains_to_agents,
    compute_failure_chain_metrics,
    cluster_errors,
    annotate_clusters_with_agents,
    compute_bottleneck_explanations,
    context_pressure_series,
    pressure_agent_choices,
    PRESSURE_ALL_AGENTS,
)
from .charts import (
    build_token_chart,
    build_duration_chart,
    build_tool_chart,
    build_agent_swimlane_chart,
    build_agent_token_chart,
    build_tool_outcome_timeline,
    build_label_phase_count_chart,
    build_label_action_count_chart,
    build_label_phase_duration_chart,
    build_label_action_duration_chart,
    build_label_timeline_chart,
    build_file_interaction_chart,
    build_plan_timeline_chart,
    build_error_classification_chart,
    build_context_growth_chart,
    build_context_pressure_chart,
    build_run_group_agent_timeline,
)

from .comparison import run_comparison
from .run_group import (
    build_run_group_scorecard,
    build_run_group_scorecard_html,
    build_run_group_behavior_html,
    normalize_run_paths,
)
from .patterns import (
    detect_tool_sequences,
    detect_failure_patterns,
    extract_plan_history,
    compute_plan_metrics,
    detect_fruitless_streaks,
    detect_tool_selection_antipatterns,
)
from .help import HELP_TEXT
from .loaders import (
    detect_format,
    check_format_selection,
    FORMAT_LABELS,
    FORMAT_DROPDOWN_CHOICES,
)
from .parser import load_labeled_json, aggregate_labels
from .formatting import format_context_pressure_html
from .rendering import (
    render_workflow_html,
    render_toc_sidebar,
    render_filter_chips,
    format_step_detail,
    render_agent_summary_cards,
    build_root_cause_html,
    build_antipattern_summary_html,
    _diag_jump_onclick,
)
from .styles import APP_CSS  # noqa: F401  (re-exported: __main__ passes it to app.launch(css=...))
from .report import ReportError, write_report_file

_DETAIL_PLACEHOLDER = (
    "<div id='wf-detail-content'>"
    "<div data-wf-detail-placeholder='1' style='padding:2em 1em;text-align:center;color:var(--ov-muted);'>"
    "<p style='font-size:15px;margin-bottom:0.5em;'>Select a step to inspect</p>"
    "<p style='font-size:12px;'>Click any card on the left, or press <kbd>j</kbd>/<kbd>k</kbd> to navigate</p>"
    "</div></div>"
)
_ROLE_FILTERS = ["Assistant", "User"]
_FEATURE_FILTERS = ["Tool Calls", "Errors", "Reasoning"]
_ALL_FEATURE_FILTER = "All"
_FILTER_CHIPS_DEFAULT = [*_ROLE_FILTERS, _ALL_FEATURE_FILTER]

# Maximum steps to process — keeps rendering, metrics, and charts bounded.
_MAX_STEPS = 2000


def _prepare_html_export(raw, steps, dark=False):
    """Build the HTML snapshot after a trajectory loads.

    Gradio's DownloadButton only saves in the same click that already has a
    file URL. Generating on click (then JS-clicking a hidden button) is
    blocked as a non-gesture download, and ``value=callable`` toasts on page
    load. So the report is prepared here and the button just downloads.
    """
    payload = raw if isinstance(raw, dict) else {}
    if not payload or payload.get("_error"):
        return gr.update(value=None, interactive=False)
    try:
        path = write_report_file(
            payload, steps=steps or None, dark=bool(dark),
        )
    except ReportError:
        return gr.update(value=None, interactive=False)
    return gr.update(value=path, interactive=True)


def _prerender_step_details(steps: list[dict]) -> str:
    """Pre-render all step details as HTML and return a base64-encoded JSON blob.

    ``format_step_detail()`` now returns styled HTML directly, so no
    markdown-it pass is needed.
    """
    details = {}
    for step in steps:
        details[str(step["index"])] = format_step_detail(step)
    b64 = base64.b64encode(json.dumps(details).encode()).decode()
    return f'<div data-b64="{b64}" style="display:none"></div>'


def _workflow_step_labels(step: dict) -> set[str]:
    """Return every Workflow filter label that applies to *step*.

    These labels intentionally overlap.  For example, a failed assistant tool
    call is labelled ``Assistant``, ``Tool Calls``, and ``Errors``.  The
    Workflow filter treats those labels as alternatives: matching any selected
    label keeps the step visible.
    """
    labels: set[str] = set()
    role = step.get("role")
    if role == "assistant":
        labels.add("Assistant")
    elif role == "user":
        labels.add("User")
    if step.get("tool_call_count", 0) > 0:
        labels.add("Tool Calls")
    if step.get("error_count", 0) > 0:
        labels.add("Errors")
    if step.get("has_reasoning"):
        labels.add("Reasoning")
    return labels


def _filter_workflow_steps(
    steps: list[dict],
    active_filters: list[str],
    keyword: str = "",
) -> list[int]:
    """Return positions matching required roles, optional features, and search.

    Roles are ORed with each other, selected features are ORed with each other,
    and the two groups are ANDed. ``All`` (or an omitted feature selection)
    means that no feature predicate is applied. Agent filtering is intentionally
    not exposed until its interaction with role-less/user steps is made explicit.
    """
    if not steps:
        return []

    keyword = (keyword or "").strip().lower()
    active = {str(value).strip() for value in active_filters if str(value).strip()}
    role_filters = active & set(_ROLE_FILTERS)
    if not role_filters:
        return []
    feature_filters = active & set(_FEATURE_FILTERS)
    restrict_features = _ALL_FEATURE_FILTER not in active and bool(feature_filters)

    filtered: list[int] = []
    for position, step in enumerate(steps):
        labels = _workflow_step_labels(step)
        if not (labels & role_filters):
            continue
        if restrict_features and not (labels & feature_filters):
            continue

        if keyword:
            text = str(step.get("text_preview") or "").lower()
            tool_names = " ".join(
                str(tool_call.get("tool_name", ""))
                for tool_call in step.get("tool_calls", [])
                if isinstance(tool_call, dict)
            ).lower()
            tool_args = " ".join(
                str(tool_call.get("input", ""))
                for tool_call in step.get("tool_calls", [])
                if isinstance(tool_call, dict)
            ).lower()
            if keyword not in text and keyword not in tool_names and keyword not in tool_args:
                continue
        filtered.append(position)
    return filtered


def _build_filtered_workflow_outputs(
    steps: list[dict],
    filter_csv: str,
    keyword: str,
    current_toc: str = "",
) -> tuple[str, str, str]:
    """Build filtered Workflow cards, count, and matching TOC HTML.

    ``current_toc`` carries the previous TOC HTML so a user-collapsed
    sidebar (``toc-hidden``) stays collapsed across re-renders.
    """
    if not steps:
        return (
            "<div style='padding:3em;color:var(--ov-muted);text-align:center;"
            "font-size:15px;'>Load a trajectory to see the step flow.</div>",
            "",
            "",
        )

    active_filters = [value.strip() for value in (filter_csv or "").split(",") if value.strip()]
    indices = _filter_workflow_steps(steps, active_filters, keyword)
    filtered_steps = [steps[position] for position in indices]

    if not (set(active_filters) & set(_ROLE_FILTERS)):
        workflow_html = (
            "<div style='padding:2em;color:var(--ov-muted);text-align:center;'>"
            "Select at least one role to see steps.</div>"
        )
    else:
        workflow_html = render_workflow_html(filtered_steps)

    count_html = f"<div class='wf-count'>Showing {len(filtered_steps)} of {len(steps)} steps</div>"
    collapsed = "toc-hidden" in (current_toc or "")
    return workflow_html, count_html, render_toc_sidebar(filtered_steps, collapsed=collapsed)


def _compute_anomalies(message_rows: list[dict]) -> list[dict]:
    """Return a list of anomaly dicts (type, step_idx, value_str) from message rows."""
    anomalies: list[dict] = []
    if not message_rows:
        return anomalies

    # Longest step by duration
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

    # Highest token step
    if message_rows:
        highest_tok = max(message_rows, key=lambda r: r["tokens_total"])
        if highest_tok["tokens_total"] > 0:
            anomalies.append(
                {
                    "type": "Most Tokens",
                    "step_idx": highest_tok["index"],
                    "value": f"{highest_tok['tokens_total']:,} tok",
                }
            )

    # Lowest cache ratio (assistant steps with tokens)
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

    # Most tool calls
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

    # Error steps
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


def _build_anomaly_strip_html(anomalies: list[dict]) -> str:
    """Render clickable anomaly badges with data-step-idx attributes."""
    if not anomalies:
        return ""
    badges = []
    for a in anomalies:
        idx = a["step_idx"]
        onclick = _diag_jump_onclick(idx)
        badges.append(
            f"<span class='anomaly-badge' data-step-idx='{idx}'"
            f" onclick=\"{onclick}\" style='cursor:pointer;'>"
            f"{html.escape(a['type'])}: #{idx} ({html.escape(a['value'])})"
            f"</span>"
        )
    return "<div class='anomaly-strip'>" + "".join(badges) + "</div>"


def _build_sparkline_svg(values: list[float], width: int = 100, height: int = 20) -> str:
    """Generate a minimal inline SVG sparkline from a list of values."""
    if not values or len(values) < 2:
        return ""
    max_v = max(values) or 1
    min_v = min(values)
    range_v = max_v - min_v or 1
    points = []
    for i, v in enumerate(values):
        x = round(i / (len(values) - 1) * width, 1)
        y = round(height - (v - min_v) / range_v * (height - 2) - 1, 1)
        points.append(f"{x},{y}")
    polyline = " ".join(points)
    return (
        f"<div class='ov-kpi-sparkline'>"
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
        f"<polyline points='{polyline}' fill='none' stroke='#3b82f6' stroke-width='1.5' "
        f"stroke-linecap='round' stroke-linejoin='round'/>"
        f"</svg></div>"
    )


def _build_session_detail_html(
    timing: dict,
    metadata: dict,
    *,
    model_id: str = "",
    agent_id: str = "",
) -> str:
    """Build Session Details panel as a chip grid of session environment fields.

    Uses the same inline chip style as the performance metric grid — small cards
    with uppercase label and value, wrapped in a flex grid.
    """
    md = metadata
    started = timing.get("started_at", "N/A")
    finished = timing.get("finished_at", "N/A")
    if isinstance(started, str) and len(started) > 19:
        started = started[:19].replace("T", " ")
    if isinstance(finished, str) and len(finished) > 19:
        finished = finished[:19].replace("T", " ")

    fields = [
        ("Model", model_id or md.get("model") or "N/A"),
        ("Agent", agent_id or md.get("agent", "N/A")),
        ("Start", str(started)),
        ("End", str(finished)),
        ("Session", (md.get("session_id") or "N/A")[:16]),
        ("Branch", md.get("branch") or "N/A"),
        ("Directory", md.get("directory_name") or "N/A"),
        ("Platform", (md.get("platform") or "N/A")[:24]),
    ]
    if md.get("server_version"):
        fields.insert(2, ("Version", md["server_version"]))

    chips = "".join(
        f"<div style='display:inline-flex;flex-direction:column;"
        f"background:var(--ov-card);border:1px solid var(--ov-border);"
        f"border-radius:8px;padding:6px 10px;min-width:100px;'>"
        f"<span style='font-size:10px;color:var(--ov-muted);text-transform:uppercase;'>"
        f"{html.escape(label)}</span>"
        f"<span style='font-size:13px;font-weight:500;'>"
        f"{html.escape(str(val))}</span></div>"
        for label, val in fields
    )
    return f"<div style='display:flex;flex-wrap:wrap;gap:6px;'>{chips}</div>"


def _build_overview_kpi_html(
    metrics: dict, wall_fmt: str, verdicts: list[dict] | None = None, message_rows: list[dict] | None = None
) -> str:
    """Build at-a-glance KPI card strip for Overview tab.

    When *verdicts* is provided, matching KPI cards get a colored left border
    and a tooltip with the verdict detail string.
    """
    # Build verdict lookup: metric label -> (status, detail)
    _verdict_map: dict[str, tuple[str, str]] = {}
    if verdicts:
        _metric_to_kpi = {
            "Tool Success": "Tool Success",
            "Throughput": "Tokens",
            "Token Efficiency": "Tokens",
            "Errors": "Steps",
        }
        for v in verdicts:
            kpi_label = _metric_to_kpi.get(v["metric"], "")
            if kpi_label:
                _verdict_map[kpi_label] = (v["status"], v["detail"])

    _status_colors = {
        "good": "#059669",
        "warn": "#d97706",
        "bad": "#dc2626",
    }

    # Build sparkline data from message_rows
    sparkline_data: dict[str, list[float]] = {}
    if message_rows:
        sparkline_data["Tokens"] = [r.get("tokens_total", 0) for r in message_rows]
        sparkline_data["Wall-Clock"] = [r.get("duration", 0) or 0 for r in message_rows]

    # Help text keys for each KPI label
    _label_to_help_key = {
        "Steps": "steps",
        "Wall-Clock": "wall_clock",
        "Tokens": "tokens",
        "Tool Success": "tool_success",
    }

    output_rate = metrics.get("output_tokens_per_sec")
    if isinstance(output_rate, (int, float)) and not isinstance(output_rate, bool):
        throughput_sub = f"{output_rate:,} output tok/s"
    else:
        throughput_sub = "Output throughput: N/A"
    timed_steps = metrics.get("output_throughput_timed_steps")
    throughput_steps = metrics.get("output_throughput_total_steps")
    if (
        metrics.get("output_throughput_incomplete")
        and isinstance(timed_steps, int)
        and isinstance(throughput_steps, int)
        and throughput_steps > 0
    ):
        throughput_sub += f" · {timed_steps}/{throughput_steps} timed"

    user_steps = metrics.get("user_steps", 0)
    cards = [
        (
            "Steps",
            f"{metrics.get('total_steps', 0):,}",
            f"{metrics.get('assistant_steps', 0)} assistant, {user_steps} user",
        ),
        ("Wall-Clock", wall_fmt, f"P95 {metrics.get('p95_duration', 0)}s"),
        ("Tokens", f"{metrics.get('tokens', {}).get('total', 0):,}", throughput_sub),
        ("Tool Success", f"{metrics.get('tool_success_rate', 0)}%", f"{metrics.get('tool_call_count', 0):,} calls"),
    ]
    card_html = []
    for label, value, sub in cards:
        verdict_info = _verdict_map.get(label)
        extra_style = ""
        title_attr = ""
        data_attr = ""
        if verdict_info:
            status, detail = verdict_info
            border_color = _status_colors.get(status, "#6b7280")
            extra_style = f" style='border-left:4px solid {border_color};'"
            title_attr = f" title='{html.escape(detail)}'"
            data_attr = f" data-status='{html.escape(status)}'"
        # Help tooltip
        help_key = _label_to_help_key.get(label, "")
        help_attr = ""
        if help_key and help_key in HELP_TEXT:
            help_attr = f" data-help='{html.escape(HELP_TEXT[help_key])}'"
        # Sparkline
        sparkline = ""
        if label in sparkline_data:
            sparkline = _build_sparkline_svg(sparkline_data[label])
        # Verdict detail as visible subtitle
        verdict_sub = ""
        if verdict_info:
            status, detail = verdict_info
            vcolor = _status_colors.get(status, "#6b7280")
            verdict_sub = f"<div style='font-size:11px;color:{vcolor};margin-top:2px;'>{html.escape(detail)}</div>"
        card_html.append(
            f"<div class='ov-kpi-card'{extra_style}{title_attr}{data_attr}>"
            f"<div class='ov-kpi-label'{help_attr}>{html.escape(str(label))}</div>"
            f"<div class='ov-kpi-value'>{html.escape(str(value))}</div>"
            f"<div class='ov-kpi-sub'>{html.escape(str(sub))}</div>"
            f"{verdict_sub}"
            f"{sparkline}"
            "</div>"
        )
    return "<div class='ov-kpi-grid'>" + "".join(card_html) + "</div>"


def _build_overview_outputs(
    steps: list[dict],
    raw: dict,
    metrics: dict,
    message_rows: list[dict],
    verdicts: list[dict],
    wfmt: str,
    file_path: str,
    trajectory_format: str | None = None,
) -> dict:
    """Compute overview-related outputs: banner, KPI, metadata, and metrics markdown."""
    banner = format_banner_html(os.path.basename(file_path), metrics, wfmt, trajectory_format=trajectory_format)
    kpi_html = _build_overview_kpi_html(metrics, wfmt, verdicts=verdicts, message_rows=message_rows)
    metrics_text = format_performance_md(metrics, wfmt)
    from .metrics import compute_diagnostic_metrics

    traj = raw.get("trajectory") or raw.get("messages") or []
    diag_metrics = compute_diagnostic_metrics(steps, traj) if traj else None
    behavior_text = format_behavioral_md(metrics, diag_metrics=diag_metrics)
    hotspots_text = _build_hotspots_md(message_rows)
    per_message_text = _build_per_message_md(message_rows)

    anomalies = _compute_anomalies(message_rows)
    anomaly_html = _build_anomaly_strip_html(anomalies)

    return {
        "banner": banner,
        "anomaly_html": anomaly_html,
        "kpi_html": kpi_html,
        "metrics_text": metrics_text,
        "behavior_text": behavior_text,
        "hotspots_text": hotspots_text,
        "per_message_text": per_message_text,
    }


def _build_chart_outputs(
    steps: list[dict],
    message_rows: list[dict],
    agent_summaries: list[dict],
    dark: bool = False,
    trajectory_format: str | None = None,
) -> dict:
    """Build all chart figures and analytics markdown."""

    # Compute diagnostic data for charts
    plan_history = extract_plan_history(steps)
    plan_metrics = compute_plan_metrics(plan_history)
    fruitless_streaks = detect_fruitless_streaks(steps)
    tool_selection = detect_tool_selection_antipatterns(steps)

    # Core charts
    tok_fig = build_token_chart(steps, dark=dark, format=trajectory_format)
    dur_fig = build_duration_chart(steps, dark=dark)
    tl_fig = build_tool_chart(steps, dark=dark)

    tool_outcome_fig = build_tool_outcome_timeline(steps, dark=dark)

    agent_tok_fig = build_agent_token_chart(agent_summaries, dark=dark)
    swimlane_fig = build_agent_swimlane_chart(steps, dark=dark)

    # New diagnostic charts
    plan_timeline_fig = build_plan_timeline_chart(plan_history, plan_metrics, dark=dark)
    error_class_fig = build_error_classification_chart(steps, dark=dark)

    context_growth_fig = build_context_growth_chart(message_rows, dark=dark)

    # New panels
    error_count = sum(1 for s in steps for tc in s.get("tool_calls", []) if tc.get("error_type"))
    antipattern_html = build_antipattern_summary_html(
        fruitless_streaks,
        tool_selection,
        plan_metrics,
        error_count=error_count,
    )

    return {
        "tok_fig": tok_fig,
        "dur_fig": dur_fig,
        "tl_fig": tl_fig,
        "tool_outcome_fig": tool_outcome_fig,
        "agent_cards_html": render_agent_summary_cards(agent_summaries),
        "agent_tok_fig": agent_tok_fig,
        "swimlane_fig": swimlane_fig,
        "plan_timeline_fig": plan_timeline_fig,
        "error_class_fig": error_class_fig,
        "antipattern_html": antipattern_html,
        "context_growth_fig": context_growth_fig,
    }


def trajectory_format_label(fmt: str | None) -> str:
    """Return a human-readable trajectory format label."""
    return FORMAT_LABELS.get(fmt or "", fmt or "Unknown")


def _build_diagnostics_outputs(
    steps: list[dict],
    step_analytics: list[dict],
    agent_summaries: list[dict],
    dark: bool = False,
    trajectory_format: str | None = None,
    raw: dict | None = None,
) -> dict:
    """Build all diagnostics outputs: file interactions, failure chains, root causes, bottlenecks."""
    _empty_fig = go.Figure()
    _empty_fig.update_layout(
        template="plotly_white", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )

    # File interaction analysis
    interactions = extract_file_interactions(steps)
    target_files = identify_target_files(steps)
    file_chart = build_file_interaction_chart(interactions, target_files, dark=dark) if interactions else _empty_fig

    # Failure chain analysis (metrics feed the summary line; UI strip removed)
    chains = detect_failure_chains(steps)
    chains = link_chains_to_agents(chains, steps, agent_summaries)
    chain_metrics = compute_failure_chain_metrics(chains, sum(1 for s in steps if s.get("role") == "assistant"))

    # Root-cause attribution. The findings panel is suppressed for OpenCode,
    # CodeArts, and Codex because their tool-error reporting is noisy (e.g.
    # OpenCode surfaces every non-zero bash exit code as a "failure") and the
    # cluster summaries become misleading.
    # Counts are still computed so the summary line stays accurate.
    clusters = cluster_errors(steps)
    clusters = annotate_clusters_with_agents(clusters, steps, agent_summaries)
    if trajectory_format in ("opencode", "codearts", "codex"):
        rootcause_html = ""
    else:
        rootcause_html = build_root_cause_html(clusters)

    # Bottleneck explanation (explanations feed the summary line; UI cards removed)
    bottleneck_explanations = compute_bottleneck_explanations(steps, step_analytics)

    pressure_series = context_pressure_series(
        steps,
        agent_key=PRESSURE_ALL_AGENTS,
        raw=raw,
    )
    pressure_fig = build_context_pressure_chart(
        steps,
        agent_key=PRESSURE_ALL_AGENTS,
        raw=raw,
        dark=dark,
    )
    pressure_html = format_context_pressure_html(pressure_series)
    pressure_choices = pressure_agent_choices(steps)
    # Hide the selector unless there is more than one agent to choose from
    # (All agents + at least two agent entries).
    pressure_dropdown = {
        "choices": pressure_choices or [("All agents", PRESSURE_ALL_AGENTS)],
        "value": PRESSURE_ALL_AGENTS,
        "visible": len(pressure_choices) > 2,
    }

    # Summary line
    parts = []
    if chain_metrics["total_chains"]:
        parts.append(f"{chain_metrics['total_chains']} failure chain(s)")
    if clusters:
        parts.append(f"{len(clusters)} root cause(s)")
    if bottleneck_explanations:
        parts.append(f"{len(bottleneck_explanations)} hotspot(s)")
    if interactions:
        unique_files = len({i["path"] for i in interactions})
        parts.append(f"{unique_files} file(s) touched · {len(target_files)} edited")
    compaction_count = len(pressure_series.get("events") or [])
    if compaction_count:
        parts.append(f"{compaction_count} compaction event(s)")
    summary = " &middot; ".join(parts) if parts else "No diagnostic issues detected."
    summary_html = f"<div style='font-size:12px;color:var(--ov-muted);margin-bottom:8px;'>{summary}</div>"

    return {
        "diag_summary_html": summary_html,
        "diag_pressure_html": pressure_html,
        "diag_pressure_dropdown": pressure_dropdown,
        "diag_pressure_chart": pressure_fig,
        "diag_file_chart": file_chart,
        "diag_rootcause_html": rootcause_html,
    }


def _build_workflow_outputs(steps: list[dict]) -> dict:
    """Build workflow HTML, TOC, filter chips, and detail store."""
    wf_html = render_workflow_html(steps)
    wf_count = f"<div class='wf-count'>Showing {len(steps)} of {len(steps)} steps</div>"
    toc_html_val = render_toc_sidebar(steps)
    detail_store_val = _prerender_step_details(steps)

    wf_chips = render_filter_chips()
    wf_filter_val = ",".join(_FILTER_CHIPS_DEFAULT)

    return {
        "wf_chips": wf_chips,
        "wf_filter_val": wf_filter_val,
        "wf_count": wf_count,
        "toc_html_val": toc_html_val,
        "wf_html": wf_html,
        "detail_store_val": detail_store_val,
    }


def _render_tool_sequences_html(sequences: list[dict]) -> str:
    """Render tool sequence patterns as an HTML table."""
    if not sequences:
        return "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>No recurring tool sequences detected (minimum frequency: 3).</div>"
    rows = []
    for s in sequences[:20]:
        seq_str = " → ".join(html.escape(t) for t in s["sequence"])
        indices = ", ".join(str(i) for i in s["step_indices"][:10])
        if len(s["step_indices"]) > 10:
            indices += f" … (+{len(s['step_indices']) - 10} more)"
        rows.append(
            f"<tr style='border-bottom:1px solid var(--ov-border);'>"
            f"<td style='padding:6px 10px;font-family:monospace;font-size:12px;'>{seq_str}</td>"
            f"<td style='padding:6px 10px;text-align:center;font-weight:700;'>{s['frequency']}</td>"
            f"<td style='padding:6px 10px;font-size:11px;color:var(--ov-muted);'>{indices}</td>"
            f"</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        "<tr style='background:var(--ov-table-header-bg);'>"
        "<th style='text-align:left;padding:6px 10px;'>Sequence</th>"
        "<th style='text-align:center;padding:6px 10px;'>Count</th>"
        "<th style='text-align:left;padding:6px 10px;'>Step Indices</th></tr>" + "".join(rows) + "</table>"
    )


def _count_bar_chart(counts: dict[str, int], title: str, dark: bool = False) -> go.Figure:
    """Build a small categorical count chart for label summaries."""
    fig = go.Figure(
        go.Bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            text=list(counts.values()),
            textposition="outside",
            marker_color="#2563eb",
        )
    )
    fig.update_layout(template="plotly_dark" if dark else "plotly_white", height=380, title=title)
    return fig


def build_label_ui_payload(file_path: str, dark: bool = False) -> dict:
    """Build UI-facing label payload for a *_labeled.json label file."""
    data = load_labeled_json(file_path)
    agg = aggregate_labels(data)
    pc_fig = build_label_phase_count_chart(agg["phase_counts"], dark=dark)
    ac_fig = build_label_action_count_chart(agg["action_counts"], agg["action_to_phase"], dark=dark)
    pd_fig = build_label_phase_duration_chart(agg["phase_durations"], dark=dark)
    ad_fig = build_label_action_duration_chart(agg["action_durations"], agg["action_to_phase"], dark=dark)
    tl_fig = build_label_timeline_chart(agg["steps"], dark=dark)

    n_steps = len(agg.get("steps", []))
    phase_counts = agg.get("phase_counts", {})
    n_phases = len(phase_counts)

    from .palette import LABEL_PHASE_COLORS

    bar_segments = "".join(
        f"<span style='flex:{count};background:{LABEL_PHASE_COLORS.get(phase, '#6b7280')};height:8px;'"
        f" title='{html.escape(str(phase))}: {count}'></span>"
        for phase, count in phase_counts.items()
        if count > 0
    )
    phase_bar = (
        (
            "<div style='display:flex;border-radius:4px;overflow:hidden;width:200px;margin-top:4px;'>"
            f"{bar_segments}</div>"
        )
        if bar_segments
        else ""
    )

    phase_chips = "".join(
        f"<span style='display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;"
        f"background:{LABEL_PHASE_COLORS.get(phase, '#6b7280')};color:white;'>"
        f"{html.escape(str(phase))}: {count}</span>"
        for phase, count in phase_counts.items()
        if count > 0
    )

    badge = (
        "<div style='display:flex;flex-direction:column;gap:6px;margin:6px 0;'>"
        "<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>"
        "<span style='background:#059669;color:white;"
        "padding:4px 12px;border-radius:12px;font-size:13px;'>"
        f"Labels loaded — {n_steps} steps, {n_phases} phases</span>"
        "<a href='#' onclick=\"var t=document.querySelectorAll('button[role=tab]');"
        "for(var i=0;i<t.length;i++){if(t[i].textContent.trim()==='Overview'){t[i].click();break;}}"
        "var r=document.querySelector(&quot;.overview-section-radio input[value='Labels']&quot;);"
        "if(r){r.click();r.scrollIntoView({behavior:'smooth',block:'center'});}"
        "return false;\" style='font-size:12px;color:#059669;text-decoration:underline;cursor:pointer;'>"
        "Jump to Labels</a>"
        "</div>"
        f"<div style='display:flex;gap:4px;flex-wrap:wrap;'>{phase_chips}</div>"
        f"{phase_bar}"
        "</div>"
    )
    return {
        "badge_html": badge,
        "status_html": "",
        "phase_count_fig": pc_fig,
        "action_count_fig": ac_fig,
        "phase_duration_fig": pd_fig,
        "action_duration_fig": ad_fig,
        "timeline_fig": tl_fig,
    }


def _render_failure_patterns_html(patterns: list[dict]) -> str:
    """Render failure pattern clusters as expandable cards."""
    if not patterns:
        return "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>No errors detected — no failure patterns to analyze.</div>"
    cards = []
    for p in patterns:
        label = html.escape(p.get("cluster_label", "Unknown"))
        count = p.get("count", 0)
        example = html.escape(str(p.get("example_error", ""))[:200])
        recovery = p.get("recovery_path")
        recovery_html = " → ".join(html.escape(t) for t in recovery) if recovery else "<em>No recovery path found</em>"
        step_ids = p.get("steps") or []
        steps_html = (
            "".join(
                f"<span style='display:inline-block;padding:1px 6px;margin:0 4px 2px 0;"
                f"border-radius:8px;background:var(--ov-table-header-bg);"
                f"font-size:11px;font-variant-numeric:tabular-nums;'>#{int(idx)}</span>"
                for idx in step_ids
            )
            if step_ids
            else "<em>—</em>"
        )
        cards.append(
            f"<div class='overview-card' style='margin-bottom:8px;'>"
            f"<div style='font-weight:700;font-size:13px;color:var(--ov-text);'>{label} "
            f"<span style='font-size:11px;color:var(--ov-muted);font-weight:400;'>({count} occurrences)</span></div>"
            f"<div style='font-size:12px;color:var(--ov-muted);margin:4px 0;'>{example}</div>"
            f"<div style='font-size:12px;margin-bottom:4px;'><strong>Steps:</strong> {steps_html}</div>"
            f"<div style='font-size:12px;'><strong>Recovery path:</strong> {recovery_html}</div>"
            f"</div>"
        )
    return "".join(cards)


def build_ui() -> gr.Blocks:
    """Build the full Gradio Blocks UI."""

    with gr.Blocks(title="TrajViz", elem_classes=["trajectory-viz"]) as app:
        # Per-session state via gr.State
        state_steps = gr.State([])
        state_dark = gr.State(False)
        state_raw = gr.State({})

        gr.HTML(
            "<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:4px;'>"
            "<span style='font-size:20px;font-weight:700;letter-spacing:-0.02em;'>TrajViz</span>"
            "<span style='font-size:12px;color:var(--ov-muted);'>Coding Agent Trajectory Analysis Dashboard</span>"
            "</div>"
        )

        # -- Upload area (collapses after load) --
        with gr.Column(elem_classes=["upload-row"]) as upload_accordion, gr.Row(equal_height=True):
            format_selector = gr.Dropdown(
                label="Format",
                choices=FORMAT_DROPDOWN_CHOICES,
                value="",
                interactive=True,
                scale=1,
                min_width=140,
            )
            with gr.Column(scale=2, min_width=200):
                file_upload = gr.File(
                    label="Trajectory (.json / .jsonl)",
                    file_types=[".json", ".jsonl"],
                    height=110,
                )
                with gr.Row():
                    load_btn = gr.Button("Load Trajectory", variant="primary", size="sm", min_width=120)
                    export_btn = gr.DownloadButton(
                        label="Export HTML",
                        variant="secondary",
                        size="sm",
                        min_width=120,
                        interactive=False,
                    )
            with gr.Column(scale=2, min_width=200):
                label_file_upload = gr.File(
                    label="Labels (optional)",
                    file_types=[".json"],
                    height=110,
                )
                label_load_btn = gr.Button("Load Labels", variant="secondary", size="sm", min_width=120)

        # Summary banner + anomaly strip (hidden until load)
        with gr.Column(visible=False) as summary_area:
            summary_banner = gr.HTML("", elem_classes=["summary-banner"])
            label_badge_html = gr.HTML("")
            anomaly_strip_html = gr.HTML("")

        # -- Tabs (hidden until trajectory loaded) --
        with gr.Tabs(visible=False) as main_tabs:
            # ===== Overview Tab (unified — includes former Analytics content) =====
            with gr.TabItem("Overview"):
                session_detail_html = gr.HTML("")
                overview_kpi_html = gr.HTML("", elem_classes=["overview-kpi-strip"])

                overview_section_names = [
                    "Performance",
                    "Efficiency",
                    "Tools",
                    "Agents",
                    "Diagnostics",
                    "Deep Dive",
                    "Labels",
                ]
                with gr.Row(elem_classes=["overview-content-layout"]):
                    with gr.Column(scale=0, min_width=160, elem_classes=["overview-section-nav"]):
                        gr.HTML("<div class='overview-nav-title'>Contents</div>")
                        overview_section = gr.Radio(
                            choices=overview_section_names,
                            value="Performance",
                            show_label=False,
                            container=False,
                            elem_classes=["overview-section-radio"],
                        )

                    with gr.Column(scale=1, min_width=0, elem_classes=["overview-section-content"]):
                        with gr.Column(visible=True) as performance_section:
                            gr.HTML(
                                f"<div class='section-subtitle'>{html.escape(HELP_TEXT['section_performance'])}</div>"
                            )
                            metrics_md = gr.Markdown("")
                            with gr.Row(equal_height=True):
                                token_chart = gr.Plot(show_label=False, label="Token Usage")
                                duration_chart = gr.Plot(show_label=False, label="Step Duration")

                        with gr.Column(visible=False) as efficiency_section:
                            gr.HTML(
                                f"<div class='section-subtitle'>{html.escape(HELP_TEXT['section_efficiency'])}</div>"
                            )
                            context_growth_chart = gr.Plot(show_label=False, label="Context Growth")

                        with gr.Column(visible=False) as tools_section:
                            gr.HTML(f"<div class='section-subtitle'>{html.escape(HELP_TEXT['section_tools'])}</div>")
                            behavior_md = gr.Markdown("")
                            with gr.Row(equal_height=True):
                                tool_chart = gr.Plot(show_label=False, label="Tool Call Frequency")
                                gr.Column(scale=1)  # reserved for future chart
                            with gr.Row(equal_height=True):
                                tool_outcome_chart = gr.Plot(show_label=False, label="Tool Outcome Timeline")

                        with gr.Column(visible=False) as agents_section:
                            gr.HTML(f"<div class='section-subtitle'>{html.escape(HELP_TEXT['section_agents'])}</div>")
                            agent_summary_html = gr.HTML("")
                            with gr.Row(equal_height=True):
                                agent_token_chart = gr.Plot(show_label=False, label="Token Breakdown by Agent")
                                gr.Column(scale=1)  # reserved for future chart
                            with gr.Row(equal_height=True):
                                agent_swimlane_chart = gr.Plot(show_label=False, label="Agent Swimlane")

                        with gr.Column(visible=False) as diagnostics_section:
                            gr.HTML(
                                f"<div class='section-subtitle'>{html.escape(HELP_TEXT['section_diagnostics'])}</div>"
                            )
                            diag_summary_html = gr.HTML("")
                            diag_file_chart = gr.Plot(
                                show_label=False, label="File Interaction Timeline", elem_classes=["resizable-chart"]
                            )
                            diag_pressure_html = gr.HTML("")
                            diag_pressure_agent = gr.Dropdown(
                                label="Agent",
                                choices=[("All agents", PRESSURE_ALL_AGENTS)],
                                value=PRESSURE_ALL_AGENTS,
                                visible=False,
                                interactive=True,
                            )
                            diag_pressure_chart = gr.Plot(
                                show_label=False,
                                label="Context Window Pressure",
                            )
                            diag_rootcause_html = gr.HTML("")
                            with gr.Row(equal_height=True):
                                error_class_chart = gr.Plot(show_label=False, label="Tool Error Classification")
                                plan_timeline_chart = gr.Plot(show_label=False, label="Plan Progress Timeline")

                        with gr.Column(visible=False) as deep_dive_section:
                            hotspots_md = gr.Markdown("")
                            per_message_md = gr.Markdown("")

                        with gr.Column(visible=False) as labels_section:
                            gr.HTML(
                                "<div class='section-subtitle'>Phase and action classification from labeled JSON</div>"
                            )
                            label_status_html = gr.HTML(
                                "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>"
                                "Upload a <code>*_labeled.json</code> file to view label distributions and timeline.</div>"
                            )
                            with gr.Row(equal_height=True, visible=False) as label_charts_row1:
                                label_phase_count_chart = gr.Plot(show_label=False, label="Phase Count Distribution")
                                label_action_count_chart = gr.Plot(show_label=False, label="Action Count Distribution")
                            with gr.Row(equal_height=True, visible=False) as label_charts_row2:
                                label_phase_dur_chart = gr.Plot(show_label=False, label="Phase Duration Distribution")
                                label_action_dur_chart = gr.Plot(show_label=False, label="Action Duration Distribution")
                            with gr.Row(equal_height=True, visible=False) as label_timeline_row:
                                label_timeline_chart = gr.Plot(show_label=False, label="Step Timeline")

            # ===== Patterns Tab =====
            with gr.TabItem("Patterns"):
                gr.HTML(
                    "<div class='section-subtitle'>Recurring patterns detected in the trajectory — tool sequences, failure clusters, and phase transition anomalies.</div>"
                )
                with gr.Accordion("Tool Sequence Patterns", open=True, elem_classes=["per-message-acc"]):
                    patterns_tool_html = gr.HTML(
                        "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>Load a trajectory to detect tool sequence patterns.</div>"
                    )
                with gr.Accordion("Failure Patterns", open=True, elem_classes=["per-message-acc"]):
                    patterns_failure_html = gr.HTML(
                        "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>Load a trajectory to detect failure patterns.</div>"
                    )
                with gr.Accordion("Anti-Pattern Summary", open=True, elem_classes=["per-message-acc"]):
                    antipattern_summary_html = gr.HTML(
                        "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>Load a trajectory to detect anti-patterns.</div>"
                    )

            # ===== Attribution Tab (DECAF failure attribution) =====
            with gr.TabItem("Attribution"):
                gr.HTML(
                    "<div class='section-subtitle'>DECAF capability failure attribution "
                    "&mdash; which of the seven workflow capabilities broke, with tiered "
                    "evidence (deductive / associational / model-inferred). Gold-grounded: "
                    "needs the reference patch + test outcome for this task.</div>"
                )
                _attr_placeholder = (
                    "<div style='padding:2em;color:var(--ov-muted);text-align:center;font-size:14px;'>"
                    "Load a trajectory in the Overview tab &mdash; diagnosis runs automatically on load (<b>Diagnose failure</b> re-runs it with the overrides below). "
                    "For a corpus trajectory (…/trajectory/&lt;agent&gt;/&lt;instance&gt;.json) the agent "
                    "and instance are auto-detected from the path; for an uploaded file, set them below."
                    "</div>"
                )
                attr_status_html = gr.HTML(_attr_placeholder)
                with gr.Row(equal_height=True):
                    attr_run_btn = gr.Button("Diagnose failure", variant="primary", size="sm", scale=1, min_width=140)
                with (
                    gr.Accordion(
                        "Override agent / instance / corpus (for uploaded files)",
                        open=False,
                        elem_classes=["per-message-acc"],
                    ),
                    gr.Row(equal_height=True),
                ):
                    attr_agent_override = gr.Textbox(label="Agent", placeholder="auto-detected from path", scale=1)
                    attr_inst_override = gr.Textbox(label="Instance id", placeholder="auto-detected from path", scale=2)
                    attr_root_override = gr.Textbox(
                        label="ARGUS corpus root", placeholder="default: sibling TraceProbe checkout", scale=2
                    )
                attr_result_html = gr.HTML("")

            # ===== Comparison Tab (Converge embedded + N-run scorecard) =====
            with gr.TabItem("Comparison"):
                _cmp_placeholder = (
                    "<div style='padding:2em;color:var(--ov-muted);text-align:center;font-size:14px;'>"
                    "Load a trajectory in the Overview tab first &mdash; it becomes the "
                    "<b>baseline</b> for <b>Run group</b> and the <b>compared</b> trajectory "
                    "for pairwise comparison."
                    "<br><span style='font-size:12px;'>"
                    "In Run group, upload one or more additional runs to scorecard against Overview."
                    "</span></div>"
                )
                cmp_status_html = gr.HTML(_cmp_placeholder)
                with gr.Accordion("Run group (N trajectories)", open=True, elem_classes=["per-message-acc"]):
                    gr.Markdown(
                        "_The trajectory loaded in **Overview** is included as the "
                        "**baseline**. Upload one or more additional runs of the same "
                        "task (different models, harnesses, or prompts). Builds a "
                        "metrics scorecard, agent timeline, behavioral similarity, "
                        "tool/skill coverage, action/file matrices, and waste patterns "
                        "vs the Overview run. For a full pairwise report, use "
                        "**Pairwise comparison** below._",
                    )
                    with gr.Row(equal_height=True):
                        rg_format_selector = gr.Dropdown(
                            label="Format hint",
                            choices=FORMAT_DROPDOWN_CHOICES,
                            value="",
                            interactive=True,
                            scale=1,
                            min_width=140,
                        )
                        rg_file_upload = gr.File(
                            label="Comparison runs (.json / .jsonl) — select one or more",
                            file_types=[".json", ".jsonl"],
                            file_count="multiple",
                            scale=3,
                        )
                    with gr.Row(equal_height=False):
                        rg_run_btn = gr.Button(
                            "Build scorecard",
                            variant="primary",
                            size="sm",
                            scale=0,
                            min_width=140,
                        )
                    rg_scorecard_html = gr.HTML(
                        "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>"
                        "Load a trajectory in <b>Overview</b>, upload one or more "
                        "comparison runs, then click <b>Build scorecard</b>.</div>"
                    )
                    rg_agent_timeline_chart = gr.Plot(
                        show_label=False,
                        label="Agent timeline (by run)",
                        visible=False,
                    )
                    rg_behavior_html = gr.HTML("")
                with gr.Accordion("Pairwise comparison", open=False, elem_classes=["per-message-acc"]):
                    with gr.Row(equal_height=True):
                        cmp_format_selector = gr.Dropdown(
                            label="Format",
                            choices=FORMAT_DROPDOWN_CHOICES,
                            value="",
                            interactive=True,
                            scale=1,
                            min_width=140,
                        )
                        cmp_file_upload = gr.File(
                            label="Reference Trajectory (.json / .jsonl)",
                            file_types=[".json", ".jsonl"],
                            scale=2,
                        )
                        cmp_anchor_upload = gr.File(
                            label="Anchor Patch (optional, .patch/.diff)",
                            file_types=[".patch", ".diff"],
                            scale=1,
                        )
                    with gr.Row(equal_height=True):
                        cmp_ref_labels_upload = gr.File(
                            label="Reference Labels (optional, *_labeled.json)",
                            file_types=[".json"],
                            scale=4,
                        )
                    with gr.Row(equal_height=False):
                        # Roles are fixed in the pipeline: the upload here is the
                        # reference/baseline, the Overview trajectory is compared.
                        gr.Markdown(
                            "_The uploaded trajectory is treated as the "
                            "**reference/baseline**; the trajectory loaded on the "
                            "Overview tab is the **compared** one._",
                        )
                        cmp_run_btn = gr.Button(
                            "Run Comparison",
                            variant="primary",
                            size="sm",
                            scale=1,
                            min_width=140,
                        )
                    cmp_report_html = gr.HTML("")
                    with gr.Row(equal_height=True):
                        cmp_phase_count_chart = gr.Plot(
                            show_label=False, label="Step Count by Phase — Reference vs Compared"
                        )
                        cmp_phase_duration_chart = gr.Plot(
                            show_label=False, label="Duration by Phase — Reference vs Compared"
                        )

            # ===== Workflow Tab =====
            with gr.TabItem("Workflow"):
                with gr.Row(equal_height=True):
                    wf_toc_toggle = gr.Button("TOC", variant="secondary", scale=0, min_width=50)
                    wf_filter_chips_html = gr.HTML(
                        render_filter_chips(),
                        elem_id="wf-filter-chips",
                    )
                    wf_search = gr.Textbox(
                        label="Search",
                        placeholder="Filter by keyword...",
                        scale=1,
                    )
                # Hidden textbox that JS writes active filters into (comma-separated)
                wf_filter_hidden = gr.Textbox(
                    value=",".join(_FILTER_CHIPS_DEFAULT),
                    # Keep the component mounted so the delegated chip handler
                    # can update it and trigger Gradio's input event.  Gradio
                    # omits visible=False components from the browser DOM.
                    visible=True,
                    elem_id="wf-filter-hidden",
                )
                wf_count_html = gr.HTML("")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=0, min_width=150):
                        toc_html = gr.HTML("", elem_id="wf-toc-container")
                    with gr.Column(scale=3, min_width=400):
                        workflow_html = gr.HTML(
                            "<div style='padding:3em;color:var(--ov-muted);text-align:center;"
                            "font-size:15px;'>Load a trajectory to see the step flow.</div>",
                            js_on_load="""
                            /* Filter chip click handler (delegated, survives re-renders) */
                            window.__syncWorkflowFilters = function(bar) {
                                if (!bar) return;
                                var active = Array.from(bar.querySelectorAll('.filter-chip.chip-active'))
                                    .map(function(c) { return c.dataset.filter; });
                                var hiddenEl = document.querySelector(
                                    '#wf-filter-hidden textarea, #wf-filter-hidden input'
                                );
                                if (!hiddenEl) return;
                                var proto = hiddenEl.tagName === 'TEXTAREA'
                                    ? window.HTMLTextAreaElement.prototype
                                    : window.HTMLInputElement.prototype;
                                var descriptor = Object.getOwnPropertyDescriptor(proto, 'value');
                                if (descriptor && descriptor.set) {
                                    descriptor.set.call(hiddenEl, active.join(','));
                                } else {
                                    hiddenEl.value = active.join(',');
                                }
                                hiddenEl.dispatchEvent(new InputEvent('input', {
                                    bubbles: true,
                                    composed: true,
                                    inputType: 'insertText',
                                    data: null
                                }));
                                hiddenEl.dispatchEvent(new Event('change', {
                                    bubbles: true,
                                    composed: true
                                }));
                            };
                            window.__setWorkflowChipActive = function(chip, active) {
                                if (!chip) return;
                                chip.classList.toggle('chip-active', active);
                                chip.setAttribute('aria-pressed', active ? 'true' : 'false');
                            };
                            window.__updateWorkflowFilterQuery = function(root) {
                                if (!root) return;
                                var roles = Array.from(root.querySelectorAll(
                                    '[data-filter-group="role"].chip-active'
                                )).map(function(c) { return c.dataset.filter; });
                                var features = Array.from(root.querySelectorAll(
                                    '[data-filter-group="feature"].chip-active'
                                )).map(function(c) { return c.dataset.filter; });
                                var query = root.querySelector('#wf-filter-query');
                                if (!query) return;
                                var featureText = features.indexOf('All') >= 0
                                    ? 'All'
                                    : features.join(' or ');
                                query.textContent = 'Role: ' + roles.join(' or ')
                                    + ' · Step feature: ' + featureText;
                            };
                            /* Pure chip state machine, unit-tested in
                               tests/test_workflow_filtering.py by executing this
                               exact source in Node. Keep it DOM-free. */
                            /* __WF_CHIP_STATE_BEGIN__ */
                            window.__wfComputeChipState = function(state, action) {
                                var roles = {};
                                var features = {};
                                Object.keys(state.roles).forEach(function(k) { roles[k] = !!state.roles[k]; });
                                Object.keys(state.features).forEach(function(k) { features[k] = !!state.features[k]; });
                                var rejected = false;
                                if (action.type === 'reset') {
                                    Object.keys(roles).forEach(function(k) { roles[k] = true; });
                                    Object.keys(features).forEach(function(k) { features[k] = (k === 'All'); });
                                } else if (action.group === 'role') {
                                    var activeRoles = Object.keys(roles).filter(function(k) { return roles[k]; });
                                    if (roles[action.name] && activeRoles.length === 1) {
                                        /* Refuse to deselect the last active role. */
                                        rejected = true;
                                    } else {
                                        roles[action.name] = !roles[action.name];
                                    }
                                } else if (action.name === 'All') {
                                    /* 'All' is exclusive with specific features. */
                                    Object.keys(features).forEach(function(k) { features[k] = (k === 'All'); });
                                } else {
                                    features[action.name] = !features[action.name];
                                    features['All'] = false;
                                    var anySpecific = Object.keys(features).some(function(k) {
                                        return k !== 'All' && features[k];
                                    });
                                    if (!anySpecific) {
                                        /* Auto-restore 'All' when nothing specific is left. */
                                        features['All'] = true;
                                    }
                                }
                                return {roles: roles, features: features, rejected: rejected};
                            };
                            /* __WF_CHIP_STATE_END__ */
                            window.__wfReadChipState = function(bar) {
                                var state = {roles: {}, features: {}};
                                bar.querySelectorAll('[data-filter-group="role"]').forEach(function(c) {
                                    state.roles[c.dataset.filter] = c.classList.contains('chip-active');
                                });
                                bar.querySelectorAll('[data-filter-group="feature"]').forEach(function(c) {
                                    state.features[c.dataset.filter] = c.classList.contains('chip-active');
                                });
                                return state;
                            };
                            window.__wfApplyChipState = function(bar, state) {
                                bar.querySelectorAll('[data-filter-group="role"]').forEach(function(c) {
                                    window.__setWorkflowChipActive(c, !!state.roles[c.dataset.filter]);
                                });
                                bar.querySelectorAll('[data-filter-group="feature"]').forEach(function(c) {
                                    window.__setWorkflowChipActive(c, !!state.features[c.dataset.filter]);
                                });
                            };
                            if (!window.__wfChipHandlerAttached) {
                                window.__wfChipHandlerAttached = true;
                                document.addEventListener('click', function(e) {
                                    var chip = e.target.closest('.filter-chip');
                                    var reset = e.target.closest('[data-wf-action="reset-filters"]');
                                    if (!chip && !reset) return;
                                    var root = (chip || reset).closest('#wf-filter-chips');
                                    if (!root) return;
                                    var bar = root.querySelector('#wf-filter-bar');
                                    if (!bar) return;
                                    var action = reset
                                        ? {type: 'reset'}
                                        : {type: 'toggle', group: chip.dataset.filterGroup, name: chip.dataset.filter};
                                    var next = window.__wfComputeChipState(
                                        window.__wfReadChipState(bar), action
                                    );
                                    if (next.rejected) {
                                        var roleGroup = chip.closest('.filter-group');
                                        if (roleGroup) {
                                            roleGroup.classList.remove('filter-group-attention');
                                            void roleGroup.offsetWidth;
                                            roleGroup.classList.add('filter-group-attention');
                                            window.setTimeout(function() {
                                                roleGroup.classList.remove('filter-group-attention');
                                            }, 900);
                                        }
                                        return;
                                    }
                                    window.__wfApplyChipState(bar, next);
                                    window.__updateWorkflowFilterQuery(root);
                                    window.__syncWorkflowFilters(bar);
                                });
                            }

                            /* Detail tab click handler (delegated, survives detail HTML replacement) */
                            if (!window.__dpTabHandlerAttached) {
                                window.__dpTabHandlerAttached = true;
                                document.addEventListener('click', function(e) {
                                    var tab = e.target.closest('.dp-tab');
                                    if (!tab) return;
                                    var panel = tab.closest('.dp-panel');
                                    if (!panel) return;
                                    panel.querySelectorAll('.dp-tab').forEach(function(x) {
                                        x.classList.remove('dp-tab-active');
                                    });
                                    panel.querySelectorAll('.dp-tab-content').forEach(function(x) {
                                        x.classList.remove('dp-tab-visible');
                                    });
                                    tab.classList.add('dp-tab-active');
                                    var content = panel.querySelector(
                                        '[data-tab-content="' + tab.dataset.tab + '"]'
                                    );
                                    if (content) content.classList.add('dp-tab-visible');
                                });
                            }

                            /* Card click handler */
                            function selectCard(card) {
                                if (!card) return;
                                element.querySelectorAll('.wf-card').forEach(function(c) {
                                    c.classList.remove('wf-active');
                                });
                                card.classList.add('wf-active');
                                var idx = card.dataset.stepIdx;
                                /* URL deep linking */
                                if (idx != null) {
                                    window.__wfSelectedStep = idx;
                                    history.replaceState(null, '', '#step-' + idx);
                                }
                                var storeEl = document.querySelector('#wf-detail-store [data-b64]');
                                var target = document.getElementById('wf-detail-content');
                                if (!storeEl || !target) return;
                                try {
                                    var details = JSON.parse(atob(storeEl.dataset.b64));
                                    if (details[idx] != null) {
                                        target.innerHTML = details[idx];
                                        var detailPanel = target.closest('.detail-panel');
                                        if (detailPanel) detailPanel.scrollTop = 0;
                                    }
                                } catch(ex) { console.error('wf-click:', ex); }
                            }
                            element.addEventListener('click', function(e) {
                                selectCard(e.target.closest('.wf-card'));
                            });

                            /* Keyboard navigation: j/k for next/prev step */
                            document.addEventListener('keydown', function(e) {
                                var tag = (e.target.tagName || '').toLowerCase();
                                if (tag === 'input' || tag === 'textarea' || e.target.isContentEditable) return;
                                if (e.key !== 'j' && e.key !== 'k') return;
                                var cards = Array.from(element.querySelectorAll('.wf-card'));
                                if (!cards.length) return;
                                var activeIdx = cards.findIndex(function(c) { return c.classList.contains('wf-active'); });
                                var nextIdx;
                                if (e.key === 'j') {
                                    nextIdx = activeIdx < 0 ? 0 : Math.min(activeIdx + 1, cards.length - 1);
                                } else {
                                    nextIdx = activeIdx < 0 ? 0 : Math.max(activeIdx - 1, 0);
                                }
                                cards[nextIdx].scrollIntoView({behavior:'smooth', block:'center'});
                                selectCard(cards[nextIdx]);
                            });

                            /* Deep link: on load, scroll to step from URL hash */
                            setTimeout(function() {
                                var hash = window.location.hash;
                                var m = hash && hash.match(/^#step-(\\d+)$/);
                                if (m) {
                                    var card = document.getElementById('wf-card-' + m[1]);
                                    if (card) {
                                        card.scrollIntoView({behavior:'smooth', block:'center'});
                                        selectCard(card);
                                    } else {
                                        /* Stale hash from an earlier session or another
                                           trajectory: drop it rather than guess. */
                                        history.replaceState(
                                            null, '',
                                            window.location.pathname + window.location.search
                                        );
                                    }
                                }
                            }, 500);

                            /* Hidden-selection watcher: when a re-render (filter,
                               search, or label upload) removes the selected step's
                               card, tell the user in the detail panel; when the
                               card comes back, restore its detail. */
                            if (!window.__wfHiddenStepObserverAttached) {
                                window.__wfHiddenStepObserverAttached = true;
                                var hiddenStepCheckPending = null;
                                var checkSelectedStepVisible = function() {
                                    hiddenStepCheckPending = null;
                                    var target = document.getElementById('wf-detail-content');
                                    if (!target) return;
                                    if (target.querySelector('[data-wf-detail-placeholder]')) {
                                        /* The app reset the detail panel (new trajectory
                                           or label upload): the old selection is gone. */
                                        window.__wfSelectedStep = null;
                                        return;
                                    }
                                    var idx = window.__wfSelectedStep;
                                    if (idx == null) return;
                                    var card = document.getElementById('wf-card-' + idx);
                                    if (card) {
                                        if (target.querySelector('[data-wf-hidden-msg]')) {
                                            selectCard(card);
                                        }
                                        return;
                                    }
                                    if (!document.querySelector('.wf-card')) return;
                                    if (target.querySelector('[data-wf-hidden-msg]')) return;
                                    target.innerHTML =
                                        "<div data-wf-hidden-msg='1'" +
                                        " style='padding:2em 1em;text-align:center;color:var(--ov-muted);'>" +
                                        "<p style='font-size:15px;margin-bottom:0.5em;'>" +
                                        "Selected step is hidden by the current filters</p>" +
                                        "<p style='font-size:12px;'>Adjust the filters to show it again.</p>" +
                                        "</div>";
                                };
                                new MutationObserver(function() {
                                    if (hiddenStepCheckPending) return;
                                    hiddenStepCheckPending = window.setTimeout(checkSelectedStepVisible, 120);
                                }).observe(document.body, {childList: true, subtree: true});
                            }

                            /* Auto-select first assistant card if no hash link */
                            setTimeout(function() {
                                if (window.location.hash) return;
                                var cards = element.querySelectorAll('.wf-card');
                                for (var i = 0; i < cards.length; i++) {
                                    var badges = cards[i].querySelectorAll('.wf-badge');
                                    for (var j = 0; j < badges.length; j++) {
                                        if (badges[j].textContent.trim() === 'Assistant') {
                                            selectCard(cards[i]);
                                            return;
                                        }
                                    }
                                }
                                if (cards.length) selectCard(cards[0]);
                            }, 600);
                            """,
                        )
                    with gr.Column(scale=2, min_width=300, elem_classes=["detail-panel"]):
                        detail_html = gr.HTML(
                            _DETAIL_PLACEHOLDER,
                            elem_id="wf-detail-panel",
                        )
                detail_store = gr.HTML("", elem_id="wf-detail-store")

            # ===== Raw Data Tab =====
            with gr.TabItem("Raw Data"):
                raw_json = gr.Code(
                    label="Full trajectory JSON",
                    language="json",
                    value="",
                    max_lines=50,
                )

        # -- Callbacks --

        overview_sections = (
            performance_section,
            efficiency_section,
            tools_section,
            agents_section,
            diagnostics_section,
            deep_dive_section,
            labels_section,
        )

        def show_overview_section(selected):
            return tuple(gr.update(visible=name == selected) for name in overview_section_names)

        overview_section.change(
            fn=show_overview_section,
            inputs=[overview_section],
            outputs=list(overview_sections),
        )

        _empty_fig = go.Figure()
        _empty_fig.update_layout(
            template="plotly_white", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )

        def _empty_result(banner="", detail="*No data*"):
            """Return the empty outputs tuple for error states."""
            f = _empty_fig
            # Surface the detail message (it previously vanished: every load
            # error's specifics were dropped and only the generic banner shown).
            if detail and detail != "*No data*":
                import html as _html

                banner += (
                    "<p style='color:var(--ov-muted);font-size:13px;margin:4px 0 0;'>"
                    f"{_html.escape(detail.strip('*'))}</p>"
                )
            return (
                gr.update(visible=False),  # main_tabs
                gr.update(visible=bool(banner)),  # summary_area (reveal so the error banner shows)
                gr.update(),  # upload_accordion (no change)
                [],  # state_steps
                banner,  # summary_banner
                "",  # anomaly_strip_html
                "",  # overview_kpi_html
                "",  # session_detail_html
                "",  # metrics_md
                f,
                f,  # token_chart, duration_chart
                f,  # context_growth_chart
                "",  # behavior_md
                f,  # tool_chart
                f,  # tool_outcome_chart
                "",  # agent_summary_html
                f,  # agent_token_chart
                f,  # agent_swimlane_chart
                "",  # diag_summary_html
                "",  # diag_pressure_html
                gr.update(
                    choices=[("All agents", PRESSURE_ALL_AGENTS)],
                    value=PRESSURE_ALL_AGENTS,
                    visible=False,
                ),  # diag_pressure_agent
                f,  # diag_pressure_chart
                f,  # diag_file_chart
                "",  # diag_rootcause_html
                # New diagnostic charts
                f,  # error_class_chart
                f,  # plan_timeline_chart
                "",  # hotspots_md
                "",  # per_message_md
                render_filter_chips(),  # wf_filter_chips_html
                ",".join(_FILTER_CHIPS_DEFAULT),  # wf_filter_hidden
                "",  # wf_count_html
                "",  # toc_html
                "<div></div>",  # workflow_html
                "",  # detail_store
                _DETAIL_PLACEHOLDER,  # detail_html
                "",  # raw_json
                # Patterns
                "",  # patterns_tool_html
                "",  # patterns_failure_html
                "",  # antipattern_summary_html
                {},  # state_raw
            )

        def _do_load_inner(upload_obj, dark=False, selected_format=""):
            """Load trajectory from uploaded file."""

            file_path = None
            if upload_obj is not None:
                file_path = upload_obj if isinstance(upload_obj, str) else upload_obj.name

            if not file_path or not os.path.isfile(file_path):
                return _empty_result(detail="*No file selected or file not found.*")

            format_hint = selected_format or None
            raw = load_trajectory(file_path, format_hint=format_hint)
            if raw.get("_error_code") == "mismatch":
                selected_key = raw.get("_selected") or selected_format
                detected_key = raw.get("_detected") or ""
                err_msg = (
                    f"Format mismatch: selected "
                    f"<b>{html.escape(FORMAT_LABELS.get(selected_key, selected_key))}</b>"
                    f" but file detected as "
                    f"<b>{html.escape(FORMAT_LABELS.get(detected_key, detected_key))}</b>."
                )
                return _empty_result(
                    banner=f"<p style='color:#dc2626;'>{err_msg}</p>",
                    detail="*Please select the correct format and try again.*",
                )
            if "_error" in raw:
                err_banner = f"<p style='color:#dc2626;'>Error: {html.escape(raw['_error'])}</p>"
                return _empty_result(banner=err_banner, detail="*Error loading file.*")

            detected = detect_format(raw)
            gate = check_format_selection(detected, selected_format)
            if gate == "unknown":
                err_msg = (
                    "Could not detect trajectory format. "
                    "Select a format from the dropdown and try again."
                )
                return _empty_result(
                    banner=f"<p style='color:#dc2626;'>{html.escape(err_msg)}</p>",
                    detail="*Unrecognized trajectory file.*",
                )
            if gate == "mismatch":
                err_msg = (
                    f"Format mismatch: selected "
                    f"<b>{html.escape(FORMAT_LABELS.get(selected_format, selected_format))}</b>"
                    f" but file detected as "
                    f"<b>{html.escape(FORMAT_LABELS.get(detected, detected))}</b>."
                )
                return _empty_result(
                    banner=f"<p style='color:#dc2626;'>{err_msg}</p>",
                    detail="*Please select the correct format and try again.*",
                )

            steps = parse_steps(raw)
            steps_total = len(steps)
            load_warnings = ""
            if steps_total > _MAX_STEPS:
                steps = steps[:_MAX_STEPS]
                load_warnings = (
                    f"<p style='color:#d97706;font-size:13px;margin:0 0 4px;'>"
                    f"&#9888; Showing first {_MAX_STEPS:,} of {steps_total:,} steps "
                    f"({steps_total - _MAX_STEPS:,} truncated).</p>"
                )
            token_warnings = validate_token_integrity(steps)
            for tw in token_warnings:
                load_warnings += (
                    f"<p style='color:#d97706;font-size:13px;margin:0 0 4px;'>&#9888; {html.escape(tw)}</p>"
                )
            message_rows = build_message_metrics(steps)
            metrics = compute_metrics(steps, raw, message_rows=message_rows)
            _, wfmt = wall_clock_fmt(metrics)

            step_analytics = compute_step_analytics(steps)
            verdicts = compute_health_verdict(metrics, step_analytics if steps else [])
            agent_summaries = compute_agent_summary(steps, raw)

            # Delegate to focused builders
            dg = _build_diagnostics_outputs(
                steps, step_analytics, agent_summaries, dark=dark, trajectory_format=detected, raw=raw
            )
            ov = _build_overview_outputs(
                steps, raw, metrics, message_rows, verdicts, wfmt, file_path, trajectory_format=detected
            )

            def _d(k):
                return raw.get(k, {}) if isinstance(raw.get(k), dict) else {}

            _md, _timing = _d("metadata"), _d("timing")
            _model_id, _, _agent_id = extract_agent_info(steps)
            session_detail = _build_session_detail_html(
                _timing,
                _md,
                model_id=_model_id,
                agent_id=_agent_id,
            )

            ch = _build_chart_outputs(steps, message_rows, agent_summaries, dark=dark, trajectory_format=detected)
            wf = _build_workflow_outputs(steps)

            # Pattern detection
            tool_seqs = detect_tool_sequences(steps)
            fail_pats = detect_failure_patterns(steps)
            pat_tool_html = _render_tool_sequences_html(tool_seqs)
            pat_fail_html = _render_failure_patterns_html(fail_pats)

            # Raw data
            raw_str = json.dumps(raw, indent=2, ensure_ascii=False, default=str)
            if len(raw_str) > 500_000:
                raw_str = raw_str[:500_000] + "\n\n... (truncated at 500KB)"

            return (
                gr.update(visible=True),  # main_tabs
                gr.update(visible=True),  # summary_area
                gr.update(),  # upload_accordion (keep visible)
                steps,  # state_steps
                load_warnings + ov["banner"],  # summary_banner
                ov["anomaly_html"],  # anomaly_strip_html
                ov["kpi_html"],  # overview_kpi_html
                session_detail,  # session_detail_html
                ov["metrics_text"],  # metrics_md
                ch["tok_fig"],  # token_chart
                ch["dur_fig"],  # duration_chart
                ch["context_growth_fig"],  # context_growth_chart
                ov["behavior_text"],  # behavior_md
                ch["tl_fig"],  # tool_chart
                ch["tool_outcome_fig"],  # tool_outcome_chart
                ch["agent_cards_html"],  # agent_summary_html
                ch["agent_tok_fig"],  # agent_token_chart
                ch["swimlane_fig"],  # agent_swimlane_chart
                dg["diag_summary_html"],  # diag_summary_html
                dg["diag_pressure_html"],  # diag_pressure_html
                gr.update(
                    choices=dg["diag_pressure_dropdown"]["choices"],
                    value=dg["diag_pressure_dropdown"]["value"],
                    visible=dg["diag_pressure_dropdown"]["visible"],
                ),  # diag_pressure_agent
                dg["diag_pressure_chart"],  # diag_pressure_chart
                dg["diag_file_chart"],  # diag_file_chart
                dg["diag_rootcause_html"],  # diag_rootcause_html
                # New diagnostic charts
                ch["error_class_fig"],  # error_class_chart
                ch["plan_timeline_fig"],  # plan_timeline_chart
                ov["hotspots_text"],  # hotspots_md
                ov["per_message_text"],  # per_message_md
                wf["wf_chips"],  # wf_filter_chips_html
                wf["wf_filter_val"],  # wf_filter_hidden
                wf["wf_count"],  # wf_count_html
                wf["toc_html_val"],  # toc_html
                wf["wf_html"],  # workflow_html
                wf["detail_store_val"],  # detail_store
                _DETAIL_PLACEHOLDER,  # detail_html
                raw_str,  # raw_json
                # Patterns
                pat_tool_html,  # patterns_tool_html
                pat_fail_html,  # patterns_failure_html
                ch["antipattern_html"],  # antipattern_summary_html
                raw,  # state_raw
            )

        def do_load(upload_obj, dark=False, selected_format=""):
            """Load wrapper: surface any unexpected failure as a visible banner
            instead of a raw traceback that leaves the UI stale."""
            try:
                return _do_load_inner(upload_obj, dark, selected_format)
            except Exception as exc:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                return _empty_result(
                    banner=(f"<p style='color:#dc2626;'>Error loading trajectory: {html.escape(str(exc))}</p>"),
                    detail="*Failed to load — see console for details.*",
                )

        all_outputs = [
            main_tabs,
            summary_area,
            upload_accordion,
            state_steps,
            summary_banner,
            anomaly_strip_html,
            overview_kpi_html,
            session_detail_html,
            metrics_md,
            token_chart,
            duration_chart,
            context_growth_chart,
            behavior_md,
            tool_chart,
            tool_outcome_chart,
            agent_summary_html,
            agent_token_chart,
            agent_swimlane_chart,
            diag_summary_html,
            diag_pressure_html,
            diag_pressure_agent,
            diag_pressure_chart,
            diag_file_chart,
            diag_rootcause_html,
            # New diagnostic outputs
            error_class_chart,
            plan_timeline_chart,
            hotspots_md,
            per_message_md,
            wf_filter_chips_html,
            wf_filter_hidden,
            wf_count_html,
            toc_html,
            workflow_html,
            detail_store,
            detail_html,
            raw_json,
            # Patterns
            patterns_tool_html,
            patterns_failure_html,
            antipattern_summary_html,
            state_raw,
        ]

        _load_ev = load_btn.click(
            fn=do_load,
            inputs=[file_upload, state_dark, format_selector],
            outputs=all_outputs,
        )
        # Auto-load when file is uploaded (no separate click needed)
        _upload_ev = file_upload.change(
            fn=do_load,
            inputs=[file_upload, state_dark, format_selector],
            outputs=all_outputs,
        )
        for _ev in (_load_ev, _upload_ev):
            _ev.success(
                fn=_prepare_html_export,
                inputs=[state_raw, state_steps, state_dark],
                outputs=export_btn,
                show_progress="minimal",
                show_progress_on=export_btn,
            )

        def on_pressure_agent_change(agent_key, steps, raw, dark):
            """Rebuild the pressure chart for the selected agent/subagent."""
            if not steps:
                empty = go.Figure()
                empty.update_layout(
                    template="plotly_white",
                    height=380,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                return empty, ""
            key = agent_key or PRESSURE_ALL_AGENTS
            fig = build_context_pressure_chart(
                steps,
                agent_key=key,
                raw=raw if isinstance(raw, dict) else None,
                dark=bool(dark),
            )
            html_strip = format_context_pressure_html(
                context_pressure_series(
                    steps,
                    agent_key=key,
                    raw=raw if isinstance(raw, dict) else None,
                )
            )
            return fig, html_strip

        diag_pressure_agent.change(
            fn=on_pressure_agent_change,
            inputs=[diag_pressure_agent, state_steps, state_raw, state_dark],
            outputs=[diag_pressure_chart, diag_pressure_html],
        )

        # -- Run-group scorecard --
        # Overview trajectory is the baseline; uploaded files are comparison runs.
        def on_run_group_scorecard(files, format_hint, dark, overview_raw):
            """Build an N-run scorecard; Overview trajectory is the baseline."""
            empty_fig = go.Figure()
            empty_fig.update_layout(
                template="plotly_white",
                height=200,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            paths = normalize_run_paths(files)
            hint = format_hint or None
            if hint == "":
                hint = None
            baseline = overview_raw if isinstance(overview_raw, dict) and overview_raw else None
            result = build_run_group_scorecard(
                paths,
                format_hint=hint,
                baseline_raw=baseline,
            )
            scorecard = build_run_group_scorecard_html(
                result,
                include_behavior=False,
            )
            timeline_runs = result.get("timeline_runs") or []
            if result.get("ok") and timeline_runs:
                fig = build_run_group_agent_timeline(
                    timeline_runs,
                    dark=bool(dark),
                )
                chart_update = gr.update(value=fig, visible=True)
            else:
                chart_update = gr.update(value=empty_fig, visible=False)
            behavior = build_run_group_behavior_html(result) if result.get("ok") else ""
            return scorecard, chart_update, behavior

        rg_run_btn.click(
            fn=on_run_group_scorecard,
            inputs=[rg_file_upload, rg_format_selector, state_dark, state_raw],
            outputs=[rg_scorecard_html, rg_agent_timeline_chart, rg_behavior_html],
        )

        def on_run_comparison(ref_file, anchor_file, ref_format, ref_labels_file, cmp_labels_file, overview_raw, dark):
            # Pairwise: the file uploaded here is the reference/baseline;
            # the Overview trajectory is the compared run.
            empty_fig = go.Figure()
            empty_fig.update_layout(
                template="plotly_white", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )

            if not overview_raw:
                return (
                    "<div style='color:var(--ov-warn);padding:1em;text-align:center;'>"
                    "Load a trajectory in the Overview tab first — it will be the compared trajectory.</div>",
                    empty_fig,
                    empty_fig,
                    "<div style='padding:2em;color:var(--ov-muted);text-align:center;'>"
                    "Load a trajectory in the Overview tab first.</div>",
                )

            if ref_file is None:
                return (
                    "<div style='color:var(--ov-warn);padding:1em;text-align:center;'>"
                    "Upload a reference trajectory first.</div>",
                    empty_fig,
                    empty_fig,
                    "<div style='padding:2em;color:var(--ov-muted);text-align:center;'>"
                    "Upload a reference trajectory.</div>",
                )

            ref_path = ref_file if isinstance(ref_file, str) else ref_file.name
            anchor_path = None
            if anchor_file is not None:
                anchor_path = anchor_file if isinstance(anchor_file, str) else anchor_file.name

            from .loaders import load_trajectory as _load_traj

            ref_raw = _load_traj(ref_path, format_hint=ref_format or None)

            result = run_comparison(
                ref_raw=ref_raw,
                cmp_raw=overview_raw,
                anchor_path=anchor_path,
                token_rate=50.0,
                fuzzy=False,
                dark=bool(dark),
            )

            # Per-phase comparison charts — only populated when BOTH trajectories
            # have a labeled JSON uploaded. Compared labels come from the
            # Overview tab's label upload; reference labels come from this tab.
            phase_count_fig = empty_fig
            phase_duration_fig = empty_fig
            ref_labels_path = (
                ref_labels_file
                if isinstance(ref_labels_file, str)
                else (ref_labels_file.name if ref_labels_file else None)
            )
            cmp_labels_path = (
                cmp_labels_file
                if isinstance(cmp_labels_file, str)
                else (cmp_labels_file.name if cmp_labels_file else None)
            )
            if ref_labels_path and cmp_labels_path:
                try:
                    from .labels import load_labeled_json, aggregate_labels
                    from .charts import (
                        build_phase_count_comparison_chart,
                        build_phase_duration_comparison_chart,
                    )

                    ref_agg = aggregate_labels(load_labeled_json(ref_labels_path))
                    cmp_agg = aggregate_labels(load_labeled_json(cmp_labels_path))
                    ref_label_name = os.path.basename(ref_path) if ref_path else "reference"
                    cmp_label_name = (
                        os.path.basename(overview_raw.get("_source_path", ""))
                        if isinstance(overview_raw, dict)
                        else "compared"
                    ) or "compared"
                    phase_count_fig = build_phase_count_comparison_chart(
                        ref_agg["phase_counts"],
                        cmp_agg["phase_counts"],
                        ref_label=ref_label_name,
                        cmp_label=cmp_label_name,
                        dark=bool(dark),
                    )
                    phase_duration_fig = build_phase_duration_comparison_chart(
                        ref_agg["phase_durations"],
                        cmp_agg["phase_durations"],
                        ref_label=ref_label_name,
                        cmp_label=cmp_label_name,
                        dark=bool(dark),
                    )
                except Exception as exc:
                    import logging

                    logging.getLogger(__name__).debug("Phase comparison chart build failed: %s", exc)

            # Branch on the pipeline's explicit ok flag (backward-compatible:
            # a result without the flag reads as success, today's behavior).
            if result.get("ok", True):
                status = "<div style='color:var(--ov-success);padding:0.5em;font-size:13px;'>Comparison complete.</div>"
            else:
                status = (
                    "<div style='color:var(--ov-warn);padding:0.5em;font-size:13px;'>"
                    "Comparison failed &mdash; see the report panel for details.</div>"
                )

            return (result["report_html"], phase_count_fig, phase_duration_fig, status)

        cmp_run_btn.click(
            fn=on_run_comparison,
            inputs=[
                cmp_file_upload,
                cmp_anchor_upload,
                cmp_format_selector,
                cmp_ref_labels_upload,
                label_file_upload,
                state_raw,
                state_dark,
            ],
            outputs=[cmp_report_html, cmp_phase_count_chart, cmp_phase_duration_chart, cmp_status_html],
        )

        # -- Attribution callback (DECAF) --
        # Self-contained (reads state_raw), so it never touches the Overview load
        # tuple. Diagnoses the DISPLAYED trajectory (via source_path + fmt), and
        # derives (agent, instance_id) from the source path for corpus files; the
        # override fields cover uploaded temp paths.
        def on_diagnose(overview_raw, agent_override, inst_override, root_override):
            from dataclasses import asdict
            from .rendering import build_attribution_html
            from . import attribution as _attr

            if not overview_raw:
                return (
                    build_attribution_html(
                        {"available": False, "reason": "Load a trajectory in the Overview tab first."}
                    ),
                    "<div style='color:var(--ov-warn);padding:0.5em;font-size:13px;'>No trajectory loaded.</div>",
                )

            src = overview_raw.get("_source_path", "") if isinstance(overview_raw, dict) else ""
            src_sha = overview_raw.get("_source_sha256") if isinstance(overview_raw, dict) else None
            agent = (agent_override or "").strip()
            inst = (inst_override or "").strip()
            root = (root_override or "").strip()
            if src:
                if not inst:
                    inst = os.path.splitext(os.path.basename(src))[0]
                if not agent:
                    # Only a real corpus layout names the agent dir; a Gradio
                    # upload's parent is a cache hash and won't be in AGENTS —
                    # diagnose() validates and asks for the override.
                    agent = os.path.basename(os.path.dirname(src))

            fmt = None
            if isinstance(overview_raw, dict):
                detected = detect_format(overview_raw)
                fmt = None if detected == "unknown" else detected

            result = _attr.diagnose(
                agent=agent or None,
                instance_id=inst or None,
                source_path=src or None,
                fmt=fmt or None,
                expected_sha=src_sha,
                argus_root=root or None,
            )
            html_out = build_attribution_html(asdict(result))
            import html as _html

            ident = _html.escape(f"{result.agent}/{result.instance_id}")
            status = (
                "<div style='color:var(--ov-success);padding:0.5em;font-size:13px;'>"
                f"Diagnosis complete &mdash; {ident}.</div>"
                if result.available
                else "<div style='color:var(--ov-muted);padding:0.5em;font-size:13px;'>"
                "No gold-grounded attribution &mdash; see the note below.</div>"
            )
            return html_out, status

        # Manual button: honors the override fields for the current file.
        # concurrency_id serializes this with the autoload below (same queue), so
        # a manual diagnosis and an autoload can never interleave/overwrite.
        attr_run_btn.click(
            fn=on_diagnose,
            inputs=[state_raw, attr_agent_override, attr_inst_override, attr_root_override],
            outputs=[attr_result_html, attr_status_html],
            concurrency_id="attribution",
        )

        # Clear stale attribution IMMEDIATELY when a new load starts (fast
        # handler in parallel with do_load), so the previous run's diagnosis is
        # never shown against a newly loaded trajectory even transiently.
        def _clear_attribution():
            return (
                "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>"
                "Diagnosing the loaded trajectory&hellip;</div>",
                "",
            )

        # Same concurrency queue as the diagnose callbacks: total ordering means
        # an in-flight (old) diagnosis always completes BEFORE this clear runs,
        # so a stale result can never overwrite the new-load placeholder.
        load_btn.click(
            fn=_clear_attribution, outputs=[attr_result_html, attr_status_html], concurrency_id="attribution"
        )
        file_upload.change(
            fn=_clear_attribution, outputs=[attr_result_html, attr_status_html], concurrency_id="attribution"
        )

        # Auto-populate on load, and IGNORE + clear the agent/instance override
        # fields so a previous case's identity can never attribute a newly loaded
        # one. The corpus root is sticky (it selects an environment, not a case).
        def on_diagnose_autoload(overview_raw, root_override):
            html_out, status = on_diagnose(overview_raw, "", "", root_override)
            return html_out, status, "", ""  # last two clear the override fields

        for _ev in (_load_ev, _upload_ev):
            _ev.then(
                fn=on_diagnose_autoload,
                inputs=[state_raw, attr_root_override],
                outputs=[attr_result_html, attr_status_html, attr_agent_override, attr_inst_override],
                concurrency_id="attribution",
            )

        # -- Workflow filter callback --
        def do_filter_workflow(steps, filter_csv, keyword, current_toc):
            """Re-render Workflow cards, count, and TOC with filters applied."""
            return _build_filtered_workflow_outputs(steps, filter_csv, keyword, current_toc)

        # -- TOC toggle callback --
        def on_toc_toggle(current_toc):
            """Toggle TOC sidebar visibility via CSS class."""
            if not current_toc:
                return current_toc
            if "toc-hidden" in current_toc:
                return current_toc.replace("toc-hidden", "").strip()
            return current_toc.replace("wf-toc-sidebar", "wf-toc-sidebar toc-hidden")

        wf_toc_toggle.click(
            fn=on_toc_toggle,
            inputs=[toc_html],
            outputs=[toc_html],
        )

        wf_filter_hidden.change(
            fn=do_filter_workflow,
            inputs=[state_steps, wf_filter_hidden, wf_search, toc_html],
            outputs=[workflow_html, wf_count_html, toc_html],
        )
        wf_search.change(
            fn=do_filter_workflow,
            inputs=[state_steps, wf_filter_hidden, wf_search, toc_html],
            outputs=[workflow_html, wf_count_html, toc_html],
        )

        # -- Label file upload callback --
        _empty_label_fig = go.Figure()
        _empty_label_fig.update_layout(
            template="plotly_white", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )

        def do_load_labels(upload_obj, dark=False):
            """Load labeled JSON and update all label UI components.

            Returns values matching the output list below:
              0: label_badge_html
              1: label_status_html
              2: label_charts_row1 (Row visibility)
              3: label_phase_count_chart (Plot)
              4: label_action_count_chart (Plot)
              5: label_charts_row2 (Row visibility)
              6: label_phase_dur_chart (Plot)
              7: label_action_dur_chart (Plot)
              8: label_timeline_row (Row visibility)
              9: label_timeline_chart (Plot)
            """
            file_path = None
            if upload_obj is not None:
                file_path = upload_obj if isinstance(upload_obj, str) else upload_obj.name

            empty = _empty_label_fig

            if not file_path or not os.path.isfile(file_path):
                return (
                    "",  # label_badge_html
                    "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>"
                    "Upload a <code>*_labeled.json</code> file to view label distributions and timeline.</div>",
                    gr.update(visible=False),
                    empty,
                    empty,
                    gr.update(visible=False),
                    empty,
                    empty,
                    gr.update(visible=False),
                    empty,
                )

            try:
                payload = build_label_ui_payload(file_path, dark=bool(dark))
            except Exception as exc:
                return (
                    "",  # label_badge_html
                    f"<div style='padding:1em;color:#dc2626;text-align:center;'>Error: {html.escape(str(exc))}</div>",
                    gr.update(visible=False),
                    empty,
                    empty,
                    gr.update(visible=False),
                    empty,
                    empty,
                    gr.update(visible=False),
                    empty,
                )

            return (
                payload["badge_html"],  # label_badge_html
                payload["status_html"],
                gr.update(visible=True),
                payload["phase_count_fig"],
                payload["action_count_fig"],
                gr.update(visible=True),
                payload["phase_duration_fig"],
                payload["action_duration_fig"],
                gr.update(visible=True),
                payload["timeline_fig"],
            )

        label_outputs = [
            label_badge_html,
            label_status_html,
            label_charts_row1,
            label_phase_count_chart,
            label_action_count_chart,
            label_charts_row2,
            label_phase_dur_chart,
            label_action_dur_chart,
            label_timeline_row,
            label_timeline_chart,
        ]
        label_inputs = [label_file_upload, state_dark]
        label_load_btn.click(
            fn=do_load_labels,
            inputs=label_inputs,
            outputs=label_outputs,
        )
        # Auto-load labels on file upload
        label_file_upload.change(
            fn=do_load_labels,
            inputs=label_inputs,
            outputs=label_outputs,
        )

        # Reset label UI whenever a new trajectory is loaded. Without this, stale
        # label state from a prior *_labeled.json upload would render data for the
        # wrong trajectory.
        def _reset_labels():
            empty = _empty_label_fig
            return (
                "",  # label_badge_html
                (
                    "<div style='padding:1em;color:var(--ov-muted);text-align:center;'>"
                    "Upload a <code>*_labeled.json</code> file to view label distributions and timeline.</div>"
                ),  # label_status_html
                gr.update(visible=False),
                empty,
                empty,  # row1 + two charts
                gr.update(visible=False),
                empty,
                empty,  # row2 + two charts
                gr.update(visible=False),
                empty,  # timeline row + chart
                gr.update(value=None),  # clear the file picker
            )

        _reset_outputs = label_outputs + [label_file_upload]
        load_btn.click(fn=_reset_labels, inputs=None, outputs=_reset_outputs)
        file_upload.change(fn=_reset_labels, inputs=None, outputs=_reset_outputs)

        # Detect browser dark mode at page load and store in state_dark.
        app.load(
            fn=lambda dark: dark,
            inputs=[state_dark],
            outputs=[state_dark],
            js="() => [window.matchMedia('(prefers-color-scheme: dark)').matches]",
        )

    return app
