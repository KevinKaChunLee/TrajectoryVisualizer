"""HTML and Plotly presenters for a LoadedSession. No Gradio imports."""

from __future__ import annotations

import base64
import html
import json
import os

import plotly.graph_objects as go

from .charts import (
    build_agent_swimlane_chart,
    build_agent_token_chart,
    build_context_growth_chart,
    build_context_pressure_chart,
    build_duration_chart,
    build_error_classification_chart,
    build_file_interaction_chart,
    build_label_action_count_chart,
    build_label_action_duration_chart,
    build_label_phase_count_chart,
    build_label_phase_duration_chart,
    build_label_timeline_chart,
    build_plan_timeline_chart,
    build_token_chart,
    build_tool_chart,
    build_tool_outcome_timeline,
)
from .diagnostics import PRESSURE_ALL_AGENTS
from .formatting import (
    format_banner_html,
    format_behavioral_md,
    format_context_pressure_html,
    format_performance_md,
    _build_hotspots_md,
    _build_per_message_md,
)
from .help import HELP_TEXT
from .labels import aggregate_labels, load_labeled_json
from .loaders import FORMAT_LABELS
from .metrics import compute_diagnostic_metrics, extract_agent_info
from .palette import LABEL_PHASE_COLORS
from .rendering import (
    _diag_jump_onclick,
    build_antipattern_summary_html,
    build_root_cause_html,
    format_step_detail,
    render_agent_summary_cards,
    render_filter_chips,
    render_toc_sidebar,
    render_workflow_html,
)
from .session import MAX_STEPS, LoadedSession

DETAIL_PLACEHOLDER = (
    "<div id='wf-detail-content'>"
    "<div data-wf-detail-placeholder='1' style='padding:2em 1em;text-align:center;color:var(--ov-muted);'>"
    "<p style='font-size:15px;margin-bottom:0.5em;'>Select a step to inspect</p>"
    "<p style='font-size:12px;'>Click any card on the left, or press <kbd>j</kbd>/<kbd>k</kbd> to navigate</p>"
    "</div></div>"
)
ROLE_FILTERS = ["Assistant", "User"]
FEATURE_FILTERS = ["Tool Calls", "Errors", "Reasoning"]
ALL_FEATURE_FILTER = "All"
FILTER_CHIPS_DEFAULT = [*ROLE_FILTERS, ALL_FEATURE_FILTER]


def trajectory_format_label(fmt: str | None) -> str:
    """Return a human-readable trajectory format label."""
    return FORMAT_LABELS.get(fmt or "", fmt or "Unknown")


def _prerender_step_details(steps: list[dict]) -> str:
    """Pre-render all step details as HTML and return a base64-encoded JSON blob."""
    details = {}
    for step in steps:
        details[str(step["index"])] = format_step_detail(step)
    b64 = base64.b64encode(json.dumps(details).encode()).decode()
    return f'<div data-b64="{b64}" style="display:none"></div>'


def _workflow_step_labels(step: dict) -> set[str]:
    """Return every Workflow filter label that applies to *step*."""
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


def filter_workflow_steps(
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
    role_filters = active & set(ROLE_FILTERS)
    if not role_filters:
        return []
    feature_filters = active & set(FEATURE_FILTERS)
    restrict_features = ALL_FEATURE_FILTER not in active and bool(feature_filters)

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


def build_filtered_workflow_outputs(
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
    indices = filter_workflow_steps(steps, active_filters, keyword)
    filtered_steps = [steps[position] for position in indices]

    if not (set(active_filters) & set(ROLE_FILTERS)):
        workflow_html = (
            "<div style='padding:2em;color:var(--ov-muted);text-align:center;'>"
            "Select at least one role to see steps.</div>"
        )
    else:
        workflow_html = render_workflow_html(filtered_steps)

    count_html = f"<div class='wf-count'>Showing {len(filtered_steps)} of {len(steps)} steps</div>"
    collapsed = "toc-hidden" in (current_toc or "")
    return workflow_html, count_html, render_toc_sidebar(filtered_steps, collapsed=collapsed)


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
    """Build Session Details panel as a chip grid of session environment fields."""
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


def build_overview_kpi_html(
    metrics: dict, wall_fmt: str, verdicts: list[dict] | None = None, message_rows: list[dict] | None = None
) -> str:
    """Build at-a-glance KPI card strip for Overview tab.

    When *verdicts* is provided, matching KPI cards get a colored left border
    and a tooltip with the verdict detail string.
    """
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

    sparkline_data: dict[str, list[float]] = {}
    if message_rows:
        sparkline_data["Tokens"] = [r.get("tokens_total", 0) for r in message_rows]
        sparkline_data["Wall-Clock"] = [r.get("duration", 0) or 0 for r in message_rows]

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
        help_key = _label_to_help_key.get(label, "")
        help_attr = ""
        if help_key and help_key in HELP_TEXT:
            help_attr = f" data-help='{html.escape(HELP_TEXT[help_key])}'"
        sparkline = ""
        if label in sparkline_data:
            sparkline = _build_sparkline_svg(sparkline_data[label])
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


def empty_plotly_fig() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="plotly_white", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def build_overview_outputs(session: LoadedSession) -> dict:
    """Compute overview-related outputs: banner, KPI, metadata, and metrics markdown."""
    steps = session.steps
    raw = session.raw
    metrics = session.metrics
    message_rows = session.message_rows
    verdicts = session.verdicts
    wfmt = session.wall_clock
    file_path = session.path
    trajectory_format = session.format

    banner = format_banner_html(os.path.basename(file_path), metrics, wfmt, trajectory_format=trajectory_format)
    kpi_html = build_overview_kpi_html(metrics, wfmt, verdicts=verdicts, message_rows=message_rows)
    metrics_text = format_performance_md(metrics, wfmt)

    traj = raw.get("trajectory") or raw.get("messages") or []
    diag_metrics = compute_diagnostic_metrics(steps, traj) if traj else None
    behavior_text = format_behavioral_md(metrics, diag_metrics=diag_metrics)
    hotspots_text = _build_hotspots_md(message_rows)
    per_message_text = _build_per_message_md(message_rows)
    anomaly_html = _build_anomaly_strip_html(session.anomalies)

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

    return {
        "banner": banner,
        "anomaly_html": anomaly_html,
        "kpi_html": kpi_html,
        "session_detail": session_detail,
        "metrics_text": metrics_text,
        "behavior_text": behavior_text,
        "hotspots_text": hotspots_text,
        "per_message_text": per_message_text,
    }


def build_chart_outputs(session: LoadedSession, dark: bool = False) -> dict:
    """Build all chart figures and analytics markdown."""
    steps = session.steps
    message_rows = session.message_rows
    agent_summaries = session.agent_summaries
    trajectory_format = session.format

    tok_fig = build_token_chart(steps, dark=dark, format=trajectory_format)
    dur_fig = build_duration_chart(steps, dark=dark)
    tl_fig = build_tool_chart(steps, dark=dark)
    tool_outcome_fig = build_tool_outcome_timeline(steps, dark=dark)
    agent_tok_fig = build_agent_token_chart(agent_summaries, dark=dark)
    swimlane_fig = build_agent_swimlane_chart(steps, dark=dark)
    plan_timeline_fig = build_plan_timeline_chart(session.plan_history, session.plan_metrics, dark=dark)
    error_class_fig = build_error_classification_chart(steps, dark=dark)
    context_growth_fig = build_context_growth_chart(message_rows, dark=dark)

    error_count = sum(1 for s in steps for tc in s.get("tool_calls", []) if tc.get("error_type"))
    antipattern_html = build_antipattern_summary_html(
        session.fruitless_streaks,
        session.tool_selection,
        session.plan_metrics,
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


def build_diagnostics_outputs(session: LoadedSession, dark: bool = False) -> dict:
    """Build diagnostics outputs from precomputed session domain data."""
    empty_fig = empty_plotly_fig()
    interactions = session.file_interactions
    target_files = session.target_files
    file_chart = (
        build_file_interaction_chart(
            interactions,
            target_files,
            dark=dark,
            steps=session.steps,
        )
        if interactions
        else empty_fig
    )

    rootcause_html = build_root_cause_html(session.clusters) if session.show_root_cause else ""

    pressure_fig = build_context_pressure_chart(
        session.steps,
        agent_key=PRESSURE_ALL_AGENTS,
        raw=session.raw,
        dark=dark,
    )
    pressure_html = format_context_pressure_html(session.pressure_series)
    pressure_choices = session.pressure_choices
    pressure_dropdown = {
        "choices": pressure_choices or [("All agents", PRESSURE_ALL_AGENTS)],
        "value": PRESSURE_ALL_AGENTS,
        "visible": len(pressure_choices) > 2,
    }

    parts = []
    chain_metrics = session.chain_metrics
    if chain_metrics["total_chains"]:
        parts.append(f"{chain_metrics['total_chains']} failure chain(s)")
    if session.clusters:
        parts.append(f"{len(session.clusters)} root cause(s)")
    if session.bottleneck_explanations:
        parts.append(f"{len(session.bottleneck_explanations)} hotspot(s)")
    if interactions:
        unique_files = len({i["path"] for i in interactions})
        parts.append(f"{unique_files} file(s) touched · {len(target_files)} edited")
    compaction_count = len(session.pressure_series.get("events") or [])
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


def build_workflow_outputs(steps: list[dict]) -> dict:
    """Build workflow HTML, TOC, filter chips, and detail store."""
    wf_html = render_workflow_html(steps)
    wf_count = f"<div class='wf-count'>Showing {len(steps)} of {len(steps)} steps</div>"
    toc_html_val = render_toc_sidebar(steps)
    detail_store_val = _prerender_step_details(steps)

    wf_chips = render_filter_chips()
    wf_filter_val = ",".join(FILTER_CHIPS_DEFAULT)

    return {
        "wf_chips": wf_chips,
        "wf_filter_val": wf_filter_val,
        "wf_count": wf_count,
        "toc_html_val": toc_html_val,
        "wf_html": wf_html,
        "detail_store_val": detail_store_val,
    }


def render_tool_sequences_html(sequences: list[dict]) -> str:
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


def render_failure_patterns_html(patterns: list[dict]) -> str:
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


def load_warnings_html(session: LoadedSession) -> str:
    """HTML warning strip for truncation and token-integrity issues."""
    chunks = []
    if session.truncated:
        extra = session.steps_total - MAX_STEPS
        chunks.append(
            f"<p style='color:#d97706;font-size:13px;margin:0 0 4px;'>"
            f"&#9888; Showing first {MAX_STEPS:,} of {session.steps_total:,} steps "
            f"({extra:,} truncated).</p>"
        )
    for tw in session.token_warnings:
        chunks.append(f"<p style='color:#d97706;font-size:13px;margin:0 0 4px;'>&#9888; {html.escape(tw)}</p>")
    return "".join(chunks)


def raw_json_text(raw: dict) -> str:
    """Pretty-print trajectory JSON, truncated at 500KB."""
    raw_str = json.dumps(raw, indent=2, ensure_ascii=False, default=str)
    if len(raw_str) > 500_000:
        raw_str = raw_str[:500_000] + "\n\n... (truncated at 500KB)"
    return raw_str
