"""Step-axis agent timelines: swimlane, N-run, and tool outcomes."""

from __future__ import annotations

from collections import defaultdict

import plotly.graph_objects as go

from ..palette import SESSION_COLORS, TOOL_OUTCOME_COLORS
from ._layout import _apply_chart_layout, _apply_dark, _empty_figure
from ._timeline import (
    _disambiguate_timeline_labels,
    _legend_label,
    _timeline_agent_id,
    _timeline_context,
    _timeline_display_label,
    bind_timeline_agents,
)


def build_agent_swimlane_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Horizontal swimlane chart showing each agent's active step ranges and token contribution.

    Always renders the user/main lane alongside any sub-agent lanes — user
    prompts (or main-orchestrator steps) are meaningful information regardless
    of whether one or many sub-agents are present.
    """
    color_map, label_map, agent_id_of = bind_timeline_agents(steps)
    real_agents = [a for a in color_map if a]
    if not real_agents:
        fig = _empty_figure(180, "No agent activity recorded.")
        _apply_dark(fig, dark)
        return fig

    fig = go.Figure()
    # Group steps by agent and find contiguous runs
    agent_runs: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    prev: dict[str, int | None] = {}

    for s in steps:
        if not isinstance(s, dict):
            continue
        agent = agent_id_of(s)
        idx = int(s.get("index") if s.get("index") is not None else 0)
        tok = int((s.get("tokens") or {}).get("total") or 0)
        tools = int(s.get("tool_call_count") or 0)
        if agent not in prev or prev[agent] is None:
            agent_runs[agent].append((idx, idx, tok, tools))
        else:
            last = agent_runs[agent][-1]
            if idx == last[1] + 1:
                agent_runs[agent][-1] = (last[0], idx, last[2] + tok, last[3] + tools)
            else:
                agent_runs[agent].append((idx, idx, tok, tools))
        prev[agent] = idx

    lane_count = 0
    for i, agent_id in enumerate(sorted(color_map.keys(), key=lambda a: color_map[a])):
        label = _legend_label(agent_id, label_map)
        lane_count += 1
        hex_c = SESSION_COLORS[i % len(SESSION_COLORS)]
        for start, end, tok, tools in agent_runs.get(agent_id, []):
            width = end - start + 1
            fig.add_trace(
                go.Bar(
                    y=[label],
                    x=[width],
                    orientation="h",
                    base=start,
                    marker_color=hex_c,
                    name=label,
                    showlegend=False,
                    text=f"{width} steps, {tok:,} tok",
                    textposition="inside",
                    hovertext=(f"{label}: steps {start}–{end}<br>{tok:,} tokens, {tools} tool calls"),
                    hoverinfo="text",
                )
            )

    _apply_chart_layout(
        fig,
        "Agent Swimlane",
        xaxis="Step Index",
        height=max(160, 80 * max(1, lane_count)),
        barmode="overlay",
        margin=dict(l=100, r=20, t=40, b=30),
    )
    _apply_dark(fig, dark)
    return fig


def build_run_group_agent_timeline(
    runs: list[dict],
    *,
    dark: bool = False,
) -> go.Figure:
    """Compact N-run agent timeline: one horizontal lane per run.

    Each run dict needs ``label`` (or ``run_id``) and ``steps`` (parsed).
    Segments within a lane are colored by session / subagent / agent mode
    (see ``_timeline_context``); shared display names share colors across runs.
    """
    usable = [r for r in runs if isinstance(r, dict) and (r.get("steps") or [])]
    if not usable:
        fig = _empty_figure(200, "No agent activity across runs.")
        _apply_dark(fig, dark)
        return fig

    # Per-run context + shared palette keyed by display label (explore/plan/main)
    run_meta: list[tuple[dict, bool, bool, dict[str, str], dict[str, str], list[dict]]] = []
    color_map: dict[str, int] = {"main": 0}
    next_idx = 1
    for run in usable:
        steps = [s for s in (run.get("steps") or []) if isinstance(s, dict)]
        multi, use_names, primary = _timeline_context(steps)
        ids_in_run: list[str] = []
        seen: set[str] = set()
        for s in steps:
            aid = _timeline_agent_id(s, multi_session=multi, use_agent_names=use_names, primary_agents=primary)
            if aid not in seen:
                ids_in_run.append(aid)
                seen.add(aid)
        label_map = _disambiguate_timeline_labels(ids_in_run, steps)
        run_meta.append((run, multi, use_names, label_map, primary, steps))
        for aid in ids_in_run:
            # Color by pre-disambiguation display name so explore matches across runs
            bucket = _timeline_display_label(aid, steps)
            if bucket not in color_map:
                color_map[bucket] = next_idx
                next_idx += 1

    fig = go.Figure()
    legend_seen: set[str] = set()
    y_labels: list[str] = []

    # First loaded run at top
    for run, multi, use_names, label_map, primary, steps in reversed(run_meta):
        label = str(run.get("label") or run.get("run_id") or "run")
        y_labels.append(label)

        segments: list[tuple[str, str, int, int, int, int]] = []
        cur_id: str | None = None
        cur_dlabel = ""
        start = end = tok = tools = 0
        for s in steps:
            aid = _timeline_agent_id(s, multi_session=multi, use_agent_names=use_names, primary_agents=primary)
            dlabel = _legend_label(aid, label_map)
            idx = int(s.get("index") if s.get("index") is not None else 0)
            stok = int((s.get("tokens") or {}).get("total") or 0)
            stools = int(s.get("tool_call_count") or 0)
            if cur_id is None:
                cur_id, cur_dlabel = aid, dlabel
                start, end, tok, tools = idx, idx, stok, stools
            elif aid == cur_id and idx == end + 1:
                end, tok, tools = idx, tok + stok, tools + stools
            else:
                segments.append((cur_id, cur_dlabel, start, end, tok, tools))
                cur_id, cur_dlabel = aid, dlabel
                start, end, tok, tools = idx, idx, stok, stools
        if cur_id is not None:
            segments.append((cur_id, cur_dlabel, start, end, tok, tools))

        for aid, dlabel, seg_start, seg_end, seg_tok, seg_tools in segments:
            width = seg_end - seg_start + 1
            bucket = _timeline_display_label(aid, steps)
            pal_i = color_map.get(bucket, color_map.get("main", 0))
            hex_c = SESSION_COLORS[pal_i % len(SESSION_COLORS)]
            show_leg = dlabel not in legend_seen
            if show_leg:
                legend_seen.add(dlabel)
            fig.add_trace(
                go.Bar(
                    y=[label],
                    x=[width],
                    orientation="h",
                    base=seg_start,
                    marker_color=hex_c,
                    name=dlabel,
                    legendgroup=dlabel,
                    showlegend=show_leg,
                    hovertext=(
                        f"<b>{label}</b><br>{dlabel}<br>"
                        f"steps {seg_start}–{seg_end} ({width})<br>"
                        f"{seg_tok:,} tokens, {seg_tools} tool calls"
                    ),
                    hoverinfo="text",
                )
            )

    n = len(usable)
    _apply_chart_layout(
        fig,
        "Agent timeline (by run)",
        xaxis="Step Index",
        height=max(220, 56 * n + 100),
        barmode="overlay",
        margin=dict(l=120, r=20, t=90, b=40),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    fig.update_layout(
        yaxis=dict(
            categoryorder="array",
            categoryarray=y_labels,
            automargin=True,
        ),
    )
    _apply_dark(fig, dark)
    return fig


def build_tool_outcome_timeline(steps: list[dict], dark: bool = False) -> go.Figure:
    """Scatter plot showing tool call outcomes (success/failure) across steps."""
    if not steps:
        fig = _empty_figure(340)
        _apply_dark(fig, dark)
        return fig

    success_x, success_y = [], []
    failure_x, failure_y = [], []

    for s in steps:
        for tc in s["tool_calls"]:
            tool_name = tc.get("tool_name") or "(unnamed)"
            if len(tool_name) > 30:
                tool_name = tool_name[:27] + "..."
            has_error = tc.get("error") or tc.get("status") == "error"
            if has_error:
                failure_x.append(s["index"])
                failure_y.append(tool_name)
            else:
                success_x.append(s["index"])
                success_y.append(tool_name)

    if not success_x and not failure_x:
        fig = _empty_figure(340, "No tool calls recorded in this trajectory.")
        _apply_dark(fig, dark)
        return fig

    fig = go.Figure()
    if success_x:
        fig.add_trace(
            go.Scatter(
                x=success_x,
                y=success_y,
                mode="markers",
                name="Success",
                marker=dict(color=TOOL_OUTCOME_COLORS["success"], size=8, symbol="circle"),
                hovertemplate="Step %{x}<br>%{y}<br>Success<extra></extra>",
            )
        )
    if failure_x:
        fig.add_trace(
            go.Scatter(
                x=failure_x,
                y=failure_y,
                mode="markers",
                name="Failure",
                marker=dict(color=TOOL_OUTCOME_COLORS["failure"], size=8, symbol="x"),
                hovertemplate="Step %{x}<br>%{y}<br>Failure<extra></extra>",
            )
        )

    all_tools = sorted(set(success_y + failure_y))
    _apply_chart_layout(
        fig,
        "Tool Outcome Timeline",
        xaxis="Step",
        height=max(300, 30 * len(all_tools)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    _apply_dark(fig, dark)
    return fig
