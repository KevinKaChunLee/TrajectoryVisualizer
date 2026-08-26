"""Step and agent usage charts: tokens, duration, tools, context growth."""

from __future__ import annotations

import statistics
from collections import defaultdict

from ._layout import _add_legend_hint, _apply_chart_layout, _apply_dark, _empty_figure
import plotly.graph_objects as go

from ..parser import infer_non_cache_input
from ..palette import (
    AGENT_COLORS,
    CHART_ACCENT,
    ROLE_COLORS,
    SESSION_COLORS,
    TOKEN_COLORS,
)
from ._timeline import _legend_label, bind_timeline_agents


def _add_token_bar_traces(
    fig: go.Figure,
    x_values: list,
    fresh_input: list,
    cache_read: list,
    output: list,
    reasoning: list,
    *,
    cache_write: list | None = None,
    include_empty: bool = False,
    x_label: str = "Step",
) -> None:
    """Add token category bar traces to a figure.

    By default only adds traces that have non-zero data to avoid misleading legends
    (e.g., trajectories that only report total tokens, no breakdown).

    Pass ``include_empty=True`` to force every trace into the legend (useful when
    the viewer wants to see which fields are tracked even if they happen to be
    zero across the entire trajectory).

    Pass ``cache_write`` to add a 5th stacked trace. For OpenCode this sample
    always reports zero, but the field is present and worth showing.
    """
    traces = [
        ("Fresh Input", fresh_input, TOKEN_COLORS["fresh_input"]),
        ("Cache Read", cache_read, TOKEN_COLORS["cache_read"]),
        ("Output", output, TOKEN_COLORS["output"]),
        ("Reasoning", reasoning, TOKEN_COLORS["reasoning"]),
    ]
    if cache_write is not None:
        traces.append(("Cache Write", cache_write, TOKEN_COLORS["cache_write"]))
    for name, values, color in traces:
        if include_empty or any(v > 0 for v in values):
            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=values,
                    name=name,
                    marker_color=color,
                    hovertemplate=f"{(x_label + ' ') if x_label else ''}%{{x}}<br>{name}: %{{y:,.0f}}<extra></extra>",
                )
            )


def _detect_outliers(values: list[float], threshold: float = 2.0) -> list[tuple[int, float, str]]:
    """Return (index, value, label) for values exceeding *threshold* σ from mean.

    Returns empty list if fewer than 10 values or no outliers found.
    """
    if len(values) < 10:
        return []
    clean = [v for v in values if v is not None and v > 0]
    if len(clean) < 5:
        return []
    mean = statistics.mean(clean)
    stdev = statistics.stdev(clean) if len(clean) > 1 else 0
    if stdev == 0:
        return []
    outliers = []
    for i, v in enumerate(values):
        if v is not None and v > 0 and (v - mean) > threshold * stdev:
            outliers.append((i, v, "spike"))
    return outliers


def build_token_chart(steps: list[dict], dark: bool = False, *, format: str | None = None) -> go.Figure:
    """Stacked bar of token breakdown over steps (non-overlapping segments).

    Default segments: fresh_input + cache_read
                      + net_output (output - reasoning) + reasoning = total

    Per-format overrides:
    - ``format in ("opencode", "codearts")``: render all five token fields
      stacked — Fresh Input, Cache Read, Output, Reasoning, Cache Write — with
      every trace forced into the legend even when a field is zero across the
      trajectory.  Both formats expose the same complete token schema.
    - No breakdown available: fall back to a single ``Total``
      bar.
    """
    if not steps:
        fig = _empty_figure(380)
        _apply_dark(fig, dark)
        return fig

    indices = list(range(len(steps)))
    cache_r = [s["tokens"]["cache_read"] for s in steps]
    fresh_input = [
        infer_non_cache_input(
            total_tokens=s["tokens"]["total"],
            input_tokens=s["tokens"]["input"],
            output_tokens=s["tokens"]["output"],
            reasoning_tokens=s["tokens"]["reasoning"],
            cache_read_tokens=s["tokens"]["cache_read"],
        )
        for s in steps
    ]
    reasoning_t = [s["tokens"]["reasoning"] for s in steps]
    output_t = [s["tokens"]["output"] for s in steps]
    net_output = [max(0, output - reasoning) for output, reasoning in zip(output_t, reasoning_t, strict=False)]
    cache_w = [s["tokens"].get("cache_write", 0) or 0 for s in steps]

    # Detect if token breakdown is available (any non-zero input/output/cache)
    has_breakdown = any(
        s["tokens"]["input"] > 0 or s["tokens"]["output"] > 0 or s["tokens"]["cache_read"] > 0 for s in steps
    )

    fig = go.Figure()
    if format in ("opencode", "codearts") and has_breakdown:
        # OpenCode / CodeArts view: show all five fields explicitly, forcing
        # zero traces into the legend so no existing legend item disappears.
        # These formats report output and reasoning as disjoint fields, so use
        # the raw output value instead of subtracting reasoning a second time.
        _add_token_bar_traces(
            fig, indices, fresh_input, cache_r, output_t, reasoning_t, cache_write=cache_w, include_empty=True
        )
    elif has_breakdown:
        _add_token_bar_traces(fig, indices, fresh_input, cache_r, net_output, reasoning_t)
    else:
        # No breakdown available — single "Total" trace.
        totals = [s["tokens"]["total"] for s in steps]
        fig.add_trace(
            go.Bar(
                x=indices,
                y=totals,
                name="Total",
                marker_color=CHART_ACCENT,
                hovertemplate="Step %{x}<br>Total: %{y:,.0f}<extra></extra>",
            )
        )

    _apply_chart_layout(
        fig,
        "Token Usage by Step",
        xaxis="Step",
        yaxis="Tokens",
        height=380,
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    _add_legend_hint(fig)
    _apply_dark(fig, dark)
    return fig


def build_duration_chart(
    steps: list[dict],
    phases: list[dict] | None = None,
    dark: bool = False,
    compression_steps: list[int] | None = None,
) -> go.Figure:
    """Bar chart of step durations with average line.

    Error steps are highlighted in red; all others use a uniform blue.
    Optional context compression markers.
    """
    if not steps:
        fig = _empty_figure(380)
        _apply_dark(fig, dark)
        return fig

    durations = [s["duration"] if s["duration"] is not None else 0 for s in steps]

    # Average over steps that actually have a duration; steps with a missing
    # duration render as 0 bars but must not drag the mean down.
    real_durations = [s["duration"] for s in steps if s["duration"] is not None]
    avg_d = sum(real_durations) / len(real_durations) if real_durations else 0

    # Split into normal and error traces for legend
    normal_x = [i for i, s in enumerate(steps) if s["error_count"] == 0]
    normal_y = [durations[i] for i in normal_x]
    error_x = [i for i, s in enumerate(steps) if s["error_count"] > 0]
    error_y = [durations[i] for i in error_x]

    # Detect outliers — will be added as scatter labels per legend group
    outlier_set = {idx for idx, _, _ in _detect_outliers(durations)}

    bar_width = 0.8  # consistent width for both traces

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=normal_x,
            y=normal_y,
            name="Normal",
            legendgroup="Normal",
            marker_color="#3b82f6",
            width=bar_width,
            hovertemplate="Step %{x}<br>%{y:.1f}s<extra></extra>",
        )
    )
    if error_x:
        fig.add_trace(
            go.Bar(
                x=error_x,
                y=error_y,
                name="Error",
                legendgroup="Error",
                marker_color=ROLE_COLORS["error"],
                width=bar_width,
                hovertemplate="Step %{x}<br>%{y:.1f}s (error)<extra></extra>",
            )
        )

    # Spike labels as scatter traces — grouped with their bar trace so they
    # show/hide together when the legend is toggled.
    for group, gx, gy in [("Normal", normal_x, normal_y), ("Error", error_x, error_y)]:
        spike_x = [gx[j] for j in range(len(gx)) if gx[j] in outlier_set]
        spike_y = [gy[j] for j in range(len(gx)) if gx[j] in outlier_set]
        spike_text = [f"{v:.1f}s" for v in spike_y]
        if spike_x:
            fig.add_trace(
                go.Scatter(
                    x=spike_x,
                    y=spike_y,
                    mode="text",
                    text=spike_text,
                    textposition="top center",
                    textfont=dict(size=9, color="#dc2626"),
                    legendgroup=group,
                    showlegend=False,
                    hoverinfo="skip",
                    cliponaxis=False,  # allow text to extend past the right edge
                )
            )
    # Avg line — dashed, lighter
    fig.add_hline(
        y=avg_d,
        line_dash="dash",
        line_color="#94a3b8",
        line_width=1,
        annotation_text=f"Avg: {avg_d:.1f}s",
        annotation_position="top left",
        annotation_font=dict(color="#64748b", size=11),
    )
    _apply_chart_layout(
        fig,
        "Step Duration",
        xaxis="Step",
        yaxis="Duration (s)",
        height=400,
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5, itemclick="toggleothers"),
    )
    fig.update_layout(margin=dict(t=70))  # extra top margin for spike labels
    fig.update_xaxes(range=[-0.5, len(steps) - 0.5])

    # Context compression markers (red vertical lines)
    if compression_steps:
        for step_idx in compression_steps:
            fig.add_vline(
                x=step_idx,
                line_dash="dot",
                line_color="#dc2626",
                line_width=1.5,
                annotation_text="compressed",
                annotation_position="top",
                annotation_font=dict(size=8, color="#dc2626"),
            )

    _apply_dark(fig, dark)
    return fig


def build_tool_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Horizontal bar chart of tool call frequency by name, stacked by agent."""
    color_map, labels, agent_id_of = bind_timeline_agents(steps)
    has_agents = len(color_map) > 1

    agent_tool: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_tools: dict[str, int] = defaultdict(int)
    for s in steps:
        if not isinstance(s, dict):
            continue
        agent = agent_id_of(s)
        for tc in s.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            name = tc.get("tool_name") or "(unnamed)"
            agent_tool[agent][name] += 1
            all_tools[name] += 1

    if not all_tools:
        fig = _empty_figure(300, "No tool calls recorded in this trajectory.")
        _apply_dark(fig, dark)
        return fig

    sorted_tools = sorted(all_tools.keys(), key=lambda t: all_tools[t])
    display_names = [n if len(n) <= 30 else n[:27] + "..." for n in sorted_tools]

    fig = go.Figure()
    if has_agents:
        # Stacked bars per agent, matching swimlane / workflow labeling
        for agent_id in sorted(color_map.keys(), key=lambda a: color_map[a]):
            label = _legend_label(agent_id, labels)
            counts = [agent_tool[agent_id].get(t, 0) for t in sorted_tools]
            if not any(counts):
                continue
            color = SESSION_COLORS[color_map[agent_id] % len(SESSION_COLORS)]
            fig.add_trace(
                go.Bar(
                    y=display_names,
                    x=counts,
                    orientation="h",
                    name=label,
                    marker_color=color,
                    hovertemplate=f"{label}: " + "%{x} call(s)<extra></extra>",
                )
            )
    else:
        counts = [all_tools[t] for t in sorted_tools]
        fig.add_trace(
            go.Bar(
                y=display_names,
                x=counts,
                orientation="h",
                marker_color=CHART_ACCENT,
                text=[str(c) for c in counts],
                textposition="outside",
                cliponaxis=False,
                customdata=sorted_tools,
                hovertemplate="%{customdata}: %{x} call(s)<extra></extra>",
            )
        )

    max_label = max(len(n) for n in display_names)
    _apply_chart_layout(
        fig,
        "Tool Call Frequency" + (" by Agent" if has_agents else ""),
        xaxis="Count",
        height=max(250, 50 * len(sorted_tools)),
        barmode="stack" if has_agents else "relative",
        margin=dict(l=max(140, max_label * 7 + 20), r=60, t=50, b=40),
    )
    if has_agents:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    _apply_dark(fig, dark)
    return fig


def build_context_growth_chart(rows: list[dict], phases: list[dict] | None = None, dark: bool = False) -> go.Figure:
    """Cumulative input tokens (context pressure) with cache-read overlay."""
    if not rows:
        fig = _empty_figure(340)
        _apply_dark(fig, dark)
        return fig

    indices = [r["index"] for r in rows]
    cum_input = []
    cum_fresh = []
    cum_cache = []
    ri, rf, rc = 0, 0, 0
    for r in rows:
        ri += r.get("tokens_input", 0)
        cache_read = r.get("cache_read", 0)
        # rows["non_cache_tokens"] is already schema-normalized in build_message_metrics()
        rf += r.get("non_cache_tokens", 0)
        rc += cache_read
        cum_input.append(ri)
        cum_fresh.append(rf)
        cum_cache.append(rc)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=cum_input,
            name="Cumulative Input",
            mode="lines+markers",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.08)",
            hovertemplate="Step %{x}<br>Cumul. Input: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=cum_fresh,
            name="Cumul. Fresh Input",
            mode="lines+markers",
            line=dict(color="#dc2626", width=2, dash="dot"),
            marker=dict(size=4),
            hovertemplate="Step %{x}<br>Cumul. Fresh: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=cum_cache,
            name="Cumul. Cache Read",
            mode="lines+markers",
            line=dict(color="#059669", width=2, dash="dash"),
            marker=dict(size=4),
            hovertemplate="Step %{x}<br>Cumul. Cache: %{y:,}<extra></extra>",
        )
    )
    _apply_chart_layout(
        fig,
        "Context Growth (Cumulative Input Tokens)",
        xaxis="Step",
        yaxis="Tokens (count)",
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    _add_legend_hint(fig)
    _apply_dark(fig, dark)
    return fig


def build_agent_token_chart(agent_summaries: list[dict], dark: bool = False) -> go.Figure:
    """Grouped bar chart showing token breakdown per agent.

    Uses the same vertical grouped-bar layout for single- and multi-agent
    sessions so the single-agent case still surfaces the four token
    categories (Fresh / Cache Read / Output / Reasoning) — just with one
    cluster instead of many.
    """
    if not agent_summaries:
        fig = _empty_figure(240, "No agent activity recorded.")
        _apply_dark(fig, dark)
        return fig

    labels = [a["label"] for a in agent_summaries]
    has_breakdown = any(a["input_tokens"] > 0 or a["output_tokens"] > 0 for a in agent_summaries)

    fig = go.Figure()
    if has_breakdown:
        from ..parser import infer_non_cache_input

        # Schema-tolerant fresh-input: handles both Claude Code (input includes
        # cache) and OpenCode (input is already cache-excluded).
        fresh_input = [
            infer_non_cache_input(
                a["total_tokens"],
                a["input_tokens"],
                a["output_tokens"],
                a["reasoning_tokens"],
                a["cache_read_tokens"],
            )
            for a in agent_summaries
        ]
        cache_read = [a["cache_read_tokens"] for a in agent_summaries]
        output = [max(0, a["output_tokens"] - a["reasoning_tokens"]) for a in agent_summaries]
        reasoning = [a["reasoning_tokens"] for a in agent_summaries]
        _add_token_bar_traces(fig, labels, fresh_input, cache_read, output, reasoning, x_label="")
    else:
        # No token breakdown available — show total tokens per agent
        session_palette = AGENT_COLORS  # shared palette so agent colors match across views
        totals = [a["total_tokens"] for a in agent_summaries]
        colors = [session_palette[i % len(session_palette)] for i in range(len(labels))]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=totals,
                name="Total Tokens",
                marker_color=colors,
                text=[f"{t:,}" for t in totals],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}<br>%{y:,.0f} tokens<extra></extra>",
            )
        )

    _apply_chart_layout(
        fig,
        "Token Breakdown by Agent" if has_breakdown else "Total Tokens by Agent",
        xaxis="Agent",
        yaxis="Tokens (count)",
        height=320,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    fig.update_layout(margin=dict(t=70))
    _apply_dark(fig, dark)
    return fig
