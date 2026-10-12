"""Step and agent usage charts: tokens, duration, and tools."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Callable

from ._layout import _add_legend_hint, _apply_chart_layout, _apply_dark, _empty_figure, _truncate_chart_labels
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..parser import infer_non_cache_input
from ..palette import (
    AGENT_COLORS,
    CHART_ACCENT,
    DURATION_ERROR_COLORS,
    SESSION_COLORS,
    TOKEN_COLORS,
)
from trajviz.tool_vocab import parse_skill_name
from ._timeline import _legend_label, bind_timeline_agents
from ..metrics import (
    spawn_wait_seconds,
    step_duration_excluding_spawn,
    tool_call_stats_duration_ms,
)
from ..shell_cmd import tool_chart_name
from ..step_errors import step_error_kind


def _subplot_rc(row: int | None, col: int | None) -> dict[str, int]:
    """Plotly subplot kwargs, or ``{}`` for a standalone figure."""
    if row is None or col is None:
        return {}
    return {"row": row, "col": col}


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
    row: int | None = None,
    col: int | None = None,
    showlegend: bool = True,
) -> None:
    """Add token category bar traces to a figure.

    By default only adds traces that have non-zero data to avoid misleading legends
    (e.g., trajectories that only report total tokens, no breakdown).

    Pass ``include_empty=True`` to force every trace into the legend (useful when
    the viewer wants to see which fields are tracked even if they happen to be
    zero across the entire trajectory).

    Pass ``cache_write`` to add a 5th stacked trace. For OpenCode this sample
    always reports zero, but the field is present and worth showing.

    Stack order (bottom → top) mirrors context-window occupancy: cache read /
    write form the base, then fresh input, then generated output / reasoning.
    """
    traces = [("Cache Read", cache_read, TOKEN_COLORS["cache_read"])]
    if cache_write is not None:
        traces.append(("Cache Write", cache_write, TOKEN_COLORS["cache_write"]))
    traces.extend(
        [
            ("Fresh Input", fresh_input, TOKEN_COLORS["fresh_input"]),
            ("Output", output, TOKEN_COLORS["output"]),
            ("Reasoning", reasoning, TOKEN_COLORS["reasoning"]),
        ]
    )
    rc = _subplot_rc(row, col)
    step_prefix = f"{x_label} " if x_label else ""
    for name, values, color in traces:
        if include_empty or any(v > 0 for v in values):
            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=values,
                    name=name,
                    legendgroup=name,
                    showlegend=showlegend,
                    marker_color=color,
                    hovertemplate=f"{step_prefix}%{{x}}<br>{name}: %{{y:,.0f}}<extra></extra>",
                ),
                **rc,
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


def _step_x_values(steps: list[dict]) -> list[int]:
    """Workflow step indices for the x-axis (same numbering as the Workflow view)."""
    return [int(s.get("index", i)) for i, s in enumerate(steps)]


def _workflow_x_range(steps: list[dict]) -> list[float]:
    xs = _step_x_values(steps)
    return [-0.5, max(xs) + 0.5] if xs else [-0.5, 0.5]


def _agent_panel_height(n_panels: int, base: int) -> int:
    return min(2400, max(base, 200 * n_panels + 100))


def _make_agent_subplots(agent_ids: list[str], labels: dict[str, str]) -> go.Figure:
    """One shared-x row per timeline agent; room for per-panel Step axes."""
    n = len(agent_ids)
    # Plotly requires vertical_spacing <= 1/(rows-1).
    return make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[_legend_label(aid, labels) for aid in agent_ids],
        vertical_spacing=min(0.10, 0.95 / max(n - 1, 1)),
    )


def _apply_shared_step_axes(fig: go.Figure, *, n_panels: int, x_range: list[float], y_title: str) -> None:
    # shared_xaxes hides tick labels on upper panels by default.
    for row in range(1, n_panels + 1):
        fig.update_yaxes(title_text=y_title, row=row, col=1)
        fig.update_xaxes(
            title_text="Step",
            range=x_range,
            showticklabels=True,
            row=row,
            col=1,
        )


def _finalize_step_chart(
    fig: go.Figure,
    *,
    title: str,
    height: int,
    barmode: str,
    legend: dict,
    dark: bool,
    xaxis: str | None = None,
    yaxis: str | None = None,
    extra_layout: dict | None = None,
) -> go.Figure:
    _apply_chart_layout(
        fig,
        title,
        xaxis=xaxis,
        yaxis=yaxis,
        height=height,
        barmode=barmode,
        legend=legend,
    )
    if extra_layout:
        fig.update_layout(**extra_layout)
    _add_legend_hint(fig)
    _apply_dark(fig, dark)
    return fig


def _claim_legend(legend_shown: dict[str, bool], name: str) -> bool:
    """True the first time *name* is drawn; keeps multi-panel legends unique."""
    if legend_shown.get(name):
        return False
    legend_shown[name] = True
    return True


def _draw_token_panel(
    fig: go.Figure,
    steps: list[dict],
    *,
    format: str | None,
    x_values: list,
    row: int | None = None,
    col: int | None = None,
    showlegend: bool = True,
    x_label: str = "Step",
) -> None:
    """Draw one stacked (or total-only) token panel onto *fig*."""
    cache_r = [s["tokens"]["cache_read"] for s in steps]
    fresh = [
        infer_non_cache_input(
            total_tokens=s["tokens"]["total"],
            input_tokens=s["tokens"]["input"],
            output_tokens=s["tokens"]["output"],
            reasoning_tokens=s["tokens"].get("reasoning", 0),
            cache_read_tokens=s["tokens"]["cache_read"],
        )
        for s in steps
    ]
    reasoning = [s["tokens"].get("reasoning", 0) for s in steps]
    output_t = [s["tokens"]["output"] for s in steps]
    # OpenCode/CodeArts report output and reasoning as disjoint fields.
    stacked_out = (
        output_t
        if format in ("opencode", "codearts")
        else [max(0, out - rea) for out, rea in zip(output_t, reasoning, strict=True)]
    )
    cache_w = [s["tokens"].get("cache_write", 0) or 0 for s in steps]
    has_breakdown = any(
        s["tokens"]["input"] > 0 or s["tokens"]["output"] > 0 or s["tokens"]["cache_read"] > 0 for s in steps
    )
    if has_breakdown:
        # Force empty categories into the legend only on the first panel; later
        # panels omit all-zero series to avoid ~5×agent dead traces.
        opencode_schema = format in ("opencode", "codearts")
        _add_token_bar_traces(
            fig,
            x_values,
            fresh,
            cache_r,
            stacked_out,
            reasoning,
            cache_write=cache_w if opencode_schema else None,
            include_empty=opencode_schema and showlegend,
            x_label=x_label,
            row=row,
            col=col,
            showlegend=showlegend,
        )
        return

    fig.add_trace(
        go.Bar(
            x=x_values,
            y=[s["tokens"]["total"] for s in steps],
            name="Total",
            legendgroup="Total",
            showlegend=showlegend,
            marker_color=CHART_ACCENT,
            hovertemplate=f"{x_label + ' ' if x_label else ''}%{{x}}<br>Total: %{{y:,.0f}}<extra></extra>",
        ),
        **_subplot_rc(row, col),
    )


def _group_steps_by_timeline_agent(
    steps: list[dict],
) -> tuple[list[str], dict[str, str], dict[str, list[dict]]]:
    """Ordered agent ids, legend labels, and per-agent step groups."""
    color_map, labels, agent_id_of = bind_timeline_agents(steps)
    groups: dict[str, list[dict]] = defaultdict(list)
    for step in steps:
        groups[agent_id_of(step)].append(step)
    agent_ids = [aid for aid in color_map if groups.get(aid)]
    return agent_ids, labels, groups


def build_token_chart(steps: list[dict], dark: bool = False, *, format: str | None = None) -> go.Figure:
    """Stacked bar of token breakdown over steps (non-overlapping segments).

    Default segments (bottom → top): cache_read + fresh_input
                      + net_output (output - reasoning) + reasoning = total

    Per-format overrides:
    - ``format in ("opencode", "codearts")``: render all five token fields
      stacked — Cache Read, Cache Write, Fresh Input, Output, Reasoning — with
      every trace forced into the legend even when a field is zero across the
      trajectory.  Both formats expose the same complete token schema.
    - No breakdown available: fall back to a single ``Total``
      bar.

    When the trajectory has multiple timeline agents (OpenCode/CodeArts
    subagent sessions, plan/build modes, …), each agent gets its own panel so
    context growth stays readable instead of zigzagging across interleaved
    sessions. Every panel shares the Workflow step index on the x-axis so a
    bar at step N lines up with Workflow step N (and with the other panels).
    """
    if not steps:
        fig = _empty_figure(380)
        _apply_dark(fig, dark)
        return fig

    agent_ids, labels, groups = _group_steps_by_timeline_agent(steps)
    legend = dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5)

    if len(agent_ids) <= 1:
        panel_steps = groups[agent_ids[0]] if agent_ids else steps
        fig = go.Figure()
        _draw_token_panel(fig, panel_steps, format=format, x_values=_step_x_values(panel_steps))
        return _finalize_step_chart(
            fig,
            title="Token Usage by Step",
            height=380,
            barmode="stack",
            legend=legend,
            dark=dark,
            xaxis="Step",
            yaxis="Tokens",
        )

    n = len(agent_ids)
    fig = _make_agent_subplots(agent_ids, labels)
    x_range = _workflow_x_range(steps)
    for row, agent_id in enumerate(agent_ids, start=1):
        panel_steps = groups[agent_id]
        _draw_token_panel(
            fig,
            panel_steps,
            format=format,
            x_values=_step_x_values(panel_steps),
            row=row,
            col=1,
            showlegend=(row == 1),
        )
    _apply_shared_step_axes(fig, n_panels=n, x_range=x_range, y_title="Tokens")
    return _finalize_step_chart(
        fig,
        title="Token Usage by Step",
        height=_agent_panel_height(n, 380),
        barmode="stack",
        legend={**legend, "y": 1.02},
        dark=dark,
    )


def _draw_duration_panel(
    fig: go.Figure,
    steps: list[dict],
    *,
    row: int | None = None,
    col: int | None = None,
    legend_shown: dict[str, bool] | None = None,
    compression_steps: list[int] | None = None,
) -> None:
    """Draw one step-duration panel (bars, spikes, avg, compression marks)."""
    rc = _subplot_rc(row, col)
    chart_durs = [step_duration_excluding_spawn(s) for s in steps]
    durations = [d if d is not None else 0 for d in chart_durs]
    spawn_adjusted = [
        d is not None and spawn_wait_seconds(s) > 0
        for s, d in zip(steps, chart_durs, strict=True)
    ]
    real_durations = [d for d in chart_durs if d is not None]
    avg_d = sum(real_durations) / len(real_durations) if real_durations else 0

    kinds = [step_error_kind(s) for s in steps]
    step_ids = _step_x_values(steps)
    outlier_steps = {step_ids[idx] for idx, _, _ in _detect_outliers(durations)}
    legend_shown = legend_shown if legend_shown is not None else {}

    series = [
        ("Normal", None, "#3b82f6", ""),
        ("System Error", "system", DURATION_ERROR_COLORS["system"], " (system error)"),
        ("Tool Error", "tool", DURATION_ERROR_COLORS["tool"], " (tool error)"),
    ]
    for name, kind, color, suffix in series:
        positions = [i for i, k in enumerate(kinds) if k == kind]
        if name != "Normal" and not positions:
            continue
        xs = [step_ids[i] for i in positions]
        ys = [durations[i] for i in positions]
        custom = [
            [step_ids[i], " (excl. task wait)" if spawn_adjusted[i] else ""]
            for i in positions
        ]
        fig.add_trace(
            go.Bar(
                x=xs,
                y=ys,
                customdata=custom,
                name=name,
                legendgroup=name,
                showlegend=_claim_legend(legend_shown, name),
                marker_color=color,
                width=0.8,
                hovertemplate=(
                    "Step %{customdata[0]}<br>%{y:.1f}s%{customdata[1]}"
                    f"{suffix}<extra></extra>"
                ),
            ),
            **rc,
        )
        spike_x = [x for x, y in zip(xs, ys, strict=True) if x in outlier_steps]
        spike_y = [y for x, y in zip(xs, ys, strict=True) if x in outlier_steps]
        if spike_x:
            fig.add_trace(
                go.Scatter(
                    x=spike_x,
                    y=spike_y,
                    mode="text",
                    text=[f"{v:.1f}s" for v in spike_y],
                    textposition="top center",
                    textfont=dict(size=9, color="#dc2626"),
                    legendgroup=name,
                    showlegend=False,
                    hoverinfo="skip",
                    cliponaxis=False,
                ),
                **rc,
            )

    fig.add_hline(
        y=avg_d,
        line_dash="dash",
        line_color="#94a3b8",
        line_width=1,
        annotation_text=f"Avg: {avg_d:.1f}s",
        annotation_position="top left",
        annotation_font=dict(color="#64748b", size=11),
        **rc,
    )
    panel_steps = set(step_ids)
    for step_idx in compression_steps or []:
        if step_idx not in panel_steps:
            continue
        fig.add_vline(
            x=step_idx,
            line_dash="dot",
            line_color="#dc2626",
            line_width=1.5,
            annotation_text="compressed",
            annotation_position="top",
            annotation_font=dict(size=8, color="#dc2626"),
            **rc,
        )


def build_duration_chart(
    steps: list[dict],
    dark: bool = False,
    compression_steps: list[int] | None = None,
) -> go.Figure:
    """Bar chart of step durations with average line.

    System errors (scaffold tools) and tool errors (agentic/custom) are
    separate legend series. Optional context compression markers.

    Spawn/delegation wait (``task``, ``Agent``, …) is subtracted from each
    bar so parent steps blocked on a child agent do not create false spikes.

    Multi-agent trajectories get one panel per timeline agent (same split as
    Token Usage by Step), sharing the Workflow step index on the x-axis.
    """
    if not steps:
        fig = _empty_figure(380)
        _apply_dark(fig, dark)
        return fig

    agent_ids, labels, groups = _group_steps_by_timeline_agent(steps)
    x_range = _workflow_x_range(steps)
    legend = dict(
        orientation="h",
        yanchor="bottom",
        y=1.06,
        xanchor="center",
        x=0.5,
        itemclick="toggle",
        itemdoubleclick="toggleothers",
    )
    duration_layout = dict(margin=dict(t=70), clickmode="event")

    if len(agent_ids) <= 1:
        panel_steps = groups[agent_ids[0]] if agent_ids else steps
        fig = go.Figure()
        _draw_duration_panel(fig, panel_steps, compression_steps=compression_steps)
        fig.update_xaxes(title_text="Step", range=x_range, showticklabels=True)
        fig.update_yaxes(title_text="Duration (s)")
        return _finalize_step_chart(
            fig,
            title="Step Duration",
            height=400,
            barmode="overlay",
            legend=legend,
            dark=dark,
            xaxis="Step",
            yaxis="Duration (s)",
            extra_layout=duration_layout,
        )

    n = len(agent_ids)
    fig = _make_agent_subplots(agent_ids, labels)
    legend_shown: dict[str, bool] = {}
    for row, agent_id in enumerate(agent_ids, start=1):
        _draw_duration_panel(
            fig,
            groups[agent_id],
            row=row,
            col=1,
            legend_shown=legend_shown,
            compression_steps=compression_steps,
        )
    _apply_shared_step_axes(fig, n_panels=n, x_range=x_range, y_title="Duration (s)")
    return _finalize_step_chart(
        fig,
        title="Step Duration",
        height=_agent_panel_height(n, 400),
        barmode="overlay",
        legend={**legend, "y": 1.02},
        dark=dark,
        extra_layout=duration_layout,
    )


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
            name = tool_chart_name(tc)
            agent_tool[agent][name] += 1
            all_tools[name] += 1

    if not all_tools:
        fig = _empty_figure(300, "No tool calls recorded in this trajectory.")
        _apply_dark(fig, dark)
        return fig

    sorted_tools = sorted(all_tools.keys(), key=lambda t: all_tools[t])
    display_names = _truncate_chart_labels(sorted_tools)

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
    _apply_tool_hbar_layout(
        fig,
        "Tool Call Frequency" + (" by Agent" if has_agents else ""),
        xaxis="Count",
        n_tools=len(sorted_tools),
        max_label=max_label,
        barmode="stack" if has_agents else "relative",
    )
    if has_agents:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),margin=dict(t=150),)
    _apply_dark(fig, dark)
    return fig


def build_tool_duration_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Horizontal bars of tool duration, stacked by individual timed calls.

    Each segment is one tool invocation. Hover shows that segment's duration
    and the step index where it ran (plus the agent label when multiple agents
    are present).

    Spawn/delegation tools (``task``, ``Agent``, …) are omitted: their duration
    is wall-clock for the whole child agent, which swamps real tool timings and
    double-counts work already shown via the child's own calls.
    """
    color_map, labels, agent_id_of = bind_timeline_agents(steps)
    has_agents = len(color_map) > 1

    calls_by_tool: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    total_secs: dict[str, float] = defaultdict(float)
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            continue
        agent = agent_id_of(s)
        step_idx = int(s.get("index", i))
        for tc in s.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            ms = tool_call_stats_duration_ms(tc)
            if ms is None:
                continue
            name = tool_chart_name(tc)
            secs = ms / 1000.0
            calls_by_tool[name].append((step_idx, secs, agent))
            total_secs[name] += secs

    if not total_secs:
        fig = _empty_figure(300, "No tool-call timing recorded in this trajectory.")
        _apply_dark(fig, dark)
        return fig

    sorted_tools = sorted(total_secs.keys(), key=lambda t: total_secs[t])
    display_names = _truncate_chart_labels(sorted_tools)

    fig = go.Figure()
    legend_seen: set[str] = set()
    # One trace per timed call, deliberately: each segment needs its own base,
    # color and step index in customdata, which is what makes a bar clickable
    # back to its Workflow step. Trace count therefore equals timed-call count.
    # Measured over 2,500 real exports: median 12 traces, worst case 177
    # (opencode_glm/sympy__sympy-15599) at 0.10s to build and 70 KB of chart
    # HTML, so the cost only bites far outside any real trajectory (10,000
    # synthetic calls: 1.4s, 3.1 MB). Batching into one array-valued Bar per
    # agent would have to keep the per-point step index resolvable.
    for tool_name, y_label in zip(sorted_tools, display_names, strict=True):
        base = 0.0
        for step_idx, secs, agent_id in calls_by_tool[tool_name]:
            if has_agents:
                color = SESSION_COLORS[color_map.get(agent_id, 0) % len(SESSION_COLORS)]
                agent_label = _legend_label(agent_id, labels)
                show_legend = agent_id not in legend_seen
                if show_legend:
                    legend_seen.add(agent_id)
                hover_extra = f"<br>{agent_label}"
            else:
                color = CHART_ACCENT
                agent_label = ""
                show_legend = False
                hover_extra = ""

            fig.add_trace(
                go.Bar(
                    y=[y_label],
                    x=[secs],
                    base=[base],
                    orientation="h",
                    name=agent_label,
                    legendgroup=agent_id if has_agents else None,
                    showlegend=show_legend,
                    marker=dict(
                        color=color,
                        line=dict(width=1, color="rgba(255,255,255,0.9)"),
                    ),
                    customdata=[step_idx],
                    hovertemplate=(
                        f"{tool_name}<br>Step %{{customdata}}"
                        f"<br>{secs:.1f}s{hover_extra}<extra></extra>"
                    ),
                )
            )
            base += secs

    max_label = max(len(n) for n in display_names)
    _apply_tool_hbar_layout(
        fig,
        "Tool Call Duration" + (" by Agent" if has_agents else ""),
        xaxis="Duration (s)",
        n_tools=len(sorted_tools),
        max_label=max_label,
        barmode="overlay",
    )
    fig.update_layout(clickmode="event")
    fig.update_yaxes(categoryorder="array", categoryarray=display_names)
    if has_agents:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.00, xanchor="right", x=1.0),margin=dict(t=200))
    _apply_dark(fig, dark)
    return fig


def _apply_tool_hbar_layout(
    fig: go.Figure,
    title: str,
    *,
    xaxis: str,
    n_tools: int,
    max_label: int,
    barmode: str,
) -> None:
    """Shared layout for horizontal tool frequency / duration bars."""
    _apply_chart_layout(
        fig,
        title,
        xaxis=xaxis,
        height=max(250, 50 * n_tools),
        barmode=barmode,
        margin=dict(l=max(140, max_label * 7 + 20), r=60, t=50, b=40),
    )


def _hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to an ``rgba()`` string."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _spread_sankey_y(n: int) -> list[float]:
    """Even y positions in (0, 1), inset so first/last nodes are not clipped."""
    if n <= 1:
        return [0.5]
    return [0.08 + 0.84 * i / (n - 1) for i in range(n)]


def _trunc_skill_label(name: str, limit: int = 28) -> str:
    return name if len(name) <= limit else name[: limit - 3] + "..."


def collect_skill_calls(
    steps: list[dict],
    agent_id_of: Callable[[dict], str],
) -> Counter[tuple[str, str]]:
    """Count ``(agent_id, skill_id)`` pairs from Skill-tool invocations."""
    counts: Counter[tuple[str, str]] = Counter()
    for step in steps:
        if not isinstance(step, dict):
            continue
        agent = agent_id_of(step)
        for tc in step.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            skill = parse_skill_name(str(tc.get("tool_name") or ""), tc.get("input"))
            if skill:
                counts[(agent, skill)] += 1
    return counts


def build_skill_agent_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Sankey diagram of Skill-tool invocations: which agent called which skill."""
    color_map, labels, agent_id_of = bind_timeline_agents(steps)
    counts = collect_skill_calls(steps, agent_id_of)
    if not counts:
        fig = _empty_figure(300, "No Skill-tool invocations recorded in this trajectory.")
        _apply_dark(fig, dark)
        return fig

    skill_totals: Counter[str] = Counter()
    agent_totals: Counter[str] = Counter()
    for (agent, skill), n in counts.items():
        skill_totals[skill] += n
        agent_totals[agent] += n

    agent_ids = [aid for aid in sorted(color_map.keys(), key=lambda a: color_map[a]) if agent_totals[aid]]
    skills = [name for name, _ in skill_totals.most_common()]

    node_labels: list[str] = []
    node_colors: list[str] = []
    node_x: list[float] = []
    node_y: list[float] = []

    agent_ys = _spread_sankey_y(len(agent_ids))
    skill_ys = _spread_sankey_y(len(skills))

    agent_index: dict[str, int] = {}
    for i, aid in enumerate(agent_ids):
        agent_index[aid] = len(node_labels)
        label = _legend_label(aid, labels)
        node_labels.append(f"{label} ({agent_totals[aid]})")
        node_colors.append(SESSION_COLORS[color_map[aid] % len(SESSION_COLORS)])
        node_x.append(0.02)
        node_y.append(agent_ys[i])

    skill_index: dict[str, int] = {}
    skill_color = "#64748b"
    for i, skill in enumerate(skills):
        skill_index[skill] = len(node_labels)
        node_labels.append(f"{_trunc_skill_label(skill)} ({skill_totals[skill]})")
        node_colors.append(skill_color)
        node_x.append(0.98)
        node_y.append(skill_ys[i])

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    link_colors: list[str] = []
    hovers: list[str] = []
    for (agent, skill), n in counts.items():
        sources.append(agent_index[agent])
        targets.append(skill_index[skill])
        values.append(n)
        agent_color = SESSION_COLORS[color_map[agent] % len(SESSION_COLORS)]
        link_colors.append(_hex_rgba(agent_color, 0.38))
        agent_label = _legend_label(agent, labels)
        noun = "call" if n == 1 else "calls"
        hovers.append(f"{agent_label} → {skill}<br>{n} {noun}")

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    pad=14,
                    thickness=16,
                    line=dict(width=0.5, color="rgba(148,163,184,0.45)"),
                    label=node_labels,
                    color=node_colors,
                    x=node_x,
                    y=node_y,
                    hovertemplate="%{label}<extra></extra>",
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    customdata=hovers,
                    hovertemplate="%{customdata}<extra></extra>",
                ),
            )
        ]
    )
    n_nodes = max(len(agent_ids), len(skills))
    _apply_chart_layout(
        fig,
        "Skill Calls by Agent",
        height=max(300, 48 * n_nodes + 80),
        margin=dict(l=16, r=16, t=90, b=24),
    )
    _apply_dark(fig, dark)
    return fig


def build_agent_token_chart(agent_summaries: list[dict], dark: bool = False) -> go.Figure:
    """Grouped bar chart showing token breakdown per agent.

    Uses the same vertical grouped-bar layout for single- and multi-agent
    sessions so the single-agent case still surfaces the four token
    categories (Cache Read / Fresh / Output / Reasoning) — just with one
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

    # Say so when a bar is a partial sum. `compute_agent_summary` drops any step
    # whose token counts are unusable (OpenCode can report a negative input), so
    # without this the chart presents an incomplete total as a definitive one —
    # and `infer_non_cache_input` clamps at 0, which reads as a confident "no
    # fresh input" rather than "not all of it could be counted". The Overview
    # token chip already carries the same caveat.
    dropped = sum(a.get("input_tokens_unusable_steps", 0) or 0 for a in agent_summaries)
    title = "Token Breakdown by Agent" if has_breakdown else "Total Tokens by Agent"
    if dropped:
        title += f" — partial, excludes {dropped} step{'s' if dropped != 1 else ''}"

    _apply_chart_layout(
        fig,
        title,
        xaxis="Agent",
        yaxis="Tokens (count)",
        height=320,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    fig.update_layout(margin=dict(t=70))
    _apply_dark(fig, dark)
    return fig
