"""Plotly chart builders for trajectory visualization."""

import statistics

# Pre-import pandas before plotly to avoid circular import error
# in plotly's basevalidators when running inside Gradio async threads.
try:
    import pandas  # noqa: F401
except ImportError:
    pass

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .parser import infer_non_cache_input
from .metrics import effective_agent
from .palette import (
    TOKEN_COLORS, PHASE_COLORS, PHASE_FILL_COLORS, PHASE_LINE_COLORS,
    AGENT_COLORS, AGENT_CSS_COLORS, ROLE_COLORS, TOOL_OUTCOME_COLORS,
    CHART_ACCENT, PLOTLY_DARK_TEMPLATE,
)


def _apply_dark(fig: go.Figure, dark: bool) -> go.Figure:
    """Apply dark-mode layout overrides when *dark* is True."""
    if dark:
        fig.update_layout(**PLOTLY_DARK_TEMPLATE)
    return fig


def _effective_agent(s: dict) -> str:
    """Alias for backward compatibility within charts module."""
    return effective_agent(s)


def build_agent_color_map(steps: list[dict]) -> dict[str, int]:
    """Return a mapping from agent-id to palette index.

    The empty string / falsy agent is always index 0 ("main").
    Sub-agents are assigned indices 1+ in the order they first appear.
    """
    mapping: dict[str, int] = {"": 0}
    next_idx = 1
    for s in steps:
        agent = _effective_agent(s)
        if agent and agent not in mapping:
            mapping[agent] = next_idx
            next_idx += 1
    return mapping


def agent_color(agent_id: str, color_map: dict[str, int]) -> str:
    """Return the hex color for an agent given its color map index."""
    idx = color_map.get(agent_id, 0)
    return AGENT_COLORS[idx % len(AGENT_COLORS)]


def _plotly_step_color(step: dict) -> str:
    """Return a hex bar-color for a step (Plotly can't use CSS variables)."""
    if step["error_count"] > 0:
        return ROLE_COLORS["error"]
    if step.get("finish") in ("stop", "end_turn"):
        return ROLE_COLORS["stop"]
    if step["tool_call_count"] > 0:
        return ROLE_COLORS["tool"]
    if step["has_reasoning"] and step["role"] == "assistant":
        return ROLE_COLORS["reasoning"]
    return ROLE_COLORS.get(step["role"], ROLE_COLORS["default"])


# -- Layout helpers -------------------------------------------------------

_TPL = "plotly_white"


def _empty_figure(height: int = 380, message: str | None = None) -> go.Figure:
    """Return a blank Plotly figure, optionally with a centered message."""
    fig = go.Figure()
    if message:
        fig.add_annotation(text=message, xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font_size=16)
    fig.update_layout(template=_TPL, height=height)
    return fig


def _apply_chart_layout(fig: go.Figure, title: str,
                         xaxis: str | None = None, yaxis: str | None = None,
                         height: int = 380, **kwargs) -> None:
    """Apply standard chart layout (template + margins + responsive sizing).

    The title is centered horizontally (``x=0.5``, ``xanchor="center"``) and
    pinned to the top of the container (``y=0.99``, ``yanchor="top"``). The
    top margin is 90px so a horizontal legend above the plot area
    (``y=1.06``) can wrap to a second row without overlapping the title.

    Centering — rather than left-aligning — avoids collisions with the
    Gradio ``gr.Plot`` chrome: Gradio draws a small tab label in the
    top-left corner of every chart widget, which would sit on top of a
    left-aligned title.
    """
    layout = dict(
        title=dict(text=title, y=0.99, x=0.5,
                   xanchor="center", yanchor="top",
                   font=dict(size=16)),
        template=_TPL,
        height=height,
        autosize=True,
        margin=dict(t=90, b=40, l=60, r=20),
    )
    if xaxis:
        layout["xaxis_title"] = xaxis
    if yaxis:
        layout["yaxis_title"] = yaxis
    layout.update(kwargs)
    fig.update_layout(**layout)


def _add_legend_hint(fig: go.Figure) -> None:
    """Add a subtle 'click legend to toggle' hint at the bottom-right."""
    fig.add_annotation(
        text="Click legend items to show/hide series",
        xref="paper", yref="paper", x=1.0, y=-0.12,
        showarrow=False, font=dict(size=9, color="#9ca3af"),
        xanchor="right",
    )


# -- Reusable trace helpers ------------------------------------------------


def _add_token_bar_traces(fig: go.Figure, x_values: list,
                          fresh_input: list, cache_read: list,
                          output: list, reasoning: list,
                          *, cache_write: list | None = None,
                          include_empty: bool = False) -> None:
    """Add token category bar traces to a figure.

    By default only adds traces that have non-zero data to avoid misleading legends
    (e.g., CodeArts trajectories only have total tokens, no breakdown).

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
            fig.add_trace(go.Bar(
                x=x_values, y=values, name=name, marker_color=color,
                hovertemplate=f"Step %{{x}}<br>{name}: %{{y:,.0f}}<extra></extra>",
            ))


# -- Annotation utilities ------------------------------------------------

_PHASE_COLORS = PHASE_FILL_COLORS
_PHASE_LINE_COLORS = PHASE_LINE_COLORS


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


def add_phase_overlays(fig: go.Figure, phases: list[dict] | None,
                       step_count: int) -> None:
    """Draw semi-transparent vertical regions for each detected phase."""
    if not phases or len(phases) <= 1:
        return
    for p in phases:
        color = _PHASE_COLORS.get(p["name"], "rgba(107,114,128,0.06)")
        label_color = _PHASE_LINE_COLORS.get(p["name"], "#6b7280")
        fig.add_vrect(
            x0=p["start_idx"] - 0.5, x1=p["end_idx"] + 0.5,
            fillcolor=color, layer="below", line_width=0,
        )
        fig.add_annotation(
            x=(p["start_idx"] + p["end_idx"]) / 2, y=1.0,
            yref="paper", text=p["name"],
            showarrow=False, font=dict(size=10, color=label_color),
            yanchor="bottom",
        )


def _add_outlier_annotations(fig: go.Figure, outliers: list[tuple[int, float, str]],
                             fmt: str = ",.0f", suffix: str = "") -> None:
    """Add annotation arrows for detected outlier points."""
    for idx, val, label in outliers[:5]:  # cap at 5 to avoid clutter
        fig.add_annotation(
            x=idx, y=val,
            text=f"{label}: {val:{fmt}}{suffix}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
            arrowcolor="#dc2626", font=dict(size=9, color="#dc2626"),
            ax=0, ay=-30,
        )


# -- Chart builders -------------------------------------------------------

def _add_agent_regions(fig: go.Figure, steps: list[dict],
                       color_map: dict[str, int]) -> None:
    """Add semi-transparent vertical shading per agent span on a chart."""
    if not steps or len(color_map) <= 1:
        return
    # Find contiguous runs of the same agent
    runs: list[tuple[str, int, int]] = []
    prev_agent = effective_agent(steps[0])
    run_start = 0
    for i, s in enumerate(steps):
        a = effective_agent(s)
        if a != prev_agent:
            runs.append((prev_agent, run_start, i - 1))
            prev_agent = a
            run_start = i
    runs.append((prev_agent, run_start, len(steps) - 1))

    for agent_id, start, end in runs:
        if not agent_id:
            continue  # don't shade main agent
        hex_color = agent_color(agent_id, color_map)
        # Convert hex to rgba with low opacity
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        fig.add_vrect(
            x0=start - 0.5, x1=end + 0.5,
            fillcolor=f"rgba({r},{g},{b},0.08)", layer="below", line_width=0,
        )
        mid = (start + end) / 2
        label = agent_id[:8] + "…" if len(agent_id) > 8 else agent_id
        fig.add_annotation(
            x=mid, y=1.0, yref="paper",
            text=f"⬤ {label}", showarrow=False,
            font=dict(size=9, color=hex_color),
            yanchor="top", yshift=-2,
        )


def build_token_chart(steps: list[dict], cumulative: bool = False,
                      phases: list[dict] | None = None,
                      dark: bool = False,
                      *, format: str | None = None) -> go.Figure:
    """Stacked bar of token breakdown over steps (non-overlapping segments).

    Default segments: fresh_input + cache_read
                      + net_output (output - reasoning) + reasoning = total

    Per-format overrides:
    - ``format == "opencode"``: render all five token fields stacked — Fresh Input,
      Cache Read, Output, Reasoning, Cache Write — with every trace forced into
      the legend even when a field is zero across the trajectory. OpenCode's
      ``tokens.cache.write`` is typically 0 except on cache-creation turns; the
      5th stacked segment sums cleanly with the others because the sample
      satisfies ``input + cache.read + output + reasoning == total``.
    - No breakdown available (e.g., CodeArts): fall back to a single ``Total``
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
    net_output = [max(0, s["tokens"]["output"] - s["tokens"]["reasoning"]) for s in steps]
    cache_w = [s["tokens"].get("cache_write", 0) or 0 for s in steps]

    # Detect if token breakdown is available (any non-zero input/output/cache)
    has_breakdown = any(
        s["tokens"]["input"] > 0 or s["tokens"]["output"] > 0 or s["tokens"]["cache_read"] > 0
        for s in steps
    )

    fig = go.Figure()
    if format == "opencode" and has_breakdown:
        # OpenCode view: show all five fields explicitly, forcing zero traces
        # into the legend so the reader can see which fields are tracked.
        _add_token_bar_traces(fig, indices, fresh_input, cache_r, net_output, reasoning_t,
                              cache_write=cache_w, include_empty=True)
    elif has_breakdown:
        _add_token_bar_traces(fig, indices, fresh_input, cache_r, net_output, reasoning_t)
    else:
        # No breakdown available (e.g., CodeArts) — single "Total" trace.
        totals = [s["tokens"]["total"] for s in steps]
        fig.add_trace(go.Bar(
            x=indices, y=totals, name="Total",
            marker_color=CHART_ACCENT,
            hovertemplate="Step %{x}<br>Total: %{y:,.0f}<extra></extra>",
        ))

    _apply_chart_layout(
        fig, "Token Usage by Step",
        xaxis="Step", yaxis="Tokens", height=380,
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    _add_legend_hint(fig)
    _apply_dark(fig, dark)
    return fig


def build_duration_chart(steps: list[dict],
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

    indices = list(range(len(steps)))
    durations = [s["duration"] if s["duration"] is not None else 0 for s in steps]

    avg_d = sum(durations) / len(durations) if durations else 0

    # Split into normal and error traces for legend
    normal_x = [i for i, s in enumerate(steps) if s["error_count"] == 0]
    normal_y = [durations[i] for i in normal_x]
    error_x = [i for i, s in enumerate(steps) if s["error_count"] > 0]
    error_y = [durations[i] for i in error_x]

    # Detect outliers — will be added as scatter labels per legend group
    outlier_set = {idx for idx, _, _ in _detect_outliers(durations)}

    bar_width = 0.8  # consistent width for both traces

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=normal_x, y=normal_y, name="Normal",
        legendgroup="Normal",
        marker_color="#3b82f6",
        width=bar_width,
        hovertemplate="Step %{x}<br>%{y:.1f}s<extra></extra>",
    ))
    if error_x:
        fig.add_trace(go.Bar(
            x=error_x, y=error_y, name="Error",
            legendgroup="Error",
            marker_color=ROLE_COLORS["error"],
            width=bar_width,
            hovertemplate="Step %{x}<br>%{y:.1f}s (error)<extra></extra>",
        ))

    # Spike labels as scatter traces — grouped with their bar trace so they
    # show/hide together when the legend is toggled.
    for group, gx, gy in [("Normal", normal_x, normal_y), ("Error", error_x, error_y)]:
        spike_x = [gx[j] for j in range(len(gx)) if gx[j] in outlier_set]
        spike_y = [gy[j] for j in range(len(gx)) if gx[j] in outlier_set]
        spike_text = [f"{v:.1f}s" for v in spike_y]
        if spike_x:
            fig.add_trace(go.Scatter(
                x=spike_x, y=spike_y,
                mode="text", text=spike_text,
                textposition="top center",
                textfont=dict(size=9, color="#dc2626"),
                legendgroup=group, showlegend=False,
                hoverinfo="skip",
                cliponaxis=False,  # allow text to extend past the right edge
            ))
    # Avg line — dashed, lighter
    fig.add_hline(y=avg_d, line_dash="dash", line_color="#94a3b8", line_width=1,
                  annotation_text=f"Avg: {avg_d:.1f}s",
                  annotation_position="top left",
                  annotation_font=dict(color="#64748b", size=11))
    _apply_chart_layout(fig, "Step Duration", xaxis="Step", yaxis="Duration (s)",
                         height=400,
                         barmode="overlay",
                         legend=dict(orientation="h", yanchor="bottom", y=1.06,
                                     xanchor="center", x=0.5, itemclick="toggleothers"))
    fig.update_layout(margin=dict(t=70))  # extra top margin for spike labels
    fig.update_xaxes(range=[-0.5, len(steps) - 0.5])

    # Context compression markers (red vertical lines)
    if compression_steps:
        for step_idx in compression_steps:
            fig.add_vline(
                x=step_idx, line_dash="dot", line_color="#dc2626", line_width=1.5,
                annotation_text="compressed", annotation_position="top",
                annotation_font=dict(size=8, color="#dc2626"),
            )

    _apply_dark(fig, dark)
    return fig


def build_tool_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Horizontal bar chart of tool call frequency by name, stacked by agent."""
    color_map = build_agent_color_map(steps)
    has_agents = len(color_map) > 1

    # Per-agent per-tool breakdown
    from collections import defaultdict
    agent_tool: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_tools: dict[str, int] = defaultdict(int)
    for s in steps:
        agent = effective_agent(s)
        for tc in s["tool_calls"]:
            name = tc.get("tool_name") or "(unnamed)"
            agent_tool[agent][name] += 1
            all_tools[name] += 1

    if not all_tools:
        fig = _empty_figure(300, "No tool calls recorded in this trajectory.")
        _apply_dark(fig, dark)
        return fig

    sorted_tools = sorted(all_tools.keys(), key=lambda t: all_tools[t])
    display_names = [n if len(n) <= 30 else n[:27] + "..." for n in sorted_tools]

    # Use the same color palette as the Context Growth chart for consistency
    _SESSION_COLORS = ["#3b82f6", "#8b5cf6", "#059669", "#d97706", "#e11d48", "#0891b2"]

    fig = go.Figure()
    if has_agents:
        # Stacked bars per agent, matching Context Growth labeling
        for i, agent_id in enumerate(sorted(color_map.keys(), key=lambda a: color_map[a])):
            if not agent_id:
                label = "main"
            else:
                short = agent_id[:12] if len(agent_id) > 12 else agent_id
                label = f"sub {short}"
            color = _SESSION_COLORS[i % len(_SESSION_COLORS)]
            counts = [agent_tool[agent_id].get(t, 0) for t in sorted_tools]
            fig.add_trace(go.Bar(
                y=display_names, x=counts, orientation="h",
                name=label,
                marker_color=color,
                hovertemplate=f"{label}: " + "%{x} call(s)<extra></extra>",
            ))
    else:
        counts = [all_tools[t] for t in sorted_tools]
        fig.add_trace(go.Bar(
            y=display_names, x=counts, orientation="h", marker_color=CHART_ACCENT,
            text=[str(c) for c in counts], textposition="outside",
            cliponaxis=False,
            customdata=sorted_tools,
            hovertemplate="%{customdata}: %{x} call(s)<extra></extra>",
        ))

    max_label = max(len(n) for n in display_names)
    _apply_chart_layout(
        fig, "Tool Call Frequency" + (" by Agent" if has_agents else ""),
        xaxis="Count",
        height=max(250, 50 * len(sorted_tools)),
        barmode="stack" if has_agents else "relative",
        margin=dict(l=max(140, max_label * 7 + 20), r=60, t=50, b=40),
    )
    if has_agents:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom",
                                       y=1.02, xanchor="center", x=0.5))
    _apply_dark(fig, dark)
    return fig


def build_cache_ratio_chart(rows: list[dict],
                            phases: list[dict] | None = None,
                            dark: bool = False) -> go.Figure:
    """Bar chart of cache-read ratio (%) per step."""
    if not rows:
        fig = _empty_figure(320)
        _apply_dark(fig, dark)
        return fig

    indices = [r["index"] for r in rows]
    ratios = [r["cache_ratio"] * 100 for r in rows]
    colors = ["#92400e" if r["role"] == "assistant" else "#1e40af" for r in rows]
    avg_ratio = statistics.mean(ratios) if ratios else 0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=indices,
        y=ratios,
        marker_color=colors,
        name="Cache Read %",
        hovertemplate="Step %{x}<br>Cache Read: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=avg_ratio, line_dash="dash", line_color="#dc2626",
                  annotation_text=f"Avg: {avg_ratio:.1f}%")
    _apply_chart_layout(fig, "Cache-Read Ratio by Step", xaxis="Step",
                         yaxis="Cache Read (%)", height=320)
    add_phase_overlays(fig, phases, len(rows))
    _apply_dark(fig, dark)
    return fig


def build_efficiency_chart(rows: list[dict],
                           phases: list[dict] | None = None,
                           steps: list[dict] | None = None,
                           dark: bool = False) -> go.Figure:
    """Tokens/sec and tool-wait share per step (behavior efficiency view)."""
    if not rows:
        fig = _empty_figure(340)
        _apply_dark(fig, dark)
        return fig

    indices = [r["index"] for r in rows]
    tok_s = [r["tokens_per_sec"] for r in rows]
    noncache_s = [r["non_cache_per_sec"] for r in rows]
    tool_wait_pct = [r["tool_time_share"] * 100 for r in rows]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=tok_s,
            mode="lines+markers",
            name="Tokens/s",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=6),
            hovertemplate="Step %{x}<br>Tokens/s: %{y:.1f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=noncache_s,
            mode="lines+markers",
            name="Fresh Input tok/s",
            line=dict(color="#059669", width=2, dash="dot"),
            marker=dict(size=5),
            hovertemplate="Step %{x}<br>Fresh Input tok/s: %{y:.1f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=indices,
            y=tool_wait_pct,
            name="Tool-wait %",
            marker_color="#f59e0b",
            opacity=0.28,
            hovertemplate="Step %{x}<br>Tool-wait: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    _apply_chart_layout(
        fig,
        "Per-Step Efficiency — Left axis: tok/s · Right axis: Tool Wait %",
        height=340, margin=dict(t=65, b=40, l=60, r=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    _add_legend_hint(fig)
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="Throughput (tok/s)", secondary_y=False)
    fig.update_yaxes(title_text="Tool Wait (%)", secondary_y=True)
    add_phase_overlays(fig, phases, len(rows))
    if steps:
        _add_agent_regions(fig, steps, build_agent_color_map(steps))
    _apply_dark(fig, dark)
    return fig


def build_analytics_heatmap(
    analytics: list[dict], phases: list[dict] | None = None,
    dark: bool = False,
) -> go.Figure:
    """Heatmap of per-step metrics normalized 0\u20131 per row."""
    if not analytics:
        fig = _empty_figure(300)
        _apply_dark(fig, dark)
        return fig

    metric_keys = [
        "cache_ratio", "tool_time_share", "tok_per_s", "out_in_ratio",
        "non_cache_tok", "idle_before_s",
    ]
    labels = [
        "Cache Read %", "Tool Time Share", "Tok/s", "Out/In Ratio",
        "Fresh Input Tokens", "Idle Gap (s)",
    ]

    z: list[list[float]] = []
    hover: list[list[str]] = []
    for mk, lab in zip(metric_keys, labels):
        row_raw = [a.get(mk) or 0 for a in analytics]
        max_v = max(row_raw) if row_raw else 1
        if max_v == 0:
            max_v = 1
        z.append([v / max_v for v in row_raw])

        row_h: list[str] = []
        for a in analytics:
            v = a.get(mk)
            if v is None:
                row_h.append(f"Step {a['index']} ({a['role']})<br>{lab}: N/A")
            elif mk in ("cache_ratio", "tool_time_share"):
                row_h.append(
                    f"Step {a['index']} ({a['role']})<br>{lab}: {v * 100:.1f}%")
            elif mk in ("tok_per_s", "non_cache_tok"):
                row_h.append(
                    f"Step {a['index']} ({a['role']})<br>{lab}: {v:,.0f}")
            elif mk == "idle_before_s":
                row_h.append(
                    f"Step {a['index']} ({a['role']})<br>{lab}: {v:.2f}s")
            else:
                row_h.append(
                    f"Step {a['index']} ({a['role']})<br>{lab}: {v:.3f}")
        hover.append(row_h)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[str(a["index"]) for a in analytics],
        y=labels,
        hovertext=hover,
        hoverinfo="text",
        colorscale="YlOrRd",
        showscale=True,
    ))

    if phases:
        for phase in phases:
            if phase["start_idx"] > 0:
                fig.add_vline(
                    x=phase["start_idx"] - 0.5,
                    line_dash="dash", line_color="#3b82f6", line_width=2,
                    annotation_text=phase["name"],
                    annotation_position="top",
                )

    _apply_chart_layout(fig, "Behavioral Heatmap (normalized per metric)",
                         xaxis="Step", height=360,
                         margin=dict(t=50, b=40, l=120, r=20))
    _apply_dark(fig, dark)
    return fig


def build_phase_chart(
    phases: list[dict], analytics: list[dict],
    dark: bool = False,
) -> go.Figure:
    """Dual-row stacked horizontal bar: token share and runtime share per phase."""
    if not phases:
        fig = _empty_figure(240)
        _apply_dark(fig, dark)
        return fig

    colors = PHASE_COLORS
    rows = ["Runtime", "Tokens"]  # bottom-to-top in Plotly categorical y

    fig = go.Figure()
    for idx, p in enumerate(phases):
        show_legend = idx == 0 or True  # show all in legend
        color = colors.get(p["name"], "#6b7280")
        step_range = f"steps {p['start_idx']}\u2013{p['end_idx']}"
        # Token share row
        fig.add_trace(go.Bar(
            y=["Tokens"], x=[p["token_share"]], orientation="h",
            name=p["name"],
            marker_color=color,
            text=f"{p['name']} {p['token_share']}%",
            textposition="inside",
            hovertext=f"{p['name']}: {step_range}, {p['token_share']}% of tokens",
            hoverinfo="text",
            legendgroup=p["name"],
            showlegend=True,
        ))
        # Runtime share row
        fig.add_trace(go.Bar(
            y=["Runtime"], x=[p["runtime_share"]], orientation="h",
            name=p["name"],
            marker_color=color,
            text=f"{p['name']} {p['runtime_share']}%",
            textposition="inside",
            hovertext=f"{p['name']}: {step_range}, {p['runtime_share']}% of runtime",
            hoverinfo="text",
            legendgroup=p["name"],
            showlegend=False,
        ))

    _apply_chart_layout(
        fig, "Phase Timeline", xaxis="Share (%)", height=240,
        barmode="stack", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
    )
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    _apply_dark(fig, dark)
    return fig


def build_context_growth_chart(rows: list[dict],
                               phases: list[dict] | None = None,
                               dark: bool = False) -> go.Figure:
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
    fig.add_trace(go.Scatter(
        x=indices, y=cum_input, name="Cumulative Input",
        mode="lines+markers",
        line=dict(color="#2563eb", width=2),
        marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
        hovertemplate="Step %{x}<br>Cumul. Input: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=indices, y=cum_fresh, name="Cumul. Fresh Input",
        mode="lines+markers",
        line=dict(color="#dc2626", width=2, dash="dot"),
        marker=dict(size=4),
        hovertemplate="Step %{x}<br>Cumul. Fresh: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=indices, y=cum_cache, name="Cumul. Cache Read",
        mode="lines+markers",
        line=dict(color="#059669", width=2, dash="dash"),
        marker=dict(size=4),
        hovertemplate="Step %{x}<br>Cumul. Cache: %{y:,}<extra></extra>",
    ))
    _apply_chart_layout(
        fig, "Context Growth (Cumulative Input Tokens)",
        xaxis="Step", yaxis="Tokens (count)", height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    _add_legend_hint(fig)
    add_phase_overlays(fig, phases, len(rows))
    _apply_dark(fig, dark)
    return fig


def build_tool_duration_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Grouped bar chart of avg / p95 / max duration per tool type."""
    from collections import defaultdict

    tool_durs: dict[str, list[float]] = defaultdict(list)
    for s in steps:
        for tc in s["tool_calls"]:
            dur_s = None
            ts = tc.get("time_start")
            te = tc.get("time_end")
            if isinstance(ts, (int, float)) and isinstance(te, (int, float)) and te >= ts:
                dur_s = (te - ts) / 1000.0
            else:
                # Fallback: metadata.totalDurationMs (e.g. Claude Code Agent tool)
                dm = tc.get("duration_ms")
                if dm is None:
                    dm = (tc.get("metadata") or {}).get("totalDurationMs")
                if isinstance(dm, (int, float)) and dm > 0:
                    dur_s = dm / 1000.0
            if dur_s is not None:
                tool_durs[tc["tool_name"]].append(dur_s)

    if not tool_durs:
        fig = _empty_figure(
            180,
            "Tool duration data not available for this format.",
        )
        _apply_dark(fig, dark)
        return fig

    tool_totals = {t: sum(durs) for t, durs in tool_durs.items()}
    sorted_tools = sorted(tool_totals.keys(), key=lambda t: tool_totals[t], reverse=True)
    names = sorted_tools
    totals = [round(tool_totals[t], 2) for t in names]
    grand_total = sum(totals) or 1.0
    pcts = [round(v / grand_total * 100, 1) for v in totals]

    fig = go.Figure()
    display_names = [n if len(n) <= 30 else n[:27] + "..." for n in names]
    fig.add_trace(go.Bar(
        y=display_names, x=totals, orientation="h",
        marker_color=TOKEN_COLORS["fresh_input"],
        text=[f"{v:.1f}s ({p}%)" for v, p in zip(totals, pcts)],
        textposition="outside", cliponaxis=False, showlegend=False,
    ))
    max_label = max(len(n) for n in display_names)
    _apply_chart_layout(
        fig, "Tool Duration Distribution (Total Time)",
        xaxis="Total Duration (s)", height=max(280, 40 * len(names)),
        margin=dict(l=max(140, max_label * 7 + 20), r=90, t=50, b=40),
    )
    _apply_dark(fig, dark)
    return fig


def build_agent_token_chart(agent_summaries: list[dict], dark: bool = False) -> go.Figure:
    """Grouped bar chart showing token breakdown per agent.

    For single-agent sessions, render a horizontal stacked bar that shows the
    one agent's token composition (Fresh Input / Cache Read / Output / Reasoning)
    instead of returning an empty placeholder.
    """
    if not agent_summaries:
        fig = _empty_figure(240, "No agent activity recorded.")
        _apply_dark(fig, dark)
        return fig

    if len(agent_summaries) == 1:
        from .parser import infer_non_cache_input
        a = agent_summaries[0]
        label = a["label"]
        # Use the schema-tolerant inference so Fresh Input doesn't collapse to
        # zero on OpenCode (where ``input_tokens`` is already cache-excluded).
        fresh_input = infer_non_cache_input(
            a["total_tokens"], a["input_tokens"], a["output_tokens"],
            a["reasoning_tokens"], a["cache_read_tokens"],
        )
        cache_read = a["cache_read_tokens"]
        output = max(0, a["output_tokens"] - a["reasoning_tokens"])
        reasoning = a["reasoning_tokens"]
        total = fresh_input + cache_read + output + reasoning

        if total == 0:
            fig = _empty_figure(180, f"No token breakdown available for {label}.")
            _apply_dark(fig, dark)
            return fig

        # Single horizontal stacked bar — segments sized by token category.
        fig = go.Figure()
        for name, value, color in [
            ("Fresh Input", fresh_input, TOKEN_COLORS["fresh_input"]),
            ("Cache Read", cache_read, TOKEN_COLORS["cache_read"]),
            ("Output", output, TOKEN_COLORS["output"]),
            ("Reasoning", reasoning, TOKEN_COLORS["reasoning"]),
        ]:
            if value <= 0:
                continue
            pct = 100 * value / total
            fig.add_trace(go.Bar(
                y=[label], x=[value], orientation="h",
                name=name, marker_color=color,
                text=[f"{name}: {value:,} ({pct:.1f}%)"] if pct >= 4 else [""],
                textposition="inside", insidetextanchor="middle",
                hovertemplate=f"{name}: %{{x:,}} ({pct:.1f}%)<extra></extra>",
            ))
        _apply_chart_layout(
            fig, f"Token Composition — {label}",
            xaxis="Tokens (count)", height=180, barmode="stack",
            legend=dict(orientation="h", yanchor="bottom", y=1.06,
                        xanchor="center", x=0.5),
        )
        fig.update_layout(margin=dict(t=70, l=80, r=20, b=40))
        fig.update_yaxes(showticklabels=False)
        _apply_dark(fig, dark)
        return fig

    labels = [a["label"] for a in agent_summaries]
    has_breakdown = any(a["input_tokens"] > 0 or a["output_tokens"] > 0
                        for a in agent_summaries)

    fig = go.Figure()
    if has_breakdown:
        from .parser import infer_non_cache_input
        # Schema-tolerant fresh-input: handles both Claude Code (input includes
        # cache) and OpenCode (input is already cache-excluded).
        fresh_input = [
            infer_non_cache_input(
                a["total_tokens"], a["input_tokens"], a["output_tokens"],
                a["reasoning_tokens"], a["cache_read_tokens"],
            )
            for a in agent_summaries
        ]
        cache_read = [a["cache_read_tokens"] for a in agent_summaries]
        output = [
            max(0, a["output_tokens"] - a["reasoning_tokens"])
            for a in agent_summaries
        ]
        reasoning = [a["reasoning_tokens"] for a in agent_summaries]
        _add_token_bar_traces(fig, labels, fresh_input, cache_read, output, reasoning)
    else:
        # No token breakdown available — show total tokens per agent
        _SESSION_COLORS = ["#3b82f6", "#8b5cf6", "#059669", "#d97706", "#e11d48", "#0891b2"]
        totals = [a["total_tokens"] for a in agent_summaries]
        colors = [_SESSION_COLORS[i % len(_SESSION_COLORS)] for i in range(len(labels))]
        fig.add_trace(go.Bar(
            x=labels, y=totals, name="Total Tokens",
            marker_color=colors,
            text=[f"{t:,}" for t in totals], textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>%{y:,.0f} tokens<extra></extra>",
        ))

    _apply_chart_layout(
        fig, "Token Breakdown by Agent" if has_breakdown else "Total Tokens by Agent",
        xaxis="Agent", yaxis="Tokens (count)",
        height=320, barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.06,
                    xanchor="center", x=0.5),
    )
    fig.update_layout(margin=dict(t=70))
    _apply_dark(fig, dark)
    return fig


def build_agent_swimlane_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Horizontal swimlane chart showing each agent's active step ranges and token contribution.

    For single-agent sessions we render one lane for that agent — user-prompt
    steps then appear naturally as gaps in the lane. We skip the user/main
    sentinel lane in that case since it would just duplicate the gaps.
    """
    color_map = build_agent_color_map(steps)
    real_agents = [a for a in color_map if a]
    if not real_agents:
        fig = _empty_figure(180, "No agent activity recorded.")
        _apply_dark(fig, dark)
        return fig

    # Hide the user "main" sentinel lane when there's only one real agent —
    # user prompts already appear as gaps in the agent's lane, so an extra
    # lane just duplicates that signal.
    show_main = len(real_agents) > 1

    fig = go.Figure()
    # Group steps by agent and find contiguous runs
    from collections import defaultdict
    agent_runs: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    prev: dict[str, int | None] = {}

    for s in steps:
        agent = _effective_agent(s)
        idx = s["index"]
        tok = s["tokens"]["total"]
        tools = s["tool_call_count"]
        if agent not in prev or prev[agent] is None:
            agent_runs[agent].append((idx, idx, tok, tools))
        else:
            last = agent_runs[agent][-1]
            if idx == last[1] + 1:
                agent_runs[agent][-1] = (last[0], idx, last[2] + tok, last[3] + tools)
            else:
                agent_runs[agent].append((idx, idx, tok, tools))
        prev[agent] = idx

    _SESSION_COLORS = ["#3b82f6", "#8b5cf6", "#059669", "#d97706", "#e11d48", "#0891b2"]
    lane_count = 0
    for i, agent_id in enumerate(sorted(color_map.keys(), key=lambda a: color_map[a])):
        if not agent_id:
            if not show_main:
                continue  # single-agent: skip user lane
            label = "main"
        elif show_main:
            short = agent_id[:12] if len(agent_id) > 12 else agent_id
            label = f"sub {short}"
        else:
            # Single-agent: use the agent name directly (no "sub " prefix
            # since there's no main agent to be sub-ordinate to).
            label = agent_id[:20] if len(agent_id) > 20 else agent_id
        lane_count += 1
        hex_c = _SESSION_COLORS[i % len(_SESSION_COLORS)]
        for start, end, tok, tools in agent_runs.get(agent_id, []):
            width = end - start + 1
            fig.add_trace(go.Bar(
                y=[label], x=[width], orientation="h",
                base=start,
                marker_color=hex_c,
                name=label,
                showlegend=False,
                text=f"{width} steps, {tok:,} tok",
                textposition="inside",
                hovertext=(
                    f"{label}: steps {start}–{end}<br>"
                    f"{tok:,} tokens, {tools} tool calls"
                ),
                hoverinfo="text",
            ))

    _apply_chart_layout(
        fig, "Agent Swimlane", xaxis="Step Index",
        height=max(160, 80 * max(1, lane_count)),
        barmode="overlay",
        margin=dict(l=100, r=20, t=40, b=30),
    )
    _apply_dark(fig, dark)
    return fig


# -- New chart types -------------------------------------------------------

def build_token_allocation_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Donut chart showing total token breakdown by category."""
    if not steps:
        fig = _empty_figure(380, "No steps to display.")
        _apply_dark(fig, dark)
        return fig

    fresh_total = sum(
        infer_non_cache_input(
            total_tokens=s["tokens"]["total"],
            input_tokens=s["tokens"]["input"],
            output_tokens=s["tokens"]["output"],
            reasoning_tokens=s["tokens"]["reasoning"],
            cache_read_tokens=s["tokens"]["cache_read"],
        )
        for s in steps
    )
    cache_total = sum(s["tokens"]["cache_read"] for s in steps)
    reasoning_total = sum(s["tokens"]["reasoning"] for s in steps)
    output_total = sum(max(0, s["tokens"]["output"] - s["tokens"]["reasoning"]) for s in steps)

    labels = []
    values = []
    colors = []
    for name, val, color_key in [
        ("Fresh Input", fresh_total, "fresh_input"),
        ("Cache Read", cache_total, "cache_read"),
        ("Output", output_total, "output"),
        ("Reasoning", reasoning_total, "reasoning"),
    ]:
        if val > 0:
            labels.append(name)
            values.append(val)
            colors.append(TOKEN_COLORS[color_key])

    if not values:
        fig = _empty_figure(180, "Token breakdown not available for this format.")
        _apply_dark(fig, dark)
        return fig

    fig = go.Figure(data=go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,} tokens (%{percent})<extra></extra>",
    ))
    _apply_chart_layout(fig, "Token Allocation", height=380)
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
        fig.add_trace(go.Scatter(
            x=success_x, y=success_y, mode="markers",
            name="Success",
            marker=dict(color=TOOL_OUTCOME_COLORS["success"], size=8, symbol="circle"),
            hovertemplate="Step %{x}<br>%{y}<br>Success<extra></extra>",
        ))
    if failure_x:
        fig.add_trace(go.Scatter(
            x=failure_x, y=failure_y, mode="markers",
            name="Failure",
            marker=dict(color=TOOL_OUTCOME_COLORS["failure"], size=8, symbol="x"),
            hovertemplate="Step %{x}<br>%{y}<br>Failure<extra></extra>",
        ))

    all_tools = sorted(set(success_y + failure_y))
    _apply_chart_layout(
        fig, "Tool Outcome Timeline",
        xaxis="Step", height=max(300, 30 * len(all_tools)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    _apply_dark(fig, dark)
    return fig


def build_error_detail_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Horizontal bar chart of error types with hover showing step indices."""
    if not steps:
        fig = _empty_figure(300)
        _apply_dark(fig, dark)
        return fig

    from collections import defaultdict
    error_types: dict[str, list[int]] = defaultdict(list)
    for s in steps:
        for tc in s["tool_calls"]:
            if tc.get("error") or tc.get("status") == "error":
                err = tc.get("error")
                if isinstance(err, str):
                    # Extract first line as error type
                    err_type = err.split("\n")[0][:60]
                elif isinstance(err, dict):
                    err_type = err.get("type", err.get("message", "Unknown error"))[:60]
                else:
                    err_type = "Unknown error"
                error_types[err_type].append(s["index"])
        if s["error_count"] > 0 and not any(
            tc.get("error") or tc.get("status") == "error" for tc in s["tool_calls"]
        ):
            error_types["Step-level error"].append(s["index"])

    if not error_types:
        fig = _empty_figure(300, "No errors recorded in this trajectory.")
        _apply_dark(fig, dark)
        return fig

    sorted_errors = sorted(error_types.keys(), key=lambda e: len(error_types[e]))
    counts = [len(error_types[e]) for e in sorted_errors]
    hover_texts = [
        f"{e}<br>Steps: {', '.join(str(i) for i in error_types[e][:20])}"
        + ("..." if len(error_types[e]) > 20 else "")
        for e in sorted_errors
    ]
    display_names = [e if len(e) <= 40 else e[:37] + "..." for e in sorted_errors]

    fig = go.Figure(go.Bar(
        y=display_names, x=counts, orientation="h",
        marker_color=ROLE_COLORS["error"],
        text=[str(c) for c in counts], textposition="outside",
        cliponaxis=False,
        hovertext=hover_texts, hoverinfo="text",
        showlegend=False,
    ))
    max_label = max((len(n) for n in display_names), default=10)
    _apply_chart_layout(
        fig, "Error Drill-Down", xaxis="Count",
        height=max(250, 40 * len(sorted_errors)),
        margin=dict(l=max(140, max_label * 7 + 20), r=60, t=50, b=40),
    )
    _apply_dark(fig, dark)
    return fig


def build_cumulative_phase_chart(
    steps: list[dict], phases: list[dict] | None = None,
    dark: bool = False,
) -> go.Figure:
    """Stacked area chart of cumulative tokens by detected phase."""
    if not steps:
        fig = _empty_figure(340)
        _apply_dark(fig, dark)
        return fig

    if not phases:
        phases = [{"name": "Unclassified", "start_idx": 0, "end_idx": len(steps) - 1,
                   "token_share": 100.0, "runtime_share": 100.0}]

    # Build per-step phase assignment
    step_phase: list[str] = ["Unclassified"] * len(steps)
    for p in phases:
        for i in range(p["start_idx"], min(p["end_idx"] + 1, len(steps))):
            step_phase[i] = p["name"]

    indices = list(range(len(steps)))
    phase_names_ordered = []
    seen = set()
    for p in phases:
        if p["name"] not in seen:
            phase_names_ordered.append(p["name"])
            seen.add(p["name"])

    fig = go.Figure()
    for phase_name in phase_names_ordered:
        cum = []
        running = 0
        for i, s in enumerate(steps):
            if step_phase[i] == phase_name:
                running += s.get("tokens", {}).get("total", 0)
            cum.append(running)

        color = PHASE_COLORS.get(phase_name, ROLE_COLORS["default"])
        # Convert hex to rgba for fill
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=indices, y=cum,
            name=phase_name,
            mode="lines",
            line=dict(color=color, width=1),
            fill="tonexty" if phase_names_ordered.index(phase_name) > 0 else "tozeroy",
            fillcolor=f"rgba({r},{g},{b},0.2)",
            stackgroup="one",
            hovertemplate=f"{phase_name}<br>Step %{{x}}<br>Cumul. Tokens: %{{y:,}}<extra></extra>",
        ))

    _apply_chart_layout(
        fig, "Cumulative Token Usage by Phase",
        xaxis="Step", yaxis="Cumulative Tokens", height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    _apply_dark(fig, dark)
    return fig


# -- Label visualization charts -------------------------------------------

from .palette import LABEL_PHASE_COLORS


_LABEL_FONT = dict(size=13)  # consistent font size across label charts


def _add_phase_legend(fig: go.Figure, phases_used: set[str]) -> None:
    """Add invisible trace per phase to create a color legend."""
    canonical = ["understand", "plan", "implement", "debug", "validate", "report"]
    for phase in canonical:
        if phase in phases_used:
            fig.add_trace(go.Bar(
                x=[None], y=[None],
                marker_color=LABEL_PHASE_COLORS.get(phase, "#6b7280"),
                name=phase, showlegend=True,
            ))


def _build_label_bar_chart(
    data: dict[str, int | float],
    color_fn,
    title: str,
    *,
    orientation: str = "v",
    value_format: str = "",
    xaxis: str | None = None,
    yaxis: str | None = None,
    empty_message: str = "No data",
    action_to_phase: dict[str, str] | None = None,
    canonical_order: list[str] | None = None,
) -> go.Figure:
    """Parameterized builder for label bar charts.

    Args:
        data: mapping of label -> numeric value
        color_fn: callable(label) -> hex color string
        title: chart title
        orientation: "v" for vertical, "h" for horizontal
        value_format: format spec for text labels (e.g. ".1f" for durations)
        xaxis/yaxis: axis titles
        empty_message: message for empty chart
        action_to_phase: if provided, adds phase legend and hover customdata
        canonical_order: if provided, orders labels in this canonical order first
    """
    if not data:
        return _empty_figure(380, empty_message)

    # Determine label ordering
    if canonical_order:
        labels = [p for p in canonical_order if p in data]
        labels += [p for p in sorted(data) if p not in canonical_order]
    else:
        labels = sorted(data.keys(), key=lambda k: data[k], reverse=True)

    values = [round(data[k], 1) if isinstance(data[k], float) else data[k] for k in labels]
    colors = [color_fn(k) for k in labels]

    # Format text labels
    if value_format:
        texts = [f"{v:{value_format}}" for v in values]
    else:
        texts = [str(v) for v in values]

    # Build bar trace kwargs
    bar_kwargs: dict = dict(
        marker_color=colors,
        text=texts, textposition="outside",
        cliponaxis=False, textfont=_LABEL_FONT,
        showlegend=False,
    )

    if orientation == "h":
        bar_kwargs.update(y=labels, x=values, orientation="h")
        if action_to_phase:
            phases = [action_to_phase.get(a, "") for a in labels]
            bar_kwargs["customdata"] = phases
            bar_kwargs["hovertemplate"] = "%{y} (%{customdata}): %{x" + (f":{value_format}" if value_format else "") + "}<extra></extra>"
        else:
            bar_kwargs["hovertemplate"] = "%{y}: %{x}<extra></extra>"
    else:
        bar_kwargs.update(x=labels, y=values)
        hover_fmt = f":{value_format}" if value_format else ""
        bar_kwargs["hovertemplate"] = "%{x}: %{y" + hover_fmt + "}<extra></extra>"

    fig = go.Figure(go.Bar(**bar_kwargs))

    # Add phase legend for action charts
    if action_to_phase:
        phases_used = set(action_to_phase.get(a, "") for a in labels)
        _add_phase_legend(fig, phases_used)

    # Layout
    layout_kwargs: dict = dict(margin=dict(t=50, b=40, l=60, r=40))
    if orientation == "h":
        max_label = max((len(a) for a in labels), default=10)
        # When the action chart shows a phase legend above the plot, give the
        # margin enough room for both title and a 2-row wrapped legend without
        # leaving a dead band between them.
        top_margin = 75 if action_to_phase else 50
        layout_kwargs.update(
            height=max(380, 32 * len(labels)),
            margin=dict(l=max(180, max_label * 7 + 30), r=80, t=top_margin, b=40),
        )
        if action_to_phase:
            layout_kwargs.update(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom",
                            y=1.02, xanchor="center", x=0.5),
            )
    else:
        layout_kwargs["height"] = 380

    _apply_chart_layout(fig, title, xaxis=xaxis, yaxis=yaxis, **layout_kwargs)
    return fig


_CANONICAL_PHASES = ["understand", "plan", "implement", "debug", "validate", "report"]


def build_label_phase_count_chart(phase_counts: dict[str, int],
                                   dark: bool = False) -> go.Figure:
    """Bar chart of step counts per phase (level 1)."""
    fig = _build_label_bar_chart(
        phase_counts,
        color_fn=lambda p: LABEL_PHASE_COLORS.get(p, "#6b7280"),
        title="Step Count by Phase",
        xaxis="Phase", yaxis="Steps",
        empty_message="No phase data",
        canonical_order=_CANONICAL_PHASES,
    )
    _apply_dark(fig, dark)
    return fig


def build_label_action_count_chart(action_counts: dict[str, int],
                                    action_to_phase: dict[str, str],
                                    dark: bool = False) -> go.Figure:
    """Bar chart of step counts per action (level 2), colored by parent phase."""
    fig = _build_label_bar_chart(
        action_counts,
        color_fn=lambda a: LABEL_PHASE_COLORS.get(action_to_phase.get(a, ""), "#6b7280"),
        title="Step Count by Action",
        orientation="h", xaxis="Steps",
        empty_message="No action data",
        action_to_phase=action_to_phase,
    )
    _apply_dark(fig, dark)
    return fig


def build_label_phase_duration_chart(phase_durations: dict[str, float],
                                      dark: bool = False) -> go.Figure:
    """Bar chart of summed duration per phase (level 1)."""
    fig = _build_label_bar_chart(
        phase_durations,
        color_fn=lambda p: LABEL_PHASE_COLORS.get(p, "#6b7280"),
        title="Duration by Phase",
        value_format=".1f",
        xaxis="Phase", yaxis="Duration (s)",
        empty_message="No phase duration data",
        canonical_order=_CANONICAL_PHASES,
    )
    _apply_dark(fig, dark)
    return fig


def build_label_action_duration_chart(action_durations: dict[str, float],
                                       action_to_phase: dict[str, str],
                                       dark: bool = False) -> go.Figure:
    """Bar chart of summed duration per action (level 2), colored by parent phase."""
    fig = _build_label_bar_chart(
        action_durations,
        color_fn=lambda a: LABEL_PHASE_COLORS.get(action_to_phase.get(a, ""), "#6b7280"),
        title="Duration by Action",
        orientation="h", value_format=".1f",
        xaxis="Duration (s)",
        empty_message="No action duration data",
        action_to_phase=action_to_phase,
    )
    _apply_dark(fig, dark)
    return fig


def _build_phase_comparison_chart(
    ref_phase_values: dict[str, float],
    cmp_phase_values: dict[str, float],
    ref_label: str,
    cmp_label: str,
    title: str,
    y_title: str,
    value_format: str = ",",
    dark: bool = False,
) -> go.Figure:
    """Grouped bar chart comparing two trajectories' per-phase metrics.

    Phases are plotted in canonical order, including any extras from either side
    appended at the end. Bars are grouped side-by-side per phase.
    """
    phases = list(_CANONICAL_PHASES)
    for p in list(ref_phase_values.keys()) + list(cmp_phase_values.keys()):
        if p not in phases:
            phases.append(p)
    # Filter phases that appear in at least one side
    phases = [p for p in phases
              if ref_phase_values.get(p, 0) or cmp_phase_values.get(p, 0)]
    if not phases:
        fig = _empty_figure(380, "Upload labeled JSONs for both trajectories to see phase comparison")
        _apply_dark(fig, dark)
        return fig

    ref_vals = [ref_phase_values.get(p, 0) for p in phases]
    cmp_vals = [cmp_phase_values.get(p, 0) for p in phases]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=phases, y=ref_vals, name=ref_label,
        marker_color="#1d4ed8",
        text=[format(v, value_format) if v else "" for v in ref_vals],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.add_trace(go.Bar(
        x=phases, y=cmp_vals, name=cmp_label,
        marker_color="#dc2626",
        text=[format(v, value_format) if v else "" for v in cmp_vals],
        textposition="outside",
        cliponaxis=False,
    ))
    # Pad y-axis headroom so outside labels on the tallest bars don't collide
    # with the legend sitting just above the plot area.
    max_val = max([*ref_vals, *cmp_vals, 0])
    _apply_chart_layout(
        fig, title, xaxis="Phase", yaxis=y_title, height=380,
        barmode="group", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
    )
    if max_val > 0:
        fig.update_yaxes(range=[0, max_val * 1.12])
    _apply_dark(fig, dark)
    return fig


def build_phase_count_comparison_chart(
    ref_phase_counts: dict[str, int],
    cmp_phase_counts: dict[str, int],
    ref_label: str = "reference",
    cmp_label: str = "compared",
    dark: bool = False,
) -> go.Figure:
    """Grouped bar chart: step count per phase for reference vs compared."""
    return _build_phase_comparison_chart(
        ref_phase_counts, cmp_phase_counts, ref_label, cmp_label,
        title="Step Count by Phase — Reference vs Compared",
        y_title="Steps", value_format=",", dark=dark,
    )


def build_phase_duration_comparison_chart(
    ref_phase_durations: dict[str, float],
    cmp_phase_durations: dict[str, float],
    ref_label: str = "reference",
    cmp_label: str = "compared",
    dark: bool = False,
) -> go.Figure:
    """Grouped bar chart: total duration (s) per phase for reference vs compared."""
    return _build_phase_comparison_chart(
        ref_phase_durations, cmp_phase_durations, ref_label, cmp_label,
        title="Duration by Phase — Reference vs Compared",
        y_title="Duration (s)", value_format=".1f", dark=dark,
    )


def build_label_timeline_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Horizontal bar timeline — one segment per step, colored by phase, action on hover."""
    if not steps:
        fig = _empty_figure(380, "No step data")
        _apply_dark(fig, dark)
        return fig

    _USER_COLOR = "#d1d5db"       # grey fill for user prompt bar
    _USER_BORDER = "#9ca3af"      # darker border

    step_indices = [s.get("index", i) for i, s in enumerate(steps)]
    step_labels = [str(idx) for idx in step_indices]
    durations = [s.get("duration_s") or 0 for s in steps]
    roles = [s.get("role", "assistant") for s in steps]
    phases = [s.get("phase", "") for s in steps]
    actions = [s.get("action", "") for s in steps]
    max_dur = max(durations) if durations else 1
    y_pos = list(range(len(steps)))

    # --- Assistant bars (only non-user rows; user rows get 0) ---
    asst_durations = [d if r != "user" else 0 for d, r in zip(durations, roles)]
    asst_colors = [LABEL_PHASE_COLORS.get(p, "#6b7280") for p in phases]
    asst_text = [
        f"{a}  ({d:.1f}s)" if r != "user" else ""
        for r, a, d in zip(roles, actions, durations)
    ]
    asst_hover = [
        (f"<b>Step {step_indices[i]}</b><br>"
         f"Phase: {phases[i]}<br>"
         f"Action: {actions[i]}<br>"
         f"Duration: {durations[i]:.1f}s<br>"
         f"Tokens: {s.get('tokens_total', 0):,}")
        if roles[i] != "user" else ""
        for i, s in enumerate(steps)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=asst_durations, y=y_pos, orientation="h",
        marker_color=asst_colors,
        text=asst_text, textposition="outside",
        outsidetextfont=dict(size=11),
        textangle=0, cliponaxis=False,
        hovertext=asst_hover, hoverinfo="text",
        showlegend=False,
    ))

    # --- User prompt bars (small fixed-width marker + label) ---
    user_bar_width = max_dur * 0.04
    user_indices = [i for i, r in enumerate(roles) if r == "user"]
    if user_indices:
        user_hover = [
            f"<b>Step {step_indices[i]} — user prompt</b><br>"
            f"{steps[i].get('text_preview', '')[:120]}"
            for i in user_indices
        ]
        fig.add_trace(go.Bar(
            x=[user_bar_width] * len(user_indices),
            y=user_indices, orientation="h",
            width=0.8,  # match agent bar height
            marker_color=_USER_COLOR,
            marker_line=dict(color=_USER_BORDER, width=1),
            hovertext=user_hover, hoverinfo="text",
            showlegend=False,
        ))
        for yi in user_indices:
            preview = steps[yi].get("text_preview", "")
            # Show a short snippet of the user message
            snippet = preview[:60].replace("<", "&lt;").replace(">", "&gt;")
            if len(preview) > 60:
                snippet += "…"
            label = f"<i>user prompt</i>: {snippet}" if snippet else "<i>user prompt</i>"
            fig.add_annotation(
                x=user_bar_width, y=yi,
                text=label,
                showarrow=False, xanchor="left", xshift=6,
                font=dict(size=10, color="#6b7280"),
            )

    # --- Legend entries (one per phase + user) ---
    seen: set[str] = set()
    for role, phase in zip(roles, phases):
        key = "user" if role == "user" else phase
        if key and key not in seen:
            seen.add(key)
            fig.add_trace(go.Bar(
                x=[None], y=[None], orientation="h",
                marker_color=_USER_COLOR if key == "user"
                    else LABEL_PHASE_COLORS.get(key, "#6b7280"),
                name="user prompt" if key == "user" else key,
                showlegend=True,
            ))

    chart_height = max(450, 28 * len(steps))
    top_margin = 90  # title (~25 px) + small gap + legend (~25 px) + breathing room
    # Position both title and legend in container (figure-relative) coords with
    # constant pixel offsets so the title sits on top with the legend just below
    # it, regardless of how tall the chart is. yref="container" keeps both in
    # the same coordinate system so they don't collide.
    title_offset_px = 8     # title top, ~8 px below figure top
    legend_offset_px = 45   # legend top, ~45 px below figure top (just under title)
    title_y = 1 - title_offset_px / chart_height
    legend_y = 1 - legend_offset_px / chart_height
    _apply_chart_layout(
        fig, "Step Timeline (colored by phase, labeled by action)",
        xaxis="Duration (s)", yaxis="Step",
        height=chart_height,
        margin=dict(l=70, r=200, t=top_margin, b=40),
        showlegend=True, barmode="overlay",
        legend=dict(orientation="h", yref="container", yanchor="top",
                    y=legend_y, xanchor="center", x=0.5),
    )
    # Pin the title above the legend in container coords.
    fig.update_layout(title=dict(
        text="Step Timeline (colored by phase, labeled by action)",
        y=title_y, yref="container", yanchor="top",
        x=0.5, xanchor="center",
    ))
    fig.update_yaxes(
        tickvals=y_pos, ticktext=step_labels,
        autorange="reversed",
    )
    _apply_dark(fig, dark)
    return fig


# -- File Interaction Timeline -----------------------------------------------

_FILE_INTERACTION_COLORS = {
    "read": "#3b82f6",    # blue
    "write": "#10b981",   # green
    "search": "#f59e0b",  # orange
}

_FILE_INTERACTION_SYMBOLS = {
    "read": "circle",
    "write": "diamond",
    "search": "triangle-up",
}


def build_file_interaction_chart(
    interactions: list[dict],
    target_files: set[str] | None = None,
    dark: bool = False,
) -> go.Figure:
    """Build a Plotly scatter chart of file interactions across steps.

    x=step index, y=file path (categorical), color=interaction type.
    Target files are highlighted with a distinct marker border.
    """
    import os

    fig = go.Figure()
    if not interactions:
        _apply_chart_layout(fig, "File Interaction Timeline (no data)")
        _apply_dark(fig, dark)
        return fig

    target_files = target_files or set()

    # Normalize target files for matching
    def _norm(p: str) -> str:
        return os.path.normpath(p) if p else p

    norm_targets = {_norm(t) for t in target_files}

    # Group by interaction type for legend
    for itype, color in _FILE_INTERACTION_COLORS.items():
        group = [i for i in interactions if i["type"] == itype]
        if not group:
            continue

        x = [i["step"] for i in group]
        y = [i["path"] for i in group]
        is_target = [_norm(i["path"]) in norm_targets for i in group]
        hover = [
            f"Step {i['step']}: {i['tool']}({i['path']})"
            for i in group
        ]
        border_colors = [
            "#dc2626" if t else color for t in is_target
        ]
        border_widths = [2 if t else 0 for t in is_target]

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            name=itype,
            marker=dict(
                color=color, size=9,
                symbol=_FILE_INTERACTION_SYMBOLS.get(itype, "circle"),
                line=dict(color=border_colors, width=border_widths),
            ),
            hovertext=hover,
            hoverinfo="text",
        ))

    # Dynamic height — scale with file count so all files are visible.
    # Drag-to-pan on y-axis is still enabled for very large trajectories.
    unique_files = len({i["path"] for i in interactions})
    chart_height = max(350, 25 * unique_files + 80)

    # Size the left margin to fit the longest path label in full (no truncation).
    max_path_len = max((len(i["path"]) for i in interactions), default=20)
    left_margin = max(150, max_path_len * 7)

    # Force the figure wider than its container when labels + plot area would
    # otherwise overflow, so the surrounding HTML wrapper can scroll left↔right.
    # Minimum plot-area width of 600px keeps the step axis readable even when
    # the label margin is huge (e.g., deep Windows paths like the screenshot).
    fig_width = left_margin + 600 + 20  # left margin + plot area + right margin

    _apply_chart_layout(
        fig, "File Interaction Timeline",
        xaxis="Step", yaxis="File",
        height=chart_height,
        margin=dict(l=left_margin, r=20, t=50, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
    )
    fig.update_layout(width=fig_width, autosize=False)
    # Enable drag-to-pan/zoom on both axes so users can also navigate via mouse.
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    fig.update_layout(dragmode="pan")
    _apply_dark(fig, dark)
    return fig


# -- Score Gauge Chart -------------------------------------------------------

def build_score_gauge_chart(
    composite_score: float | None,
    verdict: str = "n/a",
    dark: bool = False,
) -> go.Figure:
    """Build a Plotly gauge (indicator) for the composite trajectory quality score."""
    fig = go.Figure()

    if composite_score is None:
        fig.add_annotation(
            text="N/A", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=28, color="#9ca3af"),
        )
        fig.update_layout(template=_TPL, height=220, margin=dict(t=30, b=10, l=30, r=30))
        _apply_dark(fig, dark)
        return fig

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=composite_score,
        number={"suffix": "/100", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#1d4ed8"},
            "steps": [
                {"range": [0, 40], "color": "#fee2e2"},
                {"range": [40, 70], "color": "#fef3c7"},
                {"range": [70, 100], "color": "#dcfce7"},
            ],
            "threshold": {
                "line": {"color": "#0f172a", "width": 3},
                "thickness": 0.8,
                "value": composite_score,
            },
        },
    ))

    fig.update_layout(
        template=_TPL,
        height=220,
        margin=dict(t=30, b=10, l=30, r=30),
    )
    _apply_dark(fig, dark)
    return fig


# -- Additional chart builders -----------------------------------------------


def build_tool_frequency_heatmap(steps: list[dict],
                                 dark: bool = False) -> go.Figure:
    """Heatmap of tool call counts: x = step index, y = tool name."""
    from collections import defaultdict

    # Gather per-step per-tool counts
    tool_names_set: set[str] = set()
    step_tool_counts: list[dict[str, int]] = []
    for s in steps:
        counts: dict[str, int] = defaultdict(int)
        for tc in s.get("tool_calls", []):
            name = tc.get("tool_name") or tc.get("name") or "(unnamed)"
            counts[name] += 1
            tool_names_set.add(name)
        step_tool_counts.append(counts)

    if not tool_names_set:
        fig = _empty_figure(400, "No tool calls in this trajectory")
        _apply_dark(fig, dark)
        return fig

    tool_names = sorted(tool_names_set)
    z = []
    for tool in tool_names:
        row = [step_tool_counts[i].get(tool, 0) for i in range(len(steps))]
        z.append(row)

    fig = go.Figure(go.Heatmap(
        x=list(range(len(steps))),
        y=tool_names,
        z=z,
        colorscale="Blues",
        hovertemplate="Step %{x}<br>%{y}<br>Calls: %{z}<extra></extra>",
    ))

    _apply_chart_layout(
        fig, "Tool Call Frequency Heatmap",
        xaxis="Step", yaxis="Tool",
        height=400,
        margin=dict(t=50, b=40, l=max(140, max(len(n) for n in tool_names) * 7 + 20), r=20),
    )
    _apply_dark(fig, dark)
    return fig


def build_token_burndown_chart(steps: list[dict],
                               dark: bool = False) -> go.Figure:
    """Stacked area chart of cumulative token usage over steps.

    Shows four token categories (fresh input, cache read, output, reasoning)
    as stacked areas, with a horizontal budget reference line at the total.
    """
    if not steps:
        fig = _empty_figure(380)
        _apply_dark(fig, dark)
        return fig

    indices = list(range(len(steps)))

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
    cache_read = [s["tokens"]["cache_read"] for s in steps]
    output = [s["tokens"]["output"] for s in steps]
    reasoning = [s["tokens"]["reasoning"] for s in steps]

    # Compute cumulative sums
    cum_fresh, cum_cache, cum_output, cum_reasoning = [], [], [], []
    total_f = total_c = total_o = total_r = 0
    for i in range(len(steps)):
        total_f += fresh_input[i]
        total_c += cache_read[i]
        total_o += output[i]
        total_r += reasoning[i]
        cum_fresh.append(total_f)
        cum_cache.append(total_c)
        cum_output.append(total_o)
        cum_reasoning.append(total_r)

    budget = total_f + total_c + total_o + total_r

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=indices, y=cum_fresh, name="Fresh Input",
        mode="lines", stackgroup="tokens",
        line=dict(width=0), fillcolor=TOKEN_COLORS["fresh_input"],
        hovertemplate="Step %{x}<br>Fresh Input: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=indices, y=cum_cache, name="Cache Read",
        mode="lines", stackgroup="tokens",
        line=dict(width=0), fillcolor=TOKEN_COLORS["cache_read"],
        hovertemplate="Step %{x}<br>Cache Read: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=indices, y=cum_output, name="Output",
        mode="lines", stackgroup="tokens",
        line=dict(width=0), fillcolor=TOKEN_COLORS["output"],
        hovertemplate="Step %{x}<br>Output: %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=indices, y=cum_reasoning, name="Reasoning",
        mode="lines", stackgroup="tokens",
        line=dict(width=0), fillcolor=TOKEN_COLORS["reasoning"],
        hovertemplate="Step %{x}<br>Reasoning: %{y:,.0f}<extra></extra>",
    ))

    # Budget reference line
    fig.add_hline(
        y=budget, line_dash="dash", line_color="#dc2626", line_width=1.5,
        annotation_text=f"Total: {budget:,.0f}",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#dc2626"),
    )

    _apply_chart_layout(
        fig, "Cumulative Token Burndown",
        xaxis="Step", yaxis="Cumulative Tokens",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    _add_legend_hint(fig)
    _apply_dark(fig, dark)
    return fig


def build_error_recovery_chart(steps: list[dict],
                               dark: bool = False) -> go.Figure:
    """Histogram of error recovery latency (steps between error and next success).

    For each step with error_count > 0, find the next step that has
    role == "assistant", error_count == 0, and at least one tool call.
    The latency is the step-index difference.
    """
    if not steps:
        fig = _empty_figure(380, "No errors detected in this trajectory")
        _apply_dark(fig, dark)
        return fig

    latencies: list[int] = []
    for i, s in enumerate(steps):
        if s["error_count"] > 0:
            # Scan forward for next successful tool-bearing step
            for j in range(i + 1, len(steps)):
                candidate = steps[j]
                if (candidate["role"] == "assistant"
                        and candidate["error_count"] == 0
                        and candidate["tool_call_count"] > 0):
                    latencies.append(j - i)
                    break

    if not latencies:
        fig = _empty_figure(380, "No errors detected in this trajectory")
        _apply_dark(fig, dark)
        return fig

    mean_lat = statistics.mean(latencies)
    median_lat = statistics.median(latencies)

    fig = go.Figure(go.Histogram(
        x=latencies,
        marker_color=CHART_ACCENT,
        hovertemplate="Latency: %{x} step(s)<br>Count: %{y}<extra></extra>",
    ))

    # Annotate mean and median
    fig.add_vline(
        x=mean_lat, line_dash="dash", line_color="#dc2626", line_width=1.5,
        annotation_text=f"Mean: {mean_lat:.1f}",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#dc2626"),
    )
    fig.add_vline(
        x=median_lat, line_dash="dot", line_color="#059669", line_width=1.5,
        annotation_text=f"Median: {median_lat:.1f}",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#059669"),
    )

    _apply_chart_layout(
        fig, "Error Recovery Latency",
        xaxis="Steps to Recovery", yaxis="Frequency",
        height=380,
    )
    _apply_dark(fig, dark)
    return fig


def build_phase_radar_chart(steps: list[dict],
                            phases: list[dict],
                            dark: bool = False) -> go.Figure:
    """Radar chart comparing phases on tokens_per_step, tool_success_rate, cache_hit_rate.

    *phases* is a list of dicts with keys: name, start_idx, end_idx.
    Falls back to a simple bar chart when only one phase is provided.
    """
    if not phases or not steps:
        fig = _empty_figure(400, "No phase data available")
        _apply_dark(fig, dark)
        return fig

    metrics = {"tokens_per_step": [], "tool_success_rate": [], "cache_hit_rate": []}
    phase_names: list[str] = []

    for p in phases:
        start = p["start_idx"]
        end = p["end_idx"]
        phase_steps = steps[start:end + 1]
        n_steps = len(phase_steps) or 1
        phase_names.append(p["name"])

        # tokens_per_step
        total_tokens = sum(s["tokens"]["total"] for s in phase_steps)
        metrics["tokens_per_step"].append(total_tokens / n_steps)

        # tool_success_rate
        total_tool_calls = sum(s["tool_call_count"] for s in phase_steps)
        successful_tool_calls = sum(
            s["tool_call_count"] for s in phase_steps if s["error_count"] == 0
        )
        if total_tool_calls > 0:
            metrics["tool_success_rate"].append(successful_tool_calls / total_tool_calls)
        else:
            metrics["tool_success_rate"].append(1.0)

        # cache_hit_rate
        cache_read_total = sum(s["tokens"]["cache_read"] for s in phase_steps)
        fresh_input_total = sum(
            infer_non_cache_input(
                total_tokens=s["tokens"]["total"],
                input_tokens=s["tokens"]["input"],
                output_tokens=s["tokens"]["output"],
                reasoning_tokens=s["tokens"]["reasoning"],
                cache_read_tokens=s["tokens"]["cache_read"],
            )
            for s in phase_steps
        )
        denom = cache_read_total + fresh_input_total
        metrics["cache_hit_rate"].append(cache_read_total / denom if denom > 0 else 0.0)

    # Normalize each metric to 0-1 across phases
    for key in metrics:
        vals = metrics[key]
        max_val = max(vals) if vals else 1
        if max_val > 0:
            metrics[key] = [v / max_val for v in vals]

    metric_labels = ["Tokens/Step", "Tool Success Rate", "Cache Hit Rate"]

    # Single-phase fallback: bar chart
    if len(phases) == 1:
        fig = go.Figure(go.Bar(
            x=metric_labels,
            y=[metrics["tokens_per_step"][0],
               metrics["tool_success_rate"][0],
               metrics["cache_hit_rate"][0]],
            marker_color=CHART_ACCENT,
            text=[f"{v:.2f}" for v in [metrics["tokens_per_step"][0],
                                        metrics["tool_success_rate"][0],
                                        metrics["cache_hit_rate"][0]]],
            textposition="outside",
        ))
        _apply_chart_layout(
            fig, f"Phase Metrics: {phase_names[0]}",
            yaxis="Normalized Value", height=400,
        )
        _apply_dark(fig, dark)
        return fig

    # Multi-phase radar chart
    fig = go.Figure()
    for i, name in enumerate(phase_names):
        r_vals = [
            metrics["tokens_per_step"][i],
            metrics["tool_success_rate"][i],
            metrics["cache_hit_rate"][i],
        ]
        # Close the polygon
        r_vals_closed = r_vals + [r_vals[0]]
        theta_closed = metric_labels + [metric_labels[0]]

        color = PHASE_COLORS.get(name, CHART_ACCENT)
        fig.add_trace(go.Scatterpolar(
            r=r_vals_closed,
            theta=theta_closed,
            fill="toself",
            name=name,
            line=dict(color=color),
            opacity=0.6,
        ))

    fig.update_layout(
        template=_TPL,
        height=400,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
        margin=dict(t=60, b=40, l=60, r=60),
        title="Phase Comparison Radar",
    )
    _apply_dark(fig, dark)
    return fig


# -- Plan Progress Timeline -----------------------------------------------

def build_plan_timeline_chart(
    plan_history: list[dict],
    plan_metrics: dict,
    dark: bool = False,
) -> go.Figure:
    """Horizontal Gantt-style chart showing plan item progress over steps.

    Each bar represents a todo item, spanning from first in_progress to completed.
    Stalled items are highlighted in red.
    """
    items = plan_metrics.get("items", [])
    if not items:
        fig = _empty_figure(300, "No plan data available")
        _apply_dark(fig, dark)
        return fig

    stalled_contents = {s["content"] for s in plan_metrics.get("stalled", [])}

    # Sort by start_step so the timeline reads top→bottom in chronological order.
    # Plotly places the first-added bar at the *bottom* of a horizontal bar chart,
    # so we iterate in reverse (latest start first) to get earliest-at-top.
    # Items without a start_step (never started) fall to the bottom.
    def _sort_key(it: dict) -> tuple[int, int]:
        s = it.get("start_step")
        return (0, s) if s is not None else (1, 0)
    items = sorted(items, key=_sort_key, reverse=True)

    fig = go.Figure()
    y_labels = []
    for i, item in enumerate(items):
        content = item["content"]
        short = content[:40] + "..." if len(content) > 40 else content
        y_labels.append(short)

        start = item.get("start_step")
        end = item.get("end_step")
        is_stalled = content in stalled_contents

        if start is not None and end is not None:
            color = "#dc2626" if is_stalled else "#059669"
            width = end - start
            fig.add_trace(go.Bar(
                y=[short], x=[max(width, 1)], orientation="h",
                base=start, marker_color=color, showlegend=False,
                hovertext=f"{content}<br>Steps {start}→{end} ({width} steps)",
                hoverinfo="text",
                text=f"{width} steps", textposition="inside",
            ))
        elif start is not None:
            fig.add_trace(go.Bar(
                y=[short], x=[1], orientation="h",
                base=start, marker_color="#d97706", showlegend=False,
                hovertext=f"{content}<br>Started at step {start}, not completed",
                hoverinfo="text",
                text="stalled", textposition="inside",
            ))
        elif end is not None:
            # Item went straight to "completed" without ever being marked
            # in_progress (common for OpenCode todo lists). Show a thin marker
            # at the completion step so the row isn't blank.
            fig.add_trace(go.Bar(
                y=[short], x=[1], orientation="h",
                base=max(0, end - 1), marker_color="#3b82f6", showlegend=False,
                hovertext=f"{content}<br>Completed at step {end} (no in_progress recorded)",
                hoverinfo="text",
                text="completed", textposition="inside",
            ))
        else:
            # Never started or completed — show a grey placeholder at x=0 so
            # the row appears in the y-axis instead of being silently dropped.
            fig.add_trace(go.Bar(
                y=[short], x=[0.5], orientation="h",
                base=0, marker_color="#9ca3af", showlegend=False,
                hovertext=f"{content}<br>Never started",
                hoverinfo="text",
                text="not started", textposition="outside",
                cliponaxis=False,
            ))

    # Compact bar height (22px each + 90px padding) keeps the chart from
    # ballooning vertically when only a few items have explicit timing.
    chart_height = max(220, 22 * len(items) + 90)

    _apply_chart_layout(
        fig, "Plan Progress Timeline",
        xaxis="Step", height=chart_height,
        margin=dict(l=max(200, max((len(l) for l in y_labels), default=10) * 6 + 20), r=40, t=50, b=40),
    )
    # Tighter bar gap so individual items don't visually balloon when the
    # surrounding container is wide (e.g., right column of a 2-col layout).
    fig.update_layout(bargap=0.25)
    _apply_dark(fig, dark)
    return fig


# -- Error Classification Chart --------------------------------------------

def build_error_classification_chart(
    steps: list[dict],
    dark: bool = False,
) -> go.Figure:
    """Horizontal bar chart of error types classified from tool output."""
    from collections import Counter
    from .loaders import _classify_tool_error
    error_types: Counter = Counter()
    error_steps: dict[str, list[int]] = {}

    for i, s in enumerate(steps):
        for tc in s.get("tool_calls", []):
            etype = tc.get("error_type")
            # Fallback for formats that don't pre-classify (OpenCode, Claude Code):
            # run the same pattern matcher used by the CodeArts loader against the
            # raw error message so we still get a non-empty chart.
            if not etype:
                err_text = tc.get("error") or ""
                if err_text:
                    etype = _classify_tool_error(err_text) or "tool_error"
            if etype:
                error_types[etype] += 1
                error_steps.setdefault(etype, []).append(s.get("index", i))

    if not error_types:
        fig = _empty_figure(250, "No classified tool errors")
        _apply_dark(fig, dark)
        return fig

    _LABELS = {
        "platform_error":   "Platform Error",
        "permission_error": "Permission / Policy",
        "missing_file":     "Missing File",
        "bad_input":        "Bad Input",
        "tool_error":       "Other Tool Error",
    }
    _COLORS = {
        "platform_error":   "#dc2626",
        "permission_error": "#d97706",
        "missing_file":     "#6366f1",
        "bad_input":        "#0891b2",
        "tool_error":       "#6b7280",
    }

    sorted_types = sorted(error_types.keys(), key=lambda t: error_types[t])
    labels = [_LABELS.get(t, t) for t in sorted_types]
    counts = [error_types[t] for t in sorted_types]
    colors = [_COLORS.get(t, CHART_ACCENT) for t in sorted_types]
    hover = [
        f"{_LABELS.get(t, t)}: {error_types[t]}<br>Steps: {', '.join(str(s) for s in error_steps.get(t, [])[:10])}"
        for t in sorted_types
    ]

    fig = go.Figure(go.Bar(
        y=labels, x=counts, orientation="h",
        marker_color=colors, showlegend=False,
        text=[str(c) for c in counts], textposition="outside",
        hovertext=hover, hoverinfo="text",
    ))
    _apply_chart_layout(
        fig, "Tool Error Classification",
        xaxis="Count", height=max(200, 40 * len(sorted_types)),
        margin=dict(l=max(140, max((len(l) for l in labels), default=10) * 7 + 20), r=60, t=50, b=40),
    )
    _apply_dark(fig, dark)
    return fig


# -- Task Mode Breakdown (#2) ------------------------------------------------

def build_task_mode_chart(
    steps: list[dict],
    trajectory: list[dict],
    dark: bool = False,
) -> go.Figure:
    """Bar chart of task mode (menuTask) distribution across steps."""
    from collections import Counter
    modes: Counter = Counter()
    for i, s in enumerate(steps):
        if s.get("role") != "assistant":
            continue
        info = trajectory[i].get("info", {}) if i < len(trajectory) else {}
        mode = info.get("menuTask", "")
        if mode:
            modes[mode] += 1

    if not modes:
        fig = _empty_figure(250, "No task mode data available")
        _apply_dark(fig, dark)
        return fig

    labels = sorted(modes.keys(), key=lambda m: modes[m], reverse=True)
    counts = [modes[m] for m in labels]

    fig = go.Figure(go.Bar(
        x=labels, y=counts, marker_color=CHART_ACCENT,
        text=[str(c) for c in counts], textposition="outside",
        hovertemplate="%{x}: %{y} steps<extra></extra>",
    ))
    _apply_chart_layout(fig, "Task Mode Distribution", xaxis="Mode", yaxis="Steps", height=300)
    _apply_dark(fig, dark)
    return fig


# -- Duration vs True Cost Gap (#6) ------------------------------------------

def build_duration_gap_chart(
    steps: list[dict],
    subagent_sessions: list[dict],
    dark: bool = False,
) -> go.Figure:
    """Bar chart overlaying reported step duration with sub-agent wait gap.

    For steps that spawn sub-agents, the reported duration is just the API call,
    but the true elapsed time includes the sub-agent's full run.
    """
    if not subagent_sessions:
        fig = _empty_figure(250, "No sub-agent sessions — no duration gap to show")
        _apply_dark(fig, dark)
        return fig

    spawn_steps = {s["spawn_step"]: s for s in subagent_sessions if s.get("spawn_step") is not None}
    if not spawn_steps:
        fig = _empty_figure(250, "No sub-agent spawn steps identified")
        _apply_dark(fig, dark)
        return fig

    indices = []
    reported = []
    actual = []
    labels = []

    for s in steps:
        idx = s.get("index", 0)
        if idx in spawn_steps:
            session = spawn_steps[idx]
            rep_dur = s.get("duration") or 0
            # Actual elapsed = gap from this step's start to next non-sub-agent step
            act_dur = session.get("total_duration", rep_dur)
            indices.append(idx)
            reported.append(rep_dur)
            actual.append(act_dur)
            labels.append(f"Step {idx} → sub-agent {session.get('session_id', '')[:8]}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=actual, name="Sub-agent duration",
        marker_color="#8b5cf6", opacity=0.5,
        hovertemplate="%{x}<br>Sub-agent: %{y:.1f}s<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=labels, y=reported, name="Reported step duration",
        marker_color=CHART_ACCENT,
        hovertemplate="%{x}<br>Reported: %{y:.1f}s<extra></extra>",
    ))
    _apply_chart_layout(
        fig, "Duration vs True Cost (Sub-Agent Spawn Steps)",
        yaxis="Duration (s)", height=300, barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    _apply_dark(fig, dark)
    return fig


# -- Context Growth (CodeArts cumulative) ------------------------------------

def build_context_growth_ca_chart(
    steps: list[dict],
    trajectory: list[dict],
    context_limit: int | None = None,
    dark: bool = False,
) -> go.Figure:
    """Line chart of cumulative context size over steps for CodeArts trajectories.

    Uses the raw cumulative total_tokens (before delta conversion) to show
    how the LLM's context window fills up.  Sessions are identified by
    ``sessionID`` (derived from ``chatId``); within each session, a drop in
    cumulative tokens flags a **compression event**.
    """
    if not trajectory:
        fig = _empty_figure(340, "No context growth data available")
        _apply_dark(fig, dark)
        return fig

    # ── 1. Extract per-step data points ──────────────────────────────
    points: list[dict] = []
    for i, entry in enumerate(trajectory):
        info = entry.get("info", {})
        if info.get("role") != "assistant":
            continue
        # Original cumulative value lives in _codearts_raw.total_tokens;
        # the delta conversion overwrote tokens["total"].
        raw_msg = entry.get("_codearts_raw", {})
        cumulative = raw_msg.get("total_tokens", 0) if isinstance(raw_msg, dict) else 0
        if cumulative == 0:
            cumulative = info.get("tokens", {}).get("total", 0)
        # Skip zero-token entries (tool dispatch steps with no LLM inference)
        if cumulative == 0:
            continue
        session_id = info.get("sessionID", "")
        is_sub = info.get("isSubAgent", False)
        step_idx = steps[i].get("index", i) if i < len(steps) else i
        # For CodeArts sub-agents, effective_agent returns the full session_id
        # from parsed steps — use it for consistent labeling with agent breakdown
        step_session = steps[i].get("session_id", "") if i < len(steps) else ""
        points.append({
            "step": step_idx,
            "cumulative": cumulative,
            "session_id": session_id,
            "is_sub": is_sub,
            "step_session_id": step_session,
        })

    if not points:
        fig = _empty_figure(340, "No context growth data available")
        _apply_dark(fig, dark)
        return fig

    # ── 2. Group by sessionID (preserves appearance order) ───────────
    from collections import OrderedDict
    session_map: OrderedDict[str, dict] = OrderedDict()
    for p in points:
        sid = p["session_id"]
        if sid not in session_map:
            session_map[sid] = {
                "x": [], "y": [], "is_sub": p["is_sub"], "id": sid,
                "step_session_id": p.get("step_session_id", ""),
            }
        session_map[sid]["x"].append(p["step"])
        session_map[sid]["y"].append(p["cumulative"])

    # ── 3. Detect compression events (token drops within a session) ──
    compressions: list[dict] = []
    for sess in session_map.values():
        prev = 0
        for x, y in zip(sess["x"], sess["y"]):
            if y < prev and prev > 0:
                compressions.append({"step": x, "from": prev, "to": y})
            prev = y

    # ── 4. Plot each session as its own trace ────────────────────────
    fig = go.Figure()
    _SESSION_COLORS = ["#3b82f6", "#8b5cf6", "#059669", "#d97706", "#e11d48", "#0891b2"]
    main_idx = 0
    sub_idx = 0
    n_main = sum(1 for s in session_map.values() if not s["is_sub"])
    n_sub = sum(1 for s in session_map.values() if s["is_sub"])
    for i, sess in enumerate(session_map.values()):
        if not sess["x"]:
            continue
        color = _SESSION_COLORS[i % len(_SESSION_COLORS)]
        if sess["is_sub"]:
            sub_idx += 1
            # Use step_session_id (matches agent breakdown grid label)
            ssid = sess.get("step_session_id", "") or sess["id"]
            short = ssid[:12] if ssid else f"#{sub_idx}"
            label = f"sub {short}"
        else:
            main_idx += 1
            label = f"main {main_idx}" if n_main > 1 else "main"
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=sess["x"], y=sess["y"],
            mode="lines+markers", name=label,
            line=dict(color=color, width=2),
            marker=dict(size=3),
            fill="tozeroy",
            fillcolor=f"rgba({r},{g},{b},0.08)",
            hovertemplate=f"Step %{{x}}<br>Context: %{{y:,.0f}} tokens<br>{label}<extra></extra>",
        ))

    # ── 5. Mark compression events ───────────────────────────────────
    if compressions:
        fig.add_trace(go.Scatter(
            x=[c["step"] for c in compressions],
            y=[c["to"] for c in compressions],
            mode="markers",
            name="Compression",
            marker=dict(symbol="triangle-down", size=10, color="#f59e0b",
                        line=dict(width=1, color="#92400e")),
            hovertemplate="Step %{x}<br>Compressed to %{y:,.0f} tokens<extra>compression</extra>",
        ))

    # Context limit line
    if context_limit and context_limit > 0:
        fig.add_hline(
            y=context_limit, line_dash="dash", line_color="#dc2626", line_width=1.5,
            annotation_text=f"Context limit: {context_limit:,}",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#dc2626"),
        )

    _apply_chart_layout(
        fig, "Context Growth (Cumulative Tokens per Session)",
        xaxis="Step", yaxis="Cumulative Tokens", height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    _apply_dark(fig, dark)
    return fig
