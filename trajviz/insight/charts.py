"""Plotly chart builders for trajectory visualization."""

import statistics
from collections import Counter, defaultdict
from collections.abc import Callable

# Pre-import pandas before plotly to avoid circular import error
# in plotly's basevalidators when running inside Gradio async threads.
try:
    import pandas  # noqa: F401
except ImportError:
    pass

import plotly.graph_objects as go

from .parser import infer_non_cache_input
from .metrics import effective_agent
from .palette import (
    TOKEN_COLORS,
    SESSION_COLORS,
    LABEL_PHASE_COLORS,
    AGENT_COLORS,
    ROLE_COLORS,
    TOOL_OUTCOME_COLORS,
    CHART_ACCENT,
    PLOTLY_DARK_TEMPLATE,
)


def _apply_dark(fig: go.Figure, dark: bool) -> go.Figure:
    """Apply dark-mode layout overrides when *dark* is True."""
    if dark:
        fig.update_layout(**PLOTLY_DARK_TEMPLATE)
    return fig


def _session_primary_agents(steps: list[dict]) -> dict[str, str]:
    """Most common non-compaction agent name per session_id."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for s in steps:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("session_id") or "")
        name = str(s.get("agent") or "").strip()
        if not sid or not name:
            continue
        if name == "compaction" or s.get("role") == "compaction" or s.get("is_compaction_checkpoint"):
            continue
        counts[sid][name] += 1
    return {sid: c.most_common(1)[0][0] for sid, c in counts.items() if c}


def _timeline_context(steps: list[dict]) -> tuple[bool, bool, dict[str, str]]:
    """Detect how to split agents on swimlane / run-group timelines.

    Returns ``(multi_session, use_agent_names, session_primary_agents)``:
    - multi_session: several ``session_id`` values (OpenCode/CodeArts subagents
      often lack ``isSubAgent`` tags — group by session instead)
    - use_agent_names: ``effective_agent`` is empty for everyone but ``agent``
      field varies (e.g. OpenCode plan/build, or Claude agent ids with a
      defaulted ``is_sub_agent=False``)
    - session_primary_agents: dominant mode per session (folds compaction)
    """
    primary = _session_primary_agents(steps)
    sessions = {str(s.get("session_id") or "") for s in steps if isinstance(s, dict) and s.get("session_id")}
    if len(sessions) > 1:
        return True, False, primary

    eas: set[str] = set()
    names: set[str] = set()
    for s in steps:
        if not isinstance(s, dict):
            continue
        if s.get("role") not in ("assistant", "user", "compaction"):
            continue
        eas.add(effective_agent(s))
        name = str(s.get("agent") or "").strip()
        if name:
            names.add(name)
    use_names = (not (eas - {""})) and len(names) > 1
    return False, use_names, primary


def _timeline_agent_id(
    step: dict,
    *,
    multi_session: bool = False,
    use_agent_names: bool = False,
    primary_agents: dict[str, str] | None = None,
) -> str:
    """Fine-grained identity for segment boundaries (session / subagent / mode).

    Multi-session ids are ``session_id`` or ``session_id::agent`` when the
    agent field is set — so plan/build modes on the same OpenCode root
    session stay distinct, and parallel explore sessions do not merge.
    Compaction steps inherit the session's primary agent name.
    """
    primary_agents = primary_agents or {}
    if multi_session:
        sid = step.get("session_id") or ""
        if isinstance(sid, str) and sid:
            name = str(step.get("agent") or "").strip()
            if name == "compaction" or step.get("role") == "compaction" or step.get("is_compaction_checkpoint"):
                name = primary_agents.get(sid, "")
            return f"{sid}::{name}" if name else sid
    ea = effective_agent(step)
    if ea:
        return ea
    if use_agent_names:
        return str(step.get("agent") or "").strip()
    return ""


def _timeline_id_session(agent_id: str) -> str:
    """Session portion of a multi-session timeline id (before ``::``)."""
    if "::" in agent_id:
        return agent_id.split("::", 1)[0]
    return agent_id


def _trunc_timeline_label(name: str) -> str:
    return name if len(name) <= 20 else name[:19] + "…"


def _mode_used_by_tagged_subagents(mode: str, steps: list[dict]) -> bool:
    if not mode:
        return False
    for s in steps:
        if not isinstance(s, dict) or not s.get("is_sub_agent"):
            continue
        if str(s.get("agent") or "").strip() == mode:
            return True
    return False


def _timeline_display_label(agent_id: str, steps: list[dict]) -> str:
    """Legend / color bucket for a timeline agent id."""
    if not agent_id:
        return "main"
    tagged_subs = any(isinstance(s, dict) and s.get("is_sub_agent") for s in steps)
    sid = _timeline_id_session(agent_id)
    mode = ""
    if "::" in agent_id:
        mode = agent_id.split("::", 1)[1].strip()

    lane_sub = False
    lane_parent = False
    title = ""
    for s in steps:
        if not isinstance(s, dict):
            continue
        step_sid = str(s.get("session_id") or "")
        if step_sid != sid and effective_agent(s) not in {agent_id, sid}:
            continue
        step_mode = str(s.get("agent") or "").strip()
        if mode and step_mode and step_mode != mode:
            if not (
                step_mode == "compaction"
                or s.get("role") == "compaction"
                or s.get("is_compaction_checkpoint")
            ):
                continue
        if s.get("is_sub_agent"):
            lane_sub = True
        elif s.get("role") in ("assistant", "user", "compaction"):
            lane_parent = True
        if not title:
            candidate = str(s.get("session_title") or "").strip()
            if candidate:
                title = candidate

    # DSH (and similar) tags child sessions with is_sub_agent while sharing a
    # generic preset name ("standard"). Prefer main / sub {id} over that preset.
    if tagged_subs:
        if lane_sub and not lane_parent:
            if title:
                return _trunc_timeline_label(title)
            short = sid[:12] if len(sid) > 12 else sid
            return f"sub {short}" if short else _trunc_timeline_label(mode or agent_id)
        if lane_parent and not lane_sub:
            if mode and not _mode_used_by_tagged_subagents(mode, steps):
                return _trunc_timeline_label(mode)
            return "main"

    # Composite multi-session id: prefer the embedded agent mode name
    if mode:
        return _trunc_timeline_label(mode)
    for s in steps:
        if not isinstance(s, dict):
            continue
        if (s.get("session_id") or "") == sid and sid == agent_id:
            name = str(s.get("agent") or "").strip()
            if name:
                return _trunc_timeline_label(name)
            title = str(s.get("session_title") or "").strip()
            if title:
                return _trunc_timeline_label(title)
            break
        if effective_agent(s) == agent_id or str(s.get("agent") or "").strip() == agent_id:
            name = str(s.get("agent") or "").strip()
            if name and name == agent_id:
                return _trunc_timeline_label(name)
    short = agent_id[:12] if len(agent_id) > 12 else agent_id
    return f"sub {short}"


def _disambiguate_timeline_labels(ids: list[str], steps: list[dict]) -> dict[str, str]:
    """Unique legend labels; suffix short session id when names collide."""
    raw = {aid: _timeline_display_label(aid, steps) for aid in ids}
    counts = Counter(raw.values())
    out: dict[str, str] = {}
    for aid, label in raw.items():
        if counts[label] > 1 and aid:
            sid = _timeline_id_session(aid)
            suffix = sid[-6:] if len(sid) > 6 else sid
            out[aid] = f"{label} ({suffix})"
        else:
            out[aid] = label
    return out


def _legend_label(agent_id: str, labels: dict[str, str]) -> str:
    """Display name for a timeline agent id; empty id is main."""
    if agent_id in labels:
        return labels[agent_id]
    return agent_id or "main"


def bind_timeline_agents(
    steps: list[dict],
) -> tuple[dict[str, int], dict[str, str], Callable[[dict], str]]:
    """Color map, legend labels, and per-step timeline identity.

    Charts and workflow cards must use this together so OpenCode
    ``session_id::mode`` keys stay consistent with the palette.
    """
    multi, use_names, primary = _timeline_context(steps)
    order: list[str] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        agent = _timeline_agent_id(s, multi_session=multi, use_agent_names=use_names, primary_agents=primary)
        if agent not in order:
            order.append(agent)
    if "" in order and order[0] != "":
        order.remove("")
        order.insert(0, "")
    if not order:
        order = [""]
    color_map = {aid: i for i, aid in enumerate(order)}
    labels = _disambiguate_timeline_labels(list(color_map.keys()), steps)

    def agent_id(step: dict) -> str:
        return _timeline_agent_id(
            step,
            multi_session=multi,
            use_agent_names=use_names,
            primary_agents=primary,
        )

    return color_map, labels, agent_id


def build_agent_color_map(steps: list[dict]) -> dict[str, int]:
    """Return a mapping from timeline agent-id to palette index.

    Empty string is index 0 (main) when a step maps to it. Other agents
    follow in first-seen order.
    """
    color_map, _labels, _agent_id = bind_timeline_agents(steps)
    return color_map


# -- Layout helpers -------------------------------------------------------

_TPL = "plotly_white"


def _empty_figure(height: int = 380, message: str | None = None) -> go.Figure:
    """Return a blank Plotly figure, optionally with a centered message."""
    fig = go.Figure()
    if message:
        fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=16)
    fig.update_layout(template=_TPL, height=height)
    return fig


def _apply_chart_layout(
    fig: go.Figure, title: str, xaxis: str | None = None, yaxis: str | None = None, height: int = 380, **kwargs
) -> None:
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
        title=dict(text=title, y=0.99, x=0.5, xanchor="center", yanchor="top", font=dict(size=16)),
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
        xref="paper",
        yref="paper",
        x=1.0,
        y=-0.12,
        showarrow=False,
        font=dict(size=9, color="#9ca3af"),
        xanchor="right",
    )


# -- Reusable trace helpers ------------------------------------------------


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


# -- Annotation utilities ------------------------------------------------


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


# -- Chart builders -------------------------------------------------------


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
        from .parser import infer_non_cache_input

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


# -- New chart types -------------------------------------------------------


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


# -- Label visualization charts -------------------------------------------


_LABEL_FONT = dict(size=13)  # consistent font size across label charts


def _add_phase_legend(fig: go.Figure, phases_used: set[str]) -> None:
    """Add invisible trace per phase to create a color legend."""
    canonical = ["understand", "plan", "implement", "debug", "validate", "report"]
    for phase in canonical:
        if phase in phases_used:
            fig.add_trace(
                go.Bar(
                    x=[None],
                    y=[None],
                    marker_color=LABEL_PHASE_COLORS.get(phase, "#6b7280"),
                    name=phase,
                    showlegend=True,
                )
            )


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
        text=texts,
        textposition="outside",
        cliponaxis=False,
        textfont=_LABEL_FONT,
        showlegend=False,
    )

    if orientation == "h":
        bar_kwargs.update(y=labels, x=values, orientation="h")
        if action_to_phase:
            phases = [action_to_phase.get(a, "") for a in labels]
            bar_kwargs["customdata"] = phases
            bar_kwargs["hovertemplate"] = (
                "%{y} (%{customdata}): %{x" + (f":{value_format}" if value_format else "") + "}<extra></extra>"
            )
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
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            )
    else:
        layout_kwargs["height"] = 380

    _apply_chart_layout(fig, title, xaxis=xaxis, yaxis=yaxis, **layout_kwargs)
    return fig


_CANONICAL_PHASES = ["understand", "plan", "implement", "debug", "validate", "report"]


def build_label_phase_count_chart(phase_counts: dict[str, int], dark: bool = False) -> go.Figure:
    """Bar chart of step counts per phase (level 1)."""
    fig = _build_label_bar_chart(
        phase_counts,
        color_fn=lambda p: LABEL_PHASE_COLORS.get(p, "#6b7280"),
        title="Step Count by Phase",
        xaxis="Phase",
        yaxis="Steps",
        empty_message="No phase data",
        canonical_order=_CANONICAL_PHASES,
    )
    _apply_dark(fig, dark)
    return fig


def build_label_action_count_chart(
    action_counts: dict[str, int], action_to_phase: dict[str, str], dark: bool = False
) -> go.Figure:
    """Bar chart of step counts per action (level 2), colored by parent phase."""
    fig = _build_label_bar_chart(
        action_counts,
        color_fn=lambda a: LABEL_PHASE_COLORS.get(action_to_phase.get(a, ""), "#6b7280"),
        title="Step Count by Action",
        orientation="h",
        xaxis="Steps",
        empty_message="No action data",
        action_to_phase=action_to_phase,
    )
    _apply_dark(fig, dark)
    return fig


def build_label_phase_duration_chart(phase_durations: dict[str, float], dark: bool = False) -> go.Figure:
    """Bar chart of summed duration per phase (level 1)."""
    fig = _build_label_bar_chart(
        phase_durations,
        color_fn=lambda p: LABEL_PHASE_COLORS.get(p, "#6b7280"),
        title="Duration by Phase",
        value_format=".1f",
        xaxis="Phase",
        yaxis="Duration (s)",
        empty_message="No phase duration data",
        canonical_order=_CANONICAL_PHASES,
    )
    _apply_dark(fig, dark)
    return fig


def build_label_action_duration_chart(
    action_durations: dict[str, float], action_to_phase: dict[str, str], dark: bool = False
) -> go.Figure:
    """Bar chart of summed duration per action (level 2), colored by parent phase."""
    fig = _build_label_bar_chart(
        action_durations,
        color_fn=lambda a: LABEL_PHASE_COLORS.get(action_to_phase.get(a, ""), "#6b7280"),
        title="Duration by Action",
        orientation="h",
        value_format=".1f",
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
    phases = [p for p in phases if ref_phase_values.get(p, 0) or cmp_phase_values.get(p, 0)]
    if not phases:
        fig = _empty_figure(380, "Upload labeled JSONs for both trajectories to see phase comparison")
        _apply_dark(fig, dark)
        return fig

    ref_vals = [ref_phase_values.get(p, 0) for p in phases]
    cmp_vals = [cmp_phase_values.get(p, 0) for p in phases]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=phases,
            y=ref_vals,
            name=ref_label,
            marker_color="#1d4ed8",
            text=[format(v, value_format) if v else "" for v in ref_vals],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.add_trace(
        go.Bar(
            x=phases,
            y=cmp_vals,
            name=cmp_label,
            marker_color="#dc2626",
            text=[format(v, value_format) if v else "" for v in cmp_vals],
            textposition="outside",
            cliponaxis=False,
        )
    )
    # Pad y-axis headroom so outside labels on the tallest bars don't collide
    # with the legend sitting just above the plot area.
    max_val = max([*ref_vals, *cmp_vals, 0])
    _apply_chart_layout(
        fig,
        title,
        xaxis="Phase",
        yaxis=y_title,
        height=380,
        barmode="group",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
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
        ref_phase_counts,
        cmp_phase_counts,
        ref_label,
        cmp_label,
        title="Step Count by Phase — Reference vs Compared",
        y_title="Steps",
        value_format=",",
        dark=dark,
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
        ref_phase_durations,
        cmp_phase_durations,
        ref_label,
        cmp_label,
        title="Duration by Phase — Reference vs Compared",
        y_title="Duration (s)",
        value_format=".1f",
        dark=dark,
    )


def build_label_timeline_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Horizontal bar timeline — one segment per step, colored by phase, action on hover."""
    if not steps:
        fig = _empty_figure(380, "No step data")
        _apply_dark(fig, dark)
        return fig

    _USER_COLOR = "#d1d5db"  # grey fill for user prompt bar
    _USER_BORDER = "#9ca3af"  # darker border

    step_indices = [s.get("index", i) for i, s in enumerate(steps)]
    step_labels = [str(idx) for idx in step_indices]
    durations = [s.get("duration_s") or 0 for s in steps]
    roles = [s.get("role", "assistant") for s in steps]
    phases = [s.get("phase", "") for s in steps]
    actions = [s.get("action", "") for s in steps]
    max_dur = max(durations) if durations else 1
    y_pos = list(range(len(steps)))

    # --- Assistant bars (only non-user rows; user rows get 0) ---
    asst_durations = [d if r != "user" else 0 for d, r in zip(durations, roles, strict=False)]
    asst_colors = [LABEL_PHASE_COLORS.get(p, "#6b7280") for p in phases]
    asst_text = [f"{a}  ({d:.1f}s)" if r != "user" else "" for r, a, d in zip(roles, actions, durations, strict=False)]
    asst_hover = [
        (
            f"<b>Step {step_indices[i]}</b><br>"
            f"Phase: {phases[i]}<br>"
            f"Action: {actions[i]}<br>"
            f"Duration: {durations[i]:.1f}s<br>"
            f"Tokens: {s.get('tokens_total', 0):,}"
        )
        if roles[i] != "user"
        else ""
        for i, s in enumerate(steps)
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=asst_durations,
            y=y_pos,
            orientation="h",
            marker_color=asst_colors,
            text=asst_text,
            textposition="outside",
            outsidetextfont=dict(size=11),
            textangle=0,
            cliponaxis=False,
            hovertext=asst_hover,
            hoverinfo="text",
            showlegend=False,
        )
    )

    # --- User prompt bars (small fixed-width marker + label) ---
    user_bar_width = max_dur * 0.04
    user_indices = [i for i, r in enumerate(roles) if r == "user"]
    if user_indices:
        user_hover = [
            f"<b>Step {step_indices[i]} — user prompt</b><br>{steps[i].get('text_preview', '')[:120]}"
            for i in user_indices
        ]
        fig.add_trace(
            go.Bar(
                x=[user_bar_width] * len(user_indices),
                y=user_indices,
                orientation="h",
                width=0.8,  # match agent bar height
                marker_color=_USER_COLOR,
                marker_line=dict(color=_USER_BORDER, width=1),
                hovertext=user_hover,
                hoverinfo="text",
                showlegend=False,
            )
        )
        for yi in user_indices:
            fig.add_annotation(
                x=user_bar_width,
                y=yi,
                text="<i>user prompt</i>",
                showarrow=False,
                xanchor="left",
                xshift=6,
                font=dict(size=10, color="#6b7280"),
            )

    # --- Legend entries (one per phase + user) ---
    seen: set[str] = set()
    for role, phase in zip(roles, phases, strict=False):
        key = "user" if role == "user" else phase
        if key and key not in seen:
            seen.add(key)
            fig.add_trace(
                go.Bar(
                    x=[None],
                    y=[None],
                    orientation="h",
                    marker_color=_USER_COLOR if key == "user" else LABEL_PHASE_COLORS.get(key, "#6b7280"),
                    name="user prompt" if key == "user" else key,
                    showlegend=True,
                )
            )

    chart_height = max(450, 28 * len(steps))
    top_margin = 90  # title (~25 px) + small gap + legend (~25 px) + breathing room
    # Position both title and legend in container (figure-relative) coords with
    # constant pixel offsets so the title sits on top with the legend just below
    # it, regardless of how tall the chart is. yref="container" keeps both in
    # the same coordinate system so they don't collide.
    title_offset_px = 8  # title top, ~8 px below figure top
    legend_offset_px = 45  # legend top, ~45 px below figure top (just under title)
    title_y = 1 - title_offset_px / chart_height
    legend_y = 1 - legend_offset_px / chart_height
    _apply_chart_layout(
        fig,
        "Step Timeline (colored by phase, labeled by action)",
        xaxis="Duration (s)",
        yaxis="Step",
        height=chart_height,
        margin=dict(l=70, r=200, t=top_margin, b=40),
        showlegend=True,
        barmode="overlay",
        legend=dict(orientation="h", yref="container", yanchor="top", y=legend_y, xanchor="center", x=0.5),
    )
    # Pin the title above the legend in container coords.
    fig.update_layout(
        title=dict(
            text="Step Timeline (colored by phase, labeled by action)",
            y=title_y,
            yref="container",
            yanchor="top",
            x=0.5,
            xanchor="center",
        )
    )
    fig.update_yaxes(
        tickvals=y_pos,
        ticktext=step_labels,
        autorange="reversed",
    )
    _apply_dark(fig, dark)
    return fig


# -- File Interaction Timeline -----------------------------------------------

_FILE_INTERACTION_COLORS = {
    "read": "#3b82f6",  # blue
    "write": "#10b981",  # green
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
    steps: list[dict] | None = None,
) -> go.Figure:
    """Build a Plotly scatter chart of file interactions across steps.

    x=step index, y=file path (categorical). Color is interaction type for
    single-agent runs, and timeline agent (same palette as the swimlane) when
    more than one agent touched files. Marker shape is always read/write/search.
    Target files are highlighted with a distinct marker border.
    """
    import os

    fig = go.Figure()
    if not interactions:
        _apply_chart_layout(fig, "File Interaction Timeline (no data)")
        _apply_dark(fig, dark)
        return fig

    target_files = target_files or set()

    def _norm(p: str) -> str:
        return os.path.normpath(p) if p else p

    norm_targets = {_norm(t) for t in target_files}

    color_map: dict[str, int] = {"": 0}
    labels: dict[str, str] = {"": "main"}
    agent_id_of = None
    step_by_idx: dict = {}
    if steps:
        color_map, labels, agent_id_of = bind_timeline_agents(steps)
        step_by_idx = {
            s["index"]: s for s in steps if isinstance(s, dict) and "index" in s
        }

    def _interaction_agent(item: dict) -> str:
        if agent_id_of is None:
            return ""
        step = step_by_idx.get(item["step"])
        return agent_id_of(step) if step else ""

    agents_present = []
    seen_agents: set[str] = set()
    for item in interactions:
        aid = _interaction_agent(item)
        if aid not in seen_agents:
            seen_agents.add(aid)
            agents_present.append(aid)
    has_agents = len(seen_agents) > 1

    def _add_group(group: list[dict], *, name: str, color: str, symbol=None) -> None:
        x = [i["step"] for i in group]
        y = [i["path"] for i in group]
        is_target = [_norm(i["path"]) in norm_targets for i in group]
        hover = [
            f"{name}<br>Step {i['step']}: {i['tool']} ({i['type']})<br>{i['path']}"
            for i in group
        ]
        border_colors = ["#dc2626" if t else color for t in is_target]
        border_widths = [2 if t else 0 for t in is_target]
        if symbol is None:
            symbol = [_FILE_INTERACTION_SYMBOLS.get(i["type"], "circle") for i in group]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=name,
                marker=dict(
                    color=color,
                    size=9,
                    symbol=symbol,
                    line=dict(color=border_colors, width=border_widths),
                ),
                hovertext=hover,
                hoverinfo="text",
            )
        )

    if has_agents:
        for agent_id in sorted(color_map.keys(), key=lambda a: color_map[a]):
            group = [i for i in interactions if _interaction_agent(i) == agent_id]
            if not group:
                continue
            label = _legend_label(agent_id, labels)
            color = SESSION_COLORS[color_map[agent_id] % len(SESSION_COLORS)]
            _add_group(group, name=label, color=color)
        for aid in agents_present:
            if aid in color_map:
                continue
            group = [i for i in interactions if _interaction_agent(i) == aid]
            if group:
                _add_group(
                    group,
                    name=_legend_label(aid, labels),
                    color=SESSION_COLORS[len(color_map) % len(SESSION_COLORS)],
                )
    else:
        for itype, color in _FILE_INTERACTION_COLORS.items():
            group = [i for i in interactions if i["type"] == itype]
            if group:
                _add_group(
                    group,
                    name=itype,
                    color=color,
                    symbol=_FILE_INTERACTION_SYMBOLS.get(itype, "circle"),
                )

    unique_files = len({i["path"] for i in interactions})
    chart_height = max(350, 25 * unique_files + 80)

    max_path_len = max((len(i["path"]) for i in interactions), default=20)
    left_margin = max(150, max_path_len * 7)
    fig_width = left_margin + 600 + 20  # left margin + plot area + right margin
    title = "File Interaction Timeline by Agent" if has_agents else "File Interaction Timeline"

    _apply_chart_layout(
        fig,
        title,
        xaxis="Step",
        yaxis="File",
        height=chart_height,
        margin=dict(l=left_margin, r=20, t=50, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_layout(width=fig_width, autosize=False)
    file_order: list[str] = []
    seen_files: set[str] = set()
    for item in interactions:
        path = item["path"]
        if path not in seen_files:
            seen_files.add(path)
            file_order.append(path)
    # categoryarray is bottom→top; reverse so the first-touched files sit at the top.
    fig.update_yaxes(
        fixedrange=False,
        categoryorder="array",
        categoryarray=list(reversed(file_order)),
    )
    fig.update_xaxes(fixedrange=False)
    fig.update_layout(dragmode="pan")
    _apply_dark(fig, dark)
    return fig


# -- Score Gauge Chart -------------------------------------------------------

# -- Additional chart builders -----------------------------------------------


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
    for item in items:
        content = item["content"]
        short = content[:40] + "..." if len(content) > 40 else content
        y_labels.append(short)

        start = item.get("start_step")
        end = item.get("end_step")
        is_stalled = content in stalled_contents

        if start is not None and end is not None:
            color = "#dc2626" if is_stalled else "#059669"
            width = end - start
            fig.add_trace(
                go.Bar(
                    y=[short],
                    x=[max(width, 1)],
                    orientation="h",
                    base=start,
                    marker_color=color,
                    showlegend=False,
                    hovertext=f"{content}<br>Steps {start}→{end} ({width} steps)",
                    hoverinfo="text",
                    text=f"{width} steps",
                    textposition="inside",
                )
            )
        elif start is not None:
            fig.add_trace(
                go.Bar(
                    y=[short],
                    x=[1],
                    orientation="h",
                    base=start,
                    marker_color="#d97706",
                    showlegend=False,
                    hovertext=f"{content}<br>Started at step {start}, not completed",
                    hoverinfo="text",
                    text="stalled",
                    textposition="inside",
                )
            )
        elif end is not None:
            # Item went straight to "completed" without ever being marked
            # in_progress (common for OpenCode todo lists). Show a thin marker
            # at the completion step so the row isn't blank.
            fig.add_trace(
                go.Bar(
                    y=[short],
                    x=[1],
                    orientation="h",
                    base=max(0, end - 1),
                    marker_color="#3b82f6",
                    showlegend=False,
                    hovertext=f"{content}<br>Completed at step {end} (no in_progress recorded)",
                    hoverinfo="text",
                    text="completed",
                    textposition="inside",
                )
            )
        else:
            # Never started or completed — show a grey placeholder at x=0 so
            # the row appears in the y-axis instead of being silently dropped.
            fig.add_trace(
                go.Bar(
                    y=[short],
                    x=[0.5],
                    orientation="h",
                    base=0,
                    marker_color="#9ca3af",
                    showlegend=False,
                    hovertext=f"{content}<br>Never started",
                    hoverinfo="text",
                    text="not started",
                    textposition="outside",
                    cliponaxis=False,
                )
            )

    # Compact bar height (22px each + 90px padding) keeps the chart from
    # ballooning vertically when only a few items have explicit timing.
    chart_height = max(220, 22 * len(items) + 90)

    _apply_chart_layout(
        fig,
        "Plan Progress Timeline",
        xaxis="Step",
        height=chart_height,
        margin=dict(l=max(200, max((len(lbl) for lbl in y_labels), default=10) * 6 + 20), r=40, t=50, b=40),
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
            # run the shared loader error-pattern matcher against the
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
        "platform_error": "Platform Error",
        "permission_error": "Permission / Policy",
        "missing_file": "Missing File",
        "bad_input": "Bad Input",
        "tool_error": "Other Tool Error",
    }
    _COLORS = {
        "platform_error": "#dc2626",
        "permission_error": "#d97706",
        "missing_file": "#6366f1",
        "bad_input": "#0891b2",
        "tool_error": "#6b7280",
    }

    sorted_types = sorted(error_types.keys(), key=lambda t: error_types[t])
    labels = [_LABELS.get(t, t) for t in sorted_types]
    counts = [error_types[t] for t in sorted_types]
    colors = [_COLORS.get(t, CHART_ACCENT) for t in sorted_types]
    hover = [
        f"{_LABELS.get(t, t)}: {error_types[t]}<br>Steps: {', '.join(str(s) for s in error_steps.get(t, [])[:10])}"
        for t in sorted_types
    ]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=counts,
            orientation="h",
            marker_color=colors,
            showlegend=False,
            text=[str(c) for c in counts],
            textposition="outside",
            hovertext=hover,
            hoverinfo="text",
        )
    )
    _apply_chart_layout(
        fig,
        "Tool Error Classification",
        xaxis="Count",
        height=max(200, 40 * len(sorted_types)),
        margin=dict(l=max(140, max((len(lbl) for lbl in labels), default=10) * 7 + 20), r=60, t=50, b=40),
    )
    _apply_dark(fig, dark)
    return fig


_COMPACTION_KIND_LABEL = {
    "compaction_part": "compacted",
    "compaction_message": "compacted",
    "summary": "compacted",
    "compress_step": "compacted",
    "tool_prune": "pruned",
    "occupancy_drop": "compacted",
}

# Occupancy must not share Fresh Input's blue — after compaction the stack is
# almost entirely fresh, and a blue occupancy line disappears into the fill.
_OCCUPANCY_LINE_COLOR = "#7c3aed"
_COMPACTION_MARKER_COLOR = "#e11d48"


def _pressure_series_colors(agents: list[dict]) -> dict[str, str]:
    """Distinct hue per pressure series, in first-appearance order."""
    return {agent.get("agent_id", ""): SESSION_COLORS[i % len(SESSION_COLORS)] for i, agent in enumerate(agents)}


def build_context_pressure_chart(
    steps: list[dict],
    *,
    agent_key: str | None = None,
    raw: dict | None = None,
    dark: bool = False,
) -> go.Figure:
    """Context-window occupancy over global step index, with compaction markers.

    Overlay mode (two or more agents) draws one occupancy line per agent in a
    distinct color. Compaction is drawn on that agent's series only — never as
    a full-height line across every session. Single-agent mode stacks fresh vs
    cache under an occupancy line and, when a window limit is known, adds
    70%/90% bands.
    """
    from .diagnostics import context_pressure_series

    series = context_pressure_series(steps, agent_key=agent_key, raw=raw)
    agents = series.get("agents") or []
    events = series.get("events") or []
    window_limit = series.get("window_limit")
    has_points = any(a.get("points") for a in agents)
    if not has_points:
        fig = _empty_figure(380, "No context occupancy data.")
        _apply_dark(fig, dark)
        return fig

    fig = go.Figure()
    single = len(agents) == 1
    color_map = _pressure_series_colors(agents)
    y_max = 0
    for agent in agents:
        for point in agent.get("points") or []:
            y_max = max(y_max, int(point.get("occupancy") or 0))
    if isinstance(window_limit, (int, float)) and window_limit > 0:
        y_max = max(y_max, int(window_limit))

    if single:
        agent = agents[0]
        color = _OCCUPANCY_LINE_COLOR
        points = agent.get("points") or []
        xs = [p["step"] for p in points]
        fresh = [p["fresh"] for p in points]
        cache = [p["cache_read"] for p in points]
        occ = [p["occupancy"] for p in points]
        turns = [p["local_turn"] for p in points]
        pct_suffix = []
        for occupancy in occ:
            if isinstance(window_limit, (int, float)) and window_limit > 0:
                pct_suffix.append(f"<br>Pressure: {100.0 * occupancy / window_limit:.1f}%")
            else:
                pct_suffix.append("")
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=fresh,
                name="Fresh input",
                mode="lines",
                line=dict(width=0.5, color=TOKEN_COLORS["fresh_input"]),
                stackgroup="occ",
                fillcolor="rgba(59,130,246,0.35)",
                hovertemplate=("Step %{x}<br>Turn %{customdata}<br>Fresh: %{y:,}<extra></extra>"),
                customdata=turns,
                legendgroup="tokens",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=cache,
                name="Cache read",
                mode="lines",
                line=dict(width=0.5, color=TOKEN_COLORS["cache_read"]),
                stackgroup="occ",
                fillcolor="rgba(52,211,153,0.35)",
                hovertemplate=("Step %{x}<br>Turn %{customdata}<br>Cache read: %{y:,}<extra></extra>"),
                customdata=turns,
                legendgroup="tokens",
            )
        )
        hover = [
            f"{agent['label']}<br>Step {x}<br>Turn {turn}<br>Occupancy: {o:,}{pct}"
            for x, turn, o, pct in zip(xs, turns, occ, pct_suffix, strict=False)
        ]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=occ,
                name="Occupancy",
                mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=6, color=color),
                hovertext=hover,
                hoverinfo="text",
                legendgroup=agent.get("agent_id", ""),
            )
        )
    else:
        for agent in agents:
            points = agent.get("points") or []
            if not points:
                continue
            agent_id = agent.get("agent_id", "")
            color = color_map.get(agent_id, SESSION_COLORS[0])
            xs = [p["step"] for p in points]
            occ = [p["occupancy"] for p in points]
            turns = [p["local_turn"] for p in points]
            hover = []
            for x, turn, o, p in zip(
                xs,
                turns,
                occ,
                points,
                strict=False,
            ):
                pct = ""
                if isinstance(window_limit, (int, float)) and window_limit > 0:
                    pct = f"<br>Pressure: {100.0 * o / window_limit:.1f}%"
                hover.append(
                    f"{agent['label']}<br>Step {x}<br>Turn {turn}"
                    f"<br>Occupancy: {o:,}<br>Fresh: {p['fresh']:,}"
                    f"<br>Cache: {p['cache_read']:,}{pct}"
                )
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=occ,
                    name=agent["label"],
                    mode="lines+markers",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=6, color=color),
                    hovertext=hover,
                    hoverinfo="text",
                    legendgroup=agent_id,
                )
            )

    _add_compaction_markers(
        fig,
        events,
        agents=agents,
        color_map=color_map,
        y_max=y_max,
        marker_color=_COMPACTION_MARKER_COLOR if single else None,
    )

    if isinstance(window_limit, (int, float)) and window_limit > 0:
        fig.add_hline(
            y=window_limit,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text=f"Window {window_limit:,}",
            annotation_position="top right",
        )
        if single:
            fig.add_hrect(
                y0=window_limit * 0.7,
                y1=window_limit * 0.9,
                fillcolor="rgba(217,119,6,0.08)",
                line_width=0,
            )
            fig.add_hrect(
                y0=window_limit * 0.9,
                y1=window_limit,
                fillcolor="rgba(220,38,38,0.10)",
                line_width=0,
            )

    title = "Context Window Pressure"
    if single:
        title = f"Context Window Pressure — {agents[0]['label']}"
    _apply_chart_layout(
        fig,
        title,
        xaxis="Step",
        yaxis="Tokens (occupancy)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    if y_max > 0:
        fig.update_yaxes(range=[0, y_max * 1.08])
    _add_legend_hint(fig)
    _apply_dark(fig, dark)
    return fig


def _add_compaction_markers(
    fig: go.Figure,
    events: list[dict],
    *,
    agents: list[dict],
    color_map: dict[str, str],
    y_max: int,
    marker_color: str | None = None,
) -> None:
    """Per-agent compaction diamonds and drop stems — not full-height vlines."""
    if not events:
        return
    from .diagnostics import coalesce_compaction_events

    labels = {a.get("agent_id", ""): a.get("label") or "main" for a in agents}
    by_agent: dict[str, list[dict]] = {}
    for event in events:
        agent_id = event.get("agent", "")
        if agent_id not in labels:
            continue
        by_agent.setdefault(agent_id, []).append(event)

    for agent_id, agent_events in by_agent.items():
        color = marker_color or color_map.get(agent_id) or SESSION_COLORS[0]
        agent_label = labels.get(agent_id, "main")
        coalesced = coalesce_compaction_events(agent_events)
        xs: list[int] = []
        ys: list[float] = []
        hover: list[str] = []
        for event in coalesced:
            step = int(event.get("step") or 0)
            kind = event.get("kind") or ""
            kind_label = _COMPACTION_KIND_LABEL.get(kind, kind)
            before = event.get("occupancy_before")
            after = event.get("occupancy_after")
            dropped = event.get("dropped")
            y = before if isinstance(before, (int, float)) and before > 0 else after
            if not isinstance(y, (int, float)) or y <= 0:
                y = y_max * 0.5 if y_max else 0
            drop_txt = f"<br>−{int(dropped):,} tokens" if dropped else ""
            after_txt = f"<br>After: {int(after):,}" if isinstance(after, (int, float)) and after > 0 else ""
            hover.append(f"{agent_label}<br>Step {step}<br>{kind_label}{drop_txt}{after_txt}")
            xs.append(step)
            ys.append(float(y))
            if (
                isinstance(before, (int, float))
                and before > 0
                and isinstance(after, (int, float))
                and 0 < after < before
            ):
                fig.add_trace(
                    go.Scatter(
                        x=[step, step],
                        y=[before, after],
                        mode="lines",
                        line=dict(color=color, width=2, dash="dash"),
                        legendgroup=agent_id,
                        showlegend=False,
                        hoverinfo="skip",
                        name=f"{agent_label} compact",
                    )
                )
        if not xs:
            continue
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                name=f"{agent_label} compact",
                mode="markers",
                marker=dict(
                    size=12,
                    symbol="diamond",
                    color=color,
                    line=dict(width=1.5, color="#fff"),
                ),
                legendgroup=agent_id,
                hovertext=hover,
                hoverinfo="text",
            )
        )


# -- Task Mode Breakdown (#2) ------------------------------------------------

# -- Duration vs True Cost Gap (#6) ------------------------------------------
