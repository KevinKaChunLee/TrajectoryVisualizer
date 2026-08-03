"""Markdown and HTML display string generation."""

import html
from typing import Any


_VERDICT_STYLES = {
    "good": ("var(--ov-good, #16a34a)", "#f0fdf4", "#bbf7d0"),
    "warn": ("var(--ov-warn, #d97706)", "#fffbeb", "#fde68a"),
    "bad":  ("var(--ov-bad, #dc2626)",  "#fef2f2", "#fecaca"),
}


def _metric_chip(label: str, value: str, *, wide: bool = False,
                 verdict: str | None = None, hint: str = "") -> str:
    """Render a single metric as a compact card chip (HTML).

    *verdict* adds a colored left border (good/warn/bad).
    *hint* adds a subtitle line below the value.
    """
    # label/value can be untrusted (tool, model, agent names); escape for HTML.
    label = html.escape(str(label))
    value = html.escape(str(value))
    hint = html.escape(str(hint)) if hint else ""
    min_w = "140px" if wide else "100px"
    border_left = ""
    bg = "#f8fafc"
    border_color = "#e2e8f0"
    if verdict and verdict in _VERDICT_STYLES:
        color, bg, border_color = _VERDICT_STYLES[verdict]
        border_left = f"border-left:3px solid {color};"
    hint_html = (f"<span style='font-size:9px;color:#94a3b8;margin-top:1px;'>{hint}</span>"
                 if hint else "")
    return (
        f"<div style='display:inline-flex;flex-direction:column;background:{bg};"
        f"border:1px solid {border_color};border-radius:8px;padding:6px 10px;"
        f"min-width:{min_w};{border_left}'>"
        f"<span style='font-size:10px;color:#64748b;text-transform:uppercase;'>{label}</span>"
        f"<span style='font-size:13px;color:#1e293b;font-weight:500;'>{value}</span>"
        f"{hint_html}"
        f"</div>"
    )


def _metric_grid(chips: list[str], title: str = "") -> str:
    """Wrap chips in a flex-wrap grid with optional title."""
    header = f"### {title}\n\n" if title else ""
    return header + f"<div style='display:flex;flex-wrap:wrap;gap:6px;'>" + "".join(chips) + "</div>\n"


_FINISH_LABELS = {
    "tool-calls": "Tool Call",
    "stop": "Completed",
    "end_turn": "End Turn",
}


def _friendly_finish(raw: str | None) -> str:
    """Map internal finish-reason enums to user-friendly labels."""
    if not raw:
        return ""
    return _FINISH_LABELS.get(raw, raw.replace("-", " ").replace("_", " ").title())


_PART_LABELS = {
    "text": "Text",
    "reasoning": "Reason",
    "tool_call": "Tool",
    "step_start": "Start",
    "step_finish": "Finish",
    "snapshot": "Snap",
    "patch": "Patch",
}


def _friendly_parts(part_mix: str) -> str:
    """Convert comma-separated part types to compact friendly labels."""
    if not part_mix:
        return ""
    types = [t.strip() for t in part_mix.split(",") if t.strip()]
    labels = [_PART_LABELS.get(t, t.replace("_", " ").title()) for t in types]
    if len(labels) <= 3:
        return " · ".join(labels)
    return " · ".join(labels[:2]) + f" +{len(labels) - 2}"


def _fmt_dict_as_table(d: dict, key_header: str = "Key", val_header: str = "Count") -> str:
    """Format a dict as a markdown table."""
    if not d:
        return "*None*"
    lines = [f"| {key_header} | {val_header} |", "|---|---|"]
    for k, v in sorted(d.items(), key=lambda x: -(x[1] if isinstance(x[1], (int, float)) else 0)):
        lines.append(f"| `{k}` | {v} |")
    return "\n".join(lines)


def _build_hotspots_md(rows: list[dict]) -> str:
    """Build markdown tables for top latency/token/cache-miss hotspots."""
    with_dur = [r for r in rows if r.get("duration") is not None]
    top_d = sorted(with_dur, key=lambda r: r["duration"], reverse=True)[:5]
    top_t = sorted(rows, key=lambda r: r["tokens_total"], reverse=True)[:5]

    def fmt_table(items: list[dict], value_field: str, value_header: str,
                  value_fmt: str, extra_cols: list[tuple[str, str, str]] | None = None) -> str:
        if not items:
            return "*No data*"
        extra = extra_cols or [("tokens_total", "Tokens", ","), ("tool_calls", "Tool Calls", "")]
        hdr = " | ".join(h for _, h, _ in extra)
        lines = [
            f"| Step | Role | {value_header} | {hdr} |",
            "|---:|---|" + "---:|" * (1 + len(extra)),
        ]
        for r in items:
            v = r[value_field]
            v_str = format(v, value_fmt) if isinstance(v, (int, float)) else str(v)
            extras = " | ".join(
                format(r[f], ef) if isinstance(r[f], (int, float)) and ef else str(r[f])
                for f, _, ef in extra
            )
            lines.append(f"| {r['index']} | `{r['role']}` | {v_str} | {extras} |")
        return "\n".join(lines)

    sections = [
        "### Message Hotspots\n\n"
        "**Top latency steps**\n\n"
        + fmt_table(top_d, "duration", "Duration (s)", ".2f")
        + "\n\n**Top token-load steps**\n\n"
        + fmt_table(top_t, "tokens_total", "Tokens", ",",
                    extra_cols=[("tool_calls", "Tool Calls", ""),
                                ("duration", "Duration (s)", ".2f")])
    ]

    # Lowest cache ratio (assistant steps with tokens, excluding 0-token steps).
    # Skip the table entirely if no step reports any cache read — otherwise the
    # table is just five rows of 0.0% (uninformative) for trajectories whose
    # provider doesn't emit cache metrics (e.g., some Codex sessions).
    asst_with_tok = [r for r in rows
                     if r.get("role") == "assistant" and r["tokens_total"] > 0]
    if asst_with_tok and any(r["cache_ratio"] > 0 for r in asst_with_tok):
        low_cache = sorted(asst_with_tok, key=lambda r: r["cache_ratio"])[:5]
        lines = [
            "| Step | Role | Cache Read % | Fresh Input | Tokens |",
            "|---:|---|---:|---:|---:|",
        ]
        for r in low_cache:
            lines.append(
                f"| {r['index']} | `{r['role']}` | {r['cache_ratio'] * 100:.1f}% | "
                f"{r['non_cache_tokens']:,} | {r['tokens_total']:,} |"
            )
        sections.append(
            "\n\n**Lowest cache read steps** (optimization targets)\n\n"
            + "\n".join(lines)
        )

    return "".join(sections)


def _build_per_message_md(rows: list[dict], limit: int = 80) -> str:
    """Build a compact per-message diagnostics table."""
    if not rows:
        return "*No messages parsed.*"

    has_agent = any(r.get("agent") for r in rows)

    lines = ["### Per-Message Diagnostics", ""]
    if has_agent:
        lines.append(
            "| Step | Role | Agent | Finish | Duration (s) | Tokens | Tok/s | Cache Read % | Fresh Input | Tool Calls |"
        )
        lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append(
            "| Step | Role | Finish | Duration (s) | Tokens | Tok/s | Cache Read % | Fresh Input | Tool Calls |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows[:limit]:
        dur = "N/A" if r["duration"] is None else f"{r['duration']:.2f}"
        tokps = "N/A" if r["tokens_per_sec"] is None else f"{r['tokens_per_sec']:.1f}"
        finish = _friendly_finish(r['finish']) or '-'
        if has_agent:
            agent = r.get("agent", "") or "Main agent"
            lines.append(
                f"| {r['index']} | `{r['role']}` | `{agent}` | `{finish}` | {dur} | "
                f"{r['tokens_total']:,} | {tokps} | {r['cache_ratio'] * 100:.1f}% | "
                f"{r['non_cache_tokens']:,} | {r['tool_calls']} |"
            )
        else:
            lines.append(
                f"| {r['index']} | `{r['role']}` | `{finish}` | {dur} | "
                f"{r['tokens_total']:,} | {tokps} | {r['cache_ratio'] * 100:.1f}% | "
                f"{r['non_cache_tokens']:,} | {r['tool_calls']} |"
            )
    if len(rows) > limit:
        lines.append(f"\n*Showing first {limit} / {len(rows)} messages.*")
    return "\n".join(lines)


def format_session_md(timing: dict, metadata: dict, retry: dict,
                      *, model_id: str = "", provider_id: str = "",
                      agent_id: str = "") -> str:
    """Format session & environment metadata as a markdown table."""
    started = timing.get("started_at", "N/A")
    finished = timing.get("finished_at", "N/A")
    if isinstance(started, str) and len(started) > 19:
        started = started[:19].replace("T", " ")
    else:
        started = str(started)
    if isinstance(finished, str) and len(finished) > 19:
        finished = finished[:19].replace("T", " ")
    else:
        finished = str(finished)

    md = metadata
    is_cc = md.get("agent") == "claude-code"

    if is_cc:
        cc_version = md.get("server_version", "N/A")
        sub_agents = md.get("sub_agent_count", 0)
        event_count = md.get("event_count", 0)
        models_str = model_id or md.get("model") or "N/A"

        pairs = [
            ("Model", models_str),
            ("Agent", "Claude Code"),
            ("Version", cc_version),
            ("Start", started),
            ("End", finished),
            ("Duration", f"{timing.get('total_duration', 'N/A')}s"),
            ("Session", md.get("session_id", "N/A")[:12] + "..."),
            ("Branch", md.get("branch", "N/A")),
            ("Directory", md.get("directory_name", "N/A")),
            ("Platform", (md.get("platform") or "N/A")[:20]),
            ("Sub-agents", str(sub_agents)),
        ]
        if event_count:
            pairs.append(("Raw events", str(event_count)))

        chips = "".join(
            f"<div style='display:inline-flex;flex-direction:column;background:#f8fafc;"
            f"border:1px solid #e2e8f0;border-radius:8px;padding:6px 10px;min-width:100px;'>"
            f"<span style='font-size:10px;color:#64748b;text-transform:uppercase;'>{k}</span>"
            f"<span style='font-size:13px;color:#1e293b;font-weight:500;'>{v}</span>"
            f"</div>"
            for k, v in pairs
        )
        return (
            f"### Session & Environment\n\n"
            f"<div style='display:flex;flex-wrap:wrap;gap:6px;'>{chips}</div>\n"
        )

    # OpenCode format: compact chips
    pairs = [
        ("Model", model_id or md.get("model") or "N/A"),
        ("Provider", provider_id or "N/A"),
        ("Agent", agent_id or md.get("agent", "N/A")),
        ("Start", started),
        ("End", finished),
        ("Duration", f"{timing.get('total_duration', 'N/A')}s"),
        ("Session", (md.get("session_id") or "N/A")[:12]),
        ("Branch", md.get("branch", "N/A")),
        ("Directory", md.get("directory_name", "N/A")),
        ("Platform", (md.get("platform") or "N/A")[:20]),
    ]
    if md.get("agent") == "codearts":
        pairs.extend([
            ("Format", "CodeArts"),
            ("Sessions", str(md.get("session_count", 1))),
            ("Sub-agents", str(md.get("sub_agent_count", 0))),
            ("Export", "Complete" if md.get("export_complete") is True else "Incomplete"),
        ])
    if retry:
        pairs.append(("Retries", f"{retry.get('total_attempts', '?')}/{retry.get('max_retries', '?')}"))

    chips = "".join(
        f"<div style='display:inline-flex;flex-direction:column;background:#f8fafc;"
        f"border:1px solid #e2e8f0;border-radius:8px;padding:6px 10px;min-width:100px;'>"
        f"<span style='font-size:10px;color:#64748b;text-transform:uppercase;'>{k}</span>"
        f"<span style='font-size:13px;color:#1e293b;font-weight:500;'>{v}</span>"
        f"</div>"
        for k, v in pairs
    )
    return (
        f"### Session & Environment\n\n"
        f"<div style='display:flex;flex-wrap:wrap;gap:6px;'>{chips}</div>\n"
    )


def format_performance_md(metrics: dict, wall_fmt: str) -> str:
    """Format performance & token metrics as a markdown table."""
    agent_section = ""
    if metrics.get("agent_breakdown"):
        agent_section = (
            "\n**Agent breakdown**\n\n"
            + _fmt_dict_as_table(metrics["agent_breakdown"], "Agent", "Steps")
            + "\n"
        )
    model_section = ""
    if metrics.get("model_breakdown"):
        model_section = (
            "\n**Model breakdown**\n\n"
            + _fmt_dict_as_table(metrics["model_breakdown"], "Model", "Steps")
            + "\n"
        )
    tok = metrics["tokens"]
    timing_chips = [
        _metric_chip("Steps", str(metrics["total_steps"])),
        _metric_chip("Wall-clock", wall_fmt),
        _metric_chip("Avg duration", f"{metrics['avg_duration']}s"),
        _metric_chip("Med / P95", f"{metrics['median_duration']}s / {metrics['p95_duration']}s", wide=True),
        _metric_chip("Max duration", f"{metrics['max_duration']}s"),
    ]
    # Token breakdown: show N/A when format doesn't provide subcategories
    has_breakdown = tok['input'] > 0 or tok['output'] > 0 or tok['cache_read'] > 0
    def _tok_val(v: int) -> str:
        return f"{v:,}" if has_breakdown else "N/A"

    token_chips = [
        _metric_chip("Total tokens", f"{tok['total']:,}"),
        _metric_chip("Input", _tok_val(tok['input'])),
        _metric_chip("Output", _tok_val(tok['output'])),
        _metric_chip("Reasoning", _tok_val(tok['reasoning'])),
        _metric_chip("Cache read", _tok_val(tok['cache_read'])),
        _metric_chip("Cache write", _tok_val(tok['cache_write'])),
        _metric_chip("Fresh input",
                     f"{metrics['non_cache_tokens']:,} ({metrics['non_cache_ratio']}%)"
                     if has_breakdown else "N/A"),
    ]
    eff_chips = [
        _metric_chip("Avg tok/step", f"{metrics['avg_tokens_per_step']:,}"),
        _metric_chip("Tok/sec", f"{metrics['tokens_per_second']:,}"),
        _metric_chip("Med tok/sec", f"{metrics['median_tokens_per_second']:,}"),
        _metric_chip("Out/In ratio", str(metrics["output_input_ratio"]) if has_breakdown else "N/A"),
        _metric_chip("Tok/tool call", f"{metrics['tokens_per_tool']:,}"),
    ]

    tool_chips = [
        _metric_chip(k, str(v))
        for k, v in sorted(metrics["tool_breakdown"].items(), key=lambda x: -x[1])
    ]
    # Build agent chips with main agent included and sub-agents labeled
    agent_bd = metrics.get("agent_breakdown", {})
    sub_agent_steps = sum(agent_bd.values())
    main_agent_steps = metrics.get("assistant_steps", 0) - sub_agent_steps
    agent_chips = []
    if agent_bd:
        agent_chips.append(_metric_chip("main agent", f"{main_agent_steps} steps", wide=True))
        for k, v in sorted(agent_bd.items(), key=lambda x: -x[1]):
            label = f"sub-agent {k[:12]}"
            agent_chips.append(_metric_chip(label, f"{v} steps", wide=True))
    model_chips = [
        _metric_chip(k, str(v))
        for k, v in sorted(metrics.get("model_breakdown", {}).items(), key=lambda x: -x[1])
    ]

    sections = [
        _metric_grid(timing_chips, "Timing"),
        _metric_grid(token_chips, "Tokens"),
        _metric_grid(eff_chips, "Efficiency"),
    ]
    if tool_chips:
        sections.append(
            f"\n**Tool calls** ({metrics['tool_call_count']} total, {metrics['tool_success_rate']}% success)\n\n"
            + _metric_grid(tool_chips)
        )
    if agent_chips:
        n_sub = len(agent_bd)
        sections.append(
            f"\n**Agent breakdown** (1 main + {n_sub} sub-agent{'s' if n_sub > 1 else ''})\n\n"
            + _metric_grid(agent_chips)
        )
    if model_chips:
        sections.append(f"\n**Model breakdown**\n\n" + _metric_grid(model_chips))

    return "\n".join(sections)


def _cache_verdict(avg_cache: float) -> str | None:
    if avg_cache >= 60:
        return "good"
    if avg_cache >= 30:
        return "warn"
    return "bad" if avg_cache > 0 else None


def _tool_wait_verdict(wait_share: float) -> str | None:
    if wait_share <= 30:
        return "good"
    if wait_share <= 60:
        return "warn"
    return "bad"


def format_behavioral_md(metrics: dict, diag_metrics: dict | None = None) -> str:
    """Format behavioral diagnostics as card grid with verdict badges."""
    avg_cache = metrics.get("avg_cache_ratio", 0)
    tool_wait = metrics.get("tool_wait_share", 0)
    has_cache_data = metrics.get("cache_read_tokens", 0) > 0 or avg_cache > 0
    has_tool_timing = metrics.get("tool_time_total", 0) > 0

    chips = [
        _metric_chip("Asst steps", str(metrics["assistant_steps"])),
        _metric_chip("Multi-tool", str(metrics["multi_tool_steps"])),
        _metric_chip("No-tool", str(metrics["no_tool_assistant_steps"])),
        _metric_chip("Med tok/step", f"{metrics['median_step_tokens']:,}"),
        _metric_chip("P95 tok/step", f"{metrics['p95_step_tokens']:,}"),
    ]
    # Cache metrics — N/A when format doesn't provide cache breakdown
    if has_cache_data:
        chips.append(_metric_chip("Avg cache %", f"{avg_cache}%",
                     verdict=_cache_verdict(avg_cache),
                     hint="≥60% good" if avg_cache < 60 else ""))
        chips.append(_metric_chip("Cache-dom", str(metrics["cache_dominant_steps"])))
    else:
        chips.append(_metric_chip("Avg cache %", "N/A", hint="not available for this format"))

    # Tool timing — N/A when format doesn't provide per-tool timestamps
    if has_tool_timing:
        chips.append(_metric_chip("Tool time", f"{metrics['tool_time_total']}s"))
        chips.append(_metric_chip("Tool-wait %", f"{tool_wait}%",
                     verdict=_tool_wait_verdict(tool_wait),
                     hint="≤30% good" if tool_wait > 30 else ""))
        chips.append(_metric_chip("Tool dur avg", f"{metrics['avg_tool_duration']}s"))
        chips.append(_metric_chip("Tool dur P95", f"{metrics['p95_tool_duration']}s"))
        chips.append(_metric_chip("Tool dur max", f"{metrics['max_tool_duration']}s"))
    else:
        chips.append(_metric_chip("Tool timing", "N/A", hint="not available for this format"))

    # Diagnostic metrics
    dm = diag_metrics or {}

    # Sub-agents
    if dm.get("subagent_session_count", 0) > 0:
        chips.append(_metric_chip("Sub-agents",
                                  f"{dm['subagent_session_count']} ({dm.get('subagent_total_steps', 0)} steps)"))

    # Tool errors
    if dm.get("classified_error_count", 0) > 0:
        chips.append(_metric_chip("Tool errors", str(dm['classified_error_count']),
                                  verdict="bad"))

    return _metric_grid(chips, "Behavioral Diagnostics")


def format_output_md(output: dict, metadata: dict, summary: dict,
                     metrics: dict, *, raw_stats_from: dict | None = None) -> str:
    """Format output & agent stats as markdown."""
    # Finish breakdown
    finish_parts = []
    for fk, fv in sorted(metrics["finish_breakdown"].items(), key=lambda x: -x[1]):
        finish_parts.append(f"{fv} {fk}")
    finish_str = ", ".join(finish_parts) if finish_parts else "N/A"

    # Tool status breakdown
    tool_status_parts = []
    for sk, sv in sorted(metrics["tool_status_breakdown"].items(), key=lambda x: -x[1]):
        tool_status_parts.append(f"{sv} {sk}")
    tool_status_str = ", ".join(tool_status_parts) if tool_status_parts else "N/A"

    # Role breakdown
    role_parts = []
    for rk, rv in sorted(metrics["messages_breakdown"].items()):
        role_parts.append(f"{rv} {rk}")
    role_str = ", ".join(role_parts) if role_parts else "N/A"

    # Output detail rows
    output_rows: list[str] = []
    if output.get("has_patch"):
        output_rows.append(
            f"| Patch | {output.get('patch_lines', 0)} lines,"
            f" {output.get('patch_length', 0):,} chars |"
        )
    if summary:
        output_rows.append(f"| Files changed | {summary.get('files', 'N/A')} |")
        output_rows.append(f"| Additions | +{summary.get('additions', 0)} |")
        output_rows.append(f"| Deletions | -{summary.get('deletions', 0)} |")
    gt_patch = metadata.get("ground_truth_patch", "")
    if gt_patch:
        suffix = "..." if len(gt_patch) > 60 else ""
        output_rows.append(f"| Ground truth | `{gt_patch[:60]}{suffix}` |")
    if output.get("error"):
        output_rows.append(f"| Error | `{output['error']}` |")

    output_table = ""
    if output_rows:
        output_table = "| Field | Value |\n|-------|-------|\n" + "\n".join(output_rows)

    # Raw trajectory breakdown (from statistics section)
    raw_stats = raw_stats_from if isinstance(raw_stats_from, dict) else {}
    raw_section = ""
    if raw_stats:
        raw_user = raw_stats.get("user_messages", 0)
        raw_asst = raw_stats.get("assistant_messages", 0)
        raw_total = raw_stats.get("total_messages", 0)
        raw_events = raw_total - raw_user - raw_asst
        raw_tools = raw_stats.get("total_tool_calls", 0)

        # Parsed-step counts from metrics
        parsed_user = metrics.get("total_steps", 0) - metrics.get("assistant_steps", 0)
        parsed_asst = metrics.get("assistant_steps", 0)

        raw_section = f"""

### Raw Trajectory Statistics

| Category | Raw | Parsed | Notes |
|----------|----:|-------:|-------|
| **Total** | **{raw_total}** | **{metrics.get('total_steps', 0)}** | |
| User turns | {raw_user} | {parsed_user} | {raw_user - parsed_user} tool-result-only messages merged |
| Assistant turns | {raw_asst} | {parsed_asst} | Multi-block responses merged; sub-agent steps added |
| Events | {raw_events} | — | Progress, snapshots, system (not shown as steps) |
| Tool calls (raw) | {raw_tools} | {metrics['tool_call_count']} | Parsed includes sub-agent tool calls |
"""

    indicator_chips = [
        _metric_chip("Steps", role_str, wide=True),
        _metric_chip("Finish states", finish_str, wide=True),
        _metric_chip("Tool calls", str(metrics["tool_call_count"])),
        _metric_chip("Tool status", tool_status_str, wide=True),
        _metric_chip("Tool success", f"{metrics['tool_success_rate']}%"),
        _metric_chip("Reasoning", str(metrics["reasoning_parts"])),
        _metric_chip("Text parts", str(metrics["text_parts"])),
    ]

    result = _metric_grid(indicator_chips, "Output & Agent Stats")
    if output_table:
        result = output_table + "\n\n" + result
    if raw_section:
        result += raw_section
    return result


def wall_clock_fmt(metrics: dict) -> tuple[float, str]:
    """Return (wall_seconds, formatted_string) for wall-clock time."""
    wall = metrics["wall_clock"] if isinstance(metrics["wall_clock"], (int, float)) else metrics["total_duration"]
    fmt = f"{wall:.0f}s" if wall < 3600 else f"{wall / 60:.1f}m"
    return wall, fmt


def format_banner_html(filename: str, metrics: dict, wall_fmt: str,
                       *, trajectory_format: str | None = None) -> str:
    """Build the one-line HTML summary banner for the loaded trajectory."""
    import html as _html
    parts = [
        f"<strong>{_html.escape(filename)}</strong> &nbsp;&mdash;&nbsp; ",
        f"{metrics['total_steps']} steps &middot; ",
        f"{metrics['tool_call_count']} tool calls ({metrics['tool_success_rate']}% success) &middot; ",
    ]
    # Only show token metrics if the format provides them
    total_tokens = metrics.get("tokens", {}).get("total", 0)
    if total_tokens > 0:
        parts.append(f"{total_tokens:,} tokens &middot; ")
        parts.append(f"{metrics['tokens_per_second']} tok/s &middot; ")
    parts.append(f"{wall_fmt} wall-clock")
    if metrics.get("reasoning_parts", 0) > 0:
        parts.append(f" &middot; {metrics['reasoning_parts']} reasoning")
    banner = "".join(parts)

    # Format-specific advisory notes
    note_style = (
        "margin-top:6px;padding:4px 10px;background:#fef3c7;"
        "border-left:3px solid #d97706;border-radius:4px;"
        "font-size:12px;color:#92400e;"
    )
    if trajectory_format in ("opencode", "codearts"):
        format_name = "CodeArts" if trajectory_format == "codearts" else "OpenCode"
        banner += (
            f"<div style='{note_style}'>"
            f"{format_name} format — Token Usage by Step shows all five fields stacked: "
            "Fresh Input + Cache Read + Output + Reasoning = Total, with Cache Write as the 5th segment. "
            "Cache Read can dominate each bar because the source records it as a running "
            "conversation prefix."
            "</div>"
        )
    return banner


def build_analytics_dataframe(step_analytics: list[dict]) -> list[dict]:
    """Convert step analytics into flat rows suitable for a DataFrame."""
    has_agents = any(a.get("agent") for a in step_analytics)
    rows = []
    for a in step_analytics:
        row: dict[str, Any] = {"idx": a["index"], "role": a["role"]}
        if has_agents:
            row["agent"] = a.get("agent", "")
        row.update({
            "Duration (s)": a["duration_s"],
            "Total Tokens": a["tok_total"],
            "Tok/s": round(a["tok_per_s"]) if a["tok_per_s"] is not None else None,
            "Cache Read %": round(a["cache_ratio"] * 100, 1),
            "Fresh Input": a["non_cache_tok"],
            "Out/In Ratio": round(a["out_in_ratio"], 3) if a["out_in_ratio"] is not None else None,
            "Tool Calls": a["tool_calls"],
            "Tool Wait %": (round(a["tool_time_share"] * 100, 1)
                            if a["tool_time_share"] is not None else None),
            "Finish": _friendly_finish(a["finish"]),
            "Parts": _friendly_parts(a["part_mix"]),
            "Idle Gap (s)": a["idle_before_s"],
        })
        rows.append(row)
    return rows
