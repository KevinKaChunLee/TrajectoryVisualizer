"""Reconstruct a Cursor-style context-window composition from a trajectory.

Trajectories rarely record the hidden system prompt or tool JSON schemas, so
the billed occupancy from token metrics is the source of truth for *how full*
the window is. Logged text (messages, skill bodies, spawn prompts, tool
outputs) is tokenized at ≈4 characters/token and attributed to buckets.
Whatever billed occupancy remains is **unattributed** rather than guessed.
"""

from __future__ import annotations

import html
import json
from typing import Any

from trajviz.tool_vocab import SPAWN_TOOL_NAMES, parse_skill_name

from .diagnostics import (
    PRESSURE_ALL_AGENTS,
    detect_compaction_events,
    infer_context_window_limit,
    pressure_agent_id,
    step_context_occupancy,
)
from .diagnostics import _is_occupancy_step, _pressure_agent
from .parser import spawned_child_session_id

# Display order follows Cursor's tray, then TrajViz-only tool_outputs
# (Cursor folds tool results into Conversation).
USAGE_CATEGORIES: tuple[tuple[str, str, str], ...] = (
    ("system", "System prompt", "#6b7280"),
    ("tools", "Tool definitions", "#7c3aed"),
    ("rules", "Rules", "#e11d48"),
    ("skills", "Skills", "#ca8a04"),
    ("mcp", "MCP", "#0d9488"),
    ("subagents", "Subagent definitions", "#2563eb"),
    ("summarized", "Summarized conversation", "#8b5cf6"),
    ("conversation", "Conversations", "#ea580c"),
    ("tool_outputs", "Tool outputs", "#059669"),
    ("unattributed", "Unattributed", "#94a3b8"),
)

_SYSTEM_ROLES = frozenset({"system", "developer"})
_TOOL_DEF_KEYS = ("tools", "tool_definitions", "toolDefinitions", "tool_schemas")
_RULE_KEYS = ("rules", "user_rules", "userRules", "always_apply_rules", "alwaysApplyRules")
_MCP_KEYS = ("mcp", "mcp_servers", "mcpServers", "mcp_tools", "mcpTools")
_SPAWN_NAMES_LOWER = frozenset(name.lower() for name in SPAWN_TOOL_NAMES)
_ACCOUNTABLE_KEYS = tuple(key for key, _label, _color in USAGE_CATEGORIES if key != "unattributed")


def estimate_tokens(text: str) -> int:
    """Approximate token count without a tokenizer (≈4 characters / token)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def format_token_count(n: int) -> str:
    """Compact token count for the usage header (76.8k, 1.2M)."""
    n = int(n)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1_000_000:
        text = f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    elif n >= 1_000:
        text = f"{n / 1_000:.1f}k".replace(".0k", "k")
    else:
        text = str(n)
    return sign + text


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        if not value:
            return ""
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _is_spawn_call(tool_call: dict) -> bool:
    name = str(tool_call.get("tool_name") or "")
    if name.lower() in _SPAWN_NAMES_LOWER:
        return True
    inp = tool_call.get("input")
    if isinstance(inp, dict) and inp.get("subagent_type"):
        return True
    return bool(spawned_child_session_id(tool_call.get("metadata") or {}))


def _raw_blob_tokens(raw: dict | None, keys: tuple[str, ...]) -> int:
    """Tokenize named blobs (tool schemas, rules, MCP catalogs) when present."""
    if not isinstance(raw, dict):
        return 0
    blocks: list[Any] = [raw]
    for key in ("metadata", "info", "agent"):
        block = raw.get(key)
        if isinstance(block, dict):
            blocks.append(block)
    candidates: list[Any] = []
    seen_ids: set[int] = set()
    for block in blocks:
        for key in keys:
            if key not in block:
                continue
            item = block.get(key)
            ident = id(item)
            if ident in seen_ids:
                continue
            seen_ids.add(ident)
            candidates.append(item)
    total = 0
    for item in candidates:
        if not item:
            continue
        total += estimate_tokens(_stringify(item))
    return total


def _last_compaction_step(steps: list[dict], agent_id: str, peak_step: int) -> int | None:
    last: int | None = None
    for event in detect_compaction_events(steps):
        if event.get("agent") != agent_id:
            continue
        step = int(event.get("step") or 0)
        if step < peak_step:
            last = step if last is None else max(last, step)
    return last


def _compaction_part_text(part: dict) -> str:
    chunks: list[str] = []
    seen: set[str] = set()
    for key in ("summary", "text", "recent"):
        value = part.get(key)
        if isinstance(value, str) and value and value not in seen:
            seen.add(value)
            chunks.append(value)
    return "\n".join(chunks)


def _is_summary_turn(step: dict) -> bool:
    role = str(step.get("role") or "")
    return (
        role == "compaction"
        or bool(step.get("is_compaction_checkpoint"))
        or step.get("summary") is True
    )


def _peak_occupancy_anchor(steps: list[dict], target: str | None) -> tuple[str, int, int]:
    """Return ``(agent_id, peak_step, occupancy)`` for the busiest matching window."""
    best: tuple[str, int, int] = ("", -1, 0)
    for step in steps:
        if not isinstance(step, dict) or not _is_occupancy_step(step):
            continue
        agent_id = _pressure_agent(step)
        if target is not None and agent_id != target:
            continue
        occ = step_context_occupancy(step)["occupancy"]
        if occ > best[2]:
            best = (agent_id, int(step.get("index", 0)), occ)
    return best


def _window_start_step(steps: list[dict], agent_id: str, peak_step: int) -> int:
    """First step still in the window: after the last compaction, else 0."""
    start = 0
    for event in detect_compaction_events(steps):
        if event.get("agent") != agent_id:
            continue
        step = int(event.get("step") or 0)
        if step < peak_step:
            start = max(start, step + 1)
    return start


def _accumulate_step(step: dict, buckets: dict[str, int], *, summarized_only: bool = False) -> None:
    role = str(step.get("role") or "")
    if role in _SYSTEM_ROLES:
        if summarized_only:
            return
        for part in step.get("parts") or []:
            if not isinstance(part, dict):
                continue
            text = part.get("text") or part.get("summary") or ""
            if isinstance(text, str) and text:
                buckets["system"] += estimate_tokens(text)
        return

    summary_turn = _is_summary_turn(step)
    for part in step.get("parts") or []:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "compaction":
            text = _compaction_part_text(part)
            if text:
                buckets["summarized"] += estimate_tokens(text)
            continue
        if summarized_only and not summary_turn:
            continue
        if ptype == "text":
            text = part.get("text") or ""
            if not isinstance(text, str) or not text:
                continue
            if part.get("synthetic"):
                if not summarized_only:
                    buckets["system"] += estimate_tokens(text)
            elif summary_turn:
                buckets["summarized"] += estimate_tokens(text)
            else:
                buckets["conversation"] += estimate_tokens(text)
        elif ptype == "reasoning":
            text = part.get("text") or ""
            if isinstance(text, str) and text:
                if summary_turn:
                    buckets["summarized"] += estimate_tokens(text)
                elif not summarized_only:
                    buckets["conversation"] += estimate_tokens(text)

    if summarized_only:
        return

    for tool_call in step.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        inp = _stringify(tool_call.get("input"))
        out = _stringify(tool_call.get("output"))
        name = str(tool_call.get("tool_name") or "")
        if parse_skill_name(name, tool_call.get("input")):
            buckets["skills"] += estimate_tokens(inp) + estimate_tokens(out)
        elif _is_spawn_call(tool_call):
            buckets["subagents"] += estimate_tokens(inp)
            buckets["tool_outputs"] += estimate_tokens(out)
        else:
            buckets["conversation"] += estimate_tokens(inp)
            buckets["tool_outputs"] += estimate_tokens(out)


def context_usage_breakdown(
    steps: list[dict],
    *,
    raw: dict | None = None,
    agent_key: str | None = None,
) -> dict:
    """Bucket logged context at peak occupancy for one agent window.

    Percentages of *window* use ``window_limit`` when known; otherwise the
    billed occupancy is the denominator (limit unknown).
    """
    empty_buckets = {key: 0 for key, _label, _color in USAGE_CATEGORIES}
    if not steps:
        return {
            "buckets": empty_buckets,
            "occupancy": 0,
            "window_limit": None,
            "loaded_pct": None,
            "agent_id": "",
            "peak_step": None,
            "scaled": False,
        }

    target = pressure_agent_id(agent_key)
    agent_id, peak_step, occupancy = _peak_occupancy_anchor(steps, target)
    window_limit = infer_context_window_limit(steps, raw)
    buckets = dict(empty_buckets)
    raw_dict = raw if isinstance(raw, dict) else None
    buckets["tools"] = _raw_blob_tokens(raw_dict, _TOOL_DEF_KEYS)
    buckets["rules"] = _raw_blob_tokens(raw_dict, _RULE_KEYS)
    buckets["mcp"] = _raw_blob_tokens(raw_dict, _MCP_KEYS)

    if peak_step >= 0:
        start = _window_start_step(steps, agent_id, peak_step)
        last_comp = _last_compaction_step(steps, agent_id, peak_step)
        if last_comp is not None and last_comp < start:
            for step in steps:
                if not isinstance(step, dict):
                    continue
                if int(step.get("index", 0)) != last_comp:
                    continue
                if _pressure_agent(step) != agent_id:
                    continue
                _accumulate_step(step, buckets, summarized_only=True)
        for step in steps:
            if not isinstance(step, dict):
                continue
            idx = int(step.get("index", 0))
            if idx < start or idx > peak_step:
                continue
            if _pressure_agent(step) != agent_id:
                continue
            _accumulate_step(step, buckets)

    accounted = sum(buckets[key] for key in _ACCOUNTABLE_KEYS)
    scaled = False
    if occupancy > accounted:
        buckets["unattributed"] = occupancy - accounted
    elif occupancy > 0 and accounted > occupancy:
        scale = occupancy / accounted
        scaled_vals = {key: int(buckets[key] * scale) for key in _ACCOUNTABLE_KEYS}
        if sum(scaled_vals.values()) == 0:
            largest = max(_ACCOUNTABLE_KEYS, key=lambda key: buckets[key])
            scaled_vals[largest] = occupancy
        else:
            drift = occupancy - sum(scaled_vals.values())
            target = max(_ACCOUNTABLE_KEYS, key=lambda key: (scaled_vals[key], buckets[key]))
            scaled_vals[target] = max(0, scaled_vals[target] + drift)
        buckets.update(scaled_vals)
        buckets["unattributed"] = 0
        scaled = True

    loaded_pct = None
    if isinstance(window_limit, (int, float)) and window_limit > 0 and occupancy > 0:
        loaded_pct = round(100.0 * occupancy / window_limit, 1)

    return {
        "buckets": buckets,
        "occupancy": occupancy,
        "window_limit": window_limit if isinstance(window_limit, (int, float)) else None,
        "loaded_pct": loaded_pct,
        "agent_id": agent_id,
        "peak_step": peak_step if peak_step >= 0 else None,
        "scaled": scaled,
        "agent_key": agent_key or PRESSURE_ALL_AGENTS,
    }


def usage_segments(breakdown: dict) -> list[dict]:
    """Legend rows for buckets with a non-zero token estimate."""
    buckets: dict[str, int] = breakdown.get("buckets") or {}
    occupancy = int(breakdown.get("occupancy") or 0)
    window = breakdown.get("window_limit")
    denom_window = int(window) if isinstance(window, (int, float)) and window > 0 else occupancy
    rows: list[dict] = []
    for key, label, color in USAGE_CATEGORIES:
        tokens = int(buckets.get(key) or 0)
        if tokens <= 0:
            continue
        window_pct = round(100.0 * tokens / denom_window, 1) if denom_window else 0.0
        loaded_pct = round(100.0 * tokens / occupancy, 1) if occupancy else 0.0
        rows.append({
            "key": key,
            "label": label,
            "color": color,
            "tokens": tokens,
            "window_pct": window_pct,
            "loaded_pct": loaded_pct,
        })
    return rows


def format_context_usage_html(breakdown: dict) -> str:
    """Stacked bar + legend for a :func:`context_usage_breakdown` result."""
    occupancy = int(breakdown.get("occupancy") or 0)
    window = breakdown.get("window_limit")
    has_window = isinstance(window, (int, float)) and window > 0
    loaded_pct = breakdown.get("loaded_pct")
    rows = usage_segments(breakdown)
    if occupancy <= 0 and not any(int(r.get("tokens") or 0) for r in rows):
        return ""

    if has_window and isinstance(loaded_pct, (int, float)):
        head_left = f"{html.escape(f'{loaded_pct:g}')}% full"
        head_right = (
            f"{html.escape(format_token_count(occupancy))} / "
            f"{html.escape(format_token_count(int(window)))}"
        )
    elif has_window:
        head_left = "Context loaded"
        head_right = (
            f"{html.escape(format_token_count(occupancy))} / "
            f"{html.escape(format_token_count(int(window)))}"
        )
    else:
        head_left = "Context loaded"
        head_right = (
            f"{html.escape(format_token_count(occupancy))} tokens"
            " · window limit unknown"
        )

    bar_total = int(window) if has_window else occupancy
    if occupancy > bar_total:
        bar_total = occupancy
    bar_parts: list[str] = []
    filled = 0
    if bar_total > 0:
        for row in rows:
            tokens = int(row.get("tokens") or 0)
            if tokens <= 0:
                continue
            filled += tokens
            width = 100.0 * tokens / bar_total
            label = html.escape(str(row["label"]))
            color = html.escape(str(row["color"]))
            bar_parts.append(
                f'<div class="ctx-usage-seg" style="width:{width:.3f}%;'
                f'background:{color}" title="{label}: {tokens:,} tokens"></div>'
            )
        free = max(0, bar_total - min(filled, bar_total))
        if free:
            width = 100.0 * free / bar_total
            bar_parts.append(
                f'<div class="ctx-usage-seg ctx-usage-free" style="width:{width:.3f}%"'
                ' title="Free"></div>'
            )
    bar_html = f'<div class="ctx-usage-bar">{"".join(bar_parts)}</div>'

    show_window_col = has_window
    legend_rows: list[str] = []
    for row in rows:
        tokens = int(row.get("tokens") or 0)
        name = (
            f'<span class="ctx-usage-swatch" style="background:{html.escape(str(row["color"]))}"></span>'
            f'{html.escape(str(row["label"]))}'
        )
        token_cell = html.escape(format_token_count(tokens))
        loaded_cell = html.escape(f'{row["loaded_pct"]:g}%')
        if show_window_col:
            window_cell = html.escape(f'{row["window_pct"]:g}%')
            legend_rows.append(
                f'<div class="ctx-usage-name">{name}</div>'
                f'<div class="ctx-usage-tokens">{token_cell}</div>'
                f'<div class="ctx-usage-pct-cell">{window_cell}</div>'
                f'<div class="ctx-usage-pct-cell">{loaded_cell}</div>'
            )
        else:
            legend_rows.append(
                f'<div class="ctx-usage-name">{name}</div>'
                f'<div class="ctx-usage-tokens">{token_cell}</div>'
                f'<div class="ctx-usage-pct-cell">{loaded_cell}</div>'
            )

    if show_window_col:
        legend_head = (
            '<div class="ctx-usage-name ctx-usage-head-cell">Category</div>'
            '<div class="ctx-usage-tokens ctx-usage-head-cell">Tokens</div>'
            '<div class="ctx-usage-pct-cell ctx-usage-head-cell">% window</div>'
            '<div class="ctx-usage-pct-cell ctx-usage-head-cell">% loaded</div>'
        )
        legend_class = "ctx-usage-legend ctx-usage-legend-window"
    else:
        legend_head = (
            '<div class="ctx-usage-name ctx-usage-head-cell">Category</div>'
            '<div class="ctx-usage-tokens ctx-usage-head-cell">Tokens</div>'
            '<div class="ctx-usage-pct-cell ctx-usage-head-cell">% loaded</div>'
        )
        legend_class = "ctx-usage-legend ctx-usage-legend-loaded"

    unattributed = int((breakdown.get("buckets") or {}).get("unattributed") or 0)
    notes: list[str] = [
        "Estimated from logged text (~4 characters/token)."
    ]
    if unattributed:
        notes.append(
            "Unattributed is billed prompt not present in the trajectory "
            "(typically the hidden system prompt and tool schemas)."
        )
    if breakdown.get("scaled"):
        notes.append(
            "Logged text exceeded billed occupancy after compaction; "
            "category sizes were scaled to occupancy."
        )
    note_html = (
        '<div class="ctx-usage-note">'
        + html.escape(" ".join(notes))
        + "</div>"
    )

    return (
        '<div class="ctx-usage">'
        '<div class="ctx-usage-head">'
        f'<span class="ctx-usage-pct">{head_left}</span>'
        f'<span class="ctx-usage-counts">{head_right}</span>'
        "</div>"
        f"{bar_html}"
        f'<div class="{legend_class}">{legend_head}{"".join(legend_rows)}</div>'
        f"{note_html}"
        "</div>"
    )
