"""Trajectory format detection and conversion (Claude Code, OpenCode, CodeArts, generic)."""

import json
import os
import re
from datetime import datetime, timezone
from typing import Any


def safe_get(d: Any, *keys: Any, default: Any = None) -> Any:
    """Safe nested dict access."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def detect_format(raw: dict) -> str:
    """Detect trajectory format: 'ccsession', 'opencode', 'codearts', or 'unknown'."""
    # Post-conversion marker: the Claude Code converter builds a fresh dict and
    # sets this flag. Check it before the raw-format markers below so already-
    # converted trajectories still report as ccsession.
    if raw.get("_cc_format") is True:
        return "ccsession"
    if raw.get("_codex_format") is True:
        return "codex"
    if raw.get("format") == "ccsession-trajectory":
        return "ccsession"
    # CodeArts exports use an OpenCode-compatible ``info + messages``
    # envelope.  Check their explicit export marker before the generic
    # OpenCode shape so the UI does not mislabel the originating product.
    export_metadata = raw.get("export_metadata")
    if raw.get("_codearts_format") is True or (
        isinstance(export_metadata, dict)
        and export_metadata.get("schema_version") == 2
        and export_metadata.get("source_format") == "codearts_opencode_sqlite"
        and isinstance(raw.get("info"), dict)
        and isinstance(raw.get("messages"), list)
    ):
        return "codearts"
    if isinstance(raw.get("info"), dict) and isinstance(raw.get("messages"), list):
        return "opencode"
    return "unknown"


def _iso_to_epoch_ms(iso_str: str | None) -> int | None:
    """Convert ISO 8601 string to epoch milliseconds."""
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return None


def _cc_extract_usage(usage: dict | None) -> dict:
    """Extract token usage from a Claude Code message's usage field."""
    if not isinstance(usage, dict):
        return {"total": 0, "input": 0, "output": 0, "reasoning": 0,
                "cache": {"read": 0, "write": 0}}
    inp = usage.get("input_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    cache_write = usage.get("cache_creation_input_tokens", 0) or 0
    # input_tokens, cache_creation_input_tokens and cache_read_input_tokens are
    # mutually exclusive in the Anthropic usage object, so the processed total is
    # their sum; cache_creation must be included to agree with the session total.
    total = inp + out + cache_read + cache_write
    return {
        "total": total, "input": inp, "output": out, "reasoning": 0,
        "cache": {"read": cache_read, "write": cache_write},
    }


def _cc_content_to_parts(content: list, tool_result_map: dict | None = None) -> list:
    """Convert Claude Code content[] items to internal parts format."""
    parts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        ctype = item.get("type", "")

        if ctype == "text":
            parts.append({"type": "text", "text": item.get("text", "")})
        elif ctype == "thinking":
            parts.append({"type": "reasoning", "text": item.get("text", "")})
        elif ctype == "tool_call" or ctype == "tool_use":
            tool_id = item.get("id", "")
            tool_name = item.get("name", "?")
            tool_input = item.get("input", {})
            # Check if we already have a result for this tool call
            result_info = (tool_result_map or {}).get(tool_id, {})
            status = result_info.get("status", "?")
            output = result_info.get("output", "")
            error = result_info.get("error")
            tool_exec = result_info.get("tool_execution")
            metadata = dict(tool_exec) if isinstance(tool_exec, dict) else {}
            parts.append({
                "type": "tool_call",
                "tool_name": tool_name,
                "tool_id": tool_id,
                "status": status,
                "title": item.get("caller", {}).get("type", "") if isinstance(item.get("caller"), dict) else "",
                "input": tool_input,
                "output": output,
                "error": error,
                "time_start": None,
                "time_end": None,
                "metadata": metadata,
            })
        elif ctype == "tool_result":
            # These are handled via tool_result_map; skip as standalone parts
            pass
        else:
            parts.append({"type": ctype, "raw": item})
    return parts


def _cc_build_tool_result_map(entries: list) -> dict:
    """Build a map of tool_use_id -> {output, status, error} from all user messages."""
    result_map: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role", "")
        content = entry.get("content", [])
        if not isinstance(content, list):
            continue

        # Direct user messages with tool_result
        if role == "user":
            tool_exec = entry.get("tool_execution")
            if not isinstance(tool_exec, dict):
                tool_exec = None
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    tool_id = item.get("tool_use_id", "") or item.get("tool_call_id", "")
                    is_error = item.get("is_error", False)
                    result_content = item.get("output", item.get("content", ""))
                    if tool_id:
                        result_map[tool_id] = {
                            "output": result_content,
                            "status": "error" if is_error else "success",
                            "error": result_content if is_error else None,
                            "tool_execution": tool_exec,
                        }

        # Events with nested user messages containing tool_result
        if role == "event":
            data = entry.get("data", {})
            if isinstance(data, dict):
                msg = data.get("message", {})
                if isinstance(msg, dict) and msg.get("type") == "user":
                    inner_msg = msg.get("message", {})
                    if isinstance(inner_msg, dict):
                        inner_tool_exec = inner_msg.get("tool_execution")
                        if not isinstance(inner_tool_exec, dict):
                            inner_tool_exec = None
                        inner_content = inner_msg.get("content", [])
                        if isinstance(inner_content, list):
                            for item in inner_content:
                                if isinstance(item, dict) and item.get("type") == "tool_result":
                                    tool_id = item.get("tool_use_id", "") or item.get("tool_call_id", "")
                                    is_error = item.get("is_error", False)
                                    result_content = item.get("output", item.get("content", ""))
                                    if tool_id:
                                        result_map[tool_id] = {
                                            "output": result_content,
                                            "status": "error" if is_error else "success",
                                            "error": result_content if is_error else None,
                                            "tool_execution": inner_tool_exec,
                                        }
    return result_map


def _cc_group_by_message_id(entries: list[dict]) -> list[dict]:
    """Group trajectory entries that share the same message_id into merged entries.

    Claude Code splits a single API response into multiple trajectory entries
    (one per content block: thinking, text, tool_call). These share the same
    message_id and request_id. We merge them to avoid double-counting tokens.

    The merged entry uses:
    - Content: concatenated from all entries (preserving order)
    - Usage: from the LAST entry (has final/cumulative output_tokens)
    - Timestamp: from the FIRST entry
    - stop_reason: from the LAST entry (only the final chunk has the real stop reason)
    - Other fields: from the first entry
    """
    if not entries:
        return []

    # Separate entries with and without message_id
    grouped: dict[str, list[dict]] = {}
    no_msgid: list[dict] = []
    order: list[str] = []  # Track insertion order of message_ids

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        mid = entry.get("message_id", "")
        role = entry.get("role", "")
        if mid and role in ("user", "assistant"):
            if mid not in grouped:
                grouped[mid] = []
                order.append(mid)
            grouped[mid].append(entry)
        else:
            no_msgid.append(entry)

    # Merge groups
    merged: list[dict] = []
    for mid in order:
        group = grouped[mid]
        if len(group) == 1:
            merged.append(group[0])
            continue

        # Sort by index to preserve order
        group.sort(key=lambda e: e.get("index", 0))
        first = group[0]
        last = group[-1]

        # Merge content arrays
        all_content = []
        for entry in group:
            content = entry.get("content", [])
            if isinstance(content, list):
                all_content.extend(content)

        merged_entry = {**first}
        merged_entry["content"] = all_content
        # Use last entry's usage (has cumulative/final output_tokens)
        merged_entry["usage"] = last.get("usage", first.get("usage"))
        # Use last entry's stop_reason (intermediate chunks have null)
        merged_entry["stop_reason"] = last.get("stop_reason") or first.get("stop_reason")
        merged.append(merged_entry)

    # Interleave back with non-message-id entries in original order
    # Use the index field to maintain ordering
    result = merged + no_msgid
    result.sort(key=lambda e: e.get("index", 0))
    return result


def _cc_convert_entry_to_step(entry: dict, tool_result_map: dict,
                              *, agent_id: str = "") -> dict | None:
    """Convert a single Claude Code trajectory entry to internal step format."""
    role = entry.get("role", "")
    if role not in ("user", "assistant"):
        return None

    content = entry.get("content", [])
    if not isinstance(content, list):
        content = []

    # Skip user messages that contain only tool_result items — their content
    # is already captured in the corresponding tool_call's output field.
    if role == "user" and content:
        has_non_result = any(
            isinstance(item, dict) and item.get("type") != "tool_result"
            for item in content
        )
        if not has_non_result:
            return None

    usage = _cc_extract_usage(entry.get("usage"))
    timestamp_ms = _iso_to_epoch_ms(entry.get("timestamp"))

    parts = _cc_content_to_parts(content, tool_result_map)

    # Compute derived fields
    tool_calls = [p for p in parts if p.get("type") == "tool_call"]
    errors = sum(1 for tc in tool_calls if tc.get("status") == "error")
    has_reasoning = any(p.get("type") == "reasoning" for p in parts)
    text_preview = ""
    for p in parts:
        if p.get("type") == "text" and p.get("text"):
            text_preview = p["text"]
            break
        if p.get("type") == "reasoning" and p.get("text") and not text_preview:
            text_preview = p["text"]
        if p.get("type") == "tool_call" and not text_preview:
            text_preview = f"[Tool: {p['tool_name']}]"

    model = entry.get("model", "")
    finish = entry.get("stop_reason", "")

    return {
        "role": role,
        "tokens": {
            "total": usage["total"],
            "input": usage["input"],
            "output": usage["output"],
            "reasoning": usage["reasoning"],
            "cache_read": usage["cache"]["read"],
            "cache_write": usage["cache"]["write"],
        },
        "duration": None,  # Will be computed from timestamps
        "parts": parts,
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "error_count": errors,
        "has_reasoning": has_reasoning,
        "text_preview": text_preview,
        "finish": finish or "",
        "model_id": model,
        "provider_id": "",
        "time_created_ms": timestamp_ms,
        "time_completed_ms": None,
        "agent": agent_id or "",
        "mode": "",
        "message_id": entry.get("message_id", entry.get("uuid", "")),
        "id": entry.get("uuid", ""),
        "parent_id": entry.get("parent_uuid", ""),
        "session_id": "",
        "cwd": entry.get("cwd", ""),
        "root": "",
    }


def _cc_flatten_events(entries: list, tool_result_map: dict) -> list[dict]:
    """Flatten event entries with nested sub-agent messages into steps."""
    steps = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("role") != "event":
            continue
        event_type = entry.get("event_type", "")
        if event_type != "progress":
            continue

        data = entry.get("data", {})
        if not isinstance(data, dict):
            continue

        agent_id = data.get("agentId", "")
        msg = data.get("message", {})
        if not isinstance(msg, dict):
            continue

        msg_type = msg.get("type", "")
        inner_msg = msg.get("message", {})
        if not isinstance(inner_msg, dict):
            continue

        # Map event message type to role
        if msg_type == "assistant":
            inner_content = inner_msg.get("content", [])
            if not isinstance(inner_content, list):
                continue
            usage = _cc_extract_usage(inner_msg.get("usage"))
            timestamp_ms = _iso_to_epoch_ms(msg.get("timestamp") or entry.get("timestamp"))

            parts = _cc_content_to_parts(inner_content, tool_result_map)
            tool_calls = [p for p in parts if p.get("type") == "tool_call"]
            errors = sum(1 for tc in tool_calls if tc.get("status") == "error")
            has_reasoning = any(p.get("type") == "reasoning" for p in parts)
            text_preview = ""
            for p in parts:
                if p.get("type") in ("text", "reasoning") and p.get("text"):
                    text_preview = p["text"]
                    break
                if p.get("type") == "tool_call" and not text_preview:
                    text_preview = f"[Tool: {p['tool_name']}]"

            model = inner_msg.get("model", "")
            finish = inner_msg.get("stop_reason", "")

            steps.append({
                "role": "assistant",
                "tokens": {
                    "total": usage["total"],
                    "input": usage["input"],
                    "output": usage["output"],
                    "reasoning": usage["reasoning"],
                    "cache_read": usage["cache"]["read"],
                    "cache_write": usage["cache"]["write"],
                },
                "duration": None,
                "parts": parts,
                "tool_calls": tool_calls,
                "tool_call_count": len(tool_calls),
                "error_count": errors,
                "has_reasoning": has_reasoning,
                "text_preview": text_preview,
                "finish": finish or "",
                "model_id": model,
                "provider_id": "",
                "time_created_ms": timestamp_ms,
                "time_completed_ms": None,
                "agent": agent_id,
                "mode": "",
                "message_id": inner_msg.get("id", ""),
                "id": msg.get("uuid", entry.get("uuid", "")),
                "parent_id": entry.get("parent_uuid", ""),
                "session_id": "",
                "cwd": entry.get("cwd", ""),
                "root": "",
            })
        elif msg_type == "user":
            # User messages in events are typically tool results — already handled via
            # tool_result_map. But if they have text content, include them.
            inner_content = inner_msg.get("content", [])
            if not isinstance(inner_content, list):
                continue
            has_text = any(
                isinstance(item, dict) and item.get("type") == "text"
                for item in inner_content
            )
            if has_text:
                timestamp_ms = _iso_to_epoch_ms(msg.get("timestamp") or entry.get("timestamp"))
                parts = _cc_content_to_parts(inner_content, tool_result_map)
                steps.append({
                    "role": "user",
                    "tokens": {"total": 0, "input": 0, "output": 0, "reasoning": 0,
                               "cache_read": 0, "cache_write": 0},
                    "duration": None,
                    "parts": parts,
                    "tool_calls": [],
                    "tool_call_count": 0,
                    "error_count": 0,
                    "has_reasoning": False,
                    "text_preview": next(
                        (p["text"] for p in parts if p.get("type") == "text" and p.get("text")),
                        ""
                    ),
                    "finish": "",
                    "model_id": "",
                    "provider_id": "",
                    "time_created_ms": timestamp_ms,
                    "time_completed_ms": None,
                    "agent": agent_id,
                    "mode": "",
                    "message_id": "",
                    "id": msg.get("uuid", entry.get("uuid", "")),
                    "parent_id": entry.get("parent_uuid", ""),
                    "session_id": "",
                    "cwd": entry.get("cwd", ""),
                    "root": "",
                })

    return steps


def _cc_convert_sub_agent_trajectory(sub_agent: dict) -> list[dict]:
    """Convert a sub_agents[] entry's full trajectory into internal steps.

    The sub_agents[] top-level array contains complete sub-agent sessions with
    their own trajectory arrays — richer than the inline progress events.
    """
    agent_id = sub_agent.get("agent_id", "")
    sa_trajectory = sub_agent.get("trajectory", [])
    if not isinstance(sa_trajectory, list):
        return []

    # Merge entries that share the same message_id before processing
    sa_trajectory = _cc_group_by_message_id(sa_trajectory)

    # Build tool result map from sub-agent's own trajectory
    tool_result_map = _cc_build_tool_result_map(sa_trajectory)

    steps = []
    for entry in sa_trajectory:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role", "")
        if role in ("user", "assistant"):
            step = _cc_convert_entry_to_step(entry, tool_result_map, agent_id=agent_id)
            if step is not None:
                steps.append(step)
    return steps


def _convert_claude_code_to_internal(raw: dict) -> dict:
    """Convert Claude Code (ccsession-trajectory) format to internal format.

    Transforms the raw JSON into the structure expected by parse_steps() and
    compute_metrics(), so downstream code works unchanged.
    """
    import logging
    logger = logging.getLogger(__name__)

    # --- Format version check ---
    fmt_version = raw.get("format_version", "")
    if fmt_version and fmt_version != "1.0":
        logger.warning("Claude Code trajectory format_version %s (expected 1.0) — "
                        "some fields may not parse correctly", fmt_version)

    session = raw.get("session", {}) if isinstance(raw.get("session"), dict) else {}
    stats_raw = raw.get("statistics", {}) if isinstance(raw.get("statistics"), dict) else {}
    trajectory = raw.get("trajectory", []) if isinstance(raw.get("trajectory"), list) else []
    sub_agents_raw = raw.get("sub_agents", []) if isinstance(raw.get("sub_agents"), list) else []

    # --- Metadata ---
    started_at = session.get("started_at", "")
    ended_at = session.get("ended_at", "")
    models_used = session.get("models_used", [])
    model = models_used[0] if isinstance(models_used, list) and models_used else None
    generator = raw.get("generator", {}) if isinstance(raw.get("generator"), dict) else {}

    metadata = {
        "session_id": session.get("id", ""),
        "slug": session.get("slug", ""),
        "directory": session.get("working_directory") or "",
        "directory_name": os.path.basename(session.get("working_directory") or ""),
        "agent": "claude-code",
        "model": model,
        "hostname": "",
        "platform": session.get("platform", ""),
        "python_version": "",
        "timestamp_utc": started_at,
        "branch": session.get("git_branch", ""),
        "ground_truth_patch": "",
        "baseline_commit": "",
        "sanitized": False,
        "server_url": "",
        "server_version": session.get("claude_code_version", ""),
        "generator_name": generator.get("name", ""),
        "generator_version": generator.get("version", ""),
        "format_version": fmt_version,
        "sub_agent_count": stats_raw.get("sub_agent_count", 0),
        "event_count": stats_raw.get("events", 0),
    }

    # --- Timing ---
    duration_seconds = session.get("duration_seconds")
    if duration_seconds is None and started_at and ended_at:
        start_ms = _iso_to_epoch_ms(started_at)
        end_ms = _iso_to_epoch_ms(ended_at)
        if start_ms is not None and end_ms is not None:
            duration_seconds = (end_ms - start_ms) / 1000.0
    timing = {
        "total_duration": round(duration_seconds, 3) if duration_seconds else 0,
        "started_at": started_at,
        "finished_at": ended_at,
    }

    # --- Stats & token usage ---
    tokens_raw = stats_raw.get("tokens", {}) if isinstance(stats_raw.get("tokens"), dict) else {}
    token_usage = {
        "total_tokens": (
            (tokens_raw.get("input", 0) or 0)
            + (tokens_raw.get("output", 0) or 0)
            + (tokens_raw.get("cache_read", 0) or 0)
            + (tokens_raw.get("cache_creation", 0) or 0)
        ),
        "prompt_tokens": tokens_raw.get("input", 0) or 0,
        "completion_tokens": tokens_raw.get("output", 0) or 0,
    }
    stats = {
        "total_messages": stats_raw.get("turns", 0),
        "user_messages": stats_raw.get("user_turns", 0),
        "assistant_messages": stats_raw.get("assistant_turns", 0),
        "total_tool_calls": stats_raw.get("tool_calls", 0),
        "tool_call_breakdown": stats_raw.get("tool_calls_by_name", {}),
        "failed_tool_calls": 0,
        "reasoning_steps": stats_raw.get("assistant_turns", 0),
    }

    # --- Convert main trajectory entries ---
    # Merge entries that share the same message_id to avoid double-counting tokens
    merged_trajectory = _cc_group_by_message_id(trajectory)

    # Build tool result map from merged trajectory
    tool_result_map = _cc_build_tool_result_map(merged_trajectory)

    converted_trajectory = []
    for entry in merged_trajectory:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role", "")
        if role in ("user", "assistant"):
            step = _cc_convert_entry_to_step(entry, tool_result_map)
            if step is not None:
                converted_trajectory.append(step)

    # --- Convert sub-agent trajectories from sub_agents[] ---
    # Prefer the full sub_agents[] trajectories over inline progress events,
    # as they contain complete message content, token usage, and timing.
    sub_agent_ids_processed = set()
    for sa in sub_agents_raw:
        if not isinstance(sa, dict):
            continue
        agent_id = sa.get("agent_id", "")
        sa_steps = _cc_convert_sub_agent_trajectory(sa)
        converted_trajectory.extend(sa_steps)
        if agent_id:
            sub_agent_ids_processed.add(agent_id)

    # Fallback: flatten inline progress events for any sub-agents NOT in sub_agents[].
    # This handles cases where progress events exist but sub_agents[] is missing.
    if not sub_agents_raw:
        event_steps = _cc_flatten_events(merged_trajectory, tool_result_map)
        converted_trajectory.extend(event_steps)
    else:
        # Only flatten events for agents not already processed via sub_agents[]
        event_steps = _cc_flatten_events(merged_trajectory, tool_result_map)
        for step in event_steps:
            if step.get("agent", "") not in sub_agent_ids_processed:
                converted_trajectory.append(step)

    # Sort chronologically for display order.  A step whose timestamp is missing
    # keeps its original position (it carries the previous step's timestamp)
    # instead of collapsing to 0 and jumping ahead of the opening user prompt.
    _orig_order = {id(s): i for i, s in enumerate(converted_trajectory)}
    _carry: dict[int, float | None] = {}
    _last_ts: float | None = None
    for s in converted_trajectory:
        ts = s.get("time_created_ms")
        if isinstance(ts, (int, float)):
            _last_ts = ts
        _carry[id(s)] = _last_ts
    converted_trajectory.sort(key=lambda s: (
        _carry[id(s)] if _carry[id(s)] is not None else float("-inf"),
        _orig_order[id(s)],
    ))

    for idx, step in enumerate(converted_trajectory):
        step["index"] = idx

    # Compute per-step durations WITHIN each agent's own timeline.  A delegated
    # sub-agent runs concurrently while the main agent is blocked on the Task
    # tool, so the gap to the next step in the globally-interleaved list would
    # otherwise charge the main agent's idle wait to the sub-agent's step.
    by_agent: dict[str, list[dict]] = {}
    for step in converted_trajectory:
        by_agent.setdefault(step.get("agent", ""), []).append(step)
    for agent_steps in by_agent.values():
        # Pair consecutive *timestamped* steps within the agent, stepping over
        # any untimed step so its missing timestamp does not blank out the
        # duration of the timestamped step before it.
        timed = [s for s in agent_steps
                 if isinstance(s.get("time_created_ms"), (int, float))]
        for i in range(len(timed) - 1):
            t1 = timed[i]["time_created_ms"]
            t2 = timed[i + 1]["time_created_ms"]
            if t2 >= t1:
                timed[i]["duration"] = round((t2 - t1) / 1000.0, 2)
                timed[i]["time_completed_ms"] = t2

    # --- Sub-agent summary for session display ---
    sub_agent_info = []
    for sa in sub_agents_raw:
        if not isinstance(sa, dict):
            continue
        sa_stats = sa.get("statistics", {}) if isinstance(sa.get("statistics"), dict) else {}
        sa_tokens = sa_stats.get("tokens", {}) if isinstance(sa_stats.get("tokens"), dict) else {}
        sub_agent_info.append({
            "agent_id": sa.get("agent_id", ""),
            "started_at": sa.get("started_at", ""),
            "ended_at": sa.get("ended_at", ""),
            "duration_seconds": sa.get("duration_seconds", 0),
            "spawned_by": sa.get("spawned_by_tool_call_id", ""),
            "turns": sa_stats.get("turns", 0),
            "tool_calls": sa_stats.get("tool_calls", 0),
            "tool_calls_by_name": sa_stats.get("tool_calls_by_name", {}),
            "tokens": {
                "input": sa_tokens.get("input", 0),
                "output": sa_tokens.get("output", 0),
                "cache_read": sa_tokens.get("cache_read", 0),
                "cache_creation": sa_tokens.get("cache_creation", 0),
            },
        })

    # Build the internal format that parse_steps()/compute_metrics() expects.
    return {
        "metadata": metadata,
        "input": {"prompt": "", "prompt_length": 0},
        "output": {"patch": "", "patch_length": 0, "patch_lines": 0,
                    "has_patch": False, "error": None},
        "timing": timing,
        "token_usage": token_usage,
        "stats": stats,
        "trajectory": [],  # Empty — we use _cc_parsed_steps instead
        "_cc_parsed_steps": converted_trajectory,
        "_cc_format": True,
        "_cc_sub_agents": sub_agent_info,
    }


def _convert_opencode_metadata(raw: dict) -> dict:
    """Populate metadata, timing, and output keys from OpenCode info structure.

    Mutates *raw* in place so that compute_metrics() and the session-detail
    renderer can read the standard fields.  Returns the same dict for convenience.
    """
    info = raw.get("info", {})
    if not isinstance(info, dict):
        return raw

    time_info = info.get("time", {}) if isinstance(info.get("time"), dict) else {}
    created_ms = time_info.get("created")
    updated_ms = time_info.get("updated")

    duration_seconds = 0.0
    started_at = ""
    finished_at = ""
    if isinstance(created_ms, (int, float)):
        started_at = datetime.fromtimestamp(created_ms / 1000.0, tz=timezone.utc).isoformat()
    if isinstance(updated_ms, (int, float)):
        finished_at = datetime.fromtimestamp(updated_ms / 1000.0, tz=timezone.utc).isoformat()
    if isinstance(created_ms, (int, float)) and isinstance(updated_ms, (int, float)):
        duration_seconds = round((updated_ms - created_ms) / 1000.0, 3)

    summary = info.get("summary", {}) if isinstance(info.get("summary"), dict) else {}
    model_info = {}
    # Try to get model from first message's info
    messages = raw.get("messages", [])
    if isinstance(messages, list) and messages:
        first_msg_info = messages[0].get("info", {}) if isinstance(messages[0], dict) else {}
        if isinstance(first_msg_info, dict):
            model_info = first_msg_info.get("model", {}) if isinstance(first_msg_info.get("model"), dict) else {}

    # Consolidated OpenCode/CodeArts exports flatten child-session messages into
    # the main messages array.  Count distinct child session IDs instead of
    # hard-coding zero.  A session_manifest is also accepted so a child with
    # no persisted messages is still represented accurately.
    sub_agent_session_ids: set[str] = set()
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict):
            continue
        message_info = message.get("info", {})
        if not isinstance(message_info, dict):
            continue
        agent_name = message_info.get("agent", "")
        is_sub_agent = bool(message_info.get("isSubAgent")) or (
            isinstance(agent_name, str) and agent_name.endswith("(subagent)")
        )
        if not is_sub_agent:
            continue
        child_session_id = message_info.get("sessionID")
        if isinstance(child_session_id, str) and child_session_id:
            sub_agent_session_ids.add(child_session_id)
    manifest = raw.get("session_manifest", [])
    for entry in manifest if isinstance(manifest, list) else []:
        if not isinstance(entry, dict) or not entry.get("depth"):
            continue
        manifest_info = entry.get("info", {})
        child_session_id = manifest_info.get("id") if isinstance(manifest_info, dict) else None
        if isinstance(child_session_id, str) and child_session_id:
            sub_agent_session_ids.add(child_session_id)
    exported_stats = raw.get("statistics", {})
    exported_sub_agent_count = (
        exported_stats.get("subagent_sessions", 0)
        if isinstance(exported_stats, dict)
        else 0
    )
    if not isinstance(exported_sub_agent_count, int):
        exported_sub_agent_count = 0
    sub_agent_count = max(len(sub_agent_session_ids), exported_sub_agent_count)
    exported_event_count = (
        exported_stats.get("event_rows", 0) if isinstance(exported_stats, dict) else 0
    )
    if not isinstance(exported_event_count, int):
        exported_event_count = 0
    manifest_entries = manifest if isinstance(manifest, list) else []
    manifest_event_count = sum(
        len(entry.get("events", []))
        for entry in manifest_entries if isinstance(entry, dict)
        if isinstance(entry.get("events", []), list)
    )
    event_count = max(exported_event_count, manifest_event_count)

    raw["metadata"] = {
        "session_id": info.get("id", ""),
        "slug": info.get("slug", ""),
        "title": info.get("title", ""),
        "directory": info.get("directory") or "",
        "directory_name": (info.get("directory") or "").replace("\\", "/").rsplit("/", 1)[-1],
        "agent": "opencode",
        "model": model_info.get("modelID", ""),
        "hostname": "",
        "platform": "",
        "python_version": "",
        "timestamp_utc": started_at,
        "branch": "",
        "ground_truth_patch": "",
        "baseline_commit": "",
        "sanitized": False,
        "server_url": "",
        "server_version": info.get("version", ""),
        "generator_name": "opencode",
        "generator_version": info.get("version", ""),
        "format_version": "",
        "sub_agent_count": sub_agent_count,
        "event_count": event_count,
    }
    raw["timing"] = {
        "total_duration": duration_seconds,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    raw["output"] = {
        "patch": "", "patch_length": 0, "patch_lines": 0,
        "has_patch": bool(summary.get("additions") or summary.get("deletions")),
        "error": None,
        "additions": summary.get("additions", 0),
        "deletions": summary.get("deletions", 0),
        "files_changed": summary.get("files", 0),
    }
    raw.setdefault("input", {"prompt": "", "prompt_length": 0})

    # Compute raw stats from messages array
    messages = raw.get("messages", [])
    user_count = 0
    asst_count = 0
    total_tool_calls = 0
    tool_breakdown: dict[str, int] = {}
    total_input = 0
    total_output = 0
    for msg in (messages if isinstance(messages, list) else []):
        if not isinstance(msg, dict):
            continue
        msg_info = msg.get("info", {}) if isinstance(msg.get("info"), dict) else {}
        role = msg_info.get("role", "")
        if role == "user":
            user_count += 1
        elif role == "assistant":
            asst_count += 1
        # Count tokens
        tok = msg_info.get("tokens", {}) if isinstance(msg_info.get("tokens"), dict) else {}
        total_input += tok.get("input", 0) or 0
        total_output += tok.get("output", 0) or 0
        # Count tool calls from parts
        for part in (msg.get("parts", []) if isinstance(msg.get("parts"), list) else []):
            if isinstance(part, dict) and part.get("type") == "tool":
                total_tool_calls += 1
                tname = part.get("tool", "?")
                tool_breakdown[tname] = tool_breakdown.get(tname, 0) + 1

    raw.setdefault("token_usage", {
        "total_tokens": total_input + total_output,
        "prompt_tokens": total_input,
        "completion_tokens": total_output,
    })
    raw.setdefault("stats", {
        "total_messages": user_count + asst_count,
        "user_messages": user_count,
        "assistant_messages": asst_count,
        "total_tool_calls": total_tool_calls,
        "tool_call_breakdown": tool_breakdown,
        "failed_tool_calls": 0,
        "reasoning_steps": asst_count,
    })
    return raw


def _convert_codearts_metadata(raw: dict) -> dict:
    """Normalize a CodeArts SQLite export without changing message data.

    CodeArts deliberately shares the OpenCode message/part schema, so the
    existing OpenCode normalization is the correct base.  This adapter keeps
    that behavior while restoring CodeArts product identity, export metadata,
    and the parent/sub-agent counts already present in the consolidated file.
    Token values are per-message values, not cumulative totals.
    """
    _convert_opencode_metadata(raw)

    export = raw.get("export_metadata", {})
    if not isinstance(export, dict):
        export = {}
    exported_stats = raw.get("statistics", {})
    if not isinstance(exported_stats, dict):
        exported_stats = {}
    manifest = raw.get("session_manifest", [])
    if not isinstance(manifest, list):
        manifest = []

    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        raw["metadata"] = metadata

    # Prefer the first explicit per-message model identifier.  User messages
    # commonly store it in ``info.model`` while assistant messages place it
    # directly in ``info.modelID``.
    model_id = metadata.get("model", "")
    for msg in raw.get("messages", []):
        if not isinstance(msg, dict):
            continue
        msg_info = msg.get("info", {})
        if not isinstance(msg_info, dict):
            continue
        nested_model = msg_info.get("model", {})
        candidate = msg_info.get("modelID")
        if not candidate and isinstance(nested_model, dict):
            candidate = nested_model.get("modelID")
        if candidate:
            model_id = candidate
            break

    # sub_agent_count/event_count were already derived by the shared
    # OpenCode normalization (isSubAgent messages, session_manifest, and
    # exporter statistics) — reuse them instead of recomputing.
    sub_agent_count = metadata.get("sub_agent_count", 0)
    event_count = metadata.get("event_count", 0)
    session_count = exported_stats.get("sessions")
    if not isinstance(session_count, int):
        session_count = len(manifest) or 1

    metadata.update({
        "agent": "codearts",
        "model": model_id,
        "generator_name": "codearts",
        "generator_version": metadata.get("server_version", ""),
        "format_version": str(export.get("schema_version", 2)),
        "sub_agent_count": sub_agent_count,
        "session_count": session_count,
        "event_count": event_count,
        "export_generated_at": export.get("generated_at", ""),
        "export_complete": export.get("complete"),
        "export_warnings": export.get("warnings", []),
    })

    # Use the authoritative exporter statistics where provided, while keeping
    # the tool-name breakdown calculated by the OpenCode-compatible adapter.
    stats = raw.get("stats", {})
    if not isinstance(stats, dict):
        stats = {}
        raw["stats"] = stats
    stats.update({
        "total_messages": exported_stats.get("total_messages", stats.get("total_messages", 0)),
        "user_messages": exported_stats.get("user_messages", stats.get("user_messages", 0)),
        "assistant_messages": exported_stats.get("assistant_messages", stats.get("assistant_messages", 0)),
        "total_tool_calls": exported_stats.get("tool_parts", stats.get("total_tool_calls", 0)),
        "reasoning_steps": exported_stats.get("reasoning_parts", stats.get("reasoning_steps", 0)),
        "sub_agent_count": sub_agent_count,
    })

    token_totals = {
        "total": 0, "input": 0, "output": 0, "reasoning": 0,
        "cache_read": 0, "cache_write": 0,
    }
    for msg in raw.get("messages", []):
        if not isinstance(msg, dict):
            continue
        msg_info = msg.get("info", {})
        tokens = msg_info.get("tokens", {}) if isinstance(msg_info, dict) else {}
        if not isinstance(tokens, dict):
            continue
        token_totals["total"] += tokens.get("total", 0) or 0
        token_totals["input"] += tokens.get("input", 0) or 0
        token_totals["output"] += tokens.get("output", 0) or 0
        token_totals["reasoning"] += tokens.get("reasoning", 0) or 0
        cache = tokens.get("cache", {})
        if isinstance(cache, dict):
            token_totals["cache_read"] += cache.get("read", 0) or 0
            token_totals["cache_write"] += cache.get("write", 0) or 0

    raw["token_usage"] = {
        "total_tokens": token_totals["total"],
        "prompt_tokens": token_totals["input"],
        "completion_tokens": token_totals["output"],
        "reasoning_tokens": token_totals["reasoning"],
        "cache_read_tokens": token_totals["cache_read"],
        "cache_write_tokens": token_totals["cache_write"],
    }
    raw["_codearts_format"] = True
    raw["_source_format"] = "codearts"
    raw["_capabilities"] = {
        "has_timing": True,
        "has_tool_calls": bool(stats.get("total_tool_calls")),
        "has_runtime_token_usage": True,
        "has_reasoning_content": bool(exported_stats.get("reasoning_parts")),
        "has_session_hierarchy": bool(sub_agent_count),
    }
    return raw


_TOOL_ERROR_PATTERNS = [
    # Platform / shell-level
    ("command not found", "platform_error"),
    ("not recognized as", "platform_error"),  # Windows "not recognized as internal or external command"
    ("command timed out", "platform_error"),
    # Permission / policy
    ("Permission denied", "permission_error"),
    ("permission denied", "permission_error"),
    ("rule which prevents", "permission_error"),  # OpenCode: "user has specified a rule which prevents..."
    ("must read file", "permission_error"),       # OpenCode: edit/write before read enforcement
    # Missing file / path
    ("No such file or directory", "missing_file"),
    ("cannot find the path", "missing_file"),
    ("ENOENT", "missing_file"),
    # Bad input / invalid args
    ("out of range", "bad_input"),
    # Generic tool failure (catch-all for tools that errored without a recognized cause)
    ("ripgrep failed", "tool_error"),
]


def _classify_tool_error(output: str) -> str | None:
    """Classify a Bash tool output as an error type, or None if no error detected."""
    if not output:
        return None
    for pattern, error_type in _TOOL_ERROR_PATTERNS:
        if pattern in output:
            return error_type
    return None


# ---------------------------------------------------------------------------


def _convert_codex_to_internal(events: list[dict]) -> dict:
    """Convert Codex CLI JSONL events into the trajviz internal format.

    Codex emits newline-delimited JSON with event types:
    - session_meta: session ID, cwd, model, version
    - turn_context: turn metadata
    - response_item: messages (user/assistant/developer), function_call, function_call_output, reasoning
    - event_msg: task_started, task_complete, token_count, agent_message

    We group these into "messages" matching the OpenCode internal format:
    each assistant turn = one message with parts (text, reasoning, tool calls).
    """
    # Extract session metadata
    session_meta = {}
    for e in events:
        if isinstance(e, dict) and e.get("type") == "session_meta":
            payload = e.get("payload")
            session_meta = payload if isinstance(payload, dict) else {}
            break

    # Group events into assistant turns.
    # Pattern: user message → (reasoning → assistant text → function_calls → function_call_outputs)* → task_complete
    messages: list[dict] = []
    pending_tool_calls: dict[str, dict] = {}  # call_id -> function_call payload
    current_parts: list[dict] = []
    current_role = None
    current_timestamp = None
    current_tokens: dict | None = None
    previous_cumulative_usage: dict | None = None
    previous_usage_snapshot: tuple | None = None

    usage_fields = (
        "total_tokens", "input_tokens", "output_tokens",
        "reasoning_output_tokens", "cached_input_tokens",
    )

    def _usage_value(usage: dict, field: str) -> int | float:
        value = usage.get(field, 0)
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0

    def _flush_message():
        nonlocal current_parts, current_role, current_timestamp, current_tokens
        if current_parts and current_role:
            info_block = {
                "role": current_role,
                "time": {"created": current_timestamp or 0},
            }
            if current_tokens:
                info_block["tokens"] = current_tokens
            messages.append({"info": info_block, "parts": current_parts})
        current_parts = []
        current_role = None
        current_timestamp = None
        current_tokens = None

    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        # Codex records timestamps as ISO-8601 strings; the internal contract is
        # epoch milliseconds, so convert here (parse_steps and every timing
        # consumer discard non-numeric timestamps).
        ts = _iso_to_epoch_ms(event.get("timestamp"))

        if etype == "response_item":
            payload = event.get("payload", {}) if isinstance(event.get("payload"), dict) else {}
            item_type = payload.get("type", "")
            role = payload.get("role", "")

            if item_type == "message":
                # Role change → flush previous message
                if role != current_role and current_parts:
                    _flush_message()

                current_role = role
                if not current_timestamp:
                    current_timestamp = ts

                for content in (payload.get("content") or []):
                    if not isinstance(content, dict):
                        continue
                    ctype = content.get("type", "")
                    text = content.get("text", "")
                    if ctype == "output_text":
                        current_parts.append({"type": "text", "text": text})
                    elif ctype == "input_text":
                        current_parts.append({"type": "text", "text": text})

            elif item_type == "reasoning":
                # If we already have tool call parts, this reasoning starts a new turn
                has_tool_parts = any(p.get("type") in ("tool_call", "tool") for p in current_parts)
                if has_tool_parts and current_role == "assistant":
                    _flush_message()
                current_role = current_role or "assistant"
                if not current_timestamp:
                    current_timestamp = ts
                summary = payload.get("summary", [])
                summary_text = " ".join(
                    s.get("text", "") for s in summary if isinstance(s, dict)
                ) if isinstance(summary, list) else ""
                current_parts.append({"type": "reasoning", "text": summary_text})

            elif item_type in ("function_call", "custom_tool_call"):
                if current_role != "assistant" and current_parts:
                    _flush_message()
                current_role = "assistant"
                if not current_timestamp:
                    current_timestamp = ts

                call_id = payload.get("call_id", "")
                name = payload.get("name", "exec_command")
                args_str = payload.get("arguments") or payload.get("input") or "{}"
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    args = {"raw": args_str}
                if not isinstance(args, dict):
                    args = {"raw": args_str}

                # Determine tool name and build normalized input
                cmd = args.get("cmd") or args.get("command") or ""
                if not cmd and isinstance(args_str, str):
                    if name == "exec":
                        commands = _extract_codex_exec_commands(args_str)
                        cmd = " ; ".join(commands) if commands else args_str
                    elif name == "apply_patch":
                        cmd = args_str
                tool_name = _classify_codex_command(name, cmd)
                normalized_input = _build_codex_tool_input(tool_name, cmd, args, name)

                pending_tool_calls[call_id] = {
                    "name": name,
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "input": normalized_input,
                    "cmd": cmd,
                }

            elif item_type in ("function_call_output", "custom_tool_call_output"):
                call_id = payload.get("call_id", "")
                output = payload.get("output", "")
                tc = pending_tool_calls.pop(call_id, {})

                # Determine status from output
                status = "success"
                if isinstance(output, str):
                    if "error" in output.lower()[:200] or "traceback" in output.lower()[:200]:
                        status = "error"
                    # Check exit code in output
                    if "exited with code" in output and "code 0" not in output:
                        status = "error"
                    try:
                        output_data = json.loads(output)
                    except json.JSONDecodeError:
                        output_data = None
                    if isinstance(output_data, dict):
                        metadata = output_data.get("metadata")
                        if isinstance(metadata, dict):
                            exit_code = metadata.get("exit_code")
                            if isinstance(exit_code, int) and exit_code != 0:
                                status = "error"

                current_parts.append({
                    "type": "tool_call" if tc else "tool",
                    "tool_name": tc.get("tool_name", "Bash"),
                    "tool_id": call_id,
                    "status": status,
                    "input": tc.get("input", {}),
                    "output": output,
                })

        elif etype == "event_msg":
            payload = event.get("payload", {}) if isinstance(event.get("payload"), dict) else {}
            msg_type = payload.get("type", "")
            if msg_type == "token_count":
                # last_token_usage is the per-response delta. Codex can repeat an
                # identical notification, so use the cumulative + last snapshots
                # together to de-duplicate it. Cumulative counters alone are not a
                # safe key: parallel agents can report the same cumulative value
                # with different, legitimate last-usage deltas.
                usage = payload.get("info", {}) if isinstance(payload.get("info"), dict) else {}
                last_usage = usage.get("last_token_usage")
                cumulative = usage.get("total_token_usage")
                tu = None
                cumulative_snapshot = (
                    tuple(_usage_value(cumulative, field) for field in usage_fields)
                    if isinstance(cumulative, dict) else None
                )
                last_snapshot = (
                    tuple(_usage_value(last_usage, field) for field in usage_fields)
                    if isinstance(last_usage, dict) else None
                )
                snapshot = (cumulative_snapshot, last_snapshot)
                if snapshot != (None, None):
                    if snapshot == previous_usage_snapshot:
                        continue
                    previous_usage_snapshot = snapshot

                if isinstance(last_usage, dict):
                    tu = last_usage
                elif isinstance(cumulative, dict):
                    if previous_cumulative_usage is None or any(
                        _usage_value(cumulative, field)
                        < _usage_value(previous_cumulative_usage, field)
                        for field in usage_fields
                    ):
                        tu = cumulative
                    else:
                        tu = {
                            field: _usage_value(cumulative, field)
                            - _usage_value(previous_cumulative_usage, field)
                            for field in usage_fields
                        }

                if isinstance(cumulative, dict):
                    previous_cumulative_usage = {
                        field: _usage_value(cumulative, field) for field in usage_fields
                    }

                if isinstance(tu, dict):
                    token_delta = {
                        "total": _usage_value(tu, "total_tokens"),
                        "input": _usage_value(tu, "input_tokens"),
                        "output": _usage_value(tu, "output_tokens"),
                        "reasoning": _usage_value(tu, "reasoning_output_tokens"),
                        "cache": {"read": _usage_value(tu, "cached_input_tokens"), "write": 0},
                    }
                    if not any(token_delta[field] for field in ("total", "input", "output", "reasoning")) \
                            and not token_delta["cache"]["read"]:
                        continue
                    if current_tokens is None:
                        current_tokens = token_delta
                    else:
                        # A displayed step may contain several model responses
                        # (for example commentary followed by a tool call). Each
                        # response has its own last_token_usage delta, so retain
                        # all of them instead of replacing the earlier usage.
                        for field in ("total", "input", "output", "reasoning"):
                            current_tokens[field] += token_delta[field]
                        current_tokens["cache"]["read"] += token_delta["cache"]["read"]
            elif msg_type == "task_complete":
                _flush_message()

    # Flush any remaining parts
    _flush_message()

    # Approximate each turn's completion as the next turn's start so per-step
    # durations exist (Codex is single-session; parse_steps backfills the final
    # turn from the session end timestamp).
    for i in range(len(messages) - 1):
        cur_t = messages[i]["info"]["time"].get("created")
        nxt_t = messages[i + 1]["info"]["time"].get("created")
        if isinstance(cur_t, (int, float)) and isinstance(nxt_t, (int, float)) and nxt_t >= cur_t:
            messages[i]["info"]["time"]["completed"] = nxt_t

    first_ts_iso = events[0].get("timestamp") if events and isinstance(events[0], dict) else None
    last_ts_iso = events[-1].get("timestamp") if events and isinstance(events[-1], dict) else None
    directory = session_meta.get("cwd", "") or ""
    start_ms = _iso_to_epoch_ms(first_ts_iso)
    end_ms = _iso_to_epoch_ms(last_ts_iso)
    total_duration = (
        round((end_ms - start_ms) / 1000.0, 3)
        if isinstance(start_ms, int) and isinstance(end_ms, int) else 0
    )
    total_tokens = sum(
        (m["info"].get("tokens") or {}).get("total", 0) for m in messages
    )

    # Build metadata
    info = {
        "id": session_meta.get("id", ""),
        "slug": "",
        "projectID": "",
        "directory": directory,
        "title": "",
        "version": session_meta.get("cli_version", ""),
        "time": {"created": start_ms or 0, "updated": end_ms or 0},
    }

    return {
        "info": info,
        "messages": messages,
        "metadata": {
            "session_id": session_meta.get("id", ""),
            "directory": directory,
            "directory_name": directory.replace("\\", "/").rsplit("/", 1)[-1],
            "agent": "codex",
            "model": session_meta.get("model", "") or "",
            "source": "codex",
            "model_provider": session_meta.get("model_provider", "openai"),
            "originator": session_meta.get("originator", "Codex CLI"),
            "server_version": session_meta.get("cli_version", ""),
            "timestamp_utc": first_ts_iso or "",
        },
        "timing": {
            "total_duration": total_duration,
            "started_at": first_ts_iso or "",
            "finished_at": last_ts_iso or "",
        },
        "output": {},
        "input": {},
        "token_usage": {"total_tokens": total_tokens},
        "stats": {},
        "_codex_format": True,
    }


def _extract_codex_exec_commands(source: str) -> list[str]:
    """Extract JSON-quoted ``cmd`` values from a modern custom ``exec`` input."""
    if not isinstance(source, str):
        return []
    pattern = re.compile(r'(?:["\']?cmd["\']?)\s*:\s*("(?:\\.|[^"\\])*")')
    commands = []
    for match in pattern.finditer(source):
        try:
            command = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(command, str):
            commands.append(command)
    return commands


def _classify_codex_command(func_name: str, cmd: str) -> str:
    """Map a Codex exec_command to a trajviz tool name.

    Codex uses exec_command for everything; we infer the intent from the command.
    """
    func_lower = str(func_name).lower().strip()
    cmd_lower = str(cmd).lower().strip()

    if func_lower == "apply_patch" or "tools.apply_patch" in cmd_lower:
        return "Write"
    if not cmd_lower and func_lower not in ("exec", "exec_command"):
        # Preserve non-shell Codex tools (for example spawn_agent or wait)
        # instead of flattening every call into an empty Bash command.
        return str(func_name) or "Bash"

    # File reading
    if any(cmd_lower.startswith(p) for p in ["cat ", "head ", "tail ", "sed -n", "less "]):
        return "Read"
    if cmd_lower.startswith("rg ") or cmd_lower.startswith("grep "):
        return "Grep"
    if cmd_lower.startswith("find ") or cmd_lower.startswith("ls "):
        return "Glob"

    # File writing
    for pattern in ["cat >", "cat >>", "tee ", "echo >", "echo >>",
                     "sed -i", "patch ", "git apply"]:
        if pattern in cmd_lower:
            return "Write"

    # Testing
    if any(p in cmd_lower for p in ["pytest", "python -m pytest", "python -m unittest",
                                     "make test", "tox ", "npm test"]):
        return "Bash"  # test execution

    # Git
    if cmd_lower.startswith("git "):
        return "Bash"

    # Python execution
    if cmd_lower.startswith("python ") or cmd_lower.startswith("python3 "):
        return "Bash"

    return "Bash"


def _build_codex_tool_input(
    tool_name: str,
    cmd: str,
    raw_args: dict,
    func_name: str = "",
) -> dict:
    """Build a normalized input dict for Codex commands.

    Maps the raw Codex exec_command args into the format that
    canonical.py expects for each tool type:
    - Read: {"file_path": "..."}
    - Write: {"file_path": "..."}
    - Grep/Glob: {"pattern": "...", "path": "..."}
    - Bash: {"command": "..."}
    """
    raw_text = raw_args.get("raw", "") if isinstance(raw_args, dict) else ""

    if tool_name in ("Read",):
        # Extract file path from commands like:
        #   cat file.py, sed -n '1,20p' file.py, head -n 50 file.py
        #   nl -ba file.py | sed ...
        parts = cmd.split("|")[0].strip().split()  # take before first pipe
        # File path is usually the last arg that doesn't start with - or '
        file_path = ""
        for p in reversed(parts):
            if not p.startswith("-") and not p.startswith("'") and not p.startswith('"'):
                if "." in p or "/" in p:
                    file_path = p
                    break
        return {"file_path": file_path, "command": cmd}

    elif tool_name in ("Write",):
        # Extract file path from: sed -i, cat >, patch, git apply
        patch_match = re.search(
            r"^\*\*\* (?:Add|Update|Delete) File: (.+)$",
            cmd,
            flags=re.MULTILINE,
        )
        if patch_match:
            file_path = patch_match.group(1).strip()
        else:
            parts = cmd.split()
            file_path = ""
            for p in reversed(parts):
                if not p.startswith("-") and ("." in p or "/" in p):
                    file_path = p
                    break
        result = {"file_path": file_path, "command": cmd}
        if str(func_name).lower() == "apply_patch" or cmd.startswith("*** Begin Patch"):
            result["patch"] = raw_text or cmd
        return result

    elif tool_name in ("Grep",):
        # Extract pattern and scope from: rg -n "pattern" dir/
        #   or: grep -rn "pattern" file
        parts = cmd.split()
        non_flag = [p for p in parts[1:] if not p.startswith("-")]
        pattern = non_flag[0].strip("'\"") if non_flag else ""
        path = non_flag[-1] if len(non_flag) > 1 else ""
        return {"pattern": pattern, "path": path, "command": cmd}

    elif tool_name in ("Glob",):
        # Extract path from: find dir, ls dir
        parts = cmd.split()
        path = parts[1] if len(parts) > 1 and not parts[1].startswith("-") else ""
        return {"path": path, "command": cmd}

    else:
        # Bash — pass cmd as "command" so canonical.py can parse it
        if not cmd and isinstance(raw_args, dict) and raw_args:
            return dict(raw_args)
        result = {"command": cmd}
        if raw_text and raw_text != cmd:
            result["raw_input"] = raw_text
        return result


def load_trajectory(file_path: str, format_hint: str | None = None) -> dict:
    """Load trajectory file with error handling.

    Auto-detects format (ccsession / opencode / codearts / codex) and
    normalizes metadata.  Supports ``.json`` and ``.jsonl`` files.

    ``format_hint`` is used as a fallback when automatic detection returns
    ``"unknown"`` — useful for JSON files that lack format-identifying fields
    (e.g., Claude Code exports without the ``format`` marker).
    """
    if file_path.endswith(".log"):
        return {"_error": "Unsupported file type: .log files are no longer supported."}

    # Handle Codex JSONL format
    if file_path.endswith(".jsonl"):
        events = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                pending: tuple[int, str] | None = None
                for line_number, raw_line in enumerate(f, start=1):
                    if not raw_line.strip():
                        continue
                    # Delay one non-empty line so only the true final line can
                    # receive the append-in-progress tolerance below.
                    if pending is None:
                        pending = (line_number, raw_line)
                        continue
                    pending_number, pending_line = pending
                    try:
                        events.append(json.loads(pending_line))
                    except json.JSONDecodeError as exc:
                        return {"_error": f"Invalid JSONL at line {pending_number}: {exc.msg}"}
                    pending = (line_number, raw_line)

                if pending is not None:
                    pending_number, pending_line = pending
                    try:
                        events.append(json.loads(pending_line))
                    except json.JSONDecodeError as exc:
                        # Rollouts are append-only. A missing newline on the final
                        # object is the signal that the writer may still be appending.
                        if pending_line.endswith(("\n", "\r")):
                            return {"_error": f"Invalid JSONL at line {pending_number}: {exc.msg}"}
        except OSError as exc:
            return {"_error": str(exc)}
        if events and isinstance(events[0], dict) and events[0].get("type") == "session_meta":
            result = _convert_codex_to_internal(events)
            result["_source_path"] = file_path
            return result
        return {"_error": "Unsupported JSONL input; expected Codex JSONL format (leading session_meta event)."}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return {"_error": str(exc)}

    fmt = detect_format(raw)
    if fmt == "unknown" and format_hint in ("ccsession", "codearts", "opencode"):
        fmt = format_hint
    if fmt == "ccsession":
        result = _convert_claude_code_to_internal(raw)
    elif fmt == "codearts":
        result = _convert_codearts_metadata(raw)
    elif fmt == "opencode":
        result = _convert_opencode_metadata(raw)
    else:
        result = raw
    result["_source_path"] = file_path
    return result
