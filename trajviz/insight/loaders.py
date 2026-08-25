"""Trajectory format detection and conversion (Claude Code, OpenCode, CodeArts, Codex, Pi, DSH)."""

import json
import os
import re
import zipfile
from datetime import datetime, UTC
from typing import Any


def safe_get(d: Any, *keys: Any, default: Any = None) -> Any:
    """Safe nested dict access."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


# Canonical display names for the detected trajectory formats.  Defined here,
# next to detect_format, as the single source of truth — UI/report modules
# import this mapping instead of maintaining their own drifting copies.
FORMAT_LABELS = {
    "ccsession": "Claude Code",
    "codearts": "CodeArts",
    "opencode": "OpenCode",
    "codex": "Codex CLI",
    "pi": "Pi",
    "dsh": "DeepSeek Harness",
}

# Dropdown choices for the Insight UI. Auto-detect is first so it is the
# default; an explicit format is still available as an override / mismatch gate.
FORMAT_DROPDOWN_CHOICES: list[tuple[str, str]] = [
    ("Auto-detect", ""),
    *((label, key) for key, label in FORMAT_LABELS.items()),
]

# Object-shaped JSON (one file = one dict) vs event-stream JSON/JSONL.
_OBJECT_FORMATS = frozenset({"ccsession", "codearts", "opencode"})
_EVENT_FORMATS = frozenset({"codex", "pi", "dsh"})
_FORMAT_STAMPS = {
    "ccsession": "_cc_format",
    "codearts": "_codearts_format",
    "codex": "_codex_format",
    "pi": "_pi_format",
    "dsh": "_dsh_format",
}


def _looks_like_pi_jsonl(events: list) -> bool:
    """True when an event stream starts with a Pi ``session`` header.

    Distinct from Codex ``session_meta`` and DeepSeek Harness (checked first).
    Requires a non-empty string ``id`` so a generic ``{"type": "session"}``
    log line is not treated as Pi.
    """
    if not events or not isinstance(events[0], dict):
        return False
    first = events[0]
    if first.get("type") != "session":
        return False
    ident = first.get("id")
    return isinstance(ident, str) and bool(ident)


def _looks_like_dsh_jsonl(events: list) -> bool:
    """True when an event stream is a DeepSeek Harness session log.

    DSH and Pi both lead with ``type: session`` plus a string ``id``. DSH
    headers carry ``createdAt`` (epoch ms) and/or ``delegationDepth`` /
    ``agentPreset`` / ``origin: subagent``; body events use slash-separated
    types (``user/message``, ``tool/call``).
    """
    if not events or not isinstance(events[0], dict):
        return False
    first = events[0]
    if first.get("type") != "session":
        return False
    ident = first.get("id")
    if not isinstance(ident, str) or not ident:
        return False
    created = first.get("createdAt")
    if isinstance(created, (int, float)) and not isinstance(created, bool):
        return True
    if "delegationDepth" in first or "agentPreset" in first:
        return True
    if first.get("origin") == "subagent":
        return True
    for event in events[1:12]:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if isinstance(etype, str) and "/" in etype:
            return True
    return False


def _detect_object_format(raw: dict) -> str:
    """Detect format of a parsed JSON object (raw export or already-converted)."""
    # Post-conversion markers: converters build/stamp a dict so a second pass
    # (UI gate, attribution, run-group) still reports the originating product.
    if raw.get("_cc_format") is True:
        return "ccsession"
    if raw.get("_codex_format") is True:
        return "codex"
    if raw.get("_pi_format") is True:
        return "pi"
    if raw.get("_dsh_format") is True:
        return "dsh"
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
    # CodeArts legacy-JSON exports (codearts_consolidator.py emits
    # source_format "codearts_legacy_json") lack the OpenCode "info" dict but
    # are still CodeArts files — detect them so the UI's format-mismatch gate
    # and labeling work instead of falling through to "unknown".
    if (
        isinstance(export_metadata, dict)
        and export_metadata.get("schema_version") == 2
        and export_metadata.get("source_format") == "codearts_legacy_json"
        and isinstance(raw.get("messages"), list)
    ):
        return "codearts"
    if isinstance(raw.get("info"), dict) and isinstance(raw.get("messages"), list):
        return "opencode"
    return "unknown"


def _detect_event_stream_format(events: list) -> str:
    """Detect Codex / DSH / Pi from a parsed event array (JSONL or a JSON list)."""
    if not events or not isinstance(events[0], dict):
        return "unknown"
    if events[0].get("type") == "session_meta":
        return "codex"
    # DSH before Pi: both lead with ``type: session`` + string ``id``.
    if _looks_like_dsh_jsonl(events):
        return "dsh"
    if _looks_like_pi_jsonl(events):
        return "pi"
    return "unknown"


def detect_format(raw: Any) -> str:
    """Detect trajectory format from parsed content.

    Accepts a JSON object or an event array. Returns ``ccsession``,
    ``opencode``, ``codearts``, ``codex``, ``pi``, ``dsh``, or ``unknown``.
    """
    if isinstance(raw, list):
        return _detect_event_stream_format(raw)
    if isinstance(raw, dict):
        return _detect_object_format(raw)
    return "unknown"


def _resolve_format(detected: str, hint: str | None) -> tuple[str, str | None]:
    """Apply ``format_hint`` to a detected format.

    Returns ``(fmt, reason)`` where ``reason`` is ``None`` (proceed),
    ``"unknown"`` (auto-detect found nothing), or ``"mismatch"``.

    Empty hint is auto-detect. A recognized hint on ``unknown`` *forces*
    that converter. A recognized hint on a different detected format is
    a mismatch (including Codex/Pi).
    """
    selected = hint if hint in FORMAT_LABELS else ""
    if not selected:
        if detected in ("", "unknown"):
            return "unknown", "unknown"
        return detected, None
    if detected in ("", "unknown"):
        return selected, None
    if detected != selected:
        return detected, "mismatch"
    return detected, None


def check_format_selection(detected: str, selected: str | None) -> str | None:
    """Return an error reason if the dropdown selection cannot load this file.

    ``None`` means proceed. Empty ``selected`` is auto-detect: any recognized
    format is accepted, and ``unknown`` is rejected so we do not render an
    empty dashboard. An explicit selection forces conversion when detection
    is ``unknown``, and rejects a different recognized format (including
    Codex/Pi).
    """
    _, reason = _resolve_format(detected, selected or None)
    return reason


def _iso_to_epoch_ms(iso_str: str | None) -> int | None:
    """Convert ISO 8601 string to epoch milliseconds.

    Non-string input (e.g. an already-numeric epoch timestamp in a
    hand-edited file) degrades to None (missing timestamp) instead of raising.
    """
    if not iso_str or not isinstance(iso_str, str):
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


def _extract_tool_results(content: list, tool_exec: dict | None, result_map: dict) -> None:
    """Record tool_result items from a user message's content into *result_map*.

    Shared by the direct-user-message and event-nested ingestion paths of
    _cc_build_tool_result_map so the id-fallback and is_error semantics
    cannot diverge between them.
    """
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
            _extract_tool_results(content, tool_exec, result_map)

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
                            _extract_tool_results(inner_content, inner_tool_exec, result_map)
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


def _cc_build_step(parts: list, *, role: str, usage: dict | None = None,
                   timestamp_ms: int | None = None, model: str = "",
                   finish: str = "", agent: str = "", message_id: str = "",
                   step_id: str = "", parent_id: str = "", cwd: str = "",
                   tool_calls: list | None = None,
                   error_count: int | None = None,
                   has_reasoning: bool | None = None,
                   text_preview: str = "") -> dict:
    """Build one internal-format step dict — the single source of the step schema.

    Derived fields (tool_calls, error_count, has_reasoning) are computed from
    *parts* unless explicitly overridden.  ``usage=None`` yields zeroed token
    counts.  Callers compute their own ``text_preview`` (the preview heuristics
    intentionally differ between the main-trajectory and event paths).
    """
    if tool_calls is None:
        tool_calls = [p for p in parts if p.get("type") == "tool_call"]
    if error_count is None:
        error_count = sum(1 for tc in tool_calls if tc.get("status") == "error")
    if has_reasoning is None:
        has_reasoning = any(p.get("type") == "reasoning" for p in parts)
    if usage is None:
        tokens = {"total": 0, "input": 0, "output": 0, "reasoning": 0,
                  "cache_read": 0, "cache_write": 0}
    else:
        tokens = {
            "total": usage["total"],
            "input": usage["input"],
            "output": usage["output"],
            "reasoning": usage["reasoning"],
            "cache_read": usage["cache"]["read"],
            "cache_write": usage["cache"]["write"],
        }
    return {
        "role": role,
        "tokens": tokens,
        "duration": None,  # Will be computed from timestamps
        "parts": parts,
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "error_count": error_count,
        "has_reasoning": has_reasoning,
        "text_preview": text_preview,
        "finish": finish or "",
        "model_id": model,
        "provider_id": "",
        "time_created_ms": timestamp_ms,
        "time_completed_ms": None,
        "agent": agent,
        "mode": "",
        "message_id": message_id,
        "id": step_id,
        "parent_id": parent_id,
        "session_id": "",
        "cwd": cwd,
        "root": "",
    }


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

    # Preview: prefer text over reasoning (keep scanning after a reasoning hit)
    text_preview = ""
    for p in parts:
        if p.get("type") == "text" and p.get("text"):
            text_preview = p["text"]
            break
        if p.get("type") == "reasoning" and p.get("text") and not text_preview:
            text_preview = p["text"]
        if p.get("type") == "tool_call" and not text_preview:
            text_preview = f"[Tool: {p['tool_name']}]"

    return _cc_build_step(
        parts,
        role=role,
        usage=usage,
        timestamp_ms=timestamp_ms,
        model=entry.get("model", ""),
        finish=entry.get("stop_reason", ""),
        agent=agent_id or "",
        message_id=entry.get("message_id", entry.get("uuid", "")),
        step_id=entry.get("uuid", ""),
        parent_id=entry.get("parent_uuid", ""),
        cwd=entry.get("cwd", ""),
        text_preview=text_preview,
    )


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
            # Preview: first text-or-reasoning part wins (event-path heuristic)
            text_preview = ""
            for p in parts:
                if p.get("type") in ("text", "reasoning") and p.get("text"):
                    text_preview = p["text"]
                    break
                if p.get("type") == "tool_call" and not text_preview:
                    text_preview = f"[Tool: {p['tool_name']}]"

            steps.append(_cc_build_step(
                parts,
                role="assistant",
                usage=usage,
                timestamp_ms=timestamp_ms,
                model=inner_msg.get("model", ""),
                finish=inner_msg.get("stop_reason", ""),
                agent=agent_id,
                message_id=inner_msg.get("id", ""),
                step_id=msg.get("uuid", entry.get("uuid", "")),
                parent_id=entry.get("parent_uuid", ""),
                cwd=entry.get("cwd", ""),
                text_preview=text_preview,
            ))
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
                steps.append(_cc_build_step(
                    parts,
                    role="user",
                    timestamp_ms=timestamp_ms,
                    agent=agent_id,
                    step_id=msg.get("uuid", entry.get("uuid", "")),
                    parent_id=entry.get("parent_uuid", ""),
                    cwd=entry.get("cwd", ""),
                    # Event-path user steps never carry tool calls/usage
                    tool_calls=[],
                    error_count=0,
                    has_reasoning=False,
                    text_preview=next(
                        (p["text"] for p in parts if p.get("type") == "text" and p.get("text")),
                        ""
                    ),
                ))

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
            # Skip assistant->user turn boundaries: any user step surviving
            # conversion is a real human prompt (tool_result-only user
            # messages were dropped), so the gap is human idle time between
            # turns, not model latency.  The turn-final assistant step keeps
            # duration=None, same as the last step of each agent timeline.
            if timed[i].get("role") == "assistant" and timed[i + 1].get("role") == "user":
                continue
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
        started_at = datetime.fromtimestamp(created_ms / 1000.0, tz=UTC).isoformat()
    if isinstance(updated_ms, (int, float)):
        finished_at = datetime.fromtimestamp(updated_ms / 1000.0, tz=UTC).isoformat()
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
                    if ctype == "output_text" or ctype == "input_text":
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
                # An output can arrive after task_complete flushed the turn
                # (role/timestamp reset to None); restore them so the final
                # flush's role guard does not silently drop this part.
                current_role = current_role or "assistant"
                current_timestamp = current_timestamp or ts
                call_id = payload.get("call_id", "")
                output = payload.get("output", "")
                tc = pending_tool_calls.pop(call_id, {})

                # Determine status from output.  Structured metadata
                # (metadata.exit_code) is authoritative; the substring
                # heuristic is only a fallback when no exit code exists.
                status = "success"
                if isinstance(output, str):
                    exit_code = None
                    try:
                        output_data = json.loads(output)
                    except json.JSONDecodeError:
                        output_data = None
                    if isinstance(output_data, dict):
                        metadata = output_data.get("metadata")
                        if isinstance(metadata, dict):
                            candidate = metadata.get("exit_code")
                            if isinstance(candidate, int) and not isinstance(candidate, bool):
                                exit_code = candidate
                    if exit_code is not None:
                        status = "error" if exit_code != 0 else "success"
                    else:
                        # Anchored fallback so benign text ("Found 0 errors",
                        # "error-free") does not flag a successful call.
                        if re.search(r"(?i)\b(?:error:|traceback \(most recent call last\))",
                                     output[:200]):
                            status = "error"
                        # Check exit code reported in the output text
                        if "exited with code" in output and "code 0" not in output:
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

    # Drain tool calls that never received a function_call_output (session
    # interrupted/truncated mid-command) so the final — often most diagnostic —
    # invocation is not silently dropped from the timeline.
    if pending_tool_calls:
        current_role = current_role or "assistant"
        for call_id, tc in pending_tool_calls.items():
            current_parts.append({
                "type": "tool_call",
                "tool_name": tc.get("tool_name", "Bash"),
                "tool_id": call_id,
                "status": "error",  # interrupted: call never produced an output
                "input": tc.get("input", {}),
                "output": "",
            })
        pending_tool_calls.clear()

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


# Pi coding-agent tools use lowercase names and a `path` argument. Map them to
# the Claude Code / OpenCode vocabulary so file-interaction charts and write
# detection work without a second set of aliases.
_PI_TOOL_NAMES = {
    "bash": "Bash",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "grep": "Grep",
    "find": "Glob",
    "ls": "Glob",
}


def _pi_event_ts_ms(event: dict) -> int | None:
    """Prefer the event ISO timestamp; fall back to nested message epoch-ms."""
    ts = _iso_to_epoch_ms(event.get("timestamp") if isinstance(event.get("timestamp"), str) else None)
    if ts is not None:
        return ts
    msg = event.get("message")
    if isinstance(msg, dict):
        nested = msg.get("timestamp")
        if isinstance(nested, (int, float)) and not isinstance(nested, bool):
            value = int(nested)
            # Pi stores nested timestamps as epoch milliseconds.
            return value if value > 10**12 else value * 1000
        if isinstance(nested, str):
            return _iso_to_epoch_ms(nested)
    return None


def _pi_int(value: Any) -> int:
    """Coerce a Pi usage field to int; bools and non-numerics become 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _pi_extract_usage(usage: Any) -> dict | None:
    """Map Pi usage {input, output, cacheRead, cacheWrite, reasoning, totalTokens}.

    Pi's ``totalTokens`` is often input+output only (reasoning/cache omitted).
    Prefer the component sum so stacked token charts and the stored total agree.
    """
    if not isinstance(usage, dict):
        return None
    inp = _pi_int(usage.get("input", 0))
    out = _pi_int(usage.get("output", 0))
    reasoning = _pi_int(usage.get("reasoning", 0))
    cache_read = _pi_int(usage.get("cacheRead", 0))
    cache_write = _pi_int(usage.get("cacheWrite", 0))
    component_total = inp + out + reasoning + cache_read + cache_write
    vendor_total = _pi_int(usage.get("totalTokens"))
    total = max(component_total, vendor_total)
    if not any((total, inp, out, reasoning, cache_read, cache_write)):
        return None
    return {
        "total": total,
        "input": inp,
        "output": out,
        "reasoning": reasoning,
        "cache": {
            "read": cache_read,
            "write": cache_write,
        },
    }


def _pi_iter_content(content: Any):
    """Yield content dicts without treating a string as a character sequence."""
    if isinstance(content, str):
        if content:
            yield {"type": "text", "text": content}
        return
    if isinstance(content, dict):
        yield content
        return
    if not isinstance(content, list):
        return
    for item in content:
        if isinstance(item, str):
            if item:
                yield {"type": "text", "text": item}
        elif isinstance(item, dict):
            yield item


def _pi_content_text(content: Any) -> str:
    chunks: list[str] = []
    for item in _pi_iter_content(content):
        if item.get("type") == "text":
            chunks.append(str(item.get("text") or ""))
    return "\n".join(chunks)


def _pi_normalize_tool(name: str, arguments: Any) -> tuple[str, dict]:
    raw_name = name or "?"
    canonical = _PI_TOOL_NAMES.get(str(raw_name).lower(), raw_name)
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    args = dict(arguments) if isinstance(arguments, dict) else (
        {} if arguments is None else {"raw": arguments}
    )
    if canonical in ("Read", "Write", "Edit") and "file_path" not in args and "path" in args:
        args["file_path"] = args["path"]
    if canonical == "Bash" and "command" not in args and "cmd" in args:
        args["command"] = args["cmd"]
    return canonical, args


def _convert_pi_to_internal(events: list[dict]) -> dict:
    """Convert Pi coding-agent JSONL events into the trajviz internal format.

    Pi emits newline-delimited JSON with event types:
    - session: session ID, cwd, version
    - model_change / thinking_level_change: metadata (not steps)
    - message: nested {role: user|assistant|toolResult, content, usage, ...}

    Assistant ``toolCall`` parts are paired with later ``toolResult`` messages
    by ``toolCallId``.  The result matches the OpenCode internal message shape
    used by parse_steps().
    """
    session: dict = {}
    messages: list[dict] = []
    pending_tools: dict[str, dict] = {}
    current_model = ""
    current_provider = ""

    def _append(role: str, ts: int | None, parts: list, tokens: dict | None = None,
                extra: dict | None = None) -> None:
        info: dict[str, Any] = {
            "role": role,
            "time": {"created": ts or 0},
        }
        if tokens:
            info["tokens"] = tokens
        if extra:
            info.update(extra)
        messages.append({"info": info, "parts": parts})

    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        ts = _pi_event_ts_ms(event)

        if etype == "session":
            session = event
            continue
        if etype == "model_change":
            if event.get("provider"):
                current_provider = str(event.get("provider"))
            if event.get("modelId"):
                current_model = str(event.get("modelId"))
            continue
        if etype in ("thinking_level_change",):
            continue
        if etype != "message":
            continue

        msg = event.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")

        if role == "user":
            parts = []
            for item in _pi_iter_content(msg.get("content")):
                if item.get("type") == "text":
                    parts.append({"type": "text", "text": item.get("text", "")})
            if parts:
                _append("user", ts, parts, extra={"id": event.get("id") or ""})

        elif role == "assistant":
            if msg.get("model"):
                current_model = str(msg.get("model"))
            if msg.get("provider"):
                current_provider = str(msg.get("provider"))
            parts = []
            for item in _pi_iter_content(msg.get("content")):
                ctype = item.get("type")
                if ctype == "thinking":
                    text = item.get("thinking") or item.get("text") or ""
                    parts.append({"type": "reasoning", "text": text})
                elif ctype == "text":
                    parts.append({"type": "text", "text": item.get("text", "")})
                elif ctype == "toolCall":
                    tool_name, tool_input = _pi_normalize_tool(
                        item.get("name", "?"), item.get("arguments"),
                    )
                    call_id = item.get("id") or ""
                    part = {
                        "type": "tool_call",
                        "tool_name": tool_name,
                        "tool_id": call_id,
                        "status": "pending",
                        "input": tool_input,
                        "output": "",
                    }
                    parts.append(part)
                    if call_id:
                        pending_tools[call_id] = part
            error_message = msg.get("errorMessage")
            if error_message:
                parts.append({"type": "text", "text": f"[error] {error_message}"})
            tokens = _pi_extract_usage(msg.get("usage"))
            extra = {
                "id": event.get("id") or "",
                "modelID": current_model,
                "providerID": current_provider,
            }
            stop = msg.get("stopReason")
            if stop:
                extra["finish"] = stop
            if parts or tokens or stop == "error":
                if not parts and stop == "error":
                    parts = [{"type": "text", "text": "[error]"}]
                _append("assistant", ts, parts, tokens=tokens, extra=extra)

        elif role == "toolResult":
            call_id = msg.get("toolCallId") or ""
            output = _pi_content_text(msg.get("content"))
            is_error = bool(msg.get("isError"))
            status = "error" if is_error else "success"
            part = pending_tools.pop(call_id, None) if call_id else None
            if part is not None:
                part["output"] = output
                part["status"] = status
                if is_error:
                    part["error"] = output
            else:
                tool_name, _ = _pi_normalize_tool(msg.get("toolName") or "?", {})
                _append("assistant", ts, [{
                    "type": "tool",
                    "tool_name": tool_name,
                    "tool_id": call_id,
                    "status": status,
                    "input": {},
                    "output": output,
                    **({"error": output} if is_error else {}),
                }])

    for part in pending_tools.values():
        if part.get("status") == "pending":
            part["status"] = "error"

    for i in range(len(messages) - 1):
        cur_t = messages[i]["info"]["time"].get("created")
        nxt_t = messages[i + 1]["info"]["time"].get("created")
        if isinstance(cur_t, (int, float)) and isinstance(nxt_t, (int, float)) and nxt_t >= cur_t:
            messages[i]["info"]["time"]["completed"] = nxt_t

    first_ts_iso = None
    last_ts_iso = None
    for event in events:
        if isinstance(event, dict) and isinstance(event.get("timestamp"), str):
            first_ts_iso = event["timestamp"]
            break
    for event in reversed(events):
        if isinstance(event, dict) and isinstance(event.get("timestamp"), str):
            last_ts_iso = event["timestamp"]
            break
    directory = session.get("cwd", "") or ""
    start_ms = _iso_to_epoch_ms(first_ts_iso)
    end_ms = _iso_to_epoch_ms(last_ts_iso)
    total_duration = (
        round((end_ms - start_ms) / 1000.0, 3)
        if isinstance(start_ms, int) and isinstance(end_ms, int) else 0
    )
    total_tokens = sum(
        (m["info"].get("tokens") or {}).get("total", 0) for m in messages
    )
    last_model = current_model
    last_provider = current_provider

    info = {
        "id": session.get("id", ""),
        "slug": "",
        "projectID": "",
        "directory": directory,
        "title": "",
        "version": str(session.get("version", "") or ""),
        "time": {"created": start_ms or 0, "updated": end_ms or 0},
    }

    return {
        "info": info,
        "messages": messages,
        "metadata": {
            "session_id": session.get("id", ""),
            "directory": directory,
            "directory_name": directory.replace("\\", "/").rsplit("/", 1)[-1] if directory else "",
            "agent": "pi",
            "model": last_model,
            "source": "pi",
            "model_provider": last_provider,
            "originator": "Pi",
            "server_version": str(session.get("version", "") or ""),
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
        "_pi_format": True,
    }


# DeepSeek Harness tools are lowercase with JSON-string arguments. Map them
# onto the Claude Code / OpenCode vocabulary so file-interaction charts,
# TodoWrite plan tracking, and spawn annotation work without extra aliases.
_DSH_TOOL_NAMES = {
    "bash": "Bash",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "glob": "Glob",
    "grep": "Grep",
    "todo_write": "TodoWrite",
    "subagent": "Agent",
    "subagent_fork": "Agent",
}


def _dsh_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _dsh_event_ts_ms(event: dict) -> int | None:
    """Epoch-ms from a DSH event envelope (``time`` / packed ``time0`` / header)."""
    for key in ("time", "time0", "createdAt"):
        value = event.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return int(value)
    return None


def _dsh_event_seq(event: dict) -> int | None:
    for key in ("seq", "seq0"):
        value = event.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _dsh_drop_seed_prefix(events: list[dict], seed_length: Any) -> list[dict]:
    """Drop a forked child's inherited parent prefix.

    Persisted DSH child logs copy the parent's events and keep their original
    ``seq`` values; ``seedLength`` is the first seq that belongs to the child.
    Compact fixtures without seq numbers treat ``seedLength`` as a slice index.
    """
    if not isinstance(seed_length, int) or isinstance(seed_length, bool) or seed_length <= 0:
        return events
    seqs = [_dsh_event_seq(event) for event in events]
    if any(seq is not None for seq in seqs):
        return [
            event for event, seq in zip(events, seqs, strict=False)
            if seq is None or seq >= seed_length
        ]
    return events[seed_length:]


def _dsh_iter_content(content: Any):
    if isinstance(content, str):
        if content:
            yield {"type": "text", "text": content}
        return
    if isinstance(content, dict):
        yield content
        return
    if not isinstance(content, list):
        return
    for item in content:
        if isinstance(item, str):
            if item:
                yield {"type": "text", "text": item}
        elif isinstance(item, dict):
            yield item


def _dsh_blocks_text(content: Any) -> str:
    chunks: list[str] = []
    for item in _dsh_iter_content(content):
        ctype = item.get("type")
        if ctype in ("text", "reasoning"):
            text = item.get("text") or ""
            if text:
                chunks.append(str(text))
        elif ctype in ("tool-result", "tool_result"):
            nested = _dsh_blocks_text(item.get("content"))
            if nested:
                chunks.append(nested)
    return "\n".join(chunks)


def _dsh_parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw} if raw else {}
        if isinstance(parsed, dict):
            return parsed
        return {"raw": parsed}
    if raw is None:
        return {}
    return {"raw": raw}


def _dsh_normalize_tool(name: str, arguments: Any) -> tuple[str, dict]:
    raw_name = name or "?"
    canonical = _DSH_TOOL_NAMES.get(str(raw_name).lower(), raw_name)
    args = _dsh_parse_args(arguments)
    if canonical in ("Read", "Write", "Edit") and "file_path" not in args:
        if "path" in args:
            args["file_path"] = args["path"]
        elif "filePath" in args:
            args["file_path"] = args["filePath"]
    if canonical == "Bash" and "command" not in args and "cmd" in args:
        args["command"] = args["cmd"]
    return canonical, args


def _dsh_extract_usage(usage: Any) -> dict | None:
    """Map DSH usage {inputTokens, outputTokens, cacheReadTokens, reasoningTokens}."""
    if not isinstance(usage, dict):
        return None
    inp = _dsh_int(usage.get("inputTokens", usage.get("input", 0)))
    out = _dsh_int(usage.get("outputTokens", usage.get("output", 0)))
    reasoning = _dsh_int(usage.get("reasoningTokens", usage.get("reasoning", 0)))
    cache_read = _dsh_int(usage.get("cacheReadTokens", usage.get("cacheRead", 0)))
    cache_write = _dsh_int(usage.get("cacheWriteTokens", usage.get("cacheWrite", 0)))
    component_total = inp + out + reasoning + cache_read + cache_write
    vendor_total = _dsh_int(usage.get("totalTokens", usage.get("total", 0)))
    total = max(component_total, vendor_total)
    if not any((total, inp, out, reasoning, cache_read, cache_write)):
        return None
    return {
        "total": total,
        "input": inp,
        "output": out,
        "reasoning": reasoning,
        "cache": {"read": cache_read, "write": cache_write},
    }


def _dsh_child_session_id_from_output(output: str) -> str:
    if not isinstance(output, str) or not output:
        return ""
    match = re.search(r"started subagent\s+(\S+)", output, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).rstrip(".,;")


def _dsh_is_human_user_source(source: Any) -> bool:
    """True when a ``user/message`` is a human prompt, not a runtime notice.

    DSH tags sandbox snapshots as ``plugin`` and background-child notices as
    ``subagent-report`` / ``subagent-settled``. Compact fixtures omit ``kind``.
    """
    if not isinstance(source, dict):
        return True
    kind = source.get("kind")
    if not isinstance(kind, str) or not kind:
        return True
    return kind == "user"


def _dsh_tool_result_payload(data: dict) -> tuple[str, str, bool]:
    """Return ``(call_id, output_text, is_error)`` from a ``tool/result`` event."""
    message = data.get("message") if isinstance(data.get("message"), dict) else {}
    call_id = ""
    source = message.get("source")
    if isinstance(source, dict) and source.get("callId"):
        call_id = str(source.get("callId"))
    is_error = bool(data.get("error"))
    chunks: list[str] = []
    for item in _dsh_iter_content(message.get("content")):
        if item.get("type") in ("tool-result", "tool_result"):
            if not call_id and item.get("toolCallId"):
                call_id = str(item.get("toolCallId"))
            if item.get("isError"):
                is_error = True
            text = _dsh_blocks_text(item.get("content"))
            if text:
                chunks.append(text)
        elif item.get("type") in ("text", "reasoning") and item.get("text"):
            chunks.append(str(item.get("text")))
    output = "\n".join(chunks)
    if not output and isinstance(data.get("error"), dict):
        err = data["error"]
        name = err.get("name") or err.get("code") or "error"
        output = str(name)
        is_error = True
    return call_id, output, is_error


def _dsh_session_to_messages(events: list[dict]) -> tuple[dict, list[dict], str, str, str]:
    """Convert one DSH session's events into internal messages.

    Returns ``(session_header, messages, last_model, last_provider, title)``.
    """
    session: dict = {}
    body: list[dict] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("type") == "session" and not session:
            session = event
            continue
        body.append(event)
    body = _dsh_drop_seed_prefix(body, session.get("seedLength"))

    session_id = session.get("id", "") if isinstance(session.get("id"), str) else ""
    parent_session_id = session.get("parentSession", "") if isinstance(
        session.get("parentSession"), str) else ""
    is_sub_agent = session.get("origin") == "subagent" or bool(parent_session_id)
    agent = session.get("agentPreset") or ""
    cwd = session.get("cwd") or ""

    messages: list[dict] = []
    pending_tools: dict[str, dict] = {}
    current_model = ""
    current_provider = ""
    title = ""

    def _append(role: str, ts: int | None, parts: list, tokens: dict | None = None,
                extra: dict | None = None) -> None:
        info: dict[str, Any] = {
            "role": role,
            "time": {"created": ts or 0},
            "sessionID": session_id,
            "isSubAgent": is_sub_agent,
            "agent": agent,
        }
        if parent_session_id:
            info["parentSessionID"] = parent_session_id
        if cwd:
            info["path"] = {"cwd": cwd}
        if tokens:
            info["tokens"] = tokens
        if extra:
            info.update(extra)
        messages.append({"info": info, "parts": parts})

    for event in body:
        etype = event.get("type")
        ts = _dsh_event_ts_ms(event)
        data = event.get("data") if isinstance(event.get("data"), dict) else {}

        if etype == "session/title":
            candidate = data.get("title")
            if isinstance(candidate, str) and candidate:
                title = candidate
            continue
        if etype == "request/context":
            if data.get("model"):
                current_model = str(data.get("model"))
            if data.get("provider"):
                current_provider = str(data.get("provider"))
            continue
        if etype == "request/header":
            header = data.get("header") if isinstance(data.get("header"), dict) else {}
            config = header.get("config") if isinstance(header.get("config"), dict) else {}
            if config.get("model"):
                current_model = str(config.get("model"))
            if config.get("provider"):
                current_provider = str(config.get("provider"))
            continue
        if etype == "tool/call":
            call_id = str(data.get("callId") or "")
            part = pending_tools.get(call_id) if call_id else None
            if part is not None and ts:
                state = part.setdefault("state", {})
                if isinstance(state, dict):
                    time_info = state.setdefault("time", {})
                    if isinstance(time_info, dict) and "start" not in time_info:
                        time_info["start"] = ts
            continue
        if etype == "tool/result":
            call_id, output, is_error = _dsh_tool_result_payload(data)
            status = "error" if is_error else "success"
            part = pending_tools.pop(call_id, None) if call_id else None
            if part is not None:
                part["output"] = output
                part["status"] = status
                if is_error:
                    part["error"] = output
                state = part.setdefault("state", {})
                if isinstance(state, dict):
                    state["status"] = status
                    state["output"] = output
                    if ts:
                        time_info = state.setdefault("time", {})
                        if isinstance(time_info, dict):
                            time_info["end"] = ts
                child_id = _dsh_child_session_id_from_output(output)
                if child_id:
                    for meta in (
                        part.setdefault("metadata", {}),
                        part.get("state", {}).setdefault("metadata", {})
                        if isinstance(part.get("state"), dict) else {},
                    ):
                        if isinstance(meta, dict):
                            meta["sessionId"] = child_id
                            if session_id:
                                meta["parentSessionId"] = session_id
            else:
                _append("assistant", ts, [{
                    "type": "tool",
                    "tool_name": "?",
                    "tool_id": call_id,
                    "status": status,
                    "input": {},
                    "output": output,
                    **({"error": output} if is_error else {}),
                }])
            continue
        if etype == "user/message":
            source = data.get("source") if isinstance(data.get("source"), dict) else {}
            if not _dsh_is_human_user_source(source):
                continue
            parts = []
            for item in _dsh_iter_content(data.get("content")):
                ctype = item.get("type")
                if ctype in ("text", "reasoning") and item.get("text"):
                    parts.append({"type": "text", "text": item.get("text", "")})
            if parts:
                extra = {"id": data.get("id") or ""}
                _append("user", ts, parts, extra=extra)
            continue
        if etype != "assistant/message":
            continue

        message = data.get("message") if isinstance(data.get("message"), dict) else {}
        msg_source = message.get("source") if isinstance(message.get("source"), dict) else {}
        if msg_source.get("model"):
            current_model = str(msg_source.get("model"))
        if msg_source.get("provider"):
            current_provider = str(msg_source.get("provider"))
        parts = []
        for item in _dsh_iter_content(message.get("content")):
            ctype = item.get("type")
            if ctype == "reasoning":
                parts.append({"type": "reasoning", "text": item.get("text") or ""})
            elif ctype == "text":
                parts.append({"type": "text", "text": item.get("text") or ""})
            elif ctype in ("tool-call", "tool_call", "toolCall"):
                tool_name, tool_input = _dsh_normalize_tool(
                    item.get("name", "?"), item.get("arguments"),
                )
                call_id = str(item.get("id") or "")
                part = {
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "tool_id": call_id,
                    "status": "pending",
                    "input": tool_input,
                    "output": "",
                    "state": {
                        "status": "pending",
                        "input": tool_input,
                        "output": "",
                        "metadata": {},
                    },
                    "metadata": {},
                }
                parts.append(part)
                if call_id:
                    pending_tools[call_id] = part
        tokens = _dsh_extract_usage(data.get("usage"))
        extra = {
            "id": message.get("id") or "",
            "modelID": current_model,
            "providerID": current_provider,
        }
        if parts or tokens:
            _append("assistant", ts, parts, tokens=tokens, extra=extra)

    for part in pending_tools.values():
        if part.get("status") == "pending":
            part["status"] = "error"
            state = part.get("state")
            if isinstance(state, dict):
                state["status"] = "error"

    return session, messages, current_model, current_provider, title


_DSH_CHILD_WALK_MAX_DEPTH = 8


def _dsh_child_paths_in_subagents_dir(
    sub_root: str, *, _depth: int = 0,
) -> list[str]:
    """Collect ``<id>/session.jsonl`` under a ``subagents/`` directory.

    Walks nested ``<id>/subagents/`` trees (grandchildren) up to
    ``_DSH_CHILD_WALK_MAX_DEPTH`` so a parent export that stores deeper
    delegations still merges.
    """
    if _depth > _DSH_CHILD_WALK_MAX_DEPTH or not os.path.isdir(sub_root):
        return []
    paths: list[str] = []
    try:
        names = sorted(os.listdir(sub_root))
    except OSError:
        return []
    seen: set[str] = set()
    for name in names:
        if any(sep in name for sep in ("/", "\\")) or name in (".", ".."):
            continue
        child_dir = os.path.join(sub_root, name)
        session = os.path.join(child_dir, "session.jsonl")
        if os.path.isfile(session):
            real = os.path.realpath(session)
            if real not in seen:
                seen.add(real)
                paths.append(session)
        nested = os.path.join(child_dir, "subagents")
        for nested_path in _dsh_child_paths_in_subagents_dir(
            nested, _depth=_depth + 1,
        ):
            real = os.path.realpath(nested_path)
            if real not in seen:
                seen.add(real)
                paths.append(nested_path)
    return paths


def _dsh_sibling_child_paths(source_path: str | None) -> list[str]:
    if not source_path:
        return []
    parent_dir = source_path if os.path.isdir(source_path) else os.path.dirname(
        os.path.abspath(source_path)
    )
    return _dsh_child_paths_in_subagents_dir(os.path.join(parent_dir, "subagents"))


def _dsh_safe_session_id(session_id: str) -> str:
    """Reject session ids that could escape a search root as a folder name."""
    if not isinstance(session_id, str) or not session_id:
        return ""
    if any(sep in session_id for sep in ("/", "\\")) or ".." in session_id:
        return ""
    if len(session_id) > 200:
        return ""
    return session_id


def _dsh_export_dir_names(session_id: str) -> list[str]:
    """Folder names used by DSH GUI / zip exports for one session id."""
    session_id = _dsh_safe_session_id(session_id)
    if not session_id:
        return []
    names = [f"dsh-session-{session_id}"]
    if session_id.startswith("session-") and len(session_id) > 8:
        names.append(f"dsh-session-{session_id[len('session-'):]}")
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _dsh_export_root_child_paths(session_id: str) -> list[str]:
    """Child logs under ``TRAJVIZ_DSH_EXPORT_ROOT`` only (same-machine / tests).

    A hosted dashboard never sees the uploader's home directory; do not crawl
    cwd or ``~/Downloads``.
    """
    extra = os.environ.get("TRAJVIZ_DSH_EXPORT_ROOT")
    if not extra:
        return []
    root = os.path.abspath(os.path.expanduser(extra))
    if not os.path.isdir(root):
        return []
    found = _dsh_child_paths_in_subagents_dir(os.path.join(root, "subagents"))
    if found:
        return found
    for name in _dsh_export_dir_names(session_id):
        found = _dsh_child_paths_in_subagents_dir(os.path.join(root, name, "subagents"))
        if found:
            return found
    return []


def _dsh_discover_child_log_paths(session_id: str, source_path: str | None) -> list[str]:
    """Find ``subagents/<id>/session.jsonl`` next to the loaded file.

    Hosted uploads are a single temp file (or a zip). Children come from a
    sibling ``subagents/`` tree, from zip members, or from an explicit
    ``TRAJVIZ_DSH_EXPORT_ROOT`` on this host.
    """
    paths = _dsh_sibling_child_paths(source_path)
    if paths:
        return paths
    return _dsh_export_root_child_paths(session_id)


def _dsh_read_event_list(path: str) -> list[dict]:
    try:
        with open(path, encoding="utf-8-sig") as handle:
            text = handle.read()
    except (OSError, UnicodeDecodeError):
        return []
    payload, err = _parse_trajectory_text(text, path)
    if err:
        return []
    payload = _normalize_payload(payload, path)
    if isinstance(payload, list):
        return [event for event in payload if isinstance(event, dict)]
    return []


def _fill_message_completion_times(messages: list[dict]) -> None:
    """Set ``time.completed`` from the next message in the *same* session.

    Parent and child logs are merged then sorted by timestamp. Filling from
    the globally next row attributes a parallel child's start to the parent
    (and sibling children to each other).
    """
    by_session: dict[str, list[dict]] = {}
    for msg in messages:
        info = msg.get("info") if isinstance(msg.get("info"), dict) else {}
        sid = info.get("sessionID")
        key = sid if isinstance(sid, str) else ""
        by_session.setdefault(key, []).append(msg)
    for group in by_session.values():
        for i in range(len(group) - 1):
            cur_info = group[i].get("info")
            nxt_info = group[i + 1].get("info")
            if not isinstance(cur_info, dict) or not isinstance(nxt_info, dict):
                continue
            cur_time = cur_info.get("time") if isinstance(cur_info.get("time"), dict) else None
            nxt_time = nxt_info.get("time") if isinstance(nxt_info.get("time"), dict) else None
            if not cur_time or not nxt_time:
                continue
            cur_t = cur_time.get("created")
            nxt_t = nxt_time.get("created")
            if (
                isinstance(cur_t, (int, float)) and not isinstance(cur_t, bool)
                and isinstance(nxt_t, (int, float)) and not isinstance(nxt_t, bool)
                and nxt_t >= cur_t
            ):
                cur_time["completed"] = nxt_t


def _dsh_ms_to_iso(ms: int | None) -> str:
    if not isinstance(ms, int) or ms <= 0:
        return ""
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=UTC).isoformat()
    except (OSError, OverflowError, ValueError):
        return ""


def _convert_dsh_to_internal(
    events: list[dict],
    *,
    source_path: str | None = None,
    child_event_lists: list[list[dict]] | None = None,
) -> dict:
    """Convert DeepSeek Harness JSONL events into the trajviz internal format.

    DSH emits newline-delimited JSON with a ``session`` header and slash-typed
    body events (``user/message``, ``assistant/message``, ``tool/call``,
    ``tool/result``). Streaming chunk rows are ignored in favour of the
    completed ``assistant/message`` (which carries usage). Child sessions live
    under a sibling ``subagents/<id>/session.jsonl`` directory (or the same
    layout inside an export zip, including nested ``subagents/<id>/subagents/``)
    and are merged when present.
    """
    session, messages, model, provider, title = _dsh_session_to_messages(events)
    if child_event_lists is None:
        origin = session.get("origin")
        parent_session = session.get("parentSession")
        is_child_log = origin == "subagent" or (
            isinstance(parent_session, str) and bool(parent_session)
        )
        if is_child_log:
            child_event_lists = []
        else:
            session_id = session.get("id", "") if isinstance(session.get("id"), str) else ""
            child_event_lists = [
                _dsh_read_event_list(path)
                for path in _dsh_discover_child_log_paths(session_id, source_path)
            ]
    sub_agent_ids: set[str] = set()
    for child_events in child_event_lists:
        if not child_events:
            continue
        child_session, child_messages, child_model, child_provider, child_title = (
            _dsh_session_to_messages(child_events)
        )
        child_id = child_session.get("id")
        if isinstance(child_id, str) and child_id:
            sub_agent_ids.add(child_id)
        if child_model and not model:
            model = child_model
        if child_provider and not provider:
            provider = child_provider
        if child_title:
            for msg in child_messages:
                info = msg.get("info") if isinstance(msg.get("info"), dict) else None
                if info is not None and not info.get("sessionTitle"):
                    info["sessionTitle"] = child_title
        messages.extend(child_messages)

    messages.sort(key=lambda msg: (
        msg.get("info", {}).get("time", {}).get("created") or 0,
        1 if msg.get("info", {}).get("isSubAgent") else 0,
    ))
    _fill_message_completion_times(messages)

    directory = session.get("cwd", "") or ""
    start_ms = _dsh_int(session.get("createdAt"))
    if not start_ms:
        for msg in messages:
            created = msg.get("info", {}).get("time", {}).get("created")
            if isinstance(created, (int, float)) and created:
                start_ms = int(created)
                break
    end_ms = 0
    for event in reversed(events):
        if isinstance(event, dict):
            ts = _dsh_event_ts_ms(event)
            if ts:
                end_ms = ts
                break
    if messages:
        last_created = messages[-1]["info"]["time"].get("created") or 0
        last_completed = messages[-1]["info"]["time"].get("completed") or 0
        end_ms = max(end_ms, int(last_created), int(last_completed or 0))
    total_duration = (
        round((end_ms - start_ms) / 1000.0, 3)
        if start_ms and end_ms and end_ms >= start_ms else 0
    )
    total_tokens = sum(
        (m["info"].get("tokens") or {}).get("total", 0) for m in messages
    )
    session_id = session.get("id", "") if isinstance(session.get("id"), str) else ""
    started_at = _dsh_ms_to_iso(start_ms or None)
    finished_at = _dsh_ms_to_iso(end_ms or None)
    version = session.get("version")
    version_str = "" if version is None else str(version)

    info = {
        "id": session_id,
        "slug": "",
        "projectID": "",
        "directory": directory,
        "title": title,
        "version": version_str,
        "time": {"created": start_ms or 0, "updated": end_ms or 0},
    }

    return {
        "info": info,
        "messages": messages,
        "metadata": {
            "session_id": session_id,
            "directory": directory,
            "directory_name": directory.replace("\\", "/").rsplit("/", 1)[-1] if directory else "",
            "agent": "dsh",
            "model": model,
            "source": "dsh",
            "model_provider": provider,
            "originator": "DeepSeek Harness",
            "server_version": version_str,
            "timestamp_utc": started_at,
            "title": title,
            "sub_agent_count": len(sub_agent_ids),
            "agent_preset": session.get("agentPreset") or "",
        },
        "timing": {
            "total_duration": total_duration,
            "started_at": started_at,
            "finished_at": finished_at,
        },
        "output": {},
        "input": {},
        "token_usage": {"total_tokens": total_tokens},
        "stats": {},
        "_dsh_format": True,
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

    # Test, git, python, and all other shell commands are Bash
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


_UNSUPPORTED_EVENT_STREAM = (
    "Unsupported event-array input; expected Codex JSONL "
    "(leading session_meta event), Pi JSONL "
    "(leading session event), or DeepSeek Harness JSONL "
    "(leading session event with createdAt / slash-typed body events)."
)


def _path_ext(file_path: str) -> str:
    return os.path.splitext(file_path)[1].lower()


def _parse_jsonl_events(text: str) -> tuple[list | None, str | None]:
    """Parse newline-delimited JSON, tolerating a truncated final line.

    Interior decode errors are fatal. A missing newline on the last
    non-empty line is treated as an in-progress append and dropped.
    """
    events: list = []
    pending: tuple[int, str] | None = None
    for line_number, raw_line in enumerate(text.splitlines(keepends=True), start=1):
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
            return None, f"Invalid JSONL at line {pending_number}: {exc.msg}"
        pending = (line_number, raw_line)

    if pending is not None:
        pending_number, pending_line = pending
        try:
            events.append(json.loads(pending_line))
        except json.JSONDecodeError as exc:
            # Rollouts are append-only. A missing newline on the final
            # object is the signal that the writer may still be appending.
            if pending_line.endswith(("\n", "\r")):
                return None, f"Invalid JSONL at line {pending_number}: {exc.msg}"
    return events, None


def _parse_trajectory_text(text: str, file_path: str) -> tuple[Any, str | None]:
    """Sniff content: JSON object, JSON array, or JSONL event stream.

    A complete JSON document (object or array) wins even when the path
    ends in ``.jsonl``, so a Claude/OpenCode dump saved with the wrong
    extension still loads. Multiple JSON values (``Extra data``) or a
    ``.jsonl`` path that is not one JSON document fall through to the
    JSONL parser (including last-line truncation tolerance).
    """
    ext = _path_ext(file_path)
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        if exc.msg == "Extra data" or ext == ".jsonl":
            events, err = _parse_jsonl_events(text)
            if err:
                return None, err
            return events, None
        return None, str(exc)

    if isinstance(raw, (dict, list)):
        return raw, None
    return None, (
        f"Unsupported JSON input: top-level value is {type(raw).__name__}; "
        "expected a JSON object or event array."
    )


def _is_event_record(raw: dict) -> bool:
    """True when a JSON object is a Codex/Pi/DSH event, not a trajectory dump."""
    t = raw.get("type")
    if t == "session_meta":
        return True
    return t == "session" and isinstance(raw.get("id"), str) and bool(raw["id"])


def _unwrap_singleton_object_payload(payload: Any) -> Any:
    """If JSONL parsed as a single object-format dict, treat it as that object.

    Happens when a Claude/OpenCode/CodeArts dump is saved as ``.jsonl`` and
    a trailing truncated line forced the JSONL parser (``json.loads`` of
    the whole file then fails).
    """
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        if _detect_object_format(payload[0]) in _OBJECT_FORMATS:
            return payload[0]
    return payload


def _looks_like_object_dump(raw: dict) -> bool:
    """True when an unknown dict still looks like a JSON trajectory object.

    Unmarked Claude dumps (no ``format`` field) must stay objects so
    ``format_hint`` can force ``ccsession``, even if the file was saved as
    ``.jsonl``.
    """
    if isinstance(raw.get("trajectory"), list):
        return True
    if isinstance(raw.get("messages"), list):
        return True
    if isinstance(raw.get("info"), dict):
        return True
    session = raw.get("session")
    return isinstance(session, dict) and raw.get("type") is None


def _normalize_payload(payload: Any, file_path: str) -> Any:
    """Canonicalize parsed content to either a trajectory object or an event list.

    A one-line ``.jsonl`` file is valid JSON, so ``json.loads`` returns a
    dict. If that dict is not an object-format dump, wrap it as a
    single-event stream (Codex/Pi, or unknown JSONL). A ``session_meta`` /
    Pi ``session`` object is wrapped even without the ``.jsonl`` suffix.
    """
    payload = _unwrap_singleton_object_payload(payload)
    if isinstance(payload, dict):
        if _detect_object_format(payload) != "unknown":
            return payload
        if _looks_like_object_dump(payload):
            return payload
        if _path_ext(file_path) == ".jsonl" or _is_event_record(payload):
            return [payload]
    return payload


def _already_converted(raw: dict, fmt: str) -> bool:
    stamp = _FORMAT_STAMPS.get(fmt)
    return bool(stamp and raw.get(stamp) is True)


def _format_mismatch_error(selected: str, detected: str) -> dict:
    return {
        "_error": (
            f"Format mismatch: selected {FORMAT_LABELS.get(selected, selected)} "
            f"but file detected as {FORMAT_LABELS.get(detected, detected)}."
        ),
        "_error_code": "mismatch",
        "_selected": selected,
        "_detected": detected,
    }


def _kind_mismatch_error(fmt: str) -> dict:
    label = FORMAT_LABELS.get(fmt, fmt)
    if fmt in _EVENT_FORMATS:
        leading = "session_meta" if fmt == "codex" else "session"
        return {"_error": f"{label} expects a JSONL event array "
                          f"(leading {leading} event)."}
    return {"_error": f"{label} expects a JSON object, not an event array."}


def _apply_format(payload: Any, fmt: str, *, source_path: str | None = None) -> dict:
    """Convert parsed content to the internal step-model dict."""
    if fmt == "unknown":
        if isinstance(payload, dict):
            return payload
        return {"_error": _UNSUPPORTED_EVENT_STREAM}

    if isinstance(payload, dict) and _already_converted(payload, fmt):
        return payload

    if fmt in _EVENT_FORMATS:
        if not isinstance(payload, list):
            return _kind_mismatch_error(fmt)
        if fmt == "codex":
            return _convert_codex_to_internal(payload)
        if fmt == "dsh":
            return _convert_dsh_to_internal(payload, source_path=source_path)
        return _convert_pi_to_internal(payload)

    if not isinstance(payload, dict):
        return _kind_mismatch_error(fmt)
    if fmt == "ccsession":
        return _convert_claude_code_to_internal(payload)
    if fmt == "codearts":
        # Legacy-JSON exports are detected as CodeArts (so the UI labels them
        # correctly) but have no parser yet: their 'sender'/'content' messages
        # are not OpenCode-shaped and would silently produce empty steps.
        if safe_get(payload, "export_metadata", "source_format") == "codearts_legacy_json":
            return {"_error": "CodeArts legacy-JSON export detected "
                              "(source_format=codearts_legacy_json): this legacy "
                              "message schema is not yet supported. Re-export the "
                              "session from the CodeArts SQLite database instead."}
        return _convert_codearts_metadata(payload)
    if fmt == "opencode":
        return _convert_opencode_metadata(payload)
    return payload if isinstance(payload, dict) else {"_error": _UNSUPPORTED_EVENT_STREAM}


def _resolve_trajectory_path(file_path: str) -> tuple[str, str | None]:
    """If *file_path* is a DSH session directory, return its ``session.jsonl``.

    Returns ``(path, error)``. Directories without ``session.jsonl`` error.
    """
    if os.path.isdir(file_path):
        nested = os.path.join(file_path, "session.jsonl")
        if os.path.isfile(nested):
            return nested, None
        return file_path, f"Directory has no session.jsonl: {file_path}"
    return file_path, None


def _zip_subagent_suffix(name: str) -> str | None:
    """Path after a ``subagents`` directory component, or None if not a child log."""
    name = name.replace("\\", "/")
    while name.startswith("./"):
        name = name[2:]
    parts = name.split("/")
    try:
        idx = parts.index("subagents")
    except ValueError:
        return None
    rest = parts[idx + 1:]
    return "/".join(rest) if rest else None


def _zip_dsh_members(names: list[str]) -> tuple[str | None, list[tuple[str, str]]]:
    """Pick the parent ``session.jsonl`` and ``subagents/<id>/session.jsonl`` members.

    Parent is the shortest path ending in ``session.jsonl`` that is not under
    a ``subagents`` directory. Children are ``subagents/<id>/session.jsonl``
    (DSH GUI zip root), ``.../subagents/<id>/session.jsonl`` (folder zip),
    or nested ``.../subagents/<id>/subagents/<id>/session.jsonl``.
    """
    jsonl_members = [
        name.replace("\\", "/") for name in names
        if name.replace("\\", "/").rstrip("/").endswith("session.jsonl")
        and not name.endswith("/")
    ]
    parents = [name for name in jsonl_members if _zip_subagent_suffix(name) is None]
    parents.sort(key=lambda name: (name.count("/"), len(name)))
    parent = parents[0] if parents else None
    children: list[tuple[str, str]] = []
    for name in jsonl_members:
        rest = _zip_subagent_suffix(name)
        if not rest:
            continue
        rest_parts = rest.split("/")
        if len(rest_parts) < 2 or rest_parts[-1] != "session.jsonl":
            continue
        child_id = rest_parts[-2]
        if child_id and child_id != "subagents":
            children.append((child_id, name))
    children.sort(key=lambda item: item[0])
    return parent, children


def _parse_zip_member_events(archive: zipfile.ZipFile, member: str) -> list[dict]:
    try:
        raw = archive.read(member)
    except KeyError:
        return []
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return []
    payload, err = _parse_trajectory_text(text, member)
    if err:
        return []
    payload = _normalize_payload(payload, member)
    if isinstance(payload, list):
        return [event for event in payload if isinstance(event, dict)]
    return []


def _load_zip_trajectory(file_path: str, format_hint: str | None, source_sha: str) -> dict:
    """Load a DSH export zip (``session.jsonl`` + optional ``subagents/``)."""
    try:
        archive = zipfile.ZipFile(file_path)
    except (OSError, zipfile.BadZipFile) as exc:
        return {"_error": f"Invalid zip archive: {exc}"}
    with archive:
        parent_member, child_members = _zip_dsh_members(archive.namelist())
        if not parent_member:
            return {"_error": "Zip archive has no session.jsonl."}
        events = _parse_zip_member_events(archive, parent_member)
        if not events:
            return {"_error": f"Could not parse {parent_member} from zip."}
        detected = detect_format(events)
        hint = format_hint if format_hint in FORMAT_LABELS else None
        fmt, reason = _resolve_format(detected, hint)
        if reason == "mismatch":
            return _format_mismatch_error(hint or "", detected)
        if fmt != "dsh":
            result = _apply_format(events, fmt, source_path=file_path)
        else:
            child_lists = [
                _parse_zip_member_events(archive, member)
                for _child_id, member in child_members
            ]
            result = _convert_dsh_to_internal(
                events, source_path=file_path, child_event_lists=child_lists,
            )
    if "_error" in result:
        return result
    result["_source_path"] = file_path
    result["_source_sha256"] = source_sha
    return result


def load_trajectory(file_path: str, format_hint: str | None = None) -> dict:
    """Load a trajectory file via the content dispatcher.

    Reads the file once (sha256 of those exact bytes), sniffs JSON object vs
    event array vs JSONL, detects format, then converts. ``format_hint``
    forces a converter when detection is ``unknown`` (unmarked Claude dumps)
    and *requires* that format when detection already succeeded.

    DeepSeek Harness exports may be a ``session.jsonl``, a session directory
    containing that file plus ``subagents/``, or a zip of that layout.
    """
    if _path_ext(file_path) == ".log":
        return {"_error": "Unsupported file type: .log files are no longer supported."}

    file_path, path_error = _resolve_trajectory_path(file_path)
    if path_error:
        return {"_error": path_error}

    try:
        with open(file_path, "rb") as f:
            data = f.read()
        source_sha = _sha256_bytes(data)
    except OSError as exc:
        return {"_error": str(exc)}

    ext = _path_ext(file_path)
    if ext == ".zip" or (data.startswith(b"PK") and ext not in {".json", ".jsonl"}):
        return _load_zip_trajectory(file_path, format_hint, source_sha)

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return {"_error": str(exc)}

    payload, parse_error = _parse_trajectory_text(text, file_path)
    if parse_error:
        return {"_error": parse_error}

    payload = _normalize_payload(payload, file_path)
    detected = detect_format(payload)
    hint = format_hint if format_hint in FORMAT_LABELS else None
    fmt, reason = _resolve_format(detected, hint)
    if reason == "mismatch":
        return _format_mismatch_error(hint or "", detected)

    result = _apply_format(payload, fmt, source_path=file_path)
    if "_error" in result:
        return result
    result["_source_path"] = file_path
    # The displayed content's immutable identity — the sha256 of the EXACT
    # bytes parsed above (one read, one buffer): attribution requires the
    # canonical corpus file to still have these bytes at diagnosis time
    # (TOCTOU guard — never diagnose bytes the UI isn't showing).
    result["_source_sha256"] = source_sha
    return result


def _sha256_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()
