"""ICode / Chrys expanded-session JSON → internal trajectory."""

from __future__ import annotations

import json
from typing import Any

from .common import _iso_to_epoch_ms
from .opencode import _convert_opencode_metadata

# ICode tools are lowercase (and `sh` for the shell). Map them onto the
# Claude Code / OpenCode vocabulary so charts, write detection, and spawn
# annotation work without a second set of aliases.
_ICODE_TOOL_NAMES = {
    "bash": "Bash",
    "sh": "Bash",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "grep": "Grep",
    "glob": "Glob",
    "find": "Glob",
    "ls": "Glob",
    "explore_agent": "Agent",
    "explore": "Agent",
}

_ICODE_KIND_TOOLS = {
    "shell": "Bash",
    "sub_agent": "Agent",
}


def _icode_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _icode_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _icode_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _icode_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _icode_props(message: dict) -> dict:
    return _icode_dict(message.get("additional_properties"))


def _icode_message_ts_ms(message: dict) -> int | None:
    props = _icode_props(message)
    ts = _iso_to_epoch_ms(props.get("_chrys_created_at") if isinstance(props.get("_chrys_created_at"), str) else None)
    if ts is not None:
        return ts
    return None


def _icode_parse_arguments(arguments: Any) -> dict:
    if isinstance(arguments, dict):
        return dict(arguments)
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}
        return dict(parsed) if isinstance(parsed, dict) else {"raw": parsed}
    return {"raw": arguments}


def _icode_normalize_tool(name: Any, arguments: Any, tool_kind: str = "") -> tuple[str, dict]:
    raw_name = name if isinstance(name, str) else ""
    canonical = _ICODE_TOOL_NAMES.get(raw_name.lower(), raw_name) if raw_name else ""
    if not canonical and tool_kind:
        canonical = _ICODE_KIND_TOOLS.get(tool_kind, "")
    if not canonical:
        canonical = raw_name or "?"
    args = _icode_parse_arguments(arguments)
    if canonical in ("Read", "Write", "Edit") and "file_path" not in args and "path" in args:
        args["file_path"] = args["path"]
    if canonical == "Bash" and "command" not in args and "cmd" in args:
        args["command"] = args["cmd"]
    return canonical, args


def _icode_iter_contents(contents: Any):
    if isinstance(contents, str):
        if contents:
            yield {"type": "text", "text": contents}
        return
    if isinstance(contents, dict):
        yield contents
        return
    if not isinstance(contents, list):
        return
    for item in contents:
        if isinstance(item, str):
            if item:
                yield {"type": "text", "text": item}
        elif isinstance(item, dict):
            yield item


def _icode_is_turn_marker(message: dict) -> bool:
    props = _icode_props(message)
    if props.get("_chrys_kind") == "turn":
        return True
    contents = _icode_list(message.get("contents"))
    if contents:
        return False
    return not _icode_str(message.get("role"))


def _icode_group_tokens(message: dict) -> dict | None:
    group = _icode_dict(_icode_props(message).get("_group"))
    count = group.get("token_count")
    if isinstance(count, bool) or not isinstance(count, (int, float)) or count <= 0:
        return None
    total = int(count)
    return {"total": total}


def _icode_tool_part(name: Any, arguments: Any, call_id: str, tool_kind: str = "") -> dict:
    tool_name, tool_input = _icode_normalize_tool(name, arguments, tool_kind)
    state = {
        "status": "pending",
        "input": tool_input,
        "output": "",
        "metadata": {},
    }
    if tool_kind:
        state["metadata"]["_chrys_tool_kind"] = tool_kind
    return {
        "type": "tool",
        "tool": tool_name,
        "tool_name": tool_name,
        "callID": call_id,
        "tool_id": call_id,
        "status": "pending",
        "input": tool_input,
        "output": "",
        "state": state,
        "metadata": dict(state["metadata"]),
    }


def _icode_apply_tool_result(
    part: dict,
    *,
    output: str,
    is_error: bool,
    error_text: str | None,
    error_type: str | None,
    ts: int | None,
    extra_metadata: dict | None = None,
) -> None:
    status = "error" if is_error else "completed"
    part["output"] = output
    part["status"] = status
    if is_error:
        part["error"] = error_text or output
        if error_type:
            part["error_type"] = error_type
    state = part.setdefault("state", {})
    if isinstance(state, dict):
        state["status"] = status
        state["output"] = output
        if is_error:
            state["error"] = error_text or output
        if ts:
            time_info = state.setdefault("time", {})
            if isinstance(time_info, dict):
                time_info["end"] = ts
        if extra_metadata:
            metadata = state.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata.update(extra_metadata)
    if extra_metadata:
        metadata = part.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata.update(extra_metadata)


def _icode_child_session_id(child_meta: dict) -> str:
    for key in ("invocation_id", "session_id", "id"):
        value = child_meta.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _icode_index_children(children: list) -> tuple[dict[str, dict], dict[str, dict]]:
    """Map invocation id and parent provider call id → child session dict."""
    by_invocation: dict[str, dict] = {}
    by_call: dict[str, dict] = {}
    for child in children:
        if not isinstance(child, dict):
            continue
        meta = _icode_dict(child.get("meta"))
        invocation = _icode_str(meta.get("invocation_id"))
        call_id = _icode_str(meta.get("parent_provider_call_id"))
        if invocation:
            by_invocation[invocation] = child
        if call_id:
            by_call[call_id] = child
    return by_invocation, by_call


def _icode_spawn_metadata(result_props: dict, call_id: str, children_by_invocation: dict, children_by_call: dict) -> tuple[dict, dict | None]:
    result_meta = _icode_dict(result_props.get("_chrys_tool_result_metadata"))
    invocation = _icode_str(result_meta.get("sub_agent_invocation_id"))
    child = None
    if invocation:
        child = children_by_invocation.get(invocation)
    if child is None and call_id:
        child = children_by_call.get(call_id)
    extra = dict(result_meta)
    extra.update({k: v for k, v in result_props.items() if k != "_chrys_tool_result_metadata"})
    if child is not None:
        child_meta = _icode_dict(child.get("meta"))
        child_id = _icode_child_session_id(child_meta)
        if child_id:
            extra["sessionId"] = child_id
        parent_id = _icode_str(child_meta.get("parent_session_id"))
        if parent_id:
            extra["parentSessionId"] = parent_id
        created = _iso_to_epoch_ms(child_meta.get("created_at") if isinstance(child_meta.get("created_at"), str) else None)
        ended = _iso_to_epoch_ms(child_meta.get("ended_at") if isinstance(child_meta.get("ended_at"), str) else None)
        if created is not None and ended is not None and ended >= created:
            extra["totalDurationMs"] = ended - created
    return extra, child


def _icode_function_result_fields(item: dict) -> tuple[str, str, bool, str | None, str | None, dict]:
    props = _icode_dict(item.get("additional_properties"))
    result = item.get("result")
    output = result if isinstance(result, str) else (str(result) if result else "")
    exception = item.get("exception")
    exception_text = exception if isinstance(exception, str) else (str(exception) if exception else "")
    error_kind = props.get("tool_error_kind")
    error_message = props.get("tool_error_message")
    failed = bool(props.get("failed")) or bool(exception_text) or bool(error_kind)
    if not output:
        output = exception_text or _icode_str(error_message)
    error_text = exception_text or _icode_str(error_message) or (output if failed else None)
    error_type = error_kind if isinstance(error_kind, str) and error_kind else None
    return output, exception_text, failed, error_text, error_type, props


def _icode_info_for_message(
    message: dict,
    *,
    role: str,
    ts: int | None,
    tokens: dict | None,
    session_id: str,
    parent_session_id: str,
    session_depth: int,
    session_title: str,
    is_sub_agent: bool,
    agent: str,
    model_id: str,
    provider_id: str,
    finish: str = "",
) -> dict:
    message_id = _icode_str(message.get("message_id"))
    info: dict[str, Any] = {
        "role": role,
        "time": {"created": ts or 0},
        "id": message_id,
        "sessionID": session_id,
        "agent": agent,
        "modelID": model_id,
        "providerID": provider_id,
        "isSubAgent": is_sub_agent,
    }
    if parent_session_id:
        info["parentSessionID"] = parent_session_id
    if session_depth:
        info["sessionDepth"] = session_depth
    if session_title:
        info["sessionTitle"] = session_title
    if tokens:
        info["tokens"] = tokens
    if finish:
        info["finish"] = finish
    return info


def _convert_icode_messages(
    messages: list,
    *,
    session_id: str,
    parent_session_id: str = "",
    session_depth: int = 0,
    session_title: str = "",
    is_sub_agent: bool = False,
    agent: str = "",
    model_id: str = "",
    provider_id: str = "",
    children_by_invocation: dict[str, dict] | None = None,
    children_by_call: dict[str, dict] | None = None,
    converted_children: dict[str, list[dict]] | None = None,
) -> list[dict]:
    """Convert one ICode session's messages into OpenCode-shaped records."""
    children_by_invocation = children_by_invocation or {}
    children_by_call = children_by_call or {}
    converted_children = converted_children or {}
    out: list[dict] = []
    pending_by_id: dict[str, list[dict]] = {}
    pending_anon: list[dict] = []
    inserts: dict[int, list[str]] = {}

    def _track(part: dict, call_id: str) -> None:
        if call_id:
            pending_by_id.setdefault(call_id, []).append(part)
        else:
            pending_anon.append(part)

    def _take(call_id: str) -> dict | None:
        if call_id:
            queued = pending_by_id.get(call_id)
            if queued:
                return queued.pop(0)
        if pending_anon:
            return pending_anon.pop(0)
        return None

    for message in messages:
        if not isinstance(message, dict):
            continue
        if _icode_is_turn_marker(message):
            continue
        role = _icode_str(message.get("role"))
        ts = _icode_message_ts_ms(message)
        tokens = _icode_group_tokens(message)

        if role == "user":
            parts = []
            for item in _icode_iter_contents(message.get("contents")):
                if item.get("type") in ("text", "reasoning", "thinking"):
                    text = item.get("text") or item.get("thinking") or ""
                    if item.get("type") in ("reasoning", "thinking"):
                        parts.append({"type": "reasoning", "text": text})
                    elif text:
                        parts.append({"type": "text", "text": text})
            if parts:
                info = _icode_info_for_message(
                    message, role="user", ts=ts, tokens=tokens,
                    session_id=session_id, parent_session_id=parent_session_id,
                    session_depth=session_depth, session_title=session_title,
                    is_sub_agent=is_sub_agent, agent=agent,
                    model_id=model_id, provider_id=provider_id,
                )
                out.append({
                    "info": info,
                    "parts": parts,
                    "message_id": info["id"],
                })
            continue

        if role == "assistant":
            parts = []
            has_tools = False
            for item in _icode_iter_contents(message.get("contents")):
                ctype = item.get("type")
                if ctype in ("reasoning", "thinking"):
                    parts.append({"type": "reasoning", "text": item.get("text") or item.get("thinking") or ""})
                elif ctype == "text":
                    text = item.get("text") or ""
                    if text:
                        parts.append({"type": "text", "text": text})
                elif ctype in ("function_call", "tool_call", "toolCall"):
                    call_id = _icode_str(item.get("call_id") or item.get("id"))
                    item_props = _icode_dict(item.get("additional_properties"))
                    tool_kind = _icode_str(item_props.get("_chrys_tool_kind"))
                    part = _icode_tool_part(item.get("name"), item.get("arguments"), call_id, tool_kind)
                    if ts:
                        time_info = part["state"].setdefault("time", {})
                        if isinstance(time_info, dict) and "start" not in time_info:
                            time_info["start"] = ts
                    parts.append(part)
                    _track(part, call_id)
                    has_tools = True
            if not parts and not tokens:
                continue
            finish = "tool-calls" if has_tools else "stop"
            info = _icode_info_for_message(
                message, role="assistant", ts=ts, tokens=tokens,
                session_id=session_id, parent_session_id=parent_session_id,
                session_depth=session_depth, session_title=session_title,
                is_sub_agent=is_sub_agent, agent=agent,
                model_id=model_id, provider_id=provider_id, finish=finish,
            )
            out.append({
                "info": info,
                "parts": parts,
                "message_id": info["id"],
            })
            continue

        if role != "tool":
            continue

        for item in _icode_iter_contents(message.get("contents")):
            if item.get("type") not in ("function_result", "tool_result", "toolResult"):
                continue
            call_id = _icode_str(item.get("call_id") or item.get("id"))
            output, _exception, failed, error_text, error_type, props = _icode_function_result_fields(item)
            extra, child = _icode_spawn_metadata(props, call_id, children_by_invocation, children_by_call)
            part = _take(call_id)
            if part is None:
                part = _icode_tool_part("?", {}, call_id)
                info = _icode_info_for_message(
                    message, role="assistant", ts=ts, tokens=tokens,
                    session_id=session_id, parent_session_id=parent_session_id,
                    session_depth=session_depth, session_title=session_title,
                    is_sub_agent=is_sub_agent, agent=agent,
                    model_id=model_id, provider_id=provider_id, finish="stop",
                )
                out.append({
                    "info": info,
                    "parts": [part],
                    "message_id": info["id"],
                })
            _icode_apply_tool_result(
                part,
                output=output,
                is_error=failed,
                error_text=error_text,
                error_type=error_type,
                ts=ts,
                extra_metadata=extra or None,
            )
            if child is not None:
                child_id = _icode_child_session_id(_icode_dict(child.get("meta")))
                if child_id and converted_children.get(child_id):
                    parent_idx = len(out) - 1
                    inserts.setdefault(parent_idx, [])
                    if child_id not in inserts[parent_idx]:
                        inserts[parent_idx].append(child_id)

    for part in (*[p for queued in pending_by_id.values() for p in queued], *pending_anon):
        if part.get("status") == "pending":
            _icode_apply_tool_result(
                part, output="", is_error=True, error_text="No tool result",
                error_type="missing_result", ts=None,
            )

    if not inserts:
        return out

    merged: list[dict] = []
    placed: set[str] = set()
    for idx, msg in enumerate(out):
        merged.append(msg)
        for child_id in inserts.get(idx, []):
            child_msgs = converted_children.get(child_id) or []
            merged.extend(child_msgs)
            placed.add(child_id)
    for child_id, child_msgs in converted_children.items():
        if child_id not in placed:
            merged.extend(child_msgs)
    return merged


def _convert_icode_session(
    session: dict,
    *,
    parent_session_id: str = "",
    session_depth: int = 0,
    is_sub_agent: bool = False,
) -> list[dict]:
    meta = _icode_dict(session.get("meta"))
    state = _icode_dict(session.get("state"))
    session_id = _icode_str(meta.get("session_id")) or _icode_child_session_id(meta) or parent_session_id
    agent = _icode_str(meta.get("agent_profile"))
    title = _icode_str(meta.get("agent_display_name") or meta.get("generated_title"))
    model_id = _icode_str(meta.get("model_id"))
    provider_id = _icode_str(meta.get("model_provider"))
    nested = _icode_list(session.get("_chrys_sub_agent_sessions"))
    by_invocation, by_call = _icode_index_children(nested)
    converted_children: dict[str, list[dict]] = {}
    for child in nested:
        if not isinstance(child, dict):
            continue
        child_meta = _icode_dict(child.get("meta"))
        child_id = _icode_child_session_id(child_meta)
        if not child_id:
            continue
        converted_children[child_id] = _convert_icode_session(
            child,
            parent_session_id=session_id,
            session_depth=session_depth + 1,
            is_sub_agent=True,
        )
    return _convert_icode_messages(
        _icode_list(state.get("messages")),
        session_id=session_id,
        parent_session_id=parent_session_id,
        session_depth=session_depth,
        session_title=title if is_sub_agent else "",
        is_sub_agent=is_sub_agent,
        agent=agent,
        model_id=model_id,
        provider_id=provider_id,
        children_by_invocation=by_invocation,
        children_by_call=by_call,
        converted_children=converted_children,
    )


def _fill_completed_times(messages: list[dict]) -> None:
    for i in range(len(messages) - 1):
        cur_t = messages[i]["info"]["time"].get("created")
        nxt_t = messages[i + 1]["info"]["time"].get("created")
        if isinstance(cur_t, (int, float)) and isinstance(nxt_t, (int, float)) and nxt_t >= cur_t:
            messages[i]["info"]["time"]["completed"] = nxt_t


def _convert_icode_to_internal(raw: dict) -> dict:
    """Convert an ICode / Chrys expanded session into the trajviz internal format.

    ICode exports a JSON object with ``meta`` + ``state.messages`` (user /
    assistant / tool roles, ``function_call`` / ``function_result`` contents)
    plus optional ``_chrys_sub_agent_sessions``.  Nested children are
    flattened into the OpenCode message/part shape used by ``parse_steps()``.
    """
    if raw.get("_icode_format") is True and isinstance(raw.get("messages"), list):
        return raw

    meta = _icode_dict(raw.get("meta"))
    state = _icode_dict(raw.get("state"))
    export = _icode_dict(raw.get("_chrys_export"))
    producer = _icode_dict(export.get("producer"))
    children = _icode_list(raw.get("_chrys_sub_agent_sessions"))

    messages = _convert_icode_session(raw, is_sub_agent=False, session_depth=0)
    _fill_completed_times(messages)

    session_id = _icode_str(meta.get("session_id"))
    directory = _icode_str(meta.get("primary_cwd"))
    created_iso = _icode_str(meta.get("created_at"))
    updated_iso = _icode_str(meta.get("updated_at"))
    created_ms = _iso_to_epoch_ms(created_iso) or 0
    updated_ms = _iso_to_epoch_ms(updated_iso) or created_ms
    if isinstance(updated_ms, int) and messages:
        last_info = messages[-1].get("info", {})
        last_time = last_info.get("time", {}) if isinstance(last_info, dict) else {}
        if isinstance(last_time, dict) and not last_time.get("completed"):
            last_created = last_time.get("created")
            if isinstance(last_created, (int, float)) and updated_ms >= last_created:
                last_time["completed"] = updated_ms

    info = {
        "id": session_id,
        "slug": "",
        "directory": directory,
        "title": _icode_str(meta.get("generated_title")),
        "version": _icode_str(meta.get("app_version")),
        "time": {"created": created_ms, "updated": updated_ms},
    }

    converted = {
        "info": info,
        "messages": messages,
        "_chrys_export": export or None,
        "_chrys_sub_agent_sessions": children,
    }
    _convert_opencode_metadata(converted)

    metadata = converted.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        converted["metadata"] = metadata

    model_id = _icode_str(meta.get("model_id")) or metadata.get("model", "")
    sub_agent_count = metadata.get("sub_agent_count", 0)
    if not isinstance(sub_agent_count, int):
        sub_agent_count = len(children)

    metadata.update({
        "session_id": session_id,
        "title": _icode_str(meta.get("generated_title")),
        "directory": directory,
        "directory_name": directory.replace("\\", "/").rsplit("/", 1)[-1] if directory else "",
        "agent": "icode",
        "model": model_id,
        "model_provider": _icode_str(meta.get("model_provider")),
        "platform": _icode_str(meta.get("os_name")),
        "originator": "ICode",
        "generator_name": _icode_str(producer.get("name")) or "icode",
        "generator_version": _icode_str(producer.get("version") or meta.get("app_version")),
        "format_version": _icode_str(export.get("format") or meta.get("schema_version")),
        "sub_agent_count": max(sub_agent_count, len(children)),
        "session_count": 1 + len(children),
        "timestamp_utc": created_iso,
        "export_complete": export.get("complete_relative_to_persisted_data"),
        "compressed_context_count": len(_icode_list(export.get("compressed_contexts"))),
        "agent_profile": _icode_str(meta.get("agent_profile")),
    })

    token_usage = {
        "total_tokens": _icode_int(state.get("total_session_tokens")),
        "prompt_tokens": _icode_int(state.get("total_session_input_tokens")),
        "completion_tokens": _icode_int(state.get("total_session_output_tokens")),
    }
    if token_usage["total_tokens"] == 0:
        token_usage["total_tokens"] = sum(
            _icode_int(_icode_dict(m.get("info", {}).get("tokens")).get("total"))
            for m in messages
            if isinstance(m, dict)
        )
    converted["token_usage"] = token_usage

    duration_seconds = 0.0
    if created_ms and updated_ms and updated_ms >= created_ms:
        duration_seconds = round((updated_ms - created_ms) / 1000.0, 3)
    converted["timing"] = {
        "total_duration": duration_seconds,
        "started_at": created_iso,
        "finished_at": updated_iso,
    }

    stats = converted.get("stats")
    if not isinstance(stats, dict):
        stats = {}
        converted["stats"] = stats
    failed_tool_calls = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for part in _icode_list(msg.get("parts")):
            if not isinstance(part, dict) or part.get("type") != "tool":
                continue
            status = _icode_str(part.get("status") or _icode_dict(part.get("state")).get("status"))
            if status == "error" or part.get("error"):
                failed_tool_calls += 1
    stats["failed_tool_calls"] = failed_tool_calls
    stats["sub_agent_count"] = metadata["sub_agent_count"]

    converted["_icode_format"] = True
    converted["_source_format"] = "icode"
    converted["_capabilities"] = {
        "has_timing": bool(created_ms),
        "has_tool_calls": bool(stats.get("total_tool_calls")),
        "has_runtime_token_usage": bool(token_usage.get("total_tokens")),
        "has_reasoning_content": bool(stats.get("reasoning_steps")),
        "has_session_hierarchy": bool(metadata["sub_agent_count"]),
    }
    return converted
