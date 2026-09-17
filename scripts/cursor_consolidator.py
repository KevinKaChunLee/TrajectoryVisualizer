#!/usr/bin/env python3
"""Read-only export of a Cursor IDE agent chat into one TrajViz JSON file.

Cursor splits a conversation across:

* ``~/.cursor/projects/<workspace>/agent-transcripts/<chat-id>/<chat-id>.jsonl``
  (user/assistant text and tool *inputs*, plus ``subagents/*.jsonl``)
* ``state.vscdb`` ``cursorDiskKV`` rows ``composerData:<chat-id>`` and
  ``bubbleId:<chat-id>:<bubble-id>`` (title, model, context-window snapshot,
  files changed, some tool *results*, thinking/turn durations)

This consolidator snapshots the DB (copy + WAL) and never writes Cursor's
files. It does **not** invent per-request billed tokens: Cursor does not
persist them. Session-level ``promptTokenBreakdown`` is a context-window
occupancy snapshot, stamped as such in ``export_metadata.token_semantics``.

Usage:
    python scripts/cursor_consolidator.py <chat-id> [output-file]
    python scripts/cursor_consolidator.py <chat-id> -o trajectory.json
    python scripts/cursor_consolidator.py --list

    If output-file is "-", JSON is written to stdout.

Environment:
    CURSOR_STATE_VSCDB      Path to state.vscdb (highest priority).
    CURSOR_USER_DATA_DIR    Cursor User/ directory (…/User/globalStorage/state.vscdb).
    CURSOR_PROJECTS_DIR     Projects root (default: ~/.cursor/projects).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Iterator, Sequence

try:
    from scripts import _common
except ImportError:  # Direct execution from scripts/.
    import _common  # type: ignore[no-redef]


SCHEMA_VERSION = 1
SOURCE_FORMAT = "cursor_composer"
GENERATOR_NAME = "cursor_consolidator"
TOKEN_SEMANTICS = "context_window_snapshot"
MAX_DEPTH = 8

# Internal Composer tool names → JSONL ``tool_use.name`` spellings.
_BUBBLE_TOOL_TO_JSONL = {
    "read_file_v2": "Read",
    "edit_file_v2": "StrReplace",
    "ripgrep_raw_search": "Grep",
    "run_terminal_command_v2": "Shell",
    "glob_file_search": "Glob",
    "task_v2": "Task",
    "todo_write": "TodoWrite",
    "get_mcp_tools": "GetDynamicTools",
    "set_active_branch": "SetActiveBranch",
    "await": "AwaitShell",
    "web_search": "WebSearch",
    "web_fetch": "WebFetch",
    "read_lints": "ReadLints",
    "delete_file": "Delete",
}

_SKIP_USERS = frozenset({
    "All Users", "Default", "Default User", "Public", "desktop.ini",
})


class ConsolidationError(RuntimeError):
    """Raised when a Cursor chat cannot be exported safely."""


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _maybe_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _as_dict(value: Any) -> dict[str, Any]:
    parsed = _maybe_json(value)
    return parsed if isinstance(parsed, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _stringify_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _tool_path(payload: Any) -> str:
    data = _as_dict(payload)
    for key in ("path", "file_path", "filePath", "target_file", "targetFile", "uri"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _uri_path(uri: Any) -> str:
    if isinstance(uri, str) and uri:
        return uri
    if not isinstance(uri, dict):
        return ""
    path = uri.get("path")
    if isinstance(path, str) and path:
        return path
    external = uri.get("external")
    if isinstance(external, str) and external:
        return external
    return ""


def default_projects_dir() -> Path:
    configured = os.environ.get("CURSOR_PROJECTS_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cursor" / "projects"


def default_state_vscdb_candidates() -> list[Path]:
    """Ordered candidate paths for Cursor's global ``state.vscdb``."""
    seen: set[Path] = set()
    ordered: list[Path] = []

    def add(path: Path) -> None:
        resolved = path.expanduser()
        if resolved in seen:
            return
        seen.add(resolved)
        ordered.append(resolved)

    configured = os.environ.get("CURSOR_STATE_VSCDB")
    if configured:
        add(Path(configured))
    user_data = os.environ.get("CURSOR_USER_DATA_DIR")
    if user_data:
        root = Path(user_data)
        add(root / "User" / "globalStorage" / "state.vscdb")
        add(root / "globalStorage" / "state.vscdb")
    appdata = os.environ.get("APPDATA")
    if appdata:
        add(Path(appdata) / "Cursor" / "User" / "globalStorage" / "state.vscdb")
    home = Path.home()
    add(home / ".config" / "Cursor" / "User" / "globalStorage" / "state.vscdb")
    add(home / "Library" / "Application Support" / "Cursor" / "User" / "globalStorage" / "state.vscdb")
    users = Path("/mnt/c/Users")
    if users.is_dir():
        for entry in sorted(users.iterdir(), key=lambda p: p.name.lower()):
            if not entry.is_dir() or entry.name in _SKIP_USERS:
                continue
            add(
                entry / "AppData" / "Roaming" / "Cursor" / "User"
                / "globalStorage" / "state.vscdb"
            )
    return ordered


def resolve_state_vscdb(explicit: str | Path | None = None) -> Path | None:
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConsolidationError(f"Database not found: {path}")
        return path.resolve()
    for candidate in default_state_vscdb_candidates():
        if candidate.is_file():
            return candidate.resolve()
    return None


def snapshot_state_vscdb(source: Path, destination_dir: Path) -> Path:
    """Copy ``state.vscdb`` plus WAL/SHM so a live Cursor process can keep writing."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    dest = destination_dir / "state.vscdb"
    try:
        shutil.copy2(source, dest)
    except OSError as exc:
        raise ConsolidationError(f"Cannot snapshot database {source}: {exc}") from exc
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(source) + suffix)
        if sidecar.is_file():
            try:
                shutil.copy2(sidecar, destination_dir / f"state.vscdb{suffix}")
            except OSError:
                pass
    return dest


def open_state_vscdb_read_only(path: Path) -> sqlite3.Connection:
    uri = f"{path.expanduser().resolve().as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
    except sqlite3.Error as exc:
        raise ConsolidationError(f"Cannot open database read-only at {path}: {exc}") from exc
    return connection


def _kv_get(connection: sqlite3.Connection, key: str) -> Any:
    try:
        row = connection.execute(
            "SELECT value FROM cursorDiskKV WHERE key = ?", (key,)
        ).fetchone()
    except sqlite3.Error as exc:
        raise ConsolidationError(f"cursorDiskKV read failed: {exc}") from exc
    if row is None:
        return None
    return _maybe_json(row[0])


def load_composer(connection: sqlite3.Connection, chat_id: str) -> dict[str, Any] | None:
    value = _kv_get(connection, f"composerData:{chat_id}")
    return value if isinstance(value, dict) else None


def load_ordered_bubbles(
    connection: sqlite3.Connection, chat_id: str, composer: dict[str, Any]
) -> list[dict[str, Any]]:
    headers = _as_list(composer.get("fullConversationHeadersOnly"))
    header_ids = [
        _as_str(item.get("bubbleId"))
        for item in headers
        if isinstance(item, dict) and _as_str(item.get("bubbleId"))
    ]
    bubbles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bubble_id in header_ids:
        payload = _kv_get(connection, f"bubbleId:{chat_id}:{bubble_id}")
        if not isinstance(payload, dict):
            continue
        payload = dict(payload)
        payload.setdefault("bubbleId", bubble_id)
        bubbles.append(payload)
        seen.add(bubble_id)
    if header_ids:
        return bubbles
    try:
        rows = connection.execute(
            "SELECT key, value FROM cursorDiskKV WHERE key LIKE ?",
            (f"bubbleId:{chat_id}:%",),
        ).fetchall()
    except sqlite3.Error:
        return bubbles
    for row in rows:
        key = row[0] if not isinstance(row, sqlite3.Row) else row["key"]
        value = _maybe_json(row[1] if not isinstance(row, sqlite3.Row) else row["value"])
        if not isinstance(value, dict):
            continue
        bubble_id = str(key).rsplit(":", 1)[-1]
        if bubble_id in seen:
            continue
        value = dict(value)
        value.setdefault("bubbleId", bubble_id)
        bubbles.append(value)
    return bubbles


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a Cursor agent-transcript JSONL file; drop a truncated final line."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConsolidationError(f"Cannot read transcript {path}: {exc}") from exc
    events: list[dict[str, Any]] = []
    pending: tuple[int, str] | None = None
    for line_number, raw_line in enumerate(text.splitlines(keepends=True), start=1):
        if not raw_line.strip():
            continue
        if pending is None:
            pending = (line_number, raw_line)
            continue
        pending_number, pending_line = pending
        try:
            parsed = json.loads(pending_line)
        except json.JSONDecodeError as exc:
            raise ConsolidationError(
                f"Invalid JSONL at {path}:{pending_number}: {exc.msg}"
            ) from exc
        if isinstance(parsed, dict):
            events.append(parsed)
        pending = (line_number, raw_line)
    if pending is not None:
        pending_number, pending_line = pending
        try:
            parsed = json.loads(pending_line)
        except json.JSONDecodeError:
            if pending_line.endswith(("\n", "\r")):
                raise ConsolidationError(
                    f"Invalid JSONL at {path}:{pending_number}"
                ) from None
            return events
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def find_transcript_locations(
    chat_id: str, projects_dir: Path
) -> tuple[Path | None, Path | None]:
    """Return ``(session_dir, jsonl_path)`` for a chat id, if present."""
    if not projects_dir.is_dir():
        return None, None
    nested = list(projects_dir.glob(f"*/agent-transcripts/{chat_id}/{chat_id}.jsonl"))
    if nested:
        jsonl = nested[0]
        return jsonl.parent, jsonl
    dirs = [
        path for path in projects_dir.glob(f"*/agent-transcripts/{chat_id}")
        if path.is_dir()
    ]
    for folder in dirs:
        candidate = folder / f"{chat_id}.jsonl"
        if candidate.is_file():
            return folder, candidate
        loose = list(folder.glob("*.jsonl"))
        if loose:
            return folder, loose[0]
    flat = list(projects_dir.glob(f"*/agent-transcripts/{chat_id}.jsonl"))
    if flat:
        return flat[0].parent, flat[0]
    return None, None


def resolve_chat_input(
    raw: str, projects_dir: Path
) -> tuple[str, Path | None, Path | None]:
    """Resolve a chat id or filesystem path to ``(chat_id, dir, jsonl)``."""
    path = Path(raw).expanduser()
    if path.exists():
        resolved = path.resolve()
        if resolved.is_file():
            chat_id = resolved.stem
            parent = resolved.parent
            if parent.name == chat_id:
                return chat_id, parent, resolved
            return chat_id, parent, resolved
        if resolved.is_dir():
            chat_id = resolved.name
            jsonl = resolved / f"{chat_id}.jsonl"
            if jsonl.is_file():
                return chat_id, resolved, jsonl
            loose = sorted(resolved.glob("*.jsonl"))
            if loose:
                return loose[0].stem, resolved, loose[0]
            raise ConsolidationError(f"No JSONL transcript in directory: {resolved}")
    chat_id = raw.strip()
    if not chat_id:
        raise ConsolidationError("Chat id is empty")
    folder, jsonl = find_transcript_locations(chat_id, projects_dir)
    return chat_id, folder, jsonl


def _child_jsonl_path(parent_dir: Path | None, child_id: str, projects_dir: Path) -> Path | None:
    if parent_dir is not None:
        nested = parent_dir / "subagents" / f"{child_id}.jsonl"
        if nested.is_file():
            return nested
    _, jsonl = find_transcript_locations(child_id, projects_dir)
    return jsonl


def discover_child_ids(
    parent_dir: Path | None,
    composer: dict[str, Any] | None,
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()

    def add(chat_id: str) -> None:
        if chat_id and chat_id not in seen:
            seen.add(chat_id)
            ids.append(chat_id)

    if parent_dir is not None:
        subagents = parent_dir / "subagents"
        if subagents.is_dir():
            for path in sorted(subagents.glob("*.jsonl")):
                add(path.stem)
    composer_ids = _as_list((composer or {}).get("subagentComposerIds"))
    for item in composer_ids:
        if isinstance(item, str):
            add(item)
    return ids


def _extract_tool_former(bubble: dict[str, Any]) -> dict[str, Any] | None:
    former = bubble.get("toolFormerData")
    if not isinstance(former, dict):
        return None
    name = _as_str(former.get("name")) or _as_str(former.get("toolName"))
    raw_args = _maybe_json(former.get("rawArgs"))
    params = _maybe_json(former.get("params"))
    result = _maybe_json(former.get("result"))
    status = _as_str(former.get("status")) or "completed"
    error = former.get("error")
    return {
        "name": name,
        "jsonl_name": _BUBBLE_TOOL_TO_JSONL.get(name, name),
        "path": _tool_path(raw_args) or _tool_path(params),
        "raw_args": raw_args if isinstance(raw_args, dict) else {},
        "params": params if isinstance(params, dict) else {},
        "result": result,
        "status": status,
        "error": error,
        "tool_call_id": _as_str(former.get("toolCallId")),
    }


def _bubble_role(bubble: dict[str, Any]) -> str:
    bubble_type = bubble.get("type")
    if bubble_type == 1:
        return "user"
    if bubble_type == 2:
        return "assistant"
    if _extract_tool_former(bubble) is not None:
        return "assistant"
    return ""


def _tools_equivalent(jsonl_name: str, bubble: dict[str, Any]) -> bool:
    mapped = _as_str(bubble.get("jsonl_name"))
    internal = _as_str(bubble.get("name"))
    if jsonl_name in (mapped, internal):
        return True
    return jsonl_name == "CallDynamicTool" and (
        internal.startswith("mcp-") or internal.startswith("mcp_")
    )


def _match_bubble_tool(
    jsonl_name: str,
    jsonl_input: dict[str, Any],
    pool: list[dict[str, Any]],
) -> dict[str, Any] | None:
    want_path = _tool_path(jsonl_input)
    for index, bubble in enumerate(pool):
        if not _tools_equivalent(jsonl_name, bubble):
            continue
        bubble_path = _as_str(bubble.get("path"))
        if want_path and bubble_path and want_path != bubble_path:
            continue
        return pool.pop(index)
    for index, bubble in enumerate(pool):
        if _tools_equivalent(jsonl_name, bubble):
            return pool.pop(index)
    return None


def _composer_info(chat_id: str, composer: dict[str, Any] | None) -> dict[str, Any]:
    composer = composer or {}
    model_config = _as_dict(composer.get("modelConfig"))
    workspace = _as_dict(composer.get("workspaceIdentifier"))
    directory = _uri_path(_as_dict(workspace.get("uri"))) or _as_str(workspace.get("id"))
    created = _as_int(composer.get("createdAt")) or 0
    updated = _as_int(composer.get("lastUpdatedAt")) or created
    file_uris = [
        _uri_path(_as_dict(item.get("uri"))) if isinstance(item, dict) else _uri_path(item)
        for item in _as_list(composer.get("newlyCreatedFiles"))
    ]
    original_states = composer.get("originalFileStates")
    original_paths = list(original_states) if isinstance(original_states, dict) else []
    return {
        "id": chat_id,
        "title": _as_str(composer.get("name")),
        "subtitle": _as_str(composer.get("subtitle")),
        "directory": directory,
        "status": _as_str(composer.get("status")),
        "unifiedMode": composer.get("unifiedMode"),
        "isAgentic": composer.get("isAgentic"),
        "model": _as_str(model_config.get("modelName")),
        "modelConfig": {
            key: model_config[key]
            for key in ("modelName", "maxMode", "selectedModels")
            if key in model_config
        },
        "time": {"created": created, "updated": updated},
        "summary": {
            "additions": composer.get("totalLinesAdded") or 0,
            "deletions": composer.get("totalLinesRemoved") or 0,
            "files": composer.get("filesChangedCount") or 0,
        },
        "promptTokenBreakdown": composer.get("promptTokenBreakdown"),
        "contextUsagePercent": composer.get("contextUsagePercent"),
        "contextTokensUsed": composer.get("contextTokensUsed"),
        "contextTokenLimit": composer.get("contextTokenLimit"),
        "newlyCreatedFiles": [path for path in file_uris if path],
        "originalFilePaths": original_paths,
        "subagentComposerIds": [
            item for item in _as_list(composer.get("subagentComposerIds"))
            if isinstance(item, str)
        ],
        "usageData": composer.get("usageData") if isinstance(composer.get("usageData"), dict) else {},
    }


def _align_role_bubbles(
    events: Sequence[dict[str, Any]], bubbles: Sequence[dict[str, Any]]
) -> list[dict[str, Any] | None]:
    """Pair JSONL role rows with Composer bubbles of the same role, in order."""
    by_role: dict[str, list[dict[str, Any]]] = {"user": [], "assistant": []}
    for bubble in bubbles:
        role = _bubble_role(bubble)
        if role in by_role:
            by_role[role].append(bubble)
    aligned: list[dict[str, Any] | None] = []
    for event in events:
        role = _as_str(event.get("role"))
        if role not in by_role:
            aligned.append(None)
            continue
        bucket = by_role[role]
        aligned.append(bucket.pop(0) if bucket else None)
    return aligned


def _assistant_time(bubble: dict[str, Any] | None, session_cursor: list[int]) -> dict[str, int]:
    if not bubble:
        return {}
    turn_ms = _as_int(bubble.get("turnDurationMs"))
    think_ms = _as_int(bubble.get("thinkingDurationMs"))
    duration = turn_ms or think_ms
    if not duration or duration <= 0:
        time_info: dict[str, int] = {}
        if think_ms and think_ms > 0:
            time_info["thinkingMs"] = think_ms
        return time_info
    start = session_cursor[0]
    if start <= 0:
        start = 1
    completed = start + duration
    session_cursor[0] = completed
    time_info = {"created": start, "completed": completed}
    if think_ms and think_ms > 0:
        time_info["thinkingMs"] = think_ms
    return time_info


def events_to_messages(
    events: Sequence[dict[str, Any]],
    *,
    session_id: str,
    parent_session_id: str = "",
    depth: int = 0,
    agent: str = "cursor",
    model: str = "",
    bubbles: Sequence[dict[str, Any]] | None = None,
    child_ids: Sequence[str] | None = None,
    session_cursor: list[int] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Convert Cursor JSONL events into OpenCode-shaped ``info`` + ``parts`` messages."""
    bubbles = list(bubbles or [])
    aligned = _align_role_bubbles(events, bubbles)
    tool_pool = [item for item in (_extract_tool_former(b) for b in bubbles) if item]
    child_queue = list(child_ids or [])
    cursor = session_cursor if session_cursor is not None else [0]
    messages: list[dict[str, Any]] = []
    tool_index = 0
    last_assistant: dict[str, Any] | None = None
    is_sub = depth > 0

    for event_index, event in enumerate(events):
        event_type = event.get("type")
        if event_type == "turn_ended" and last_assistant is not None:
            finish = _as_str(event.get("status")) or "success"
            last_assistant["info"]["finish"] = finish
            continue
        role = _as_str(event.get("role"))
        if role not in ("user", "assistant"):
            continue
        bubble = aligned[event_index] if event_index < len(aligned) else None
        bubble_id = _as_str((bubble or {}).get("bubbleId"))
        message_id = bubble_id or f"{session_id}:{role}:{event_index}"
        content = _as_list(_as_dict(event.get("message")).get("content"))
        parts: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append({
                        "type": "text",
                        "text": text,
                        "id": f"{message_id}:text:{len(parts)}",
                        "sessionID": session_id,
                        "messageID": message_id,
                    })
            elif btype in ("tool_use", "tool_call"):
                name = _as_str(block.get("name")) or "?"
                tool_input = block.get("input") if isinstance(block.get("input"), dict) else {}
                matched = _match_bubble_tool(name, tool_input, tool_pool)
                call_id = (
                    _as_str((matched or {}).get("tool_call_id"))
                    or _as_str(block.get("id"))
                    or f"{session_id}:tool:{tool_index}"
                )
                tool_index += 1
                status = "unknown"
                output = ""
                error = None
                metadata: dict[str, Any] = {}
                if matched is not None:
                    raw_status = _as_str(matched.get("status")).lower()
                    if matched.get("error") or raw_status in {"error", "failed", "failure"}:
                        status = "error"
                        error = matched.get("error")
                    else:
                        status = "completed"
                    output = _stringify_output(matched.get("result"))
                if name == "Task" and child_queue:
                    child_id = child_queue.pop(0)
                    metadata["sessionId"] = child_id
                    metadata["parentSessionId"] = session_id
                state = {
                    "status": status,
                    "input": tool_input,
                    "output": output,
                    "metadata": metadata,
                }
                if error is not None:
                    state["error"] = error
                parts.append({
                    "type": "tool",
                    "tool": name,
                    "callID": call_id,
                    "state": state,
                    "id": call_id,
                    "sessionID": session_id,
                    "messageID": message_id,
                })

        info: dict[str, Any] = {
            "role": role,
            "id": message_id,
            "sessionID": session_id,
            "agent": agent,
            "isSubAgent": is_sub,
            "sessionDepth": depth,
        }
        if parent_session_id:
            info["parentSessionID"] = parent_session_id
        if model:
            info["modelID"] = model
        if role == "assistant":
            time_info = _assistant_time(bubble, cursor)
            if time_info:
                info["time"] = time_info
            last_assistant = None
        message = {"info": info, "parts": parts}
        if role == "assistant":
            last_assistant = message
        messages.append(message)
    return messages, tool_index


def _session_entry(
    chat_id: str,
    *,
    parent_id: str,
    depth: int,
    jsonl_path: Path | None,
    message_count: int,
    title: str = "",
) -> dict[str, Any]:
    return {
        "id": chat_id,
        "parent_id": parent_id or None,
        "depth": depth,
        "title": title,
        "transcript_path": str(jsonl_path) if jsonl_path else "",
        "message_count": message_count,
    }


def consolidate_chat(
    chat_id: str,
    *,
    projects_dir: Path | None = None,
    database_path: Path | None = None,
    transcript_dir: Path | None = None,
    jsonl_path: Path | None = None,
) -> dict[str, Any]:
    """Export one Cursor chat and reachable subagent transcripts."""
    projects_dir = (projects_dir or default_projects_dir()).expanduser()
    warnings: list[str] = []
    folder = transcript_dir
    if jsonl_path is None or folder is None:
        found_dir, found_jsonl = find_transcript_locations(chat_id, projects_dir)
        folder = folder or found_dir
        jsonl_path = jsonl_path or found_jsonl
    if jsonl_path is None:
        raise ConsolidationError(
            f"No agent-transcript JSONL for chat {chat_id} under {projects_dir}"
        )

    snapshot_dir: tempfile.TemporaryDirectory[str] | None = None
    connection: sqlite3.Connection | None = None
    db_source = database_path if database_path is not None else resolve_state_vscdb()
    db_used: Path | None = None
    try:
        if db_source is not None:
            snapshot_dir = tempfile.TemporaryDirectory(prefix="cursor-state-")
            snapped = snapshot_state_vscdb(db_source, Path(snapshot_dir.name))
            connection = open_state_vscdb_read_only(snapped)
            db_used = db_source
        elif database_path is None:
            warnings.append(
                "state.vscdb not found; export is JSONL-only (no context snapshot, "
                "model, or tool results)."
            )

        return _consolidate_with_connection(
            chat_id,
            projects_dir=projects_dir,
            folder=folder,
            jsonl_path=jsonl_path,
            connection=connection,
            db_used=db_used,
            warnings=warnings,
        )
    finally:
        if connection is not None:
            connection.close()
        if snapshot_dir is not None:
            snapshot_dir.cleanup()


def _consolidate_with_connection(
    chat_id: str,
    *,
    projects_dir: Path,
    folder: Path | None,
    jsonl_path: Path | None,
    connection: sqlite3.Connection | None,
    db_used: Path | None,
    warnings: list[str],
    parent_folder: Path | None = None,
    parent_id: str = "",
    depth: int = 0,
    visited: set[str] | None = None,
    all_messages: list[dict[str, Any]] | None = None,
    sessions: list[dict[str, Any]] | None = None,
    session_cursor: list[int] | None = None,
) -> dict[str, Any]:
    visited = visited if visited is not None else set()
    all_messages = all_messages if all_messages is not None else []
    sessions = sessions if sessions is not None else []
    if chat_id in visited:
        warnings.append(f"Chat {chat_id} already exported, skipping cycle/diamond")
        return {}
    if depth > MAX_DEPTH:
        warnings.append(f"Max subagent depth {MAX_DEPTH} reached at {chat_id}")
        return {}
    visited.add(chat_id)

    composer = load_composer(connection, chat_id) if connection is not None else None
    if connection is not None and composer is None and depth == 0:
        warnings.append(
            f"composerData:{chat_id} missing from state.vscdb; metadata will be sparse."
        )
    bubbles = (
        load_ordered_bubbles(connection, chat_id, composer or {})
        if connection is not None
        else []
    )

    if jsonl_path is None:
        jsonl_path = _child_jsonl_path(parent_folder, chat_id, projects_dir)
        if jsonl_path is not None:
            folder = jsonl_path.parent if jsonl_path.parent.name == "subagents" else jsonl_path.parent
            if jsonl_path.parent.name == "subagents":
                folder = jsonl_path.parent.parent

    events: list[dict[str, Any]] = []
    if jsonl_path is not None and jsonl_path.is_file():
        events = parse_jsonl(jsonl_path)
    elif depth > 0:
        warnings.append(f"No JSONL transcript for subagent {chat_id}")
    else:
        raise ConsolidationError(f"No JSONL transcript for chat {chat_id}")

    info = _composer_info(chat_id, composer)
    if session_cursor is None:
        session_cursor = [int(info["time"].get("created") or 0)]
    child_ids = discover_child_ids(folder, composer)
    agent = "cursor" if depth == 0 else "cursor (subagent)"
    messages, _tool_count = events_to_messages(
        events,
        session_id=chat_id,
        parent_session_id=parent_id,
        depth=depth,
        agent=agent,
        model=_as_str(info.get("model")),
        bubbles=bubbles,
        child_ids=child_ids,
        session_cursor=session_cursor,
    )
    all_messages.extend(messages)
    sessions.append(
        _session_entry(
            chat_id,
            parent_id=parent_id,
            depth=depth,
            jsonl_path=jsonl_path,
            message_count=len(messages),
            title=_as_str(info.get("title")),
        )
    )

    for child_id in child_ids:
        _consolidate_with_connection(
            child_id,
            projects_dir=projects_dir,
            folder=None,
            jsonl_path=None,
            connection=connection,
            db_used=db_used,
            warnings=warnings,
            parent_folder=folder,
            parent_id=chat_id,
            depth=depth + 1,
            visited=visited,
            all_messages=all_messages,
            sessions=sessions,
            session_cursor=session_cursor,
        )

    if depth > 0:
        return {}

    user_messages = sum(
        1 for msg in all_messages
        if _as_dict(msg.get("info")).get("role") == "user"
    )
    assistant_messages = sum(
        1 for msg in all_messages
        if _as_dict(msg.get("info")).get("role") == "assistant"
    )
    tool_parts = 0
    tools_with_output = 0
    for msg in all_messages:
        for part in _as_list(msg.get("parts")):
            if not isinstance(part, dict) or part.get("type") != "tool":
                continue
            tool_parts += 1
            state = _as_dict(part.get("state"))
            if state.get("output"):
                tools_with_output += 1

    export_metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_format": SOURCE_FORMAT,
        "generator_name": GENERATOR_NAME,
        "generated_at": _now_iso(),
        "chat_id": chat_id,
        "transcript_path": str(jsonl_path) if jsonl_path else "",
        "database_path": str(db_used) if db_used else "",
        "token_semantics": TOKEN_SEMANTICS,
        "complete": not warnings,
        "warnings": warnings,
    }
    statistics = {
        "sessions": len(sessions),
        "subagent_sessions": max(0, len(sessions) - 1),
        "total_messages": len(all_messages),
        "user_messages": user_messages,
        "assistant_messages": assistant_messages,
        "tool_parts": tool_parts,
        "tool_parts_with_output": tools_with_output,
    }
    return {
        "info": info,
        "messages": all_messages,
        "sessions": sessions,
        "statistics": statistics,
        "export_metadata": export_metadata,
    }


def write_output(data: dict[str, Any], output_path: str | Path) -> None:
    if str(output_path) == "-":
        json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _common.write_json_atomic(path, data)


def ensure_output_is_not_source(
    output_path: str | Path, source_paths: Iterable[str | Path]
) -> None:
    _common.ensure_output_does_not_overwrite(
        output_path,
        source_paths,
        exc=ConsolidationError,
        allow_stdout_dash=True,
    )


def iter_transcript_chats(projects_dir: Path) -> Iterator[tuple[str, Path, float]]:
    if not projects_dir.is_dir():
        return
    seen: set[str] = set()
    for jsonl in projects_dir.glob("*/agent-transcripts/*/*.jsonl"):
        if jsonl.parent.name == "subagents":
            continue
        chat_id = jsonl.parent.name
        if chat_id in seen:
            continue
        seen.add(chat_id)
        yield chat_id, jsonl, jsonl.stat().st_mtime
    for jsonl in projects_dir.glob("*/agent-transcripts/*.jsonl"):
        chat_id = jsonl.stem
        if chat_id in seen:
            continue
        seen.add(chat_id)
        yield chat_id, jsonl, jsonl.stat().st_mtime


def list_chats(projects_dir: Path, database_path: Path | None) -> int:
    rows = sorted(iter_transcript_chats(projects_dir), key=lambda item: item[2], reverse=True)
    if not rows:
        print(f"No agent-transcripts under {projects_dir}", file=sys.stderr)
        return 1
    titles: dict[str, str] = {}
    snapshot_dir: tempfile.TemporaryDirectory[str] | None = None
    connection: sqlite3.Connection | None = None
    db_source = database_path if database_path is not None else resolve_state_vscdb()
    try:
        if db_source is not None:
            snapshot_dir = tempfile.TemporaryDirectory(prefix="cursor-state-")
            snapped = snapshot_state_vscdb(db_source, Path(snapshot_dir.name))
            connection = open_state_vscdb_read_only(snapped)
            for chat_id, _path, _mtime in rows:
                composer = load_composer(connection, chat_id)
                if composer and _as_str(composer.get("name")):
                    titles[chat_id] = _as_str(composer.get("name"))
    except ConsolidationError as exc:
        print(f"Warning: {exc}", file=sys.stderr)
    finally:
        if connection is not None:
            connection.close()
        if snapshot_dir is not None:
            snapshot_dir.cleanup()
    for chat_id, path, mtime in rows:
        stamp = datetime.fromtimestamp(mtime, tz=UTC).strftime("%Y-%m-%d %H:%M")
        title = titles.get(chat_id, "")
        suffix = f"  {title}" if title else ""
        print(f"{chat_id}  {stamp}  {path}{suffix}")
    return 0


def _print_summary(data: dict[str, Any], output: str | Path) -> None:
    stats = data.get("statistics", {})
    chat_id = data.get("export_metadata", {}).get("chat_id") or data.get("info", {}).get("id")
    print(f"Consolidated: {chat_id}", file=sys.stderr)
    print(
        f"  Sessions: {stats.get('sessions', 1)}; "
        f"messages: {stats.get('total_messages', 0)}; "
        f"tools: {stats.get('tool_parts', 0)} "
        f"({stats.get('tool_parts_with_output', 0)} with outputs)",
        file=sys.stderr,
    )
    print(f"  Output: {output}", file=sys.stderr)
    export_metadata = data.get("export_metadata", {})
    for warning in _as_list(export_metadata.get("warnings")):
        print(f"  Warning: {warning}", file=sys.stderr)
    if export_metadata.get("complete") is False:
        print("  Completeness: partial (see warnings above)", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a Cursor IDE agent chat (JSONL + Composer DB) to TrajViz JSON.",
    )
    parser.add_argument(
        "chat_id",
        nargs="?",
        help="Composer/chat UUID, or a path to the transcript JSONL / session folder.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output JSON path. Defaults to export-<chat-id>.json. Use '-' for stdout.",
    )
    parser.add_argument("-o", "--output-file", dest="output_file", help="Output JSON path (overrides positional).")
    parser.add_argument("--db", dest="database", help="Path to Cursor state.vscdb.")
    parser.add_argument(
        "--projects-dir",
        dest="projects_dir",
        help="Cursor projects root (default: ~/.cursor/projects or CURSOR_PROJECTS_DIR).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List chat ids found under the projects directory and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    projects_dir = Path(args.projects_dir).expanduser() if args.projects_dir else default_projects_dir()
    database = Path(args.database).expanduser() if args.database else None
    try:
        if args.list:
            return list_chats(projects_dir, database)
        if not args.chat_id:
            parser.print_help(sys.stderr)
            return 2
        chat_id, folder, jsonl_path = resolve_chat_input(args.chat_id, projects_dir)
        output = args.output_file or args.output or f"export-{chat_id}.json"
        sources = [path for path in (jsonl_path, folder, database) if path is not None]
        ensure_output_is_not_source(output, sources)
        data = consolidate_chat(
            chat_id,
            projects_dir=projects_dir,
            database_path=database,
            transcript_dir=folder,
            jsonl_path=jsonl_path,
        )
        write_output(data, output)
        if str(output) != "-":
            _print_summary(data, output)
        return 0
    except ConsolidationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
