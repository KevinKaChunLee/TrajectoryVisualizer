"""Pattern detection for recurring tool sequences, workflow phases, and anti-patterns."""

from __future__ import annotations

import shlex
from bisect import bisect_right
from collections import Counter

from trajviz.insight.parser import spawned_child_session_id


# ---------------------------------------------------------------------------
# Expected phase ordering for anomaly detection
# ---------------------------------------------------------------------------

_PHASE_ORDER = ["understand", "plan", "implement", "debug", "validate", "report"]
_PHASE_RANK: dict[str, int] = {name: i for i, name in enumerate(_PHASE_ORDER)}

# Cap to avoid pathological inputs
_MAX_STEPS = 2000

_PLAN_TOOL_NAMES = {
    "TodoWrite", "todowrite", "TodoUpdate", "TaskCreate", "TaskUpdate",
    "TaskList", "EnterPlanMode",
}
_READ_TOOL_NAMES = {"Read", "read", "WebFetch"}
_SEARCH_TOOL_NAMES = {
    "Bash", "bash", "Grep", "Glob", "grep", "glob", "find", "ToolSearch", "WebSearch",
}
# Single source of truth for write-tool names: trajviz.tool_vocab.
from trajviz.tool_vocab import WRITE_TOOL_NAMES as _WRITE_TOOL_NAMES  # noqa: E402
_VALIDATION_COMMAND_PATTERNS = (
    "pytest", "python -m pytest", "unittest", "tox", "nox", "go test",
    "cargo test", "npm test", "pnpm test", "yarn test", "jest", "vitest",
    "mvn test", "gradle test", "bazel test", "make test", "ctest", "ruff",
    "flake8", "pylint", "mypy", "eslint", "lint", "check", "verify",
)


def _has_plan_snapshot(step: dict) -> bool:
    """Return True when a step carries an explicit TODO/task snapshot."""
    for part in step.get("parts", []):
        if part.get("type") != "snapshot":
            continue
        data = part.get("data", {})
        if isinstance(data, dict) and ("todos" in data or "items" in data):
            return True
    return False


def _tool_failed(tc: dict) -> bool:
    """Return True when a tool call clearly failed."""
    status = tc.get("status", "")
    if status in ("error", "failed", "failure", "cancelled", "timeout"):
        return True
    meta = tc.get("metadata", {})
    return isinstance(meta, dict) and meta.get("exit") not in (None, 0)


def _is_validation_command(command: str) -> bool:
    """Return True when a shell command appears to run validation."""
    lowered = " ".join(str(command).lower().split())
    if not lowered:
        return False
    return any(pattern in lowered for pattern in _VALIDATION_COMMAND_PATTERNS)


def classify_structural_phase(step: dict) -> str:
    """Classify one assistant step into a deterministic workflow phase.

    The classifier is intentionally conservative. It maps directly observable
    tool behavior to one of the six workflow phases used by the semantic
    labeler: understand, plan, implement, debug, validate, report.
    """
    if step.get("role") != "assistant":
        return "unknown"

    tool_calls = step.get("tool_calls", [])
    if _has_plan_snapshot(step):
        return "plan"

    has_plan_tool = False
    has_write = False
    has_read_or_search = False
    has_validation = False
    has_failure = False

    for tc in tool_calls:
        tool_name = tc.get("tool_name", "")
        if tool_name in _PLAN_TOOL_NAMES:
            has_plan_tool = True
        if tool_name in _WRITE_TOOL_NAMES:
            has_write = True
        if tool_name in _READ_TOOL_NAMES or tool_name in _SEARCH_TOOL_NAMES:
            has_read_or_search = True
        if _tool_failed(tc):
            has_failure = True

        inp = tc.get("input", {})
        if isinstance(inp, dict):
            command = inp.get("command", "")
            if command and _is_validation_command(command):
                has_validation = True

    if has_plan_tool:
        return "plan"
    if has_write:
        return "implement"
    if has_validation:
        return "validate"
    if has_failure:
        return "debug"
    if has_read_or_search:
        return "understand"
    if step.get("finish") in ("stop", "end_turn") or not tool_calls:
        return "report"
    return "unknown"


def build_structural_phase_segments(steps: list[dict]) -> list[dict]:
    """Group assistant steps into contiguous structural workflow phases."""
    segments: list[dict] = []
    current: dict | None = None

    for step in steps[:_MAX_STEPS]:
        phase = classify_structural_phase(step)
        if phase == "unknown":
            continue

        step_idx = step.get("index", 0)
        if current is None or current["name"] != phase:
            if current is not None:
                segments.append(current)
            current = {
                "name": phase,
                "start_idx": step_idx,
                "end_idx": step_idx,
            }
        else:
            current["end_idx"] = step_idx

    if current is not None:
        segments.append(current)

    return segments


# ---------------------------------------------------------------------------
# 1. Tool Sequence Detection
# ---------------------------------------------------------------------------

def _extract_tool_names(step: dict) -> list[str]:
    """Extract tool-call names from a single step."""
    names: list[str] = []
    for tc in step.get("tool_calls", []):
        name = tc.get("tool_name") or tc.get("name", "")
        if name:
            names.append(name)
    return names


def detect_tool_sequences(
    steps: list[dict],
    min_freq: int = 3,
    max_n: int = 5,
) -> list[dict]:
    """Find recurring tool-call n-grams across the trajectory.

    Parameters
    ----------
    steps : list[dict]
        Parsed trajectory steps, each with a ``tool_calls`` list.
    min_freq : int
        Minimum frequency for an n-gram to be reported.
    max_n : int
        Maximum n-gram length to consider (inclusive).

    Returns
    -------
    list[dict]
        Each entry: ``{"sequence": [str], "frequency": int, "step_indices": [int]}``,
        sorted by frequency descending.
    """
    if not steps:
        return []

    # Build flat list of (tool_name, originating_step_index) pairs
    tool_stream: list[tuple[str, int]] = []
    for step in steps[:_MAX_STEPS]:
        idx = step.get("index", 0)
        for name in _extract_tool_names(step):
            tool_stream.append((name, idx))

    if len(tool_stream) < 2:
        return []

    results: list[dict] = []

    for n in range(2, max_n + 1):
        if n > len(tool_stream):
            break

        # Count n-grams and track their start indices in the stream
        ngram_indices: dict[tuple[str, ...], list[int]] = {}
        for i in range(len(tool_stream) - n + 1):
            gram = tuple(tool_stream[i + k][0] for k in range(n))
            ngram_indices.setdefault(gram, []).append(i)

        for gram, stream_positions in ngram_indices.items():
            # Count NON-OVERLAPPING occurrences so a single contiguous burst of
            # one tool (Read×4 -> three overlapping (Read,Read)) is not inflated
            # into a frequent sequence.
            occ_positions: list[int] = []
            last_end = -1
            for pos in sorted(stream_positions):
                if pos >= last_end:
                    occ_positions.append(pos)
                    last_end = pos + n
            if len(occ_positions) < min_freq:
                continue
            step_indices = sorted({tool_stream[pos][1] for pos in occ_positions})
            results.append({
                "sequence": list(gram),
                "frequency": len(occ_positions),
                "step_indices": step_indices,
            })

    # Sort by frequency descending, then by shorter sequences first for ties
    results.sort(key=lambda r: (-r["frequency"], len(r["sequence"])))
    return results


# ---------------------------------------------------------------------------
# 2. Failure Pattern Detection
# ---------------------------------------------------------------------------

def _step_has_success(step: dict) -> bool:
    """Return True if the step has at least one successful tool call."""
    for tc in step.get("tool_calls", []):
        status = tc.get("status", "")
        if status and status not in ("error", "failed", "failure", "cancelled", "timeout"):
            return True
        # No explicit error status and no bad exit code -> treat as success
        meta = tc.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}
        if status == "" and meta.get("exit", 0) in (None, 0):
            return True
    return False


def detect_failure_patterns(steps: list[dict]) -> list[dict]:
    """Detect recurring failure patterns and their recovery paths.

    Reuses :func:`.diagnostics.cluster_errors` to group errors, then for
    each cluster finds the most common sequence of tool calls between the
    error and the next successful step.

    Returns
    -------
    list[dict]
        Each entry: ``{"cluster_label": str, "count": int,
        "example_error": str, "recovery_path": [str] | None}``.
    """
    if not steps:
        return []

    from .diagnostics import cluster_errors

    clusters = cluster_errors(steps)
    if not clusters:
        return []

    # Build step-index lookup
    step_map: dict[int, dict] = {s.get("index", i): s for i, s in enumerate(steps)}
    all_indices = sorted(step_map.keys())

    results: list[dict] = []

    for cluster in clusters:
        # Collect recovery paths for every error occurrence in this cluster
        recovery_paths: list[tuple[str, ...]] = []

        for err_step_idx in cluster["steps"]:
            # Find steps between the error and the next success
            path_names: list[str] = []
            found_recovery = False

            # Walk forward from the step *after* the error
            pos = _bisect_right(all_indices, err_step_idx)
            for j in range(pos, min(pos + 20, len(all_indices))):
                next_idx = all_indices[j]
                next_step = step_map[next_idx]
                names = _extract_tool_names(next_step)
                path_names.extend(names)
                if _step_has_success(next_step):
                    found_recovery = True
                    break

            if found_recovery and path_names:
                recovery_paths.append(tuple(path_names))

        # Most common recovery path
        recovery_path: list[str] | None = None
        if recovery_paths:
            counter = Counter(recovery_paths)
            most_common = counter.most_common(1)[0][0]
            recovery_path = list(most_common)

        results.append({
            "cluster_label": f"{cluster['tool']}: {cluster['pattern']}",
            "count": cluster["count"],
            "example_error": cluster["pattern"],
            "recovery_path": recovery_path,
            "steps": list(cluster.get("steps", [])),
        })

    return results


_bisect_right = bisect_right


# ---------------------------------------------------------------------------
# 3. Phase Anomaly Detection
# ---------------------------------------------------------------------------

def detect_phase_anomalies(
    steps: list[dict],
    phases: list[dict],
) -> list[dict]:
    """Detect backward phase transitions that may indicate rework or regression.

    Parameters
    ----------
    steps : list[dict]
        Full parsed trajectory steps (used to compute total step count).
    phases : list[dict]
        Phase list with keys ``name``, ``start_idx``, ``end_idx`` (e.g.
        from manual annotation or an external classifier).

    Returns
    -------
    list[dict]
        Each entry: ``{"from_phase": str, "to_phase": str, "step_idx": int,
        "confidence": float, "category": str, "explanation": str}``.
        ``category`` is ``"intentional_iteration"`` or ``"unintentional_drift"``.
    """
    if not phases or not steps:
        return []

    total_steps = len(steps[:_MAX_STEPS])
    if total_steps == 0:
        return []

    anomalies: list[dict] = []

    for i in range(1, len(phases)):
        prev_phase = phases[i - 1]
        curr_phase = phases[i]

        prev_name = prev_phase.get("name", "").lower().strip()
        curr_name = curr_phase.get("name", "").lower().strip()

        prev_rank = _PHASE_RANK.get(prev_name)
        curr_rank = _PHASE_RANK.get(curr_name)

        # Skip phases not in the expected ordering
        if prev_rank is None or curr_rank is None:
            continue

        # Backward transition
        if curr_rank < prev_rank:
            transition_step = curr_phase.get("start_idx", 0)
            regressed_steps = curr_phase.get("end_idx", transition_step) - transition_step + 1
            confidence = round(regressed_steps / total_steps, 4) if total_steps > 0 else 0.0

            # Categorize: check if a planning step precedes the regression
            category = "unintentional_drift"
            window_start = max(0, transition_step - 3)
            for s in steps[window_start:transition_step]:
                for tc in s.get("tool_calls", []):
                    if tc.get("tool_name") in _PLAN_TOOL_NAMES:
                        category = "intentional_iteration"
                        break
                if category == "intentional_iteration":
                    break

            if category == "intentional_iteration":
                explanation = (
                    f"Backward transition from '{prev_name}' to '{curr_name}' "
                    f"at step {transition_step}. "
                    f"A planning step precedes the transition, suggesting "
                    f"intentional iteration (spans {regressed_steps} step(s), "
                    f"{confidence * 100:.1f}% of trajectory)."
                )
            else:
                explanation = (
                    f"Backward transition from '{prev_name}' to '{curr_name}' "
                    f"at step {transition_step}. "
                    f"The regressed phase spans {regressed_steps} step(s) "
                    f"({confidence * 100:.1f}% of trajectory), "
                    f"suggesting rework or unexpected context switch."
                )

            anomalies.append({
                "from_phase": prev_name,
                "to_phase": curr_name,
                "step_idx": transition_step,
                "confidence": confidence,
                "category": category,
                "explanation": explanation,
            })

    return anomalies


# ---------------------------------------------------------------------------
# 4. Plan Progress Analysis (TodoWrite tracking)
# ---------------------------------------------------------------------------

def extract_plan_history(steps: list[dict]) -> list[dict]:
    """Extract plan snapshots from TodoWrite tool calls.

    Returns a list of plan snapshots:
    ``[{"step": int, "items": [{"content": str, "status": str}]}]``
    """
    history: list[dict] = []
    for s in steps[:_MAX_STEPS]:
        for tc in s.get("tool_calls", []):
            name = tc.get("tool_name") or tc.get("name", "")
            # Match TodoWrite (Claude Code), todowrite (OpenCode), TaskCreate, etc.
            if name.lower() not in ("todowrite", "todo_write", "taskcreate", "taskupdate"):
                continue
            inp = tc.get("input", tc.get("arguments", {}))
            if not isinstance(inp, dict):
                continue
            todos = inp.get("todos", [])
            if not todos:
                continue
            items = [
                {"content": t.get("content", ""), "status": t.get("status", "?")}
                for t in todos if isinstance(t, dict)
            ]
            history.append({"step": s.get("index", 0), "items": items})
    return history


def detect_plan_phases(plan_history: list[dict]) -> list[dict]:
    """Group plan snapshots into phases.

    A phase boundary is detected when the item set changes completely
    (no content overlap with the previous snapshot).
    """
    if not plan_history:
        return []

    phases: list[dict] = []
    current_contents: set[str] = set()
    phase_start = 0

    for i, snapshot in enumerate(plan_history):
        new_contents = {item["content"] for item in snapshot["items"]}
        if i > 0 and current_contents and not current_contents & new_contents:
            # Complete content change — new phase
            phases.append({
                "phase_index": len(phases),
                "start_snapshot": phase_start,
                "end_snapshot": i - 1,
                "start_step": plan_history[phase_start]["step"],
                "end_step": plan_history[i - 1]["step"],
                "item_count": len(current_contents),
            })
            phase_start = i
        current_contents = new_contents

    # Final phase
    if plan_history:
        phases.append({
            "phase_index": len(phases),
            "start_snapshot": phase_start,
            "end_snapshot": len(plan_history) - 1,
            "start_step": plan_history[phase_start]["step"],
            "end_step": plan_history[-1]["step"],
            "item_count": len(current_contents),
        })

    return phases


def compute_plan_metrics(plan_history: list[dict]) -> dict:
    """Compute plan-level metrics from plan history.

    Returns dict with: item durations, stalled items, plan reset count.
    """
    if not plan_history:
        return {"items": [], "stalled": [], "plan_resets": 0, "total_items": 0}

    # Track per-item first in_progress and completed steps
    item_first_progress: dict[str, int] = {}
    item_completed: dict[str, int] = {}

    for snapshot in plan_history:
        step = snapshot["step"]
        for item in snapshot["items"]:
            content = item["content"]
            status = item["status"]
            if status == "in_progress" and content not in item_first_progress:
                item_first_progress[content] = step
            if status == "completed" and content not in item_completed:
                item_completed[content] = step

    items = []
    stalled = []
    all_contents: set[str] = set()
    for snapshot in plan_history:
        for item in snapshot["items"]:
            all_contents.add(item["content"])

    for content in all_contents:
        start = item_first_progress.get(content)
        end = item_completed.get(content)
        duration = (end - start) if start is not None and end is not None else None
        entry = {"content": content, "start_step": start, "end_step": end, "duration_steps": duration}
        items.append(entry)
        if start is not None and end is None or duration is not None and duration > 20:
            stalled.append(entry)

    phases = detect_plan_phases(plan_history)
    return {
        "items": items,
        "stalled": stalled,
        "plan_resets": max(0, len(phases) - 1),
        "total_items": len(all_contents),
    }


# ---------------------------------------------------------------------------
# 5. Sub-Agent Delegation Analysis
# ---------------------------------------------------------------------------

def _get_step_info(step: dict, trajectory: list[dict]) -> dict:
    """Retrieve the trajectory 'info' dict corresponding to a parsed step.

    Matches by the step's 'raw_index' (original position in trajectory);
    returns {} if raw_index is missing or out of range.
    """
    raw_idx = step.get("raw_index")
    if raw_idx is not None and 0 <= raw_idx < len(trajectory):
        entry = trajectory[raw_idx]
        info = entry.get("info", {}) if isinstance(entry, dict) else {}
        return info if isinstance(info, dict) else {}
    return {}


def extract_subagent_sessions(
    steps: list[dict], trajectory: list[dict],
) -> list[dict]:
    """Identify sub-agent sessions by grouping consecutive isSubAgent steps.

    Returns list of sessions:
    ``[{"session_id": str, "start_step": int, "end_step": int,
        "step_count": int, "spawn_step": int | None}]``
    """
    sessions: list[dict] = []
    current_session: dict | None = None

    for s in steps[:_MAX_STEPS]:
        info = _get_step_info(s, trajectory)
        # Prefer step fields (may be inferred from Task spawn metadata).
        is_sub = bool(s.get("is_sub_agent")) or bool(info.get("isSubAgent", False))
        session_id = s.get("session_id") or info.get("sessionID", "")
        step_idx = s.get("index", 0)

        if is_sub and session_id:
            if current_session is None or current_session["session_id"] != session_id:
                if current_session is not None:
                    sessions.append(current_session)
                current_session = {
                    "session_id": session_id,
                    "start_step": step_idx,
                    "end_step": step_idx,
                    "step_count": 1,
                    "spawn_step": None,
                }
            else:
                current_session["end_step"] = step_idx
                current_session["step_count"] += 1
        else:
            if current_session is not None:
                sessions.append(current_session)
                current_session = None

    if current_session is not None:
        sessions.append(current_session)

    # OpenCode-style exports record the spawned child session on the parent
    # task tool part's metadata rather than on the child messages.  Build the
    # child-session -> delegation-step map in one pass so consolidated
    # CodeArts/OpenCode exports can identify the actual delegation step.
    spawn_by_child: dict[str, int] = {}
    for step in steps[:_MAX_STEPS]:
        raw_idx = step.get("raw_index")
        if not isinstance(raw_idx, int) or not (0 <= raw_idx < len(trajectory)):
            continue
        entry = trajectory[raw_idx]
        parts = entry.get("parts", []) if isinstance(entry, dict) else []
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict) or part.get("type") != "tool":
                continue
            state = part.get("state", {})
            metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
            child_id = spawned_child_session_id(
                metadata,
                caller_session_id=str(step.get("session_id") or ""),
            )
            if child_id:
                spawn_by_child.setdefault(child_id, step.get("index", 0))

    for session in sessions:
        if session["spawn_step"] is None:
            session["spawn_step"] = spawn_by_child.get(session["session_id"])

    return sessions


def compute_subagent_metrics(
    sessions: list[dict], steps: list[dict],
) -> list[dict]:
    """Compute per-session metrics: tokens, tools, duration."""
    results = []
    for session in sessions:
        start = session["start_step"]
        end = session["end_step"]
        session_steps = [s for s in steps if start <= s.get("index", 0) <= end]

        total_tokens = sum(s["tokens"]["total"] for s in session_steps)
        total_tools = sum(s["tool_call_count"] for s in session_steps)
        durations = [s["duration"] for s in session_steps if s.get("duration") is not None]
        total_duration = sum(durations)

        results.append({
            **session,
            "total_tokens": total_tokens,
            "total_tools": total_tools,
            "total_duration": round(total_duration, 2),
        })
    return results


# ---------------------------------------------------------------------------
# 6. Fruitless Search & Anti-Pattern Detection
# ---------------------------------------------------------------------------

_SEARCH_BASH_PREFIXES = ("grep", "rg", "ag", "find", "locate", "fgrep", "egrep", "ripgrep")
_SHELL_PUNCTUATION = ";&|()\n"
_WRAPPER_OPTIONS_WITH_VALUES = {
    "env": {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"},
    "git": {
        "-C", "-c", "--git-dir", "--work-tree", "--namespace",
        "--super-prefix", "--config-env",
    },
    "sudo": {
        "-u", "--user", "-g", "--group", "-h", "--host", "-p", "--prompt",
        "-C", "--close-from", "-r", "--role", "-t", "--type", "-T",
        "--command-timeout", "-D", "--chdir", "-R", "--chroot", "-U",
        "--other-user",
    },
    "time": {"-f", "--format", "-o", "--output"},
    "nice": {"-n", "--adjustment"},
    "timeout": {"-k", "--kill-after", "-s", "--signal"},
    "xargs": {
        "-a", "--arg-file", "-E", "--eof", "-I", "--replace", "-L",
        "--max-lines", "-n", "--max-args", "-P", "--max-procs", "-s",
        "--max-chars",
    },
}


def _shell_segments(command: str) -> list[list[str]]:
    """Lex shell command segments without evaluating or executing anything."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=_SHELL_PUNCTUATION)
    # Preserve newlines as command boundaries while still honoring quoted ones.
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        # Be conservative for malformed/unclosed quoting.
        return []

    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token and all(char in _SHELL_PUNCTUATION for char in token):
            if current:
                segments.append(current)
                current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def _shell_name(token: str) -> str:
    """Return a case-insensitive executable basename."""
    return token.rsplit("/", 1)[-1].lower()


def _is_shell_assignment(token: str) -> bool:
    """Return True for a simple POSIX-style NAME=value assignment."""
    name, separator, _ = token.partition("=")
    return bool(
        separator
        and name
        and (name[0].isalpha() or name[0] == "_")
        and all(char.isalnum() or char == "_" for char in name)
    )


def _skip_wrapper_options(tokens: list[str], index: int, wrapper: str) -> int:
    """Skip known wrapper flags, including flags whose value is separate."""
    value_options = _WRAPPER_OPTIONS_WITH_VALUES.get(wrapper, set())
    while index < len(tokens):
        option = tokens[index]
        if option == "--":
            return index + 1
        if not option.startswith("-") or option == "-":
            return index
        index += 1
        if option in value_options and index < len(tokens):
            index += 1
    return index


def _segment_runs_search(tokens: list[str], nesting: int = 0) -> bool:
    """Recognize a search executable at the head of one shell segment."""
    index = 0
    # Bound wrapper traversal even for adversarially repetitive input.
    for _ in range(12):
        while index < len(tokens) and _is_shell_assignment(tokens[index]):
            index += 1
        if index >= len(tokens):
            return False

        command = _shell_name(tokens[index])
        if command in _SEARCH_BASH_PREFIXES:
            return True

        if command == "git":
            index = _skip_wrapper_options(tokens, index + 1, "git")
            return index < len(tokens) and _shell_name(tokens[index]) == "grep"

        if command == "command":
            index += 1
            if index < len(tokens) and tokens[index] in ("-v", "-V"):
                return False
            index = _skip_wrapper_options(tokens, index, "command")
            continue

        if command in ("env", "sudo", "time", "nice", "nohup", "xargs"):
            index = _skip_wrapper_options(tokens, index + 1, command)
            continue

        if command == "timeout":
            index = _skip_wrapper_options(tokens, index + 1, command)
            # timeout's first positional argument is the duration.
            index += 1
            continue

        if command == "busybox":
            index += 1
            continue

        if command in ("bash", "dash", "ksh", "sh", "zsh") and nesting < 3:
            # A quoted ``sh -c`` script is data to this process.  Lex it again
            # with the same non-executing tokenizer instead of invoking a shell.
            option_index = index + 1
            while option_index < len(tokens) and tokens[option_index].startswith("-"):
                flags = tokens[option_index].lstrip("-")
                if "c" in flags and option_index + 1 < len(tokens):
                    return any(
                        _segment_runs_search(segment, nesting + 1)
                        for segment in _shell_segments(tokens[option_index + 1])
                    )
                option_index += 1
            return False

        if command in ("if", "then", "elif", "while", "until", "do", "!"):
            index += 1
            continue

        return False
    return False


def _is_search_call(tc: dict) -> bool:
    """A tool call counts as a search only if it actually searches.

    Bash is in ``_SEARCH_TOOL_NAMES`` because agents run grep/rg/find through
    it, but ordinary shell commands (mkdir, cp, chmod, git add) that legitimately
    emit no output must not be mistaken for empty searches.
    """
    name = tc.get("tool_name")
    if name not in _SEARCH_TOOL_NAMES:
        return False
    if name in ("Bash", "bash"):
        inp = tc.get("input", {})
        cmd = inp.get("command", "") if isinstance(inp, dict) else ""
        if not isinstance(cmd, str) or not cmd.strip():
            return False
        return any(_segment_runs_search(segment) for segment in _shell_segments(cmd))
    return True


def _is_fruitless_step(step: dict) -> bool:
    """Check if a step's search/grep tool calls all returned empty results.

    Supports two detection methods:
    1. tool_call output/result is empty or contains no matches
    2. tool_call status indicates no results
    """
    tool_calls = step.get("tool_calls", [])
    if not tool_calls:
        return False

    search_calls = [tc for tc in tool_calls if _is_search_call(tc)]
    if not search_calls:
        return False

    # Method 1: Check individual tool call outputs/results
    for tc in search_calls:
        output = tc.get("output", tc.get("result", ""))
        if isinstance(output, str):
            stripped = output.strip()
            if stripped and stripped not in ("", "No matches found", "No results", "No files found"):
                return False
        elif isinstance(output, dict):
            content = output.get("content", output.get("text", ""))
            if isinstance(content, str) and content.strip():
                return False
            # Content-block lists (e.g. [{"type": "text", "text": ...}]) are
            # non-empty results too — mirror the bare-list handling below.
            if isinstance(content, list) and content:
                return False
        elif isinstance(output, list) and output:
            return False

    return True


def detect_fruitless_streaks(steps: list[dict]) -> list[dict]:
    """Detect consecutive steps with empty search/grep tool output.

    Returns list of streaks:
    ``[{"start_step": int, "end_step": int, "length": int, "tools": [str]}]``
    """
    streaks: list[dict] = []
    current_streak: dict | None = None

    for s in steps[:_MAX_STEPS]:
        if not s.get("tool_calls"):
            if current_streak is not None and current_streak["length"] >= 3:
                streaks.append(current_streak)
            current_streak = None
            continue

        if _is_fruitless_step(s):
            tool_names = [tc.get("tool_name", "?") for tc in s["tool_calls"]]
            step_idx = s.get("index", 0)
            if current_streak is None:
                current_streak = {
                    "start_step": step_idx,
                    "end_step": step_idx,
                    "length": 1,
                    "tools": tool_names,
                }
            else:
                current_streak["end_step"] = step_idx
                current_streak["length"] += 1
                current_streak["tools"].extend(tool_names)
        else:
            if current_streak is not None and current_streak["length"] >= 3:
                streaks.append(current_streak)
            current_streak = None

    if current_streak is not None and current_streak["length"] >= 3:
        streaks.append(current_streak)

    return streaks


def compute_autonomy_ratio(steps: list[dict]) -> float:
    """Compute autonomy ratio from trigger fields.

    Returns ratio of autonomous steps to total assistant steps (0.0 to 1.0).
    Falls back to 1 - (user_steps / total_steps) if trigger field is absent.
    """
    assistant_steps = [s for s in steps if s.get("role") == "assistant"]
    if not assistant_steps:
        return 0.0

    # Autonomy = share of turns not directly driven by the user.
    user_steps = sum(1 for s in steps if s.get("role") == "user")
    total = len(steps)
    return round(1.0 - (user_steps / total), 4) if total > 0 else 0.0


def detect_tool_selection_antipatterns(steps: list[dict]) -> list[dict]:
    """Detect Bash commands used for file reading when Read tool exists.

    Returns list of flagged steps.
    """
    import re
    # Only flag commands where sed/cat/head IS the primary command reading a file,
    # not when used in pipes (e.g., "grep ... | head -20" is legitimate).
    _BASH_READ_PATTERNS = [
        re.compile(r"^sed\s+-n\s+'"),          # sed as primary command
        re.compile(r"^cat\s+[\"']?[/\\a-zA-Z]"),  # cat as primary command
        re.compile(r"^head\s+-[n0-9]"),        # head as primary command
        re.compile(r"^tail\s+-[n0-9]"),        # tail as primary command
    ]

    flagged: list[dict] = []
    for s in steps[:_MAX_STEPS]:
        for tc in s.get("tool_calls", []):
            # OpenCode emits lowercase tool names — match both spellings,
            # like _is_search_call above.
            if tc.get("tool_name") not in ("Bash", "bash"):
                continue
            inp = tc.get("input", {})
            cmd = inp.get("command", "") if isinstance(inp, dict) else ""
            for pattern in _BASH_READ_PATTERNS:
                if pattern.search(cmd):
                    flagged.append({
                        "step": s.get("index", 0),
                        "command": cmd[:80],
                        "pattern": pattern.pattern,
                    })
                    break
    return flagged


# ---------------------------------------------------------------------------
# 7. Semantic Anti-Pattern Detection (requires step labels)
# ---------------------------------------------------------------------------

_SEMANTIC_PHASE_ORDER = _PHASE_ORDER  # reuse: understand, plan, implement, debug, validate, report


def load_step_labels(labeled_json_path: str) -> dict[int, dict[str, str]]:
    """Load a ``_labeled.json`` file into a step-index → {phase, action} mapping.

    Gracefully returns an empty dict on failure.
    """
    import json
    try:
        with open(labeled_json_path, encoding="utf-8") as f:
            data = json.load(f)
        return {
            s["index"]: {"phase": s.get("phase", "unknown"), "action": s.get("action", "unknown")}
            for s in data.get("steps", [])
            if isinstance(s, dict) and "index" in s
        }
    except Exception:
        return {}


def detect_semantic_antipatterns(
    steps: list[dict],
    step_labels: dict[int, dict[str, str]],
) -> dict[str, list[dict]]:
    """Detect anti-patterns from LLM-derived phase/action label sequences.

    Requires ``step_labels`` from :func:`load_step_labels` (or the step labeler).
    Returns a dict mapping anti-pattern type to a list of instances.
    Returns empty collections when labels are unavailable.

    Six anti-pattern types:
    1. phase_oscillation — repeated back-and-forth between two phases
    2. premature_implementation — implement before any plan phase
    3. semantic_fruitless_exploration — understand/code_reading streak with no overlap
       to subsequent implementation targets
    4. validation_avoidance — high implement:validate ratio
    5. debug_without_hypothesis — repeated debug_reproduction with no root-cause step
    6. semantic_plan_stall — extended plan phase with no implement steps
    """
    if not step_labels:
        return {
            "phase_oscillation": [],
            "premature_implementation": [],
            "semantic_fruitless_exploration": [],
            "validation_avoidance": [],
            "debug_without_hypothesis": [],
            "semantic_plan_stall": [],
        }

    # Build ordered label sequence for assistant steps
    labeled_seq: list[tuple[int, str, str]] = []  # (step_index, phase, action)
    for s in steps[:_MAX_STEPS]:
        if s.get("role") != "assistant":
            continue
        idx = s.get("index", 0)
        lbl = step_labels.get(idx, {})
        phase = lbl.get("phase", "unknown")
        action = lbl.get("action", "unknown")
        if phase != "unknown":
            labeled_seq.append((idx, phase, action))

    if not labeled_seq:
        return {k: [] for k in (
            "phase_oscillation", "premature_implementation",
            "semantic_fruitless_exploration", "validation_avoidance",
            "debug_without_hypothesis", "semantic_plan_stall",
        )}

    results: dict[str, list[dict]] = {
        "phase_oscillation": [],
        "premature_implementation": [],
        "semantic_fruitless_exploration": [],
        "validation_avoidance": [],
        "debug_without_hypothesis": [],
        "semantic_plan_stall": [],
    }

    # --- 1. Phase oscillation ---
    # Detect ≥3 transitions between the same two phases within a sliding window
    phase_seq = [(idx, phase) for idx, phase, _ in labeled_seq]
    if len(phase_seq) >= 4:
        for i in range(len(phase_seq) - 3):
            window = phase_seq[i:i + 6]
            phases_in_window = [p for _, p in window]
            unique_phases = set(phases_in_window)
            if len(unique_phases) == 2:
                transitions = sum(
                    1 for j in range(len(phases_in_window) - 1)
                    if phases_in_window[j] != phases_in_window[j + 1]
                )
                if transitions >= 3:
                    p1, p2 = sorted(unique_phases)
                    # Means "at least one oscillating phase is a known workflow
                    # phase" — not a direction check (the two phases here are
                    # always distinct, so ranked phases always differ).
                    is_regression = p1 in _PHASE_RANK or p2 in _PHASE_RANK
                    results["phase_oscillation"].append({
                        "phases": [p1, p2],
                        "step_range": [window[0][0], window[-1][0]],
                        "transitions": transitions,
                        "involves_regression": is_regression,
                    })
                    break  # report first occurrence only

    # --- 2. Premature implementation ---
    first_plan_idx = None
    first_impl_idx = None
    for idx, phase, _ in labeled_seq:
        if phase == "plan" and first_plan_idx is None:
            first_plan_idx = idx
        if phase == "implement" and first_impl_idx is None:
            first_impl_idx = idx

    if first_impl_idx is not None:
        if first_plan_idx is None or first_impl_idx < first_plan_idx:
            # Check if followed by debug within 5 steps
            impl_pos = next(
                (i for i, (idx, _, _) in enumerate(labeled_seq) if idx == first_impl_idx),
                None,
            )
            followed_by_debug = False
            if impl_pos is not None:
                for _, phase, _ in labeled_seq[impl_pos + 1:impl_pos + 6]:
                    if phase == "debug":
                        followed_by_debug = True
                        break
            results["premature_implementation"].append({
                "first_implement_step": first_impl_idx,
                "first_plan_step": first_plan_idx,
                "followed_by_debug": followed_by_debug,
            })

    # --- 3. Semantic fruitless exploration ---
    # Consecutive understand/code_reading or understand/file_discovery steps
    # where the read targets don't appear in subsequent implement steps
    impl_targets: set[str] = set()
    for s in steps[:_MAX_STEPS]:
        idx = s.get("index", 0)
        lbl = step_labels.get(idx, {})
        if lbl.get("phase") == "implement":
            for tc in s.get("tool_calls", []):
                inp = tc.get("input", {})
                if isinstance(inp, dict):
                    fp = inp.get("file_path", "") or inp.get("filePath", "")
                    if fp:
                        impl_targets.add(fp.split("/")[-1])

    current_reading_run: list[tuple[int, str]] = []
    for s in steps[:_MAX_STEPS]:
        idx = s.get("index", 0)
        lbl = step_labels.get(idx, {})
        if lbl.get("action") in ("code_reading", "file_discovery"):
            targets = []
            for tc in s.get("tool_calls", []):
                inp = tc.get("input", {})
                if isinstance(inp, dict):
                    fp = inp.get("file_path", "") or inp.get("filePath", "") or inp.get("path", "")
                    if fp:
                        targets.append(fp.split("/")[-1])
            for t in targets:
                current_reading_run.append((idx, t))
        else:
            if len(current_reading_run) >= 5:
                unused = [(idx, t) for idx, t in current_reading_run if t not in impl_targets]
                if len(unused) >= 4:
                    results["semantic_fruitless_exploration"].append({
                        "step_range": [current_reading_run[0][0], current_reading_run[-1][0]],
                        "files_read": len(current_reading_run),
                        "files_unused": len(unused),
                    })
            current_reading_run = []

    if len(current_reading_run) >= 5:
        unused = [(idx, t) for idx, t in current_reading_run if t not in impl_targets]
        if len(unused) >= 4:
            results["semantic_fruitless_exploration"].append({
                "step_range": [current_reading_run[0][0], current_reading_run[-1][0]],
                "files_read": len(current_reading_run),
                "files_unused": len(unused),
            })

    # --- 4. Validation avoidance ---
    impl_count = sum(1 for _, p, _ in labeled_seq if p == "implement")
    validate_count = sum(1 for _, p, _ in labeled_seq if p == "validate")
    if impl_count >= 5 and validate_count == 0:
        results["validation_avoidance"].append({
            "implement_steps": impl_count,
            "validate_steps": 0,
            "ratio": None,
        })
    elif impl_count >= 8 and validate_count > 0 and impl_count / validate_count > 5:
        results["validation_avoidance"].append({
            "implement_steps": impl_count,
            "validate_steps": validate_count,
            "ratio": round(impl_count / validate_count, 1),
        })

    # --- 5. Debug without hypothesis ---
    # ≥3 consecutive debug_reproduction without debug_root_cause or debug_hypothesis_test
    repro_run = 0
    repro_start = None
    for idx, phase, action in labeled_seq:
        if action == "debug_reproduction":
            if repro_run == 0:
                repro_start = idx
            repro_run += 1
        elif action in ("debug_root_cause", "debug_hypothesis_test"):
            repro_run = 0
            repro_start = None
        elif phase != "debug":
            if repro_run >= 3:
                results["debug_without_hypothesis"].append({
                    "step_range": [repro_start, idx],
                    "reproduction_count": repro_run,
                })
            repro_run = 0
            repro_start = None

    if repro_run >= 3:
        results["debug_without_hypothesis"].append({
            "step_range": [repro_start, labeled_seq[-1][0]],
            "reproduction_count": repro_run,
        })

    # --- 6. Semantic plan stall ---
    # ≥5 consecutive plan-phase steps with no implement step between them
    plan_run = 0
    plan_start = None
    for idx, phase, _ in labeled_seq:
        if phase == "plan":
            if plan_run == 0:
                plan_start = idx
            plan_run += 1
        elif phase == "implement":
            if plan_run >= 5:
                results["semantic_plan_stall"].append({
                    "step_range": [plan_start, idx],
                    "plan_steps": plan_run,
                })
            plan_run = 0
            plan_start = None
        else:
            pass  # non-plan, non-implement phases don't break the stall

    if plan_run >= 5:
        results["semantic_plan_stall"].append({
            "step_range": [plan_start, labeled_seq[-1][0]],
            "plan_steps": plan_run,
        })

    return results
