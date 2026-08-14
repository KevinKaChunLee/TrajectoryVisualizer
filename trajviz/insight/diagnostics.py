"""Trajectory diagnostics: file interaction, failure chains, root-cause attribution, bottleneck explanation."""

from __future__ import annotations

from trajviz.tool_vocab import (WRITE_TOOL_NAMES as _WRITE_TOOL_SET,
                                write_target_path as _write_target_path)

import os
import re

from .metrics import effective_agent
from .parser import infer_non_cache_input

# Tool-call statuses that open/continue a failure chain. Shared by
# _step_has_error, classify_chain_steps, and cluster_errors so the chain
# detector and the chain classifier can never drift apart again.
_ERROR_STATUSES = ("error", "failed", "failure", "cancelled", "timeout")


# ---------------------------------------------------------------------------
# 1. File Interaction Analysis
# ---------------------------------------------------------------------------

# Tools whose input dict has a structured file path field
_TOOL_FILE_FIELDS: dict[str, tuple[tuple[str, ...], str]] = {
    # tool_name -> (candidate_input_keys, interaction_type)
    # Multiple keys per tool let us handle scaffolds that use different field
    # spellings — e.g. Claude Code emits ``file_path`` (snake) while OpenCode
    # emits ``filePath`` (camel). The lookup picks whichever the call carries.
    "Read":         (("file_path", "filePath"), "read"),
    "read":         (("file_path", "filePath"), "read"),
    "Write":        (("file_path", "filePath"), "write"),
    "write":        (("file_path", "filePath"), "write"),
    "Edit":         (("file_path", "filePath"), "write"),
    "edit":         (("file_path", "filePath"), "write"),
    "NotebookEdit": (("notebook_path", "notebookPath"), "write"),
}

# Tools that search (return multiple file references in pattern/path fields)
_TOOL_SEARCH_FIELDS: dict[str, tuple[str, str]] = {
    "Glob": ("pattern", "search"),
    "glob": ("pattern", "search"),
    "Grep": ("pattern", "search"),
    "grep": ("pattern", "search"),
}

# Regex for path-like tokens in bash commands
# Matches: /unix/path, ./relative, ../parent, d:/windows/path
# Handles both unquoted and quoted paths ("d:/foo", 'd:/foo')
_PATH_RE = re.compile(
    r"""(?:^|[\s=;|&(])"""                       # preceded by whitespace or delimiter
    r"""(?:"""
    r"""["']((?:/|\.\.?/|[a-zA-Z]:[/\\])[^"']*?)["']"""   # quoted path
    r"""|"""
    r"""((?:/|\.\.?/|[a-zA-Z]:[/\\])[^\s;|&)'"]+)"""      # unquoted path
    r""")"""
)


def _extract_bash_paths(command: str) -> list[str]:
    """Extract file-path-like tokens from a bash command string."""
    paths = []
    for m in _PATH_RE.finditer(command):
        p = m.group(1) or m.group(2)  # group 1 = quoted, group 2 = unquoted
        if not p:
            continue
        p = p.rstrip(",:")
        # Skip very short or unlikely paths
        if len(p) < 2 or p.rstrip("/\\") in ("", "/", "./", ".."):
            continue
        # Must contain at least one path component with alphanumeric chars
        parts = [seg for seg in p.replace("\\", "/").split("/") if seg]
        if parts and any(c.isalnum() for c in parts[-1]):
            paths.append(p)
    return paths


def extract_file_interactions(steps: list[dict]) -> list[dict]:
    """Extract file path references from every tool call in parsed steps.

    Returns a list of records: {step, tool, path, type, tokens}
    where type is one of: read, write, search.
    """
    interactions: list[dict] = []

    for step in steps:
        step_idx = step["index"]
        step_tokens = step["tokens"]["total"]

        for tc in step.get("tool_calls", []):
            tool_name = tc.get("tool_name", "")
            inp = tc.get("input", {})
            if not isinstance(inp, dict):
                inp = {}

            found_paths: list[tuple[str, str]] = []  # (path, interaction_type)

            # Structured file tools — try every candidate field name so the
            # extractor works across schema spellings (snake_case / camelCase).
            if tool_name in _TOOL_FILE_FIELDS:
                fields, itype = _TOOL_FILE_FIELDS[tool_name]
                for field in fields:
                    path = inp.get(field, "")
                    if path:
                        found_paths.append((str(path), itype))
                        break

            # Search tools — extract path (directory) and pattern (for Glob only)
            elif tool_name in _TOOL_SEARCH_FIELDS:
                _, itype = _TOOL_SEARCH_FIELDS[tool_name]
                path = inp.get("path", "")
                if path:
                    found_paths.append((str(path), itype))
                # Glob patterns are file paths; Grep patterns are text regexes
                if tool_name in ("Glob", "glob"):
                    pattern = inp.get("pattern", "")
                    if pattern:
                        found_paths.append((str(pattern), itype))

            # Bash — heuristic path extraction
            elif tool_name in ("Bash", "bash", "BashCommand"):
                command = inp.get("command", "")
                if command:
                    for p in _extract_bash_paths(command):
                        # Classify: if command looks like a write operation
                        write_cmds = ("mv ", "cp ", "rm ", "mkdir ", "touch ",
                                      "> ", ">> ", "tee ")
                        itype = "write" if any(command.strip().startswith(c) or f" {c}" in command for c in write_cmds) else "read"
                        found_paths.append((p, itype))

            for path, itype in found_paths:
                # Normalize backslashes to forward slashes for consistency
                path = path.replace("\\", "/")
                interactions.append({
                    "step": step_idx,
                    "tool": tool_name,
                    "path": path,
                    "type": itype,
                    "tokens": step_tokens,
                })

    return interactions


def identify_target_files(steps: list[dict]) -> set[str]:
    """Identify target files: files in patch parts + successful Edit/Write calls."""
    targets: set[str] = set()

    for step in steps:
        for part in step.get("parts", []):
            if part.get("type") == "patch":
                for f in part.get("files", []):
                    if f:
                        targets.add(str(f).replace("\\", "/"))

        for tc in step.get("tool_calls", []):
            tool_name = tc.get("tool_name", "")
            status = tc.get("status", "")
            if tool_name in _WRITE_TOOL_SET and status not in ("error", "failed", "failure"):
                inp = tc.get("input", {})
                if isinstance(inp, dict):
                    # OpenCode uses ``filePath``; Claude Code uses ``file_path``.
                    path = _write_target_path(inp)
                    if path:
                        # Match extract_file_interactions, which normalizes
                        # backslashes; _paths_match/normpath won't on POSIX.
                        targets.add(str(path).replace("\\", "/"))

    return targets


def _normalize_path(path: str) -> str:
    """Normalize a path for comparison (resolve . and .., strip trailing /)."""
    return os.path.normpath(path) if path else path


def _paths_match(a: str, b: str) -> bool:
    """Check if two paths refer to the same file (suffix match for relative vs absolute)."""
    na, nb = _normalize_path(a), _normalize_path(b)
    if na == nb:
        return True
    # One might be absolute, the other relative — check suffix match
    return na.endswith("/" + nb) or nb.endswith("/" + na)


def compute_file_targeting_metrics(
    interactions: list[dict],
    target_files: set[str],
    total_steps: int,
) -> dict:
    """Compute file-targeting efficiency metrics.

    Returns dict with:
    - steps_to_first_touch: dict[file, {absolute, relative}]
    - avg_steps_to_first_touch: float
    - exploration_ratio: float
    - per_file_token_cost: dict[file, int]
    """
    if not interactions or not target_files or total_steps == 0:
        return {
            "steps_to_first_touch": {},
            "avg_steps_to_first_touch": None,
            "exploration_ratio": None,
            "per_file_token_cost": {},
        }

    # First touch per target file
    first_touch: dict[str, int] = {}
    for target in target_files:
        for inter in interactions:
            if _paths_match(inter["path"], target):
                if target not in first_touch or inter["step"] < first_touch[target]:
                    first_touch[target] = inter["step"]

    steps_to_first_touch = {}
    for f, step_idx in first_touch.items():
        steps_to_first_touch[f] = {
            "absolute": step_idx,
            "relative": round(step_idx / total_steps, 4) if total_steps > 0 else 0,
        }

    avg_sft = None
    if first_touch:
        avg_sft = round(sum(first_touch.values()) / len(first_touch), 2)

    # Exploration ratio: unique files touched / target file count
    all_files = {_normalize_path(i["path"]) for i in interactions if i["type"] in ("read", "search")}
    exploration_ratio = round(len(all_files) / len(target_files), 2) if target_files else None

    # Per-file token cost: divide step tokens equally among files in that step
    file_token_cost: dict[str, float] = {}
    # Group interactions by step
    step_files: dict[int, list[str]] = {}
    step_tokens: dict[int, int] = {}
    for inter in interactions:
        sid = inter["step"]
        step_files.setdefault(sid, []).append(inter["path"])
        step_tokens[sid] = inter["tokens"]

    for sid, files in step_files.items():
        tok = step_tokens.get(sid, 0)
        if files and tok > 0:
            share = tok / len(files)
            for f in files:
                nf = _normalize_path(f)
                file_token_cost[nf] = file_token_cost.get(nf, 0) + share

    per_file_token_cost = {f: round(v) for f, v in sorted(file_token_cost.items(), key=lambda x: -x[1])}

    return {
        "steps_to_first_touch": steps_to_first_touch,
        "avg_steps_to_first_touch": avg_sft,
        "exploration_ratio": exploration_ratio,
        "per_file_token_cost": per_file_token_cost,
    }


# ---------------------------------------------------------------------------
# 2. Failure Chain Analysis
# ---------------------------------------------------------------------------

def _step_has_error(step: dict) -> bool:
    """Check if a step has at least one error tool call or non-zero exit code."""
    if step.get("error_count", 0) > 0:
        return True
    for tc in step.get("tool_calls", []):
        status = tc.get("status", "")
        if status in _ERROR_STATUSES:
            return True
        meta = tc.get("metadata", {})
        if isinstance(meta, dict) and meta.get("exit") not in (None, 0):
            return True
    return False


def _error_tool_target(tc: dict) -> tuple[str, str]:
    """Extract (tool_name, primary_target) from a tool call for comparison."""
    tool_name = tc.get("tool_name", "")
    inp = tc.get("input", {})
    if not isinstance(inp, dict):
        return (tool_name, "")
    for k in ("file_path", "command", "pattern", "path", "query"):
        if inp.get(k):
            return (tool_name, str(inp[k])[:80])
    return (tool_name, "")


def detect_failure_chains(steps: list[dict]) -> list[dict]:
    """Find maximal sequences of consecutive assistant steps with errors.

    Returns list of chain dicts: {start, end, steps: [indices]}
    """
    chains: list[dict] = []
    current_chain: list[int] = []

    for step in steps:
        if step.get("role") != "assistant":
            continue  # user steps don't break chains
        if _step_has_error(step):
            current_chain.append(step["index"])
        else:
            if current_chain:
                chains.append({
                    "start": current_chain[0],
                    "end": current_chain[-1],
                    "steps": list(current_chain),
                })
                current_chain = []

    if current_chain:
        chains.append({
            "start": current_chain[0],
            "end": current_chain[-1],
            "steps": list(current_chain),
        })

    return chains


def classify_chain_steps(chain: dict, steps: list[dict]) -> list[dict]:
    """Classify each step in a failure chain as first_error, recovery_attempt, or cascade.

    Returns list of {step_idx, classification} dicts.
    """
    step_map = {s["index"]: s for s in steps}
    chain_steps = chain["steps"]

    if not chain_steps:
        return []

    # Get first error's tool+target signature
    first_step = step_map.get(chain_steps[0], {})
    first_error_sigs = set()
    for tc in first_step.get("tool_calls", []):
        # Same status set as _step_has_error: a chain opened by a cancelled or
        # timed-out call must still yield first-error signatures, or identical
        # retries would all be classified "cascade".
        if tc.get("status") in _ERROR_STATUSES or (
            isinstance(tc.get("metadata"), dict) and tc["metadata"].get("exit") not in (None, 0)
        ):
            first_error_sigs.add(_error_tool_target(tc))

    result = [{"step_idx": chain_steps[0], "classification": "first_error"}]

    for idx in chain_steps[1:]:
        step = step_map.get(idx, {})
        step_sigs = set()
        for tc in step.get("tool_calls", []):
            step_sigs.add(_error_tool_target(tc))

        # Recovery if same tool+target as first error
        if step_sigs & first_error_sigs:
            result.append({"step_idx": idx, "classification": "recovery_attempt"})
        else:
            result.append({"step_idx": idx, "classification": "cascade"})

    return result


def link_chains_to_agents(
    chains: list[dict],
    steps: list[dict],
    agent_summaries: list[dict],
) -> list[dict]:
    """Annotate failure chains with parent agent spawning info.

    Returns chains with added spawning_agent and spawning_step fields where applicable.
    """
    if not agent_summaries:
        return chains

    from .metrics import effective_agent

    # Build agent_id -> spawned_by_step lookup
    spawn_map: dict[str, int] = {}
    for a in agent_summaries:
        if a.get("spawned_by_step") is not None:
            spawn_map[a["agent_id"]] = a["spawned_by_step"]

    step_map = {s["index"]: s for s in steps}

    for chain in chains:
        first_step = step_map.get(chain["start"], {})
        agent_id = effective_agent(first_step)
        if agent_id and agent_id in spawn_map:
            chain["spawning_step"] = spawn_map[agent_id]
            chain["spawning_agent"] = "main"

    return chains


def compute_failure_chain_metrics(chains: list[dict], total_assistant_steps: int) -> dict:
    """Aggregate failure chain metrics."""
    if not chains:
        return {
            "total_chains": 0,
            "total_chain_steps": 0,
            "longest_chain": 0,
            "chain_step_pct": 0.0,
        }

    total_chain_steps = sum(len(c["steps"]) for c in chains)
    longest = max(len(c["steps"]) for c in chains)

    return {
        "total_chains": len(chains),
        "total_chain_steps": total_chain_steps,
        "longest_chain": longest,
        "chain_step_pct": round(total_chain_steps / total_assistant_steps * 100, 1) if total_assistant_steps > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# 3. Root-Cause Attribution
# ---------------------------------------------------------------------------

def _error_pattern(tc: dict) -> str:
    """Extract a short error pattern from a tool call."""
    # Check explicit error field
    err = tc.get("error") or ""
    if err:
        return str(err).strip().split("\n")[0][:120]

    # Check exit code
    meta = tc.get("metadata", {})
    if isinstance(meta, dict) and meta.get("exit") not in (None, 0):
        return f"exit code {meta['exit']}"

    # Check status
    status = tc.get("status", "")
    if status in ("error", "failed", "failure"):
        output = tc.get("output", "")
        if output:
            return str(output).strip().split("\n")[0][:120]
        return f"status: {status}"

    return "unknown error"


def cluster_errors(steps: list[dict]) -> list[dict]:
    """Group error tool calls by (tool_name, error_pattern).

    Returns sorted list of cluster dicts:
    {tool, pattern, count, steps: [step_indices], first_step, last_step}
    """
    clusters: dict[tuple[str, str], dict] = {}

    for step in steps:
        for tc in step.get("tool_calls", []):
            is_error = (
                tc.get("status") in _ERROR_STATUSES
                or (isinstance(tc.get("metadata"), dict) and tc["metadata"].get("exit") not in (None, 0))
            )
            if not is_error:
                continue

            tool = tc.get("tool_name", "unknown")
            pattern = _error_pattern(tc)
            key = (tool, pattern)

            if key not in clusters:
                clusters[key] = {
                    "tool": tool,
                    "pattern": pattern,
                    "count": 0,
                    "steps": [],
                    "first_step": step["index"],
                    "last_step": step["index"],
                }
            c = clusters[key]
            c["count"] += 1
            if step["index"] not in c["steps"]:
                c["steps"].append(step["index"])
            c["last_step"] = max(c["last_step"], step["index"])

    return sorted(clusters.values(), key=lambda c: (-c["count"], c["first_step"]))


def annotate_clusters_with_agents(
    clusters: list[dict],
    steps: list[dict],
    agent_summaries: list[dict],
) -> list[dict]:
    """Annotate error clusters with parent agent info when errors occur in sub-agents."""
    if not agent_summaries:
        return clusters

    from .metrics import effective_agent

    spawn_map: dict[str, int] = {}
    for a in agent_summaries:
        if a.get("spawned_by_step") is not None:
            spawn_map[a["agent_id"]] = a["spawned_by_step"]

    step_map = {s["index"]: s for s in steps}

    for cluster in clusters:
        first_step = step_map.get(cluster["first_step"], {})
        agent_id = effective_agent(first_step)
        if agent_id and agent_id in spawn_map:
            cluster["parent_agent"] = "main"
            cluster["parent_step"] = spawn_map[agent_id]

    return clusters


def format_root_cause_summary(clusters: list[dict]) -> list[str]:
    """Generate human-readable summary text per root-cause cluster."""
    summaries: list[str] = []
    for c in clusters:
        steps_range = f"steps {c['first_step']}..{c['last_step']}" if c["first_step"] != c["last_step"] else f"step {c['first_step']}"
        text = f"{c['count']}x {c['tool']} failures: '{c['pattern']}' ({steps_range})"
        if c.get("parent_agent"):
            text += f" \u2014 traced to {c['parent_agent']} step {c['parent_step']}"
        summaries.append(text)
    return summaries


# ---------------------------------------------------------------------------
# 4. Bottleneck Explanation
# ---------------------------------------------------------------------------

def decompose_hotspot_duration(
    step: dict,
    analytics_row: dict | None = None,
    idle_gap: float | None = None,
) -> dict:
    """Decompose a step's duration into tool execution, LLM inference, and idle components.

    Returns {tool_s, inference_s, idle_s, tool_pct, inference_pct, idle_pct,
             timing_incomplete, dominant_tool}
    """
    duration = step.get("duration") or 0
    if duration <= 0:
        return {
            "tool_s": 0, "inference_s": 0, "idle_s": 0,
            "tool_pct": 0, "inference_pct": 0, "idle_pct": 0,
            "timing_incomplete": True, "dominant_tool": None,
        }

    idle_s = max(0, idle_gap or 0)

    # Sum tool call durations
    tool_time_ms = 0
    timing_complete = True
    dominant_tool_name = ""
    dominant_tool_target = ""
    dominant_tool_dur = 0

    from .metrics import tool_call_duration_ms

    for tc in step.get("tool_calls", []):
        v = tool_call_duration_ms(tc)
        dur_ms = v if v is not None else 0
        if v is None:
            timing_complete = False

        tool_time_ms += dur_ms
        if dur_ms > dominant_tool_dur:
            dominant_tool_dur = dur_ms
            dominant_tool_name = tc.get("tool_name", "")
            inp = tc.get("input", {})
            if isinstance(inp, dict):
                for k in ("file_path", "command", "pattern", "path"):
                    if inp.get(k):
                        dominant_tool_target = str(inp[k])[:50]
                        break

    tool_s = min(tool_time_ms / 1000.0, duration)  # cap at step duration
    # inference = the step's own duration minus tool time. idle_s is the gap
    # BEFORE the step (prev completion -> this start) and is disjoint from the
    # step's duration, so it must not be subtracted here.
    inference_s = max(0, duration - tool_s)

    dominant_tool = None
    if dominant_tool_name:
        dominant_tool = {
            "name": dominant_tool_name,
            "target": dominant_tool_target,
            "duration_s": round(dominant_tool_dur / 1000.0, 2),
        }

    # All three percentages share one base — the full wall-clock span the step
    # accounts for (its own duration plus the disjoint pre-step idle gap) — so
    # they sum to ~100% and idle_pct can never exceed 100.
    total = duration + idle_s

    return {
        "tool_s": round(tool_s, 2),
        "inference_s": round(inference_s, 2),
        "idle_s": round(idle_s, 2),
        "tool_pct": round(tool_s / total * 100, 1) if total > 0 else 0,
        "inference_pct": round(inference_s / total * 100, 1) if total > 0 else 0,
        "idle_pct": round(idle_s / total * 100, 1) if total > 0 else 0,
        "timing_incomplete": not timing_complete,
        "dominant_tool": dominant_tool,
    }


def explain_hotspot(step: dict, decomposition: dict) -> str:
    """Generate a one-line explanation for a hotspot step."""
    idx = step["index"]
    dur = step.get("duration") or 0
    d = decomposition

    parts = []

    # Dominant component first
    if d["tool_pct"] >= d["inference_pct"] and d["tool_pct"] >= d["idle_pct"]:
        dt = d.get("dominant_tool")
        if dt:
            target = f": {dt['target']}" if dt["target"] else ""
            parts.append(f"{d['tool_s']}s executing tools ({dt['name']}{target} {dt['duration_s']}s)")
        else:
            parts.append(f"{d['tool_s']}s executing tools")
        if d["inference_s"] > 0:
            tok = step["tokens"]["total"]
            parts.append(f"{d['inference_s']}s LLM inference ({tok:,} tokens)")
    elif d["idle_pct"] >= d["inference_pct"]:
        parts.append(f"{d['idle_s']}s idle gap before step (queuing or rate limiting)")
        if d["tool_s"] > 0:
            parts.append(f"{d['tool_s']}s tool execution")
        # Mirror the other branches: never drop a component of the step's own
        # duration just because the pre-step idle gap dominates.
        if d["inference_s"] > 0:
            tok = step["tokens"]["total"]
            parts.append(f"{d['inference_s']}s LLM inference ({tok:,} tokens)")
    else:
        tok = step["tokens"]["total"]
        parts.append(f"{d['inference_s']}s LLM inference ({tok:,} tokens)")
        if d["tool_s"] > 0:
            parts.append(f"{d['tool_s']}s tool execution")

    if d["idle_s"] > 0 and d["idle_pct"] < d["tool_pct"] and d["idle_pct"] < d["inference_pct"]:
        parts.append(f"{d['idle_s']}s idle")

    incomplete = " [timing incomplete]" if d["timing_incomplete"] else ""
    return f"Step {idx}: {dur:.1f}s \u2014 {', '.join(parts)}{incomplete}"


def compute_bottleneck_explanations(
    steps: list[dict],
    step_analytics: list[dict],
    n: int = 5,
) -> list[dict]:
    """Compute duration decomposition and explanation for top-N hotspot steps.

    Returns list of {step_idx, duration, decomposition, explanation} dicts.
    """
    # Find top-N assistant steps by duration
    asst = [s for s in steps if s.get("role") == "assistant" and s.get("duration")]
    asst.sort(key=lambda s: -(s.get("duration") or 0))
    hotspots = asst[:n]

    # Build analytics lookup
    analytics_map = {a["index"]: a for a in step_analytics}

    results: list[dict] = []
    for step in hotspots:
        idx = step["index"]
        analytics_row = analytics_map.get(idx)
        idle_gap = analytics_row.get("idle_before_s") if analytics_row else None

        decomp = decompose_hotspot_duration(step, analytics_row, idle_gap)
        explanation = explain_hotspot(step, decomp)

        results.append({
            "step_idx": idx,
            "duration": step.get("duration", 0),
            "decomposition": decomp,
            "explanation": explanation,
        })

    return results


# ---------------------------------------------------------------------------
# 5. Context-window pressure
# ---------------------------------------------------------------------------

PRESSURE_ALL_AGENTS = "__all__"
PRESSURE_MAIN_AGENT = "__main__"

# Occupancy is a compaction candidate when it falls below this fraction of the
# previous non-zero same-agent occupancy.
_OCCUPANCY_DROP_RATIO = 0.7

# High-confidence model-id prefixes only — occupancy % is omitted when unknown.
_MODEL_CONTEXT_LIMITS: tuple[tuple[str, int], ...] = (
    ("claude", 200_000),
    ("gpt-4o", 128_000),
)


def pressure_agent_key(agent_id: str) -> str:
    """Dropdown value for an ``effective_agent`` / pressure-agent id."""
    return PRESSURE_MAIN_AGENT if not agent_id else str(agent_id)


def pressure_agent_id(dropdown_key: str | None) -> str | None:
    """Map a dropdown value to an agent id. ``None`` means all agents."""
    if dropdown_key in (None, "", PRESSURE_ALL_AGENTS):
        return None
    if dropdown_key == PRESSURE_MAIN_AGENT:
        return ""
    return dropdown_key


def _pressure_agent(step: dict) -> str:
    """Context-window identity for a step.

    Each OpenCode/CodeArts session has its own window, even when child
    sessions are not tagged ``isSubAgent`` / ``(subagent)``. Prefer
    ``session_id`` when present; otherwise fall back to ``effective_agent``.
    """
    session_id = step.get("session_id") or ""
    if isinstance(session_id, str) and session_id:
        return session_id
    role = step.get("role", "")
    if role not in ("user", "compaction"):
        return effective_agent(step)
    agent = step.get("agent", "") or ""
    suffix_sub = isinstance(agent, str) and agent.endswith("(subagent)")
    if step.get("is_sub_agent") or suffix_sub:
        return agent or ""
    return ""


def _is_occupancy_step(step: dict) -> bool:
    """True when the step contributes a live context-window occupancy point."""
    if step.get("is_compaction_checkpoint") or step.get("role") == "compaction":
        return False
    return step.get("role") == "assistant"


def _agent_pressure_label(agent_id: str, steps: list[dict]) -> str:
    if not agent_id:
        return "main"
    for step in steps:
        if _pressure_agent(step) != agent_id:
            continue
        name = step.get("agent") or ""
        title = step.get("session_title") or ""
        if name:
            return name if len(name) <= 40 else name[:39] + "\u2026"
        if title:
            return title if len(title) <= 40 else title[:39] + "\u2026"
        break
    return agent_id[:8] + "\u2026" if len(agent_id) > 8 else agent_id


def _disambiguate_pressure_labels(agent_ids: list[str], steps: list[dict]) -> dict[str, str]:
    """Unique dropdown/legend labels; suffix a short session id on collisions."""
    from collections import Counter

    raw = {agent_id: _agent_pressure_label(agent_id, steps) for agent_id in agent_ids}
    counts = Counter(raw.values())
    labels: dict[str, str] = {}
    for agent_id, label in raw.items():
        if counts[label] > 1 and agent_id:
            suffix = agent_id[-6:] if len(agent_id) > 6 else agent_id
            labels[agent_id] = f"{label} ({suffix})"
        else:
            labels[agent_id] = label
    return labels


def step_context_occupancy(step: dict) -> dict:
    """Schema-normalized prompt occupancy for one step.

    ``occupancy = fresh input + cache read``. Fresh input is inferred by
    :func:`trajviz.insight.parser.infer_non_cache_input` so Claude Code
    (cache-exclusive input) and OpenCode (cache-excluded input) agree.
    """
    tokens = step.get("tokens") if isinstance(step.get("tokens"), dict) else {}
    tok_total = tokens.get("total", 0) or 0
    tok_input = tokens.get("input", 0) or 0
    tok_output = tokens.get("output", 0) or 0
    tok_reasoning = tokens.get("reasoning", 0) or 0
    cache_read = tokens.get("cache_read", 0) or 0
    fresh = infer_non_cache_input(
        total_tokens=tok_total,
        input_tokens=tok_input,
        output_tokens=tok_output,
        reasoning_tokens=tok_reasoning,
        cache_read_tokens=cache_read,
    )
    occupancy = fresh + (cache_read or 0)
    return {
        "fresh": int(fresh),
        "cache_read": int(cache_read or 0),
        "occupancy": int(occupancy),
    }


def infer_context_window_limit(
    steps: list[dict],
    raw: dict | None = None,
) -> int | None:
    """Return a context-window token limit, or None when it cannot be known.

    Prefers an explicit CodeArts ``context_tokens`` / ``contextToken`` field,
    then a small model-id prefix table. Never guesses from peak occupancy.
    """
    if isinstance(raw, dict):
        md = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        for key in ("context_tokens", "contextToken"):
            value = md.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                return int(value)
        base = raw.get("chat_base_info")
        if isinstance(base, dict):
            value = base.get("contextToken")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                return int(value)
    for step in steps:
        model_id = (step.get("model_id") or "").lower()
        if not model_id:
            continue
        for prefix, limit in _MODEL_CONTEXT_LIMITS:
            if model_id.startswith(prefix):
                return limit
    return None


def _previous_agent_occupancy(
    steps: list[dict],
    index: int,
    agent_id: str,
) -> int | None:
    for step in reversed(steps[:index]):
        if not _is_occupancy_step(step):
            continue
        if _pressure_agent(step) != agent_id:
            continue
        occ = step_context_occupancy(step)["occupancy"]
        if occ > 0:
            return occ
    return None


def _next_agent_occupancy(
    steps: list[dict],
    index: int,
    agent_id: str,
) -> int | None:
    for step in steps[index + 1:]:
        if not _is_occupancy_step(step):
            continue
        if _pressure_agent(step) != agent_id:
            continue
        occ = step_context_occupancy(step)["occupancy"]
        if occ > 0:
            return occ
    return None


def coalesce_compaction_events(events: list[dict]) -> list[dict]:
    """Merge adjacent same-session compaction signals into one event."""
    if not events:
        return []
    ordered = sorted(events, key=lambda e: int(e.get("step") or 0))
    merged: list[dict] = [dict(ordered[0])]
    for event in ordered[1:]:
        prev = merged[-1]
        step = int(event.get("step") or 0)
        if step - int(prev.get("step") or 0) <= 1:
            after = event.get("occupancy_after")
            if isinstance(after, (int, float)) and after > 0:
                prev["occupancy_after"] = int(after)
            dropped = event.get("dropped")
            if isinstance(dropped, (int, float)) and dropped > (prev.get("dropped") or 0):
                prev["dropped"] = int(dropped)
            if event.get("kind") == "summary":
                prev["kind"] = "summary"
            continue
        merged.append(dict(event))
    for event in merged:
        before = event.get("occupancy_before")
        after = event.get("occupancy_after")
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            event["dropped"] = max(0, int(before) - int(after))
    return merged


def _splice_compaction_into_points(
    points: list[dict],
    events: list[dict],
) -> list[dict]:
    """Insert a vertical occupancy cliff at each compaction step.

    Live occupancy is recorded on assistant turns, so without this splice the
    line slopes between the last pre-compaction turn and the first summary
    turn, and the compaction marker floats above the line.
    """
    if not points or not events:
        return [dict(p) for p in points]
    out = [dict(p) for p in points]
    for event in coalesce_compaction_events(events):
        step = int(event.get("step") or 0)
        before = event.get("occupancy_before")
        after = event.get("occupancy_after")
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            continue
        if not (0 < after < before):
            continue
        prev = nxt = None
        at_step: list[dict] = []
        for point in out:
            if point["step"] < step:
                prev = point
            elif point["step"] == step:
                at_step.append(point)
            elif nxt is None:
                nxt = point
        src_before = prev or (at_step[0] if at_step else None)
        src_after = nxt or (at_step[-1] if at_step else src_before)
        if src_before is None:
            continue
        if src_after is None:
            src_after = src_before
        out.append({
            "step": step,
            "local_turn": src_before.get("local_turn", 0),
            "fresh": src_before.get("fresh", 0),
            "cache_read": src_before.get("cache_read", 0),
            "occupancy": int(before),
        })
        out.append({
            "step": step,
            "local_turn": src_after.get("local_turn", 0),
            "fresh": src_after.get("fresh", 0),
            "cache_read": src_after.get("cache_read", 0),
            "occupancy": int(after),
        })
    out.sort(key=lambda p: (p["step"], -int(p["occupancy"])))
    deduped: list[dict] = []
    for point in out:
        if (
            deduped
            and deduped[-1]["step"] == point["step"]
            and deduped[-1]["occupancy"] == point["occupancy"]
        ):
            continue
        deduped.append(point)
    return deduped


def detect_compaction_events(steps: list[dict]) -> list[dict]:
    """Detect compaction / prune / occupancy-drop events per agent.

    Explicit log signals are preferred. Occupancy-drop is a fallback only
    when the drop is *per session* and *persists* on the next turn — a
    one-step dip that recovers is cache jitter, not compaction.
    """
    events: list[dict] = []
    explicit_steps: set[int] = set()

    def _event(step: dict, kind: str, agent_id: str, *, position: int | None = None, **extra) -> dict:
        idx = int(step.get("index", 0))
        pos = position if position is not None else idx
        before = extra.get("occupancy_before")
        if before is None:
            before = _previous_agent_occupancy(steps, pos, agent_id)
        after = extra.get("occupancy_after")
        if after is None:
            after = step_context_occupancy(step)["occupancy"]
        # Compaction parts live on user turns with no token totals — look
        # ahead to the next live occupancy of this window.
        if not after:
            after = _next_agent_occupancy(steps, pos, agent_id)
        dropped = None
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            dropped = max(0, int(before) - int(after))
        return {
            "step": idx,
            "agent": agent_id,
            "kind": kind,
            "occupancy_before": int(before) if isinstance(before, (int, float)) else None,
            "occupancy_after": int(after) if isinstance(after, (int, float)) else None,
            "dropped": dropped,
        }

    for i, step in enumerate(steps):
        idx = int(step.get("index", i))
        agent_id = _pressure_agent(step)

        if step.get("is_compaction_checkpoint") or step.get("role") == "compaction":
            events.append(_event(step, "compaction_message", agent_id, position=i))
            explicit_steps.add(idx)
            continue

        for part in step.get("parts") or []:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype == "compaction":
                events.append(_event(step, "compaction_part", agent_id, position=i))
                explicit_steps.add(idx)
            elif ptype in ("step_start", "step_finish"):
                name_l = str(part.get("name", "")).lower()
                if "compress" in name_l or "compact" in name_l:
                    events.append(_event(step, "compress_step", agent_id, position=i))
                    explicit_steps.add(idx)

        if _is_occupancy_step(step) and step.get("summary") is True:
            events.append(_event(step, "summary", agent_id, position=i))
            explicit_steps.add(idx)

        for tc in step.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            compacted = tc.get("time_compacted")
            if compacted is None:
                continue
            events.append(_event(step, "tool_prune", agent_id, position=i))
            explicit_steps.add(idx)
            break

    from collections import defaultdict

    occ_seq: dict[str, list[tuple[int, int, dict]]] = defaultdict(list)
    for step in steps:
        if not _is_occupancy_step(step):
            continue
        occ = step_context_occupancy(step)["occupancy"]
        if occ <= 0:
            continue
        occ_seq[_pressure_agent(step)].append(
            (int(step.get("index", 0)), occ, step),
        )

    explicit_at = {(e["agent"], e["step"]) for e in events}
    for agent_id, points in occ_seq.items():
        for i in range(1, len(points)):
            _prev_idx, prev_occ, _prev_step = points[i - 1]
            idx, occ, step = points[i]
            if occ >= prev_occ * _OCCUPANCY_DROP_RATIO:
                continue
            if idx in explicit_steps or (agent_id, idx) in explicit_at:
                continue
            # Adjacent explicit compaction (part on the previous user turn, etc.)
            if any(abs(idx - estep) <= 1 and eagent == agent_id
                   for eagent, estep in explicit_at):
                continue
            if i + 1 < len(points):
                next_occ = points[i + 1][1]
                # Recovered on the next turn → cache/prefix jitter, not compaction.
                if next_occ >= prev_occ * 0.8:
                    continue
            elif occ >= prev_occ * 0.4:
                # Last point: only count a severe drop we cannot confirm.
                continue
            events.append(_event(
                step, "occupancy_drop", agent_id,
                occupancy_before=prev_occ, occupancy_after=occ,
            ))
            explicit_steps.add(idx)

    events.sort(key=lambda e: (e["step"], e["kind"]))
    return events


def pressure_agent_choices(steps: list[dict]) -> list[tuple[str, str]]:
    """``(label, value)`` pairs for the Diagnostics agent dropdown."""
    order: list[str] = []
    seen: set[str] = set()
    for step in steps:
        if not _is_occupancy_step(step):
            continue
        agent_id = _pressure_agent(step)
        if agent_id not in seen:
            seen.add(agent_id)
            order.append(agent_id)
    labels = _disambiguate_pressure_labels(order, steps)
    choices = [("All agents", PRESSURE_ALL_AGENTS)]
    for agent_id in order:
        choices.append((labels[agent_id], pressure_agent_key(agent_id)))
    return choices


def context_pressure_series(
    steps: list[dict],
    *,
    agent_key: str | None = None,
    raw: dict | None = None,
) -> dict:
    """Build per-agent occupancy series and compaction events for charting.

    Occupancy points skip compaction-only checkpoints so the line shows the
    live window (the sawtooth). ``agent_key`` of ``__all__`` / ``None``
    overlays every agent; ``__main__`` is the empty effective-agent id.
    """
    target = pressure_agent_id(agent_key)
    events = detect_compaction_events(steps)
    if target is not None:
        events = [e for e in events if e["agent"] == target]

    agents_order: list[str] = []
    seen: set[str] = set()
    points_by_agent: dict[str, list[dict]] = {}
    local_turn: dict[str, int] = {}
    for step in steps:
        if not _is_occupancy_step(step):
            continue
        agent_id = _pressure_agent(step)
        if target is not None and agent_id != target:
            continue
        if agent_id not in seen:
            seen.add(agent_id)
            agents_order.append(agent_id)
            points_by_agent[agent_id] = []
            local_turn[agent_id] = 0
        local_turn[agent_id] += 1
        occ = step_context_occupancy(step)
        points_by_agent[agent_id].append({
            "step": int(step.get("index", 0)),
            "local_turn": local_turn[agent_id],
            "fresh": occ["fresh"],
            "cache_read": occ["cache_read"],
            "occupancy": occ["occupancy"],
        })

    window_limit = infer_context_window_limit(steps, raw)
    labels = _disambiguate_pressure_labels(agents_order, steps)
    agents = []
    for agent_id in agents_order:
        agent_events = [e for e in events if e.get("agent") == agent_id]
        agents.append({
            "agent_id": agent_id,
            "key": pressure_agent_key(agent_id),
            "label": labels[agent_id],
            "points": _splice_compaction_into_points(
                points_by_agent[agent_id], agent_events,
            ),
        })
    return {
        "agents": agents,
        "events": events,
        "window_limit": window_limit,
        "agent_key": agent_key or PRESSURE_ALL_AGENTS,
    }


def context_pressure_stats(series: dict) -> dict:
    """Peak occupancy, optional peak %, compaction count, and largest drop."""
    peak = 0
    for agent in series.get("agents") or []:
        for point in agent.get("points") or []:
            peak = max(peak, int(point.get("occupancy") or 0))
    events = series.get("events") or []
    largest_drop = 0
    for event in events:
        dropped = event.get("dropped")
        if isinstance(dropped, (int, float)):
            largest_drop = max(largest_drop, int(dropped))
    window_limit = series.get("window_limit")
    peak_pct = None
    if isinstance(window_limit, (int, float)) and window_limit > 0 and peak > 0:
        peak_pct = round(100.0 * peak / window_limit, 1)
    return {
        "peak_occupancy": peak,
        "peak_pct": peak_pct,
        "compaction_count": len(events),
        "largest_drop": largest_drop,
        "window_limit": window_limit if isinstance(window_limit, (int, float)) else None,
    }
