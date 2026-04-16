"""Metric computation, health verdicts, and agent summaries."""

import statistics
from typing import Any


def effective_agent(s: dict) -> str:
    """Return the best agent identifier for a step.

    User turns always map to ``""`` (main).  For assistant turns:
    - Claude Code / OpenCode: uses ``agent`` field (sub-agent ID).
    - CodeArts: uses ``session_id`` as the grouping key since each
      sub-agent gets a distinct session, and ``agent`` may be the same
      generic name (e.g. "Agent") for both main and sub-agent messages.
    Falls back to empty string for the main agent.
    """
    if s.get("role") == "user":
        return ""
    # CodeArts: prefer session_id for sub-agents (agent field is often
    # the same generic name like "Agent" for all messages)
    if s.get("is_sub_agent") and s.get("session_id"):
        return s["session_id"]
    agent = s.get("agent", "")
    if agent:
        return agent
    return ""


def _percentile(values: list[float], q: float) -> float:
    """Compute percentile using nearest-rank (q in [0, 1])."""
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    vals = sorted(values)
    idx = max(0, min(len(vals) - 1, int((len(vals) - 1) * q)))
    return vals[idx]


def validate_token_integrity(steps: list[dict]) -> list[str]:
    """Check for steps with missing or all-zero token data.

    Returns a list of human-readable warning strings (empty if all OK).
    """
    zero_token_steps = []
    for s in steps:
        tokens = s.get("tokens", {})
        total = tokens.get("total", 0) or 0
        inp = tokens.get("input", 0) or 0
        out = tokens.get("output", 0) or 0
        cache = tokens.get("cache_read", 0) or 0
        if total == 0 and inp == 0 and out == 0 and cache == 0:
            if s.get("role") == "assistant":
                zero_token_steps.append(s.get("index", "?"))

    warnings: list[str] = []
    if zero_token_steps:
        n = len(zero_token_steps)
        examples = ", ".join(str(i) for i in zero_token_steps[:5])
        suffix = f" and {n - 5} more" if n > 5 else ""
        warnings.append(
            f"{n} assistant step(s) have zero token data "
            f"(steps {examples}{suffix}) \u2014 metrics may be inaccurate."
        )
    return warnings


def build_message_metrics(steps: list[dict]) -> list[dict]:
    """Build per-message metrics used for diagnostics tables and charts."""
    from .parser import infer_non_cache_input

    rows: list[dict] = []
    for s in steps:
        tokens = s.get("tokens", {})
        tok_total = tokens.get("total", 0) or 0
        tok_input = tokens.get("input", 0) or 0
        tok_output = tokens.get("output", 0) or 0
        tok_reasoning = tokens.get("reasoning", 0) or 0
        cache_read = tokens.get("cache_read", 0) or 0
        non_cache = infer_non_cache_input(
            total_tokens=tok_total,
            input_tokens=tok_input,
            output_tokens=tok_output,
            reasoning_tokens=tok_reasoning,
            cache_read_tokens=cache_read,
        )
        duration = s.get("duration")

        tool_time_sum = 0.0
        for tc in s.get("tool_calls", []):
            ts = tc.get("time_start")
            te = tc.get("time_end")
            if isinstance(ts, (int, float)) and isinstance(te, (int, float)) and te >= ts:
                tool_time_sum += (te - ts) / 1000.0
            else:
                dm = tc.get("duration_ms")
                if dm is None:
                    dm = (tc.get("metadata") or {}).get("totalDurationMs")
                if isinstance(dm, (int, float)) and dm > 0:
                    tool_time_sum += dm / 1000.0

        part_counts: dict[str, int] = {}
        for p in s.get("parts", []):
            pt = p.get("type", "unknown")
            part_counts[pt] = part_counts.get(pt, 0) + 1

        rows.append({
            "index": s.get("index", 0),
            "role": s.get("role", "?"),
            "agent": effective_agent(s),
            "model_id": s.get("model_id", ""),
            "finish": s.get("finish", ""),
            "duration": duration,
            "tokens_total": tok_total,
            "tokens_input": tok_input,
            "tokens_output": tok_output,
            "cache_read": cache_read,
            "non_cache_tokens": non_cache,
            "cache_ratio": (cache_read / tok_total) if tok_total else 0.0,
            "tokens_per_sec": (tok_total / duration) if duration and duration > 0 else None,
            "non_cache_per_sec": (non_cache / duration) if duration and duration > 0 else None,
            "output_input_ratio": (tok_output / max(1, tok_input)),
            "tool_calls": s.get("tool_call_count", 0),
            "errors": s.get("error_count", 0),
            "tool_time_sum": tool_time_sum,
            "tool_time_share": (tool_time_sum / duration) if duration and duration > 0 else 0.0,
            "reasoning_parts": part_counts.get("reasoning", 0),
            "text_parts": part_counts.get("text", 0),
            "patch_parts": part_counts.get("patch", 0),
        })
    return rows


def _compute_command_metrics(steps: list[dict]) -> dict:
    """Compute command execution success rate from tool calls with exit codes."""
    cmd_total = 0
    cmd_failures = 0
    for s in steps:
        for tc in s.get("tool_calls", []):
            meta = tc.get("metadata", {})
            if not isinstance(meta, dict):
                continue
            if "exit" in meta:
                cmd_total += 1
                if meta["exit"] != 0:
                    cmd_failures += 1
    if cmd_total == 0:
        return {"command_success_rate": None, "command_call_count": None, "command_failures": None}
    return {
        "command_success_rate": round((cmd_total - cmd_failures) / cmd_total, 4),
        "command_call_count": cmd_total,
        "command_failures": cmd_failures,
    }


def _compute_timing_metrics(steps: list[dict], total_output_tokens: int) -> dict:
    """Compute throughput timing: TTFT, output tok/s, TTLT."""
    first_user_created = None
    first_asst_completed = None
    last_asst_completed = None
    total_asst_duration = 0.0

    for s in steps:
        role = s.get("role", "")
        t_created = s.get("time_created_ms")
        t_completed = s.get("time_completed_ms")
        if role == "user" and first_user_created is None and isinstance(t_created, (int, float)):
            first_user_created = t_created
        if role == "assistant":
            if isinstance(t_completed, (int, float)):
                if first_asst_completed is None:
                    first_asst_completed = t_completed
                last_asst_completed = t_completed
            d = s.get("duration")
            if d is not None:
                total_asst_duration += d

    result: dict = {}
    if first_user_created is not None and first_asst_completed is not None:
        result["time_to_first_token"] = round((first_asst_completed - first_user_created) / 1000, 3)
    else:
        result["time_to_first_token"] = None

    result["output_tokens_per_sec"] = (
        round(total_output_tokens / total_asst_duration, 1)
        if total_asst_duration > 0 and total_output_tokens > 0 else None
    )

    if first_user_created is not None and last_asst_completed is not None:
        result["time_to_last_token"] = round((last_asst_completed - first_user_created) / 1000, 3)
    else:
        result["time_to_last_token"] = None

    return result


def _compute_plan_metrics(steps: list[dict]) -> dict:
    """Compute plan tracking from todo snapshot parts."""
    snapshots = []
    for s in steps:
        for p in s.get("parts", []):
            if p.get("type") == "snapshot":
                data = p.get("data", {})
                if isinstance(data, dict) and ("todos" in data or "items" in data):
                    snapshots.append(data)
    if not snapshots:
        return {"plan_items": None, "plan_completion_ratio": None, "plan_update_count": None}

    # First snapshot for plan_items count
    first = snapshots[0]
    items = first.get("todos", first.get("items", []))
    plan_items = len(items) if isinstance(items, list) else 0

    # Last snapshot for completion ratio
    last = snapshots[-1]
    last_items = last.get("todos", last.get("items", []))
    if isinstance(last_items, list) and last_items:
        completed = sum(
            1 for item in last_items
            if (isinstance(item, dict) and item.get("completed", False))
            or (isinstance(item, dict) and item.get("status") in ("completed", "done"))
        )
        plan_completion_ratio = round(completed / len(last_items), 4)
    else:
        plan_completion_ratio = None

    return {
        "plan_items": plan_items,
        "plan_completion_ratio": plan_completion_ratio,
        "plan_update_count": len(snapshots),
    }


def _compute_token_stats(total_tokens, total_duration, steps, message_rows, raw):
    """Token breakdown, throughput, and cache metrics."""
    output = raw.get("output", {}) if isinstance(raw.get("output"), dict) else {}
    session_raw = raw.get("session_raw", {}) if isinstance(raw.get("session_raw"), dict) else {}
    summary = session_raw.get("summary") if isinstance(session_raw.get("summary"), dict) else None

    assistant_rows = [r for r in message_rows if r.get("role") == "assistant"]
    assistant_tokens = [r["tokens_total"] for r in assistant_rows]
    token_rates = [r["tokens_per_sec"] for r in assistant_rows if r.get("tokens_per_sec") is not None]
    cache_ratios = [r["cache_ratio"] for r in assistant_rows if r["tokens_total"] > 0]
    non_cache_total = sum(r["non_cache_tokens"] for r in message_rows)
    cache_dominant = sum(1 for r in assistant_rows if r["tokens_total"] > 0 and r["cache_ratio"] >= 0.90)
    total_io = total_tokens["input"] + total_tokens["output"]
    churn = (summary["additions"] + summary["deletions"]) if summary and "additions" in summary and "deletions" in summary else 0
    return {
        "tokens": total_tokens,
        "non_cache_tokens": non_cache_total,
        "non_cache_ratio": round(non_cache_total / total_tokens["total"] * 100, 1) if total_tokens["total"] else 0,
        "avg_tokens_per_step": round(total_tokens["total"] / len(steps)) if steps else 0,
        "tokens_per_second": round(total_tokens["total"] / total_duration, 1) if total_duration else 0,
        "output_input_ratio": round(total_tokens["output"] / max(1, total_tokens["input"]), 3),
        "median_step_tokens": round(statistics.median(assistant_tokens)) if assistant_tokens else 0,
        "p95_step_tokens": round(_percentile(assistant_tokens, 0.95)) if assistant_tokens else 0,
        "median_tokens_per_second": round(statistics.median(token_rates), 1) if token_rates else 0,
        "avg_cache_ratio": round(statistics.mean(cache_ratios) * 100, 1) if cache_ratios else 0,
        "cache_dominant_steps": cache_dominant,
        "assistant_steps": len(assistant_rows),
        "user_steps": sum(1 for r in message_rows if r.get("role") == "user"),
        "input_tokens": total_tokens["input"],
        "output_tokens": total_tokens["output"],
        "cache_read_tokens": total_tokens["cache_read"],
        "cache_utilization_ratio": (
            round(total_tokens["cache_read"] / (total_tokens["cache_read"] + total_tokens["input"]), 4)
            if (total_tokens["cache_read"] + total_tokens["input"]) > 0 and total_tokens["cache_read"] > 0 else None
        ),
        "tokens_per_patch_line": round(total_io / output.get("patch_lines", 0), 1) if output.get("patch_lines", 0) > 0 else None,
        "tokens_per_churn_line": round(total_io / churn, 1) if churn > 0 else None,
    }


def _compute_tool_stats(steps, total_tokens_total, total_duration, message_rows):
    """Tool frequency, success rate, duration, and load metrics."""
    tool_count = 0
    tool_breakdown: dict[str, int] = {}
    tool_status_breakdown: dict[str, int] = {}
    tool_success = 0
    tool_fail = 0
    tool_durations: list[float] = []

    for s in steps:
        tool_count += s["tool_call_count"]
        for tc in s["tool_calls"]:
            name = tc["tool_name"]
            tool_breakdown[name] = tool_breakdown.get(name, 0) + 1
            status = tc.get("status", "unknown")
            tool_status_breakdown[status] = tool_status_breakdown.get(status, 0) + 1
            if status in {"error", "failed", "failure", "cancelled", "canceled", "timeout", "timed_out"}:
                tool_fail += 1
            elif status in {"?", "unknown", ""}:
                # Unknown status: treat as success unless it has a classified error_type
                if tc.get("error_type"):
                    tool_fail += 1
                else:
                    tool_success += 1
            else:
                tool_success += 1
            ts, te = tc.get("time_start"), tc.get("time_end")
            if isinstance(ts, (int, float)) and isinstance(te, (int, float)) and te >= ts:
                tool_durations.append((te - ts) / 1000.0)
            else:
                dm = tc.get("duration_ms")
                if dm is None:
                    dm = (tc.get("metadata") or {}).get("totalDurationMs")
                if isinstance(dm, (int, float)) and dm > 0:
                    tool_durations.append(dm / 1000.0)

    assistant_rows = [r for r in message_rows if r.get("role") == "assistant"]
    tool_time_total = sum(r["tool_time_sum"] for r in message_rows)
    avg_td = statistics.mean(tool_durations) if tool_durations else 0
    return {
        "tool_call_count": tool_count,
        "tool_breakdown": tool_breakdown,
        "tool_status_breakdown": tool_status_breakdown,
        "tool_success": tool_success,
        "tool_fail": tool_fail,
        "tool_success_rate": round(tool_success / tool_count * 100, 1) if tool_count else 0,
        "tokens_per_tool": round(total_tokens_total / tool_count) if tool_count else 0,
        "tool_time_total": round(tool_time_total, 2),
        "tool_wait_share": round(tool_time_total / total_duration * 100, 1) if total_duration else 0,
        "avg_tool_duration": round(avg_td, 3),
        "p95_tool_duration": round(_percentile(tool_durations, 0.95), 3) if tool_durations else 0,
        "max_tool_duration": round(max(tool_durations), 3) if tool_durations else 0,
        "multi_tool_steps": sum(1 for r in assistant_rows if r["tool_calls"] >= 2),
        "no_tool_assistant_steps": sum(1 for r in assistant_rows if r["tool_calls"] == 0),
        "patch_steps": sum(1 for r in assistant_rows if r["patch_parts"] > 0),
        "tool_calls_per_min": round(tool_count / (total_duration / 60), 2) if total_duration > 0 else None,
        "tool_time_fraction": round(tool_time_total / total_duration, 4) if total_duration > 0 else None,
        "tool_system_failure_rate": round(tool_fail / tool_count, 4) if tool_count > 0 else None,
    }


def _compute_efficiency_stats(steps, message_rows, raw):
    """Behavioral, structural, and change-scope metrics."""
    roles: dict[str, int] = {}
    agent_breakdown: dict[str, int] = {}
    model_breakdown: dict[str, int] = {}
    finish_breakdown: dict[str, int] = {}
    reasoning_parts = text_parts = snapshot_parts = 0
    for s in steps:
        roles[s["role"]] = roles.get(s["role"], 0) + 1
        agent = effective_agent(s)
        if agent:
            agent_breakdown[agent] = agent_breakdown.get(agent, 0) + 1
        model = s.get("model_id", "")
        if model:
            model_breakdown[model] = model_breakdown.get(model, 0) + 1
        finish = s.get("finish", "")
        if finish:
            finish_breakdown[finish] = finish_breakdown.get(finish, 0) + 1
        for p in s.get("parts", []):
            pt = p.get("type", "")
            if pt == "reasoning":
                reasoning_parts += 1
            elif pt == "text":
                text_parts += 1
            elif pt == "snapshot":
                snapshot_parts += 1

    output = raw.get("output", {}) if isinstance(raw.get("output"), dict) else {}
    session_raw = raw.get("session_raw", {}) if isinstance(raw.get("session_raw"), dict) else {}
    summary = session_raw.get("summary") if isinstance(session_raw.get("summary"), dict) else None
    file_status_raw = raw.get("file_status")
    asst_durs = [s["duration"] for s in steps if s.get("role") == "assistant" and s.get("duration") is not None]
    user_n, asst_n = roles.get("user", 0), roles.get("assistant", 0)
    return {
        "messages_breakdown": roles,
        "agent_breakdown": agent_breakdown,
        "model_breakdown": model_breakdown,
        "finish_breakdown": finish_breakdown,
        "reasoning_parts": reasoning_parts,
        "text_parts": text_parts,
        "snapshot_parts": snapshot_parts,
        "patch_lines": output.get("patch_lines", 0),
        "has_patch": output.get("has_patch", False),
        "patch_error": output.get("error"),
        "files_changed": (
            summary.get("files") if summary and "files" in summary
            else len(file_status_raw) if isinstance(file_status_raw, list) else None
        ),
        "additions": summary.get("additions") if summary else None,
        "deletions": summary.get("deletions") if summary else None,
        "churn": (summary["additions"] + summary["deletions"]) if summary and "additions" in summary and "deletions" in summary else None,
        "net_change": (summary["additions"] - summary["deletions"]) if summary and "additions" in summary and "deletions" in summary else None,
        "user_turns": user_n,
        "assistant_turns": asst_n,
        "autonomy_ratio": round(asst_n / (user_n + asst_n), 4) if (user_n + asst_n) > 0 else None,
        "p50_duration": round(_percentile(asst_durs, 0.50), 2) if asst_durs else None,
        "p90_duration": round(_percentile(asst_durs, 0.90), 2) if asst_durs else None,
        "p99_duration": round(_percentile(asst_durs, 0.99), 2) if asst_durs else None,
    }


def compute_metrics(steps: list[dict], raw: dict, message_rows: list[dict] | None = None) -> dict:
    """Aggregate metrics from parsed steps and raw trajectory."""
    if message_rows is None:
        message_rows = build_message_metrics(steps)

    # Duration stats
    durations = [s["duration"] for s in steps if s.get("duration") is not None]
    total_duration = sum(durations)
    total_tokens = {"total": 0, "input": 0, "output": 0, "reasoning": 0,
                    "cache_read": 0, "cache_write": 0}
    for s in steps:
        for k in total_tokens:
            total_tokens[k] += s["tokens"].get(k, 0)

    timing = raw.get("timing", {}) if isinstance(raw.get("timing"), dict) else {}

    return {
        "total_steps": len(steps),
        "total_duration": round(total_duration, 2),
        "avg_duration": round(total_duration / len(durations), 2) if durations else 0,
        "median_duration": round(statistics.median(durations), 2) if durations else 0,
        "p95_duration": round(_percentile(durations, 0.95), 2) if durations else 0,
        "max_duration": round(max(durations), 2) if durations else 0,
        "wall_clock": timing.get("total_duration", total_duration),
        **_compute_token_stats(total_tokens, total_duration, steps, message_rows, raw),
        **_compute_tool_stats(steps, total_tokens["total"], total_duration, message_rows),
        **_compute_efficiency_stats(steps, message_rows, raw),
        **_compute_command_metrics(steps),
        **_compute_timing_metrics(steps, total_tokens["output"]),
        **_compute_plan_metrics(steps),
    }


def compute_diagnostic_metrics(
    steps: list[dict],
    trajectory: list[dict],
    step_labels: dict[int, dict[str, str]] | None = None,
) -> dict:
    """Compute diagnostic metrics from trajectory analysis.

    These metrics require the raw trajectory data (not just parsed steps)
    for sub-agent detection, fruitless streak analysis, etc.

    Args:
        step_labels: Optional mapping from step index to {phase, action}
            from the step labeler. Enables semantic anti-pattern detection.
    """
    from .patterns import (
        extract_plan_history, compute_plan_metrics as _plan_metrics,
        extract_subagent_sessions, compute_subagent_metrics,
        detect_fruitless_streaks, compute_autonomy_ratio,
        detect_tool_selection_antipatterns,
        build_structural_phase_segments, detect_phase_anomalies,
        detect_semantic_antipatterns,
    )

    plan_history = extract_plan_history(steps)
    plan_m = _plan_metrics(plan_history)
    sessions = extract_subagent_sessions(steps, trajectory)
    sa_metrics = compute_subagent_metrics(sessions, steps)
    streaks = detect_fruitless_streaks(steps, trajectory)
    autonomy = compute_autonomy_ratio(steps, trajectory)
    tool_sel = detect_tool_selection_antipatterns(steps)
    error_count = sum(1 for s in steps for tc in s.get("tool_calls", []) if tc.get("error_type"))

    # Edit precision: successful edits / total edit attempts
    edit_tools = {"Edit", "edit", "Write", "write", "MultiEdit", "multiedit",
                  "str_replace_editor", "create_file"}
    edit_total = 0
    edit_success = 0
    for s in steps:
        for tc in s.get("tool_calls", []):
            if tc.get("tool_name") in edit_tools:
                edit_total += 1
                if tc.get("status") not in ("error", "failed", "failure"):
                    edit_success += 1

    # Search-to-action ratio: read/search calls per edit/write call
    search_tools = {"Read", "read", "Grep", "grep", "Glob", "glob",
                    "WebFetch", "WebSearch"}
    search_count = sum(1 for s in steps for tc in s.get("tool_calls", [])
                       if tc.get("tool_name") in search_tools)

    # Retry rate: same tool+target called multiple times (excess calls only)
    tool_targets: dict[str, int] = {}
    for s in steps:
        for tc in s.get("tool_calls", []):
            name = tc.get("tool_name", "")
            inp = tc.get("input", {})
            target = inp.get("file_path", inp.get("command", inp.get("pattern", ""))) if isinstance(inp, dict) else ""
            key = f"{name}:{str(target)[:80]}"
            tool_targets[key] = tool_targets.get(key, 0) + 1
    retry_calls = sum(v - 1 for v in tool_targets.values() if v > 1)
    total_calls = sum(tool_targets.values()) or 1

    # Context compression events — deduplicate between part scan and token drop
    compression_steps: set[int] = set()
    for i, s in enumerate(steps):
        for p in s.get("parts", []):
            if (p.get("type") in ("step_start", "step_finish")
                    and "compress" in p.get("name", "").lower()):
                compression_steps.add(i)
    # Token-drop heuristic: only applies when tokens grow cumulatively across
    # steps (e.g., Claude Code context window).  For formats with per-step
    # deltas (CodeArts) tokens naturally vary, so drops are not compressions.
    # Detect cumulative pattern: tokens should generally be non-decreasing.
    asst_tokens = [s.get("tokens", {}).get("total", 0) or 0
                   for s in steps if s.get("role") == "assistant"]
    if len(asst_tokens) >= 5:
        increasing = sum(1 for a, b in zip(asst_tokens, asst_tokens[1:]) if b >= a)
        is_cumulative = increasing / (len(asst_tokens) - 1) > 0.7
    else:
        is_cumulative = False
    if is_cumulative:
        for i in range(1, len(steps)):
            if steps[i].get("role") != "assistant":
                continue
            prev_tok = steps[i - 1].get("tokens", {}).get("total", 0) or 0
            curr_tok = steps[i].get("tokens", {}).get("total", 0) or 0
            if prev_tok > 0 and curr_tok > 0 and curr_tok < prev_tok * 0.3:
                if i not in compression_steps:
                    compression_steps.add(i)
    compression_count = len(compression_steps)

    structural_phases = build_structural_phase_segments(steps)
    structural_regressions = detect_phase_anomalies(steps, structural_phases)

    # Semantic anti-patterns (requires step labels from the step labeler)
    sem = detect_semantic_antipatterns(steps, step_labels or {})

    result = {
        "plan_stall_count": len(plan_m.get("stalled", [])),
        "plan_reset_count": plan_m.get("plan_resets", 0),
        "plan_total_items": plan_m.get("total_items", 0),
        "subagent_session_count": len(sessions),
        "subagent_total_steps": sum(s.get("step_count", 0) for s in sa_metrics),
        "subagent_total_tokens": sum(s.get("total_tokens", 0) for s in sa_metrics),
        "fruitless_streak_count": len(streaks),
        "fruitless_streak_max": max((s["length"] for s in streaks), default=0),
        "autonomy_ratio": autonomy,
        "tool_selection_flags": len(tool_sel),
        "classified_error_count": error_count,
        "edit_total": edit_total,
        "edit_success": edit_success,
        "edit_precision": round(edit_success / edit_total * 100, 1) if edit_total else None,
        "search_to_action": round(search_count / edit_total, 1) if edit_total else None,
        "retry_rate": round(retry_calls / total_calls * 100, 1),
        "compression_count": compression_count,
        "structural_phase_count": len(structural_phases),
        "structural_phase_regression_count": len(structural_regressions),
        "structural_phases": structural_phases,
        "structural_phase_regressions": structural_regressions,
    }

    # Append semantic anti-pattern counts when labels are available
    if step_labels:
        result["phase_oscillation_count"] = len(sem["phase_oscillation"])
        result["premature_implementation"] = len(sem["premature_implementation"]) > 0
        result["semantic_fruitless_exploration_count"] = len(sem["semantic_fruitless_exploration"])
        result["validation_avoidance"] = len(sem["validation_avoidance"]) > 0
        result["debug_without_hypothesis_count"] = len(sem["debug_without_hypothesis"])
        result["semantic_plan_stall_count"] = len(sem["semantic_plan_stall"])
        result["semantic_antipatterns"] = sem

    return result


def compute_health_verdict(metrics: dict, step_analytics: list[dict]) -> list[dict]:
    """Compute a health verdict with color-coded status for key metrics."""
    verdicts = []

    # Cache efficiency
    avg_cache = metrics.get("avg_cache_ratio", 0)
    if avg_cache >= 60:
        status, detail = "good", f"Avg cache read {avg_cache}% — strong cache reuse"
    elif avg_cache >= 30:
        status, detail = "warn", f"Avg cache read {avg_cache}% — moderate cache reuse"
    else:
        status, detail = "bad", f"Avg cache read {avg_cache}% — most input tokens are fresh"
    verdicts.append({"metric": "Cache Efficiency", "status": status, "label": f"{avg_cache}%", "detail": detail})

    # Tool success rate
    tool_rate = metrics.get("tool_success_rate", 0)
    tool_count = metrics.get("tool_call_count", 0)
    if tool_count == 0:
        verdicts.append({"metric": "Tool Success", "status": "good", "label": "N/A", "detail": "No tool calls"})
    elif tool_rate >= 95:
        verdicts.append({"metric": "Tool Success", "status": "good", "label": f"{tool_rate}%", "detail": f"{tool_rate}% success across {tool_count} calls"})
    elif tool_rate >= 80:
        verdicts.append({"metric": "Tool Success", "status": "warn", "label": f"{tool_rate}%", "detail": f"{tool_rate}% success — {metrics.get('tool_fail', 0)} failures out of {tool_count} calls"})
    else:
        verdicts.append({"metric": "Tool Success", "status": "bad", "label": f"{tool_rate}%", "detail": f"{tool_rate}% success — high failure rate across {tool_count} calls"})

    # Token efficiency (tok/s)
    tok_per_s = metrics.get("tokens_per_second", 0)
    if tok_per_s >= 50:
        status, detail = "good", f"{tok_per_s} tok/s — strong throughput"
    elif tok_per_s >= 20:
        status, detail = "warn", f"{tok_per_s} tok/s — moderate throughput"
    else:
        status, detail = "bad", f"{tok_per_s} tok/s — low throughput"
    verdicts.append({"metric": "Throughput", "status": status, "label": f"{tok_per_s} tok/s", "detail": detail})

    # Error rate — use metrics directly (step_analytics dicts don't have "parts")
    error_steps = metrics.get("tool_fail", 0)
    if error_steps == 0:
        status, detail = "good", "No error steps detected"
    elif error_steps <= 2:
        status, detail = "warn", f"{error_steps} error step(s) detected"
    else:
        status, detail = "bad", f"{error_steps} error steps — agent may be struggling"
    verdicts.append({"metric": "Errors", "status": status, "label": str(error_steps), "detail": detail})

    return verdicts


def extract_agent_info(steps: list[dict]) -> tuple[str, str, str]:
    """Return (model_id, provider_id, agent_id) from the first assistant step."""
    model_id = provider_id = agent_id = ""
    for s in steps:
        if s["role"] == "assistant" and s.get("model_id"):
            model_id = s["model_id"]
            provider_id = s.get("provider_id", "")
            if s.get("agent"):
                agent_id = s["agent"]
            break
    if not agent_id:
        for s in steps:
            if s.get("agent"):
                agent_id = s["agent"]
                break
    return model_id, provider_id, agent_id


def compute_agent_summary(steps: list[dict], raw: dict) -> list[dict]:
    """Compute per-agent summary statistics from parsed steps.

    Returns a list of agent summary dicts sorted by first appearance,
    containing tokens, duration, tool calls, errors, and cache efficiency.
    """
    from collections import defaultdict

    if not steps:
        return []

    # Determine agent ordering by first appearance (assistant steps only)
    agent_order: list[str] = []
    seen: set[str] = set()
    for s in steps:
        if s.get("role") != "assistant":
            continue
        agent = effective_agent(s)
        if agent not in seen:
            agent_order.append(agent)
            seen.add(agent)

    # Accumulate per-agent stats
    stats: dict[str, dict] = defaultdict(lambda: {
        "step_count": 0, "total_tokens": 0, "input_tokens": 0,
        "output_tokens": 0, "reasoning_tokens": 0, "cache_read_tokens": 0,
        "total_duration_s": 0.0, "tool_call_count": 0, "error_count": 0,
    })
    for s in steps:
        if s.get("role") != "assistant":
            continue
        agent = effective_agent(s)
        d = stats[agent]
        d["step_count"] += 1
        tok = s.get("tokens", {})
        d["total_tokens"] += tok.get("total", 0)
        d["input_tokens"] += tok.get("input", 0)
        d["output_tokens"] += tok.get("output", 0)
        d["reasoning_tokens"] += tok.get("reasoning", 0)
        d["cache_read_tokens"] += tok.get("cache_read", 0)
        dur = s.get("duration")
        if isinstance(dur, (int, float)):
            d["total_duration_s"] += dur
        d["tool_call_count"] += s.get("tool_call_count", 0)
        d["error_count"] += s.get("error_count", 0)

    # Build spawning map from _cc_sub_agents
    spawned_by_map: dict[str, str] = {}  # agent_id -> spawned_by_tool_call_id
    cc_sub_agents = raw.get("_cc_sub_agents", [])
    if isinstance(cc_sub_agents, list):
        for sa in cc_sub_agents:
            if isinstance(sa, dict) and sa.get("agent_id"):
                spawned_by_map[sa["agent_id"]] = sa.get("spawned_by", "")

    # Build tool_call_id -> step_index map for spawning correlation
    tool_call_step_map: dict[str, int] = {}
    for s in steps:
        for tc in s.get("tool_calls", []):
            tid = tc.get("tool_id", "")
            if tid:
                tool_call_step_map[tid] = s.get("index", 0)

    result = []
    for agent_id in agent_order:
        d = stats[agent_id]
        label = "main" if not agent_id else (
            agent_id[:8] + "\u2026" if len(agent_id) > 8 else agent_id
        )
        total_tok = d["total_tokens"]
        cache_read = d["cache_read_tokens"]
        cache_pct = round(cache_read / total_tok * 100, 1) if total_tok > 0 else 0.0
        dur = d["total_duration_s"]
        tok_per_s = round(total_tok / dur, 1) if dur > 0 else 0.0

        # Spawning correlation
        spawned_by_tool = spawned_by_map.get(agent_id, "")
        spawned_by_step = tool_call_step_map.get(spawned_by_tool) if spawned_by_tool else None

        result.append({
            "agent_id": agent_id,
            "label": label,
            "step_count": d["step_count"],
            "total_tokens": total_tok,
            "input_tokens": d["input_tokens"],
            "output_tokens": d["output_tokens"],
            "reasoning_tokens": d["reasoning_tokens"],
            "cache_read_tokens": cache_read,
            "total_duration_s": round(dur, 2),
            "tool_call_count": d["tool_call_count"],
            "error_count": d["error_count"],
            "cache_efficiency_pct": cache_pct,
            "tokens_per_second": tok_per_s,
            "spawned_by_step": spawned_by_step,
        })
    return result


def generate_agent_insights(agent_summaries: list[dict]) -> list[str]:
    """Generate insight strings for multi-agent patterns."""
    if len(agent_summaries) <= 1:
        return []

    insights: list[str] = []
    total_tokens = sum(a["total_tokens"] for a in agent_summaries)
    if total_tokens <= 0:
        return insights

    for a in agent_summaries:
        share = a["total_tokens"] / total_tokens * 100
        if share > 60:
            insights.append(
                f"Agent {a['label']} consumed {share:.0f}% of total tokens"
            )
        if a["cache_efficiency_pct"] == 0 and a["total_tokens"] > 0:
            insights.append(
                f"Agent {a['label']} has 0% cache efficiency — no prompt caching"
            )
        if a["step_count"] > 0 and a["error_count"] / a["step_count"] > 0.10:
            rate = a["error_count"] / a["step_count"] * 100
            insights.append(
                f"Agent {a['label']} has {rate:.0f}% error rate ({a['error_count']} errors in {a['step_count']} steps)"
            )
        tool_share = a["tool_call_count"] / max(1, sum(x["tool_call_count"] for x in agent_summaries)) * 100
        if tool_share > 70 and len(agent_summaries) > 1:
            insights.append(
                f"Agent {a['label']} made {tool_share:.0f}% of all tool calls"
            )

    return insights
