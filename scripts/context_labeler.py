"""
Context labeler — LLM-based context utilization rating of trajectory turns.

Reads a trajectory file (Claude Code JSON, OpenCode JSON, CodeArts JSON,
or Lingxi .log), extracts assistant steps, and labels each turn with a
**context utilization ratio** that measures what fraction of the
accumulated conversation context was actually important/useful for the
agent's action in that specific turn.

This is a forward-looking metric, NOT backward-looking goal relevance:

  For each assistant turn, the labeler:
    1. Builds a compact timeline of ALL prior conversation turns
       (accumulated context — user messages, prior assistant actions,
       tool calls, compaction events).
    2. Sends that timeline + the current turn's reasoning & tool calls
       to the LLM.
    3. The LLM returns a utilization_ratio (0.0-1.0) representing the
       fraction of accumulated context that was actually needed for
       this turn's action, along with summaries of what was important
       vs what was noise.

The ratio naturally decreases over time as context grows — each
individual turn typically only needs a fraction of the total available
information. Compaction events may cause the ratio to jump back up.

LLM configuration is read from .env (LABEL_BASE_URL, LABEL_API_KEY,
LABEL_MODEL, LABEL_TEMPERATURE, LABEL_MAX_TOKENS).  CLI flags override
.env values.

Usage:
    python scripts/context_labeler.py samples/op_trajectory.json

    python scripts/context_labeler.py trajectory.json \
        --output context_labeled.json \
        --model vibe-coding \
        --base-url https://api.example.com \
        --api-key sk-xxx
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import step_labeler

# ── Reuse step_labeler's .env loader (already executed on import) ───────
_env_float = step_labeler._env_float
_env_int = step_labeler._env_int

# ── Constants ───────────────────────────────────────────────────────────

SCHEMA_VERSION = "context_utilization.v1"
_CONFIDENCE_LEVELS = {"high", "medium", "low"}


# ── Accumulated context builder ─────────────────────────────────────────


def _short_tool_desc(tc: dict) -> str:
    """Build a compact description of a single tool call."""
    name = tc.get("tool_name", "?")
    inp = tc.get("input", {})
    if not isinstance(inp, dict):
        inp = {}
    if name == "read":
        fp = inp.get("filePath") or inp.get("file_path") or inp.get("path") or ""
        short = str(fp).split("/")[-1] if fp else "?"
        return f"read:{short}"
    if name in ("write", "edit"):
        fp = inp.get("filePath") or inp.get("file_path") or inp.get("path") or ""
        short = str(fp).split("/")[-1] if fp else "?"
        return f"{name}:{short}"
    if name == "bash":
        cmd = str(inp.get("command", ""))[:50]
        return f"bash:{cmd}"
    if name == "grep":
        pattern = str(inp.get("pattern", ""))[:30]
        return f"grep:{pattern}"
    if name == "glob":
        pattern = str(inp.get("pattern", ""))[:30]
        return f"glob:{pattern}"
    if name in ("task", "subagent"):
        return f"task:{str(inp.get('description', ''))[:40]}"
    return name


def build_accumulated_context(
    all_steps: list[dict],
    step_index: int,
    max_chars: int = 20000,
) -> tuple[str, list[dict]]:
    """Build a compact timeline of prior conversation turns.

    When a compaction event occurs, all prior turns are compressed into a
    summary by the system and removed from the agent's context window.
    Therefore, only turns AFTER the most recent compaction are included.

    Returns (content_string, turn_info) where turn_info is a list of
    {"index": step_index, "role": role, "label": short_label} for each
    actual turn (user/assistant), excluding compaction markers.
    """
    lines: list[str] = []
    turn_info: list[dict] = []
    for s in all_steps:
        if s.get("index") == step_index:
            break

        role = s.get("role", "")
        idx = s.get("index", "?")

        for p in s.get("parts", []):
            if p.get("type") == "compaction":
                lines = [
                    f"[COMPACTION at step {idx}] Prior context was compressed "
                    f"into a summary by the system. The turns above are no "
                    f"longer in the agent's context window."
                ]
                turn_info = []

        if role == "user":
            text = (s.get("text_preview", "") or "")[:200]
            if text:
                lines.append(f"[User] {text}")
                turn_info.append({"index": idx, "role": "user", "label": f"User {idx}"})
        elif role == "assistant":
            agent = s.get("agent", "") or s.get("agent_id", "")
            agent_short = agent[:25] if agent else "main"
            text = (s.get("text_preview", "") or "")[:150]

            tool_calls = s.get("tool_calls", [])
            tool_parts = [_short_tool_desc(tc) for tc in tool_calls]
            tool_str = f" tools=[{', '.join(tool_parts)}]" if tool_parts else ""

            line = f"[Step {idx}] [{agent_short}]{tool_str}"
            if text:
                line += f" {text}"
            lines.append(line)
            turn_info.append({
                "index": idx, "role": "assistant",
                "label": f"Step {idx}",
                "agent": agent_short,
                "tools": [tc.get("tool_name", "?") for tc in tool_calls],
            })

    content = "\n".join(lines)
    if len(content) > max_chars:
        truncated_msg = "...[earlier context truncated]...\n"
        budget = max_chars - len(truncated_msg)
        content = truncated_msg + content[-budget:]
    return content, turn_info


# ── Prompt builders ─────────────────────────────────────────────────────


def build_rating_system_prompt() -> str:
    return """You are an expert at analyzing AI agent trajectories and evaluating context efficiency.

For each turn in an agent's conversation, you will receive:
1. ACCUMULATED CONTEXT: A timeline of all prior conversation turns
2. CURRENT TURN: The agent's reasoning and actions in this specific turn

Your task: Assign a utilization ratio (0.0-1.0) to EACH prior turn based on how
important that turn's information is for the agent's action in the current turn.

- IMPORTANT context = information the agent directly used or needed to decide
  what to do in this turn (file paths discovered earlier, prior findings,
  task instructions, error messages being debugged, relevant search results, etc.)
- NOISE = information that was available but not needed for this specific turn
  (unrelated file reads, completed sub-tasks no longer relevant, irrelevant
  search results, reference docs not needed for this action, etc.)

REALISTIC EXPECTATIONS:
- Early turns (few prior steps) will have HIGH ratios — most available context
  is relevant because there is little of it.
- Later turns (many prior steps) will have LOWER ratios — each turn typically
  only needs a fraction of the total accumulated context. This is normal and
  expected, NOT a failure.
- After a compaction event, the accumulated context RESETS — only turns after
  the compaction are included. The compaction marker indicates that prior
  context was compressed into a summary and is no longer available in full.
- A low ratio does NOT mean the context was "wasted" — it means the agent
  didn't need it for THIS specific turn. The context may have been needed
  for earlier turns.

HOW TO ASSESS (be brief — do NOT analyze each turn one by one):
1. Look at what the agent is doing in the current turn (reasoning + tool calls).
2. Identify which prior turns are directly relevant vs. not needed.
3. Assign a ratio to each prior turn: 0.0=irrelevant, 0.25=low, 0.5=moderate,
   0.75=high, 1.0=essential.
4. Keep your reasoning to 2-3 sentences total, NOT a per-turn breakdown.

Respond with ONLY this JSON object:
{
  "utilization_ratios": [0.1, 0.05, 0.8, ...],
  "important_context_summary": "Which prior context was needed and why (2-3 sentences)",
  "noise_context_summary": "Which prior context was not needed (2-3 sentences)",
  "reasoning": "Your overall analysis of why these ratios are appropriate (2-3 sentences)",
  "confidence": "high|medium|low"
}

The utilization_ratios array must have exactly one value per prior turn in the
accumulated context, in order. For example, if there are 10 prior turns, the
array must have 10 values.

- confidence: high = you have enough context to be sure;
  medium = reasonable but some uncertainty;
  low = limited context about this turn's needs."""



def build_rating_message(
    step: dict,
    accumulated_context: str,
    max_chars: int = 30000,
    compaction_context: str = "",
    expected_ratios: int | None = None,
) -> str:
    """Build the user message for a single context-utilization rating call."""
    parts = []

    parts.append(f"ACCUMULATED CONTEXT (all prior turns):\n{accumulated_context}")

    if compaction_context:
        parts.append(compaction_context)

    parts.append(
        "CURRENT TURN TO EVALUATE:\n"
        + step_labeler.build_step_message(step, max_chars=6000)
    )

    if expected_ratios is not None:
        parts.append(
            f"IMPORTANT: The accumulated context contains exactly {expected_ratios} "
            f"prior turns. You MUST return exactly {expected_ratios} values in the "
            f"utilization_ratios array — one per prior turn, in order. "
            f"Do not include more or fewer than {expected_ratios} values."
        )

    content = "\n\n".join(parts)
    if len(content) > max_chars:
        separator = "\n\n[... truncated ...]\n\n"
        budget = max_chars - len(separator)
        if budget <= 0:
            content = content[:max_chars]
        else:
            head_chars = (budget * 3) // 4
            tail_chars = budget - head_chars
            content = content[:head_chars] + separator + content[-tail_chars:]
    return content


# ── File-read enrichment ────────────────────────────────────────────────


def extract_file_reads(step: dict) -> list[dict]:
    """Extract read tool calls from a step, returning normalized dicts."""
    reads = []
    for tc in step.get("tool_calls", []):
        tool_name = tc.get("tool_name", "")
        if tool_name != "read":
            continue
        inp = tc.get("input", {})
        if not isinstance(inp, dict):
            inp = {}
        file_path = inp.get("filePath") or inp.get("file_path") or inp.get("path") or ""
        status = tc.get("status", "?")
        error = tc.get("error") or ""
        output = tc.get("output", "") or ""
        reads.append({
            "tool_name": tool_name,
            "file_path": str(file_path) if file_path else "",
            "status": status,
            "error": str(error) if error else "",
            "output": str(output) if output else "",
        })
    return reads


def read_file_from_disk(
    file_path: str,
    max_lines: int = 80,
    max_chars: int = 3000,
) -> tuple[bool, str]:
    """Try to read a file from disk."""
    if not file_path or not os.path.isfile(file_path):
        return False, ""
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            lines = []
            total_chars = 0
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                if total_chars + len(line) > max_chars:
                    remaining = max_chars - total_chars
                    if remaining > 0:
                        lines.append(line[:remaining])
                    lines.append("... [truncated]\n")
                    break
                lines.append(line)
                total_chars += len(line)
            snippet = "".join(lines)
            return True, snippet
    except (OSError, UnicodeDecodeError):
        return False, ""


def enrich_file_reads(
    reads: list[dict],
    file_snippet_lines: int = 80,
    max_file_chars: int = 3000,
) -> list[dict]:
    """Enrich file read dicts with on-disk content and tool-output snippets."""
    enriched = []
    for r in reads:
        path = r.get("file_path", "")
        error = r.get("error", "")
        tool_output = r.get("output", "")
        entry: dict[str, Any] = {"path": path}

        if error:
            entry["exists_on_disk"] = False
            entry["error"] = error
            entry["snippet"] = ""
            entry["tool_output"] = ""
        else:
            exists, snippet = read_file_from_disk(
                path, max_lines=file_snippet_lines, max_chars=max_file_chars
            )
            entry["exists_on_disk"] = exists
            entry["snippet"] = snippet
            entry["tool_output"] = tool_output[:1000] if not exists else ""

        enriched.append(entry)
    return enriched


# ── File-read history (per-agent tracking) ──────────────────────────────


def build_file_read_history(
    all_steps: list[dict],
    target_step_index: int,
    target_agent: str = "",
) -> tuple[dict[str, int], dict[str, int]]:
    """Build cumulative file-read counts before the target step."""
    agent_history: dict[str, int] = {}
    global_history: dict[str, int] = {}
    for step in all_steps:
        if step.get("index") == target_step_index:
            break
        if step.get("role") != "assistant":
            continue
        step_agent = step.get("agent", "") or step.get("agent_id", "")
        for tc in step.get("tool_calls", []):
            if tc.get("tool_name") != "read":
                continue
            inp = tc.get("input", {})
            if not isinstance(inp, dict):
                continue
            fp = inp.get("filePath") or inp.get("file_path") or inp.get("path") or ""
            if fp:
                fp = str(fp)
                global_history[fp] = global_history.get(fp, 0) + 1
                if step_agent == target_agent:
                    agent_history[fp] = agent_history.get(fp, 0) + 1
    return agent_history, global_history


# ── Response parsing & normalization ────────────────────────────────────


def parse_json_object(text: str) -> dict:
    """Parse a model response that should contain a single JSON object."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def normalize_confidence(value: str) -> str:
    conf = str(value).strip().lower()
    if conf in _CONFIDENCE_LEVELS:
        return conf
    return "medium"


def normalize_files_read(enriched_files: list[dict]) -> list[dict]:
    """Build files_read output from enriched file data (no LLM per-file rating)."""
    result = []
    for ef in enriched_files:
        result.append({
            "path": ef.get("path", ""),
            "exists_on_disk": ef.get("exists_on_disk", False),
            "prior_reads_by_agent": ef.get("prior_reads_by_agent", 0),
        })
    return result


def normalize_rating(raw: dict, enriched_files: list[dict]) -> dict:
    """Normalize the LLM utilization rating response into a validated dict."""
    ratios_raw = raw.get("utilization_ratios", raw.get("utilization_ratio", 0.0))

    per_turn_ratios: list[float] = []
    if isinstance(ratios_raw, list):
        for r in ratios_raw:
            try:
                per_turn_ratios.append(max(0.0, min(1.0, float(r))))
            except (ValueError, TypeError):
                per_turn_ratios.append(0.0)
    else:
        try:
            per_turn_ratios.append(max(0.0, min(1.0, float(ratios_raw))))
        except (ValueError, TypeError):
            per_turn_ratios.append(0.0)

    if per_turn_ratios:
        overall_ratio = sum(per_turn_ratios) / len(per_turn_ratios)
    else:
        overall_ratio = 0.0

    confidence = normalize_confidence(raw.get("confidence", ""))
    important = str(raw.get("important_context_summary", ""))[:600]
    noise = str(raw.get("noise_context_summary", ""))[:600]
    reasoning = str(raw.get("reasoning", ""))[:600]
    files = normalize_files_read(enriched_files)

    return {
        "utilization_ratio": round(overall_ratio, 4),
        "utilization_percentage": int(round(overall_ratio * 100)),
        "per_turn_ratios": [round(r, 4) for r in per_turn_ratios],
        "important_context_summary": important,
        "noise_context_summary": noise,
        "reasoning": reasoning,
        "confidence": confidence,
        "files_read": files,
    }


# ── Compaction detection ────────────────────────────────────────────────


def detect_compaction_events(all_steps: list[dict]) -> tuple[list[dict], set[int]]:
    """Scan parsed steps for compaction events."""
    events: list[dict] = []
    post_compaction_indices: set[int] = set()

    for i, step in enumerate(all_steps):
        for p in step.get("parts", []):
            if p.get("type") != "compaction":
                continue
            raw = p.get("raw", {})
            if not isinstance(raw, dict):
                raw = {}
            time_obj = raw.get("time", {})
            if not isinstance(time_obj, dict):
                time_obj = {}
            events.append({
                "step_index": step.get("index", i),
                "auto": bool(raw.get("auto", False)),
                "overflow": bool(raw.get("overflow", False)),
                "time_created_ms": time_obj.get("created"),
                "time_updated_ms": time_obj.get("updated"),
                "tail_start_id": raw.get("tail_start_id", ""),
            })
            for j in range(i + 1, len(all_steps)):
                if all_steps[j].get("role") == "assistant":
                    post_compaction_indices.add(all_steps[j].get("index", j))
                    break
            break

    return events, post_compaction_indices


# ── Step-level rating ───────────────────────────────────────────────────


def is_empty_step(step: dict) -> bool:
    """Check if a step has no text, no tool calls, and no reasoning."""
    has_text = bool((step.get("text_preview", "") or "").strip())
    has_tools = bool(step.get("tool_calls"))
    has_reasoning = any(
        p.get("type") == "reasoning" and p.get("text")
        for p in step.get("parts", [])
    )
    return not has_text and not has_tools and not has_reasoning


def rate_step(
    step: dict,
    all_steps: list[dict],
    file_snippet_lines: int,
    *,
    base_url: str,
    api_key: str,
    model: str,
    provider: str = "openai",
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_content_chars: int = 30000,
    max_context_chars: int = 20000,
    max_retries: int = 2,
    retry_delay: float = 1.0,
    debug: bool = False,
    post_compaction_indices: set[int] | None = None,
) -> dict:
    """Rate a single assistant step for context utilization."""
    reads = extract_file_reads(step)
    enriched_files = enrich_file_reads(reads, file_snippet_lines=file_snippet_lines)

    step_idx = step.get("index")
    step_agent = step.get("agent", "") or step.get("agent_id", "")
    agent_read_history, _ = build_file_read_history(
        all_steps, step_idx, target_agent=step_agent,
    )

    for ef in enriched_files:
        ef["prior_reads_by_agent"] = agent_read_history.get(ef.get("path", ""), 0)

    accumulated_context, prior_turns = build_accumulated_context(
        all_steps, step_idx, max_chars=max_context_chars,
    )

    compaction_context = ""
    if post_compaction_indices and step_idx in post_compaction_indices:
        compaction_context = (
            "COMPACTION NOTE:\n"
            "This step occurred immediately after a context compaction event. "
            "The conversation history was compressed into a summary. The "
            "accumulated context above starts from the compaction point — "
            "all prior turns have been replaced by the compressed summary "
            "and are no longer in the agent's context window."
        )

    system_prompt = build_rating_system_prompt()

    raw = {}
    last_exc: Exception | None = None
    n_prior = len(prior_turns)
    user_message = build_rating_message(
        step=step,
        accumulated_context=accumulated_context,
        max_chars=max_content_chars,
        compaction_context=compaction_context,
    )

    if debug:
        print(f"[debug] Rating prompt for step {step_idx}:\n"
              f"{user_message[:2000]}\n[...]", file=sys.stderr)

    for attempt in range(max_retries):
        try:
            if attempt > 0 and n_prior > 0:
                user_message = build_rating_message(
                    step=step,
                    accumulated_context=accumulated_context,
                    max_chars=max_content_chars,
                    compaction_context=compaction_context,
                    expected_ratios=n_prior,
                )
            response = step_labeler.call_llm(
                base_url=base_url,
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_message=user_message,
                provider=provider,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"enable_thinking": True},
            )
            if debug:
                print(f"[debug] LLM response (attempt {attempt+1}): {response[:500]}", file=sys.stderr)
            raw = parse_json_object(response)
            if n_prior > 0:
                ratios_raw = raw.get("utilization_ratios", raw.get("utilization_ratio", 0.0))
                got = len(ratios_raw) if isinstance(ratios_raw, list) else 1
                if got != n_prior:
                    print(f"[warn] Step {step_idx}: expected {n_prior} ratios, got {got} "
                          f"(attempt {attempt+1}/{max_retries})", file=sys.stderr)
                    if attempt < max_retries - 1:
                        if retry_delay > 0:
                            time.sleep(retry_delay)
                        continue
            break
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1 and retry_delay > 0:
                time.sleep(retry_delay)

    if not raw and last_exc:
        print(f"[error: {last_exc}]", file=sys.stderr)
        rating = {
            "utilization_ratio": 0.0,
            "utilization_percentage": 0,
            "per_turn_ratios": [],
            "prior_turns": prior_turns,
            "important_context_summary": "",
            "noise_context_summary": "",
            "reasoning": f"LLM call failed: {last_exc}",
            "confidence": "low",
            "files_read": normalize_files_read(enriched_files),
        }
    else:
        rating = normalize_rating(raw, enriched_files)
        rating["prior_turns"] = prior_turns
        ptr = rating.get("per_turn_ratios", [])
        n = len(prior_turns)
        if len(ptr) > n:
            rating["per_turn_ratios"] = ptr[:n]
        elif len(ptr) < n:
            rating["per_turn_ratios"] = ptr + [0.0] * (n - len(ptr))
        if rating["per_turn_ratios"]:
            rating["utilization_ratio"] = round(
                sum(rating["per_turn_ratios"]) / len(rating["per_turn_ratios"]), 4
            )
            rating["utilization_percentage"] = int(
                round(rating["utilization_ratio"] * 100)
            )

    return rating


# ── Main pipeline ───────────────────────────────────────────────────────


def label_context(
    trajectory_path: str,
    output_path: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    provider: str = "openai",
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_content_chars: int = 30000,
    max_context_chars: int = 20000,
    file_snippet_lines: int = 80,
    delay: float = 0.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    limit: int | None = None,
    debug: bool = False,
) -> None:
    """Label assistant steps in a trajectory and write output JSON."""
    print(f"Loading trajectory: {trajectory_path}", file=sys.stderr)

    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from trajviz.insight.loaders import load_trajectory
    from trajviz.insight.parser import parse_steps

    raw = load_trajectory(trajectory_path)
    if "_error" in raw:
        raise ValueError(f"Failed to load trajectory: {raw['_error']}")
    all_steps = parse_steps(raw)
    assistant_steps = [s for s in all_steps if s.get("role") == "assistant"]
    print(f"Found {len(assistant_steps)} assistant steps", file=sys.stderr)

    if not assistant_steps:
        print("No assistant steps found — nothing to label.", file=sys.stderr)
        return

    compaction_events, post_compaction_indices = detect_compaction_events(all_steps)
    if compaction_events:
        print(f"Detected {len(compaction_events)} compaction event(s) at steps: "
              f"{[e['step_index'] for e in compaction_events]}", file=sys.stderr)

    steps_to_rate = assistant_steps if not limit else assistant_steps[:limit]
    labeled_steps: list[dict] = []

    def _write_output() -> None:
        ratio_sum = sum(s["utilization_ratio"] for s in labeled_steps)
        avg_ratio = round(ratio_sum / len(labeled_steps), 4) if labeled_steps else 0.0
        output = {
            "schema_version": SCHEMA_VERSION,
            "trajectory_file": os.path.abspath(trajectory_path),
            "labeled_at": datetime.now(UTC).isoformat(),
            "model": model,
            "compaction_events": compaction_events,
            "total_steps": len(labeled_steps),
            "avg_utilization_ratio": avg_ratio,
            "avg_utilization_percentage": int(round(avg_ratio * 100)),
            "steps": labeled_steps,
        }
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, output_path)

    for i, step in enumerate(steps_to_rate):
        step_idx = step.get("index", i)
        print(
            f"Rating step {i + 1}/{len(steps_to_rate)} (idx {step_idx})...",
            file=sys.stderr, end=" ", flush=True,
        )

        if is_empty_step(step):
            print("(empty step — skipped)", file=sys.stderr)
            reads = extract_file_reads(step)
            enriched_files = enrich_file_reads(reads, file_snippet_lines=file_snippet_lines)
            rating = {
                "utilization_ratio": 0.0,
                "utilization_percentage": 0,
                "per_turn_ratios": [],
                "prior_turns": [],
                "important_context_summary": "",
                "noise_context_summary": "",
                "reasoning": "Empty step with no text, tool calls, or reasoning.",
                "confidence": "high",
                "files_read": normalize_files_read(enriched_files),
            }
        else:
            rating = rate_step(
                step=step,
                all_steps=all_steps,
                file_snippet_lines=file_snippet_lines,
                base_url=base_url,
                api_key=api_key,
                model=model,
                provider=provider,
                temperature=temperature,
                max_tokens=max_tokens,
                max_content_chars=max_content_chars,
                max_context_chars=max_context_chars,
                max_retries=max_retries,
                retry_delay=retry_delay,
                debug=debug,
                post_compaction_indices=post_compaction_indices,
            )

        pct = rating["utilization_percentage"]
        print(f"{pct}%", file=sys.stderr)

        tool_names = [tc.get("tool_name", "?") for tc in step.get("tool_calls", [])]
        tokens = step.get("tokens", {})

        labeled_steps.append({
            "index": step_idx,
            "role": "assistant",
            "utilization_ratio": rating["utilization_ratio"],
            "utilization_percentage": rating["utilization_percentage"],
            "per_turn_ratios": rating.get("per_turn_ratios", []),
            "prior_turns": rating.get("prior_turns", []),
            "important_context_summary": rating["important_context_summary"],
            "noise_context_summary": rating["noise_context_summary"],
            "reasoning": rating["reasoning"],
            "confidence": rating["confidence"],
            "files_read": rating["files_read"],
            "is_post_compaction": step_idx in post_compaction_indices,
            "time_created_ms": step.get("time_created_ms"),
            "time_completed_ms": step.get("time_completed_ms"),
            "duration_s": step.get("duration"),
            "tokens_total": tokens.get("total", 0),
            "tool_calls": tool_names,
            "finish": step.get("finish", ""),
            "agent": step.get("agent", "") or step.get("agent_id", ""),
            "model_id": step.get("model_id", ""),
            "text_preview": (step.get("text_preview", "") or "")[:200],
            "session_id": step.get("session_id", ""),
            "is_sub_agent": step.get("is_sub_agent", False),
        })

        _write_output()

        if delay > 0 and i < len(steps_to_rate) - 1:
            time.sleep(delay)

    ratio_sum = sum(s["utilization_ratio"] for s in labeled_steps)
    avg_pct = int(round(ratio_sum / len(labeled_steps) * 100)) if labeled_steps else 0
    print(
        f"\nDone: {len(labeled_steps)} steps labeled. "
        f"Avg utilization: {avg_pct}%. "
        f"Output: {output_path}",
        file=sys.stderr,
    )


# ── CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate context utilization ratio for each turn in a trajectory using an LLM.",
    )
    parser.add_argument("input", help="Path to trajectory file (JSON or Lingxi .log)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON path (default: <input>_context_labeled.json)")
    parser.add_argument("--model", default=None,
                        help="LLM model (overrides LABEL_MODEL env)")
    parser.add_argument("--base-url", default=None,
                        help="LLM API base URL (overrides LABEL_BASE_URL env)")
    parser.add_argument("--api-key", default=None,
                        help="LLM API key (overrides LABEL_API_KEY env)")
    parser.add_argument("--provider", default=None,
                        help="LLM provider: openai or anthropic (overrides LABEL_PROVIDER env)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (overrides LABEL_TEMPERATURE env)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max response tokens (overrides LABEL_MAX_TOKENS env)")
    parser.add_argument("--max-content-chars", type=int, default=30000,
                        help="Max chars per step prompt sent to LLM (default: 30000)")
    parser.add_argument("--max-context-chars", type=int, default=20000,
                        help="Max chars for accumulated context timeline (default: 20000)")
    parser.add_argument("--file-snippet-lines", type=int, default=80,
                        help="Lines of file content to read from disk per file (default: 80)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Delay in seconds between LLM calls (default: 0)")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max LLM call attempts per step (default: 2)")
    parser.add_argument("--retry-delay", type=float, default=1.0,
                        help="Delay between retries in seconds (default: 1.0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only rate the first N assistant steps (default: all)")
    parser.add_argument("--debug", action="store_true",
                        help="Show full LLM prompts and responses for debugging")

    args = parser.parse_args()

    base_url = args.base_url or os.getenv("LABEL_BASE_URL", "")
    api_key = args.api_key or os.getenv("LABEL_API_KEY", "")
    model = args.model or os.getenv("LABEL_MODEL", "")
    provider = args.provider or os.getenv("LABEL_PROVIDER", "openai")
    temperature = args.temperature
    if temperature is None:
        temperature = _env_float("LABEL_TEMPERATURE", 0.3)
    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = _env_int("LABEL_MAX_TOKENS", 4096)

    if not base_url:
        print("Error: LABEL_BASE_URL not set (use --base-url or .env)", file=sys.stderr)
        sys.exit(1)
    if not api_key:
        print("Error: LABEL_API_KEY not set (use --api-key or .env)", file=sys.stderr)
        sys.exit(1)
    if not model:
        print("Error: LABEL_MODEL not set (use --model or .env)", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        inp = Path(args.input)
        output_path = str(inp.parent / f"{inp.stem}_context_labeled.json")

    label_context(
        trajectory_path=args.input,
        output_path=output_path,
        base_url=base_url,
        api_key=api_key,
        model=model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        max_content_chars=args.max_content_chars,
        max_context_chars=args.max_context_chars,
        file_snippet_lines=args.file_snippet_lines,
        delay=args.delay,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        limit=args.limit,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
