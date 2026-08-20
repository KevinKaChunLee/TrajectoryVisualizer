"""N-run scorecard + behavioral comparison for the same task.

Phase 1 — per-run metrics side-by-side.
Phase 2 — all-pairs alignment F1, consensus vs unique actions/files, tool /
MCP / skill coverage, and waste patterns relative to a baseline (first
successfully loaded run).
"""

from __future__ import annotations

import html
import math
import os
from collections import Counter
from typing import Any

from trajviz.converge.alignment import (
    _detect_success,
    align_trajectories,
    compute_alignment_metrics,
)
from trajviz.converge.canonical import (
    CanonicalAction,
    assign_effect_labels,
    canonicalize_steps,
)
from trajviz.converge.divergence import classify_divergences, compute_pattern_costs
from trajviz.insight.diagnostics import (
    context_pressure_series,
    context_pressure_stats,
)
from trajviz.insight.formatting import wall_clock_fmt
from trajviz.insight.loaders import detect_format, load_trajectory
from trajviz.insight.parser import build_message_metrics, compute_metrics, parse_steps

# Pattern labels shown in the UI (omit low-signal names when empty).
_PATTERN_LABELS = {
    "reverted_and_rewritten": "reverted writes",
    "iterative_refinement": "iterative refinement",
    "broad_exploration": "broad exploration",
    "error_recovery_overhead": "error recovery",
    "redundant_search": "redundant search",
    "ordering_inefficiency": "reordered vs baseline",
    "dead_end_branch": "dead-end branch",
    "premature_validation": "premature validation",
}


def _file_path(entry: Any) -> str | None:
    """Normalize a Gradio File value (str, file-like, or None) to a path."""
    if entry is None:
        return None
    if isinstance(entry, str):
        return entry
    name = getattr(entry, "name", None)
    return name if isinstance(name, str) and name else None


def normalize_run_paths(files: Any) -> list[str]:
    """Flatten Gradio single/multi file uploads into existing path strings."""
    if files is None:
        return []
    if not isinstance(files, (list, tuple)):
        files = [files]
    paths: list[str] = []
    seen: set[str] = set()
    for entry in files:
        path = _file_path(entry)
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def default_run_label(path: str) -> str:
    """Short label from basename without extension."""
    base = os.path.basename(path)
    stem, _ext = os.path.splitext(base)
    return stem or base


def _action_signature(action: CanonicalAction) -> tuple[str, str] | None:
    """Stable (type, target) key for action coverage.

    Skips REASON, empty targets, and FILE_READ/FILE_WRITE (those belong in
    file coverage with read/write counts).
    """
    if action.action_type in ("", "REASON", "FILE_READ", "FILE_WRITE"):
        return None
    target = (action.target or "").strip()
    if not target and action.action_type != "AGENT_SPAWN":
        return None
    if action.action_type == "AGENT_SPAWN":
        return ("AGENT_SPAWN", "*")
    return (action.action_type, target)


def _signature_set(actions: list[CanonicalAction]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for action in actions:
        sig = _action_signature(action)
        if sig is not None:
            out.add(sig)
    return out


def _file_touch_map(actions: list[CanonicalAction]) -> dict[str, dict[str, int]]:
    """path → {read, write} counts for one run's canonical file actions."""
    touches: dict[str, dict[str, int]] = {}
    for action in actions:
        if action.action_type not in ("FILE_READ", "FILE_WRITE") or not action.target:
            continue
        cell = touches.setdefault(action.target, {"read": 0, "write": 0})
        if action.action_type == "FILE_READ":
            cell["read"] += 1
        else:
            cell["write"] += 1
    return touches


def _short_path(path: str, limit: int = 56) -> str:
    base = os.path.basename(path) or path
    if len(path) <= limit:
        return path
    if len(base) + 3 >= limit:
        return "…" + base[-(limit - 1) :]
    return "…" + path[-(limit - 1) :]


def _fmt_signature(sig: tuple[str, str]) -> str:
    atype, target = sig
    if atype == "AGENT_SPAWN":
        return "AGENT_SPAWN"
    short = target if len(target) <= 64 else target[:61] + "…"
    return f"{atype}({short})"


_SKILL_TOOL_NAMES = frozenset(
    {
        "skill",
        "skills",
        "Skill",
        "Skills",
        "invoke_skill",
        "InvokeSkill",
        "run_skill",
        "RunSkill",
    }
)


def _parse_skill_name(tool_name: str, tool_input: Any) -> str | None:
    """Return skill id when this call invokes a Skill tool."""
    name_l = (tool_name or "").lower()
    if name_l not in {n.lower() for n in _SKILL_TOOL_NAMES}:
        return None
    if isinstance(tool_input, dict):
        for key in ("skill", "name", "skill_name", "skillName", "id", "skill_id"):
            val = tool_input.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    if isinstance(tool_input, str) and tool_input.strip():
        return tool_input.strip()
    return "(unnamed skill)"


def extract_capability_usage(steps: list[dict]) -> dict[str, Counter[str]]:
    """Count tools and skill triggers from parsed steps.

    - ``tools``: all tool names (includes MCP and Skill as tool types)
    - ``skills``: skill ids from Skill-tool invocations
    """
    tools: Counter[str] = Counter()
    skills: Counter[str] = Counter()
    for step in steps:
        for tc in step.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            name = str(tc.get("tool_name") or "").strip()
            if not name or name == "?":
                continue
            tools[name] += 1
            skill = _parse_skill_name(name, tc.get("input"))
            if skill:
                skills[skill] += 1
    return {"tools": tools, "skills": skills}


def _coverage_kind(n_runs: int, thresh: int) -> str:
    if n_runs >= thresh:
        return "consensus"
    if n_runs == 1:
        return "unique"
    return "partial"


def _build_count_matrix(
    counts_by_run: dict[str, Counter[str]],
    run_ids: list[str],
    thresh: int,
    *,
    limit: int = 50,
) -> tuple[list[dict], int]:
    """Build presence/count matrix rows for string keys across runs."""
    key_counts: Counter[str] = Counter()
    for rid in run_ids:
        for key in counts_by_run.get(rid, ()):
            key_counts[key] += 1
    kind_rank = {"consensus": 0, "partial": 1, "unique": 2}
    rows: list[dict] = []
    for key, n_runs in key_counts.items():
        cells: dict[str, dict] = {}
        for rid in run_ids:
            count = int(counts_by_run.get(rid, Counter()).get(key, 0))
            cells[rid] = {"present": count > 0, "count": count}
        rows.append(
            {
                "key": key,
                "short": _short_path(key, limit=64),
                "kind": _coverage_kind(n_runs, thresh),
                "n_runs": n_runs,
                "cells": cells,
            }
        )
    rows.sort(key=lambda r: (kind_rank[r["kind"]], -r["n_runs"], r["key"].lower()))
    return rows[:limit], len(rows)


def build_run_scorecard_row(
    raw: dict,
    *,
    path: str = "",
    label: str | None = None,
    run_id: str | None = None,
    steps: list[dict] | None = None,
) -> dict:
    """Compute one scorecard row from an already-loaded trajectory dict."""
    if "_error" in raw:
        return {
            "run_id": run_id or default_run_label(path) or "run",
            "label": label or default_run_label(path) or "run",
            "path": path,
            "error": str(raw.get("_error") or "load failed"),
            "finished": False,
            "steps": None,
            "wall_clock_s": None,
            "wall_clock_fmt": "—",
            "tokens": None,
            "tool_calls": None,
            "tool_success_pct": None,
            "peak_occupancy": None,
            "peak_pct": None,
            "compactions": None,
            "format": None,
        }

    if steps is None:
        steps = parse_steps(raw)
    message_rows = build_message_metrics(steps)
    metrics = compute_metrics(steps, raw, message_rows=message_rows)
    wall_s, wall_fmt = wall_clock_fmt(metrics)
    pressure = context_pressure_stats(
        context_pressure_series(steps, raw=raw),
    )
    fmt = detect_format(raw)
    rid = run_id or default_run_label(path) or "run"
    return {
        "run_id": rid,
        "label": label or rid,
        "path": path,
        "error": None,
        "finished": bool(_detect_success(steps)),
        "steps": int(metrics.get("total_steps") or 0),
        "wall_clock_s": float(wall_s) if isinstance(wall_s, (int, float)) else None,
        "wall_clock_fmt": wall_fmt,
        "tokens": int((metrics.get("tokens") or {}).get("total") or 0),
        "tool_calls": int(metrics.get("tool_call_count") or 0),
        "tool_success_pct": float(metrics.get("tool_success_rate") or 0),
        "peak_occupancy": int(pressure.get("peak_occupancy") or 0),
        "peak_pct": pressure.get("peak_pct"),
        "compactions": int(pressure.get("compaction_count") or 0),
        "format": fmt if fmt != "unknown" else None,
    }


def _pair_f1(
    ref_actions: list[CanonicalAction],
    cmp_actions: list[CanonicalAction],
    *,
    fuzzy: bool = False,
) -> float:
    alignment = align_trajectories(ref_actions, cmp_actions, fuzzy_commands=fuzzy)
    metrics = compute_alignment_metrics(alignment, ref_actions, cmp_actions)
    return float(metrics.get("alignment_f1") or 0.0)


def _consensus_threshold(n: int) -> int:
    """Signatures in at least this many runs count as consensus."""
    if n <= 1:
        return 1
    return max(2, math.ceil(n / 2))


def build_behavioral_comparison(
    runs: list[dict],
    *,
    fuzzy: bool = False,
) -> dict:
    """All-pairs F1, consensus/unique signatures, baseline-relative patterns.

    Each ``runs`` entry needs ``run_id``, ``label``, and ``actions``
    (list[CanonicalAction]). Optional ``steps`` enables tool / MCP / skill
    coverage matrices. Baseline is the first run.
    """
    if len(runs) < 2:
        return {}

    ids = [r["run_id"] for r in runs]
    labels = {r["run_id"]: r["label"] for r in runs}
    actions_by_id = {r["run_id"]: r["actions"] for r in runs}
    baseline_id = ids[0]

    # All-pairs F1 matrix (symmetric; diagonal = 1)
    matrix: dict[str, dict[str, float]] = {i: {} for i in ids}
    for i, a in enumerate(ids):
        matrix[a][a] = 1.0
        for b in ids[i + 1 :]:
            f1 = _pair_f1(actions_by_id[a], actions_by_id[b], fuzzy=fuzzy)
            matrix[a][b] = f1
            matrix[b][a] = f1

    # Consensus / unique on action signatures and files
    sig_sets = {rid: _signature_set(actions_by_id[rid]) for rid in ids}
    sig_counts_by_run: dict[str, Counter[tuple[str, str]]] = {rid: Counter() for rid in ids}
    for rid, actions in actions_by_id.items():
        for action in actions:
            sig = _action_signature(action)
            if sig is not None:
                sig_counts_by_run[rid][sig] += 1

    touch_by_run = {rid: _file_touch_map(actions_by_id[rid]) for rid in ids}
    sig_counts: Counter[tuple[str, str]] = Counter()
    for sigs in sig_sets.values():
        for sig in sigs:
            sig_counts[sig] += 1

    all_paths: set[str] = set()
    for touches in touch_by_run.values():
        all_paths.update(touches)

    thresh = _consensus_threshold(len(ids))

    # Action coverage matrix (type + target × runs)
    kind_rank = {"consensus": 0, "partial": 1, "unique": 2}
    action_rows: list[dict] = []
    for sig, n_runs in sig_counts.items():
        atype, target = sig
        cells: dict[str, dict] = {}
        for rid in ids:
            count = int(sig_counts_by_run[rid].get(sig, 0))
            cells[rid] = {"present": count > 0, "count": count}
        action_rows.append(
            {
                "type": atype,
                "target": target,
                "short_target": _short_path(target, limit=48) if target != "*" else "",
                "label": _fmt_signature(sig),
                "kind": _coverage_kind(n_runs, thresh),
                "n_runs": n_runs,
                "cells": cells,
            }
        )
    action_rows.sort(
        key=lambda r: (kind_rank[r["kind"]], -r["n_runs"], r["type"], r["target"]),
    )
    action_matrix = action_rows[:60]

    file_rows: list[dict] = []
    for path in all_paths:
        cells_f: dict[str, dict[str, int]] = {}
        n_runs = 0
        for rid in ids:
            cell = touch_by_run[rid].get(path)
            if cell and (cell.get("read") or cell.get("write")):
                cells_f[rid] = {
                    "read": int(cell.get("read") or 0),
                    "write": int(cell.get("write") or 0),
                }
                n_runs += 1
            else:
                cells_f[rid] = {"read": 0, "write": 0}
        file_rows.append(
            {
                "path": path,
                "short": _short_path(path),
                "kind": _coverage_kind(n_runs, thresh),
                "n_runs": n_runs,
                "cells": cells_f,
            }
        )

    file_rows.sort(key=lambda r: (kind_rank[r["kind"]], -r["n_runs"], r["path"]))
    file_matrix = file_rows[:50]

    # Tools / skills from parsed steps (when available)
    usage_by_run = {r["run_id"]: extract_capability_usage(r.get("steps") or []) for r in runs}
    tool_counts = {rid: usage_by_run[rid]["tools"] for rid in ids}
    skill_counts = {rid: usage_by_run[rid]["skills"] for rid in ids}
    tool_matrix, tool_total = _build_count_matrix(tool_counts, ids, thresh, limit=50)
    skill_matrix, skill_total = _build_count_matrix(skill_counts, ids, thresh, limit=40)

    # Waste patterns: each non-baseline run vs baseline alignment extras
    patterns_by_run: dict[str, list[dict]] = {baseline_id: []}
    base_actions = actions_by_id[baseline_id]
    for rid in ids:
        if rid == baseline_id:
            continue
        cmp_actions = actions_by_id[rid]
        alignment = align_trajectories(base_actions, cmp_actions, fuzzy_commands=fuzzy)
        matched_cmp = {j for _, j in alignment["matched_pairs"]}
        extras = [
            cmp_actions[j]
            for j in alignment["extra"]
            if j < len(cmp_actions) and cmp_actions[j].action_type != "REASON"
        ]
        matched_actions = [cmp_actions[j] for j in sorted(matched_cmp) if j < len(cmp_actions)]
        patterns = classify_divergences(
            extras,
            matched_actions,
            cmp_actions,
            matched_pairs=alignment["matched_pairs"],
            reference_actions=base_actions,
        )
        compute_pattern_costs(patterns)
        by_type: Counter[str] = Counter()
        for p in patterns:
            by_type[p.get("type") or "unknown"] += 1
        patterns_by_run[rid] = [
            {
                "type": ptype,
                "label": _PATTERN_LABELS.get(ptype, ptype),
                "count": count,
            }
            for ptype, count in by_type.most_common()
        ]

    return {
        "run_ids": ids,
        "labels": labels,
        "baseline_run_id": baseline_id,
        "similarity": matrix,
        "consensus_threshold": thresh,
        "action_matrix": action_matrix,
        "action_matrix_total": len(action_rows),
        "file_matrix": file_matrix,
        "file_matrix_total": len(file_rows),
        "tool_matrix": tool_matrix,
        "tool_matrix_total": tool_total,
        "skill_matrix": skill_matrix,
        "skill_matrix_total": skill_total,
        "patterns_vs_baseline": patterns_by_run,
    }


def build_run_group_scorecard(
    paths: list[str],
    *,
    labels: list[str] | None = None,
    format_hint: str | None = None,
    fuzzy: bool = False,
    baseline_raw: dict | None = None,
    baseline_label: str | None = None,
) -> dict:
    """Load N trajectories, scorecard rows, and behavioral comparison.

    When ``baseline_raw`` is provided (e.g. the Overview trajectory), it is
    always the first run and the baseline for waste patterns. Uploaded
    ``paths`` are additional comparison runs — only one upload is required
    in that case.

    Returns ``{"rows", "behavior", "timeline_runs", "ok", "error"}``.
    """
    paths = list(paths or [])
    has_baseline = isinstance(baseline_raw, dict) and baseline_raw and "_error" not in baseline_raw
    # Need ≥2 runs total: baseline+≥1 path, or ≥2 paths with no baseline
    min_paths = 1 if has_baseline else 2
    if not has_baseline and not paths:
        return {
            "rows": [],
            "behavior": None,
            "timeline_runs": [],
            "ok": False,
            "error": (
                "Load a trajectory in Overview (baseline), then upload at least "
                "one more run — or upload two or more trajectories here."
            ),
        }
    if len(paths) < min_paths:
        if has_baseline:
            return {
                "rows": [],
                "behavior": None,
                "timeline_runs": [],
                "ok": False,
                "error": ("Upload at least one comparison trajectory (Overview is already included as the baseline)."),
            }
        return {
            "rows": [],
            "behavior": None,
            "timeline_runs": [],
            "ok": False,
            "error": "Upload at least two trajectories to build a run-group scorecard.",
        }

    rows: list[dict] = []
    loaded_runs: list[dict] = []
    used_ids: set[str] = set()

    def _append_raw(
        raw: dict,
        *,
        path: str,
        label: str | None,
        prefer_id: str | None = None,
    ) -> None:
        base_id = prefer_id or default_run_label(path) or f"run-{len(used_ids) + 1}"
        run_id = base_id
        suffix = 2
        while run_id in used_ids:
            run_id = f"{base_id}-{suffix}"
            suffix += 1
        used_ids.add(run_id)

        if "_error" in raw:
            rows.append(
                build_run_scorecard_row(
                    raw,
                    path=path,
                    label=label or run_id,
                    run_id=run_id,
                )
            )
            return

        steps = parse_steps(raw)
        row = build_run_scorecard_row(
            raw,
            path=path,
            label=label or run_id,
            run_id=run_id,
            steps=steps,
        )
        rows.append(row)
        actions = canonicalize_steps(steps)
        assign_effect_labels(actions, steps, None)
        loaded_runs.append(
            {
                "run_id": run_id,
                "label": row["label"],
                "actions": actions,
                "steps": steps,
            }
        )

    if has_baseline:
        src = str(baseline_raw.get("_source_path") or "")
        if baseline_label:
            blabel = baseline_label
        elif src:
            blabel = f"{default_run_label(src)} (baseline)"
        else:
            blabel = "Overview (baseline)"
        prefer_id = (default_run_label(src) if src else None) or "overview"
        _append_raw(
            baseline_raw,
            path=src or "(overview)",
            label=blabel,
            prefer_id=prefer_id,
        )

    for i, path in enumerate(paths):
        label = None
        # labels list aligns with uploaded paths only (not baseline)
        if labels and i < len(labels) and labels[i]:
            label = labels[i]
        raw = load_trajectory(path, format_hint=format_hint)
        _append_raw(raw, path=path, label=label)

    if not loaded_runs:
        return {
            "rows": rows,
            "behavior": None,
            "timeline_runs": [],
            "ok": False,
            "error": "None of the trajectories could be loaded.",
        }

    if len(loaded_runs) < 2:
        return {
            "rows": rows,
            "behavior": None,
            "timeline_runs": [],
            "ok": False,
            "error": (
                "Need at least two successfully loaded runs (Overview baseline + one comparison, or two uploads)."
            ),
        }

    behavior = build_behavioral_comparison(loaded_runs, fuzzy=fuzzy)

    timeline_runs = [
        {
            "run_id": r["run_id"],
            "label": r["label"],
            "steps": r.get("steps") or [],
        }
        for r in loaded_runs
    ]
    return {
        "rows": rows,
        "behavior": behavior,
        "timeline_runs": timeline_runs,
        "ok": True,
        "error": None,
    }


def _best_worst_flags(rows: list[dict]) -> dict[str, dict[str, set[str]]]:
    """Per numeric column, which run_ids are best / worst among successful rows."""
    usable = [r for r in rows if not r.get("error")]
    flags: dict[str, dict[str, set[str]]] = {}
    lower_better = ("steps", "wall_clock_s", "tokens", "peak_occupancy", "compactions")
    higher_better = ("tool_success_pct",)

    def _collect(key: str, prefer_low: bool) -> None:
        vals = [(r["run_id"], r[key]) for r in usable if isinstance(r.get(key), (int, float))]
        if len(vals) < 2:
            return
        numbers = [v for _, v in vals]
        best_v = min(numbers) if prefer_low else max(numbers)
        worst_v = max(numbers) if prefer_low else min(numbers)
        if best_v == worst_v:
            return
        flags[key] = {
            "best": {rid for rid, v in vals if v == best_v},
            "worst": {rid for rid, v in vals if v == worst_v},
        }

    for key in lower_better:
        _collect(key, prefer_low=True)
    for key in higher_better:
        _collect(key, prefer_low=False)
    return flags


def _cell_class(run_id: str, key: str, flags: dict[str, dict[str, set[str]]]) -> str:
    entry = flags.get(key)
    if not entry:
        return ""
    if run_id in entry.get("best", ()):
        return " style='color:var(--ov-success);font-weight:600;'"
    if run_id in entry.get("worst", ()):
        return " style='color:var(--ov-warn);font-weight:600;'"
    return ""


def _f1_bg(f1: float) -> str:
    """Background color for F1 cell (0 → cool gray, 1 → green)."""
    t = max(0.0, min(1.0, f1))
    # interpolate toward emerald
    r = int(226 - 140 * t)
    g = int(232 - 40 * t)
    b = int(240 - 120 * t)
    return f"background:rgb({r},{g},{b});"


def _render_similarity_html(behavior: dict) -> str:
    ids = behavior.get("run_ids") or []
    labels = behavior.get("labels") or {}
    matrix = behavior.get("similarity") or {}
    if len(ids) < 2:
        return ""
    parts = [
        "<h4 style='margin:1.25em 0 0.35em;font-size:14px;'>Behavioral similarity (alignment F1)</h4>",
        "<div style='font-size:12px;color:var(--ov-muted);margin-bottom:6px;'>"
        "Ordered action overlap via Converge LCS. 1.0 = identical tool sequence "
        "(ignoring REASON). Use pairwise Comparison for a full report on any pair."
        "</div>",
        '<table class="cvg-outcome-table"><thead><tr><th></th>',
    ]
    for rid in ids:
        parts.append(f"<th>{html.escape(str(labels.get(rid, rid)))}</th>")
    parts.append("</tr></thead><tbody>")
    for a in ids:
        parts.append(f"<tr><th>{html.escape(str(labels.get(a, a)))}</th>")
        for b in ids:
            f1 = float((matrix.get(a) or {}).get(b) or 0.0)
            style = _f1_bg(f1)
            parts.append(f"<td style='text-align:center;{style}'>{f1:.2f}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _rw_badges(cell: dict) -> str:
    """Compact read/write count chips for one run×file cell."""
    read = int(cell.get("read") or 0)
    write = int(cell.get("write") or 0)
    if read <= 0 and write <= 0:
        return "<span class='rg-file-empty'>—</span>"
    bits: list[str] = []
    if read > 0:
        label = "R" if read == 1 else f"R×{read}"
        bits.append(
            f"<span class='rg-badge rg-badge-r' title='{read} read(s)'>{label}</span>"
        )
    if write > 0:
        label = "W" if write == 1 else f"W×{write}"
        bits.append(
            f"<span class='rg-badge rg-badge-w' title='{write} write(s)'>{label}</span>"
        )
    return "".join(bits)


def _kind_badge(kind: str, n_runs: int, n_total: int) -> str:
    labels = {
        "consensus": ("shared", "rg-kind-shared"),
        "partial": ("partial", "rg-kind-partial"),
        "unique": ("unique", "rg-kind-unique"),
    }
    text, cls = labels.get(kind, (kind, "rg-kind-partial"))
    return f"<span class='rg-kind {cls}'>{html.escape(text)}</span><span class='rg-cov'>{n_runs}/{n_total}</span>"


def _action_type_badge(atype: str) -> str:
    short = {
        "FILE_READ": ("READ", "rg-atype-read"),
        "FILE_WRITE": ("WRITE", "rg-atype-write"),
        "SEARCH": ("SEARCH", "rg-atype-search"),
        "COMMAND": ("CMD", "rg-atype-cmd"),
        "AGENT_SPAWN": ("SPAWN", "rg-atype-spawn"),
    }.get(atype, (atype[:6], "rg-atype-other"))
    text, cls = short
    return f"<span class='rg-atype {cls}'>{html.escape(text)}</span>"


def _action_cell(cell: dict) -> str:
    if not cell.get("present"):
        return "<span class='rg-file-empty'>—</span>"
    count = int(cell.get("count") or 1)
    if count <= 1:
        return "<span class='rg-action-hit' title='Present'>✓</span>"
    return f"<span class='rg-action-hit' title='{count} times'>✓×{count}</span>"


def _render_action_matrix_html(behavior: dict) -> str:
    ids = behavior.get("run_ids") or []
    labels = behavior.get("labels") or {}
    rows = behavior.get("action_matrix") or []
    total = int(behavior.get("action_matrix_total") or len(rows))
    n = len(ids)
    parts = [
        "<h4 style='margin:1em 0 0.35em;font-size:14px;'>Action coverage</h4>",
        "<div style='font-size:12px;color:var(--ov-muted);margin-bottom:6px;'>"
        "Non-file canonical actions (search, command, spawn, …). "
        "File read/write live in <b>File coverage</b> below. "
        "<span class='rg-action-hit'>✓</span> = present; ×N = repeats in that run."
        "</div>",
    ]
    if not rows:
        parts.append(
            "<div style='font-size:12px;color:var(--ov-muted);'>No non-REASON actions across these runs.</div>"
        )
        return "".join(parts)

    parts.append(
        "<div class='rg-file-scroll'><table class='cvg-outcome-table rg-file-table'><thead><tr><th>Action</th>"
    )
    for rid in ids:
        parts.append(f"<th style='text-align:center;'>{html.escape(str(labels.get(rid, rid)))}</th>")
    parts.append("<th>Coverage</th></tr></thead><tbody>")

    for row in rows:
        atype = str(row.get("type") or "")
        target = str(row.get("target") or "")
        short = str(row.get("short_target") or target)
        kind = str(row.get("kind") or "partial")
        n_runs = int(row.get("n_runs") or 0)
        cells = row.get("cells") or {}
        full = str(row.get("label") or _fmt_signature((atype, target)))
        target_html = (
            f"<code title='{html.escape(target)}'>{html.escape(short)}</code>" if target and target != "*" else ""
        )
        parts.append(f"<tr class='rg-row-{kind}'>")
        parts.append(
            f"<td class='rg-file-path' title='{html.escape(full)}'>{_action_type_badge(atype)} {target_html}</td>"
        )
        for rid in ids:
            cell = cells.get(rid) or {"present": False, "count": 0}
            parts.append(f"<td style='text-align:center;white-space:nowrap;'>{_action_cell(cell)}</td>")
        parts.append(f"<td style='white-space:nowrap;'>{_kind_badge(kind, n_runs, n)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    if total > len(rows):
        parts.append(
            f"<div style='font-size:12px;color:var(--ov-muted);margin-top:4px;'>"
            f"Showing {len(rows)} of {total} actions</div>"
        )
    return "".join(parts)


def _render_file_matrix_html(behavior: dict) -> str:
    ids = behavior.get("run_ids") or []
    labels = behavior.get("labels") or {}
    rows = behavior.get("file_matrix") or []
    total = int(behavior.get("file_matrix_total") or len(rows))
    n = len(ids)
    parts = [
        "<h4 style='margin:1em 0 0.35em;font-size:14px;'>File coverage</h4>",
        "<div style='font-size:12px;color:var(--ov-muted);margin-bottom:6px;'>"
        "<span class='rg-badge rg-badge-r'>R</span> / "
        "<span class='rg-badge rg-badge-r'>R×N</span> read count &nbsp;"
        "<span class='rg-badge rg-badge-w'>W</span> / "
        "<span class='rg-badge rg-badge-w'>W×N</span> write count &nbsp;·&nbsp; "
        "shared = consensus threshold, unique = one run only"
        "</div>",
    ]
    if not rows:
        parts.append("<div style='font-size:12px;color:var(--ov-muted);'>No file reads/writes across these runs.</div>")
        return "".join(parts)

    parts.append("<div class='rg-file-scroll'><table class='cvg-outcome-table rg-file-table'><thead><tr><th>File</th>")
    for rid in ids:
        parts.append(f"<th style='text-align:center;'>{html.escape(str(labels.get(rid, rid)))}</th>")
    parts.append("<th>Coverage</th></tr></thead><tbody>")

    for row in rows:
        path = str(row.get("path") or "")
        short = str(row.get("short") or path)
        kind = str(row.get("kind") or "partial")
        n_runs = int(row.get("n_runs") or 0)
        cells = row.get("cells") or {}
        row_cls = f"rg-row-{kind}"
        parts.append(f"<tr class='{row_cls}'>")
        parts.append(f"<td class='rg-file-path' title='{html.escape(path)}'><code>{html.escape(short)}</code></td>")
        for rid in ids:
            cell = cells.get(rid) or {"read": 0, "write": 0}
            parts.append(f"<td style='text-align:center;white-space:nowrap;'>{_rw_badges(cell)}</td>")
        parts.append(f"<td style='white-space:nowrap;'>{_kind_badge(kind, n_runs, n)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    if total > len(rows):
        parts.append(
            f"<div style='font-size:12px;color:var(--ov-muted);margin-top:4px;'>"
            f"Showing {len(rows)} of {total} files</div>"
        )
    return "".join(parts)


def _render_count_matrix_html(
    behavior: dict,
    *,
    matrix_key: str,
    total_key: str,
    title: str,
    blurb: str,
    empty: str,
    noun: str,
) -> str:
    """Shared renderer for tool / MCP / skill presence matrices."""
    ids = behavior.get("run_ids") or []
    labels = behavior.get("labels") or {}
    rows = behavior.get(matrix_key) or []
    total = int(behavior.get(total_key) or len(rows))
    n = len(ids)
    parts = [
        f"<h4 style='margin:1em 0 0.35em;font-size:14px;'>{html.escape(title)}</h4>",
        f"<div style='font-size:12px;color:var(--ov-muted);margin-bottom:6px;'>{html.escape(blurb)}</div>",
    ]
    if not rows:
        parts.append(f"<div style='font-size:12px;color:var(--ov-muted);'>{html.escape(empty)}</div>")
        return "".join(parts)

    parts.append(
        "<div class='rg-file-scroll'>"
        "<table class='cvg-outcome-table rg-file-table'><thead><tr>"
        f"<th>{html.escape(noun)}</th>"
    )
    for rid in ids:
        parts.append(f"<th style='text-align:center;'>{html.escape(str(labels.get(rid, rid)))}</th>")
    parts.append("<th>Coverage</th></tr></thead><tbody>")

    for row in rows:
        key = str(row.get("key") or "")
        short = str(row.get("short") or key)
        kind = str(row.get("kind") or "partial")
        n_runs = int(row.get("n_runs") or 0)
        cells = row.get("cells") or {}
        parts.append(f"<tr class='rg-row-{kind}'>")
        parts.append(f"<td class='rg-file-path' title='{html.escape(key)}'><code>{html.escape(short)}</code></td>")
        for rid in ids:
            cell = cells.get(rid) or {"present": False, "count": 0}
            parts.append(f"<td style='text-align:center;white-space:nowrap;'>{_action_cell(cell)}</td>")
        parts.append(f"<td style='white-space:nowrap;'>{_kind_badge(kind, n_runs, n)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    if total > len(rows):
        parts.append(
            f"<div style='font-size:12px;color:var(--ov-muted);margin-top:4px;'>"
            f"Showing {len(rows)} of {total} {html.escape(noun.lower())}s</div>"
        )
    return "".join(parts)


def _render_behavior_html(behavior: dict | None) -> str:
    if not behavior:
        return ""
    labels = behavior.get("labels") or {}
    baseline = behavior.get("baseline_run_id") or ""
    base_label = labels.get(baseline, baseline)
    thresh = behavior.get("consensus_threshold") or 0
    parts = [
        "<hr style='border:none;border-top:1px solid var(--ov-border);margin:1.25em 0;'>",
        "<h3 style='margin:0 0 0.35em;font-size:15px;'>Behavioral comparison</h3>",
        f"<div style='font-size:12px;color:var(--ov-muted);margin-bottom:8px;'>"
        f"Baseline for waste patterns: <b>{html.escape(str(base_label))}</b> "
        f"(first loaded run). Consensus = item seen in ≥{thresh} runs."
        f"</div>",
        _render_similarity_html(behavior),
        _render_count_matrix_html(
            behavior,
            matrix_key="tool_matrix",
            total_key="tool_matrix_total",
            title="Tool coverage",
            blurb="All tool names used across runs (including MCP tools).",
            empty="No tools recorded across these runs.",
            noun="Tool",
        ),
        _render_count_matrix_html(
            behavior,
            matrix_key="skill_matrix",
            total_key="skill_matrix_total",
            title="Skill coverage",
            blurb="Skills triggered via the Skill tool (name from tool input).",
            empty="No Skill-tool invocations recorded across these runs.",
            noun="Skill",
        ),
        _render_action_matrix_html(behavior),
        _render_file_matrix_html(behavior),
    ]

    # Patterns vs baseline
    patterns = behavior.get("patterns_vs_baseline") or {}
    parts.append(
        f"<h4 style='margin:1em 0 0.35em;font-size:14px;'>"
        f"Waste patterns vs baseline ({html.escape(str(base_label))})</h4>"
        "<div style='font-size:12px;color:var(--ov-muted);margin-bottom:6px;'>"
        "Divergence patterns on actions not matched to the baseline trajectory."
        "</div>"
    )
    any_pat = False
    for rid in behavior.get("run_ids") or []:
        if rid == baseline:
            continue
        plist = patterns.get(rid) or []
        label = labels.get(rid, rid)
        if not plist:
            parts.append(
                f"<div style='font-size:13px;margin:0.35em 0;'>"
                f"<b>{html.escape(str(label))}</b> — no classified waste patterns"
                f"</div>"
            )
            continue
        any_pat = True
        bits = ", ".join(f"{html.escape(p['label'])} ×{p['count']}" for p in plist[:6])
        parts.append(f"<div style='font-size:13px;margin:0.35em 0;'><b>{html.escape(str(label))}</b> — {bits}</div>")
    if not any_pat and len(behavior.get("run_ids") or []) <= 1:
        parts.append("<div style='font-size:12px;color:var(--ov-muted);'>No other runs.</div>")

    return "".join(parts)


def build_run_group_scorecard_html(
    result: dict,
    *,
    include_behavior: bool = True,
) -> str:
    """Render scorecard table, optionally plus behavioral comparison sections."""
    error = result.get("error")
    rows = result.get("rows") or []
    if error and not rows:
        return f"<div style='color:var(--ov-warn);padding:1em;text-align:center;'>{html.escape(str(error))}</div>"
    if not rows:
        return (
            "<div style='padding:1.5em;color:var(--ov-muted);text-align:center;'>"
            "Upload two or more trajectories and click <b>Build scorecard</b>.</div>"
        )

    flags = _best_worst_flags(rows)
    parts: list[str] = []
    if error:
        parts.append(f"<div style='color:var(--ov-warn);margin-bottom:0.5em;'>{html.escape(str(error))}</div>")
    n_ok = sum(1 for r in rows if not r.get("error"))
    parts.append(
        f"<div style='font-size:12px;color:var(--ov-muted);margin-bottom:6px;'>"
        f"{n_ok} of {len(rows)} run(s) loaded"
        f" &middot; green = best, amber = worst among loaded runs"
        f" &middot; Finished means normal stop, not task correctness"
        f"</div>"
    )
    parts.append('<table class="cvg-outcome-table"><thead><tr>')
    headers = [
        "Condition",
        "Format",
        "Finished",
        "Steps",
        "Time",
        "Tokens",
        "Tools",
        "Success %",
        "Peak ctx",
        "Compacts",
    ]
    for h in headers:
        parts.append(f"<th>{html.escape(h)}</th>")
    parts.append("</tr></thead><tbody>")

    for row in rows:
        rid = row.get("run_id") or ""
        if row.get("error"):
            parts.append("<tr>")
            parts.append(f"<td>{html.escape(str(row.get('label') or rid))}</td>")
            parts.append("<td colspan='9' style='color:var(--ov-bad);'>")
            parts.append(html.escape(str(row["error"])))
            parts.append("</td></tr>")
            continue

        finished = "✓" if row.get("finished") else "✗"
        fin_style = (
            " style='color:var(--ov-success);font-weight:600;'"
            if row.get("finished")
            else " style='color:var(--ov-bad);'"
        )
        peak = row.get("peak_occupancy") or 0
        peak_pct = row.get("peak_pct")
        if isinstance(peak_pct, (int, float)):
            peak_txt = f"{peak:,} ({peak_pct:g}%)"
        else:
            peak_txt = f"{peak:,}" if peak else "—"

        cells = [
            (html.escape(str(row.get("label") or rid)), ""),
            (html.escape(str(row.get("format") or "—")), ""),
            (finished, fin_style),
            (f"{row.get('steps') or 0}", _cell_class(rid, "steps", flags)),
            (html.escape(str(row.get("wall_clock_fmt") or "—")), _cell_class(rid, "wall_clock_s", flags)),
            (f"{row.get('tokens') or 0:,}", _cell_class(rid, "tokens", flags)),
            (f"{row.get('tool_calls') or 0}", ""),
            (f"{row.get('tool_success_pct') or 0:g}%", _cell_class(rid, "tool_success_pct", flags)),
            (html.escape(peak_txt), _cell_class(rid, "peak_occupancy", flags)),
            (f"{row.get('compactions') or 0}", _cell_class(rid, "compactions", flags)),
        ]
        parts.append("<tr>")
        for text, style in cells:
            parts.append(f"<td{style}>{text}</td>")
        parts.append("</tr>")

    parts.append("</tbody></table>")
    if include_behavior:
        parts.append(_render_behavior_html(result.get("behavior")))
    return "".join(parts)


def build_run_group_behavior_html(result: dict) -> str:
    """Render behavioral comparison sections for a scorecard result."""
    return _render_behavior_html(result.get("behavior") if result else None)
