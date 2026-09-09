"""Overview Issues triage: rank session signals into a foldable panel."""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Literal

from ..rendering import (
    _indices_for_step_range,
    _step_link_chips,
)
from ..session import LoadedSession
from .patterns import count_tool_errors

IssueKind = Literal["error", "antipattern"]

# Lower = more urgent. Errors before waste.
_SEVERITY: dict[IssueKind, int] = {
    "error": 0,
    "antipattern": 1,
}

_BORDER: dict[IssueKind, str] = {
    "error": "var(--ov-bad)",
    "antipattern": "var(--ov-warn)",
}


@dataclass(frozen=True)
class OverviewIssue:
    """One Overview triage item (session diagnostics only; no Fix/Change yet)."""

    kind: IssueKind
    title: str
    detail: str
    why: str = ""
    steps: tuple[int, ...] = ()


def rank_issues(issues: list[OverviewIssue]) -> list[OverviewIssue]:
    """Sort by severity, then earliest step, then title. Show all (panel is foldable)."""
    return sorted(
        issues,
        key=lambda i: (
            _SEVERITY[i.kind],
            min(i.steps) if i.steps else 10**9,
            i.title,
        ),
    )


def collect_overview_issues(session: LoadedSession) -> list[OverviewIssue]:
    """Map LoadedSession diagnostics into Issues (no re-detection).

    Slow-step bottlenecks are omitted — latency is usually not author-controllable
    the way errors and anti-patterns are.
    """
    issues: list[OverviewIssue] = []
    issues.extend(_from_failure_patterns(session))
    issues.extend(_from_antipatterns(session))
    return issues


def _from_failure_patterns(session: LoadedSession) -> list[OverviewIssue]:
    out: list[OverviewIssue] = []
    for pat in session.failure_patterns or []:
        label = str(pat.get("cluster_label") or "Unknown error")
        count = int(pat.get("count") or 0)
        example = str(pat.get("example_error") or "")[:200]
        recovery = pat.get("recovery_path")
        if recovery and isinstance(recovery, (list, tuple)):
            why = "Typical recovery: " + " → ".join(str(t) for t in recovery)
        else:
            why = "No recovery path found after this error cluster."
        steps = tuple(int(s) for s in (pat.get("steps") or []) if s is not None)
        out.append(
            OverviewIssue(
                kind="error",
                title=f"{label} ({count}×)" if count else label,
                detail=example,
                why=why,
                steps=steps,
            )
        )
    return out


def _from_antipatterns(session: LoadedSession) -> list[OverviewIssue]:
    out: list[OverviewIssue] = []

    # Skip tool-error rollup when failure_patterns already cover the same failures.
    if not session.failure_patterns:
        error_count, error_steps = count_tool_errors(session.steps)
        if error_count > 0:
            out.append(
                OverviewIssue(
                    kind="error",
                    title=f"{error_count} tool error(s)",
                    detail="detected from tool output (platform, permission, missing file)",
                    why=(
                        "Failed tool calls cost tokens and turns to recover from, and often "
                        "indicate environment problems rather than agent mistakes."
                    ),
                    steps=tuple(error_steps),
                )
            )

    streaks = session.fruitless_streaks or []
    if streaks:
        total_wasted = sum(int(s.get("length") or 0) for s in streaks)
        shown = streaks[:3]
        streak_desc = ", ".join(
            f"steps {s.get('start_step')}-{s.get('end_step')} ({s.get('length')})"
            for s in shown
        )
        remaining = len(streaks) - len(shown)
        if remaining > 0:
            remaining_len = sum(int(s.get("length") or 0) for s in streaks[len(shown):])
            streak_desc += f", +{remaining} more ({remaining_len})"
        streak_indices: list[int] = []
        for s in streaks:
            streak_indices.extend(
                _indices_for_step_range(s.get("start_step"), s.get("end_step"))
            )
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=f"{len(streaks)} fruitless search streak(s)",
                detail=f"{total_wasted} wasted steps — {streak_desc}",
                why=(
                    "Three or more consecutive searches that returned no matches; "
                    "sustained streaks suggest looking in the wrong place."
                ),
                steps=tuple(streak_indices),
            )
        )

    tool_selection = session.tool_selection or []
    if tool_selection:
        bash_steps = [
            int(f["step"]) for f in tool_selection if f.get("step") is not None
        ]
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=f"{len(tool_selection)} Bash-for-reading",
                detail="steps used sed/cat/head instead of Read tool",
                why=(
                    "Shell reads bypass structured Read tooling — no line numbers, "
                    "weaker cache, larger context."
                ),
                steps=tuple(bash_steps),
            )
        )

    stalled = (session.plan_metrics or {}).get("stalled") or []
    if stalled:
        items_desc = ", ".join(f"'{s.get('content', '')[:30]}'" for s in stalled[:2])
        stall_steps: list[int] = []
        for s in stalled:
            stall_steps.extend(
                _indices_for_step_range(s.get("start_step"), s.get("end_step"))
            )
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=f"{len(stalled)} stalled plan item(s)",
                detail=items_desc,
                why=(
                    "Todo items stayed in_progress without completion — often a "
                    "context switch that never closed the loop."
                ),
                steps=tuple(stall_steps),
            )
        )

    return out


def _issue_card(issue: OverviewIssue) -> str:
    """Render one issue card: title, detail, step links, optional why."""
    title = html.escape(issue.title)
    detail = html.escape(issue.detail)
    border = _BORDER[issue.kind]
    steps_html = _step_link_chips(list(issue.steps))

    why_html = ""
    if issue.why:
        why_html = (
            f"<div style='font-size:11px;color:var(--ov-muted);font-style:italic;"
            f"margin-top:4px;'>Why it matters: {html.escape(issue.why)}</div>"
        )

    return (
        f"<div style='padding:10px 12px;background:var(--ov-card);"
        f"border-left:3px solid {border};border-radius:4px;margin-bottom:8px;'>"
        f"<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>"
        f"<span style='font-size:12px;font-weight:600;'>{title}</span>"
        f"<span style='font-size:12px;color:var(--ov-muted);'>{detail}</span>"
        f"</div>"
        f"{steps_html}"
        f"{why_html}"
        f"</div>"
    )


def render_overview_issues_html(issues: list[OverviewIssue]) -> str:
    """Render a foldable Issues panel (healthy empty state when *issues* is empty).

    Uses ``<details>`` so the panel can be collapsed without Gradio rewiring.
    Non-empty panels start expanded; empty stays collapsed.
    """
    if not issues:
        return (
            "<details class='overview-issues-panel' id='overview-issues'>"
            "<summary class='overview-issues-summary'>"
            "<span>Issues</span>"
            "<span class='overview-issues-summary-meta'>none detected</span>"
            "</summary>"
            "<div class='overview-issues-body'>"
            "<div style='padding:8px 0 4px;color:var(--ov-muted);text-align:center;font-size:13px;'>"
            "No major workflow issues detected."
            "</div></div></details>"
        )

    count = len(issues)
    count_label = f"{count} issue{'s' if count != 1 else ''}"
    return (
        "<details class='overview-issues-panel' id='overview-issues' open>"
        "<summary class='overview-issues-summary'>"
        "<span>Issues</span>"
        f"<span class='overview-issues-summary-meta'>{html.escape(count_label)}</span>"
        "</summary>"
        "<div class='overview-issues-body'>"
        "<div style='font-size:12px;color:var(--ov-muted);margin:0 0 8px;'>"
        "Ranked workflow problems — click a step to open Workflow"
        "</div>"
        + "".join(_issue_card(issue) for issue in issues)
        + "</div></details>"
    )


def build_overview_issues_html(session: LoadedSession) -> str:
    """Collect, rank, and render Overview Issues for *session*."""
    return render_overview_issues_html(rank_issues(collect_overview_issues(session)))
