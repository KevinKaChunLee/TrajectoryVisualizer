"""Overview Issues triage: rank session signals into a foldable panel."""

from __future__ import annotations

import html
from dataclasses import dataclass, replace
from typing import Literal

from ..rendering import (
    _indices_for_step_range,
    _step_link_chips,
)
from ..session import LoadedSession
from .patterns import count_tool_errors

IssueKind = Literal["error", "antipattern", "bottleneck"]
Confidence = Literal["high", "medium", "low"]

# Lower = more urgent. Errors before waste before latency.
_SEVERITY: dict[IssueKind, int] = {
    "error": 0,
    "antipattern": 1,
    "bottleneck": 2,
}

_BORDER: dict[IssueKind, str] = {
    "error": "var(--ov-bad)",
    "antipattern": "var(--ov-warn)",
    "bottleneck": "var(--ov-accent)",
}


@dataclass(frozen=True)
class IssueJudgment:
    """LLM-suggested Change/Fix for one OverviewIssue (on-demand judge)."""

    where: str
    fix: str
    also: str = ""
    confidence: Confidence = "medium"


@dataclass(frozen=True)
class OverviewIssue:
    """One Overview triage item (session diagnostics; optional LLM judgment)."""

    kind: IssueKind
    title: str
    detail: str
    why: str = ""
    steps: tuple[int, ...] = ()
    source_id: str = ""
    judgment: IssueJudgment | None = None

    def with_judgment(self, judgment: IssueJudgment) -> OverviewIssue:
        return replace(self, judgment=judgment)


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
    """Map LoadedSession diagnostics into Issues (no re-detection)."""
    issues: list[OverviewIssue] = []
    issues.extend(_from_failure_patterns(session))
    issues.extend(_from_failure_chains(session))
    issues.extend(_from_antipatterns(session))
    issues.extend(_from_bottlenecks(session))
    return issues


def _from_failure_patterns(session: LoadedSession) -> list[OverviewIssue]:
    out: list[OverviewIssue] = []
    for i, pat in enumerate(session.failure_patterns or []):
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
                source_id=f"fail:{i}:{label}",
            )
        )
    return out


def _from_failure_chains(session: LoadedSession) -> list[OverviewIssue]:
    """Consecutive assistant error runs (cascades). Skip length-1 (covered by clusters)."""
    out: list[OverviewIssue] = []
    chains = getattr(session, "failure_chains", None) or []
    for i, chain in enumerate(chains):
        steps = tuple(int(s) for s in (chain.get("steps") or []) if s is not None)
        if len(steps) < 2:
            continue
        start = chain.get("start", steps[0])
        end = chain.get("end", steps[-1])
        out.append(
            OverviewIssue(
                kind="error",
                title=f"{len(steps)}-step failure cascade",
                detail=f"steps {start}–{end}",
                why=(
                    "Consecutive assistant steps failed without a clean recovery in between; "
                    "cascades often amplify one root error into thrash."
                ),
                steps=steps,
                source_id=f"cascade:{i}:{start}-{end}",
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
                    source_id="antipattern:tool_errors",
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
                source_id="antipattern:fruitless",
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
                source_id="antipattern:bash_read",
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
                source_id="antipattern:stalled_plan",
            )
        )

    plan_resets = int((session.plan_metrics or {}).get("plan_resets") or 0)
    if plan_resets > 0:
        plan_steps = sorted({
            int(snap["step"])
            for snap in (getattr(session, "plan_history", None) or [])
            if snap.get("step") is not None
        })
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=f"{plan_resets} plan reset(s)",
                detail="todo list content replaced with no overlapping items",
                why=(
                    "A full plan rewrite mid-run usually means the agent abandoned context "
                    "instead of completing or revising items in place."
                ),
                steps=tuple(plan_steps[:12]),
                source_id="antipattern:plan_resets",
            )
        )

    for i, thrash in enumerate(getattr(session, "edit_thrash", None) or []):
        path = str(thrash.get("path") or "")
        count = int(thrash.get("count") or 0)
        steps = tuple(int(s) for s in (thrash.get("steps") or []) if s is not None)
        short = path if len(path) <= 48 else ("…" + path[-47:])
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=f"Edit thrash on {short} ({count}×)",
                detail=f"steps {thrash.get('start_step')}–{thrash.get('end_step')}",
                why=(
                    "Repeated Write/Edit on the same file in a short window often means "
                    "thrashing without a stable approach."
                ),
                steps=steps,
                source_id=f"antipattern:edit_thrash:{i}:{path}",
            )
        )

    for i, rep in enumerate(getattr(session, "repeated_searches", None) or []):
        display = str(rep.get("display") or rep.get("signature") or "")
        count = int(rep.get("count") or 0)
        steps = tuple(int(s) for s in (rep.get("steps") or []) if s is not None)
        short = display if len(display) <= 48 else (display[:45] + "…")
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=f"Repeated empty search ({count}×)",
                detail=short,
                why=(
                    "The same search ran multiple times with empty results (not only as a "
                    "consecutive streak) — the query or path is likely wrong."
                ),
                steps=steps,
                source_id=f"antipattern:repeated_search:{i}",
            )
        )

    return out


def _from_bottlenecks(session: LoadedSession) -> list[OverviewIssue]:
    """Map detected performance bottlenecks (outlier + clear cause), not top-N slow steps."""
    out: list[OverviewIssue] = []
    for i, bn in enumerate(getattr(session, "performance_bottlenecks", None) or []):
        step_idx = bn.get("step_idx")
        if step_idx is None:
            continue
        idx = int(step_idx)
        title = str(bn.get("title") or f"Performance bottleneck at #{idx}")
        detail = str(bn.get("detail") or bn.get("explanation") or "")[:200]
        why = str(bn.get("why") or (
            "Session-relative duration outlier with a dominant tool, idle/queue, "
            "or context/inference cause."
        ))
        out.append(
            OverviewIssue(
                kind="bottleneck",
                title=title,
                detail=detail,
                why=why,
                steps=(idx,),
                source_id=f"bottleneck:{bn.get('cause', 'unknown')}:{i}:{idx}",
            )
        )
    return out


def _issue_card(issue: OverviewIssue) -> str:
    """Render one issue card; Change/Fix only when an LLM judgment is present."""
    title = html.escape(issue.title)
    detail = html.escape(issue.detail)
    border = _BORDER[issue.kind]
    steps_html = _step_link_chips(list(issue.steps))

    judgment_html = ""
    if issue.judgment is not None:
        j = issue.judgment
        also_html = ""
        if j.also:
            also_html = (
                f"<div style='font-size:11px;color:var(--ov-muted);margin-top:6px;"
                f"line-height:1.35;'>Also: {html.escape(j.also)}</div>"
            )
        conf = html.escape(j.confidence)
        judgment_html = (
            "<div style='margin-top:8px;font-size:12px;line-height:1.4;'>"
            "<span style='font-size:10px;font-weight:600;letter-spacing:0.04em;"
            "text-transform:uppercase;color:var(--ov-muted);'>Change</span>"
            f"<div style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
            f"font-size:12px;color:var(--ov-text);margin-top:2px;'>"
            f"{html.escape(j.where)}</div></div>"
            "<div style='margin-top:8px;padding:8px 10px;background:var(--ov-bg);"
            "border-radius:4px;'>"
            "<div style='font-size:10px;font-weight:600;letter-spacing:0.04em;"
            "text-transform:uppercase;color:var(--ov-muted);margin-bottom:4px;'>"
            f"Fix <span style='font-weight:500;letter-spacing:0;text-transform:none;"
            f"color:var(--ov-muted);'>({conf})</span></div>"
            f"<div style='font-size:13px;line-height:1.4;color:var(--ov-text);'>"
            f"{html.escape(j.fix)}</div>"
            f"{also_html}</div>"
        )

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
        f"{judgment_html}"
        f"{why_html}"
        f"</div>"
    )


def render_overview_issues_html(
    issues: list[OverviewIssue],
    *,
    banner: str = "",
) -> str:
    """Render a foldable Issues panel (healthy empty state when *issues* is empty)."""
    banner_html = ""
    if banner:
        banner_html = (
            f"<div style='font-size:12px;color:var(--ov-muted);margin:0 0 8px;'>"
            f"{html.escape(banner)}</div>"
        )

    if not issues:
        return (
            "<details class='overview-issues-panel' id='overview-issues'>"
            "<summary class='overview-issues-summary'>"
            "<span>Issues</span>"
            "<span class='overview-issues-summary-meta'>none detected</span>"
            "</summary>"
            "<div class='overview-issues-body'>"
            f"{banner_html}"
            "<div style='padding:8px 0 4px;color:var(--ov-muted);text-align:center;font-size:13px;'>"
            "No major workflow issues detected."
            "</div></div></details>"
        )

    count = len(issues)
    judged = sum(1 for i in issues if i.judgment is not None)
    count_label = f"{count} issue{'s' if count != 1 else ''}"
    if judged:
        count_label += f" · {judged} with LLM fix"

    hint = (
        "LLM Change/Fix attached — click a step to open Workflow"
        if judged
        else "Ranked workflow problems — click a step to open Workflow, or Suggest fixes"
    )
    return (
        "<details class='overview-issues-panel' id='overview-issues' open>"
        "<summary class='overview-issues-summary'>"
        "<span>Issues</span>"
        f"<span class='overview-issues-summary-meta'>{html.escape(count_label)}</span>"
        "</summary>"
        "<div class='overview-issues-body'>"
        f"{banner_html}"
        f"<div style='font-size:12px;color:var(--ov-muted);margin:0 0 8px;'>{hint}</div>"
        + "".join(_issue_card(issue) for issue in issues)
        + "</div></details>"
    )


def build_overview_issues_html(
    session: LoadedSession,
    *,
    issues: list[OverviewIssue] | None = None,
    banner: str = "",
) -> str:
    """Collect, rank, and render Overview Issues for *session*."""
    if issues is None:
        ranked = rank_issues(collect_overview_issues(session))
    else:
        ranked = list(issues)
    return render_overview_issues_html(ranked, banner=banner)
