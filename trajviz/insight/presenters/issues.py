"""Overview Issues triage: rank session signals into fix-first cards."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Literal

from ..rendering import (
    _indices_for_step_range,
    _step_link_chips,
)
from ..session import LoadedSession

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
    """One Overview triage item with a single primary fix (session data only)."""

    kind: IssueKind
    title: str
    detail: str
    why: str
    fix: str = ""
    also: str = ""
    steps: tuple[int, ...] = ()
    source_id: str = ""
    severity: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", _SEVERITY[self.kind])


def _earliest_step(issue: OverviewIssue) -> int:
    return min(issue.steps) if issue.steps else 10**9


def rank_issues(issues: list[OverviewIssue]) -> list[OverviewIssue]:
    """Sort by severity, then earliest step, then title. Show all (panel is foldable)."""
    return sorted(issues, key=lambda i: (i.severity, _earliest_step(i), i.title))


def collect_overview_issues(session: LoadedSession) -> list[OverviewIssue]:
    """Map LoadedSession diagnostics into Issues (no re-detection).

    Bottlenecks / slow steps are omitted: wall-clock hotspots are often
    model or tool latency the author cannot control the way they can fix
    errors and anti-patterns.
    """
    issues: list[OverviewIssue] = []
    issues.extend(_from_failure_patterns(session.failure_patterns))
    issues.extend(_from_antipatterns(session))
    return issues


def _prescription_for_failure(
    label: str, recovery: list | None
) -> tuple[str, str]:
    """Return (fix, also) — one primary change, optional supporting note."""
    label_l = label.lower()

    if recovery:
        path = " → ".join(str(t) for t in recovery)
        fix = (
            f"Encode this recovery in the system prompt or a skill before the failing "
            f"tool retries: {path}."
        )
    else:
        fix = (
            "Add an early exit or explicit failure handler so the agent stops after this "
            "error instead of retrying blindly."
        )

    if "bash" in label_l or "exit" in label_l:
        also = (
            "Before Bash: Glob/Read the path and check cwd/permissions; fail closed "
            "when preconditions are unmet."
        )
    elif "permission" in label_l or "denied" in label_l:
        also = (
            "Fix sandbox/permissions/auth in the environment — prompt retries will not "
            "clear a denied call."
        )
    elif "not found" in label_l or "enoent" in label_l or "missing" in label_l:
        also = "Require Glob → Read before Edit/Bash, or seed the project layout in the prompt."
    else:
        also = (
            "Confirm environment preconditions before changing prompts if the error looks "
            "platform- or permission-related."
        )
    return fix, also


def _from_failure_patterns(patterns: list[dict]) -> list[OverviewIssue]:
    out: list[OverviewIssue] = []
    for i, pat in enumerate(patterns):
        label = str(pat.get("cluster_label") or "Unknown error")
        count = int(pat.get("count") or 0)
        example = str(pat.get("example_error") or "")[:200]
        recovery = pat.get("recovery_path")
        if recovery:
            why = "Typical recovery: " + " → ".join(str(t) for t in recovery)
        else:
            why = "No recovery path found after this error cluster."
        steps = tuple(int(s) for s in (pat.get("steps") or []) if s is not None)
        fix, also = _prescription_for_failure(
            label, list(recovery) if recovery else None
        )
        out.append(
            OverviewIssue(
                kind="error",
                title=f"{label} ({count}×)" if count else label,
                detail=example,
                why=why,
                fix=fix,
                also=also,
                steps=steps,
                source_id=f"fail:{i}:{label}",
            )
        )
    return out


def _from_antipatterns(session: LoadedSession) -> list[OverviewIssue]:
    out: list[OverviewIssue] = []

    # Tool-error rollup only when failure_patterns did not already surface clusters
    # (both come from tool failures; keep Issues from double-listing).
    if not session.failure_patterns:
        error_steps: list[int] = []
        error_count = 0
        for step in session.steps:
            errs = [tc for tc in (step.get("tool_calls") or []) if tc.get("error_type")]
            if not errs:
                continue
            error_count += len(errs)
            error_steps.append(int(step.get("index", 0)))
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
                    fix=(
                        "If the error is platform/permission, fix cwd, deps, sandbox, or "
                        "auth before changing prompts."
                    ),
                    also=(
                        "Add a precondition check (exists? readable? authorized?) before "
                        "retrying that tool class."
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
                fix=(
                    "In the prompt or skill: require Glob/path confirmation before a "
                    "search loop, and stop after N empty results to re-plan."
                ),
                also="Seed known-good paths or project layout so the agent does not invent directories.",
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
                fix=(
                    "Update tool guidance: read files with Read; use Bash only for "
                    "commands that mutate state or run programs."
                ),
                also="Discourage or remove shell-for-read examples from the system prompt.",
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
                fix=(
                    "Require TodoWrite complete/cancel on the current item before the "
                    "agent starts unrelated work."
                ),
                also="Split oversized todos into smaller units the agent can finish in one loop.",
                steps=tuple(stall_steps),
                source_id="antipattern:stalled_plan",
            )
        )

    return out


def _issue_card(issue: OverviewIssue) -> str:
    """Render one prescription card: evidence + steps + a single Fix."""
    title = html.escape(str(issue.title))
    detail = html.escape(str(issue.detail))
    why = html.escape(str(issue.why))
    border = _BORDER[issue.kind]
    steps_html = _step_link_chips(list(issue.steps))

    fix_text = (issue.fix or issue.why or "").strip()
    also_text = (issue.also or "").strip()

    fix_html = ""
    if fix_text:
        fix_html = (
            "<div style='margin-top:8px;padding:8px 10px;background:var(--ov-bg);"
            "border-radius:4px;'>"
            "<div style='font-size:10px;font-weight:600;letter-spacing:0.04em;"
            "text-transform:uppercase;color:var(--ov-muted);margin-bottom:4px;'>Fix</div>"
            f"<div style='font-size:13px;line-height:1.4;color:var(--ov-text);'>"
            f"{html.escape(fix_text)}</div>"
        )
        if also_text:
            fix_html += (
                f"<div style='font-size:11px;color:var(--ov-muted);margin-top:6px;"
                f"line-height:1.35;'>Also: {html.escape(also_text)}</div>"
            )
        fix_html += "</div>"

    evidence = ""
    if why and why != html.escape(fix_text):
        evidence = (
            f"<div style='font-size:11px;color:var(--ov-muted);margin-top:4px;'>"
            f"{why}</div>"
        )

    return (
        f"<div class='overview-issue-card' style='padding:10px 12px;background:var(--ov-card);"
        f"border-left:3px solid {border};border-radius:4px;margin-bottom:8px;'>"
        f"<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>"
        f"<span style='font-size:12px;font-weight:600;'>{title}</span>"
        f"<span style='font-size:12px;color:var(--ov-muted);'>{detail}</span>"
        f"</div>"
        f"{steps_html}"
        f"{fix_html}"
        f"{evidence}"
        f"</div>"
    )


def render_overview_issues_html(
    issues: list[OverviewIssue],
    *,
    expanded: bool = True,
) -> str:
    """Render a foldable Issues panel (healthy empty state when *issues* is empty).

    Uses ``<details>`` so the panel can be collapsed without Gradio rewiring.
    Defaults to expanded when there are issues; empty state stays collapsed.
    All collected issues are shown — fold when the list is long.
    """
    if not issues:
        return (
            "<details class='overview-issues-panel' id='overview-issues'>"
            "<summary class='overview-issues-summary'>"
            "<span class='overview-issues-summary-title'>Issues</span>"
            "<span class='overview-issues-summary-meta'>none detected</span>"
            "</summary>"
            "<div class='overview-issues-body'>"
            "<div style='padding:8px 0 4px;color:var(--ov-muted);text-align:center;font-size:13px;'>"
            "No major workflow issues detected."
            "</div></div></details>"
        )

    cards = [_issue_card(issue) for issue in issues]
    count = len(issues)
    count_label = f"{count} issue{'s' if count != 1 else ''}"

    open_attr = " open" if expanded else ""
    return (
        f"<details class='overview-issues-panel' id='overview-issues'{open_attr}>"
        "<summary class='overview-issues-summary'>"
        "<span class='overview-issues-summary-title'>Issues</span>"
        f"<span class='overview-issues-summary-meta'>{html.escape(count_label)}</span>"
        "</summary>"
        "<div class='overview-issues-body'>"
        "<div style='font-size:12px;color:var(--ov-muted);margin:0 0 8px;'>"
        "One change per issue — open a step to verify, then apply the Fix"
        "</div>"
        + "".join(cards)
        + "</div></details>"
    )


def build_overview_issues_html(session: LoadedSession) -> str:
    """Collect, rank, and render Overview Issues for *session*."""
    return render_overview_issues_html(rank_issues(collect_overview_issues(session)))
