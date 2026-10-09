"""Overview Issues triage: rank session signals into a foldable panel."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, replace
from itertools import groupby
from typing import Literal

from ..rendering import (
    _indices_for_step_range,
    _step_link_chips,
)
from ..context_usage import format_token_count
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

ISSUE_KIND_COLORS: dict[IssueKind, str] = {
    "error": "var(--ov-bad)",
    "antipattern": "var(--ov-warn)",
    "bottleneck": "var(--ov-accent)",
}

_ISSUE_KIND_LABELS_ZH: dict[IssueKind, str] = {
    "error": "错误",
    "antipattern": "行为问题",
    "bottleneck": "耗时过长",
}

_CONFIDENCE_ZH: dict[Confidence, str] = {"high": "高", "medium": "中", "low": "低"}


@dataclass(frozen=True)
class IssueJudgment:
    """LLM-suggested Change/Fix for one OverviewIssue (on-demand judge)."""

    where: str
    fix: str
    also: str = ""
    confidence: Confidence = "medium"


@dataclass(frozen=True)
class OverviewIssue:
    """One Overview triage item (session diagnostics; optional LLM judgment).

    *why* is shown as a hover tooltip and sent to the LLM judge as context.
    """

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
    issues.extend(_from_premature_compactions(session))
    issues.extend(_from_bottlenecks(session))
    return issues


_SYSTEM_ERROR_HIGH = 5


def _count_label(label: str, count: int, unit: str = "次") -> str:
    return f"{label}（{count} {unit}）" if count else label


def _zh_step_range(start: object, end: object) -> str:
    return f"第{start}步" if start == end else f"第{start}–{end}步"


# Harnesses that wrap errors in JSON (e.g. Cursor's
# {"clientVisibleErrorMessage": "...", "modelVisibleErrorMessage": "..."}).
# The pattern is clipped upstream, so the value may lack its closing quote.
_JSON_ERROR_MESSAGE = re.compile(
    r'"(?:clientVisibleErrorMessage|modelVisibleErrorMessage|message|error)"\s*:\s*'
    r'"((?:[^"\\]|\\.)*)(")?'
)
_ERROR_WRAPPER = re.compile(r"</?tool_use_error>|^\s*error:\s*", re.IGNORECASE)

# Prefix-matched (case-insensitive) against the unwrapped message; any
# remainder after the match is kept verbatim after a "：". Unmatched messages
# stay in their original language.
_ERROR_MESSAGES_ZH: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), zh)
    for pattern, zh in (
        (r"exit code (-?\d+)", r"退出码 \1"),
        (r"status: (\w+)", r"状态：\1"),
        (r"unknown error", "未知错误"),
        (r"tool execution error", "工具执行出错"),
        (r"incorrect tool arguments", "工具参数错误"),
        (r"invalid arguments:\s*(\w+): required", r"参数无效，缺少 \1"),
        (r"invalid arguments", "参数无效"),
        (r"argument parsing failed", "参数解析失败"),
        (r"file not found|file does not exist", "文件不存在"),
        (r"(?:the )?string to replace (?:was )?not found in (?:the )?file", "文件中找不到要替换的文本"),
        (r"offset (\d+) is beyond file length \((\d+) lines\)", r"偏移量 \1 超出文件长度（共 \2 行）"),
        (r'tool "(.+?)" not found in namespace "(.+?)"', r"命名空间 \2 中没有工具 \1"),
        (r"no (?:matches|files) found", "未找到匹配项"),
        (r"permission denied", "权限不足"),
        (r"enoent: no such file or directory", "文件或目录不存在"),
        (r"eisdir: illegal operation on a directory", "不能对目录执行此操作"),
        (r"cancelled: parallel tool call", "已取消（同批并行调用出错）"),
    )
)


def _unwrap_error_message(raw: str) -> str:
    match = _JSON_ERROR_MESSAGE.search(raw)
    if match:
        value, closed = match.group(1), match.group(2)
        try:
            text = json.loads(f'"{value}"')
        except ValueError:
            text = value.replace("\\n", "\n").replace('\\"', '"')
        raw = text if closed else text.rstrip() + "…"
    return _ERROR_WRAPPER.sub("", raw).strip()


def _zh_error_message(raw: str) -> str:
    """Readable Chinese for a clustered tool error (unknown messages pass through)."""
    message = _unwrap_error_message(raw)
    for pattern, zh in _ERROR_MESSAGES_ZH:
        match = pattern.match(message)
        if match:
            translated = match.expand(zh)
            rest = message[match.end():].strip(" ,:.\n")
            return f"{translated}：{rest}" if rest else translated
    return " ".join(message.split()) or "未知错误"


def _from_failure_patterns(session: LoadedSession) -> list[OverviewIssue]:
    out: list[OverviewIssue] = []
    for i, pat in enumerate(session.failure_patterns or []):
        label = str(pat.get("cluster_label") or "Unknown error")
        message = _zh_error_message(str(pat.get("example_error") or ""))
        error_label = f"{pat['tool']}：{message}"
        count = int(pat.get("count") or 0)
        recovery = pat.get("recovery_path")
        steps = tuple(int(s) for s in (pat.get("steps") or []) if s is not None)
        error_class = str(pat.get("error_class") or "tool").lower()

        if error_class == "system":
            if count >= _SYSTEM_ERROR_HIGH:
                title = f"系统错误频发 · {_count_label(error_label, count)}"
                why = (
                    "读取、搜索、编辑类工具在本次运行中多次失败。单次失败通常风险不高，"
                    "但数量多往往说明路径、权限或运行环境配置有问题，值得检查。"
                )
            else:
                title = f"系统错误 · {_count_label(error_label, count)}"
                why = (
                    "读取、搜索或写入类工具的常见失败，通常风险不高，"
                    "智能体一般能自行恢复，不需要修改配置。"
                )
            out.append(
                OverviewIssue(
                    kind="antipattern",
                    title=title,
                    detail="",
                    why=why,
                    steps=steps,
                    source_id=f"fail:system:{i}:{label}",
                )
            )
            continue

        if recovery and isinstance(recovery, (list, tuple)):
            runs = [(name, len(list(group))) for name, group in groupby(str(t) for t in recovery)]
            why = (
                "这类错误要花额外的步骤和 token 才能恢复。常见恢复路径："
                + " → ".join(f"{name} ×{n}" if n > 1 else name for name, n in runs)
            )
        else:
            why = "出现这类错误后没有找到恢复路径，智能体可能并没有真正解决它。"
        out.append(
            OverviewIssue(
                kind="error",
                title=_count_label(error_label, count),
                detail="",
                why=why,
                steps=steps,
                source_id=f"fail:{i}:{label}",
            )
        )
    return out


def _from_failure_chains(session: LoadedSession) -> list[OverviewIssue]:
    """Consecutive assistant error runs (cascades). Skip length-1 (covered by clusters)."""
    out: list[OverviewIssue] = []
    chains = session.failure_chains or []
    for i, chain in enumerate(chains):
        steps = tuple(int(s) for s in (chain.get("steps") or []) if s is not None)
        if len(steps) < 2:
            continue
        start = chain.get("start", steps[0])
        end = chain.get("end", steps[-1])
        out.append(
            OverviewIssue(
                kind="error",
                title=_count_label("连续失败", len(steps), "步"),
                detail=_zh_step_range(start, end),
                why=(
                    "多个连续步骤都失败了，中间没有成功恢复；"
                    "一个根本错误常会这样被放大成反复试错。"
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
                    title=_count_label("工具调用错误", error_count),
                    detail="从工具输出中检测到（平台、权限、文件缺失等）",
                    why=(
                        "失败的工具调用要花额外的步骤和 token 来恢复，"
                        "而且往往是运行环境的问题，而不是智能体的错误。"
                    ),
                    steps=tuple(error_steps),
                    source_id="antipattern:tool_errors",
                )
            )

    streaks = session.fruitless_streaks or []
    if streaks:
        total_wasted = sum(int(s.get("length") or 0) for s in streaks)
        shown = streaks[:3]
        streak_desc = "、".join(
            f"{_zh_step_range(s.get('start_step'), s.get('end_step'))}（{s.get('length')} 步）"
            for s in shown
        )
        remaining = len(streaks) - len(shown)
        if remaining > 0:
            remaining_len = sum(int(s.get("length") or 0) for s in streaks[len(shown):])
            streak_desc += f"，另有 {remaining} 段（{remaining_len} 步）"
        streak_indices: list[int] = []
        for s in streaks:
            streak_indices.extend(
                _indices_for_step_range(s.get("start_step"), s.get("end_step"))
            )
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=_count_label("连续无结果搜索", len(streaks)),
                detail=f"浪费 {total_wasted} 步 — {streak_desc}",
                why=(
                    "连续 3 次或以上搜索都没有结果，"
                    "说明智能体很可能找错了地方，而不是在改进搜索条件。"
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
                title=_count_label("用 Bash 读取文件", len(tool_selection)),
                detail="使用 sed/cat/head 读取文件，而不是 Read 工具",
                why=(
                    "用 shell 命令读文件绕过了 Read 工具：没有行号、缓存效果差，"
                    "还会让上下文变大。"
                ),
                steps=tuple(bash_steps),
                source_id="antipattern:bash_read",
            )
        )

    stalled = (session.plan_metrics or {}).get("stalled") or []
    if stalled:
        items_desc = "、".join(f"“{s.get('content', '')[:30]}”" for s in stalled[:2])
        stall_steps: list[int] = []
        for s in stalled:
            stall_steps.extend(
                _indices_for_step_range(s.get("start_step"), s.get("end_step"))
            )
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=_count_label("停滞的待办项", len(stalled), "项"),
                detail=items_desc,
                why=(
                    "待办项一直处于进行中却没有完成，"
                    "通常是智能体转去做别的事后忘了收尾。"
                ),
                steps=tuple(stall_steps),
                source_id="antipattern:stalled_plan",
            )
        )

    plan_resets = int((session.plan_metrics or {}).get("plan_resets") or 0)
    if plan_resets > 0:
        plan_steps = sorted({
            int(snap["step"])
            for snap in (session.plan_history or [])
            if snap.get("step") is not None
        })
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=_count_label("计划重置", plan_resets),
                detail="待办列表被整体替换，新旧条目没有重叠",
                why=(
                    "运行中途整体重写待办列表，通常意味着智能体放弃了原来的思路，"
                    "而不是逐项完成或调整。"
                ),
                steps=tuple(plan_steps[:12]),
                source_id="antipattern:plan_resets",
            )
        )

    for i, thrash in enumerate(session.edit_thrash or []):
        path = str(thrash.get("path") or "")
        count = int(thrash.get("count") or 0)
        fail_count = int(thrash.get("fail_count") or 0)
        steps = tuple(int(s) for s in (thrash.get("steps") or []) if s is not None)
        short = path if len(path) <= 48 else ("…" + path[-47:])
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=_count_label(f"编辑失败后反复重试：{short}", count),
                detail=(
                    f"{_zh_step_range(thrash.get('start_step'), thrash.get('end_step'))}中"
                    f"有 {fail_count} 次写入失败"
                ),
                why=(
                    "同一个文件在写入失败后被反复修改，通常是路径错误、补丁对不上，"
                    "或需要先检查前提条件。"
                ),
                steps=steps,
                source_id=f"antipattern:edit_thrash:{i}:{path}",
            )
        )

    for i, rep in enumerate(session.repeated_searches or []):
        display = str(rep.get("display") or rep.get("signature") or "")
        count = int(rep.get("count") or 0)
        steps = tuple(int(s) for s in (rep.get("steps") or []) if s is not None)
        short = display if len(display) <= 48 else (display[:45] + "…")
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=_count_label("重复的空搜索", count),
                detail=short,
                why=(
                    "同一个搜索多次执行都没有结果（不只是连续出现），"
                    "搜索条件或路径很可能有误。"
                ),
                steps=steps,
                source_id=f"antipattern:repeated_search:{i}",
            )
        )

    return out


def _from_premature_compactions(session: LoadedSession) -> list[OverviewIssue]:
    """Compactions that ran before the window was full, or grew it."""
    out: list[OverviewIssue] = []
    flagged = session.premature_compactions or []
    if not flagged:
        return out

    def _fmt(n: object) -> str:
        return format_token_count(int(n)) if isinstance(n, (int, float)) else "?"

    grew = [c for c in flagged if c.get("grew")]
    if grew:
        worst = max(grew, key=lambda c: int(c.get("occupancy_after") or 0))
        detail = "，".join(
            f"第{c.get('step')}步：{_fmt(c.get('occupancy_before'))} → {_fmt(c.get('occupancy_after'))}"
            for c in grew[:4]
        )
        if len(grew) > 4:
            detail += f"，另有 {len(grew) - 4} 次"
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=_count_label("压缩后上下文反而变大", len(grew)),
                detail=detail,
                why=(
                    f"压缩后上下文窗口没有变小（最严重的一次："
                    f"{_fmt(worst.get('occupancy_before'))} → {_fmt(worst.get('occupancy_after'))}）："
                    "保存的摘要比被压缩掉的消息占用更多 token，这次压缩白做了。"
                ),
                steps=tuple(int(c.get("step", 0)) for c in grew),
                source_id="antipattern:compaction_grew",
            )
        )

    premature = [c for c in flagged if not c.get("grew")]
    if premature:
        pcts = [
            float(c["before_pct"])
            for c in premature
            if isinstance(c.get("before_pct"), (int, float))
        ]
        limit = _fmt(premature[0].get("window_limit"))
        pct_desc = (
            f"假定 {limit} token 窗口的 {min(pcts):.0f}–{max(pcts):.0f}%"
            if pcts
            else "远低于窗口上限"
        )
        out.append(
            OverviewIssue(
                kind="antipattern",
                title=_count_label("过早压缩上下文", len(premature)),
                detail=f"压缩时窗口占用：{pct_desc}",
                why=(
                    "上下文窗口远未用满就被压缩了，通常是智能体主动调用压缩，"
                    "而不是系统强制的。上下文过早丢失，请确认保留的摘要包含了关键信息。"
                ),
                steps=tuple(int(c.get("step", 0)) for c in premature),
                source_id="antipattern:compaction_premature",
            )
        )
    return out


_BOTTLENECK_WHY: dict[str, str] = {
    "idle": (
        "这一步开始前空闲了很久，通常是排队或限流导致的——"
        "应减少并发压力或调整等待策略，而不是修改提示词。"
    ),
    "tool": (
        "这一步的大部分时间花在工具上——可改用更轻量的工具、缩小范围，"
        "或缓存结果，避免重复执行耗时操作。"
    ),
    "context": (
        "这一步模型推理耗时异常长，而且上下文很大或缓存命中率低——"
        "可精简上下文、提高缓存复用，避免每轮重新读取大文件。"
    ),
    "inference": (
        "这一步几乎没有等待工具，但模型推理耗时异常长——"
        "通常是推理过长或提示词过大，可收紧指令或拆分任务。"
    ),
}


def _zh_duration(seconds: float) -> str:
    if round(seconds, 1) < 60:
        return f"{seconds:.1f} 秒"
    hours, rem = divmod(round(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours} 小时 {minutes} 分"
    return f"{minutes} 分 {secs} 秒"


def _dominant_tool_label(decomp: dict) -> str:
    dt = decomp.get("dominant_tool") or {}
    if not dt.get("name"):
        return ""
    return f"{dt['name']}: {dt['target']}" if dt.get("target") else str(dt["name"])


def _bottleneck_title(bn: dict, idx: int) -> str:
    decomp = bn.get("decomposition") or {}
    cause = bn["cause"]
    if cause == "idle":
        idle = _zh_duration(float(decomp.get("idle_s") or 0))
        return f"空闲/排队瓶颈：第{idx}步前等待 {idle}"
    if cause == "tool":
        tool = _zh_duration(float(decomp.get("tool_s") or 0))
        label = _dominant_tool_label(decomp)
        if not label:
            return f"工具瓶颈：第{idx}步（{tool}）"
        short = label if len(label) <= 40 else (label[:37] + "…")
        return f"工具瓶颈：{short}（{tool}）"
    duration = _zh_duration(float(bn.get("duration") or 0))
    if cause == "context":
        return f"上下文/缓存瓶颈：第{idx}步（{duration}）"
    return f"推理瓶颈：第{idx}步（{duration}）"


def _bottleneck_detail(bn: dict, idx: int) -> str:
    """Duration breakdown, largest component first; token/cache load for model-side causes."""
    decomp = bn.get("decomposition") or {}
    duration = float(bn.get("duration") or 0)
    parts: list[tuple[float, str]] = []

    tool_s = float(decomp.get("tool_s") or 0)
    if tool_s > 0:
        text = f"工具执行 {_zh_duration(tool_s)}"
        label = _dominant_tool_label(decomp)
        if label:
            dt_s = float(decomp["dominant_tool"].get("duration_s") or 0)
            text += f"（{label} {_zh_duration(dt_s)}）"
        parts.append((tool_s, text))
    inference_s = float(decomp.get("inference_s") or 0)
    if inference_s > 0:
        parts.append((inference_s, f"模型推理 {_zh_duration(inference_s)}"))
    idle_s = float(decomp.get("idle_s") or 0)
    if idle_s > 0:
        parts.append((idle_s, f"开始前空闲 {_zh_duration(idle_s)}（排队或限流）"))

    detail = f"第{idx}步耗时 {_zh_duration(duration)}"
    if parts:
        parts.sort(key=lambda p: p[0], reverse=True)
        detail += "：" + "，".join(text for _, text in parts)
    if decomp.get("timing_incomplete"):
        detail += "（计时不完整）"

    if bn["cause"] in ("context", "inference"):
        load: list[str] = []
        if bn["tokens"]:
            load.append(f"本步 {format_token_count(bn['tokens'])} token")
        if bn["cache_ratio"] is not None:
            load.append(f"缓存命中率 {bn['cache_ratio']:.0%}")
        if load:
            detail += "；" + "，".join(load)
    return detail[:200]


def _from_bottlenecks(session: LoadedSession) -> list[OverviewIssue]:
    """Map detected performance bottlenecks (outlier + clear cause), not top-N slow steps."""
    out: list[OverviewIssue] = []
    for i, bn in enumerate(session.performance_bottlenecks or []):
        step_idx = bn.get("step_idx")
        if step_idx is None:
            continue
        idx = int(step_idx)
        out.append(
            OverviewIssue(
                kind="bottleneck",
                title=_bottleneck_title(bn, idx),
                detail=_bottleneck_detail(bn, idx),
                why=_BOTTLENECK_WHY[bn["cause"]],
                steps=(idx,),
                source_id=f"bottleneck:{bn.get('cause', 'unknown')}:{i}:{idx}",
            )
        )
    return out


def _issue_card(issue: OverviewIssue, number: int) -> str:
    """Compact scan row: number + kind badge + title + step chips + optional LLM fix."""
    title = html.escape(issue.title)
    detail = html.escape(issue.detail)
    border = ISSUE_KIND_COLORS[issue.kind]
    steps_html = _step_link_chips(list(issue.steps), label="第{n}步", more="另有 {extra} 步")

    judgment_html = ""
    if issue.judgment is not None:
        j = issue.judgment
        also_html = ""
        if j.also:
            also_html = (
                f"<div style='font-size:11px;color:var(--ov-muted);margin-top:6px;"
                f"line-height:1.35;'>补充：{html.escape(j.also)}</div>"
            )
        judgment_html = (
            "<div style='margin-top:8px;font-size:12px;line-height:1.4;'>"
            "<span style='font-size:10px;font-weight:600;letter-spacing:0.04em;"
            "color:var(--ov-muted);'>修改位置</span>"
            f"<div style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
            f"font-size:12px;color:var(--ov-text);margin-top:2px;'>"
            f"{html.escape(j.where)}</div></div>"
            "<div style='margin-top:8px;padding:8px 10px;background:var(--ov-bg);"
            "border-radius:4px;'>"
            "<div style='font-size:10px;font-weight:600;letter-spacing:0.04em;"
            "color:var(--ov-muted);margin-bottom:4px;'>"
            f"修复建议 <span style='font-weight:500;letter-spacing:0;"
            f"color:var(--ov-muted);'>（置信度：{_CONFIDENCE_ZH[j.confidence]}）</span></div>"
            f"<div style='font-size:13px;line-height:1.4;color:var(--ov-text);'>"
            f"{html.escape(j.fix)}</div>"
            f"{also_html}</div>"
        )

    detail_html = (
        f"<span class='overview-issue-detail'>{detail}</span>" if issue.detail else ""
    )
    why_attr = info_html = ""
    if issue.why:
        why_attr = f" title='{html.escape(issue.why)}'"
        info_html = f"<span class='overview-issue-info'{why_attr} aria-hidden='true'>ⓘ</span>"
    return (
        f"<div class='overview-issue-card' style='border-left-color:{border};'>"
        f"<div class='overview-issue-head'>"
        f"<span class='overview-issue-index'>#{number}</span>"
        f"<span class='overview-issue-kind' style='color:{border};'>"
        f"{_ISSUE_KIND_LABELS_ZH[issue.kind]}</span>"
        f"<span class='overview-issue-title'{why_attr}>{title}</span>"
        f"{info_html}"
        f"{detail_html}"
        f"</div>"
        f"{steps_html}"
        f"{judgment_html}"
        f"</div>"
    )


def _issue_cards_html(issues: list[OverviewIssue]) -> str:
    """Render all issue cards (no preview cap), numbered from 1 in ranked order."""
    return "".join(
        _issue_card(issue, number) for number, issue in enumerate(issues, start=1)
    )


def render_overview_issues_html(
    issues: list[OverviewIssue],
    *,
    banner: str = "",
    progress: str = "",
) -> str:
    """Render a foldable Issues panel (healthy empty state when *issues* is empty).

    *progress* is a live status line shown inside the panel while the LLM judge runs.
    """
    banner_html = ""
    if banner:
        banner_html = (
            f"<div style='font-size:12px;color:var(--ov-muted);margin:0 0 8px;'>"
            f"{html.escape(banner)}</div>"
        )

    progress_html = ""
    if progress:
        # SMIL (not CSS) so prefers-reduced-motion / Gradio HTML swaps don't freeze it.
        progress_html = (
            "<div class='overview-issues-progress' role='status' aria-live='polite'>"
            "<span class='overview-issues-progress-label'>思考中…</span>"
            "<svg class='overview-issues-progress-spinner' width='14' height='14' "
            "viewBox='0 0 24 24' aria-hidden='true' "
            "style='flex-shrink:0;display:block'>"
            "<circle cx='12' cy='12' r='10' fill='none' "
            "stroke='currentColor' stroke-opacity='0.25' stroke-width='3'/>"
            "<path d='M12 2a10 10 0 0 1 10 10' fill='none' stroke='currentColor' "
            "stroke-width='3' stroke-linecap='round'>"
            "<animateTransform attributeName='transform' type='rotate' "
            "from='0 12 12' to='360 12 12' dur='0.7s' repeatCount='indefinite'/>"
            "</path></svg>"
            f"<span>{html.escape(progress)}</span>"
            "</div>"
        )

    if not issues:
        return (
            "<details class='overview-issues-panel' id='overview-issues'>"
            "<summary class='overview-issues-summary'>"
            "<span>Issues</span>"
            "<span class='overview-issues-summary-meta'>未检测到问题</span>"
            "</summary>"
            "<div class='overview-issues-body'>"
            f"{progress_html}"
            f"{banner_html}"
            "<div style='padding:8px 0 4px;color:var(--ov-muted);text-align:center;font-size:13px;'>"
            "未检测到明显的工作流问题。"
            "</div></div></details>"
        )

    count = len(issues)
    judged = sum(1 for i in issues if i.judgment is not None)
    count_label = f"{count} 个问题"
    if progress:
        count_label += " · 正在生成修复建议…"
    elif judged:
        count_label += f" · 已为 {judged} 个问题生成修复建议"

    lead = (
        "各问题下方为 LLM 给出的修改位置和修复建议"
        if judged and not progress
        else "按严重程度排序"
    )
    hint = (
        f"<div class='overview-issues-hint'>{lead} · 点击步骤可跳转到 Workflow"
        " · 鼠标悬停 ⓘ 查看说明</div>"
    )
    return (
        "<details class='overview-issues-panel' id='overview-issues' open>"
        "<summary class='overview-issues-summary'>"
        "<span>Issues</span>"
        f"<span class='overview-issues-summary-meta'>{html.escape(count_label)}</span>"
        "</summary>"
        "<div class='overview-issues-body'>"
        f"{progress_html}"
        f"{banner_html}"
        f"{hint}"
        + _issue_cards_html(issues)
        + "</div></details>"
    )


def build_overview_issues_html(
    session: LoadedSession,
    *,
    issues: list[OverviewIssue] | None = None,
    banner: str = "",
    progress: str = "",
) -> str:
    """Collect, rank, and render Overview Issues for *session*."""
    if issues is None:
        ranked = rank_issues(collect_overview_issues(session))
    else:
        ranked = list(issues)
    return render_overview_issues_html(ranked, banner=banner, progress=progress)
