"""Self-contained HTML snapshot of Overview, charts, and Patterns.

The live dashboard is Gradio. This module reuses the same builders and writes
a single file that opens in a browser without the server — charts stay
interactive via Plotly's CDN bundle.
"""

from __future__ import annotations

import html
import os
import re
import tempfile
from datetime import UTC, datetime
from typing import Any

import plotly.graph_objects as go

from trajviz.insight.analytics import compute_step_analytics
from trajviz.insight.formatting import wall_clock_fmt
from trajviz.insight.loaders import FORMAT_LABELS, detect_format
from trajviz.insight.metrics import (
    build_message_metrics,
    compute_agent_summary,
    compute_health_verdict,
    compute_metrics,
    extract_agent_info,
)
from trajviz.insight.parser import parse_steps
from trajviz.insight.patterns import detect_failure_patterns, detect_tool_sequences
from trajviz.insight.styles import APP_CSS

_MAX_STEPS = 2000
_PLOTLY_CDN = "cdn"

_MD_TABLE_SEP = re.compile(r"^\|[\s:|\-]+\|$")
_HEADING = re.compile(r"^(#{2,3})\s+(.*)$")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")
_STYLE_TAG = re.compile(r"</?style[^>]*>", re.IGNORECASE)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


class ReportError(ValueError):
    """Raised when a trajectory cannot be turned into a report."""


def build_report_html(
    raw: dict,
    *,
    steps: list[dict] | None = None,
    file_path: str | None = None,
    dark: bool = False,
) -> str:
    """Return a full HTML document for *raw* (already loaded via ``load_trajectory``)."""
    if not isinstance(raw, dict) or not raw:
        raise ReportError("Load a trajectory first.")
    if raw.get("_error"):
        raise ReportError(str(raw["_error"]))

    src = file_path or raw.get("_source_path") or "trajectory"
    if steps is None:
        steps = parse_steps(raw)
    steps_total = len(steps)
    truncated_note = ""
    if steps_total > _MAX_STEPS:
        steps = steps[:_MAX_STEPS]
        truncated_note = (
            f"Showing first {_MAX_STEPS:,} of {steps_total:,} steps "
            f"({steps_total - _MAX_STEPS:,} truncated), same cap as the dashboard."
        )

    detected = detect_format(raw)
    message_rows = build_message_metrics(steps)
    metrics = compute_metrics(steps, raw, message_rows=message_rows)
    _, wfmt = wall_clock_fmt(metrics)
    step_analytics = compute_step_analytics(steps)
    verdicts = compute_health_verdict(metrics, step_analytics if steps else [])
    agent_summaries = compute_agent_summary(steps, raw)

    # Lazy import: insight.py imports this module at UI-build time.
    from .insight import (
        _build_chart_outputs,
        _build_diagnostics_outputs,
        _build_overview_outputs,
        _build_session_detail_html,
        _render_failure_patterns_html,
        _render_tool_sequences_html,
    )

    dg = _build_diagnostics_outputs(
        steps, step_analytics, agent_summaries,
        dark=dark, trajectory_format=detected, raw=raw,
    )
    ov = _build_overview_outputs(
        steps, raw, metrics, message_rows, verdicts, wfmt, src,
        trajectory_format=detected,
    )
    meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    timing = raw.get("timing") if isinstance(raw.get("timing"), dict) else {}
    model_id, _, agent_id = extract_agent_info(steps)
    session_detail = _build_session_detail_html(
        timing, meta, model_id=model_id, agent_id=agent_id,
    )
    ch = _build_chart_outputs(
        steps, message_rows, agent_summaries, dark=dark, trajectory_format=detected,
    )
    pat_tool = _render_tool_sequences_html(detect_tool_sequences(steps))
    pat_fail = _render_failure_patterns_html(detect_failure_patterns(steps))

    figures: list[tuple[str, go.Figure]] = [
        ("Token usage", ch["tok_fig"]),
        ("Step duration", ch["dur_fig"]),
        ("Context growth", ch["context_growth_fig"]),
        ("Tool-call frequency", ch["tl_fig"]),
        ("Tool outcome timeline", ch["tool_outcome_fig"]),
        ("Tokens by agent", ch["agent_tok_fig"]),
        ("Agent swimlane", ch["swimlane_fig"]),
        ("Context-window pressure", dg["diag_pressure_chart"]),
        ("File interactions", dg["diag_file_chart"]),
        ("Tool-error classification", ch["error_class_fig"]),
        ("Plan timeline", ch["plan_timeline_fig"]),
    ]

    basename = os.path.basename(src) or "trajectory"
    fmt_label = FORMAT_LABELS.get(detected, detected or "unknown")
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    title = f"TrajViz report — {basename}"

    include_js: str | bool = _PLOTLY_CDN

    def charts(items: list[tuple[str, go.Figure]]) -> str:
        nonlocal include_js
        html_part, include_js = _charts_html(items, include_js)
        return html_part

    sections: list[str] = [
        _header_html(title, basename, fmt_label, generated, truncated_note),
        _section("Summary", ov["banner"] + ov["anomaly_html"] + ov["kpi_html"] + session_detail),
        _section("Performance", _mixed_md_to_html(ov["metrics_text"])),
        charts(figures[:3]),
        _section("Tools", _mixed_md_to_html(ov["behavior_text"])),
        charts(figures[3:5]),
        _section("Agents", ch["agent_cards_html"]),
        charts(figures[5:7]),
        _section(
            "Diagnostics",
            dg["diag_summary_html"] + dg["diag_pressure_html"] + dg["diag_rootcause_html"],
        ),
        charts(figures[7:11]),
        _section("Hotspots", _mixed_md_to_html(ov["hotspots_text"])),
        _section("Per-message metrics", _mixed_md_to_html(ov["per_message_text"])),
        _section(
            "Patterns",
            "<h3>Recurring tool sequences</h3>" + pat_tool
            + "<h3>Failure patterns</h3>" + pat_fail
            + ch["antipattern_html"],
        ),
    ]

    body = "\n".join(s for s in sections if s)
    return (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        f"<meta charset='utf-8'>\n<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        f"<title>{html.escape(title)}</title>\n"
        f"<style>\n{_embedded_css()}\n</style>\n"
        "</head>\n<body>\n"
        f"<article class='tv-report'>\n{body}\n</article>\n"
        "</body>\n</html>\n"
    )


def write_report_file(
    raw: dict,
    dest: str | None = None,
    *,
    steps: list[dict] | None = None,
    file_path: str | None = None,
    dark: bool = False,
) -> str:
    """Write the report to *dest* (file or directory) and return the path."""
    src = file_path or (raw.get("_source_path") if isinstance(raw, dict) else None) or "trajectory"
    name = _report_basename(str(src))
    if dest is None:
        dest = os.path.join(tempfile.mkdtemp(prefix="trajviz-report-"), name)
    elif os.path.isdir(dest):
        dest = os.path.join(dest, name)
    html_out = build_report_html(raw, steps=steps, file_path=str(src), dark=dark)
    parent = os.path.dirname(os.path.abspath(dest))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as handle:
        handle.write(html_out)
    return dest


def _report_basename(file_path: str) -> str:
    stem = os.path.splitext(os.path.basename(file_path or "trajectory"))[0] or "trajectory"
    safe = re.sub(r"[^\w.-]+", "-", stem).strip("-.") or "trajectory"
    return f"{safe}-trajviz-report.html"


def _embedded_css() -> str:
    app = _STYLE_TAG.sub("", APP_CSS)
    app = _HTML_COMMENT.sub("", app)
    layout = """
body { margin: 0; background: var(--ov-bg, #f6f8fc); color: var(--ov-text, #0f172a); }
.tv-report { max-width: 1100px; margin: 0 auto; padding: 24px 20px 72px; }
.tv-report-kicker { font-size: 12px; color: var(--ov-muted); letter-spacing: 0.04em; text-transform: uppercase; }
.tv-report h1 { font-size: 22px; margin: 4px 0 8px; letter-spacing: -0.02em; }
.tv-meta { font-size: 13px; color: var(--ov-muted); margin-bottom: 20px; }
.tv-note { margin: 8px 0 16px; padding: 8px 12px; background: var(--ov-anomaly-bg, #fef3c7);
  border-left: 3px solid var(--ov-warn, #b45309); color: var(--ov-anomaly-text, #92400e); font-size: 13px; }
.tv-section { margin: 28px 0; }
.tv-section h2 { font-size: 16px; margin: 0 0 12px; padding-bottom: 6px;
  border-bottom: 1px solid var(--ov-border, #dce3ef); }
.tv-section h3 { font-size: 14px; margin: 16px 0 8px; }
.tv-chart { margin: 8px 0 20px; }
.tv-md table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 8px 0 16px; }
.tv-md th, .tv-md td { border-bottom: 1px solid var(--ov-border, #dce3ef); padding: 6px 10px; text-align: left; }
.tv-md th { background: var(--ov-table-header-bg, #f1f5f9); font-size: 11px; text-transform: uppercase; }
.tv-md p { margin: 0 0 8px; }
.tv-md code { font-size: 12px; background: var(--ov-code-bg, #eef3ff); padding: 1px 4px; border-radius: 3px; }
@media print {
  .tv-chart { break-inside: avoid; }
}
"""
    return layout + "\n" + app


def _header_html(title: str, basename: str, fmt_label: str, generated: str, note: str) -> str:
    note_html = f"<p class='tv-note'>{html.escape(note)}</p>" if note else ""
    return (
        "<header>"
        "<div class='tv-report-kicker'>TrajViz HTML report</div>"
        f"<h1>{html.escape(basename)}</h1>"
        f"<p class='tv-meta'>{html.escape(fmt_label)} &middot; generated {html.escape(generated)}"
        " &middot; Overview, charts, and Patterns (not a live Gradio session)</p>"
        f"{note_html}"
        "</header>"
    )


def _section(title: str, inner: str) -> str:
    if not (inner or "").strip():
        return ""
    return f"<section class='tv-section' id='{html.escape(_slug(title))}'><h2>{html.escape(title)}</h2>{inner}</section>"


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _figure_is_empty(fig: Any) -> bool:
    if fig is None or not isinstance(fig, go.Figure):
        return True
    return len(fig.data) == 0


def _charts_html(items: list[tuple[str, go.Figure]], include_js: str | bool) -> tuple[str, str | bool]:
    parts: list[str] = []
    js = include_js
    for title, fig in items:
        if _figure_is_empty(fig):
            continue
        div = fig.to_html(
            full_html=False,
            include_plotlyjs=js,
            config={"displaylogo": False, "responsive": True},
        )
        js = False
        parts.append(
            f"<section class='tv-section tv-chart' id='{html.escape(_slug(title))}'>"
            f"<h2>{html.escape(title)}</h2>{div}</section>"
        )
    return "\n".join(parts), js


def _mixed_md_to_html(text: str) -> str:
    """Render dashboard markdown that already embeds HTML metric chips."""
    if not text or not str(text).strip():
        return ""
    lines = str(text).splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        raw_line = lines[i]
        stripped = raw_line.strip()
        heading = _HEADING.match(stripped)
        if heading:
            tag = "h2" if heading.group(1) == "##" else "h3"
            out.append(f"<{tag}>{_inline(heading.group(2))}</{tag}>")
            i += 1
            continue
        if (
            stripped.startswith("|")
            and i + 1 < n
            and _MD_TABLE_SEP.match(lines[i + 1].strip())
        ):
            block = [stripped, lines[i + 1].strip()]
            i += 2
            while i < n and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            out.append(_md_table_to_html(block))
            continue
        if stripped.startswith("<"):
            out.append(raw_line)
            i += 1
            continue
        if not stripped:
            i += 1
            continue
        out.append(f"<p>{_inline(stripped)}</p>")
        i += 1
    return "<div class='tv-md'>" + "\n".join(out) + "</div>"


def _md_table_to_html(block: list[str]) -> str:
    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header = cells(block[0])
    body_rows = [cells(row) for row in block[2:]]
    head = "".join(f"<th>{_inline(c)}</th>" for c in header)
    rows = []
    for row in body_rows:
        padded = row + [""] * (len(header) - len(row))
        rows.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in padded[: len(header)]) + "</tr>")
    return (
        "<table><thead><tr>" + head + "</tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def _inline(text: str) -> str:
    if "<" in text and ">" in text:
        text = _BOLD.sub(r"<strong>\1</strong>", text)
        return _CODE.sub(r"<code>\1</code>", text)
    escaped = html.escape(text)
    escaped = _BOLD.sub(r"<strong>\1</strong>", escaped)
    escaped = _CODE.sub(r"<code>\1</code>", escaped)
    if escaped.startswith("*") and escaped.endswith("*") and len(escaped) > 2:
        escaped = f"<em>{escaped[1:-1]}</em>"
    return escaped
