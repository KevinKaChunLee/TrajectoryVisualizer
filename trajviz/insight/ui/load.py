"""Load-path Gradio packer: LoadedSession / LoadError → the 40-output tuple."""

from __future__ import annotations

import html
import traceback

import gradio as gr
import plotly.graph_objects as go

from ..diagnostics import PRESSURE_ALL_AGENTS
from ..loaders import FORMAT_LABELS
from ..presenters import (
    DETAIL_PLACEHOLDER,
    FILTER_CHIPS_DEFAULT,
    _build_chart_outputs,
    _build_diagnostics_outputs,
    _build_overview_outputs,
    _build_workflow_outputs,
    _render_failure_patterns_html,
    _render_tool_sequences_html,
    load_warnings_html,
    raw_json_text,
)
from ..rendering import render_filter_chips
from ..session import LoadError, LoadedSession, load_session


def _empty_fig() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def resolve_upload_path(upload_obj) -> str | None:
    """Filesystem path from a Gradio File value (string or file-like)."""
    if upload_obj is None:
        return None
    if isinstance(upload_obj, str):
        return upload_obj
    name = getattr(upload_obj, "name", None)
    return name if name else None


def empty_load_outputs(banner: str = "", detail: str = "*No data*") -> tuple:
    """Return the empty outputs tuple for error states (40 slots)."""
    f = _empty_fig()
    if detail and detail != "*No data*":
        banner += (
            "<p style='color:var(--ov-muted);font-size:13px;margin:4px 0 0;'>"
            f"{html.escape(detail.strip('*'))}</p>"
        )
    return (
        gr.update(visible=False),  # main_tabs
        gr.update(visible=bool(banner)),  # summary_area
        gr.update(),  # upload_accordion
        [],  # state_steps
        banner,  # summary_banner
        "",  # anomaly_strip_html
        "",  # overview_kpi_html
        "",  # session_detail_html
        "",  # metrics_md
        f,
        f,  # token_chart, duration_chart
        f,  # context_growth_chart
        "",  # behavior_md
        f,  # tool_chart
        f,  # tool_outcome_chart
        "",  # agent_summary_html
        f,  # agent_token_chart
        f,  # agent_swimlane_chart
        "",  # diag_summary_html
        "",  # diag_pressure_html
        gr.update(
            choices=[("All agents", PRESSURE_ALL_AGENTS)],
            value=PRESSURE_ALL_AGENTS,
            visible=False,
        ),  # diag_pressure_agent
        f,  # diag_pressure_chart
        f,  # diag_file_chart
        "",  # diag_rootcause_html
        f,  # error_class_chart
        f,  # plan_timeline_chart
        "",  # hotspots_md
        "",  # per_message_md
        render_filter_chips(),  # wf_filter_chips_html
        ",".join(FILTER_CHIPS_DEFAULT),  # wf_filter_hidden
        "",  # wf_count_html
        "",  # toc_html
        "<div></div>",  # workflow_html
        "",  # detail_store
        DETAIL_PLACEHOLDER,  # detail_html
        "",  # raw_json
        "",  # patterns_tool_html
        "",  # patterns_failure_html
        "",  # antipattern_summary_html
        {},  # state_raw
    )


def _pack_error(err: LoadError) -> tuple:
    if err.code == "not_found":
        return empty_load_outputs(detail=f"*{err.message}*")
    if err.code == "mismatch":
        selected_key = err.selected or ""
        detected_key = err.detected or ""
        err_msg = (
            f"Format mismatch: selected "
            f"<b>{html.escape(FORMAT_LABELS.get(selected_key, selected_key))}</b>"
            f" but file detected as "
            f"<b>{html.escape(FORMAT_LABELS.get(detected_key, detected_key))}</b>."
        )
        return empty_load_outputs(
            banner=f"<p style='color:#dc2626;'>{err_msg}</p>",
            detail="*Please select the correct format and try again.*",
        )
    if err.code == "unknown":
        return empty_load_outputs(
            banner=f"<p style='color:#dc2626;'>{html.escape(err.message)}</p>",
            detail="*Unrecognized trajectory file.*",
        )
    # read_error
    return empty_load_outputs(
        banner=f"<p style='color:#dc2626;'>Error: {html.escape(err.message)}</p>",
        detail="*Error loading file.*",
    )


def pack_load_outputs(result: LoadedSession | LoadError, dark: bool = False) -> tuple:
    """Map a load result onto the 40 Gradio outputs `do_load` returns."""
    if isinstance(result, LoadError):
        return _pack_error(result)

    ov = _build_overview_outputs(result)
    ch = _build_chart_outputs(result, dark=dark)
    dg = _build_diagnostics_outputs(result, dark=dark)
    wf = _build_workflow_outputs(result.steps)
    warnings = load_warnings_html(result)
    return (
        gr.update(visible=True),  # main_tabs
        gr.update(visible=True),  # summary_area
        gr.update(),  # upload_accordion
        result.steps,  # state_steps
        warnings + ov["banner"],  # summary_banner
        ov["anomaly_html"],
        ov["kpi_html"],
        ov["session_detail"],
        ov["metrics_text"],
        ch["tok_fig"],
        ch["dur_fig"],
        ch["context_growth_fig"],
        ov["behavior_text"],
        ch["tl_fig"],
        ch["tool_outcome_fig"],
        ch["agent_cards_html"],
        ch["agent_tok_fig"],
        ch["swimlane_fig"],
        dg["diag_summary_html"],
        dg["diag_pressure_html"],
        gr.update(
            choices=dg["diag_pressure_dropdown"]["choices"],
            value=dg["diag_pressure_dropdown"]["value"],
            visible=dg["diag_pressure_dropdown"]["visible"],
        ),
        dg["diag_pressure_chart"],
        dg["diag_file_chart"],
        dg["diag_rootcause_html"],
        ch["error_class_fig"],
        ch["plan_timeline_fig"],
        ov["hotspots_text"],
        ov["per_message_text"],
        wf["wf_chips"],
        wf["wf_filter_val"],
        wf["wf_count"],
        wf["toc_html_val"],
        wf["wf_html"],
        wf["detail_store_val"],
        DETAIL_PLACEHOLDER,
        raw_json_text(result.raw),
        _render_tool_sequences_html(result.tool_sequences),
        _render_failure_patterns_html(result.failure_patterns),
        ch["antipattern_html"],
        result.raw,
    )


def do_load(upload_obj, dark=False, selected_format=""):
    """Load wrapper: surface unexpected failures as a visible banner."""
    try:
        file_path = resolve_upload_path(upload_obj)
        result = load_session(file_path or "", format_hint=selected_format or "")
        return pack_load_outputs(result, dark=bool(dark))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return empty_load_outputs(
            banner=(f"<p style='color:#dc2626;'>Error loading trajectory: {html.escape(str(exc))}</p>"),
            detail="*Failed to load — see console for details.*",
        )


def bind_load(*, file_upload, load_btn, format_selector, state_dark, all_outputs):
    """Wire Load Trajectory click and auto-load on file change. Returns both events."""
    _load_ev = load_btn.click(
        fn=do_load,
        inputs=[file_upload, state_dark, format_selector],
        outputs=all_outputs,
    )
    _upload_ev = file_upload.change(
        fn=do_load,
        inputs=[file_upload, state_dark, format_selector],
        outputs=all_outputs,
    )
    return _load_ev, _upload_ev
