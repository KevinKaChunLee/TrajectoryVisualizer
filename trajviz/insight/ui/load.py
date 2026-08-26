"""Load-path Gradio packer: LoadedSession / LoadError → a named slot dict."""

from __future__ import annotations

import html
import traceback
from typing import Any

import gradio as gr

from ..diagnostics import PRESSURE_ALL_AGENTS
from ..loaders import FORMAT_LABELS
from ..presenters import (
    DETAIL_PLACEHOLDER,
    FILTER_CHIPS_DEFAULT,
    build_chart_outputs,
    build_diagnostics_outputs,
    build_overview_outputs,
    build_workflow_outputs,
    empty_plotly_fig,
    load_warnings_html,
    raw_json_text,
    render_failure_patterns_html,
    render_tool_sequences_html,
)
from ..rendering import render_filter_chips
from ..session import LoadError, LoadedSession, load_session
from . import overview_tab, patterns_tab, raw_tab, upload, workflow_tab
from .shared import SharedState


def _slot_defaults(banner: str = "") -> dict[str, Any]:
    """Schema + empty values for every load slot. Adding a tab starts here."""
    fig = empty_plotly_fig()
    return {
        "main_tabs": gr.update(visible=False),
        "summary_area": gr.update(visible=bool(banner)),
        "upload_accordion": gr.update(),
        "state_steps": [],
        "summary_banner": banner,
        "anomaly_strip_html": "",
        "overview_kpi_html": "",
        "session_detail_html": "",
        "metrics_md": "",
        "token_chart": fig,
        "duration_chart": fig,
        "context_growth_chart": fig,
        "behavior_md": "",
        "tool_chart": fig,
        "tool_outcome_chart": fig,
        "agent_summary_html": "",
        "agent_token_chart": fig,
        "agent_swimlane_chart": fig,
        "diag_summary_html": "",
        "diag_pressure_html": "",
        "diag_pressure_agent": gr.update(
            choices=[("All agents", PRESSURE_ALL_AGENTS)],
            value=PRESSURE_ALL_AGENTS,
            visible=False,
        ),
        "diag_pressure_chart": fig,
        "diag_file_chart": fig,
        "diag_rootcause_html": "",
        "error_class_chart": fig,
        "plan_timeline_chart": fig,
        "hotspots_md": "",
        "per_message_md": "",
        "wf_filter_chips_html": render_filter_chips(),
        "wf_filter_hidden": ",".join(FILTER_CHIPS_DEFAULT),
        "wf_count_html": "",
        "toc_html": "",
        "workflow_html": "<div></div>",
        "detail_store": "",
        "detail_html": DETAIL_PLACEHOLDER,
        "raw_json": "",
        "patterns_tool_html": "",
        "patterns_failure_html": "",
        "antipattern_summary_html": "",
        "state_raw": {},
    }


_LOAD_SLOT_KEYS: frozenset[str] | None = None


def load_slot_keys() -> frozenset[str]:
    """Names the packer and `merge_load_slots` must agree on."""
    global _LOAD_SLOT_KEYS
    if _LOAD_SLOT_KEYS is None:
        _LOAD_SLOT_KEYS = frozenset(_slot_defaults())
    return _LOAD_SLOT_KEYS


def _overlay(packed: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    extra = updates.keys() - packed.keys()
    if extra:
        raise ValueError(f"Unknown load slot keys: {sorted(extra)}")
    packed.update(updates)
    return packed


def merge_load_slots(
    *,
    main_tabs,
    shared: SharedState,
    upload_refs: upload.UploadRefs,
    overview: overview_tab.OverviewRefs,
    patterns: patterns_tab.PatternsRefs,
    workflow: workflow_tab.WorkflowRefs,
    raw: raw_tab.RawRefs,
) -> dict[str, Any]:
    """Component map for `do_load`. A new tab adds `**tab.load_slots(refs)` here."""
    slots = {
        "main_tabs": main_tabs,
        "state_steps": shared.state_steps,
        "state_raw": shared.state_raw,
        **upload.load_slots(upload_refs),
        **overview_tab.load_slots(overview),
        **patterns_tab.load_slots(patterns),
        **workflow_tab.load_slots(workflow),
        **raw_tab.load_slots(raw),
    }
    expected = load_slot_keys()
    got = frozenset(slots)
    if got != expected:
        raise ValueError(
            f"Load slot keys mismatch vs packer: missing={sorted(expected - got)} extra={sorted(got - expected)}"
        )
    return slots


def resolve_upload_path(upload_obj) -> str | None:
    """Filesystem path from a Gradio File value (string or file-like)."""
    if upload_obj is None:
        return None
    if isinstance(upload_obj, str):
        return upload_obj
    name = getattr(upload_obj, "name", None)
    return name if name else None


def empty_load_outputs(banner: str = "", detail: str = "*No data*") -> dict[str, Any]:
    """Named empty/error outputs. Keys match `pack_load_outputs`."""
    if detail and detail != "*No data*":
        banner += (
            f"<p style='color:var(--ov-muted);font-size:13px;margin:4px 0 0;'>{html.escape(detail.strip('*'))}</p>"
        )
    return _slot_defaults(banner=banner)


def _pack_error(err: LoadError) -> dict[str, Any]:
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
    return empty_load_outputs(
        banner=f"<p style='color:#dc2626;'>Error: {html.escape(err.message)}</p>",
        detail="*Error loading file.*",
    )


def pack_load_outputs(result: LoadedSession | LoadError, dark: bool = False) -> dict[str, Any]:
    """Map a load result onto named Gradio slots."""
    if isinstance(result, LoadError):
        return _pack_error(result)

    ov = build_overview_outputs(result)
    ch = build_chart_outputs(result, dark=dark)
    dg = build_diagnostics_outputs(result, dark=dark)
    wf = build_workflow_outputs(result.steps)
    warnings = load_warnings_html(result)
    return _overlay(
        _slot_defaults(),
        {
            "main_tabs": gr.update(visible=True),
            "summary_area": gr.update(visible=True),
            "state_steps": result.steps,
            "summary_banner": warnings + ov["banner"],
            "anomaly_strip_html": ov["anomaly_html"],
            "overview_kpi_html": ov["kpi_html"],
            "session_detail_html": ov["session_detail"],
            "metrics_md": ov["metrics_text"],
            "token_chart": ch["tok_fig"],
            "duration_chart": ch["dur_fig"],
            "context_growth_chart": ch["context_growth_fig"],
            "behavior_md": ov["behavior_text"],
            "tool_chart": ch["tl_fig"],
            "tool_outcome_chart": ch["tool_outcome_fig"],
            "agent_summary_html": ch["agent_cards_html"],
            "agent_token_chart": ch["agent_tok_fig"],
            "agent_swimlane_chart": ch["swimlane_fig"],
            "diag_summary_html": dg["diag_summary_html"],
            "diag_pressure_html": dg["diag_pressure_html"],
            "diag_pressure_agent": gr.update(
                choices=dg["diag_pressure_dropdown"]["choices"],
                value=dg["diag_pressure_dropdown"]["value"],
                visible=dg["diag_pressure_dropdown"]["visible"],
            ),
            "diag_pressure_chart": dg["diag_pressure_chart"],
            "diag_file_chart": dg["diag_file_chart"],
            "diag_rootcause_html": dg["diag_rootcause_html"],
            "error_class_chart": ch["error_class_fig"],
            "plan_timeline_chart": ch["plan_timeline_fig"],
            "hotspots_md": ov["hotspots_text"],
            "per_message_md": ov["per_message_text"],
            "wf_filter_chips_html": wf["wf_chips"],
            "wf_filter_hidden": wf["wf_filter_val"],
            "wf_count_html": wf["wf_count"],
            "toc_html": wf["toc_html_val"],
            "workflow_html": wf["wf_html"],
            "detail_store": wf["detail_store_val"],
            "detail_html": DETAIL_PLACEHOLDER,
            "raw_json": raw_json_text(result.raw),
            "patterns_tool_html": render_tool_sequences_html(result.tool_sequences),
            "patterns_failure_html": render_failure_patterns_html(result.failure_patterns),
            "antipattern_summary_html": ch["antipattern_html"],
            "state_raw": result.raw,
        },
    )


def do_load(upload_obj, dark=False, selected_format="", *, slots: dict[str, Any] | None = None):
    """Load wrapper: surface unexpected failures as a visible banner.

    When *slots* is provided (the live dashboard), return a Gradio component
    dict. Tests call ``pack_load_outputs`` / ``empty_load_outputs`` directly.
    """
    try:
        file_path = resolve_upload_path(upload_obj)
        result = load_session(file_path or "", format_hint=selected_format or "")
        packed = pack_load_outputs(result, dark=bool(dark))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        packed = empty_load_outputs(
            banner=(f"<p style='color:#dc2626;'>Error loading trajectory: {html.escape(str(exc))}</p>"),
            detail="*Failed to load — see console for details.*",
        )
    if slots is None:
        return packed
    return {slots[key]: packed[key] for key in slots}


def bind_load(*, file_upload, load_btn, format_selector, state_dark, slots: dict[str, Any]):
    """Wire Load Trajectory click and auto-load on file change. Returns both events."""
    expected = load_slot_keys()
    got = frozenset(slots)
    if got != expected:
        raise ValueError(
            f"Load slot keys mismatch vs packer: missing={sorted(expected - got)} extra={sorted(got - expected)}"
        )

    def _do_load(upload_obj, dark=False, selected_format=""):
        return do_load(upload_obj, dark, selected_format, slots=slots)

    outputs = list(slots.values())
    _load_ev = load_btn.click(
        fn=_do_load,
        inputs=[file_upload, state_dark, format_selector],
        outputs=outputs,
    )
    _upload_ev = file_upload.change(
        fn=_do_load,
        inputs=[file_upload, state_dark, format_selector],
        outputs=outputs,
    )
    return _load_ev, _upload_ev
