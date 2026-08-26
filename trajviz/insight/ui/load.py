"""Load-path Gradio packer: merge per-tab named slot dicts."""

from __future__ import annotations

import html
import traceback
from collections.abc import Callable, Mapping
from typing import Any

import gradio as gr

from ..loaders import FORMAT_LABELS
from ..session import LoadError, LoadedSession, load_session
from . import overview_tab, patterns_tab, raw_tab, upload, workflow_tab
from .shared import SharedState

PackFn = Callable[..., dict[str, Any]]


def pack_shell(session: LoadedSession | None = None, *, dark: bool = False, banner: str = "") -> dict[str, Any]:
    """Tabs visibility and Gradio state filled on load."""
    del dark, banner
    if session is None:
        return {
            "main_tabs": gr.update(visible=False),
            "state_steps": [],
            "state_raw": {},
        }
    return {
        "main_tabs": gr.update(visible=True),
        "state_steps": session.steps,
        "state_raw": session.raw,
    }


_PACKERS: tuple[PackFn, ...] = (
    pack_shell,
    upload.pack_load,
    overview_tab.pack_load,
    patterns_tab.pack_load,
    workflow_tab.pack_load,
    raw_tab.pack_load,
)


def merge_packer_dicts(parts: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Union packer dicts; duplicate keys are a bug."""
    packed: dict[str, Any] = {}
    for part in parts:
        overlap = packed.keys() & part.keys()
        if overlap:
            raise ValueError(f"Duplicate load slot keys: {sorted(overlap)}")
        packed.update(part)
    return packed


def collect_pack(session: LoadedSession | None = None, *, dark: bool = False, banner: str = "") -> dict[str, Any]:
    """Run every tab packer and merge."""
    return merge_packer_dicts([pack(session, dark=dark, banner=banner) for pack in _PACKERS])


_LOAD_SLOT_KEYS: frozenset[str] | None = None


def load_slot_keys() -> frozenset[str]:
    """Union of keys every packer emits (empty load)."""
    global _LOAD_SLOT_KEYS
    if _LOAD_SLOT_KEYS is None:
        _LOAD_SLOT_KEYS = frozenset(collect_pack())
    return _LOAD_SLOT_KEYS


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
    slots = merge_packer_dicts(
        [
            {
                "main_tabs": main_tabs,
                "state_steps": shared.state_steps,
                "state_raw": shared.state_raw,
            },
            upload.load_slots(upload_refs),
            overview_tab.load_slots(overview),
            patterns_tab.load_slots(patterns),
            workflow_tab.load_slots(workflow),
            raw_tab.load_slots(raw),
        ]
    )
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
    return collect_pack(banner=banner)


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
    packed = collect_pack(result, dark=dark)
    expected = load_slot_keys()
    got = frozenset(packed)
    if got != expected:
        raise ValueError(f"Load packer keys mismatch: missing={sorted(expected - got)} extra={sorted(got - expected)}")
    return packed


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
