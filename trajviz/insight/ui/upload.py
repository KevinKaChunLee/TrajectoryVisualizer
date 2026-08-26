"""Upload row, title, and summary banner."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from ..loaders import FORMAT_DROPDOWN_CHOICES
from ..presenters.overview import build_summary_outputs, load_warnings_html
from ..session import LoadedSession


@dataclass
class UploadRefs:
    upload_accordion: gr.Column
    format_selector: gr.Dropdown
    file_upload: gr.File
    load_btn: gr.Button
    label_file_upload: gr.File
    label_load_btn: gr.Button
    summary_area: gr.Column
    summary_banner: gr.HTML
    label_badge_html: gr.HTML
    anomaly_strip_html: gr.HTML


def layout() -> UploadRefs:
    gr.HTML(
        "<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:4px;'>"
        "<span style='font-size:20px;font-weight:700;letter-spacing:-0.02em;'>TrajViz</span>"
        "<span style='font-size:12px;color:var(--ov-muted);'>Coding Agent Trajectory Analysis Dashboard</span>"
        "</div>"
    )

    with gr.Column(elem_classes=["upload-row"]) as upload_accordion, gr.Row(equal_height=True):
        format_selector = gr.Dropdown(
            label="Format",
            choices=FORMAT_DROPDOWN_CHOICES,
            value="",
            interactive=True,
            scale=1,
            min_width=140,
        )
        with gr.Column(scale=2, min_width=200):
            file_upload = gr.File(
                label="Trajectory (.json / .jsonl / .zip)",
                file_types=[".json", ".jsonl", ".zip"],
                height=110,
            )
            load_btn = gr.Button("Load Trajectory", variant="primary", size="sm", min_width=120)
        with gr.Column(scale=2, min_width=200):
            label_file_upload = gr.File(
                label="Labels (optional)",
                file_types=[".json"],
                height=110,
            )
            label_load_btn = gr.Button("Load Labels", variant="secondary", size="sm", min_width=120)

    with gr.Column(visible=False) as summary_area:
        summary_banner = gr.HTML("", elem_classes=["summary-banner"])
        label_badge_html = gr.HTML("")
        anomaly_strip_html = gr.HTML("")

    return UploadRefs(
        upload_accordion=upload_accordion,
        format_selector=format_selector,
        file_upload=file_upload,
        load_btn=load_btn,
        label_file_upload=label_file_upload,
        label_load_btn=label_load_btn,
        summary_area=summary_area,
        summary_banner=summary_banner,
        label_badge_html=label_badge_html,
        anomaly_strip_html=anomaly_strip_html,
    )


def load_slots(refs: UploadRefs) -> dict:
    """Named Gradio components filled by the load packer."""
    return {
        "summary_area": refs.summary_area,
        "upload_accordion": refs.upload_accordion,
        "summary_banner": refs.summary_banner,
        "anomaly_strip_html": refs.anomaly_strip_html,
    }


def pack_load(session: LoadedSession | None = None, *, dark: bool = False, banner: str = "") -> dict:
    """Named upload-row values for a load (error banner when *session* is None)."""
    del dark
    if session is None:
        return {
            "summary_area": gr.update(visible=bool(banner)),
            "upload_accordion": gr.update(),
            "summary_banner": banner,
            "anomaly_strip_html": "",
        }
    summary = build_summary_outputs(session)
    return {
        "summary_area": gr.update(visible=True),
        "upload_accordion": gr.update(),
        "summary_banner": load_warnings_html(session) + summary["banner"],
        "anomaly_strip_html": summary["anomaly_html"],
    }
