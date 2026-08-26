"""Raw Data tab."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr


@dataclass
class RawRefs:
    raw_json: gr.Code


def layout() -> RawRefs:
    with gr.TabItem("Raw Data"):
        raw_json = gr.Code(
            label="Full trajectory JSON",
            language="json",
            value="",
            max_lines=50,
        )
    return RawRefs(raw_json=raw_json)
