"""Analysis sidebar layout and chat callbacks."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from ..assistant import analyze_loaded_trajectory, answer_question
from ..llm_config import config_status_html
from .shared import SharedState


@dataclass
class SidebarRefs:
    analysis_status: gr.HTML
    analysis_chatbot: gr.Chatbot
    analysis_input: gr.Textbox
    analysis_clear: gr.Button


def layout() -> SidebarRefs:
    with gr.Sidebar(
        label="AI Trajectory Analysis",
        position="right",
        width="max(480px, 25vw)",
        open=False,
        elem_id="analysis-sidebar",
        elem_classes=["analysis-sidebar"],
    ):
        gr.HTML(
            "<div class='analysis-panel-title'>🤖 AI Trajectory Analysis</div>"
        )
        analysis_status = gr.HTML(config_status_html(loaded_steps=0))
        analysis_chatbot = gr.Chatbot(
            value=[],
            label="Analysis",
            show_label=False,
            height=380,
            resizable=True,
            layout="panel",
            placeholder=(
                "Load a trajectory to start analysis, then ask follow-up questions."
            ),
            buttons=["copy"],
            feedback_options=None,
            elem_classes=["analysis-chatbot"],
        )
        with gr.Row():
            analysis_input = gr.Textbox(
                label="Question",
                placeholder="e.g. What is the biggest performance bottleneck?",
                scale=4,
                container=False,
                submit_btn=True,
                elem_id="analysis-input",
            )
        analysis_clear = gr.Button("Clear chat", size="sm", variant="secondary")
    return SidebarRefs(
        analysis_status=analysis_status,
        analysis_chatbot=analysis_chatbot,
        analysis_input=analysis_input,
        analysis_clear=analysis_clear,
    )


def bind(refs: SidebarRefs, shared: SharedState, load_events) -> None:
    state_analysis_brief = shared.state_analysis_brief

    def on_trajectory_for_analysis(steps, raw):
        """Pack dashboard stats, then run the first analysis pass."""
        if not steps:
            return "", [], config_status_html(loaded_steps=0)
        brief, history = analyze_loaded_trajectory(
            steps, raw if isinstance(raw, dict) else {},
        )
        return brief, history, config_status_html(loaded_steps=len(steps))

    for _ev in load_events:
        _ev.then(
            fn=on_trajectory_for_analysis,
            inputs=[shared.state_steps, shared.state_raw],
            outputs=[state_analysis_brief, refs.analysis_chatbot, refs.analysis_status],
        )

    def on_analysis_ask(question, history, brief):
        return answer_question(question, history, brief), ""

    refs.analysis_input.submit(
        fn=on_analysis_ask,
        inputs=[refs.analysis_input, refs.analysis_chatbot, state_analysis_brief],
        outputs=[refs.analysis_chatbot, refs.analysis_input],
    )
    refs.analysis_clear.click(fn=lambda: [], outputs=[refs.analysis_chatbot])
