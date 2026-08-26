"""Gradio UI for TrajViz — Blocks composer."""

from __future__ import annotations

import gradio as gr

from .llm_config import load_env_files
from .presenters import trajectory_format_label  # noqa: F401  (public re-export)
from .styles import APP_CSS  # noqa: F401  (re-exported: __main__ passes it to app.launch)
from .ui import (
    attribution_tab,
    comparison_tab,
    overview_tab,
    patterns_tab,
    raw_tab,
    sidebar,
    upload,
    workflow_tab,
)
from .ui.load import bind_load
from .ui.shared import SharedState


def build_ui() -> gr.Blocks:
    """Build the full Gradio Blocks UI."""
    load_env_files()

    with gr.Blocks(title="TrajViz", elem_classes=["trajectory-viz"]) as app:
        shared = SharedState(
            state_steps=gr.State([]),
            state_dark=gr.State(False),
            state_raw=gr.State({}),
            state_analysis_brief=gr.State(""),
        )
        state_steps = shared.state_steps
        state_dark = shared.state_dark
        state_raw = shared.state_raw
        state_analysis_brief = shared.state_analysis_brief  # noqa: F841  (analysis tests)

        sidebar_refs = sidebar.layout()
        upload_refs = upload.layout()
        with gr.Tabs(visible=False) as main_tabs:
            overview = overview_tab.layout()
            patterns = patterns_tab.layout()
            attribution = attribution_tab.layout()
            comparison = comparison_tab.layout()
            workflow = workflow_tab.layout()
            raw = raw_tab.layout()

        summary_area = upload_refs.summary_area
        upload_accordion = upload_refs.upload_accordion
        file_upload = upload_refs.file_upload
        load_btn = upload_refs.load_btn
        format_selector = upload_refs.format_selector
        summary_banner = upload_refs.summary_banner
        anomaly_strip_html = upload_refs.anomaly_strip_html

        all_outputs = [
            main_tabs,
            summary_area,
            upload_accordion,
            state_steps,
            summary_banner,
            anomaly_strip_html,
            overview.overview_kpi_html,
            overview.session_detail_html,
            overview.metrics_md,
            overview.token_chart,
            overview.duration_chart,
            overview.context_growth_chart,
            overview.behavior_md,
            overview.tool_chart,
            overview.tool_outcome_chart,
            overview.agent_summary_html,
            overview.agent_token_chart,
            overview.agent_swimlane_chart,
            overview.diag_summary_html,
            overview.diag_pressure_html,
            overview.diag_pressure_agent,
            overview.diag_pressure_chart,
            overview.diag_file_chart,
            overview.diag_rootcause_html,
            overview.error_class_chart,
            overview.plan_timeline_chart,
            overview.hotspots_md,
            overview.per_message_md,
            workflow.wf_filter_chips_html,
            workflow.wf_filter_hidden,
            workflow.wf_count_html,
            workflow.toc_html,
            workflow.workflow_html,
            workflow.detail_store,
            workflow.detail_html,
            raw.raw_json,
            patterns.patterns_tool_html,
            patterns.patterns_failure_html,
            patterns.antipattern_summary_html,
            state_raw,
        ]

        _load_ev, _upload_ev = bind_load(
            file_upload=file_upload,
            load_btn=load_btn,
            format_selector=format_selector,
            state_dark=state_dark,
            all_outputs=all_outputs,
        )
        load_events = (_load_ev, _upload_ev)

        sidebar.bind(sidebar_refs, shared, load_events)
        overview_tab.bind(overview, shared, upload_refs)
        attribution_tab.bind(attribution, shared, upload_refs, load_events)
        comparison_tab.bind(comparison, shared, upload_refs)
        workflow_tab.bind(workflow, shared)

        app.load(
            fn=lambda dark: dark,
            inputs=[state_dark],
            outputs=[state_dark],
            js="""() => {
                const btn = document.querySelector('#analysis-sidebar .toggle-button');
                if (btn) {
                    btn.setAttribute('aria-label', 'AI Trajectory Analysis');
                    btn.setAttribute('title', 'AI Trajectory Analysis');
                }
                return [window.matchMedia('(prefers-color-scheme: dark)').matches];
            }""",
        )

    return app
