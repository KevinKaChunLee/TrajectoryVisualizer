"""Gradio UI for TrajViz — Blocks composer."""

from __future__ import annotations

import gradio as gr

from .llm_config import load_env_files
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
from .ui.load import bind_load, merge_load_slots
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

        sidebar_refs = sidebar.layout()
        upload_refs = upload.layout()
        with gr.Tabs(visible=False) as main_tabs:
            overview = overview_tab.layout()
            patterns = patterns_tab.layout()
            attribution = attribution_tab.layout()
            comparison = comparison_tab.layout()
            workflow = workflow_tab.layout()
            raw = raw_tab.layout()

        slots = merge_load_slots(
            main_tabs=main_tabs,
            shared=shared,
            refs={
                upload: upload_refs,
                overview_tab: overview,
                patterns_tab: patterns,
                workflow_tab: workflow,
                raw_tab: raw,
            },
        )
        _load_ev, _upload_ev = bind_load(
            file_upload=upload_refs.file_upload,
            load_btn=upload_refs.load_btn,
            format_selector=upload_refs.format_selector,
            state_dark=shared.state_dark,
            slots=slots,
        )
        load_events = (_load_ev, _upload_ev)
        upload.bind_export(upload_refs, shared, load_events)

        sidebar.bind(sidebar_refs, shared, load_events)
        overview_tab.bind(overview, shared, upload_refs)
        attribution_tab.bind(attribution, shared, upload_refs, load_events)
        comparison_tab.bind(comparison, shared, upload_refs)
        workflow_tab.bind(workflow, shared)

        app.load(
            fn=lambda _dark=False: False,
            outputs=[shared.state_dark],
            js="""() => {
                document.documentElement.style.colorScheme = 'light';
                document.documentElement.classList.remove('dark');
                if (document.body) document.body.classList.remove('dark');
                const btn = document.querySelector('#analysis-sidebar .toggle-button');
                if (btn) {
                    btn.setAttribute('aria-label', 'AI Trajectory Analysis');
                    btn.setAttribute('title', 'AI Trajectory Analysis');
                }
                if (!window.__tvFileTimelineBound) {
                    window.__tvFileTimelineBound = true;
                    const ROW = 28, CHROME = 120, MIN = 340;
                    window.tvExpandFileTimeline = function () {
                        document.querySelectorAll('.resizable-chart').forEach((root) => {
                            const gd = root.querySelector('.js-plotly-plot')
                                || root.querySelector('[data-testid="plotly"]');
                            if (!gd || !gd.layout) return;
                            const meta = gd.layout.meta || {};
                            const cats = (gd.layout.yaxis && gd.layout.yaxis.categoryarray)
                                || (gd._fullLayout && gd._fullLayout.yaxis
                                    && gd._fullLayout.yaxis._categories)
                                || [];
                            const n = Array.isArray(cats) ? cats.length : 0;
                            const h = Number(meta.tv_chart_height)
                                || (n ? Math.max(MIN, ROW * n + CHROME) : 0);
                            if (!h) return;
                            const plot = gd.classList.contains('js-plotly-plot')
                                ? gd
                                : (gd.querySelector('.js-plotly-plot') || gd);
                            if (plot.style.height === h + 'px'
                                && Math.abs(plot.clientHeight - h) < 4) return;
                            plot.style.height = h + 'px';
                            plot.style.minHeight = h + 'px';
                            root.style.height = 'auto';
                            root.style.maxHeight = 'none';
                            if (window.Plotly && window.Plotly.Plots) {
                                window.Plotly.Plots.resize(plot);
                            }
                        });
                    };
                    let timer = null;
                    const schedule = () => {
                        if (timer) clearTimeout(timer);
                        timer = setTimeout(window.tvExpandFileTimeline, 60);
                    };
                    new MutationObserver(schedule).observe(document.documentElement, {
                        childList: true,
                        subtree: true,
                        attributes: true,
                        attributeFilter: ['style', 'class', 'hidden'],
                    });
                    window.addEventListener('resize', schedule);
                    [0, 80, 250, 600].forEach((ms) => setTimeout(window.tvExpandFileTimeline, ms));
                }
                return [false];
            }""",
        )

    return app
