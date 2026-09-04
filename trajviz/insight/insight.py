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
                if (!window.__tvChartUiBound) {
                    window.__tvChartUiBound = true;
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
                    window.tvClickMainTab = function (label) {
                        const tabs = document.querySelectorAll('button[role=tab]');
                        for (let ti = 0; ti < tabs.length; ti++) {
                            if (tabs[ti].textContent.trim() === label) {
                                window.__tvHistoryNav = true;
                                try {
                                    tabs[ti].click();
                                } finally {
                                    window.__tvHistoryNav = false;
                                }
                                return tabs[ti];
                            }
                        }
                        return null;
                    };
                    window.tvActiveMainTab = function () {
                        const tabs = document.querySelectorAll('button[role=tab]');
                        for (let ti = 0; ti < tabs.length; ti++) {
                            const t = tabs[ti];
                            const selected = t.getAttribute('aria-selected');
                            if (selected === 'true' || selected === true) {
                                return t.textContent.trim();
                            }
                            if (t.classList.contains('selected') || t.getAttribute('data-selected') === 'true') {
                                return t.textContent.trim();
                            }
                        }
                        return '';
                    };
                    /* Remember the tab we left so Back can restore Overview (etc.). */
                    window.tvPushTabReturnPoint = function (tabLabel) {
                        if (!tabLabel || tabLabel === 'Workflow') return;
                        const cur = history.state || {};
                        if (cur.tvTab === tabLabel && !window.location.hash.startsWith('#step-')) {
                            return;
                        }
                        history.pushState(
                            { tvTab: tabLabel },
                            '',
                            window.location.pathname + window.location.search + window.location.hash
                        );
                    };
                    window.tvFocusWorkflowCard = function (card, opts) {
                        if (!card) return;
                        opts = opts || {};
                        const flash = opts.flash !== false;
                        const pushHistory = opts.pushHistory !== false;
                        document.querySelectorAll('.wf-card.wf-flash').forEach((el) => {
                            el.classList.remove('wf-flash');
                        });
                        card.scrollIntoView({behavior: 'smooth', block: 'center'});
                        /* selectCard reads __tvSelectOpts when invoked via click. */
                        window.__tvSelectOpts = { pushHistory: pushHistory };
                        card.click();
                        window.__tvSelectOpts = null;
                        if (!flash) return;
                        card.classList.add('wf-flash');
                        if (card.__tvFlashTimer) clearTimeout(card.__tvFlashTimer);
                        card.__tvFlashTimer = setTimeout(() => {
                            card.classList.remove('wf-flash');
                            card.__tvFlashTimer = null;
                        }, 2000);
                    };
                    window.tvGotoWorkflowStep = function (idx, opts) {
                        opts = opts || {};
                        const fromHistory = !!opts.fromHistory;
                        if (!fromHistory) {
                            const active = window.tvActiveMainTab();
                            if (active && active !== 'Workflow') {
                                window.tvPushTabReturnPoint(active);
                            }
                        }
                        window.tvClickMainTab('Workflow');
                        let attempts = 0;
                        const tryFocus = function () {
                            const c = document.getElementById('wf-card-' + idx);
                            if (c) {
                                window.tvFocusWorkflowCard(c, {
                                    flash: !fromHistory,
                                    pushHistory: !fromHistory,
                                });
                                return;
                            }
                            if (attempts++ < 12) {
                                setTimeout(tryFocus, 80);
                            }
                        };
                        setTimeout(tryFocus, fromHistory ? 80 : 200);
                    };
                    if (!window.__tvPopstateBound) {
                        window.__tvPopstateBound = true;
                        window.addEventListener('popstate', function () {
                            const state = history.state || {};
                            if (state.tvTab && state.tvTab !== 'Workflow') {
                                window.tvClickMainTab(state.tvTab);
                                return;
                            }
                            const m = String(window.location.hash || '').match(/^#step-(\\d+)$/);
                            if (m) {
                                window.tvGotoWorkflowStep(Number(m[1]), { fromHistory: true });
                            }
                        });
                    };
                    if (!window.__tvTabHistoryBound) {
                        window.__tvTabHistoryBound = true;
                        /* Capture phase: read the prior tab before Gradio switches. */
                        document.addEventListener('click', function (e) {
                            const tab = e.target && e.target.closest && e.target.closest('button[role=tab]');
                            if (!tab) return;
                            const next = tab.textContent.trim();
                            if (next !== 'Workflow') return;
                            if (window.__tvHistoryNav) return;
                            const prev = window.tvActiveMainTab();
                            if (prev && prev !== 'Workflow') {
                                window.tvPushTabReturnPoint(prev);
                            }
                        }, true);
                    };
                    window.tvBindChartWorkflowJumps = function () {
                        ['duration-chart', 'tool-outcome-chart', 'tool-duration-chart'].forEach((id) => {
                            const root = document.getElementById(id);
                            if (!root) return;
                            const plots = root.querySelectorAll('.js-plotly-plot');
                            plots.forEach((gd) => {
                                if (gd.__tvJumpBound || typeof gd.on !== 'function') return;
                                gd.__tvJumpBound = true;
                                gd.style.cursor = 'pointer';
                                gd.on('plotly_click', function (data) {
                                    const pt = data && data.points && data.points[0];
                                    if (!pt) return;
                                    let idx = pt.customdata;
                                    if (Array.isArray(idx)) idx = idx[0];
                                    const n = Number(idx);
                                    if (!Number.isFinite(n)) return;
                                    window.tvGotoWorkflowStep(Math.trunc(n));
                                });
                            });
                        });
                    };
                    window.tvBindDurationJump = window.tvBindChartWorkflowJumps;
                    let timer = null;
                    const schedule = () => {
                        if (timer) clearTimeout(timer);
                        timer = setTimeout(() => {
                            window.tvExpandFileTimeline();
                            window.tvBindChartWorkflowJumps();
                        }, 60);
                    };
                    new MutationObserver(schedule).observe(document.documentElement, {
                        childList: true,
                        subtree: true,
                        attributes: true,
                        attributeFilter: ['style', 'class', 'hidden'],
                    });
                    window.addEventListener('resize', schedule);
                    [0, 80, 250, 600].forEach((ms) => setTimeout(schedule, ms));
                }
                return [false];
            }""",
        )

    return app
