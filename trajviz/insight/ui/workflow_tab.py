"""Workflow tab: filter chips, TOC, step cards, and client JS."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from ..presenters import (
    DETAIL_PLACEHOLDER,
    FILTER_CHIPS_DEFAULT,
    _build_filtered_workflow_outputs,
)
from ..rendering import render_filter_chips
from .shared import SharedState

WORKFLOW_JS = """
                            /* Filter chip click handler (delegated, survives re-renders) */
                            window.__syncWorkflowFilters = function(bar) {
                                if (!bar) return;
                                var active = Array.from(bar.querySelectorAll('.filter-chip.chip-active'))
                                    .map(function(c) { return c.dataset.filter; });
                                var hiddenEl = document.querySelector(
                                    '#wf-filter-hidden textarea, #wf-filter-hidden input'
                                );
                                if (!hiddenEl) return;
                                var proto = hiddenEl.tagName === 'TEXTAREA'
                                    ? window.HTMLTextAreaElement.prototype
                                    : window.HTMLInputElement.prototype;
                                var descriptor = Object.getOwnPropertyDescriptor(proto, 'value');
                                if (descriptor && descriptor.set) {
                                    descriptor.set.call(hiddenEl, active.join(','));
                                } else {
                                    hiddenEl.value = active.join(',');
                                }
                                hiddenEl.dispatchEvent(new InputEvent('input', {
                                    bubbles: true,
                                    composed: true,
                                    inputType: 'insertText',
                                    data: null
                                }));
                                hiddenEl.dispatchEvent(new Event('change', {
                                    bubbles: true,
                                    composed: true
                                }));
                            };
                            window.__setWorkflowChipActive = function(chip, active) {
                                if (!chip) return;
                                chip.classList.toggle('chip-active', active);
                                chip.setAttribute('aria-pressed', active ? 'true' : 'false');
                            };
                            window.__updateWorkflowFilterQuery = function(root) {
                                if (!root) return;
                                var roles = Array.from(root.querySelectorAll(
                                    '[data-filter-group="role"].chip-active'
                                )).map(function(c) { return c.dataset.filter; });
                                var features = Array.from(root.querySelectorAll(
                                    '[data-filter-group="feature"].chip-active'
                                )).map(function(c) { return c.dataset.filter; });
                                var query = root.querySelector('#wf-filter-query');
                                if (!query) return;
                                var featureText = features.indexOf('All') >= 0
                                    ? 'All'
                                    : features.join(' or ');
                                query.textContent = 'Role: ' + roles.join(' or ')
                                    + ' · Step feature: ' + featureText;
                            };
                            /* Pure chip state machine, unit-tested in
                               tests/test_workflow_filtering.py by executing this
                               exact source in Node. Keep it DOM-free. */
                            /* __WF_CHIP_STATE_BEGIN__ */
                            window.__wfComputeChipState = function(state, action) {
                                var roles = {};
                                var features = {};
                                Object.keys(state.roles).forEach(function(k) { roles[k] = !!state.roles[k]; });
                                Object.keys(state.features).forEach(function(k) { features[k] = !!state.features[k]; });
                                var rejected = false;
                                if (action.type === 'reset') {
                                    Object.keys(roles).forEach(function(k) { roles[k] = true; });
                                    Object.keys(features).forEach(function(k) { features[k] = (k === 'All'); });
                                } else if (action.group === 'role') {
                                    var activeRoles = Object.keys(roles).filter(function(k) { return roles[k]; });
                                    if (roles[action.name] && activeRoles.length === 1) {
                                        /* Refuse to deselect the last active role. */
                                        rejected = true;
                                    } else {
                                        roles[action.name] = !roles[action.name];
                                    }
                                } else if (action.name === 'All') {
                                    /* 'All' is exclusive with specific features. */
                                    Object.keys(features).forEach(function(k) { features[k] = (k === 'All'); });
                                } else {
                                    features[action.name] = !features[action.name];
                                    features['All'] = false;
                                    var anySpecific = Object.keys(features).some(function(k) {
                                        return k !== 'All' && features[k];
                                    });
                                    if (!anySpecific) {
                                        /* Auto-restore 'All' when nothing specific is left. */
                                        features['All'] = true;
                                    }
                                }
                                return {roles: roles, features: features, rejected: rejected};
                            };
                            /* __WF_CHIP_STATE_END__ */
                            window.__wfReadChipState = function(bar) {
                                var state = {roles: {}, features: {}};
                                bar.querySelectorAll('[data-filter-group="role"]').forEach(function(c) {
                                    state.roles[c.dataset.filter] = c.classList.contains('chip-active');
                                });
                                bar.querySelectorAll('[data-filter-group="feature"]').forEach(function(c) {
                                    state.features[c.dataset.filter] = c.classList.contains('chip-active');
                                });
                                return state;
                            };
                            window.__wfApplyChipState = function(bar, state) {
                                bar.querySelectorAll('[data-filter-group="role"]').forEach(function(c) {
                                    window.__setWorkflowChipActive(c, !!state.roles[c.dataset.filter]);
                                });
                                bar.querySelectorAll('[data-filter-group="feature"]').forEach(function(c) {
                                    window.__setWorkflowChipActive(c, !!state.features[c.dataset.filter]);
                                });
                            };
                            if (!window.__wfChipHandlerAttached) {
                                window.__wfChipHandlerAttached = true;
                                document.addEventListener('click', function(e) {
                                    var chip = e.target.closest('.filter-chip');
                                    var reset = e.target.closest('[data-wf-action="reset-filters"]');
                                    if (!chip && !reset) return;
                                    var root = (chip || reset).closest('#wf-filter-chips');
                                    if (!root) return;
                                    var bar = root.querySelector('#wf-filter-bar');
                                    if (!bar) return;
                                    var action = reset
                                        ? {type: 'reset'}
                                        : {type: 'toggle', group: chip.dataset.filterGroup, name: chip.dataset.filter};
                                    var next = window.__wfComputeChipState(
                                        window.__wfReadChipState(bar), action
                                    );
                                    if (next.rejected) {
                                        var roleGroup = chip.closest('.filter-group');
                                        if (roleGroup) {
                                            roleGroup.classList.remove('filter-group-attention');
                                            void roleGroup.offsetWidth;
                                            roleGroup.classList.add('filter-group-attention');
                                            window.setTimeout(function() {
                                                roleGroup.classList.remove('filter-group-attention');
                                            }, 900);
                                        }
                                        return;
                                    }
                                    window.__wfApplyChipState(bar, next);
                                    window.__updateWorkflowFilterQuery(root);
                                    window.__syncWorkflowFilters(bar);
                                });
                            }

                            /* Detail tab click handler (delegated, survives detail HTML replacement) */
                            if (!window.__dpTabHandlerAttached) {
                                window.__dpTabHandlerAttached = true;
                                document.addEventListener('click', function(e) {
                                    var tab = e.target.closest('.dp-tab');
                                    if (!tab) return;
                                    var panel = tab.closest('.dp-panel');
                                    if (!panel) return;
                                    panel.querySelectorAll('.dp-tab').forEach(function(x) {
                                        x.classList.remove('dp-tab-active');
                                    });
                                    panel.querySelectorAll('.dp-tab-content').forEach(function(x) {
                                        x.classList.remove('dp-tab-visible');
                                    });
                                    tab.classList.add('dp-tab-active');
                                    var content = panel.querySelector(
                                        '[data-tab-content="' + tab.dataset.tab + '"]'
                                    );
                                    if (content) content.classList.add('dp-tab-visible');
                                });
                            }

                            /* Card click handler */
                            function selectCard(card) {
                                if (!card) return;
                                element.querySelectorAll('.wf-card').forEach(function(c) {
                                    c.classList.remove('wf-active');
                                });
                                card.classList.add('wf-active');
                                var idx = card.dataset.stepIdx;
                                /* URL deep linking */
                                if (idx != null) {
                                    window.__wfSelectedStep = idx;
                                    history.replaceState(null, '', '#step-' + idx);
                                }
                                var storeEl = document.querySelector('#wf-detail-store [data-b64]');
                                var target = document.getElementById('wf-detail-content');
                                if (!storeEl || !target) return;
                                try {
                                    var details = JSON.parse(atob(storeEl.dataset.b64));
                                    if (details[idx] != null) {
                                        target.innerHTML = details[idx];
                                        var detailPanel = target.closest('.detail-panel');
                                        if (detailPanel) detailPanel.scrollTop = 0;
                                    }
                                } catch(ex) { console.error('wf-click:', ex); }
                            }
                            element.addEventListener('click', function(e) {
                                selectCard(e.target.closest('.wf-card'));
                            });

                            /* Keyboard navigation: j/k for next/prev step */
                            document.addEventListener('keydown', function(e) {
                                var tag = (e.target.tagName || '').toLowerCase();
                                if (tag === 'input' || tag === 'textarea' || e.target.isContentEditable) return;
                                if (e.key !== 'j' && e.key !== 'k') return;
                                var cards = Array.from(element.querySelectorAll('.wf-card'));
                                if (!cards.length) return;
                                var activeIdx = cards.findIndex(function(c) { return c.classList.contains('wf-active'); });
                                var nextIdx;
                                if (e.key === 'j') {
                                    nextIdx = activeIdx < 0 ? 0 : Math.min(activeIdx + 1, cards.length - 1);
                                } else {
                                    nextIdx = activeIdx < 0 ? 0 : Math.max(activeIdx - 1, 0);
                                }
                                cards[nextIdx].scrollIntoView({behavior:'smooth', block:'center'});
                                selectCard(cards[nextIdx]);
                            });

                            /* Deep link: on load, scroll to step from URL hash */
                            setTimeout(function() {
                                var hash = window.location.hash;
                                var m = hash && hash.match(/^#step-(\\d+)$/);
                                if (m) {
                                    var card = document.getElementById('wf-card-' + m[1]);
                                    if (card) {
                                        card.scrollIntoView({behavior:'smooth', block:'center'});
                                        selectCard(card);
                                    } else {
                                        /* Stale hash from an earlier session or another
                                           trajectory: drop it rather than guess. */
                                        history.replaceState(
                                            null, '',
                                            window.location.pathname + window.location.search
                                        );
                                    }
                                }
                            }, 500);

                            /* Hidden-selection watcher: when a re-render (filter,
                               search, or label upload) removes the selected step's
                               card, tell the user in the detail panel; when the
                               card comes back, restore its detail. */
                            if (!window.__wfHiddenStepObserverAttached) {
                                window.__wfHiddenStepObserverAttached = true;
                                var hiddenStepCheckPending = null;
                                var checkSelectedStepVisible = function() {
                                    hiddenStepCheckPending = null;
                                    var target = document.getElementById('wf-detail-content');
                                    if (!target) return;
                                    if (target.querySelector('[data-wf-detail-placeholder]')) {
                                        /* The app reset the detail panel (new trajectory
                                           or label upload): the old selection is gone. */
                                        window.__wfSelectedStep = null;
                                        return;
                                    }
                                    var idx = window.__wfSelectedStep;
                                    if (idx == null) return;
                                    var card = document.getElementById('wf-card-' + idx);
                                    if (card) {
                                        if (target.querySelector('[data-wf-hidden-msg]')) {
                                            selectCard(card);
                                        }
                                        return;
                                    }
                                    if (!document.querySelector('.wf-card')) return;
                                    if (target.querySelector('[data-wf-hidden-msg]')) return;
                                    target.innerHTML =
                                        "<div data-wf-hidden-msg='1'" +
                                        " style='padding:2em 1em;text-align:center;color:var(--ov-muted);'>" +
                                        "<p style='font-size:15px;margin-bottom:0.5em;'>" +
                                        "Selected step is hidden by the current filters</p>" +
                                        "<p style='font-size:12px;'>Adjust the filters to show it again.</p>" +
                                        "</div>";
                                };
                                new MutationObserver(function() {
                                    if (hiddenStepCheckPending) return;
                                    hiddenStepCheckPending = window.setTimeout(checkSelectedStepVisible, 120);
                                }).observe(document.body, {childList: true, subtree: true});
                            }

                            /* Auto-select first assistant card if no hash link */
                            setTimeout(function() {
                                if (window.location.hash) return;
                                var cards = element.querySelectorAll('.wf-card');
                                for (var i = 0; i < cards.length; i++) {
                                    var badges = cards[i].querySelectorAll('.wf-badge');
                                    for (var j = 0; j < badges.length; j++) {
                                        if (badges[j].textContent.trim() === 'Assistant') {
                                            selectCard(cards[i]);
                                            return;
                                        }
                                    }
                                }
                                if (cards.length) selectCard(cards[0]);
                            }, 600);
                            """


@dataclass
class WorkflowRefs:
    wf_toc_toggle: gr.Button
    wf_filter_chips_html: gr.HTML
    wf_search: gr.Textbox
    wf_filter_hidden: gr.Textbox
    wf_count_html: gr.HTML
    toc_html: gr.HTML
    workflow_html: gr.HTML
    detail_html: gr.HTML
    detail_store: gr.HTML


def layout() -> WorkflowRefs:
    with gr.TabItem("Workflow"):
        with gr.Row(equal_height=True):
            wf_toc_toggle = gr.Button("TOC", variant="secondary", scale=0, min_width=50)
            wf_filter_chips_html = gr.HTML(
                render_filter_chips(),
                elem_id="wf-filter-chips",
            )
            wf_search = gr.Textbox(
                label="Search",
                placeholder="Filter by keyword...",
                scale=1,
            )
        # Hidden textbox that JS writes active filters into (comma-separated)
        wf_filter_hidden = gr.Textbox(
            value=",".join(FILTER_CHIPS_DEFAULT),
            # Keep the component mounted so the delegated chip handler
            # can update it and trigger Gradio's input event.  Gradio
            # omits visible=False components from the browser DOM.
            visible=True,
            elem_id="wf-filter-hidden",
        )
        wf_count_html = gr.HTML("")
        with gr.Row(equal_height=False):
            with gr.Column(scale=0, min_width=150):
                toc_html = gr.HTML("", elem_id="wf-toc-container")
            with gr.Column(scale=3, min_width=400):
                workflow_html = gr.HTML(
                    "<div style='padding:3em;color:var(--ov-muted);text-align:center;"
                    "font-size:15px;'>Load a trajectory to see the step flow.</div>",
                    js_on_load=WORKFLOW_JS,
                )
            with gr.Column(scale=2, min_width=300, elem_classes=["detail-panel"]):
                detail_html = gr.HTML(
                    DETAIL_PLACEHOLDER,
                    elem_id="wf-detail-panel",
                )
        detail_store = gr.HTML("", elem_id="wf-detail-store")
    return WorkflowRefs(
        wf_toc_toggle=wf_toc_toggle,
        wf_filter_chips_html=wf_filter_chips_html,
        wf_search=wf_search,
        wf_filter_hidden=wf_filter_hidden,
        wf_count_html=wf_count_html,
        toc_html=toc_html,
        workflow_html=workflow_html,
        detail_html=detail_html,
        detail_store=detail_store,
    )


def bind(refs: WorkflowRefs, shared: SharedState) -> None:
    state_steps = shared.state_steps
    wf_filter_hidden = refs.wf_filter_hidden
    wf_search = refs.wf_search
    toc_html = refs.toc_html
    workflow_html = refs.workflow_html
    wf_count_html = refs.wf_count_html

    def do_filter_workflow(steps, filter_csv, keyword, current_toc):
        """Re-render Workflow cards, count, and TOC with filters applied."""
        return _build_filtered_workflow_outputs(steps, filter_csv, keyword, current_toc)

    def on_toc_toggle(current_toc):
        """Toggle TOC sidebar visibility via CSS class."""
        if not current_toc:
            return current_toc
        if "toc-hidden" in current_toc:
            return current_toc.replace("toc-hidden", "").strip()
        return current_toc.replace("wf-toc-sidebar", "wf-toc-sidebar toc-hidden")

    refs.wf_toc_toggle.click(
        fn=on_toc_toggle,
        inputs=[toc_html],
        outputs=[toc_html],
    )

    wf_filter_hidden.change(
        fn=do_filter_workflow,
        inputs=[state_steps, wf_filter_hidden, wf_search, toc_html],
        outputs=[workflow_html, wf_count_html, toc_html],
    )
    wf_search.change(
        fn=do_filter_workflow,
        inputs=[state_steps, wf_filter_hidden, wf_search, toc_html],
        outputs=[workflow_html, wf_count_html, toc_html],
    )
