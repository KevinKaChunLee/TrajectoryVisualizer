# Integrating DECAF failure attribution into TrajViz — implementation plan

**Branch:** `feat/decaf-trajviz-integration`
**Goal:** surface DECAF's per-trajectory *capability failure attribution* (which of the 7 workflow capabilities broke, with tiered evidence) inside the TrajectoryVisualizer (TrajViz) dashboard as a new **Attribution** view — the existing DECAF method, a new UI.

## 1. Non-goals (explicit scope guards)

- **No unifying the two trajectory normalizers.** DECAF (`awe`) and TrajViz each normalize raw agent logs differently (see §3); unifying them is invasive and out of scope. We integrate at the *diagnosis boundary* only.
- **No changes to DECAF's attribution logic or numbers.** DECAF is consumed as a library; the v0.33 dossier stays canonical. Golden tests enforce this.
- **No fabricated attribution when gold is absent.** Capability blame is gold-grounded; without a gold reference the UI degrades honestly (§5.3), it does not guess.
- **No new LLM spend by default.** The default path is the offline deductive/associational slice (5 capabilities, no API key).

## 2. What the codebase review established (load-bearing facts)

| Fact | Source | Consequence for the plan |
|---|---|---|
| DECAF core is loosely bound: `detect(instance_id, agent, patch_override, resolved_override, trajectory_override, use_judge)` + `build_blame(d)` + `evidence.build_chain(...)` accept in-memory objects | `awe/detect.py:346`, `awe/rootcause.py`, `awe/evidence.py:251` | We call the core directly with injected objects — never `dossier.case_record` (fully disk-bound). |
| **Gold is the one hard binding** — `detect` always calls `build_gold_reference(instance_id)` (no override), which reads `requirements/<id>.json` | `awe/detect.py:356`, `awe/gold.py:73` | The central design problem. Handled by the three gold modes in §5.3. |
| 5 capabilities (localization, editing, verification, self-repair, tool-use) are pure deductive/associational — **no key**; 2 (requirement, planning) need the judge, but verdicts are **cached and read back with no key** | `awe/detect.py:105,423`, `awe/judge.py:424` | Default offline 5-cap; auto-upgrade to 7 when a cache exists; lazy compute is opt-in. |
| DECAF adapters take a raw dict + `(agent, instance_id)` and TrajViz already holds that raw dict and detects its format | `awe/adapters.py:82/166/358`, `trajviz/insight/loaders.py:20,1452` | **Reuse DECAF's adapters** on TrajViz's raw dict — no lossy converter. |
| TrajViz is Gradio 6 + Plotly, Python **3.11+**, builder-based dashboard; a new tab = 1 builder + 3 positional wiring touch-points | `trajviz/insight/insight.py:804,1472,1620,1428` | UI addition is well-bounded. |
| **rendering.py already ships unused renderers** for capability scorecards (`build_dimension_cards_html:971`) and a judge/dossier panel (`build_judge_result_html:1021`), with matching CSS in `styles.py` | `trajviz/insight/rendering.py` | The UI surface for scorecards + dossier largely exists; we wire and adapt it. |
| TrajViz already does proto-attribution (error clustering, failure chains, anti-patterns) but produces **no capability verdict/scorecard/dossier** | `trajviz/insight/diagnostics.py`, `patterns.py` | DECAF fills exactly the gap; we cross-link rather than duplicate. |
| Both projects key off `TraceProbe/data` (`requirements/`, `patch/<agent>/`, `trajectory/<agent>/`, `eval_<agent>.json`), but TrajViz has **no `instance_id` concept** — it loads a bare file path | `awe/config.py:15-24`, `trajviz/insight/loaders.py:1452` | TrajViz must start carrying `(agent, instance_id)`; trivially derivable from the corpus path `trajectory/<agent>/<id>.json`. |
| Python: system `python3` is 3.9 (TrajViz needs 3.11+); the repo `.venv` is **3.13**; DECAF needs ≥3.9; `pytest` is not a declared TrajViz dep | prior session + review | Standardize on the repo `.venv` (3.13); add `pytest` to TrajViz dev deps. |
| The current OpenRouter key has `anthropic/*` and `openai/*` **403-blocked** (only deepseek/glm work) | this session | The judge default model (`claude-sonnet-4.5`) is unusable on this key — lazy LLM compute must use `z-ai/glm-5.2`. Default stays offline. |

## 3. Architecture

```
 TrajViz (Gradio)                         New seam                         DECAF (awe, library)
 ┌─────────────────┐   raw dict + fmt   ┌──────────────────────┐   objects   ┌────────────────────┐
 │ loaders.detect_ │ ─────────────────▶ │ trajviz/insight/     │ ──────────▶ │ adapters.adapt_<f> │
 │ format / raw    │  (agent,inst_id)   │ attribution.py       │             │ → NormalizedTraj   │
 │ state_raw       │                    │  diagnose(...)       │             │ detect(overrides)  │
 └─────────────────┘                    │  · reuse DECAF adptr │             │ build_blame(d)     │
        │ steps                         │  · resolve gold      │             │ evidence.build_    │
        ▼                               │  · patch+outcome     │ ◀────────── │   chain(...)       │
 ┌─────────────────┐   AttributionResult│  · call core         │  blame+     └────────────────────┘
 │ Attribution tab │ ◀───────────────── │  → AttributionResult │  chains
 │ (rendering.py   │                    └──────────────────────┘
 │  scorecards +   │
 │  dossier)       │
 └─────────────────┘
```

**One new module, one one-directional dependency.** `trajviz` imports `awe`; `awe` never imports `trajviz`. All DECAF access is funneled through a single new file `trajviz/insight/attribution.py`, so the coupling surface is auditable and the two projects stay independently testable.

## 4. The library seam — `trajviz/insight/attribution.py`

A thin, UI-agnostic wrapper (pure functions, no Gradio imports — so it is unit-testable headless):

```python
# trajviz/insight/attribution.py
from dataclasses import dataclass

@dataclass
class AttributionResult:
    available: bool                 # False => degraded (see reason)
    reason: str | None              # why unavailable (e.g. "no gold reference for <id>")
    mode: str                       # "corpus" | "gold_provided" | "gold_free"
    instance_id: str | None
    agent: str | None
    blame_status: str | None        # primary | conjunctive | refuted_unattributed
    primary: dict | None            # {capability, error_type, ...}
    scorecard: list[dict]           # per-capability: {capability, blamed, weight, tier, top_error}
    faults: list[dict]              # each: {label, capability, error_type, confidence,
                                    #        blame_weight, is_primary, evidence_chain, audit,
                                    #        code_evidence, recommendation}
    used_judge: bool                # 7-cap (cache present) vs 5-cap offline

def diagnose(raw: dict, fmt: str, agent: str | None, instance_id: str | None,
             argus_root: str | None = None, use_judge: str = "auto") -> AttributionResult: ...
```

`diagnose` steps:
1. **Adapt** the raw dict via the DECAF adapter for `fmt` (`adapt_claude_code` / `adapt_opencode` / `adapt_codex`) → `NormalizedTrajectory`. (CodeArts → degraded until §8 adds `adapt_codearts`.)
2. **Resolve gold** (§5.3): corpus → `build_gold_reference(instance_id)` via `AWE_ARGUS_ROOT`; gold-provided → construct+prime; gold-free → return `available=False, reason=...`.
3. **Patch + outcome:** `patch_override` from `patch/<agent>/<id>.diff` (or the trajectory's final patch); `resolved_override` from `eval_<agent>.json` (or user-supplied). Build `AgentPatch` via the `patches.py:40-57` logic.
4. **Diagnose:** `d = detect(instance_id, agent, patch_override=…, resolved_override=…, trajectory_override=nt, use_judge=<auto>)`; `b = build_blame(d, adjudicate=arbiter.auto_adjudicate(agent))`.
5. **Shape** into `AttributionResult`: derive the per-capability scorecard from `b["capability_blame"]`, build each fault's `evidence_chain` via `evidence.build_chain(...)`, attach `audit`/`code_evidence`.

Returns are plain dicts/dataclasses → the UI layer and tests both consume them without Gradio.

## 5. UI design — the Attribution tab

### 5.1 Placement & wiring (bounded, mechanical)
- New `with gr.TabItem("Attribution"):` after Patterns (`insight.py:957`).
- New builder `_build_attribution_outputs(steps, raw, fmt, agent, instance_id)` beside `_build_diagnostics_outputs` (`insight.py:582`) → returns a dict.
- Three positional touch-points (must stay in sync): the `_do_load_inner` return tuple (`insight.py:1564`), the `all_outputs` list (`insight.py:1620`), the `_empty_result` tuple (`insight.py:1428`).

### 5.2 Renderers (reuse the pre-built scaffolding)
- **Scorecard grid** → `build_dimension_cards_html` (`rendering.py:971`): map DECAF's 7 capabilities onto dimension cards (blamed/clean + weight + driving error type + evidence-tier badge). Extend `_DIMENSION_DISPLAY_NAMES`/`_VERDICT_*` to the DECAF capability set.
- **Fault dossier + evidence chain** → generalize `build_judge_result_html` (`rendering.py:1021`) into a per-fault panel: `claim`, a **strength badge** (deductive / associational / model-inferred), the ordered `links[]` (`•` observation / `↳` inference / `⇒` conclusion, matching `dossier.py:334`), verbatim `quotes` on judge-cited links, and the `audit` verdict badge. Clickable flagged-step links reuse the existing judge-panel step-jump JS.
- **Primary-cause banner** + the `code_evidence` side-by-side diff reuse `_render_diff_lines`/`_split_diff_by_file` (`rendering.py:567/587`).

### 5.3 Gold modes (the central UX decision)
- **Corpus mode (primary):** trajectory came from `TraceProbe/data/trajectory/<agent>/<id>.json`. TrajViz derives `(agent, instance_id)` from the path (or an optional pair of fields in the upload row) and points `AWE_ARGUS_ROOT` at TraceProbe → full gold-grounded attribution.
- **Gold-provided mode:** user supplies a `requirements/<id>.json` (or a gold patch) alongside the trajectory → construct a `GoldReference` and prime `build_gold_reference`'s `lru_cache`.
- **Gold-free mode (arbitrary upload):** no gold → Attribution tab shows a clear notice ("Capability attribution needs the reference patch + test outcome; showing trajectory-only signals") and links to TrajViz's existing diagnostics/patterns. **No fabricated verdict.**

### 5.4 Cross-link, don't duplicate
TrajViz's existing error clusters / failure chains (`diagnostics.py`) become supporting evidence *under* the DECAF primary cause, not a competing verdict. The Workflow tab optionally gets per-step fault badges (Phase 4) via `format_step_detail` (`rendering.py:767`).

## 6. Phased plan (each phase independently shippable + verifiable)

**Phase 0 — dependency wiring & baseline.** Make `trajviz` import `awe` under the repo `.venv` (3.13): add DECAF as a path/editable dep, add `pytest` to TrajViz dev deps, pin the interpreter in the test docs. DoD: `python -c "import awe, trajviz"` under `.venv`; existing TrajViz suite (107) + DECAF suite (448) both green under 3.13.

**Phase 1 — the library seam (no UI).** Implement `attribution.py::diagnose` (adapter reuse + gold resolution + detect/build_blame/build_chain + `AttributionResult`). DoD: unit tests pass — adapter parity (§7), golden diagnosis (§7), gold-free degradation, offline (no key). This is the risk-carrying phase; UI is trivial after it.

**Phase 2 — the Attribution tab.** Add the tab + builder + 3 wiring points; adapt `build_dimension_cards_html` + `build_judge_result_html`; implement the three gold modes' UX. DoD: end-to-end UI verification (§7) — launch the app, load a known corpus trajectory, Attribution tab renders the correct primary cause + scorecard + evidence chain matching `faults.json`; a gold-free upload shows the degradation notice without error.

**Phase 3 — LLM layers (opt-in).** Read cached judge/arbiter verdicts offline → 7-capability view when a cache exists (`use_judge="auto"`). Add an opt-in "compute judge" action guarded by `OPENROUTER_API_KEY` **and** a non-blocked model (default `z-ai/glm-5.2`, since anthropic/openai are 403-blocked on this key). DoD: 7-cap view appears iff cache present; lazy compute writes cache and is a no-op without a key.

**Phase 4 — polish & breadth.** Per-step fault badges in the Workflow tab; `adapt_codearts` in DECAF so CodeArts trajectories are diagnosable; user docs + a screenshot walkthrough. DoD: CodeArts corpus case diagnoses; Workflow badges match the dossier.

## 7. Quality strategy (the "ensure quality" mandate)

**Test matrix (all under the 3.13 `.venv`):**
| Test | Guards | Location |
|---|---|---|
| **Adapter parity** — DECAF adapter on TrajViz's raw dict ≡ `awe.load_normalized` on the corpus file, for claude_code/opencode/codex | the reuse seam doesn't silently diverge | `TrajectoryVisualizer/tests/test_attribution_parity.py` |
| **Golden diagnosis** — `diagnose()` on N committed corpus cases reproduces the exact `blame_status` + primary capability in `DECAF/data/dossier/faults.json` | in-memory path ≡ batch path; DECAF numbers unchanged | `tests/test_attribution_golden.py` |
| **Offline / no-key** — `diagnose(use_judge=False)` with `OPENROUTER_API_KEY` unset yields the 5-cap slice, no network | the default path never needs a key | `tests/test_attribution_offline.py` |
| **Cache read-back** — with judge cache present, 7 caps appear with no key | Phase 3 read path | same |
| **Gold-free degradation** — arbitrary upload → `available=False`, clean reason, no exception, no fabricated blame | the honesty guard | `tests/test_attribution_degraded.py` |
| **UI smoke** — build the Attribution tab outputs for a known case; assert scorecard + dossier HTML contain the expected primary capability | wiring & renderers | `tests/test_attribution_ui.py` |

**End-to-end verification (not just unit tests):** launch the Gradio app under `.venv`, load `trajectory/claude_code/<a-known-failing-id>.json`, open Attribution, and confirm the rendered primary cause + evidence chain match that case's `faults.json` record. Do this before each phase's PR (the `verify`/`run` workflow).

**Non-negotiables:** DECAF suite stays 448-green and `run_validation` PASS after integration (DECAF is untouched except a possible additive `adapt_codearts`); TrajViz suite stays green; no new default LLM spend; golden test wired into CI so a DECAF change that shifts numbers surfaces here.

## 8. Risks & mitigations

1. **Gold availability (central).** Corpus trajectories have gold; arbitrary uploads don't. → three explicit modes (§5.3), honest degradation, and `(agent, instance_id)` derived from the corpus path.
2. **Normalizer divergence drift.** Two adapters for the same logs. → reuse DECAF's adapters at the seam (single source there) + parity tests; unification tracked as future work, not attempted now.
3. **Blocked LLM providers.** `anthropic/*`, `openai/*` are 403 on this key. → default offline; lazy compute defaults to `z-ai/glm-5.2`; never hard-depend on the judge.
4. **Python version split.** 3.9 vs 3.11+ vs 3.13. → standardize on the repo `.venv` (3.13); document it; add `pytest`.
5. **Cross-project coupling.** → one-directional (`trajviz`→`awe`), funneled through one module, both suites still run independently.
6. **CodeArts unsupported by DECAF.** → degrade for CodeArts in Phases 1–3; add `adapt_codearts` in Phase 4.
7. **Positional wiring fragility** in `insight.py` (tuple/`all_outputs`/`_empty_result` must stay aligned). → a builder-returns-dict pattern + the UI smoke test catches misalignment.

## 9. Future work (explicitly deferred)
- Unify the two trajectory normalizers into one shared model (kills the duplicated adapters).
- Wire DECAF's `counterfactual.status` (currently `not_tested`) into an actual interventional re-run driven from the UI.
- Batch/corpus mode in TrajViz (diagnose a whole agent's run, not one trajectory) — reuses `scoring.agent_capability_scores`.
