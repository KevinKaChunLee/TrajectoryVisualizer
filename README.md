# TrajectoryVisualizer

**Offline analytics & visualization for LLM agent trajectories.**

TrajectoryVisualizer loads a single agent trajectory (or compares two), parses it into a normalized step model, and renders an interactive Gradio + Plotly dashboard covering tokens, timing, tool-use patterns, phase composition, anti-pattern detections, and cross-trajectory divergence.

Supports trajectories from **Claude Code**, **OpenCode**, **Codex**, and **CodeArts** out of the box.

---

## Features

- **Overview tab** — KPI cards, token-usage timeline, step-duration heatmap, tool-outcome timeline, agent swimlane, error classification, context growth.
- **Workflow tab** — step-by-step cards with syntax-highlighted code, tool I/O, reasoning, and interactive detail pane.
- **Patterns tab** — 27+ anti-pattern detectors across six phases (intake, understand, plan, implement, validate, report) plus cross-cutting and semantic patterns.
- **Comparison tab** — two-trajectory alignment with milestone timelines, divergence waterfalls, segment-cost charts, and anchor-class analysis.
- **Scoring** — composite quality score with dimension breakdown and health verdict.
- **Judge (optional)** — LLM-as-judge trajectory evaluation via `openai` (install with `[judge]` extra).
- **Light & dark modes**, CJK font fallbacks, Pygments syntax highlighting, responsive layout.

---

## Install

```bash
git clone https://github.com/rshu/TrajectoryVisualizer.git
cd TrajectoryVisualizer
pip install -e .
```

With the optional LLM-judge feature:

```bash
pip install -e ".[judge]"
```

Requires Python 3.11+.

---

## Run

Launch the dashboard:

```bash
python -m trajectory_visualizer.insight
# or
trajectory-visualizer
```

Default: `http://localhost:7860`. Upload a trajectory JSON via the file picker at the top of the UI.

Custom port or public share link:

```bash
python -m trajectory_visualizer.insight --port 8080 --share
```

---

## Supported trajectory formats

TrajectoryVisualizer auto-detects and normalizes the following formats on load:

| Format | Detection | Notes |
|---|---|---|
| Claude Code (`ccsession-trajectory`) | `_cc_format: true` or `format` field | Full support: tokens, cache, tool calls, thinking |
| OpenCode | `info` + `messages` shape | Includes sub-agent sessions |
| Codex | `format: codex` | Auto-normalized to canonical step model |
| CodeArts | `format: codearts` | Sub-agent `session_id` threading |
| Generic/unknown | Fallback | Best-effort parse |

See [INSIGHT_GUIDE.md](./INSIGHT_GUIDE.md) for detailed chart and metric documentation.

---

## Project layout

```
trajectory_visualizer/
├── core/            # Detector framework (DetectorContext, PatternDetection, catalog)
├── insight/         # Single-trajectory dashboard
│   ├── loaders.py       # Format detection & normalization
│   ├── parser.py        # Step model
│   ├── metrics.py       # Per-step & session metrics
│   ├── analytics.py     # Phase detection, behavioral analytics
│   ├── charts.py        # Plotly chart builders (40+ figures)
│   ├── rendering.py     # HTML rendering (workflow cards, code blocks)
│   ├── diagnostics.py   # Failure chains, root causes
│   ├── patterns.py      # Tool sequences, anti-patterns
│   ├── scoring.py       # Quality scoring
│   ├── judge.py         # LLM-as-judge (optional)
│   ├── comparison.py    # Bridge to converge pipeline
│   ├── styles.py        # CSS (light/dark)
│   ├── insight.py       # Gradio UI builder
│   ├── __main__.py      # CLI entry
│   └── detectors/       # Anti-pattern detector modules
└── converge/        # Two-trajectory comparison pipeline
    ├── canonical.py     # Step canonicalization
    ├── alignment.py     # DP alignment algorithm
    ├── milestones.py    # Milestone extraction & comparison
    ├── divergence.py    # Divergence classification
    ├── charts.py        # Comparison charts
    └── rendering.py     # Comparison HTML report
tests/               # Pytest suite (insight, converge, core, detectors)
INSIGHT_GUIDE.md     # Chart-by-chart & metric reference
```

---

## Run tests

```bash
pip install -e ".[dev]"
pytest
```

Tests cover loaders, parser, metrics, analytics, diagnostics, scoring, judge, dark-mode rendering, pattern detectors, and converge alignment.

---

## License

MIT — see [LICENSE](./LICENSE).
