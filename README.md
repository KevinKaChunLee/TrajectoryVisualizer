# TrajectoryVisualizer

**Offline analytics & visualization for LLM agent trajectories.**

TrajectoryVisualizer loads a single agent trajectory (or compares two), parses it into a normalized step model, and renders an interactive Gradio + Plotly dashboard covering tokens, timing, tool-use patterns, phase composition, anti-pattern detections, and cross-trajectory divergence.

Supports trajectories from **Claude Code**, **OpenCode**, and **CodeArts** out of the box.

---

## Install

```bash
git clone https://github.com/rshu/TrajectoryVisualizer.git
cd TrajectoryVisualizer
pip install -e .                     # package-managed (recommended)
# — or —
pip install -r requirements.txt      # plain requirements file
```

Requires Python 3.12+.

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
| Claude Code | `format: ccsession-trajectory` | Full support: tokens, cache, tool calls, thinking. Produced by [ccsession](https://github.com/rshu/ccsession) (see below). |
| OpenCode | `info` + `messages` shape | Includes sub-agent sessions |
| CodeArts | `format: codearts` | Sub-agent `session_id` threading |

---

## Collecting trajectories

TrajectoryVisualizer only **reads** trajectory files — producing them is the agent's responsibility. This section shows how to obtain a valid trajectory JSON for each supported agent.

### Claude Code

Claude Code writes one JSONL file per session under
`~/.claude/projects/<url-encoded-cwd>/<session-id>.jsonl`. The raw JSONL is not
directly consumable — use the [**ccsession**](https://github.com/rshu/ccsession)
exporter to convert it into the `ccsession-trajectory` JSON format that
TrajectoryVisualizer expects.

1. Install `ccsession` (see the tool's README for current instructions).
2. Run a Claude Code session against your repo as normal (`claude` CLI or VS
   Code extension).
3. Find the session ID — it is the basename of the most recent `*.jsonl`
   under `~/.claude/projects/<project>/`, and is also echoed at the top of
   each Claude Code session.
4. Export that specific session with `ccsession`'s export mode. Example with
   a real session ID:
   ```bash
   ccsession export --session-id f33cdb42-0a41-40d4-91eb-c89c109af38a
   ```
   This writes an export folder containing
   several files (`trajectory.json`, `raw_messages.jsonl`,
   `conversation_full.md`, …).
5. Upload the resulting `trajectory.json` in the
   Insight dashboard — the loader detects the `format: ccsession-trajectory`
   marker and normalizes the step model automatically.

### OpenCode

OpenCode stores sessions in its local store (`~/.local/share/opencode/`) and
exposes an `export` command that writes trajectory JSON to stdout.

1. Run an OpenCode session as normal.
2. List recent sessions and pick the one you want — **run this from the same
   working directory where the session was created**; `opencode session list`
   is scoped to the current directory's project ID:
   ```bash
   cd /path/to/your/project
   opencode session list --format json -n 10
   ```
3. Export the chosen session to JSON (banner output goes to stderr, so
   redirection produces a clean file):
   ```bash
   opencode export <session_id> > op_trajectory.json
   ```
4. Upload `op_trajectory.json` in the Insight dashboard. The loader detects the
   `info` + `messages` shape automatically; sub-agent sessions are threaded in.

### CodeArts

A CodeArts session is a **folder**, not a single file. Each session directory
must contain **both** of the following files for the consolidator to pick it
up:

- `chat_baseInfo.json` — session metadata (title, chatId, timestamp, agent info)
- `messages_0.json` — the raw message list

TrajectoryVisualizer expects a single consolidated JSON, so use the helper in
`scripts/` (run from the repo root):

```bash
# Single session — pick the output path yourself
python scripts/codearts_consolidator.py path/to/<session-id> --output ca_trajectory.json
```

The consolidator wraps `chat_baseInfo.json` + `messages_0.json` into a single
JSON with `"format": "codearts"` at the top level. Upload the consolidated
file in the dashboard.

---

## Scripts

Helper utilities that live in `scripts/` (run from the repo root):

| File | Purpose |
|---|---|
| `codearts_consolidator.py` | Merge a CodeArts session folder (`chat_baseInfo.json` + `messages_0.json`) into a single consolidated JSON. Single-session and `--batch` modes. See the **CodeArts** collection section above. |
| `step_labeler.py` | LLM-based per-step classifier. Reads a trajectory and emits a sidecar `*_labeled.json` with phase and action tags from the taxonomy. |
| `TAXONOMY_REFERENCE.md` | Authoritative list of phase and action tags the labeler emits. Auto-loaded by `step_labeler.py` from its own directory. |

### Labeling a trajectory

`step_labeler.py` makes live LLM calls via `requests`, which is installed by
default with the rest of the project.

The labeler needs three config values — provide them via a `.env` file (in
the repo root or CWD) or as CLI flags. A sample
[`.env.example`](./.env.example) ships with the repo; copy it and fill in
your own key:

```bash
cp .env.example .env
# then edit .env and set LABEL_BASE_URL / LABEL_API_KEY / LABEL_MODEL
```

Required variables and their CLI-flag equivalents:

| Variable | CLI flag | Meaning |
|---|---|---|
| `LABEL_BASE_URL` | `--base-url` | API base URL (OpenAI-compatible) |
| `LABEL_API_KEY`  | `--api-key`  | API key |
| `LABEL_MODEL`    | `--model`    | Model name, e.g. `gpt-4o-mini`, `glm-4.6` |

Optional: `LABEL_PROVIDER` (`openai` | `anthropic`, default `openai`),
`LABEL_TEMPERATURE` (default `0.3`), `LABEL_MAX_TOKENS` (default `1024`).

Example invocations:

```bash
# Using a .env file in the repo root
python scripts/step_labeler.py cc_trajectory.json --output cc_trajectory_labeled.json

# Overriding config on the command line
python scripts/step_labeler.py cc_trajectory.json \
    --output cc_trajectory_labeled.json \
    --base-url https://api.openai.com/v1 \
    --api-key sk-... \
    --model gpt-4o-mini
```

Upload the trajectory **and** its labels sidecar in the dashboard's two upload
slots to unlock the semantic pattern detectors and label-phase charts.

---

## Project layout

```
TrajectoryVisualizer/
├── trajectory_visualizer/       # Python package
│   ├── core/                    # Detector framework (DetectorContext, PatternDetection, catalog)
│   ├── insight/                 # Single-trajectory dashboard
│   │   ├── loaders.py           # Format detection & normalization
│   │   ├── parser.py            # Step model
│   │   ├── metrics.py           # Per-step & session metrics
│   │   ├── analytics.py         # Phase detection, behavioral analytics
│   │   ├── charts.py            # Plotly chart builders (40+ figures)
│   │   ├── rendering.py         # HTML rendering (workflow cards, code blocks)
│   │   ├── diagnostics.py       # Failure chains, root causes
│   │   ├── patterns.py          # Tool sequences, anti-patterns
│   │   ├── scoring.py           # Quality scoring
│   │   ├── judge.py             # LLM-as-judge
│   │   ├── comparison.py        # Bridge to converge pipeline
│   │   ├── styles.py            # CSS (light/dark)
│   │   ├── insight.py           # Gradio UI builder
│   │   ├── __main__.py          # CLI entry
│   │   └── detectors/           # Anti-pattern detector modules
│   └── converge/                # Two-trajectory comparison pipeline
│       ├── canonical.py         # Step canonicalization
│       ├── alignment.py         # DP alignment algorithm
│       ├── milestones.py        # Milestone extraction & comparison
│       ├── divergence.py        # Divergence classification
│       ├── charts.py            # Comparison charts
│       └── rendering.py         # Comparison HTML report
├── scripts/                     # Trajectory helpers
│   ├── codearts_consolidator.py # CodeArts session → single JSON
│   ├── step_labeler.py          # LLM-based step classifier
│   └── TAXONOMY_REFERENCE.md    # Phase/action tag catalog
├── pyproject.toml               # Package metadata & dependencies
├── requirements.txt             # Mirror of pyproject runtime dependencies
├── .env.example                 # Sample env for the step labeler
├── LICENSE
└── README.md
```

---

## License

MIT — see [LICENSE](./LICENSE).
