# TrajViz

**Offline analytics & visualization for LLM agent trajectories.**

TrajViz loads a single agent trajectory (or compares two), parses it into a normalized step model, and renders an interactive Gradio + Plotly dashboard covering tokens, timing, tool-use patterns, phase composition, anti-pattern detections, step-label analysis, and cross-trajectory divergence.

Supports trajectories from **Claude Code**, **OpenCode**, **CodeArts**, and **Codex CLI** out of the box.

---

## Install

```bash
git clone https://github.com/rshu/TrajectoryVisualizer.git
cd TrajectoryVisualizer
```

Pick one of the workflows below. Requires Python 3.11+.

### uv (recommended)

[`uv`](https://github.com/astral-sh/uv) is the fastest path and produces a
reproducible `uv.lock`. From the repo root:

```bash
uv sync                              # creates .venv, installs the project + deps
```

That's it — no separate `pip install -e .` step. To add or upgrade a dependency
later, use `uv add <pkg>` / `uv lock --upgrade`.

### pip

```bash
pip install -e .                     # package-managed
# — or —
pip install -r requirements.txt      # plain requirements file
```

---

## Run

Launch the dashboard:

```bash
# uv
uv run trajviz
# or
uv run python -m trajviz.insight

# pip / activated venv
python -m trajviz.insight
# or
trajviz
```

Default: `http://localhost:7860`. Upload a trajectory JSON via the file picker at the top of the UI.

Custom port or host (prefix any of these with `uv run` when using uv):

```bash
# Different port
python -m trajviz.insight --port 8080

# Access from any IP on the network
python -m trajviz.insight --host 0.0.0.0
# Then access it via your server's IP address: `http://YOUR_IP:7860`

# Create a public share link
python -m trajviz.insight --share
```

---

## Common Issues

### Server cannot start

If the dashboard fails to start or shows connection errors, your environment may be using a proxy configuration that interferes with localhost connections. Add `127.0.0.1,localhost` to the `no_proxy` environment variable:

```bash
# Linux/Mac
export no_proxy="127.0.0.1,localhost,$no_proxy"

# Windows (PowerShell)
$env:no_proxy = "127.0.0.1,localhost,$env:no_proxy"

# Windows (CMD)
set no_proxy=127.0.0.1,localhost;%no_proxy%
```

### Port already in use

If port 7860 is already in use, specify a different port:

```bash
python -m trajviz.insight --port 8080
```

---

## Supported trajectory formats

trajviz auto-detects and normalizes the following formats on load:

| Format | Detection | Notes |
|---|---|---|
| Claude Code | `format: ccsession-trajectory` | Full support: tokens, cache, tool calls, thinking. Produced by [ccsession](https://github.com/rshu/ccsession) (see below). |
| OpenCode | `info` + `messages` shape | Includes sub-agent sessions |
| CodeArts | `export_metadata.source_format: codearts_opencode_sqlite` with schema version 2 | Preserved token breakdown and consolidated parent/sub-agent sessions |
| Codex CLI | `.jsonl` rollout starting with a `session_meta` event | Normalized into the shared step model (select **Codex** in the format dropdown); tool intent (Read / Grep / Glob / Write / Bash) inferred from classic `exec_command` calls and modern `exec` / `apply_patch` records |

---

## Collecting trajectories

trajviz only **reads** trajectory files — producing them is the agent's responsibility. This section shows how to obtain a valid trajectory JSON for each supported agent.

### Claude Code

Claude Code writes one JSONL file per session under
`~/.claude/projects/<url-encoded-cwd>/<session-id>.jsonl`. The raw JSONL is not
directly consumable — use the [**ccsession**](https://github.com/rshu/ccsession)
exporter to convert it into the `ccsession-trajectory` JSON format that
trajviz expects.

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
3. Export the chosen session to JSON. Two paths depending on whether the
   session spawned sub-agents:

   **Typical case — no special sub-agents.** Use OpenCode's built-in
   exporter (banner output goes to stderr, so redirection produces a clean
   file):
   ```bash
   opencode export <session-id> > op_trajectory.json
   ```

   **Customized case — session has special sub-agents.** Sub-agent invocations live
   under their own session IDs, so `opencode export` on the parent only gives
   you the parent's messages. Use the consolidator helper in `scripts/` to
   recursively pull the parent and every child session into one JSON:
   ```bash
   python scripts/opencode_consolidator.py <session-id> op_trajectory.json
   ```
   The consolidator follows tool-call metadata to discover child session IDs
   and emits a flat `{"sessions": [...]}` structure that the loader threads
   into a single trajectory.
4. Upload `op_trajectory.json` in the Insight dashboard. The loader detects the
   `info` + `messages` shape automatically; sub-agent sessions are threaded in.

### CodeArts

Current CodeArts AgentKernel builds persist sessions in an `opencode.db`
SQLite store. Export a root session together with all child/sub-agent
sessions using the read-only consolidator:

```bash
python scripts/codearts_consolidator.py path/to/opencode.db \
  --session-id <session-id> --output ca_trajectory.json
```

The consolidator follows both `session.parent_id` and tool-result session
metadata, preserves stored token dictionaries and precise part timing, and
emits the `info` + `messages` envelope with an `export_metadata` block whose
`source_format` is `codearts_opencode_sqlite` — the marker the dashboard
detects. Pass `--no-children` only when a root-only export is intentional. A
bare session ID can also be used when `CODEARTS_DATABASE` or
`OPENCODE_DATABASE` points at the database.

Upload the export JSON and select **CodeArts**. Already-consolidated export
files can of course be uploaded directly.

The consolidator can also merge an old folder-based session
(`chat_baseInfo.json` + every `messages_<n>.json` shard) into a single
archival JSON. That legacy output preserves the original message records for
reproducibility but is **not** loadable by the dashboard, which only supports
the current export format.

### Codex CLI

Codex CLI records every session automatically as a JSONL rollout under
`~/.codex/sessions/` (date-bucketed `rollout-<timestamp>-<session-id>.jsonl`
files) — no exporter needed.

1. Run a Codex session as normal.
2. Locate the rollout file for the session — the most recent
   `rollout-*.jsonl` under `~/.codex/sessions/`.
3. Upload the `.jsonl` file and select **Codex** as the format. The loader
   detects the leading `session_meta` event and threads the rollout into the
   shared step model. Per-step tool intent (Read / Grep / Glob / Write / Bash)
   is inferred from classic `exec_command` calls and modern `exec` /
   `apply_patch` records.

---

## Scripts

Helper utilities that live in `scripts/` (run from the repo root):

| File | Purpose |
|---|---|
| `codearts_consolidator.py` | Read-only export from a CodeArts `opencode.db` with recursive child sessions, or lossless archival merge of legacy `messages_<n>.json` shards. See the **CodeArts** collection section above. |
| `opencode_consolidator.py` | Recursively merge an OpenCode parent session and child sub-agent sessions into a single JSON. See the **OpenCode** collection section above. |
| `step_labeler.py` | LLM-based per-step classifier. Reads a trajectory and emits a sidecar `*_labeled.json` with phase and action tags from the taxonomy. |
| `step_labeler_v2.py` | Variant of `step_labeler.py` that emits one record for **every** parsed step: assistant steps via the LLM, user steps as deterministic `user/user_prompt`, with `index`/`raw_index` preserved for exact source mapping. |
| `TAXONOMY_REFERENCE.md` | Authoritative list of phase and action tags the labeler emits. Auto-loaded by `step_labeler.py` from its own directory. |

### Labeling a trajectory

`step_labeler.py` makes live LLM calls via
`requests`, which is installed by default with the rest of the project.

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
# Step behavior labels using a .env file in the repo root
python scripts/step_labeler.py cc_trajectory.json --output cc_trajectory_labeled.json

# Overriding config on the command line
python scripts/step_labeler.py cc_trajectory.json \
    --output cc_trajectory_labeled.json \
    --base-url https://api.openai.com/v1 \
    --api-key sk-... \
    --model gpt-4o-mini
```

Upload the trajectory **and** its labels sidecar in the dashboard's two upload
slots to unlock phase-aware analytics and label-phase charts.

---

## Project layout

```
TrajectoryVisualizer/
├── trajviz/       # Python package
│   ├── insight/                 # Single-trajectory dashboard
│   │   ├── loaders.py           # Format detection & normalization
│   │   ├── parser.py            # Step model
│   │   ├── metrics.py           # Per-step & session metrics
│   │   ├── analytics.py         # Phase detection, behavioral analytics
│   │   ├── charts.py            # Plotly chart builders
│   │   ├── rendering.py         # HTML rendering (workflow cards, code blocks)
│   │   ├── diagnostics.py       # Failure chains, root causes
│   │   ├── patterns.py          # Tool sequences, anti-patterns
│   │   ├── labels.py            # Phase/action label model
│   │   ├── comparison.py        # Bridge to converge pipeline
│   │   ├── formatting.py        # Markdown/HTML metric grids
│   │   ├── palette.py           # Shared chart & phase colors
│   │   ├── help.py              # Metric tooltip registry
│   │   ├── styles.py            # CSS (light/dark)
│   │   ├── insight.py           # Gradio UI builder
│   │   └── __main__.py          # CLI entry
│   └── converge/                # Two-trajectory comparison pipeline
│       ├── canonical.py         # Step canonicalization
│       ├── alignment.py         # DP alignment algorithm
│       ├── milestones.py        # Milestone extraction & comparison
│       ├── divergence.py        # Divergence classification
│       ├── anchor.py            # Ground-truth patch grounding
│       ├── eval_layers.py       # Diagnostic evaluation layers
│       ├── batch.py             # Manifest batch mode & aggregation
│       ├── intervention.py      # Before/after intervention comparison
│       ├── charts.py            # Comparison charts
│       ├── rendering.py         # Comparison HTML report
│       ├── cli.py               # Pairwise / batch / before-after CLI
│       ├── app.py               # Standalone Gradio comparison app
│       └── styles.py            # Comparison report CSS
├── scripts/                     # Trajectory helpers
│   ├── codearts_consolidator.py # CodeArts opencode.db → single export JSON
│   ├── opencode_consolidator.py # OpenCode parent + sub-agent sessions → single JSON
│   ├── step_labeler.py          # LLM-based step classifier
│   ├── step_labeler_v2.py       # Every-step labeler (assistant via LLM, user deterministic)
│   └── TAXONOMY_REFERENCE.md    # Phase/action tag catalog
├── tests/                       # Workflow UI regression tests
├── pyproject.toml               # Package metadata & dependencies
├── requirements.txt             # Mirror of pyproject runtime dependencies
├── .env.example                 # Sample env for the step labeler
├── LICENSE
└── README.md
```

---

## License

MIT — see [LICENSE](./LICENSE).
