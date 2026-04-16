# TrajectoryVisualizer Guide

A reference for every chart, table, and card in the TrajectoryVisualizer dashboard. Organized by the UI layout so you can look up what you see on screen. Includes a [Metrics Reference](#metrics-reference) explaining how each metric is computed and what it reveals.

---

## Table of Contents

- [File Loader](#file-loader)
- [Overview Tab](#overview-tab)
  - [Session Context Bar](#session-context-bar)
  - [KPI Cards](#kpi-cards)
  - [Performance](#performance)
  - [Efficiency](#efficiency)
  - [Tools](#tools)
  - [Agents](#agents)
  - [Diagnostics](#diagnostics)
  - [Analytics](#analytics)
  - [Per-Step Deep Dive](#per-step-deep-dive)
  - [Labels](#labels)
- [Workflow Tab](#workflow-tab)
- [Patterns Tab](#patterns-tab)
- [Comparison Tab](#comparison-tab)
- [Summary Banner](#summary-banner)
- [Metrics Reference](#metrics-reference)
  - [Per-Step Metrics](#per-step-metrics)
  - [Session-Level Metrics](#session-level-metrics)
  - [Health Verdicts](#health-verdicts)

---

## File Loader

Always visible at the top of the page. Single-row layout with three columns:
- **Format dropdown** (left, scale 1): Select trajectory format — Claude Code, CodeArts, or OpenCode. The file is validated against the selected format on load; mismatches produce an error.
- **Trajectory file upload** (center, scale 2): Accepts `.json` files. Drag-and-drop or click to upload. "Load Trajectory" button below.
- **Labels file upload** (right, scale 2): Optional labeled JSON. "Load Labels" button below.

The upload area remains visible after loading so you can see the loaded filename and drag a new file at any time. The summary banner appears below once a trajectory is loaded. Warnings for step truncation (>2000 steps) and zero-token data appear above the banner.

---

## Overview Tab

### Session Context Bar

Displayed above the KPI cards (always visible, no accordion). A flex-wrap grid of small chip cards answering "who ran what, where, and when." Built by `_build_session_detail_html` in `insight.py`. These are **identity fields** extracted from the trajectory JSON metadata, not computed metrics.

| Field | Source | What It Means |
|-------|--------|---------------|
| **Model** | `model_id` from the first assistant step, or `metadata.model` as fallback | Which LLM model executed the task. Critical for comparing runs. |
| **Agent** | `agent` field from the first assistant step with a non-empty agent ID | The agent framework instance identifier (hex ID for Claude Code). |
| **Version** | `metadata.server_version` (only shown if present) | Agent framework version (e.g., Claude Code `2.1.71`). |
| **Start** | `timing.started_at`, truncated to `YYYY-MM-DD HH:MM:SS` | When the session began. |
| **End** | `timing.finished_at`, truncated to `YYYY-MM-DD HH:MM:SS` | When the session finished. |
| **Session** | `metadata.session_id`, truncated to first 16 characters | Unique session identifier for cross-referencing with logs. |
| **Branch** | `metadata.branch` | Git branch the agent was working on. |
| **Directory** | `metadata.directory_name` | Working directory name (project/repo). |
| **Platform** | `metadata.platform`, truncated to 24 chars | OS/platform (e.g., `linux`, `darwin`). |

---

### KPI Cards

A single row of 4 cards with colored left borders (green/yellow/red verdict), sparkline trends (for Tokens and Wall-Clock), and hover tooltips. Built by `_build_overview_kpi_html` in `insight.py`.

| Card | Value | Computation | Insight | Verdict Thresholds |
|------|-------|-------------|---------|-------------------|
| **Steps** | Total step count | `len(steps)`. Subtitle shows assistant/user breakdown. | Orientation metric. Verdict color from error count. | Errors: 0=good, 1-2=warn, 3+=bad |
| **Wall-Clock** | Total duration | `timing.total_duration` or sum of step durations. Subtitle shows P95. Sparkline shows per-step durations. | How long the agent ran. P95 shows the slow tail. | N/A |
| **Tokens** | Total token count | `sum(s["tokens"]["total"])`. Subtitle shows tok/s. Sparkline shows per-step tokens. | Cost proxy. | Throughput: >=50=good, 20-50=warn, <20=bad |
| **Tool Success** | Success rate % | `successful / total * 100`. Unknown-status calls count in denominator only. | Reliability. Below 80% = systematic issues. | >=95%=good, 80-95%=warn, <80%=bad |

---

### Performance

#### Performance Metrics Panel

Chip grids grouped into 6 sections. Rendered by `format_performance_md()` in `formatting.py`. All values from `compute_metrics()` in `metrics.py`.

**Timing** — step duration statistics:

| Chip | Computation | What It Reveals |
|------|-------------|-----------------|
| Steps | `len(steps)` | Total parsed conversation steps. |
| Wall-clock | `timing.total_duration` or sum of step durations | Total elapsed time including idle gaps. |
| Avg duration | `total_duration / count(durations)` | Mean step duration. Skewed by outliers. |
| Med / P95 | Median and 95th percentile of step durations | Typical step (median) vs slow-tail (P95). |
| Max duration | `max(durations)` | Single slowest step. |

**Tokens** — token breakdown by category:

| Chip | Computation | What It Reveals |
|------|-------------|-----------------|
| Total tokens | `sum(s["tokens"]["total"])` | Overall token consumption (cost proxy). |
| Input | `sum(s["tokens"]["input"])` | Total input tokens. May include cache reads depending on provider. |
| Output | `sum(s["tokens"]["output"])` | Total generated tokens. |
| Reasoning | `sum(s["tokens"]["reasoning"])` | Chain-of-thought tokens. High = heavy deliberation. |
| Cache read | `sum(s["tokens"]["cache_read"])` | Tokens reused from KV cache. High = efficient. |
| Cache write | `sum(s["tokens"]["cache_write"])` | Tokens written to cache for future turns. |
| Fresh input | `infer_non_cache_input()` summed, as `count (ratio%)` | Genuinely new content. Auto-detects provider token schema. |

**Efficiency** — throughput and cost ratios:

| Chip | Computation | What It Reveals |
|------|-------------|-----------------|
| Avg tok/step | `total_tokens / total_steps` | Average token cost per step. |
| Tok/sec | `total_tokens / total_duration` | Overall throughput (includes idle time). |
| Med tok/sec | Median of per-assistant-step `tokens / duration` | Typical per-step throughput. |
| Out/In ratio | `total_output / total_input` | Generation intensity. Low (<0.1) = reading. High (>0.5) = writing. |
| Tok/tool call | `total_tokens / tool_call_count` | Token cost per tool invocation. |

**Tool calls** — one chip per tool name (count), sorted by frequency. Header shows total and success rate.

**Agent breakdown** — shows "main agent" with step count, plus "sub-agent {id}" chips for each sub-agent. Only shown for multi-agent sessions. Header shows composition (e.g., "1 main + 2 sub-agents").

**Model breakdown** — one chip per model (step count). Only shown when multiple models were used.

#### Token Usage Chart

- **What it shows**: Stacked bar chart with one bar per step showing token breakdown: **Fresh Input** (blue), **Cache Read** (emerald), **Output** (amber), **Reasoning** (violet). Per-step view only.
- **Data source**: Per-step token counts. See `build_token_chart` in `charts.py`.
- **How to read it**: Tall bars = expensive steps. Bars dominated by Cache Read (green) mean most input was cached. Outlier annotations flag unusually large steps. Click legend to toggle categories.
- **Note**: For sessions with Anthropic prompt caching, Cache Read typically dominates (99%+) — this is expected and indicates efficient context reuse.

#### Step Duration Chart

- **What it shows**: Bar chart with one bar per step showing duration in seconds. Blue bars = normal steps; red bars = error steps (overlay mode). Grey dashed line = average.
- **Data source**: Per-step duration from timestamps. See `build_duration_chart` in `charts.py`.
- **How to read it**: The average line shows the mean step duration. Outlier labels appear above extreme bars and are tied to their trace — they hide when the trace is toggled off. Click a legend item to isolate that trace (e.g., click "Error" to see only error steps). Double-click to restore all.
- **Legend behavior**: `itemclick="toggleothers"` — single click isolates, double click restores. X-axis is fixed to the full step range regardless of which trace is visible. Bar width is fixed to prevent Plotly from auto-expanding sparse traces.

---

### Efficiency

#### Context Growth Chart

- **What it shows**: Line chart of cumulative context size (tokens) over steps, split by session. A dashed red line marks the context window limit (192K for CodeArts). Orange triangle-down markers flag **compression events** (token drops within a session).
- **Data source**: Raw cumulative `total_tokens` from CodeArts trajectory entries (before delta conversion). Sessions are identified by `sessionID` (derived from `chatId`), not by token drops. Zero-token tool dispatch steps are skipped. See `build_context_growth_ca_chart` in `charts.py`.
- **How to read it**: Each colored trace is a separate session — "main" for the primary agent, "sub {id}" for sub-agents (ID matches the agent breakdown grid). When a line approaches the context limit, the agent risks compression or degraded performance. Compression markers show where the context was truncated mid-session.
- **Note**: Only meaningful for CodeArts trajectories. For Claude Code/OpenCode, the chart shows "No context growth data available".

---

### Tools

#### Behavioral Diagnostics Grid

A chip grid of behavioral metrics with verdict badges (green/amber/red left borders) and hint text. Built by `format_behavioral_md()` in `formatting.py`. Metrics come from `compute_metrics()` and `compute_diagnostic_metrics()`. Chips that depend on format-specific data (cache breakdown, tool timestamps) show "N/A" when the data is unavailable.

| Chip | What It Measures | Computation | Verdict | Why It Matters |
|------|-----------------|-------------|---------|----------------|
| **Asst steps** | Number of assistant (LLM) turns | Count of steps with `role == "assistant"` | — | Trajectory size baseline. All ratios below are relative to this. |
| **Multi-tool** | Steps where the agent invoked 2+ tools in one turn | Count of assistant steps with `len(tool_calls) >= 2` | — | High count may indicate efficient parallel tool use, or thrashing if paired with errors. |
| **No-tool** | Assistant steps with zero tool calls | Count of assistant steps with empty `tool_calls` | — | Pure reasoning/summary steps. A high ratio (>30%) may mean excessive deliberation. |
| **Med tok/step** | Typical token cost per step | Median of per-assistant-step `tokens.total` | — | Robust central tendency — not skewed by occasional large steps. |
| **P95 tok/step** | How expensive the heavy-tail steps are | 95th percentile of per-assistant-step `tokens.total` | — | If P95 >> Median, a few steps dominate token consumption. |
| **Avg cache %** | Cache reuse effectiveness | Mean of per-step `cache_read / total_tokens` | ≥60% good, ≥30% warn, <30% bad | Shows "N/A" for formats without cache data (e.g., CodeArts). |
| **Cache-dom** | Steps where cache dominates input | Count of steps with `cache_ratio >= 0.90` | — | Only shown when cache data is available. |
| **Tool time** | Absolute time spent waiting for tools | `sum(tool_call_durations)` in seconds | — | Only shown when tool timestamps are available. |
| **Tool-wait %** | Fraction of wall-clock in tools vs inference | `tool_time / total_duration * 100` | ≤30% good, ≤60% warn, >60% bad | >60% = tool-bound session. Only shown when tool timestamps are available. |
| **Tool dur avg/P95/max** | Tool call latency distribution | Mean, 95th percentile, max of tool durations | — | Only shown when tool timestamps are available. |
| **Sub-agents** | Delegation to child agents | Count of sub-agent sessions + total steps | — | Only shown when sub-agents detected. |
| **Tool errors** | Errors from tool output content | Count of tool calls with `error_type` set | any = bad | Only shown when >0. |

#### Tool Call Frequency Chart

- **What it shows**: Horizontal bar chart of tool call counts by name. Stacked by agent in multi-agent sessions. Uses the same color palette as the Context Growth chart (`_SESSION_COLORS`). Sub-agents labeled as `"sub {id[:12]}"`.
- **Data source**: Per-step tool call data. See `build_tool_chart` in `charts.py`.
- **Layout**: Left half of a row; right half reserved for a future chart.

#### Tool Outcome Timeline

- **What it shows**: Scatter plot — circles = success, X marks = failure. Color-coded green/red. Each dot is one tool call positioned at its step index.
- **Data source**: Per-tool-call error/status flags. See `build_tool_outcome_timeline` in `charts.py`.
- **How to read it**: A field of green with isolated red marks = healthy. Clusters of red = systematic failures at specific steps. Tools with `status == "?"` (unknown, common in CodeArts) are treated as success.

---

### Agents

#### Agent Summary Cards

- **What it shows**: HTML card grid — one card per agent (assistant steps only) with steps, tokens, duration, tool calls, errors, cache %, tok/s. Cache % shows "N/A" when cache data is unavailable.
- **Data source**: Per-agent metrics from `compute_agent_summary()` in `metrics.py`. See `render_agent_summary_cards` in `rendering.py`.
- **Note**: User steps are excluded from agent counts. Sub-agents labeled as `"sub {id[:12]}"` using the same color palette as other agent charts.

#### Token Breakdown by Agent Chart

- **What it shows**: Grouped bar of token composition per agent (Fresh Input, Cache Read, Output, Reasoning) when breakdown data is available. Falls back to a simple total-tokens-per-agent bar chart when only totals are available (CodeArts).
- **Data source**: Per-agent aggregated tokens. See `build_agent_token_chart` in `charts.py`.
- **Layout**: Left half of a row; right half reserved for a future chart.

#### Agent Swimlane Chart

- **What it shows**: Horizontal bars showing each agent's active step ranges. Uses the same `_SESSION_COLORS` palette and `"sub {id[:12]}"` labels as other agent charts. Multi-agent only.
- **Data source**: Per-step agent assignment. See `build_agent_swimlane_chart` in `charts.py`.

---

### Diagnostics

#### Diagnostics Summary

- **What it shows**: One-line summary: failure chains, root causes, hotspots, files touched, files edited.
- **Example**: `"2 failure chain(s) · 2 root cause(s) · 5 hotspot(s) · 27 file(s) touched · 1 edited"`

#### File Interaction Timeline

- **What it shows**: Scatter plot (step x file path). circle/blue = read, diamond/green = write, triangle/orange = search. Target files (successfully edited) get red border. Full file paths shown on y-axis.
- **Data source**: Extracted from structured tool calls (Read, Write, Edit, Grep path, Glob pattern) and Bash commands via regex (Unix and Windows paths, quoted and unquoted). See `extract_file_interactions` in `diagnostics.py` and `build_file_interaction_chart` in `charts.py`.
- **How to read it**: Drag up/down to pan through files. Chart height is resizable (drag bottom edge). Dense clusters of blue dots = heavy reading phase. Green diamonds = file modifications. Red-bordered markers = target files.
- **Note**: Grep text patterns (e.g., "ImageVolume") are excluded — only Grep's `path` field (directory) is extracted. Glob patterns (e.g., `**/types.go`) are included as they represent file search targets.

#### Failure Chain Strip

- **What it shows**: Badge strip of contiguous error step sequences.
- **Data source**: Error chain analysis. See `build_failure_chain_strip_html` in `rendering.py`.

#### Bottleneck Cards

- **What it shows**: Cards for slowest steps with time decomposition: tool wait / inference / idle.
- **Data source**: Per-step time decomposition. See `build_bottleneck_cards_html` in `rendering.py`.

#### Root Cause Panel

- **What it shows**: Ranked root cause candidates with explanations.
- **Data source**: Root cause attribution. See `build_root_cause_html` in `rendering.py`.

#### Tool Error Classification Chart

- **What it shows**: Horizontal bar chart of error types classified from Bash tool output content (operateCacheData). Categories: Platform Error (wrong OS commands), Permission Denied, Missing File.
- **Data source**: `_classify_tool_error()` in `loaders.py` parses `operateCacheData.bash.content` for error patterns. See `build_error_classification_chart` in `charts.py`.
- **How to read it**: Each bar = one error type with count. Hover shows affected step indices. Only appears when CodeArts trajectories have `operateCacheData`.

#### Plan Progress Timeline

- **What it shows**: Horizontal Gantt-style chart showing each todo item as a bar spanning from first `in_progress` to `completed` step. Green = completed, red = stalled (>20 steps without completion), amber = in_progress but not completed. Plan resets appear as separate groups.
- **Data source**: `extract_plan_history()` and `compute_plan_metrics()` in `patterns.py`, tracking TodoWrite tool calls. See `build_plan_timeline_chart` in `charts.py`.
- **How to read it**: Wide bars = items that took many steps. Red bars = stalled items (agent was stuck). Multiple groups = plan resets between phases.

#### Task Mode Distribution

- **What it shows**: Bar chart of task mode (menuTask) values across assistant steps. Values like `plan-execute`, `code-generate`, `code-fix`.
- **Data source**: `menuTask` field from `codeOperateData` in CodeArts messages. See `build_task_mode_chart` in `charts.py`.
- **How to read it**: Shows what type of work the agent was doing. Heavy `code-fix` suggests debugging struggles. CodeArts-specific — hidden when field is absent.

---

### Analytics

#### Error Detail Chart

- **What it shows**: Horizontal bar of error types with counts. Hover shows affected step indices.
- **Data source**: Per-tool-call error fields. See `build_error_detail_chart` in `charts.py`.

#### Per-Phase Efficiency Radar Chart

- **What it shows**: Radar chart comparing phases on Tokens/Step, Tool Success Rate, Cache Hit Rate. Normalized 0-1.
- **Data source**: Per-phase aggregated metrics. See `build_phase_radar_chart` in `charts.py`.

---

### Per-Step Deep Dive

#### Hotspots

- **What it shows**: Tables of top 5 slowest steps, highest-token steps, and lowest-cache-ratio steps.
- **Data source**: Sorted per-step metrics. See `_build_hotspots_md` in `formatting.py`.

#### Per-Message Metrics Table

- **What it shows**: Table with one row per step (up to 80): index, role, duration, tokens, tool calls, cache ratio, errors.
- **Data source**: Per-step parsed metrics. See `_build_per_message_md` in `formatting.py`.

---

### Labels

These charts appear when a labeled JSON is loaded (phase/action annotations).

#### Phase Count Chart

- **What it shows**: Bar chart of step counts per phase (understand, plan, implement, debug, validate, report).

#### Action Count Chart

- **What it shows**: Horizontal bar of step counts per action, colored by parent phase.

#### Phase Duration Chart

- **What it shows**: Bar chart of total duration per phase.

#### Action Duration Chart

- **What it shows**: Horizontal bar of total duration per action, colored by phase.

#### Label Timeline Chart

- **What it shows**: Horizontal bar timeline — one bar per step, colored by phase, labeled by action. User prompts shown as grey markers.
- **Tips**: Phase oscillation (implement → debug → implement) indicates the agent struggled.

---

## Workflow Tab

### Workflow Step Cards

- **What it shows**: Vertical card stack — one per step with index, role badge, duration, tokens, collapsible content with syntax highlighting.
- Cards color-coded: red = error, teal = tool calls, purple = reasoning, blue = assistant, green = user.

### Filter Chips

- Toggleable buttons: Assistant, User, Tool Calls, Errors, Reasoning. Per-agent chips in multi-agent sessions.

### TOC Sidebar

- Compact navigation listing step numbers with role badges. Click to scroll.

---

## Patterns Tab

Detects recurring structural patterns: tool sequences, failure clusters, phase anomalies.

### Tool Sequence Patterns

- Recurring tool call sequences (e.g., "read → edit → read" repeated N times).

### Failure Patterns

- Clusters of similar errors grouped by type.

### Phase Anomalies

- Backward phase transitions categorized as **intentional iteration** (preceded by a planning step like TodoWrite/EnterPlanMode within 3 steps) or **unintentional drift** (no planning step before the regression). Each anomaly includes a confidence score proportional to the regressed phase's step span.

### Anti-Pattern Summary

- **What it shows**: Card-style panel summarizing detected anti-patterns with count, affected steps, and severity badges. Categories: tool errors (platform confusion), fruitless search streaks, Bash-for-reading (tool selection), stalled plan items.
- **Data source**: `detect_fruitless_streaks()`, `detect_tool_selection_antipatterns()`, `compute_plan_metrics()`, `_classify_tool_error()` in `patterns.py`/`loaders.py`. See `build_antipattern_summary_html` in `rendering.py`.
- **How to read it**: Each card = one anti-pattern type. Color-coded borders: red = errors, amber = fruitless/stalled, blue = tool selection. Shows "No anti-patterns detected" for clean trajectories.

---

## Comparison Tab

Converge-style trajectory comparison against a reference trajectory or anchor file.

### Comparison Report

- Divergence analysis, alignment scores, per-phase differences. Token rate normalization for fair comparison.

---

## Summary Banner

Appears below the upload area after loading. Shows: **filename**, step count, tool calls (success rate), total tokens, tok/s, wall-clock, reasoning count. Compact single-line format.

### Anomaly Strip

- Badges for statistical outliers: slowest step, most tokens, lowest cache ratio, most tool calls. Clickable.

---

## Metrics Reference

### Per-Step Metrics

Computed by `build_message_metrics()` in `metrics.py` and `compute_step_analytics()` in `analytics.py`.

| Metric | Computation | What It Reveals |
|--------|-------------|-----------------|
| **Fresh Input Tokens** | `total - output - reasoning - cache_read` via `infer_non_cache_input`. Auto-detects provider schema by comparing both interpretations against `total`. | Genuinely new content processed. |
| **Cache Ratio** | `cache_read / total_tokens` per step | Cache efficiency. 0% = all new; 90%+ = strong reuse. |
| **Tokens/sec** | `total_tokens / duration` per step | LLM throughput including tool wait time. |
| **Fresh Input tok/sec** | `non_cache_tokens / duration` per step | Throughput of new content only. |
| **Tool Time Share** | `sum(tool_call_durations) / step_duration` | Fraction of step spent waiting for tools. >80% = tool-bound. |
| **Output/Input Ratio** | `output_tokens / input_tokens` | Low = reading. High = heavy generation. |
| **Idle Gap** | `(current_step.time_created - previous_step.time_completed) / 1000` | Dead time between steps. |

### Session-Level Metrics

Aggregated by `compute_metrics()` in `metrics.py`.

**Token Metrics:**

| Metric | Computation | What It Reveals |
|--------|-------------|-----------------|
| **Non-Cache Ratio** | `non_cache_tokens / total_tokens * 100` | % of tokens that were fresh. Lower = better caching. |
| **Avg Tokens/Step** | `total_tokens / num_steps` | Average step cost. |
| **Median Step Tokens** | Median of assistant-step token counts | Robust central tendency (not skewed by outliers). |
| **Avg Cache Ratio** | Mean of per-assistant-step cache ratios | Overall cache effectiveness. >60% = strong. |
| **Tokens/Patch Line** | `(input + output) / patch_lines` | Token cost per line of code changed. |

**Tool Metrics:**

| Metric | Computation | What It Reveals |
|--------|-------------|-----------------|
| **Tool Success Rate** | `successful / total * 100` | Overall reliability. Unknown-status in denominator only. |
| **Tool Wait Share** | `total_tool_time / total_duration * 100` | Fraction of session spent on tools. |
| **Tool Calls/Min** | `tool_count / (total_duration / 60)` | Tool usage intensity. |

**Timing Metrics:**

| Metric | Computation | What It Reveals |
|--------|-------------|-----------------|
| **TTFT** | `first_assistant_completed - first_user_created` | Time to first response. |
| **Output tok/s** | `total_output_tokens / total_assistant_duration` | Pure generation throughput. |
| **Autonomy Ratio** | `assistant_turns / (user + assistant)` | 1.0 = fully autonomous. |

### Diagnostic Metrics

Computed by `compute_diagnostic_metrics()` in `metrics.py`. Requires raw trajectory data. Shown in the Behavioral Diagnostics chip grid.

| Metric | Computation | What It Reveals |
|--------|-------------|-----------------|
| **Sub-agent Sessions** | Count of distinct sub-agent session IDs | Delegation frequency. |
| **Sub-agent Total Tokens** | Sum of token deltas across sub-agent steps | Hidden cost from delegation. |
| **Fruitless Streak Count** | Number of streaks (>=3 consecutive empty tool outputs) | Search inefficiency. |
| **Fruitless Streak Max** | Length of longest streak | Worst-case blind exploration. |
| **Plan Stall Count** | Items `in_progress` for >20 steps without completion | Planning effectiveness. |
| **Plan Reset Count** | Times the plan item set changed completely | Re-planning frequency. |
| **Tool Selection Flags** | Bash commands using `sed/cat/head` that could use Read tool | Tool choice quality. |
| **Classified Error Count** | Bash errors detected from output content (platform, permission, missing file) | Errors invisible in standard error_count. |

### Health Verdicts

Computed by `compute_health_verdict()` in `metrics.py`. Displayed as colored KPI card borders.

**KPI Card Verdicts:**

| Metric | Good (green) | Warn (yellow) | Bad (red) |
|--------|-------------|---------------|-----------|
| **Cache Efficiency** | Avg cache ratio >= 60% | 30-60% | < 30% |
| **Tool Success** | >= 95% | 80-95% | < 80% |
| **Throughput** | >= 50 tok/s | 20-50 tok/s | < 20 tok/s |
| **Errors** | 0 error steps | 1-2 error steps | > 2 error steps |

**Behavioral Diagnostics Verdicts** (chip left-border colors):

| Metric | Good (green) | Warn (yellow) | Bad (red) |
|--------|-------------|---------------|-----------|
| **Avg cache %** | >= 60% | 30-60% | < 30% (None if 0%) |
| **Tool-wait %** | <= 30% | 30-60% | > 60% |
