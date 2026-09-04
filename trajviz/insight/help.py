"""Centralized help text registry for metric tooltips and section explanations."""

# Tooltip/subtitle text keyed by identifier. Only keys referenced by
# presenters/overview.py / overview_tab.py are live: the four Overview KPI cards
# (rendered as data-help tooltips) and the Overview section subtitles.
# Add an entry here only together with the UI code that renders it.
HELP_TEXT: dict[str, str] = {
    # KPI card metrics
    "steps": "Total number of conversation turns (user + assistant messages) in the trajectory.",
    "wall_clock": "Elapsed wall-clock time from first to last step, including idle gaps between steps.",
    "tokens": "Total tokens consumed across all steps: input + output + reasoning + cache read.",
    "tool_success": "Percentage of tool calls that completed without errors. 100% means no tool failures.",
    # Section subtitles
    "section_performance": "Token consumption, step timing, and overall resource usage patterns.",
    "section_context_utilization": "How loaded tokens break down. Empty categories are omitted. Window limit defaults to 128k (or the inferred model size) and can be changed. Select one agent to inspect the window before a compaction. Harness system definitions (not included in log) is billed overhead this export did not record.",
    "section_tools": "Tool usage frequency, per-call duration by tool (hover for step), Skill-tool calls by agent, outcome timeline (by agent when multi-agent), and behavioral diagnostics.",
    "section_agents": "Multi-agent breakdown, spawning relationships, and per-agent performance.",
    "section_diagnostics": "File targeting, errors, and root-cause attribution. The file timeline legend maps marker shape: circle = read, square = write, triangle = search, star = skill.",
}
