"""Centralized help text registry for metric tooltips and section explanations."""

# Tooltip/subtitle text keyed by identifier. Only keys referenced by
# insight.py are live: the four Overview KPI cards (rendered as data-help
# tooltips) and the Overview section subtitles. Add an entry here only
# together with the UI code that renders it.
HELP_TEXT: dict[str, str] = {
    # KPI card metrics
    "steps": "Total number of conversation turns (user + assistant messages) in the trajectory.",
    "wall_clock": "Elapsed wall-clock time from first to last step, including idle gaps between steps.",
    "tokens": "Total tokens consumed across all steps: input + output + reasoning + cache read.",
    "tool_success": "Percentage of tool calls that completed without errors. 100% means no tool failures.",
    # Section subtitles
    "section_performance": "Token consumption, step timing, and overall resource usage patterns.",
    "section_efficiency": "Context growth over time and compression events.",
    "section_tools": "Tool usage frequency, outcome timeline, and behavioral diagnostics.",
    "section_agents": "Multi-agent breakdown, spawning relationships, and per-agent performance.",
    "section_diagnostics": "File targeting, errors, and context-window pressure — occupancy of each agent's window over steps, with compaction drops.",
}
