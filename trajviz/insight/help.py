"""Centralized help text registry for metric tooltips and section explanations."""

# Metric tooltip definitions keyed by metric/section identifier.
# Used by rendering code to add data-help attributes for CSS tooltips.
HELP_TEXT: dict[str, str] = {
    # KPI card metrics
    "steps": "Total number of conversation turns (user + assistant messages) in the trajectory.",
    "wall_clock": "Elapsed wall-clock time from first to last step, including idle gaps between steps.",
    "tokens": "Total tokens consumed across all steps: input + output + reasoning + cache read.",
    "tool_success": "Percentage of tool calls that completed without errors. 100% means no tool failures.",
    "cache_read_pct": "Percentage of input tokens served from prompt cache vs. fresh computation. Higher values indicate better caching efficiency.",
    "fresh_input": "Percentage and count of input tokens that were not served from cache, requiring full computation.",

    # Performance metrics
    "tokens_per_second": "Average token throughput: total tokens divided by total wall-clock time.",
    "p95_duration": "95th percentile step duration — 95% of steps completed faster than this value.",
    "avg_duration": "Mean duration across all steps, in seconds.",
    "median_duration": "Median (50th percentile) step duration, less sensitive to outliers than the mean.",

    # Efficiency metrics
    "cache_ratio": "Per-step ratio of cache-read tokens to total tokens. Higher means more prompt reuse.",
    "context_growth": "Cumulative input tokens over time, showing how context window pressure increases.",
    "tokens_per_sec_step": "Per-step throughput: tokens produced divided by step duration.",
    "tool_time_share": "Fraction of step duration spent waiting for tool call results (vs. model inference).",
    "out_in_ratio": "Ratio of output tokens to input tokens for a step. High values suggest generative steps.",

    # Tool metrics
    "tool_call_count": "Total number of tool invocations across all steps.",
    "tool_call_frequency": "How many times each tool was called, sorted by frequency.",
    "tool_duration": "Total wall-clock time spent in each tool type, including I/O wait.",

    # Agent metrics
    "agent_tokens": "Token breakdown per agent: fresh input, cache read, output, and reasoning.",
    "agent_swimlane": "Timeline showing which agent was active at each step range.",

    # Analytics metrics
    "heatmap": "Normalized (0–1) per-step metrics across 6 dimensions. Darker cells indicate higher relative values.",

    # Section explanations
    "section_performance": "Token consumption, step timing, and overall resource usage patterns.",
    "section_efficiency": "Context growth over time and compression events.",
    "section_tools": "Tool usage frequency, outcome timeline, and behavioral diagnostics.",
    "section_agents": "Multi-agent breakdown, spawning relationships, and per-agent performance.",
    "section_analytics": "Behavioral patterns, phase segmentation, and anomaly detection.",

    # Filter features
    "filter_chips": "Use these filters to focus on specific step types. Click chips to toggle visibility.",
    "keyword_search": "Filter steps by keyword — matches against text content, tool names, and tool arguments.",
}
