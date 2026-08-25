"""AI-powered trajectory analysis.

Extracts trajectory data and generates automated insights using LLM.
"""

import logging
from typing import Any

import html

from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class AIAnalyzer:
    """Analyzes trajectories using LLM."""

    def __init__(self, client: LLMClient):
        """Initialize analyzer with LLM client.

        Args:
            client: LLM client instance
        """
        self.client = client

    def _extract_trajectory_summary(
        self,
        steps: list[dict],
        metrics: dict,
        patterns: dict | None = None,
        raw: dict | None = None,
        diagnostics: dict | None = None,
        agent_summaries: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Extract key information for analysis.

        Args:
            steps: Parsed step list
            metrics: Computed metrics
            patterns: Detected patterns
            raw: Raw trajectory data
            diagnostics: Diagnostic metrics including file interactions, failure chains, etc.
            agent_summaries: Per-agent summary statistics

        Returns:
            Summary dictionary for LLM context
        """
        summary = {
            "steps": len(steps),
            "metrics": {
                "total_tokens": metrics.get("tokens", {}).get("total", 0),
                "total_duration": metrics.get("duration", 0),
                "tool_call_count": metrics.get("tool_call_count", 0),
                "tool_success_rate": metrics.get("tool_success_rate", 0),
                "assistant_steps": metrics.get("assistant_steps", 0),
                "user_steps": metrics.get("user_steps", 0),
                "tokens_per_second": metrics.get("tokens_per_second", 0),
                "tokens_per_tool": metrics.get("tokens_per_tool", 0),
                "p95_tool_duration": metrics.get("p95_tool_duration", 0),
                "max_tool_duration": metrics.get("max_tool_duration", 0),
                "tool_system_failure_rate": metrics.get("tool_system_failure_rate", 0),
                "latency": metrics.get("latency", 0),
                "tool_wait_pct": metrics.get("tool_wait_pct", 0),
                "tool_time_share": metrics.get("tool_time_share", 0),
            },
        }

        if raw:
            if "metadata" in raw:
                summary["model"] = raw["metadata"].get("model", "unknown")
                summary["session_id"] = raw["metadata"].get("session_id", "unknown")[:16]
            if "timing" in raw:
                summary["started"] = raw["timing"].get("started_at", "")
                summary["finished"] = raw["timing"].get("finished_at", "")

            # Extract user's initial message(s) to understand the objective
            if "trajectory" in raw and isinstance(raw["trajectory"], list):
                user_messages = []
                for msg in raw["trajectory"][:10]:  # Look at first 10 messages
                    if msg.get("role") == "user" and msg.get("text"):
                        user_messages.append(msg["text"][:500])
                if user_messages:
                    summary["user_messages"] = user_messages

        if patterns:
            if "tool_sequences" in patterns:
                summary["top_tool_patterns"] = patterns["tool_sequences"][:5]
            if "failure_patterns" in patterns:
                summary["failure_count"] = len(patterns["failure_patterns"])

        step_types = {}
        for step in steps:
            role = step.get("role", "unknown")
            step_types[role] = step_types.get(role, 0) + 1
        
        summary["step_distribution"] = step_types

        token_by_step = [s.get("tokens", {}).get("total", 0) for s in steps]
        if token_by_step:
            summary["token_stats"] = {
                "min": min(token_by_step),
                "max": max(token_by_step),
                "avg": sum(token_by_step) / len(token_by_step),
            }

        # Add diagnostics information
        if diagnostics:
            if "file_targeting" in diagnostics:
                ft = diagnostics["file_targeting"]
                summary["file_interactions"] = {
                    "unique_files": ft.get("unique_files", 0),
                    "edited_files": ft.get("target_files_count", 0),
                    "read_count": ft.get("read_count", 0),
                    "write_count": ft.get("write_count", 0),
                    "search_count": ft.get("search_count", 0),
                }

            if "chain_metrics" in diagnostics:
                cm = diagnostics["chain_metrics"]
                summary["failure_chains"] = {
                    "total_chains": cm.get("total_chains", 0),
                    "longest_chain_length": cm.get("longest_chain_length", 0),
                    "avg_chain_length": cm.get("avg_chain_length", 0),
                    "chained_error_rate": cm.get("chained_error_rate", 0),
                }

            if "error_clusters" in diagnostics:
                clusters = diagnostics["error_clusters"]
                if clusters:
                    summary["error_clusters"] = [
                        {
                            "pattern": c.get("pattern", ""),
                            "count": c.get("count", 0),
                            "steps": c.get("steps", [])[:3],
                        } for c in clusters[:5]
                    ]

            if "bottleneck_explanations" in diagnostics:
                bottlenecks = diagnostics["bottleneck_explanations"]
                if bottlenecks:
                    summary["bottlenecks"] = [
                        {
                            "step_num": b.get("step_num"),
                            "explanation": b.get("explanation", "")[:150],
                        } for b in bottlenecks[:3]
                    ]

        # Add agent swimlane information
        if agent_summaries:
            summary["agent_swimlane"] = [
                {
                    "agent_id": a.get("agent_id", ""),
                    "label": a.get("label", "unknown"),
                    "step_count": a.get("step_count", 0),
                    "total_tokens": a.get("total_tokens", 0),
                    "cache_efficiency": a.get("cache_efficiency_pct", 0),
                    "error_count": a.get("error_count", 0),
                } for a in agent_summaries
            ]

        # Add context management analysis
        trajectory = raw.get("trajectory", []) if raw else []
        if trajectory:
            from .diagnostics import generate_context_management_warnings
            context_warnings = generate_context_management_warnings(steps, trajectory)
            if context_warnings.get("subagent_skill_issues"):
                summary["context_management"] = {
                    "issues_found": len(context_warnings["subagent_skill_issues"]),
                    "warnings": context_warnings["subagent_skill_issues"][:5],  # Limit to top 5
                    "summary": context_warnings.get("summary", []),
                }
            elif context_warnings.get("summary"):
                summary["context_management"] = {
                    "issues_found": 0,
                    "summary": context_warnings.get("summary", []),
                }

        return summary

    def _get_language_guidelines(self) -> str:
        """Get language guidelines for LLM responses."""
        return """- **Language**:
- **All thinking, analysis, reasoning, explanations, and descriptions must be in Chinese (中文) unless otherwise specified**
- English only for: code, technical identifiers, JSON keys, file paths, `task_id` values"""

    def _get_html_usage_guidelines(self) -> str:
        """Get HTML usage guidelines for LLM responses."""
        return """Use HTML elements like <strong>, <code>, <ul>, <li> for emphasis."""

    def _build_trajectory_context(
        self,
        summary: dict[str, Any],
        steps: list[dict] | None = None,
        include_detailed_steps: bool = False,
    ) -> str:
        """Build trajectory context for LLM prompts.

        Args:
            summary: Trajectory summary
            steps: Optional detailed steps for step-by-step queries
            include_detailed_steps: Whether to include detailed step information

        Returns:
            Formatted trajectory context string
"""
        context = f"""## 轨迹参考数据
- 步数: {summary['steps']}
- 总Tokens: {summary['metrics']['total_tokens']:,}
- 时长: {summary['metrics']['total_duration']:.1f}s
- 工具成功率: {summary['metrics']['tool_success_rate']}%
- 助手步数: {summary['metrics']['assistant_steps']}
- 用户步数: {summary['metrics']['user_steps']}
"""

        # Additional metrics for deeper analysis
        context += f"""## 性能指标
- Tokens/秒: {summary['metrics']['tokens_per_second']:.2f}
- Tokens/工具: {summary['metrics']['tokens_per_tool']:.0f}
- P95工具时长: {summary['metrics']['p95_tool_duration']:.3f}s
- 最大工具时长: {summary['metrics']['max_tool_duration']:.3f}s
- 工具系统失败率: {summary['metrics']['tool_system_failure_rate']}%
- 延迟: {summary['metrics']['latency']:.3f}s
- 工具等待占比: {summary['metrics']['tool_wait_pct']:.1f}%
- 工具执行时间占比: {summary['metrics']['tool_time_share']:.1f}%
"""

        if "user_messages" in summary:
            context += "\\n## 用户目标\\n"
            context += "以下是用户的初始消息，理解任务目标：\\n"
            for i, msg in enumerate(summary["user_messages"][:3], 1):
                context += f"{i}. {msg}\\n"

        if "model" in summary:
            context += f"- 模型: {summary['model']}\\n"

        if "step_distribution" in summary:
            context += "\n## 步骤分布\n"
            for role, count in summary["step_distribution"].items():
                context += f"- {role}: {count}\n"

        if "token_stats" in summary:
            ts = summary["token_stats"]
            context += f"\n## Token统计\n- 最小值: {ts['min']:,}\n- 最大值: {ts['max']:,}\n- 平均值: {ts['avg']:,.0f}\n"

        if "top_tool_patterns" in summary:
            context += "\n## 重复工具模式\n"
            for i, pattern in enumerate(summary["top_tool_patterns"], 1):
                seq = " → ".join(pattern.get("sequence", []))
                context += f"{i}. {seq} (出现 {pattern.get('frequency', 0)} 次)\n"

        if "failure_count" in summary:
            context += f"\n## 问题\n- {summary['failure_count']} 个失败模式检测到\n"

        if "file_interactions" in summary:
            fi = summary["file_interactions"]
            context += f"\n## 文件交互\n"
            context += f"- 唯一文件数: {fi['unique_files']}\n"
            if fi['edited_files'] > 0:
                context += f"- 编辑文件数: {fi['edited_files']}\n"
            context += f"- 读取: {fi['read_count']} 次\n"
            context += f"- 写入: {fi['write_count']} 次\n"
            context += f"- 搜索: {fi['search_count']} 次\n"

        if "failure_chains" in summary:
            fc = summary["failure_chains"]
            if fc['total_chains'] > 0:
                context += f"\n## 失败链\n"
                context += f"- 总失败链数: {fc['total_chains']}\n"
                context += f"- 最长链: {fc['longest_chain_length']} 步\n"
                context += f"- 平均链长度: {fc['avg_chain_length']:.1f}\n"
                context += f"- 链式错误率: {fc['chained_error_rate']:.1f}%\n"

        if "error_clusters" in summary:
            ec = summary["error_clusters"]
            if ec:
                context += f"\\n## 错误聚类\\n"
                for cluster in ec:
                    pattern = cluster.get('pattern', '')
                    count = cluster.get('count', 0)
                    cluster_steps = cluster.get('steps') or []
                    context += f"- 模式: {pattern[:50]} (出现 {count} 次, 步骤 {cluster_steps[:5]})\\n"

        if "bottlenecks" in summary:
            bottlenecks = summary["bottlenecks"]
            if bottlenecks:
                context += f"\n## 性能瓶颈\n"
                for b in bottlenecks:
                    context += f"- 步骤 {b['step_num']}: {b['explanation']}\n"

        if "agent_swimlane" in summary:
            agents = summary["agent_swimlane"]
            if len(agents) > 1:
                context += f"\n## 多智能体泳道\n"
                for agent in agents:
                    label = agent['label']
                    context += f"- {label}: {agent['step_count']} 步, {agent['total_tokens']} tokens"
                    if agent['cache_efficiency'] > 0:
                        context += f", 缓存效率 {agent['cache_efficiency']}%"
                    if agent['error_count'] > 0:
                        context += f", {agent['error_count']} 错误"
                    context += "\\n"

        if "context_management" in summary:
            cm = summary["context_management"]
            if cm.get("issues_found", 0) > 0:
                context += "\\n## ⚠️ 上下文管理警告\\n"
                context += f"检测到 {cm['issues_found']} 个子智能体上下文管理问题：\\n"
                for warning in cm.get("warnings", []):
                    context += f"- {warning.get('warning', '')}\\n"
            else:
                for summary_line in cm.get("summary", []):
                    context += f"\\n{summary_line}"

        if include_detailed_steps and steps:
            context += "\n## 详细步骤信息\n"
            context += "以下是轨迹中的详细信息，用于回答关于特定步骤的问题：\n\n"

            steps_to_include = []

            for step in steps[:50]:
                step_info = f"步骤 #{step.get('index', '?')}\n"
                if 'role' in step:
                    step_info += f"角色: {step['role']}\n"
                if 'text_preview' in step and step['text_preview']:
                    step_info += f"内容概要: {step['text_preview'][:200]}...\n"
                if 'duration' in step and step['duration']:
                    step_info += f"时长: {step['duration']:.2f}s\n"
                if 'tool_calls' in step and step['tool_calls']:
                    tool_names = [tc.get('tool_name', 'unknown') for tc in step['tool_calls'][:3]]
                    step_info += f"工具调用: {', '.join(tool_names)}\n"
                if 'errors' in step and step['errors']:
                    step_info += f"错误: {len(step['errors'])} 个错误\n"
                if 'tokens' in step and step['tokens'].get('total', 0) > 0:
                    step_info += f"Tokens: {step['tokens']['total']}\n"
                steps_to_include.append(step_info)

            for i in range(50, len(steps), 10):
                if i < len(steps) - 50:
                    step = steps[i]
                    step_info = f"步骤 #{step.get('index', '?')}\n"
                    if 'role' in step:
                        step_info += f"角色: {step['role']}\n"
                    if 'text_preview' in step and step['text_preview']:
                        step_info += f"内容概要: {step['text_preview'][:200]}...\n"
                    if 'duration' in step and step['duration']:
                        step_info += f"时长: {step['duration']:.2f}s\n"
                    if 'tool_calls' in step and step['tool_calls']:
                        tool_names = [tc.get('tool_name', 'unknown') for tc in step['tool_calls'][:3]]
                        step_info += f"工具调用: {', '.join(tool_names)}\n"
                    if 'errors' in step and step['errors']:
                        step_info += f"错误: {len(step['errors'])} 个错误\n"
                    if 'tokens' in step and step['tokens'].get('total', 0) > 0:
                        step_info += f"Tokens: {step['tokens']['total']}\n"
                    steps_to_include.append(step_info)

            for step in steps[-50:]:
                step_info = f"步骤 #{step.get('index', '?')}\n"
                if 'role' in step:
                    step_info += f"角色: {step['role']}\n"
                if 'text_preview' in step and step['text_preview']:
                    step_info += f"内容概要: {step['text_preview'][:200]}...\n"
                if 'duration' in step and step['duration']:
                    step_info += f"时长: {step['duration']:.2f}s\n"
                if 'tool_calls' in step and step['tool_calls']:
                    tool_names = [tc.get('tool_name', 'unknown') for tc in step['tool_calls'][:3]]
                    step_info += f"工具调用: {', '.join(tool_names)}\n"
                if 'errors' in step and step['errors']:
                    step_info += f"错误: {len(step['errors'])} 个错误\n"
                if 'tokens' in step and step['tokens'].get('total', 0) > 0:
                    step_info += f"Tokens: {step['tokens']['total']}\n"
                steps_to_include.append(step_info)

            context += "\\n".join(steps_to_include)

        return context

    def _build_analysis_prompt(
        self,
        summary: dict[str, Any],
        steps: list[dict] | None = None,
    ) -> tuple[str, str]:
        """Build analysis prompt and system message.

        Args:
            summary: Trajectory summary
            steps: Optional detailed steps for context

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = f"""You are an expert at analyzing OpenCode trajectories. Analyze the trajectory to help developers identify performance bottlenecks, error patterns, and provide specific optimization suggestions.

Output the analysis in the following format:

Based on this trace log, I will help you analyze the main performance bottlenecks and issues:

Main Performance Bottlenecks
1. Step X - [Phase Name] ([Severity Description])
Time taken: [Specific time]
Issue: [Issue description]
Result: [Execution result]
Analysis: [Detailed analysis, including why this step is slow or fails]
2. Step Y - [Phase Name]
...

Error Pattern Analysis
[Identified major error patterns, grouped by type, and described the steps where they occur]

Root Cause
[In-depth analysis based on metrics and trace data, explaining the root cause rather than just the symptoms]

Suggested Optimization Directions
1. [Optimization Direction 1]: [Specific suggestion]
2. [Optimization Direction 2]: [Specific suggestion]
...
Key requirements:
- Identify different pipeline stages (depending on the specific task, these could be: planning, code generation, code transformation, optimization, compilation/build, testing, error repair, verification, etc.)
- Use metrics data (tool_time_share, tool_wait_pct, P95_tool_duration, max_tool_duration, etc.) to support your analysis
- For each bottleneck, specify the exact step number, duration, and why this approach is inefficient

{self._get_language_guidelines()}

{self._get_html_usage_guidelines()} for emphasis when helpful, but don't force rigid structure. Answer naturally.
"""

        trajectory_context = self._build_trajectory_context(summary, steps, include_detailed_steps=True)

        user_prompt = f"{trajectory_context}\\n\\nPlease analyze this trajectory and provide insights in Markdown format."

        return system_prompt, user_prompt

    def chat(
        self,
        steps: list[dict],
        metrics: dict,
        patterns: dict | None = None,
        raw: dict | None = None,
        diagnostics: dict | None = None,
        agent_summaries: list[dict] | None = None,
        conversation_history: list[tuple[str, str]] | None = None,
        user_question: str = "",
    ) -> dict[str, Any]:
        """Chat about a trajectory using LLM with conversation support.

        Args:
            steps: Parsed step list
            metrics: Computed metrics
            patterns: Detected patterns
            raw: Raw trajectory data
            diagnostics: Diagnostic metrics including file interactions, failure chains, etc.
            agent_summaries: Per-agent summary statistics
            conversation_history: Previous chat messages as [(user, assistant), ...]
            user_question: Current question from user

        Returns:
            Chat response with content and metadata
        """
        summary = self._extract_trajectory_summary(steps, metrics, patterns, raw, diagnostics, agent_summaries)

        system_prompt = f"""You are an expert at analyzing LLM agent trajectories. You help users understand trajectory execution, patterns, and performance metrics.

Your responses should be:
- Natural and conversational (not rigid executive summaries)
- Specific and data-driven when needed
- Clear and direct in answering questions
- Focused on what the user is actually asking about

When users ask about specific steps (like "what\'s happening at step 50"), look at the detailed step information provided in the trajectory context. Each step includes role, content summary, duration, tool calls, errors, and token counts.

{self._get_language_guidelines()}

{self._get_html_usage_guidelines()} for emphasis when helpful, but don't force rigid structure. Answer naturally."""

        # Build trajectory context for reference
        trajectory_context = self._build_trajectory_context(summary, steps, include_detailed_steps=True)

        try:
            messages = []
            
            # Add conversation history to messages
            if conversation_history:
                # Handle both tuple and dict formats safely
                for item in conversation_history[-5:]:  # Last 5 exchanges for context
                    try:
                        if isinstance(item, dict):
                            # Already in message format
                            if "content" in item and "role" in item:
                                messages.append(item)
                        elif isinstance(item, (tuple, list)) and len(item) >= 2:
                            # Expect (user_msg, assistant_msg) format
                            user_msg = item[0] if isinstance(item[0], str) else ""
                            assistant_msg = item[1] if isinstance(item[1], str) else ""
                            if user_msg:
                                messages.append({"role": "user", "content": user_msg})
                            if assistant_msg:
                                messages.append({"role": "assistant", "content": assistant_msg})
                        elif isinstance(item, str):
                            # Simple string, treat as user message
                            messages.append({"role": "user", "content": item})
                    except (IndexError, TypeError, ValueError):
                        # Skip problematic items
                        continue
            
            # Build the current message with context
            current_message = f"{trajectory_context}\n\n"
            if user_question:
                current_message += f"用户问题: {user_question}"
            
            messages.append({"role": "user", "content": current_message})

            response = self.client.chat(
                messages=messages,
                system_prompt=system_prompt,
            )

            return {
                "success": True,
                "content": response,
                "metadata": summary,
            }

        except Exception as e:
            logger.error("Chat failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "content": f"<div class='ai-analysis-error'>"
                          f"<p><strong>Chat unavailable:</strong> {html.escape(str(e))}</p>"
                          f"<p>Ensure your LLM service is running and configured correctly.</p>"
                          f"</div>",
                "metadata": summary,
            }

    def analyze(
        self,
        steps: list[dict],
        metrics: dict,
        patterns: dict | None = None,
        raw: dict | None = None,
        diagnostics: dict | None = None,
        agent_summaries: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Analyze a trajectory using LLM.

        Args:
            steps: Parsed step list
            metrics: Computed metrics
            patterns: Detected patterns
            raw: Raw trajectory data
            diagnostics: Diagnostic metrics including file interactions, failure chains, etc.
            agent_summaries: Per-agent summary statistics

        Returns:
            Analysis result with HTML content and metadata
        """
        summary = self._extract_trajectory_summary(steps, metrics, patterns, raw, diagnostics, agent_summaries)

        system_prompt, user_prompt = self._build_analysis_prompt(summary, steps)

        try:
            response = self.client.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )

            return {
                "success": True,
                "content": response,
                "metadata": summary,
            }

        except Exception as e:
            logger.error("Analysis failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "content": f"<div class='ai-analysis-error'>"
                          f"<p><strong>Analysis unavailable:</strong> {html.escape(str(e))}</p>"
                          f"<p>Ensure your LLM service is running and configured correctly.</p>"
                          f"</div>",
                "metadata": summary,
            }
