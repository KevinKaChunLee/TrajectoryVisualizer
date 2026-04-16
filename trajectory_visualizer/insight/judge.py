"""Optional LLM-as-judge for trajectory quality evaluation."""

from __future__ import annotations

import json
import re
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_JUDGE_TEMPLATE = """\
You are evaluating the quality of an AI agent's tool-calling trajectory.

## Trajectory Scores (deterministic analysis)

Composite Score: {composite_score}/100 ({composite_verdict})

### Dimension Breakdown
{dimension_summary}

### Failure Chains
{failure_chain_summary}

### Performance Bottlenecks
{bottleneck_summary}

## Your Task

Given the metrics above, evaluate whether this trajectory's quality is acceptable.
Consider:
- Are the error patterns (if any) justified or indicative of a flawed approach?
- Is the exploration overhead reasonable for the task complexity?
- Are the bottlenecks caused by necessary computation or unnecessary retries?

Respond in **exactly** this JSON format:
```json
{{
  "verdict": "acceptable" | "poor" | "uncertain",
  "reasoning": "1-3 sentences explaining your assessment",
  "flagged_steps": [list of step indices that are most problematic, or empty list]
}}
```
"""


def build_judge_prompt(
    score_result: dict,
    failure_chains: list[dict] | None = None,
    bottleneck_explanations: list[dict] | None = None,
) -> str:
    """Construct a focused LLM judge prompt with metric context."""
    # Dimension summary
    dim_lines = []
    for name, dim in score_result.get("dimensions", {}).items():
        score = dim.get("score")
        verdict = dim.get("verdict", "n/a")
        score_str = f"{score:.1f}" if score is not None else "N/A"
        metrics_str = ", ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in (dim.get("metrics") or {}).items()
            if v is not None
        )
        dim_lines.append(f"- {name}: {score_str}/100 ({verdict}) — {metrics_str}")
    dimension_summary = "\n".join(dim_lines) if dim_lines else "No dimension data available."

    # Failure chain summary
    chain_lines = []
    if failure_chains:
        for c in failure_chains[:5]:
            start, end = c.get("start", "?"), c.get("end", "?")
            n = len(c.get("steps", []))
            chain_lines.append(f"- Steps {start}–{end} ({n} steps)")
    failure_chain_summary = "\n".join(chain_lines) if chain_lines else "No failure chains detected."

    # Bottleneck summary
    bn_lines = []
    if bottleneck_explanations:
        for e in bottleneck_explanations[:5]:
            bn_lines.append(f"- {e.get('explanation', 'N/A')}")
    bottleneck_summary = "\n".join(bn_lines) if bn_lines else "No significant bottlenecks."

    composite = score_result.get("composite_score")
    composite_str = f"{composite:.1f}" if composite is not None else "N/A"

    return _JUDGE_TEMPLATE.format(
        composite_score=composite_str,
        composite_verdict=score_result.get("composite_verdict", "n/a"),
        dimension_summary=dimension_summary,
        failure_chain_summary=failure_chain_summary,
        bottleneck_summary=bottleneck_summary,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_VALID_VERDICTS = {"acceptable", "poor", "uncertain"}


def _parse_judge_response(raw_response: str) -> dict:
    """Extract structured judge result from LLM output.

    Returns {verdict, reasoning, flagged_steps}.
    Falls back to uncertain if parsing fails.
    """
    fallback = {
        "verdict": "uncertain",
        "reasoning": "Failed to parse judge response",
        "flagged_steps": [],
    }

    if not raw_response:
        return fallback

    # Try to extract JSON block
    json_str = raw_response
    m = _JSON_BLOCK_RE.search(raw_response)
    if m:
        json_str = m.group(1)
    else:
        # Try to find bare JSON object
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start >= 0 and end > start:
            json_str = raw_response[start:end + 1]

    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return fallback

    if not isinstance(data, dict):
        return fallback

    verdict = data.get("verdict", "uncertain")
    if verdict not in _VALID_VERDICTS:
        verdict = "uncertain"

    reasoning = str(data.get("reasoning", ""))
    flagged = data.get("flagged_steps", [])
    if not isinstance(flagged, list):
        flagged = []
    flagged = [int(s) for s in flagged if isinstance(s, (int, float))]

    return {
        "verdict": verdict,
        "reasoning": reasoning,
        "flagged_steps": flagged,
    }


# ---------------------------------------------------------------------------
# Default LLM backend
# ---------------------------------------------------------------------------

_OPENAI_CLIENT = None
_JUDGE_MODEL = "gpt-4o-mini"


def _default_openai_callable(prompt: str) -> str:
    """Default LLM backend using OpenAI chat completion (lazy import).

    Reuses a single client instance across calls to avoid per-request
    connection overhead.
    """
    global _OPENAI_CLIENT
    try:
        import openai
    except ImportError:
        raise ImportError(
            "The 'openai' package is required for the LLM judge. "
            "Install it with: pip install openai"
        )

    if _OPENAI_CLIENT is None:
        import os
        _OPENAI_CLIENT = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = _OPENAI_CLIENT.chat.completions.create(
        model=_JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Judge orchestrator
# ---------------------------------------------------------------------------

def invoke_judge(
    score_result: dict,
    failure_chains: list[dict] | None = None,
    bottleneck_explanations: list[dict] | None = None,
    llm_callable: Callable[[str], str] | None = None,
    uncertain_band: tuple[float, float] = (35, 65),
) -> dict | None:
    """Invoke the LLM judge if the composite score is in the uncertain band.

    Returns judge result dict or None if the judge was not invoked.
    """
    composite = score_result.get("composite_score")

    # Skip if score is clearly good or clearly bad
    if composite is None:
        return None
    if composite < uncertain_band[0] or composite > uncertain_band[1]:
        return None

    # Build prompt
    prompt = build_judge_prompt(score_result, failure_chains, bottleneck_explanations)

    # Select callable
    callable_fn = llm_callable or _default_openai_callable

    try:
        raw_response = callable_fn(prompt)
    except Exception as e:
        return {
            "verdict": "uncertain",
            "reasoning": f"LLM call failed: {e}",
            "flagged_steps": [],
        }

    return _parse_judge_response(raw_response)
