"""Scoring configuration: thresholds, weights, and profiles for trajectory quality scoring."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# ---------------------------------------------------------------------------
# Default thresholds per metric
# ---------------------------------------------------------------------------
# Each metric defines (good, warn, bad) breakpoints.
# For "lower is better" metrics (exploration_ratio, chain_count, etc.):
#   score=100 when value <= good, score=0 when value >= bad
# For "higher is better" metrics (tool_success_rate, cache_utilization):
#   the `invert` flag is set and the axis is reversed.

DEFAULT_THRESHOLDS: dict[str, dict[str, Any]] = {
    # -- Targeting dimension --
    "targeting": {
        "avg_steps_to_first_touch": {"good": 3.0, "warn": 8.0, "bad": 15.0},
        "exploration_ratio": {"good": 2.0, "warn": 5.0, "bad": 10.0},
    },
    # -- Error Resilience dimension --
    "error_resilience": {
        "failure_chain_count": {"good": 0, "warn": 2, "bad": 5},
        "longest_chain": {"good": 1, "warn": 3, "bad": 6},
        "error_cluster_count": {"good": 0, "warn": 2, "bad": 5},
        "tool_success_rate": {"good": 95.0, "warn": 80.0, "bad": 60.0, "invert": True},
    },
    # -- Execution Efficiency dimension --
    "execution_efficiency": {
        "avg_hotspot_inference_pct": {"good": 40.0, "warn": 70.0, "bad": 90.0},
        "tool_retry_count": {"good": 2, "warn": 6, "bad": 12},
        "steps_per_patch_line": {"good": 0.5, "warn": 2.0, "bad": 5.0},
    },
    # -- Cost Efficiency dimension --
    "cost_efficiency": {
        "non_cache_ratio": {"good": 20.0, "warn": 50.0, "bad": 80.0},
        "tokens_per_patch_line": {"good": 500, "warn": 2000, "bad": 5000},
        "cache_utilization": {"good": 0.6, "warn": 0.3, "bad": 0.1, "invert": True},
    },
}

# Default dimension weights for composite score
DEFAULT_WEIGHTS: dict[str, float] = {
    "targeting": 0.30,
    "error_resilience": 0.25,
    "execution_efficiency": 0.25,
    "cost_efficiency": 0.20,
}

# Verdict thresholds for composite and per-dimension scores
VERDICT_THRESHOLDS = {"good": 70, "warn": 40}

# Default uncertain band for LLM judge invocation
DEFAULT_UNCERTAIN_BAND = (35, 65)


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

PROFILES: dict[str, dict[str, Any]] = {
    "default": {
        "thresholds": DEFAULT_THRESHOLDS,
        "weights": DEFAULT_WEIGHTS,
        "uncertain_band": DEFAULT_UNCERTAIN_BAND,
    },
    "strict": {
        "thresholds": {
            "targeting": {
                "avg_steps_to_first_touch": {"good": 2.0, "warn": 5.0, "bad": 10.0},
                "exploration_ratio": {"good": 1.5, "warn": 3.0, "bad": 6.0},
            },
            "error_resilience": {
                "failure_chain_count": {"good": 0, "warn": 1, "bad": 3},
                "longest_chain": {"good": 1, "warn": 2, "bad": 4},
                "error_cluster_count": {"good": 0, "warn": 1, "bad": 3},
                "tool_success_rate": {"good": 98.0, "warn": 90.0, "bad": 75.0, "invert": True},
            },
            "execution_efficiency": {
                "avg_hotspot_inference_pct": {"good": 30.0, "warn": 55.0, "bad": 80.0},
                "tool_retry_count": {"good": 1, "warn": 3, "bad": 8},
                "steps_per_patch_line": {"good": 0.3, "warn": 1.0, "bad": 3.0},
            },
            "cost_efficiency": {
                "non_cache_ratio": {"good": 15.0, "warn": 35.0, "bad": 60.0},
                "tokens_per_patch_line": {"good": 300, "warn": 1000, "bad": 3000},
                "cache_utilization": {"good": 0.7, "warn": 0.4, "bad": 0.2, "invert": True},
            },
        },
        "weights": DEFAULT_WEIGHTS,
        "uncertain_band": (40, 60),
    },
    "lenient": {
        "thresholds": {
            "targeting": {
                "avg_steps_to_first_touch": {"good": 5.0, "warn": 12.0, "bad": 25.0},
                "exploration_ratio": {"good": 4.0, "warn": 10.0, "bad": 20.0},
            },
            "error_resilience": {
                "failure_chain_count": {"good": 1, "warn": 4, "bad": 8},
                "longest_chain": {"good": 2, "warn": 5, "bad": 10},
                "error_cluster_count": {"good": 1, "warn": 4, "bad": 8},
                "tool_success_rate": {"good": 90.0, "warn": 70.0, "bad": 50.0, "invert": True},
            },
            "execution_efficiency": {
                "avg_hotspot_inference_pct": {"good": 50.0, "warn": 80.0, "bad": 95.0},
                "tool_retry_count": {"good": 4, "warn": 10, "bad": 20},
                "steps_per_patch_line": {"good": 1.0, "warn": 4.0, "bad": 10.0},
            },
            "cost_efficiency": {
                "non_cache_ratio": {"good": 30.0, "warn": 65.0, "bad": 90.0},
                "tokens_per_patch_line": {"good": 1000, "warn": 4000, "bad": 10000},
                "cache_utilization": {"good": 0.4, "warn": 0.15, "bad": 0.05, "invert": True},
            },
        },
        "weights": DEFAULT_WEIGHTS,
        "uncertain_band": (25, 70),
    },
}


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------

def get_profile(name: str = "default") -> dict[str, Any]:
    """Return a deep copy of a built-in profile by name.

    Raises KeyError if the profile name is not recognized.
    """
    if name not in PROFILES:
        raise KeyError(f"Unknown scoring profile: {name!r}. Available: {list(PROFILES.keys())}")
    return deepcopy(PROFILES[name])


def merge_profile(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge overrides into a base profile.

    Override keys replace base keys at the leaf level. This allows
    users to override specific thresholds or weights without
    specifying the entire profile.
    """
    result = deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_profile(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
