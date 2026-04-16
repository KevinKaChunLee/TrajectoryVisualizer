"""Trajectory quality scoring engine: dimension computation, composite score, and export."""

from __future__ import annotations

from typing import Any

from .scoring_config import (
    DEFAULT_WEIGHTS,
    VERDICT_THRESHOLDS,
    get_profile,
    merge_profile,
)


# ---------------------------------------------------------------------------
# Core scoring primitives
# ---------------------------------------------------------------------------

def _score_metric(
    value: float | int | None,
    good: float,
    warn: float,
    bad: float,
    invert: bool = False,
) -> float | None:
    """Map a single metric value to a 0–100 score via piecewise-linear interpolation.

    For "lower is better" metrics (invert=False):
      value <= good → 100
      good < value <= warn → linear 100→50
      warn < value <= bad → linear 50→0
      value > bad → 0

    For "higher is better" metrics (invert=True):
      value >= good → 100
      warn <= value < good → linear 50→100
      bad <= value < warn → linear 0→50
      value < bad → 0

    Returns None if value is None.
    """
    if value is None:
        return None

    if invert:
        # Higher is better: flip the axis
        if value >= good:
            return 100.0
        if value >= warn:
            return 50.0 + 50.0 * (value - warn) / (good - warn) if good != warn else 50.0
        if value >= bad:
            return 50.0 * (value - bad) / (warn - bad) if warn != bad else 0.0
        return 0.0
    else:
        # Lower is better
        if value <= good:
            return 100.0
        if value <= warn:
            return 100.0 - 50.0 * (value - good) / (warn - good) if warn != good else 50.0
        if value <= bad:
            return 50.0 - 50.0 * (value - warn) / (bad - warn) if bad != warn else 0.0
        return 0.0


def _score_dimension(
    metrics: dict[str, float | int | None],
    thresholds: dict[str, dict[str, Any]],
    aggregation: str = "weighted_min",
) -> tuple[float | None, dict[str, float | None]]:
    """Compute a dimension sub-score from individual metric scores.

    Aggregation strategies:
    - "min": pure minimum (original behavior, penalizes single outliers heavily)
    - "weighted_min": blend of 60% minimum + 40% mean (default, balances
      sensitivity to outliers with overall performance)

    Returns (dimension_score, {metric_name: metric_score}).
    If all metrics are None, returns (None, {...}).
    """
    metric_scores: dict[str, float | None] = {}

    for metric_name, threshold in thresholds.items():
        value = metrics.get(metric_name)
        score = _score_metric(
            value,
            good=threshold["good"],
            warn=threshold["warn"],
            bad=threshold["bad"],
            invert=threshold.get("invert", False),
        )
        metric_scores[metric_name] = score

    valid_scores = [s for s in metric_scores.values() if s is not None]
    if not valid_scores:
        return None, metric_scores

    if aggregation == "min":
        dimension_score = round(min(valid_scores), 1)
    else:
        # weighted_min: 60% min + 40% mean
        score_min = min(valid_scores)
        score_mean = sum(valid_scores) / len(valid_scores)
        dimension_score = round(0.6 * score_min + 0.4 * score_mean, 1)

    return dimension_score, metric_scores


def _classify_verdict(score: float | None) -> str:
    """Classify a score into good/warn/bad verdict."""
    if score is None:
        return "n/a"
    if score >= VERDICT_THRESHOLDS["good"]:
        return "good"
    if score >= VERDICT_THRESHOLDS["warn"]:
        return "warn"
    return "bad"


# ---------------------------------------------------------------------------
# Per-dimension scorers
# ---------------------------------------------------------------------------

def compute_targeting_score(
    file_targeting: dict | None,
    thresholds: dict[str, dict],
) -> dict:
    """Compute the targeting dimension score."""
    if not file_targeting:
        return {"score": None, "verdict": "n/a", "metrics": {}}

    metrics = {
        "avg_steps_to_first_touch": file_targeting.get("avg_steps_to_first_touch"),
        "exploration_ratio": file_targeting.get("exploration_ratio"),
    }
    score, metric_scores = _score_dimension(metrics, thresholds)
    return {"score": score, "verdict": _classify_verdict(score), "metrics": metric_scores}


def compute_error_resilience_score(
    chain_metrics: dict | None,
    error_clusters: list[dict] | None,
    tool_success_rate: float | None,
    thresholds: dict[str, dict],
) -> dict:
    """Compute the error resilience dimension score."""
    metrics = {
        "failure_chain_count": chain_metrics.get("total_chains") if chain_metrics else 0,
        "longest_chain": chain_metrics.get("longest_chain") if chain_metrics else 0,
        "error_cluster_count": len(error_clusters) if error_clusters else 0,
        "tool_success_rate": tool_success_rate,
    }
    score, metric_scores = _score_dimension(metrics, thresholds)
    return {"score": score, "verdict": _classify_verdict(score), "metrics": metric_scores}


def compute_execution_efficiency_score(
    bottleneck_explanations: list[dict] | None,
    metrics: dict | None,
    thresholds: dict[str, dict],
) -> dict:
    """Compute the execution efficiency dimension score."""
    # Average inference % across hotspot steps
    avg_inference_pct = None
    if bottleneck_explanations:
        inference_pcts = [
            e["decomposition"]["inference_pct"]
            for e in bottleneck_explanations
            if e.get("decomposition")
        ]
        if inference_pcts:
            avg_inference_pct = sum(inference_pcts) / len(inference_pcts)

    # Tool retry count (from insights / metrics)
    tool_retry_count = None
    if metrics:
        # Approximate: tool failures as proxy for retries
        tool_retry_count = metrics.get("tool_fail", 0)

    # Steps per patch line
    steps_per_patch = None
    if metrics:
        total_steps = metrics.get("total_steps", 0)
        patch_lines = metrics.get("patch_lines", 0)
        if patch_lines and patch_lines > 0:
            steps_per_patch = total_steps / patch_lines

    dim_metrics = {
        "avg_hotspot_inference_pct": avg_inference_pct,
        "tool_retry_count": tool_retry_count,
        "steps_per_patch_line": steps_per_patch,
    }
    score, metric_scores = _score_dimension(dim_metrics, thresholds)
    return {"score": score, "verdict": _classify_verdict(score), "metrics": metric_scores}


def compute_cost_efficiency_score(
    metrics: dict | None,
    thresholds: dict[str, dict],
) -> dict:
    """Compute the cost efficiency dimension score."""
    if not metrics:
        return {"score": None, "verdict": "n/a", "metrics": {}}

    dim_metrics = {
        "non_cache_ratio": metrics.get("non_cache_ratio"),
        "tokens_per_patch_line": metrics.get("tokens_per_patch_line"),
        "cache_utilization": metrics.get("cache_utilization_ratio"),
    }
    score, metric_scores = _score_dimension(dim_metrics, thresholds)
    return {"score": score, "verdict": _classify_verdict(score), "metrics": metric_scores}


# ---------------------------------------------------------------------------
# Composite score orchestrator
# ---------------------------------------------------------------------------

def compute_trajectory_score(
    metrics: dict,
    diagnostics: dict,
    profile: str | dict = "default",
) -> dict:
    """Compute the full trajectory quality score.

    Args:
        metrics: Aggregate metrics from compute_metrics().
        diagnostics: Dict with keys: file_targeting, chain_metrics,
                     error_clusters, bottleneck_explanations.
        profile: Profile name (str) or custom profile dict.

    Returns:
        Score result dict with composite_score, composite_verdict,
        dimensions, and profile name.
    """
    # Resolve profile
    if isinstance(profile, str):
        prof = get_profile(profile)
        profile_name = profile
    else:
        prof = merge_profile(get_profile("default"), profile)
        profile_name = "custom"

    thresholds = prof["thresholds"]
    weights = prof.get("weights", DEFAULT_WEIGHTS)

    # Compute per-dimension scores
    dimensions = {
        "targeting": compute_targeting_score(
            diagnostics.get("file_targeting"),
            thresholds.get("targeting", {}),
        ),
        "error_resilience": compute_error_resilience_score(
            diagnostics.get("chain_metrics"),
            diagnostics.get("error_clusters"),
            metrics.get("tool_success_rate"),
            thresholds.get("error_resilience", {}),
        ),
        "execution_efficiency": compute_execution_efficiency_score(
            diagnostics.get("bottleneck_explanations"),
            metrics,
            thresholds.get("execution_efficiency", {}),
        ),
        "cost_efficiency": compute_cost_efficiency_score(
            metrics,
            thresholds.get("cost_efficiency", {}),
        ),
    }

    # Composite: weighted average of available dimensions
    available = {
        name: dim for name, dim in dimensions.items()
        if dim["score"] is not None
    }

    if not available:
        return {
            "composite_score": None,
            "composite_verdict": "n/a",
            "dimensions": dimensions,
            "profile": profile_name,
            "judge": None,
        }

    # Re-normalize weights
    total_weight = sum(weights.get(name, 0) for name in available)
    if total_weight <= 0:
        total_weight = len(available)
        norm_weights = {name: 1.0 / total_weight for name in available}
    else:
        norm_weights = {name: weights.get(name, 0) / total_weight for name in available}

    composite = sum(
        dim["score"] * norm_weights[name]
        for name, dim in available.items()
    )
    composite = round(composite, 1)

    return {
        "composite_score": composite,
        "composite_verdict": _classify_verdict(composite),
        "dimensions": dimensions,
        "profile": profile_name,
        "judge": None,
    }
