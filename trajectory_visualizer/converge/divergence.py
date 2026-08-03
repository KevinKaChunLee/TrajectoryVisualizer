"""Divergence pattern classification, confidence scoring, and cost attribution."""

from __future__ import annotations

from .canonical import CanonicalAction, compute_action_cost, DEFAULT_TOKEN_RATE


# ---------------------------------------------------------------------------
# Default confidence scores per pattern type
# ---------------------------------------------------------------------------

DEFAULT_PATTERN_CONFIDENCE: dict[str, float] = {
    "reverted_and_rewritten": 0.7,
    "iterative_refinement": 0.4,
    "broad_exploration": 0.6,
    "error_recovery_overhead": 0.8,
    "redundant_search": 0.7,
    "ordering_inefficiency": 0.5,
    "dead_end_branch": 0.6,
    "premature_validation": 0.5,
}

# Step distance threshold: rewrites within this many steps are instability
_CLOSE_REWRITE_STEPS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_subtree(path: str, depth: int = 2) -> str:
    """Extract the first N path segments as a directory subtree identifier."""
    parts = path.replace("\\", "/").strip("/").split("/")
    return "/".join(parts[:depth]) if len(parts) >= depth else "/".join(parts)


def _action_id(action: CanonicalAction, index_in_extras: int) -> tuple[int, str, int]:
    """Unique identifier for a single action instance.

    Uses (step_index, target, position_in_extras) so that sibling actions on the
    same step (e.g., two FILE_WRITEs) are not suppressed by a shared step_index.
    """
    return (action.step_index, action.target, index_in_extras)


def _make_pattern(
    ptype: str,
    evidence: list[str],
    steps: list[int],
    tokens: int | float,
    confidence: float | None = None,
    parent_type: str | None = None,
) -> dict:
    """Create a pattern dict with standard fields."""
    conf = confidence if confidence is not None else DEFAULT_PATTERN_CONFIDENCE.get(ptype, 0.5)
    result = {
        "type": ptype,
        "evidence": evidence,
        "steps": steps,
        "estimated_extra_cost": {"steps": len(steps), "tokens": int(tokens)},
        "confidence": round(conf, 2),
        "evidence_level": "single_pair_hypothesis",
    }
    if parent_type:
        result["parent_type"] = parent_type
    return result


# ---------------------------------------------------------------------------
# Pattern classification
# ---------------------------------------------------------------------------

def classify_divergences(
    extra_actions: list[CanonicalAction],
    matched_actions: list[CanonicalAction],
    all_compared_actions: list[CanonicalAction],
    matched_pairs: list[tuple[int, int]] | None = None,
    anchor_files: set[str] | None = None,
) -> list[dict]:
    """Classify extra (unmatched) compared actions into divergence patterns.

    Returns list of pattern dicts with type, evidence, steps, estimated_extra_cost,
    confidence, and evidence_level.
    """
    patterns: list[dict] = []
    classified: set[tuple[int, str, int]] = set()

    def _is_classified(action: CanonicalAction, idx: int) -> bool:
        return _action_id(action, idx) in classified

    def _mark_classified(action: CanonicalAction, idx: int) -> None:
        classified.add(_action_id(action, idx))

    has_writes = any(a.action_type == "FILE_WRITE" for a in all_compared_actions)

    # Build anchor subtrees for broad_exploration check
    anchor_subtrees: set[str] | None = None
    if anchor_files:
        anchor_subtrees = {_get_subtree(f) for f in anchor_files}

    # ── write_retry split: reverted_and_rewritten vs iterative_refinement ──
    write_targets_seen: dict[str, list[tuple[int, CanonicalAction]]] = {}
    for i, a in enumerate(extra_actions):
        if a.action_type == "FILE_WRITE" and a.effect_label == "reverted":
            write_targets_seen.setdefault(a.target, []).append((i, a))

    for target, reverted_entries in write_targets_seen.items():
        all_writes_to_target = [
            a for a in all_compared_actions
            if a.action_type == "FILE_WRITE" and a.target == target
        ]
        for extra_idx, ra in reverted_entries:
            if _is_classified(ra, extra_idx):
                continue
            replacement = None
            for w in all_writes_to_target:
                if w.step_index > ra.step_index:
                    replacement = w
                    break

            step_distance = (replacement.step_index - ra.step_index) if replacement else 999

            if step_distance <= _CLOSE_REWRITE_STEPS:
                ptype = "reverted_and_rewritten"
                conf = min(0.9, 0.7 + ((_CLOSE_REWRITE_STEPS - step_distance) * 0.1))
            else:
                ptype = "iterative_refinement"
                conf = max(0.3, 0.5 - (step_distance - _CLOSE_REWRITE_STEPS) * 0.02)

            patterns.append(_make_pattern(
                ptype=ptype,
                evidence=[f"FILE_WRITE({target}) [reverted] → rewrite at +{step_distance} steps"],
                steps=[ra.step_index],
                tokens=ra.cost.token_share,
                confidence=conf,
                parent_type="write_retry",
            ))
            _mark_classified(ra, extra_idx)

    # ── error_recovery_overhead ──
    for i, a in enumerate(extra_actions):
        if _is_classified(a, i):
            continue
        if a.effect_label == "failed":
            for j, b in enumerate(extra_actions[i + 1:], start=i + 1):
                if b.action_type == a.action_type and b.target == a.target:
                    patterns.append(_make_pattern(
                        ptype="error_recovery_overhead",
                        evidence=[f"{a.action_type}({a.target}) [failed]", f"{b.action_type}({b.target})"],
                        steps=[a.step_index, b.step_index],
                        tokens=a.cost.token_share + b.cost.token_share,
                        confidence=0.8,
                    ))
                    _mark_classified(a, i)
                    _mark_classified(b, j)
                    break

    # ── premature_validation ──
    if has_writes:
        first_write_step = min(
            (a.step_index for a in all_compared_actions if a.action_type == "FILE_WRITE"),
            default=float("inf"),
        )
        from .milestones import DEFAULT_VALIDATION_PATTERNS
        for i, a in enumerate(extra_actions):
            if _is_classified(a, i):
                continue
            if a.action_type == "COMMAND" and a.step_index < first_write_step:
                base_cmd = a.target.split("/")[-1] if "/" in a.target else a.target
                if any(vp in base_cmd for vp in DEFAULT_VALIDATION_PATTERNS):
                    patterns.append(_make_pattern(
                        ptype="premature_validation",
                        evidence=[f"COMMAND({a.target}) before first write"],
                        steps=[a.step_index],
                        tokens=a.cost.token_share,
                    ))
                    _mark_classified(a, i)

    # ── redundant_search ──
    matched_search_targets: set[str] = set()
    for a in matched_actions:
        if a.action_type == "SEARCH":
            matched_search_targets.add(a.target)
    extra_search_counts: dict[str, int] = {}
    for i, a in enumerate(extra_actions):
        if _is_classified(a, i):
            continue
        if a.action_type == "SEARCH":
            extra_search_counts[a.target] = extra_search_counts.get(a.target, 0) + 1
            if a.target in matched_search_targets or extra_search_counts[a.target] > 1:
                patterns.append(_make_pattern(
                    ptype="redundant_search",
                    evidence=[f"SEARCH({a.target}) repeated"],
                    steps=[a.step_index],
                    tokens=a.cost.token_share,
                ))
                _mark_classified(a, i)

    # ── broad_exploration (with subtree check when anchor available) ──
    # Phase labels refine confidence: debug/debug_root_cause reads are
    # more likely targeted investigation than generic browsing.
    _UNKNOWN_LABELS = {"unknown"}
    _INVESTIGATION_PHASES = {"debug"}
    _INVESTIGATION_ACTIONS = {"debug_root_cause", "debug_hypothesis_test", "code_reading"}
    for i, a in enumerate(extra_actions):
        if _is_classified(a, i):
            continue
        if a.action_type == "FILE_READ" and a.effect_label in _UNKNOWN_LABELS:
            is_dead_end = not any(
                b.target == a.target and b.step_index > a.step_index
                for b in matched_actions
            )
            if not is_dead_end:
                continue

            if anchor_subtrees:
                file_subtree = _get_subtree(a.target)
                if file_subtree in anchor_subtrees:
                    continue
                conf = 0.7
            else:
                conf = 0.5

            # Phase-label adjustment: lower confidence when the step is
            # labeled as targeted investigation (debug, root cause analysis)
            if a.phase_label in _INVESTIGATION_PHASES or a.action_label in _INVESTIGATION_ACTIONS:
                conf = max(0.3, conf - 0.2)

            patterns.append(_make_pattern(
                ptype="broad_exploration",
                evidence=[f"FILE_READ({a.target})"
                          + (f" [phase={a.phase_label}]" if a.phase_label else "")],
                steps=[a.step_index],
                tokens=a.cost.token_share,
                confidence=conf,
            ))
            _mark_classified(a, i)

    # ── ordering_inefficiency ──
    # Compare positional ranks within each sequence, not raw indices which
    # belong to different arrays and are not directly comparable.
    if matched_pairs:
        n_compared = len(all_compared_actions)
        n_pairs = len(matched_pairs)
        threshold = max(3, int(n_pairs * 0.3))
        # Rank each matched pair by its position in the reference order and in
        # the compared order; a large rank gap means the action was performed
        # out of order (raw cross-array indices are not comparable).
        ref_rank = {k: r for r, k in enumerate(
            sorted(range(n_pairs), key=lambda k: matched_pairs[k][0]))}
        cmp_rank = {k: r for r, k in enumerate(
            sorted(range(n_pairs), key=lambda k: matched_pairs[k][1]))}
        for pair_pos, (ref_idx, cmp_idx) in enumerate(matched_pairs):
            if cmp_idx >= n_compared:
                continue
            positional_gap = abs(cmp_rank[pair_pos] - ref_rank[pair_pos])
            if positional_gap > threshold:
                a = all_compared_actions[cmp_idx]
                aid = _action_id(a, -1 - pair_pos)  # negative index for non-extra actions
                if aid not in classified:
                    patterns.append(_make_pattern(
                        ptype="ordering_inefficiency",
                        evidence=[f"{a.action_type}({a.target}) reordered — "
                                  f"reference rank {ref_rank[pair_pos]} vs compared rank {cmp_rank[pair_pos]}"],
                        steps=[a.step_index],
                        tokens=0,
                    ))
                    classified.add(aid)

    # ── dead_end_branch ──
    dead_end_run: list[tuple[int, CanonicalAction]] = []

    def _flush_dead_end():
        if len(dead_end_run) >= 2:
            conf = min(0.8, 0.5 + len(dead_end_run) * 0.1)
            patterns.append(_make_pattern(
                ptype="dead_end_branch",
                evidence=[f"{x.action_type}({x.target})" for _, x in dead_end_run],
                steps=[x.step_index for _, x in dead_end_run],
                tokens=sum(x.cost.token_share for _, x in dead_end_run),
                confidence=conf,
            ))
            for idx, x in dead_end_run:
                _mark_classified(x, idx)

    for i, a in enumerate(extra_actions):
        if _is_classified(a, i):
            _flush_dead_end()
            dead_end_run = []
            continue
        if a.action_type in ("FILE_READ", "SEARCH") and a.effect_label in _UNKNOWN_LABELS:
            dead_end_run.append((i, a))
        else:
            _flush_dead_end()
            dead_end_run = []

    _flush_dead_end()

    return patterns


# ---------------------------------------------------------------------------
# Cost attribution
# ---------------------------------------------------------------------------

def compute_pattern_costs(
    patterns: list[dict],
    token_rate: float = DEFAULT_TOKEN_RATE,
) -> None:
    """Ensure each pattern has estimated_extra_cost. Mutates in place."""
    for p in patterns:
        if "estimated_extra_cost" not in p:
            p["estimated_extra_cost"] = {"steps": len(p.get("steps", [])), "tokens": 0}
