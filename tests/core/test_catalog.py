"""Tests for the pattern catalog.

Lock the counts (20 [S] + 7 [H] + 6 divergence = 33) and invariants that
downstream code depends on. If a catalog record is renamed, counts changed,
or thresholds mutated, these tests fail loudly.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from trajectory_visualizer.core.catalog import (
    ALL_PATTERNS,
    PatternRecord,
    by_band,
    by_id,
    by_phase,
    thresholds_for,
)


# ---------------------------------------------------------------------------
# Counts — locked to the paper's catalog
# ---------------------------------------------------------------------------

def test_total_count_matches_paper() -> None:
    assert len(ALL_PATTERNS) == 33  # 20 [S] + 7 [H] + 6 divergence


def test_structural_count() -> None:
    assert len(by_band("[S]")) == 20


def test_hypothesis_count() -> None:
    assert len(by_band("[H]")) == 7


def test_divergence_count() -> None:
    assert len(by_band("divergence")) == 6


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------

def test_ids_are_unique() -> None:
    ids = [p.id for p in ALL_PATTERNS]
    assert len(ids) == len(set(ids)), "duplicate pattern ids in catalog"


def test_legacy_aliases_do_not_collide_with_ids() -> None:
    live_ids = {p.id for p in ALL_PATTERNS}
    for p in ALL_PATTERNS:
        for alias in p.legacy_aliases:
            assert alias not in live_ids, (
                f"legacy alias {alias!r} on {p.id!r} collides with a live id"
            )


# ---------------------------------------------------------------------------
# Per-band invariants
# ---------------------------------------------------------------------------

def test_every_h_record_requires_labels() -> None:
    for p in by_band("[H]"):
        assert p.requires_semantic_labels is True, (
            f"{p.id}: [H] record must have requires_semantic_labels=True"
        )


def test_no_s_record_requires_labels() -> None:
    for p in by_band("[S]"):
        assert p.requires_semantic_labels is False, (
            f"{p.id}: [S] record must not require semantic labels"
        )


def test_every_divergence_has_a_tier() -> None:
    for p in by_band("divergence"):
        assert p.tier in {"high", "med", "low"}, (
            f"{p.id}: divergence record must have tier in high/med/low, got {p.tier!r}"
        )


def test_every_divergence_lists_required_signals() -> None:
    for p in by_band("divergence"):
        assert len(p.required_signals) > 0, (
            f"{p.id}: divergence record must list required_signals"
        )


def test_only_divergence_records_have_a_tier() -> None:
    for p in ALL_PATTERNS:
        if p.band != "divergence":
            assert p.tier is None, f"{p.id}: only divergence records may carry a tier"


def test_anti_patterns_have_a_phase() -> None:
    """All [S] and [H] records must declare a lifecycle phase."""
    for p in ALL_PATTERNS:
        if p.band in {"[S]", "[H]"}:
            assert p.phase is not None, f"{p.id}: anti-pattern missing phase"


def test_divergence_records_have_no_phase() -> None:
    for p in by_band("divergence"):
        assert p.phase is None, f"{p.id}: divergence records must not carry a phase"


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

def test_record_is_frozen() -> None:
    record = ALL_PATTERNS[0]
    with pytest.raises(FrozenInstanceError):
        record.name = "mutated"  # type: ignore[misc]


def test_thresholds_mapping_is_read_only() -> None:
    """thresholds is a MappingProxyType, so item assignment must raise."""
    record = by_id("search-loop")
    with pytest.raises(TypeError):
        record.thresholds["min_consecutive_steps"] = 999  # type: ignore[index]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def test_by_id_live_name() -> None:
    r = by_id("search-loop")
    assert r.name == "Search loop"
    assert r.band == "[S]"


def test_by_id_legacy_alias() -> None:
    r = by_id("fruitless_streak")  # legacy alias
    assert r.id == "search-loop"


def test_by_id_unknown_raises() -> None:
    with pytest.raises(KeyError):
        by_id("definitely-not-a-pattern")


def test_by_phase_returns_only_matching() -> None:
    intake = by_phase("intake")
    assert {p.id for p in intake} >= {"memory-bypass", "premature-code-action"}
    for p in intake:
        assert p.phase == "intake"


def test_thresholds_for_returns_defaults() -> None:
    t = thresholds_for("search-loop")
    assert t["min_consecutive_steps"] == 4
