"""DECAF failure-attribution seam for TrajViz.

The single, one-directional bridge between TrajViz and DECAF's ``awe`` package:
this module imports ``awe``; ``awe`` never imports ``trajviz``. Every DECAF call
the UI makes goes through :func:`diagnose`, which returns a plain
:class:`AttributionResult` (no Gradio types) so it is unit-testable headless.

DECAF attributes a coding-agent *failure* to one of seven workflow capabilities
with tiered evidence (deductive / associational / model-inferred). That
attribution is **gold-grounded** — it needs the reference patch + test outcome —
so :func:`diagnose` operates in three modes:

* ``corpus``     – the trajectory is an on-disk ``TraceProbe`` case; we call
                   DECAF's own ``case_record`` (the battle-tested path that
                   produced ``faults.json``) for maximum fidelity.
* ``gold_free``  – no reference patch is available (an arbitrary upload); we
                   return ``available=False`` with a reason. We never fabricate
                   a verdict.
* (``gold_provided`` — an off-corpus trajectory plus a supplied reference — is
   scaffolded via DECAF's in-memory ``detect(trajectory_override=...)`` overrides
   and lands in a later phase; see docs/decaf-integration-plan.md.)

DECAF is stdlib-only and not published to PyPI, so it is located at runtime via
``AWE_DECAF_PATH`` (default: a sibling ``../DECAF`` checkout). The corpus root
defaults to a sibling ``../TraceProbe`` and can be re-pointed with
:func:`configure`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Causal capability order (mirrors DECAF's UPSTREAM_ORDER). Used to lay out the
# scorecard; DECAF remains the source of truth for which are blamed.
CAPABILITIES: tuple[str, ...] = (
    "requirement_understanding", "task_planning", "code_localization",
    "code_editing", "code_verification", "self_repair_loop", "tool_use",
)
_JUDGE_CAPS = frozenset({"requirement_understanding", "task_planning"})


# --------------------------------------------------------------------------- #
# DECAF discovery + import (stdlib-only, not a pip package)
# --------------------------------------------------------------------------- #
def _decaf_root() -> Path | None:
    """Locate a DECAF checkout containing the ``awe`` package."""
    candidates = []
    env = os.environ.get("AWE_DECAF_PATH")
    if env:
        candidates.append(Path(env))
    # default: …/ARGUS/DECAF, a sibling of this repo
    # attribution.py -> insight -> trajviz -> <repo> -> <parent>
    candidates.append(Path(__file__).resolve().parents[3] / "DECAF")
    for c in candidates:
        if (c / "awe" / "__init__.py").is_file():
            return c
    return None


_DECAF_ROOT = _decaf_root()
if _DECAF_ROOT is not None and str(_DECAF_ROOT) not in sys.path:
    sys.path.insert(0, str(_DECAF_ROOT))

# The corpus root must be set BEFORE importing awe.config (it freezes ARGUS_ROOT
# at import). Default to a sibling TraceProbe; a later configure() can re-point.
os.environ.setdefault(
    "AWE_ARGUS_ROOT", str(Path(__file__).resolve().parents[3] / "TraceProbe"))
# DECAF's judge/arbiter verdict caches are partitioned by model slug. The
# checked-in caches were produced with z-ai/glm-5.2, so we must read them back
# under THAT namespace — the awe default (anthropic/claude-sonnet-4.5) points at
# an empty namespace, which silently drops the arbiter (refuted faults would then
# show as blamed) and the two judge capabilities. Override AWE_JUDGE_MODEL to
# change this; it only selects which cache namespace is read (no key needed).
os.environ.setdefault("AWE_JUDGE_MODEL", "z-ai/glm-5.2")

DECAF_AVAILABLE = False
_IMPORT_ERROR: str | None = None
try:
    if _DECAF_ROOT is None:
        raise ImportError(
            "DECAF (awe) not found — set AWE_DECAF_PATH to a DECAF checkout")
    from awe import config as _cfg  # noqa: E402
    DECAF_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment-dependent
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def configure(argus_root: str | os.PathLike) -> None:
    """Re-point the DECAF corpus root at runtime (e.g. from a UI field).

    ``awe.config`` freezes its path constants at import, so we reassign the
    derived ones and clear the gold-reference cache. No-op if DECAF is absent.
    """
    if not DECAF_AVAILABLE:
        return
    root = Path(argus_root).resolve()
    os.environ["AWE_ARGUS_ROOT"] = str(root)
    _cfg.ARGUS_ROOT = root
    _cfg.DATA = root / "data"
    _cfg.REQUIREMENTS_DIR = _cfg.DATA / "requirements"
    _cfg.PATCH_DIR = _cfg.DATA / "patch"
    _cfg.TRAJECTORY_DIR = _cfg.DATA / "trajectory"
    _cfg.LABELS_DIR = _cfg.DATA / "labels"
    # Clear DECAF's per-instance memoized caches so a corpus switch cannot serve a
    # previous root's gold/outcome for a same-named instance_id (both are
    # @lru_cache keyed by id, not by root).
    for mod, fn in (("awe.gold", "build_gold_reference"),
                    ("awe.outcomes", "resolved")):
        try:
            import importlib
            getattr(importlib.import_module(mod), fn).cache_clear()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Result model (plain data — no Gradio, no awe types leak to the UI)
# --------------------------------------------------------------------------- #
@dataclass
class CapabilityScore:
    capability: str
    assessed: bool                # was this capability in scope for this case?
    blamed: bool                  # NON-ZERO blame — arbiter-refuted faults are not blamed
    weight: float                 # summed blame weight for this capability (0..1)
    tier: str | None              # deductive | associational | model_inferred
    top_error: str | None         # dominant error_type, if blamed


# TrajViz format -> DECAF adapter (reused so no re-normalization / drift). CodeArts
# has no DECAF adapter yet -> those trajectories can't be diagnosed (degrade).
_FMT_ADAPTER = {
    "ccsession": ("adapt_claude_code", "json"),
    "opencode": ("adapt_opencode", "json"),
    "codex": ("adapt_codex", "lines"),
}
# Always assessed on a failed case (the deductive core); the rest are conditional
# on an opportunity in the trajectory.
_ALWAYS_ASSESSED = frozenset({"code_localization", "code_editing"})


@dataclass
class AttributionResult:
    available: bool
    reason: str | None = None
    mode: str = "gold_free"       # corpus | gold_provided | gold_free
    agent: str | None = None
    instance_id: str | None = None
    blame_status: str | None = None      # primary | conjunctive | refuted_unattributed | unattributed
    primary: dict | None = None          # {capability, error_type} of the elected primary
    scorecard: list[CapabilityScore] = field(default_factory=list)
    faults: list[dict] = field(default_factory=list)   # DECAF fault records (verbatim)
    arbiter: dict | None = None
    used_judge: bool = False             # 7-cap (judge cache present) vs 5-cap deductive
    task: dict | None = None             # {problem_statement, gold_files, ...}


# --------------------------------------------------------------------------- #
# Diagnosis
# --------------------------------------------------------------------------- #
def _adapt_displayed(source_path, fmt, agent, instance_id):
    """Adapt the DISPLAYED trajectory file into a DECAF NormalizedTrajectory via
    DECAF's own adapters, so diagnosis reflects what the UI shows rather than the
    canonical corpus copy. Returns (nt, reason): nt=None with reason=None means
    'no readable source — fall back to the corpus copy'; a non-None reason is a
    real adapt failure worth surfacing."""
    spec = _FMT_ADAPTER.get(fmt or "")
    if spec is None:
        return None, (f"format '{fmt}' has no DECAF adapter "
                      f"(CodeArts is not yet diagnosable)" if fmt else None)
    if not source_path or not Path(source_path).is_file():
        return None, None
    fn_name, kind = spec
    import importlib
    import json as _json
    adapt = getattr(importlib.import_module("awe.adapters"), fn_name)
    p = Path(source_path)
    try:
        raw = (p.read_text(errors="replace").splitlines() if kind == "lines"
               else _json.loads(p.read_text(errors="replace")))
        return adapt(raw, agent, instance_id), None
    except Exception as exc:
        return None, f"could not adapt the displayed trajectory: {type(exc).__name__}: {exc}"


def diagnose(*, agent: str | None, instance_id: str | None,
             source_path: str | os.PathLike | None = None, fmt: str | None = None,
             argus_root: str | os.PathLike | None = None) -> AttributionResult:
    """Diagnose the failure of the *displayed* trajectory and return an
    AttributionResult.

    Requires the on-disk gold reference at
    ``<argus_root>/data/requirements/<instance_id>.json``. When ``source_path`` +
    ``fmt`` are given, the displayed trajectory file is adapted and diagnosed
    directly (so we never silently diagnose the canonical corpus copy); otherwise
    the on-disk corpus trajectory is used as a fallback. Returns
    ``available=False`` (never raises, never fabricates) when the gold-grounded
    inputs are absent.
    """
    if not DECAF_AVAILABLE:
        return AttributionResult(False, reason=f"DECAF unavailable ({_IMPORT_ERROR})")
    if argus_root is not None:
        configure(argus_root)
    if not (agent and instance_id):
        return AttributionResult(
            False, mode="gold_free", agent=agent, instance_id=instance_id,
            reason="capability attribution needs (agent, instance_id); an arbitrary "
                   "trajectory upload has no reference patch to ground blame")

    from awe import config
    if not config.requirements_path(instance_id).is_file():
        return AttributionResult(
            False, mode="gold_free", agent=agent, instance_id=instance_id,
            reason=f"no gold reference (data/requirements/{instance_id}.json) under "
                   f"{config.ARGUS_ROOT} — attribution needs the reference patch + "
                   f"test outcome; showing trajectory-only signals instead")
    if agent not in config.AGENTS:
        return AttributionResult(
            False, mode="corpus", agent=agent, instance_id=instance_id,
            reason=f"could not determine the agent for '{instance_id}' "
                   f"(got '{agent}'). Set the agent override to one of: "
                   f"{', '.join(config.AGENTS)}")

    # Diagnose what's displayed: adapt the loaded file and pass it as an override.
    nt, adapt_reason = _adapt_displayed(source_path, fmt, agent, instance_id)
    if nt is None and adapt_reason and not config.patch_path(agent, instance_id).is_file():
        # a genuine off-corpus upload whose trajectory couldn't be adapted and has
        # no corpus fallback -> surface the reason rather than guess.
        return AttributionResult(False, mode="gold_free", agent=agent,
                                 instance_id=instance_id, reason=adapt_reason)

    from awe.dossier import case_record
    from awe.detect import detect
    try:
        rec = case_record(agent, instance_id, trajectory_override=nt)
        d = detect(instance_id, agent, trajectory_override=nt)  # for opportunities
    except Exception as exc:  # never surface a raw traceback to the UI
        return AttributionResult(
            False, mode="corpus", agent=agent, instance_id=instance_id,
            reason=f"diagnosis failed for {agent}/{instance_id}: {type(exc).__name__}: {exc}")
    if rec is None:
        return AttributionResult(
            False, mode="corpus", agent=agent, instance_id=instance_id,
            reason="this run resolved (or its outcome is unknown); DECAF attributes "
                   "failures only")
    return _shape(rec, d.get("opportunities", {}),
                  traj_source=("displayed" if nt is not None else "corpus"))


def _shape(rec: dict, opportunities: dict,
           traj_source: str = "corpus") -> AttributionResult:
    """Turn a DECAF case_record dict + the detection's opportunities into an
    AttributionResult. Blame is NON-ZERO only (arbiter-refuted faults carry weight
    0 and are not 'blamed'); a capability with no opportunity is 'not assessed'
    (n/a), distinct from 'assessed clean'."""
    faults = rec.get("faults", [])
    primary = next(({"capability": f["capability"], "error_type": f["error_type"]}
                    for f in faults if f.get("is_primary")), None)

    agg: dict[str, dict] = {}
    for f in faults:
        cap = f["capability"]
        w = float(f.get("blame_weight") or 0.0)
        a = agg.setdefault(cap, {"weight": 0.0, "tier": None, "errors": {}})
        a["weight"] += w
        if w > 0:   # tier / top_error reflect BLAMED faults only (skip refuted)
            a["tier"] = (f.get("evidence_chain") or {}).get("strength") or a["tier"]
            a["errors"][f["error_type"]] = a["errors"].get(f["error_type"], 0.0) + w

    scorecard = []
    for cap in CAPABILITIES:
        a = agg.get(cap)
        weight = round(a["weight"], 3) if a else 0.0
        blamed = weight > 0
        assessed = blamed or cap in _ALWAYS_ASSESSED or bool(opportunities.get(cap))
        top = (max(a["errors"], key=a["errors"].get) if (a and a["errors"]) else None)
        scorecard.append(CapabilityScore(cap, assessed, blamed, weight,
                                         (a["tier"] if a else None), top))

    used_judge = any(f["capability"] in _JUDGE_CAPS and float(f.get("blame_weight") or 0) > 0
                     for f in faults) or any(
        (f.get("evidence_chain") or {}).get("strength") == "model_inferred"
        for f in faults)

    return AttributionResult(
        available=True, mode="corpus",
        agent=rec.get("agent"), instance_id=rec.get("instance_id"),
        blame_status=rec.get("blame_status"), primary=primary,
        scorecard=scorecard, faults=faults, arbiter=rec.get("arbiter"),
        used_judge=used_judge, task=rec.get("task"),
    )
