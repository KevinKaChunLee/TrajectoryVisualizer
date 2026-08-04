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
    try:
        from awe.gold import build_gold_reference
        build_gold_reference.cache_clear()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Result model (plain data — no Gradio, no awe types leak to the UI)
# --------------------------------------------------------------------------- #
@dataclass
class CapabilityScore:
    capability: str
    blamed: bool
    weight: float                 # summed blame weight for this capability (0..1)
    tier: str | None              # deductive | associational | model_inferred
    top_error: str | None         # dominant error_type, if blamed


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
def diagnose(*, agent: str | None, instance_id: str | None,
             argus_root: str | os.PathLike | None = None) -> AttributionResult:
    """Diagnose one (agent, instance) failure and return an AttributionResult.

    Corpus mode only in this phase: requires the on-disk reference at
    ``<argus_root>/data/requirements/<instance_id>.json``. Returns
    ``available=False`` (never raises, never fabricates) when the inputs the
    gold-grounded method needs are absent.
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
    # An uploaded file lands at a temp path, so the corpus agent dir isn't always
    # recoverable — require a valid agent (blame reads patch/<agent>/ + eval_<agent>).
    if agent not in config.AGENTS:
        return AttributionResult(
            False, mode="corpus", agent=agent, instance_id=instance_id,
            reason=f"could not determine the agent for '{instance_id}' "
                   f"(got '{agent}'). Set the agent override to one of: "
                   f"{', '.join(config.AGENTS)}")

    from awe.dossier import case_record
    try:
        rec = case_record(agent, instance_id)
    except Exception as exc:  # never surface a raw traceback to the UI
        return AttributionResult(
            False, mode="corpus", agent=agent, instance_id=instance_id,
            reason=f"diagnosis failed for {agent}/{instance_id}: {type(exc).__name__}: {exc}")
    if rec is None:
        return AttributionResult(
            False, mode="corpus", agent=agent, instance_id=instance_id,
            reason="this run resolved (or its outcome is unknown); DECAF attributes "
                   "failures only")
    return _shape(rec)


def _shape(rec: dict) -> AttributionResult:
    """Turn a DECAF case_record dict into an AttributionResult + scorecard."""
    faults = rec.get("faults", [])
    primary = next(({"capability": f["capability"], "error_type": f["error_type"]}
                    for f in faults if f.get("is_primary")), None)

    # per-capability rollup for the scorecard
    agg: dict[str, dict] = {}
    for f in faults:
        cap = f["capability"]
        a = agg.setdefault(cap, {"weight": 0.0, "tier": None, "errors": {}})
        a["weight"] += float(f.get("blame_weight") or 0.0)
        a["tier"] = (f.get("evidence_chain") or {}).get("strength") or a["tier"]
        a["errors"][f["error_type"]] = a["errors"].get(f["error_type"], 0.0) + \
            float(f.get("blame_weight") or 0.0)
    scorecard = []
    for cap in CAPABILITIES:
        a = agg.get(cap)
        if a:
            top = max(a["errors"], key=a["errors"].get) if a["errors"] else None
            scorecard.append(CapabilityScore(cap, True, round(a["weight"], 3),
                                             a["tier"], top))
        else:
            scorecard.append(CapabilityScore(cap, False, 0.0, None, None))

    used_judge = any(f["capability"] in _JUDGE_CAPS for f in faults) or any(
        (f.get("evidence_chain") or {}).get("strength") == "model_inferred"
        for f in faults)

    return AttributionResult(
        available=True, mode="corpus",
        agent=rec.get("agent"), instance_id=rec.get("instance_id"),
        blame_status=rec.get("blame_status"), primary=primary,
        scorecard=scorecard, faults=faults, arbiter=rec.get("arbiter"),
        used_judge=used_judge, task=rec.get("task"),
    )
