"""DECAF failure-attribution seam for TrajViz.

The single, one-directional bridge between TrajViz and DECAF's ``awe`` package:
this module imports ``awe``; ``awe`` never imports ``trajviz``. Every DECAF call
the UI makes goes through :func:`diagnose`, which returns a plain
:class:`AttributionResult` (no Gradio types) so it is unit-testable headless.

Integrity model (hardened over three review rounds):

* **Gold-grounded.** DECAF's blame needs the run's reference patch + test
  outcome; without them we degrade honestly (never fabricate).
* **Canonical identity.** A displayed file must be byte-identical to the file
  DECAF itself parses for (agent, instance) — resolved by DECAF's own
  ``canonical_trajectory_path`` (codex prefers ``.jsonl``) so verification and
  diagnosis can never read different sources.
* **Cache provenance.** DECAF's normtraj/trajsig caches are content-keyed
  (v0.34). Judge/arbiter verdicts are accepted only when their stamped
  ``trajectory_sha256`` matches the canonical trajectory; unverifiable verdicts
  disable the LLM layers for that diagnosis (noted in the result) rather than
  shape it.
* **Isolation.** ``diagnose`` always configures an explicit corpus root (the
  immutable import-time default unless the caller passes one) under a module
  lock, so no request inherits another session's root and concurrent callers
  cannot interleave configuration and diagnosis.
* **Input validation.** ``agent`` must be a known corpus agent and
  ``instance_id`` a plain filename stem — path traversal cannot reach the
  filesystem.
"""
from __future__ import annotations

import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

# Causal capability order (mirrors DECAF's UPSTREAM_ORDER).
CAPABILITIES: tuple[str, ...] = (
    "requirement_understanding", "task_planning", "code_localization",
    "code_editing", "code_verification", "self_repair_loop", "tool_use",
)
_JUDGE_CAPS = frozenset({"requirement_understanding", "task_planning"})
# Always assessed on a failed case (the deductive core); the rest are conditional
# on an opportunity in the trajectory / a judged verdict.
_ALWAYS_ASSESSED = frozenset({"code_localization", "code_editing"})

# A safe instance id: a single filename stem (no separators, no leading dot).
_VALID_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# DECAF discovery + import (stdlib-only, not a pip package)
# --------------------------------------------------------------------------- #
def _decaf_root() -> Path | None:
    candidates = []
    env = os.environ.get("AWE_DECAF_PATH")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parents[3] / "DECAF")
    for c in candidates:
        if (c / "awe" / "__init__.py").is_file():
            return c
    return None


_DECAF_ROOT = _decaf_root()
if _DECAF_ROOT is not None and str(_DECAF_ROOT) not in sys.path:
    sys.path.insert(0, str(_DECAF_ROOT))

# The corpus root default is captured ONCE at import (immutable thereafter):
# env override, else a sibling TraceProbe. Every diagnose() call explicitly
# configures this default or the caller's root — never "whatever the previous
# caller left behind".
_DEFAULT_ROOT = Path(os.environ.get(
    "AWE_ARGUS_ROOT", str(Path(__file__).resolve().parents[3] / "TraceProbe")))
os.environ.setdefault("AWE_ARGUS_ROOT", str(_DEFAULT_ROOT))
# DECAF's judge/arbiter caches are partitioned by model slug; the checked-in
# caches were produced with z-ai/glm-5.2 — the awe default (claude-sonnet-4.5)
# points at an empty namespace, silently dropping the arbiter and judge.
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
    """Point DECAF at a corpus root and clear its root-dependent memory caches.

    Called by diagnose() under the module lock on EVERY request (explicit root
    or the immutable default) — configuration is per-call, never inherited.
    The outcome lru_cache lives on load_outcomes (resolved() is an uncached
    wrapper); clearing must fail loudly on DECAF API drift, not except-pass.
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
    from awe.gold import build_gold_reference
    from awe.outcomes import load_outcomes
    build_gold_reference.cache_clear()
    load_outcomes.cache_clear()


# --------------------------------------------------------------------------- #
# Result model
# --------------------------------------------------------------------------- #
@dataclass
class CapabilityScore:
    capability: str
    assessed: bool                # was this capability in scope for this case?
    blamed: bool                  # NON-ZERO blame — refuted faults are not blamed
    weight: float
    tier: str | None              # deductive | associational | model_inferred
    top_error: str | None


@dataclass
class AttributionResult:
    available: bool
    reason: str | None = None
    mode: str = "gold_free"       # corpus | gold_free
    agent: str | None = None
    instance_id: str | None = None
    blame_status: str | None = None
    primary: dict | None = None
    scorecard: list[CapabilityScore] = field(default_factory=list)
    faults: list[dict] = field(default_factory=list)
    arbiter: dict | None = None
    used_judge: bool = False      # judge verdict AVAILABLE for this case
    task: dict | None = None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sha256(p: Path) -> str | None:
    import hashlib
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def _llm_layer_policy(agent: str, instance_id: str, canon_sha: str):
    """Decide whether cached judge/arbiter verdicts may shape this diagnosis.

    A verdict is trusted only when its stamped trajectory_sha256 equals the
    canonical trajectory's hash. Legacy/unstamped or mismatched verdicts are
    provenance-unverifiable -> the corresponding layer is disabled (a stale
    verdict must never refute or judge a different trajectory).

    Returns (use_judge, adjudicate, notes): use_judge False disables the judge;
    adjudicate None disables the arbiter; _AUTO keeps DECAF's default.
    """
    from awe.judge import verdict_cached as judge_cached
    import awe.arbiter as _arb
    from awe.dossier import _AUTO
    notes: list[str] = []

    use_judge: bool | None = None      # DECAF auto
    jrec = judge_cached(agent, instance_id)
    if jrec is not None and jrec.get("trajectory_sha256") != canon_sha:
        use_judge = False
        notes.append("judge verdict ignored: its recorded trajectory hash does "
                     "not match this trajectory (provenance unverifiable)")

    adjudicate = _AUTO
    adir = _arb.ARBITER_CACHE_DIR / _cfg.model_slug() / agent
    if adir.is_dir():
        import json as _json
        for f in adir.glob(f"{instance_id}*.json"):
            try:
                rec = _json.loads(f.read_text())
            except Exception:
                continue
            if rec.get("trajectory_sha256") != canon_sha:
                adjudicate = None
                notes.append("arbiter verdict ignored: its recorded trajectory "
                             "hash does not match this trajectory (provenance "
                             "unverifiable)")
                break
    return use_judge, adjudicate, notes


# --------------------------------------------------------------------------- #
# Diagnosis
# --------------------------------------------------------------------------- #
def diagnose(*, agent: str | None, instance_id: str | None,
             source_path: str | os.PathLike | None = None, fmt: str | None = None,
             argus_root: str | os.PathLike | None = None) -> AttributionResult:
    """Diagnose the failure of the *displayed* trajectory. Never raises; never
    fabricates; degrades with an explicit reason. See the module docstring for
    the integrity model."""
    if not DECAF_AVAILABLE:
        return AttributionResult(False, reason=f"DECAF unavailable ({_IMPORT_ERROR})")
    with _LOCK:
        return _diagnose_locked(agent=agent, instance_id=instance_id,
                                source_path=source_path, fmt=fmt,
                                argus_root=argus_root)


def _diagnose_locked(*, agent, instance_id, source_path, fmt, argus_root):
    # per-call, explicit configuration — never inherited from a previous caller
    configure(argus_root if argus_root else _DEFAULT_ROOT)

    if not (agent and instance_id):
        return AttributionResult(
            False, mode="gold_free", agent=agent, instance_id=instance_id,
            reason="capability attribution needs (agent, instance_id); an arbitrary "
                   "trajectory upload has no reference patch to ground blame")
    # Validate BEFORE any filesystem access: agent from the known corpus list,
    # instance_id a single safe filename stem (no traversal, no separators).
    from awe import config
    if agent not in config.AGENTS:
        return AttributionResult(
            False, mode="corpus", agent=agent, instance_id=instance_id,
            reason=f"could not determine the agent for '{instance_id}' "
                   f"(got '{agent}'). Set the agent override to one of: "
                   f"{', '.join(config.AGENTS)}")
    if not _VALID_ID.fullmatch(instance_id) or ".." in instance_id:
        return AttributionResult(
            False, mode="gold_free", agent=agent, instance_id=instance_id,
            reason="invalid instance id (must be a plain corpus instance name)")

    if not config.requirements_path(instance_id).is_file():
        return AttributionResult(
            False, mode="gold_free", agent=agent, instance_id=instance_id,
            reason=f"no gold reference (data/requirements/{instance_id}.json) under "
                   f"{config.ARGUS_ROOT} — attribution needs the reference patch + "
                   f"test outcome; showing trajectory-only signals instead")

    # Canonical identity: resolve via DECAF's OWN source resolution so we verify
    # the very file diagnosis will parse (codex prefers .jsonl).
    from awe.adapters import canonical_trajectory_path
    canon = canonical_trajectory_path(agent, instance_id)
    if canon is None:
        return AttributionResult(
            False, mode="corpus", agent=agent, instance_id=instance_id,
            reason=f"no corpus trajectory for {agent}/{instance_id} under "
                   f"{config.ARGUS_ROOT} — cannot verify the displayed trajectory "
                   f"belongs to this run")
    canon_sha = _sha256(canon)
    if source_path is not None:
        sp = Path(source_path)
        if not (sp.resolve() == canon.resolve() or _sha256(sp) == canon_sha):
            return AttributionResult(
                False, mode="gold_free", agent=agent, instance_id=instance_id,
                reason=f"the displayed trajectory does not match the canonical "
                       f"{agent}/{instance_id} run — its patch and test outcome "
                       f"belong to a different execution, so a gold-grounded "
                       f"verdict would mix run provenance. Attribution of "
                       f"arbitrary uploads needs their own patch + outcome "
                       f"(not yet supported)")

    # LLM layers only when their cached verdicts verifiably belong to THIS
    # trajectory content.
    use_judge, adjudicate, notes = _llm_layer_policy(agent, instance_id, canon_sha)

    from awe.dossier import case_record
    from awe.detect import detect
    try:
        rec = case_record(agent, instance_id, use_judge=use_judge,
                          adjudicate=adjudicate)
        d = detect(instance_id, agent, use_judge=use_judge)
    except Exception as exc:  # never surface a raw traceback to the UI
        return AttributionResult(
            False, mode="corpus", agent=agent, instance_id=instance_id,
            reason=f"diagnosis failed for {agent}/{instance_id}: {type(exc).__name__}: {exc}")
    if rec is None:
        return AttributionResult(
            False, mode="corpus", agent=agent, instance_id=instance_id,
            reason="this run resolved (or its outcome is unknown); DECAF attributes "
                   "failures only")
    return _shape(rec, d.get("opportunities", {}), notes)


def _shape(rec: dict, opportunities: dict, notes: list[str]) -> AttributionResult:
    """case_record dict + detection opportunities -> AttributionResult.

    Blame is NON-ZERO only (arbiter-refuted faults carry weight 0). Assessment
    is per capability: the deductive core always; every other capability —
    including EACH judge capability separately — only when its own opportunity
    opened (DECAF can judge requirement_understanding while task_planning is out
    of scope for the same case)."""
    faults = rec.get("faults", [])
    primary = next(({"capability": f["capability"], "error_type": f["error_type"]}
                    for f in faults if f.get("is_primary")), None)

    agg: dict[str, dict] = {}
    for f in faults:
        cap = f["capability"]
        w = float(f.get("blame_weight") or 0.0)
        a = agg.setdefault(cap, {"weight": 0.0, "errors": {}, "tier_of": {}})
        a["weight"] += w
        if w > 0:
            et = f["error_type"]
            a["errors"][et] = a["errors"].get(et, 0.0) + w
            a["tier_of"][et] = (f.get("evidence_chain") or {}).get("strength")

    scorecard = []
    for cap in CAPABILITIES:
        a = agg.get(cap)
        weight = round(a["weight"], 3) if a else 0.0
        blamed = weight > 0
        assessed = blamed or cap in _ALWAYS_ASSESSED or bool(opportunities.get(cap))
        top = (max(a["errors"], key=a["errors"].get) if (a and a["errors"]) else None)
        tier = a["tier_of"].get(top) if (a and top) else None
        scorecard.append(CapabilityScore(cap, assessed, blamed, weight, tier, top))

    used_judge = any(bool(opportunities.get(c)) for c in _JUDGE_CAPS) or \
        any(f["capability"] in _JUDGE_CAPS for f in faults)

    return AttributionResult(
        available=True, mode="corpus",
        agent=rec.get("agent"), instance_id=rec.get("instance_id"),
        blame_status=rec.get("blame_status"), primary=primary,
        scorecard=scorecard, faults=faults, arbiter=rec.get("arbiter"),
        used_judge=used_judge, task=rec.get("task"), notes=notes,
    )
