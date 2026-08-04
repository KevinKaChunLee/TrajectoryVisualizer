"""Phase 1 tests for the DECAF attribution seam (trajviz/insight/attribution.py).

These de-risk the integration boundary:
  * golden       — diagnose() on a corpus case reproduces DECAF's committed
                   faults.json record (in-memory path == the batch pipeline;
                   DECAF's numbers are unchanged).
  * offline      — works with no OPENROUTER_API_KEY (reads cached verdicts).
  * degradation  — an off-corpus / missing-gold request returns available=False
                   with a reason, never raises, never fabricates a verdict.
  * parity       — DECAF's adapter on the raw trajectory file agrees with
                   DECAF's own load_normalized (guards the in-memory seam).

They are skipped (not failed) when DECAF or the TraceProbe corpus is absent, so
the suite still runs on a checkout without the data tree.
"""
import json
import os

import pytest

from trajviz.insight import attribution

GOLD_AGENT = "claude_code"
GOLD_INST = "astropy__astropy-13033"

pytestmark = pytest.mark.skipif(
    not attribution.DECAF_AVAILABLE, reason="DECAF (awe) not importable")


def _corpus_present() -> bool:
    if not attribution.DECAF_AVAILABLE:
        return False
    from awe import config
    return config.requirements_path(GOLD_INST).is_file()


def _faults_record():
    """The committed DECAF dossier record for the golden case, or None."""
    root = attribution._DECAF_ROOT
    fj = root / "data" / "dossier" / "faults.json"
    if not fj.is_file():
        return None
    for r in json.loads(fj.read_text()):
        if r.get("agent") == GOLD_AGENT and r.get("instance_id") == GOLD_INST:
            return r
    return None


requires_corpus = pytest.mark.skipif(
    not _corpus_present(), reason="TraceProbe corpus not present")


@requires_corpus
def test_corpus_diagnose_matches_committed_faults_json():
    ref = _faults_record()
    if ref is None:
        pytest.skip("faults.json golden record not present")
    res = attribution.diagnose(agent=GOLD_AGENT, instance_id=GOLD_INST)

    assert res.available is True
    assert res.mode == "corpus"
    assert res.blame_status == ref["blame_status"]

    ref_primary = next(({"capability": f["capability"], "error_type": f["error_type"]}
                        for f in ref["faults"] if f.get("is_primary")), None)
    assert res.primary == ref_primary

    got = {(f["capability"], f["error_type"]) for f in res.faults}
    exp = {(f["capability"], f["error_type"]) for f in ref["faults"]}
    assert got == exp

    # every returned fault carries the evidence chain the UI renders
    for f in res.faults:
        chain = f.get("evidence_chain") or {}
        assert chain.get("strength") in {"deductive", "associational", "model_inferred"}
        assert isinstance(chain.get("links"), list) and chain["links"]


@requires_corpus
def test_scorecard_marks_primary_capability_blamed():
    res = attribution.diagnose(agent=GOLD_AGENT, instance_id=GOLD_INST)
    assert res.available
    by_cap = {s.capability: s for s in res.scorecard}
    assert set(by_cap) == set(attribution.CAPABILITIES)   # all 7 present
    assert res.primary is not None
    assert by_cap[res.primary["capability"]].blamed is True
    assert by_cap[res.primary["capability"]].weight > 0


@requires_corpus
def test_diagnose_offline_without_api_key(monkeypatch):
    # cached judge/arbiter verdicts must be read back with no key and no network
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    res = attribution.diagnose(agent=GOLD_AGENT, instance_id=GOLD_INST)
    assert res.available is True
    assert res.blame_status is not None


@requires_corpus
def test_adapter_parity_claude_code():
    """DECAF's adapter on the raw trajectory file == DECAF's own load_normalized."""
    from awe import config
    from awe.adapters import adapt_claude_code, load_normalized

    raw = json.loads((config.TRAJECTORY_DIR / GOLD_AGENT / f"{GOLD_INST}.json").read_text())
    direct = adapt_claude_code(raw, GOLD_AGENT, GOLD_INST)
    disk = load_normalized(GOLD_AGENT, GOLD_INST)
    assert disk is not None
    assert len(direct.steps) == len(disk.steps)
    for a, b in zip(direct.steps, disk.steps):
        assert (a.actor, a.type, a.canonical_action, tuple(a.files or []),
                a.test_run, a.tool_error) == \
               (b.actor, b.type, b.canonical_action, tuple(b.files or []),
                b.test_run, b.tool_error)


def test_gold_free_upload_degrades_cleanly():
    res = attribution.diagnose(agent=GOLD_AGENT, instance_id="nonexistent__instance-99999")
    assert res.available is False
    assert res.mode == "gold_free"
    reason = (res.reason or "").lower()
    assert "gold" in reason or "reference" in reason
    assert res.faults == []
    assert res.primary is None


def test_missing_identifiers_degrade_cleanly():
    res = attribution.diagnose(agent=None, instance_id=None)
    assert res.available is False
    assert res.mode == "gold_free"
    assert res.faults == []
