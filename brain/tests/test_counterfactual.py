"""Tests for Phase 10 (#17): Counterfactual Evaluation Engine.

Pins the contract that matters:
  - DATA-GATED dormant by default (200+ outcomes, 500+ evals) — emits nothing,
    but still accumulates toward the gate.
  - When the gate opens, surfaces grounded 'missed opportunity' findings.
  - The counterfactual is grounded in lived policy history (best alternative tool
    for the decision's topic), deterministic, no-LLM.
  - read-only: never writes to policy memory; the regret signal is SHADOW only
    (live_influence=False) — synthetic regret must never drive live policy.
  - dedup watermark: a decision is evaluated once, ever.
  - state persists across restarts.
"""
from __future__ import annotations

import types

import pytest

from autonomy.policy_memory import PolicyOutcome
import epistemic.counterfactual.engine as cfmod
from epistemic.counterfactual.engine import (
    MIN_OUTCOMES,
    MIN_BUFFER,
    CounterfactualEngine,
)


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Point persisted state at a tmp file and reset the singleton per test."""
    monkeypatch.setattr(cfmod, "STATE_PATH", str(tmp_path / "cf_state.json"))
    CounterfactualEngine._instance = None
    yield
    CounterfactualEngine._instance = None


def _outcome(tool, tags, delta, ts, warmup=False, intent_id=None):
    return PolicyOutcome(
        intent_id=intent_id or f"{tool}-{ts}",
        intent_type="metric:test",
        tool_used=tool,
        topic_tags=tuple(tags),
        net_delta=delta,
        worked=delta > 0,
        warmup=warmup,
        timestamp=float(ts),
    )


def _fake_pm(outcomes):
    """A minimal stand-in for AutonomyPolicyMemory: just exposes _outcomes."""
    pm = types.SimpleNamespace(_outcomes=list(outcomes))
    pm.record_outcome = lambda *a, **k: pm.__dict__.setdefault("_writes", []).append((a, k))
    return pm


def _open_gate(monkeypatch, *, outcomes_floor=5, buffer_floor=5, full_conf=5):
    """Lower the data gate + confidence scale so a compact dataset activates it."""
    monkeypatch.setattr(cfmod, "MIN_OUTCOMES", outcomes_floor)
    monkeypatch.setattr(cfmod, "MIN_BUFFER", buffer_floor)
    monkeypatch.setattr(cfmod, "SAMPLES_FOR_FULL_CONFIDENCE", full_conf)


def _perf_dataset():
    """6 'codebase' wins (+0.3) and 6 'web' duds (0.0) on tag 'perf'."""
    outs = []
    ts = 1
    for _ in range(6):
        outs.append(_outcome("codebase", ["perf"], 0.30, ts)); ts += 1
    for _ in range(6):
        outs.append(_outcome("web", ["perf"], 0.00, ts)); ts += 1
    return outs


# ---------------------------------------------------------------------------
# Data gate
# ---------------------------------------------------------------------------

class TestDataGate:
    def test_dormant_by_default_emits_nothing(self):
        eng = CounterfactualEngine()
        emitted = eng.evaluate(policy_memory=_fake_pm(_perf_dataset()))
        assert emitted == []
        st = eng.get_state()
        assert st["active"] is False
        assert "data_gated" in st["reason"]
        assert st["missed_opportunity_count"] == 0

    def test_dormant_still_accumulates_toward_gate(self):
        eng = CounterfactualEngine()
        eng.evaluate(policy_memory=_fake_pm(_perf_dataset()))
        # 12 non-warmup decisions were evaluated even though nothing was emitted.
        assert eng.get_state()["total_evaluations"] == 12

    def test_gate_constants_match_spec(self):
        # #17: "Data-gated (200+ outcomes, buffer >500)" — pin so it can't drift.
        assert MIN_OUTCOMES == 200
        assert MIN_BUFFER == 500


# ---------------------------------------------------------------------------
# Active behaviour
# ---------------------------------------------------------------------------

class TestActive:
    def test_emits_grounded_missed_opportunity(self, monkeypatch):
        _open_gate(monkeypatch)
        eng = CounterfactualEngine()
        emitted = eng.evaluate(policy_memory=_fake_pm(_perf_dataset()))
        assert len(emitted) >= 1
        # Every emitted finding points the 'web' decisions at the better 'codebase'.
        for cf in emitted:
            assert cf.actual_tool == "web"
            assert cf.alternative_tool == "codebase"
            assert cf.regret == pytest.approx(0.30, abs=1e-6)
            assert cf.confidence >= cfmod.MIN_CONFIDENCE
        assert eng.get_state()["active"] is True
        assert eng.get_state()["missed_opportunity_count"] == len(emitted)

    def test_better_choice_is_not_flagged(self, monkeypatch):
        _open_gate(monkeypatch)
        eng = CounterfactualEngine()
        emitted = eng.evaluate(policy_memory=_fake_pm(_perf_dataset()))
        # The 'codebase' decisions made the right call — none should be flagged.
        assert all(cf.actual_tool != "codebase" for cf in emitted)

    def test_no_finding_without_tags(self, monkeypatch):
        _open_gate(monkeypatch)
        outs = [_outcome("web", [], 0.0, i) for i in range(1, 13)]
        eng = CounterfactualEngine()
        assert eng.evaluate(policy_memory=_fake_pm(outs)) == []

    def test_regret_below_threshold_not_flagged(self, monkeypatch):
        _open_gate(monkeypatch)
        # codebase only marginally better than web (< REGRET_THRESHOLD)
        outs = []
        ts = 1
        for _ in range(6):
            outs.append(_outcome("codebase", ["perf"], 0.02, ts)); ts += 1
        for _ in range(6):
            outs.append(_outcome("web", ["perf"], 0.00, ts)); ts += 1
        eng = CounterfactualEngine()
        assert eng.evaluate(policy_memory=_fake_pm(outs)) == []


# ---------------------------------------------------------------------------
# Read-only + shadow-only reward
# ---------------------------------------------------------------------------

class TestReadOnlyAndShadow:
    def test_never_writes_to_policy_memory(self, monkeypatch):
        _open_gate(monkeypatch)
        pm = _fake_pm(_perf_dataset())
        before = list(pm._outcomes)
        eng = CounterfactualEngine()
        eng.evaluate(policy_memory=pm)
        assert pm._outcomes == before          # unchanged
        assert "_writes" not in pm.__dict__    # record_outcome never called

    def test_live_influence_always_off(self, monkeypatch):
        _open_gate(monkeypatch)
        eng = CounterfactualEngine()
        eng.evaluate(policy_memory=_fake_pm(_perf_dataset()))
        st = eng.get_state()
        assert st["live_influence"] is False
        # regret accumulates as a negative shadow signal, but only in shadow.
        assert st["shadow_reward_sum"] < 0


# ---------------------------------------------------------------------------
# Determinism, dedup, persistence
# ---------------------------------------------------------------------------

class TestDeterminismDedupPersistence:
    def test_deterministic(self, monkeypatch, tmp_path):
        _open_gate(monkeypatch)
        # Two engines from CLEAN state (separate files) must agree.
        monkeypatch.setattr(cfmod, "STATE_PATH", str(tmp_path / "det_a.json"))
        a = CounterfactualEngine().evaluate(policy_memory=_fake_pm(_perf_dataset()))
        CounterfactualEngine._instance = None
        monkeypatch.setattr(cfmod, "STATE_PATH", str(tmp_path / "det_b.json"))
        b = CounterfactualEngine().evaluate(policy_memory=_fake_pm(_perf_dataset()))
        assert [f.to_dict() for f in a] == [f.to_dict() for f in b]

    def test_dedup_watermark(self, monkeypatch):
        _open_gate(monkeypatch)
        eng = CounterfactualEngine()
        pm = _fake_pm(_perf_dataset())
        eng.evaluate(policy_memory=pm)
        n_after_first = eng.get_state()["total_evaluations"]
        emitted2 = eng.evaluate(policy_memory=pm)   # same data, nothing new
        assert emitted2 == []
        assert eng.get_state()["total_evaluations"] == n_after_first

    def test_new_decision_after_watermark_is_evaluated(self, monkeypatch):
        _open_gate(monkeypatch)
        eng = CounterfactualEngine()
        outs = _perf_dataset()
        pm = _fake_pm(outs)
        eng.evaluate(policy_memory=pm)
        n1 = eng.get_state()["total_evaluations"]
        pm._outcomes.append(_outcome("web", ["perf"], 0.0, 999))  # newer ts
        eng.evaluate(policy_memory=pm)
        assert eng.get_state()["total_evaluations"] == n1 + 1

    def test_state_persists_across_restart(self, monkeypatch):
        _open_gate(monkeypatch)
        eng = CounterfactualEngine()
        eng.evaluate(policy_memory=_fake_pm(_perf_dataset()))
        saved_evals = eng.get_state()["total_evaluations"]
        saved_miss = eng.get_state()["missed_opportunity_count"]
        CounterfactualEngine._instance = None
        eng2 = CounterfactualEngine()  # reloads from the tmp state file
        assert eng2.get_state()["total_evaluations"] == saved_evals
        assert eng2.get_state()["missed_opportunity_count"] == saved_miss


# ---------------------------------------------------------------------------
# Layer-9 scanner integration
# ---------------------------------------------------------------------------

class TestLayer9Integration:
    def test_missed_opportunity_is_an_audit_category(self):
        from epistemic.reflective_audit import engine as audit_mod
        # The Literal type can't be introspected directly; assert the scanner exists
        assert hasattr(audit_mod.ReflectiveAuditEngine, "_scan_missed_opportunities")

    def test_scanner_returns_list_when_dormant(self):
        from epistemic.reflective_audit.engine import ReflectiveAuditEngine
        audit = ReflectiveAuditEngine()
        findings = audit._scan_missed_opportunities()  # engine dormant -> []
        assert isinstance(findings, list)
        assert findings == []
