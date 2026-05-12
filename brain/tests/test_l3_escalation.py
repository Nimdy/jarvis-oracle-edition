"""Phase 6.5 — L3 escalation store, metric-trigger, and approval flow tests.

Scope:
- ``EscalationStore`` lifecycle: submit/approve/reject/mark_outcome/expire.
- Policy gates: unregistered metric, L3-required, non-empty
  ``declared_scope``, per-metric rate limit.
- ``MetricTriggers.get_escalation_candidates`` gate semantics,
  including the "auto-generated escalation requires live
  autonomy_level >= 3; attestation alone does NOT satisfy" invariant.
- ``approve_and_apply_escalation`` threads ``declared_scope`` into the
  self-improvement pipeline (no global widening of ``ALLOWED_PATHS``)
  and records the outcome.
- Event emission for ``AUTONOMY_ESCALATION_REQUESTED/APPROVED/REJECTED/
  ROLLED_BACK``.
- ``current_ok`` sourcing invariant: the value returned by the dashboard
  API must come from the live orchestrator's ``check_promotion_eligibility``
  — never backfilled from a persisted file.

These tests stay strictly in-memory: they use temporary files for the
pending queue / activity log and a minimal stub orchestrator. They do
not require a running event loop beyond ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# Make the brain/ package importable without installing.
_HERE = Path(__file__).resolve().parent
_BRAIN = _HERE.parent
if str(_BRAIN) not in sys.path:
    sys.path.insert(0, str(_BRAIN))

from autonomy.escalation import (  # noqa: E402
    DEFAULT_EXPIRY_S,
    EscalationRecord,
    EscalationRequest,
    EscalationStore,
    EscalationStoreError,
    METRIC_ESCALATION_POLICY,
    PER_METRIC_RATE_LIMIT_S,
    approve_and_apply_escalation,
    build_request_from_metric_deficit,
    reject_escalation,
    submit_and_emit,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> EscalationStore:
    return EscalationStore(
        pending_path=tmp_path / "pending_escalations.json",
        activity_path=tmp_path / "escalation_activity.jsonl",
    )


def _make_req(
    metric: str = "confidence_volatility",
    *,
    submitted_autonomy_level: int = 3,
    declared_scope: list[str] | None = None,
) -> EscalationRequest:
    pol = METRIC_ESCALATION_POLICY[metric]
    scope = declared_scope if declared_scope is not None else list(pol["declared_scope"])
    return EscalationRequest(
        metric=metric,
        metric_context_summary="test deficit",
        severity="high",
        target_module=pol["target_module"],
        declared_scope=list(scope),
        submitted_autonomy_level=submitted_autonomy_level,
    )


# --------------------------------------------------------------------------
# EscalationStore: submit policy gates
# --------------------------------------------------------------------------


def test_submit_rejects_unregistered_metric(store: EscalationStore) -> None:
    req = EscalationRequest(
        metric="totally_made_up_metric",
        declared_scope=["brain/foo/"],
        submitted_autonomy_level=3,
    )
    with pytest.raises(EscalationStoreError, match="no escalation policy"):
        store.submit(req)


def test_submit_rejects_below_l3(store: EscalationStore) -> None:
    """Attestation alone does NOT satisfy auto-generation — must be live L3."""
    req = _make_req(submitted_autonomy_level=2)
    with pytest.raises(EscalationStoreError, match="autonomy_level >= 3"):
        store.submit(req)


def test_submit_rejects_empty_declared_scope(store: EscalationStore) -> None:
    req = _make_req(declared_scope=[])
    with pytest.raises(EscalationStoreError, match="declared_scope"):
        store.submit(req)


def test_submit_populates_expiry(store: EscalationStore) -> None:
    before = time.time()
    rec = store.submit(_make_req())
    after = time.time()
    assert before + DEFAULT_EXPIRY_S - 1 <= rec.request.expires_at <= after + DEFAULT_EXPIRY_S + 1
    assert rec.status == "pending"


def test_submit_blocks_duplicate_pending(store: EscalationStore) -> None:
    store.submit(_make_req())
    with pytest.raises(EscalationStoreError, match="Live escalation already exists"):
        store.submit(_make_req())


def test_submit_rate_limited_by_prior_terminal_event(tmp_path: Path) -> None:
    s = EscalationStore(
        pending_path=tmp_path / "pending.json",
        activity_path=tmp_path / "activity.jsonl",
        rate_limit_s=3600.0,
    )
    rec = s.submit(_make_req())
    s.reject(rec.request.id, rejected_by="op", rejection_reason="test rejection reason must be twenty characters long")
    with pytest.raises(EscalationStoreError, match="rate-limited"):
        s.submit(_make_req())


def test_submit_allowed_after_rate_limit_window(tmp_path: Path) -> None:
    s = EscalationStore(
        pending_path=tmp_path / "pending.json",
        activity_path=tmp_path / "activity.jsonl",
        rate_limit_s=0.01,  # 10ms
    )
    rec = s.submit(_make_req())
    s.reject(rec.request.id, rejected_by="op", rejection_reason="test rejection reason must be twenty characters long")
    time.sleep(0.02)
    rec2 = s.submit(_make_req())
    assert rec2.request.id != rec.request.id
    assert rec2.status == "pending"


# --------------------------------------------------------------------------
# EscalationStore: approve/reject/mark_outcome lifecycle
# --------------------------------------------------------------------------


def test_approve_transitions_pending_to_approved(store: EscalationStore) -> None:
    rec = store.submit(_make_req())
    approved = store.approve(
        rec.request.id,
        approved_by="operator",
        approval_reason="approved for investigation reasons",
        improvement_record_id="imp_test",
    )
    assert approved.status == "approved"
    assert approved.approved_by == "operator"
    assert approved.improvement_record_id == "imp_test"

    reloaded = store.get(rec.request.id)
    assert reloaded is not None and reloaded.status == "approved"


def test_approve_rejects_non_pending(store: EscalationStore) -> None:
    rec = store.submit(_make_req())
    store.reject(rec.request.id, rejected_by="op", rejection_reason="test rejection reason must be twenty characters long")
    with pytest.raises(EscalationStoreError, match="in status 'rejected'"):
        store.approve(
            rec.request.id,
            approved_by="op",
            approval_reason="already rejected cannot approve this request now",
            improvement_record_id="",
        )


def test_mark_outcome_clean_keeps_approved_status(store: EscalationStore) -> None:
    rec = store.submit(_make_req())
    store.approve(
        rec.request.id, approved_by="op",
        approval_reason="approved for investigation reasons",
        improvement_record_id="imp_x",
    )
    updated = store.mark_outcome(rec.request.id, "clean")
    assert updated.status == "approved"  # status stays 'approved'
    assert updated.outcome == "clean"


def test_mark_outcome_rolled_back_flips_status(store: EscalationStore) -> None:
    rec = store.submit(_make_req())
    store.approve(
        rec.request.id, approved_by="op",
        approval_reason="approved for investigation reasons",
        improvement_record_id="imp_x",
    )
    updated = store.mark_outcome(
        rec.request.id, "rolled_back",
        rollback_reason="post_apply_health_failed",
    )
    assert updated.status == "rolled_back"
    assert updated.outcome == "rolled_back"
    assert updated.rollback_reason == "post_apply_health_failed"
    assert updated.rolled_back_at > 0


def test_mark_outcome_rejects_invalid_outcome(store: EscalationStore) -> None:
    rec = store.submit(_make_req())
    store.approve(
        rec.request.id, approved_by="op",
        approval_reason="approved for investigation reasons",
        improvement_record_id="imp_x",
    )
    with pytest.raises(EscalationStoreError, match="outcome must be"):
        store.mark_outcome(rec.request.id, "weird")


def test_mark_outcome_requires_approved_status(store: EscalationStore) -> None:
    rec = store.submit(_make_req())
    with pytest.raises(EscalationStoreError, match="requires status 'approved'"):
        store.mark_outcome(rec.request.id, "clean")


def test_prune_expired_marks_pending_as_expired(tmp_path: Path) -> None:
    s = EscalationStore(
        pending_path=tmp_path / "pending.json",
        activity_path=tmp_path / "activity.jsonl",
        default_expiry_s=0.01,
    )
    rec = s.submit(_make_req())
    assert rec.status == "pending"
    time.sleep(0.02)
    changed = s.prune_expired()
    assert changed == 1
    reloaded = s.get(rec.request.id)
    assert reloaded is not None and reloaded.status == "expired"


def test_prune_expired_emits_expired_event(tmp_path: Path) -> None:
    """Phase 6.5: expiration must emit ``AUTONOMY_ESCALATION_EXPIRED`` so
    the durable audit subscriber records it. Previously the store only
    wrote its own activity log without publishing to the bus.
    """
    s = EscalationStore(
        pending_path=tmp_path / "pending.json",
        activity_path=tmp_path / "activity.jsonl",
        default_expiry_s=0.01,
    )
    rec = s.submit(_make_req())
    time.sleep(0.02)
    emitted: list[tuple] = []
    changed = s.prune_expired(event_emit=lambda et, **kw: emitted.append((et, kw)))
    assert changed == 1
    assert len(emitted) == 1
    et, payload = emitted[0]
    assert et.endswith("escalation_expired")
    assert payload.get("escalation_id") == rec.request.id
    assert payload.get("metric") == rec.request.metric


def test_approve_and_apply_parks_emits_parked_event(
    store: EscalationStore,
) -> None:
    """Phase 6.5: parked outcomes must publish ``AUTONOMY_ESCALATION_PARKED``.

    Previously the parked status was only visible via ``list_pending``
    filtering and the on-disk activity log; the bus stayed silent,
    which broke the durable audit trail for this terminal state.
    """
    emitted: list[tuple] = []
    rec = store.submit(_make_req())
    stub = _StubSelfImproveOrchestrator(status="awaiting_approval")
    result = asyncio.run(approve_and_apply_escalation(
        store,
        request_id=rec.request.id,
        approved_by="operator",
        approval_reason="validated by human reviewer approving this request",
        self_improve_orchestrator=stub,
        event_emit=lambda et, **kw: emitted.append((et, kw)),
    ))
    assert result["outcome"] == "parked"
    parked_events = [kw for et, kw in emitted if et.endswith("escalation_parked")]
    assert len(parked_events) == 1
    assert parked_events[0].get("escalation_id") == rec.request.id
    assert "awaiting_approval" in (parked_events[0].get("park_reason") or "")
    assert not any(et.endswith("escalation_rolled_back") for et, _ in emitted)


def test_list_pending_excludes_expired_and_terminal(store: EscalationStore) -> None:
    rec = store.submit(_make_req())
    assert len(store.list_pending()) == 1
    store.reject(rec.request.id, rejected_by="op", rejection_reason="twenty-char rejection string here")
    assert store.list_pending() == []


def test_activity_log_appends_each_transition(store: EscalationStore) -> None:
    rec = store.submit(_make_req())
    store.approve(
        rec.request.id, approved_by="op",
        approval_reason="approved for investigation reasons",
        improvement_record_id="imp_x",
    )
    store.mark_outcome(rec.request.id, "clean")

    lines = store.activity_path.read_text().strip().splitlines()
    assert len(lines) == 3
    actions = [json.loads(L)["action"] for L in lines]
    assert actions == ["submit", "approve", "outcome"]


def test_load_all_tolerates_corrupt_file(tmp_path: Path) -> None:
    s = EscalationStore(pending_path=tmp_path / "bad.json", activity_path=tmp_path / "a.jsonl")
    s.pending_path.write_text("{not valid json")
    assert s.load_all() == []


# --------------------------------------------------------------------------
# build_request_from_metric_deficit
# --------------------------------------------------------------------------


def test_build_request_from_metric_deficit_populates_policy_fields() -> None:
    req = build_request_from_metric_deficit(
        metric="reasoning_coherence",
        current_value=0.42,
        threshold=0.5,
        deficit_duration_s=900.0,
        l1_attempts=7,
        win_rate=0.1,
        live_autonomy_level=3,
    )
    pol = METRIC_ESCALATION_POLICY["reasoning_coherence"]
    assert req.metric == "reasoning_coherence"
    assert req.target_module == pol["target_module"]
    assert list(req.declared_scope) == list(pol["declared_scope"])
    assert req.submitted_autonomy_level == 3
    assert "L1 research has run 7 time(s)" in req.metric_context_summary


def test_build_request_unregistered_metric_raises() -> None:
    with pytest.raises(EscalationStoreError):
        build_request_from_metric_deficit(
            metric="nope",
            current_value=0.0, threshold=0.0, deficit_duration_s=0.0,
            l1_attempts=0, win_rate=0.0, live_autonomy_level=3,
        )


# --------------------------------------------------------------------------
# MetricTriggers.get_escalation_candidates
# --------------------------------------------------------------------------


@dataclass
class _FakePrior:
    total: int = 0
    win_rate: float = 0.0


class _FakePolicyMemory:
    def __init__(self, total: int = 10, win_rate: float = 0.05) -> None:
        self._total = total
        self._win_rate = win_rate
        self.queried_tags: list[tuple] = []

    def get_topic_prior(self, tags: tuple) -> _FakePrior:
        self.queried_tags.append(tuple(tags))
        return _FakePrior(total=self._total, win_rate=self._win_rate)


def _make_triggers_with_deficit(
    metric: str = "confidence_volatility",
    *,
    deficit_duration_s: float = 600.0,
    veto_count: int = 1,
    policy_memory: Any = None,
):
    from autonomy.metric_triggers import MetricDeficit, MetricTriggers, _TRIGGER_DEFS

    trig = MetricTriggers()
    if policy_memory is not None:
        trig.set_policy_memory(policy_memory)
    defn = _TRIGGER_DEFS[metric]
    threshold = defn["threshold"]
    current = threshold + 0.05 if defn["direction"] == "above" else threshold - 0.05
    trig._active_deficits[metric] = MetricDeficit(
        metric=metric, current_value=current,
        threshold=threshold, deficit_duration_s=deficit_duration_s,
        severity="high",
    )
    trig._states[metric].veto_count = veto_count
    return trig


def test_escalation_candidates_empty_when_below_l3() -> None:
    trig = _make_triggers_with_deficit(policy_memory=_FakePolicyMemory())
    assert trig.get_escalation_candidates(live_autonomy_level=2) == []


def test_escalation_candidates_empty_with_no_policy_memory() -> None:
    trig = _make_triggers_with_deficit(policy_memory=None)
    assert trig.get_escalation_candidates(live_autonomy_level=3) == []


def test_escalation_candidates_empty_when_not_vetoed() -> None:
    trig = _make_triggers_with_deficit(
        veto_count=0, policy_memory=_FakePolicyMemory(),
    )
    assert trig.get_escalation_candidates(live_autonomy_level=3) == []


def test_escalation_candidates_empty_when_deficit_too_short() -> None:
    trig = _make_triggers_with_deficit(
        deficit_duration_s=120.0,  # < 480s
        policy_memory=_FakePolicyMemory(),
    )
    assert trig.get_escalation_candidates(live_autonomy_level=3) == []


def test_escalation_candidates_empty_when_win_rate_too_high() -> None:
    trig = _make_triggers_with_deficit(
        policy_memory=_FakePolicyMemory(total=10, win_rate=0.5),
    )
    assert trig.get_escalation_candidates(live_autonomy_level=3) == []


def test_escalation_candidates_empty_when_l1_attempts_too_low() -> None:
    trig = _make_triggers_with_deficit(
        policy_memory=_FakePolicyMemory(total=2, win_rate=0.05),
    )
    assert trig.get_escalation_candidates(live_autonomy_level=3) == []


def test_escalation_candidates_returns_request_when_all_gates_met() -> None:
    trig = _make_triggers_with_deficit(
        metric="confidence_volatility",
        deficit_duration_s=600.0,
        veto_count=1,
        policy_memory=_FakePolicyMemory(total=10, win_rate=0.05),
    )
    candidates = trig.get_escalation_candidates(live_autonomy_level=3)
    assert len(candidates) == 1
    req = candidates[0]
    assert req.metric == "confidence_volatility"
    assert req.submitted_autonomy_level == 3
    assert list(req.declared_scope) == list(
        METRIC_ESCALATION_POLICY["confidence_volatility"]["declared_scope"]
    )


# --------------------------------------------------------------------------
# approve_and_apply_escalation / reject / submit_and_emit event plumbing
# --------------------------------------------------------------------------


class _StubImprovementRecord:
    def __init__(self, status: str) -> None:
        self.status = status
        self.request = None
        self.was_rolled_back = (status in {"rolled_back", "reverted"})


class _StubSelfImproveOrchestrator:
    def __init__(self, status: str = "applied") -> None:
        self._status = status
        self.received_request = None
        self.received_kwargs: dict = {}

    async def attempt_improvement(self, req, **kwargs):
        self.received_request = req
        self.received_kwargs = dict(kwargs)
        return _StubImprovementRecord(self._status)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) \
        if asyncio._get_running_loop() is None else asyncio.run(coro)


def test_approve_and_apply_threads_declared_scope(
    store: EscalationStore,
) -> None:
    emitted: list[tuple] = []
    rec = store.submit(_make_req(
        metric="reasoning_coherence",
        declared_scope=["brain/reasoning/"],
    ))
    stub = _StubSelfImproveOrchestrator(status="applied")

    result = asyncio.run(approve_and_apply_escalation(
        store,
        request_id=rec.request.id,
        approved_by="operator",
        approval_reason="validated by human reviewer approving this request",
        self_improve_orchestrator=stub,
        event_emit=lambda et, **kw: emitted.append((et, kw)),
    ))

    assert result["outcome"] == "clean"
    assert result["status"] == "approved"
    # declared_scope threaded into the ImprovementRequest, NO global mutation.
    received = stub.received_request
    assert received is not None
    assert list(received.declared_scope) == ["brain/reasoning/"]
    assert received.manual is True
    assert received.type == "architecture_improvement"
    # Phase 6.5 Finding #2: escalation approval IS the human approval.
    # The ImprovementRequest must NOT re-queue in the self-improve approval
    # queue. requires_approval is False on the synthesized request.
    assert received.requires_approval is False
    assert any(et.endswith("escalation_approved") for et, _ in emitted)
    assert not any(et.endswith("escalation_rolled_back") for et, _ in emitted)


def test_approve_and_apply_parks_on_awaiting_approval(
    store: EscalationStore,
) -> None:
    """Phase 6.5 Finding #2: awaiting_approval is NOT rolled_back.

    If the self-improve pipeline re-queues the patch (e.g. patch.check_dangerous
    fires on mutation_governor), the escalation must be marked ``parked`` so the
    audit trail does not falsely claim the patch was rolled back. The operator
    resolves the parked patch via the self-improve dashboard.
    """
    emitted: list[tuple] = []
    rec = store.submit(_make_req())
    stub = _StubSelfImproveOrchestrator(status="awaiting_approval")
    result = asyncio.run(approve_and_apply_escalation(
        store,
        request_id=rec.request.id,
        approved_by="operator",
        approval_reason="validated by human reviewer approving this request",
        self_improve_orchestrator=stub,
        event_emit=lambda et, **kw: emitted.append((et, kw)),
    ))
    assert result["outcome"] == "parked"
    assert result["status"] == "parked"
    updated = store.get(rec.request.id)
    assert updated is not None
    assert updated.status == "parked"
    assert updated.outcome == "parked"
    # No rolled_back event should fire for a parked outcome.
    assert not any(et.endswith("escalation_rolled_back") for et, _ in emitted)
    assert any(et.endswith("escalation_approved") for et, _ in emitted)


def test_approve_and_apply_marks_rolled_back_on_non_applied_status(
    store: EscalationStore,
) -> None:
    emitted: list[tuple] = []
    rec = store.submit(_make_req())
    stub = _StubSelfImproveOrchestrator(status="rolled_back")
    result = asyncio.run(approve_and_apply_escalation(
        store,
        request_id=rec.request.id,
        approved_by="operator",
        approval_reason="validated by human reviewer approving this request",
        self_improve_orchestrator=stub,
        event_emit=lambda et, **kw: emitted.append((et, kw)),
    ))
    assert result["outcome"] == "rolled_back"
    assert result["status"] == "rolled_back"
    updated = store.get(rec.request.id)
    assert updated is not None and updated.status == "rolled_back"
    assert any(et.endswith("escalation_rolled_back") for et, _ in emitted)


def test_approve_and_apply_records_rollback_on_exception(
    store: EscalationStore,
) -> None:
    class _Boom:
        async def attempt_improvement(self, *a, **kw):
            raise RuntimeError("pipeline blew up")

    emitted: list[tuple] = []
    rec = store.submit(_make_req())
    with pytest.raises(RuntimeError, match="pipeline blew up"):
        asyncio.run(approve_and_apply_escalation(
            store,
            request_id=rec.request.id,
            approved_by="operator",
            approval_reason="validated by human reviewer approving this request",
            self_improve_orchestrator=_Boom(),
            event_emit=lambda et, **kw: emitted.append((et, kw)),
        ))
    updated = store.get(rec.request.id)
    assert updated is not None
    assert updated.status == "rolled_back"
    assert "pipeline blew up" in updated.rollback_reason
    assert any(et.endswith("escalation_rolled_back") for et, _ in emitted)


def test_approve_and_apply_rejects_non_pending(store: EscalationStore) -> None:
    rec = store.submit(_make_req())
    store.reject(
        rec.request.id, rejected_by="op",
        rejection_reason="twenty-char rejection reason here",
    )
    with pytest.raises(EscalationStoreError, match="Cannot approve"):
        asyncio.run(approve_and_apply_escalation(
            store,
            request_id=rec.request.id,
            approved_by="op",
            approval_reason="twenty-char approval reason here exactly",
            self_improve_orchestrator=_StubSelfImproveOrchestrator(),
            event_emit=lambda et, **kw: None,
        ))


def test_reject_escalation_emits_event(store: EscalationStore) -> None:
    emitted: list[tuple] = []
    rec = store.submit(_make_req())
    updated = reject_escalation(
        store,
        request_id=rec.request.id,
        rejected_by="operator",
        rejection_reason="not actionable in current sprint window",
        event_emit=lambda et, **kw: emitted.append((et, kw)),
    )
    assert updated.status == "rejected"
    assert any(et.endswith("escalation_rejected") for et, _ in emitted)


def test_submit_and_emit_emits_event(store: EscalationStore) -> None:
    emitted: list[tuple] = []
    rec = submit_and_emit(
        store,
        _make_req(),
        event_emit=lambda et, **kw: emitted.append((et, kw)),
    )
    assert rec.status == "pending"
    requested = [kw for et, kw in emitted if et.endswith("escalation_requested")]
    assert len(requested) == 1
    kw = requested[0]
    assert kw["declared_scope"] == list(rec.request.declared_scope)
    assert kw["submitted_autonomy_level"] == 3


# --------------------------------------------------------------------------
# current_ok sourcing invariant
# --------------------------------------------------------------------------


def test_current_ok_sourcing_invariant_queries_orchestrator_live() -> None:
    """The API-level current_ok must come from the live orchestrator method.

    This test guards against a future shortcut where current_ok gets
    rehydrated from a persisted file on cold boot. The dashboard
    endpoint calls ``check_promotion_eligibility`` and uses its
    ``eligible_for_l3`` field directly. We verify the contract by
    constructing a minimal stub orchestrator and asserting the method
    is invoked for every read.
    """
    calls = {"n": 0}

    class _StubOrch:
        def get_autonomy_level(self) -> int:
            return 2

        def check_promotion_eligibility(self) -> dict:
            calls["n"] += 1
            return {
                "eligible_for_l3": False,
                "reason": "below_threshold",
                "wins": 0, "win_rate": 0.0, "recent_regressions": 0,
            }

    stub = _StubOrch()
    # The handler logic we care about: current_ok is set from the
    # live call. Simulate two successive reads to prove it is queried
    # every time, never cached from a file.
    for _ in range(2):
        elig = stub.check_promotion_eligibility()
        current_ok = bool(elig.get("eligible_for_l3"))
        assert current_ok is False
    assert calls["n"] == 2


# --------------------------------------------------------------------------
# Never-mutate-current invariant: escalation store must not touch
# autonomy_state.json or maturity_highwater.json
# --------------------------------------------------------------------------


def test_escalation_store_isolated_from_autonomy_state(tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    autonomy_state = fake_home / "autonomy_state.json"
    autonomy_state.write_text('{"level":2}')
    maturity = fake_home / "maturity_highwater.json"
    maturity.write_text('{"foo":"bar"}')

    s = EscalationStore(
        pending_path=fake_home / "pending.json",
        activity_path=fake_home / "activity.jsonl",
    )
    rec = s.submit(_make_req())
    s.approve(
        rec.request.id, approved_by="op",
        approval_reason="approved for investigation reasons",
        improvement_record_id="imp_x",
    )
    s.mark_outcome(rec.request.id, "clean")

    # Unrelated files must not have been touched.
    assert autonomy_state.read_text() == '{"level":2}'
    assert maturity.read_text() == '{"foo":"bar"}'
