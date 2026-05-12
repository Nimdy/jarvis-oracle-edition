"""Phase 6.5 regression guard: L3 is never auto-promoted.

The Pillar 10 / Pillar 8 invariant is that transitions into autonomy L3
must be explicitly operator-driven. The `_check_and_apply_promotion()`
loop may observe live eligibility and must announce it via
`AUTONOMY_L3_ELIGIBLE` exactly once per session, but it must never call
`set_autonomy_level(3)`. Manual promotion through the API must always
supply an `evidence_path`.

These tests make the invariant machine-enforced so a future refactor
cannot silently re-enable auto-promotion.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

MIN_MEANINGFUL_DELTA = 0.02


@dataclass
class FakePolicyOutcome:
    net_delta: float = 0.05
    worked: bool = True
    warmup: bool = False
    intent_id: str = ""
    intent_type: str = ""
    tool_used: str = ""
    topic_tags: tuple[str, ...] = ()
    question_summary: str = ""
    stable: bool = True
    confidence: float = 0.5
    cost_tokens: int = 0
    cost_seconds: float = 0.0
    risk_score: float = 0.0
    timestamp: float = field(default_factory=time.time)


class FakePolicyMemory:
    """Stub AutonomyPolicyMemory with controllable stats.

    Produces >= L3 gate conditions (25+ wins, 50%+ win rate, 0 regressions
    in last 10) so the promotion check always considers L3 eligible.
    """

    def __init__(
        self,
        outcomes: list[FakePolicyOutcome] | None = None,
        wins: int = 100,
        total: int = 101,
        win_rate: float = 0.99,
    ) -> None:
        self._outcomes = outcomes or [
            FakePolicyOutcome(net_delta=0.1, worked=True) for _ in range(10)
        ]
        self._wins = wins
        self._total = total
        self._win_rate = win_rate

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_outcomes": self._total,
            "total_wins": self._wins,
            "overall_win_rate": self._win_rate,
            "total_losses": self._total - self._wins,
            "warmup_outcomes": 0,
            "in_warmup": False,
            "warmup_remaining_s": 0.0,
            "avoid_patterns": [],
            "unique_tools": [],
            "unique_types": [],
        }


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _build_orchestrator_at_l2(tmp_dir: Path):
    """Build an orchestrator bypassing __init__, pre-seeded at L2.

    Mirrors the pattern in test_autonomy_reconcile.py. The orchestrator
    is ready to have `_check_and_apply_promotion()` called against it.
    """
    state_path = tmp_dir / "autonomy_state.json"

    with mock.patch("autonomy.orchestrator._AUTONOMY_STATE_PATH", state_path), \
         mock.patch("autonomy.orchestrator._JARVIS_DIR", tmp_dir):
        from autonomy.orchestrator import AutonomyOrchestrator

        orch = AutonomyOrchestrator.__new__(AutonomyOrchestrator)

        orch._detector = mock.MagicMock()
        orch._governor = mock.MagicMock()
        orch._query_interface = mock.MagicMock()
        orch._integrator = mock.MagicMock()
        orch._scorer = mock.MagicMock()
        orch._metric_triggers = mock.MagicMock()
        orch._metric_history = mock.MagicMock()
        orch._delta_tracker = mock.MagicMock()
        orch._delta_tracker.get_stats.return_value = {}
        orch._policy_memory = FakePolicyMemory()
        orch._calibrator = mock.MagicMock()
        orch._episode_recorder = mock.MagicMock()
        orch._bridge = mock.MagicMock()
        orch._drive_manager = None
        orch._queue = []
        orch._completed = []
        orch._intent_metadata = {}
        orch._intent_ledger_ids = {}
        orch._metadata_prune_counter = 0
        orch._last_process_time = 0.0
        orch._last_metrics_feed_time = 0.0
        orch._last_drive_eval_time = 0.0
        orch._saturated_topics = set()
        orch._topic_recall_misses = {}
        orch._last_saturation_clear = time.time()
        orch._enabled = True
        orch._started = False
        orch._engine_ref = None
        # Pre-seeded L2 state, warmup bypassed via _level_restored_from_disk.
        orch._autonomy_level = 2
        orch._level_restored_from_disk = True
        orch._persisted_autonomy_data = None
        orch._current_mode = ""
        orch._goal_callback = None
        orch._goal_manager = None
        orch._last_promotion_check = 0.0
        orch._last_eval_replay_time = 0.0
        orch._boot_time = time.time()
        orch._l3_eligibility_announced = False
        orch._escalation_store = None
        orch._escalation_wire_last_error_log_ts = 0.0

    return orch


class _EventRecorder:
    """Captures events emitted through orchestrator._emit_event."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event_type: str, **kwargs: Any) -> None:
        self.events.append((event_type, kwargs))

    def count(self, event_type: str) -> int:
        return sum(1 for t, _ in self.events if t == event_type)

    def only(self, event_type: str) -> list[dict[str, Any]]:
        return [payload for t, payload in self.events if t == event_type]


def test_no_auto_promote_to_l3_ever(tmp_dir):
    """100 ticks with 99% win rate and zero regressions cannot reach L3.

    This is the core Pillar 10 invariant: live-runtime eligibility is a
    fact, promotion is an act. The act requires manual approval.
    """
    orch = _build_orchestrator_at_l2(tmp_dir)
    recorder = _EventRecorder()

    with mock.patch.object(orch, "_emit_event", recorder.emit):
        for _ in range(100):
            orch._check_and_apply_promotion()

    assert orch._autonomy_level == 2, (
        "L3 auto-promotion must not occur under any live-runtime conditions. "
        "Autonomy level changed from 2 after 100 promotion ticks."
    )
    assert recorder.count("autonomy:level_changed") == 0, (
        "Level-changed event must not fire when the invariant holds."
    )


def test_l3_eligibility_announced_exactly_once(tmp_dir):
    """AUTONOMY_L3_ELIGIBLE must fire once per session, not per tick."""
    orch = _build_orchestrator_at_l2(tmp_dir)
    recorder = _EventRecorder()

    with mock.patch.object(orch, "_emit_event", recorder.emit):
        for _ in range(100):
            orch._check_and_apply_promotion()

    assert recorder.count("autonomy:l3_eligible") == 1, (
        "AUTONOMY_L3_ELIGIBLE must fire exactly once per session "
        f"(observed {recorder.count('autonomy:l3_eligible')} times)"
    )


def test_l3_eligibility_event_payload_shape(tmp_dir):
    """AUTONOMY_L3_ELIGIBLE payload must carry live-runtime fields only."""
    orch = _build_orchestrator_at_l2(tmp_dir)
    recorder = _EventRecorder()

    with mock.patch.object(orch, "_emit_event", recorder.emit):
        orch._check_and_apply_promotion()

    payloads = recorder.only("autonomy:l3_eligible")
    assert len(payloads) == 1
    payload = payloads[0]
    assert "reason" in payload
    assert "wins" in payload
    assert "win_rate" in payload
    assert "regressions_in_last_10" in payload
    assert payload["wins"] == 100
    assert payload["win_rate"] == pytest.approx(0.99, abs=0.01)
    assert payload["regressions_in_last_10"] == 0


def test_set_autonomy_level_3_without_evidence_path_refuses(tmp_dir):
    """Internal callers cannot reach L3; explicit evidence_path is required."""
    orch = _build_orchestrator_at_l2(tmp_dir)

    with pytest.raises(PermissionError, match="requires explicit evidence_path"):
        orch.set_autonomy_level(3)

    assert orch._autonomy_level == 2


def test_set_autonomy_level_3_with_current_eligible_emits_promoted(tmp_dir):
    """Manual promotion with a valid evidence_path transitions to L3 cleanly."""
    orch = _build_orchestrator_at_l2(tmp_dir)
    recorder = _EventRecorder()

    with mock.patch.object(orch, "_emit_event", recorder.emit):
        orch.set_autonomy_level(
            3,
            evidence_path="current_eligible",
            approval_source="test",
            caller_id="pytest",
        )

    assert orch._autonomy_level == 3

    promoted = recorder.only("autonomy:l3_promoted")
    assert len(promoted) == 1, "AUTONOMY_L3_PROMOTED must fire exactly once"
    payload = promoted[0]
    assert payload["evidence_path"] == "current_eligible"
    assert payload["approval_source"] == "test"
    assert payload["caller_id"] == "pytest"
    assert payload["outcome"] == "clean"
    assert payload["prior_level"] == 2
    assert "promoted_at" in payload


def test_set_autonomy_level_3_no_event_when_refused(tmp_dir):
    """A refused promotion must not emit AUTONOMY_L3_PROMOTED.

    Regression guard for the plan's invariant that AUTONOMY_L3_PROMOTED
    is the single authoritative record that L3 became active. Refusal
    is not promotion.
    """
    orch = _build_orchestrator_at_l2(tmp_dir)
    recorder = _EventRecorder()

    with mock.patch.object(orch, "_emit_event", recorder.emit):
        with pytest.raises(PermissionError):
            orch.set_autonomy_level(3)

    assert recorder.count("autonomy:l3_promoted") == 0
    assert recorder.count("autonomy:level_changed") == 0


def test_set_autonomy_level_3_refusal_emits_activation_denied(tmp_dir):
    """Phase 6.5 Finding #4: every L3 denial must be observable via event.

    A refused L3 promotion (no evidence_path) emits
    ``AUTONOMY_L3_ACTIVATION_DENIED`` with reason="missing_evidence_path"
    and the current_level so downstream audit consumers (dashboard,
    consciousness replay) can render the denial without re-reading logs.
    """
    orch = _build_orchestrator_at_l2(tmp_dir)
    recorder = _EventRecorder()

    with mock.patch.object(orch, "_emit_event", recorder.emit):
        with pytest.raises(PermissionError):
            orch.set_autonomy_level(
                3, approval_source="test", caller_id="pytest",
            )

    denied = recorder.only("autonomy:l3_activation_denied")
    assert len(denied) == 1, (
        "AUTONOMY_L3_ACTIVATION_DENIED must fire exactly once per refusal"
    )
    payload = denied[0]
    assert payload["reason"] == "missing_evidence_path"
    assert payload["caller_id"] == "pytest"
    assert payload["approval_source"] == "test"
    assert payload["current_level"] == 2
    assert "denied_at" in payload


def test_set_autonomy_level_2_unchanged_behavior(tmp_dir):
    """L2 transitions must not be affected by the L3 invariant.

    Regression guard: promotion to L2 continues to fire level_changed
    without requiring evidence_path.
    """
    orch = _build_orchestrator_at_l2(tmp_dir)
    orch._autonomy_level = 1
    recorder = _EventRecorder()

    with mock.patch.object(orch, "_emit_event", recorder.emit):
        orch.set_autonomy_level(2)

    assert orch._autonomy_level == 2
    assert recorder.count("autonomy:level_changed") == 1
    assert recorder.count("autonomy:l3_promoted") == 0


def test_l3_eligibility_not_announced_when_level_already_3(tmp_dir):
    """Once L3 is active, the promotion check should short-circuit.

    The eligibility event is a "you just earned this" signal; it must not
    fire while already at L3.
    """
    orch = _build_orchestrator_at_l2(tmp_dir)
    orch._autonomy_level = 3
    recorder = _EventRecorder()

    with mock.patch.object(orch, "_emit_event", recorder.emit):
        for _ in range(10):
            orch._check_and_apply_promotion()

    assert recorder.count("autonomy:l3_eligible") == 0


# --------------------------------------------------------------------------
# Phase 6.5 Finding #1: escalation wire-up in the autonomy tick
# --------------------------------------------------------------------------


def test_escalation_wire_does_not_fire_below_l3(tmp_dir):
    """At L<3 the escalation trigger must not be consulted at all."""
    orch = _build_orchestrator_at_l2(tmp_dir)
    orch._autonomy_level = 2
    orch._metric_triggers.get_escalation_candidates = mock.MagicMock(return_value=[])

    orch._evaluate_l3_escalations()

    orch._metric_triggers.get_escalation_candidates.assert_not_called()


def test_escalation_wire_consults_trigger_at_l3(tmp_dir):
    """At L3 the tick must call get_escalation_candidates exactly once."""
    orch = _build_orchestrator_at_l2(tmp_dir)
    orch._autonomy_level = 3
    orch._metric_triggers.get_escalation_candidates = mock.MagicMock(return_value=[])

    orch._evaluate_l3_escalations()

    orch._metric_triggers.get_escalation_candidates.assert_called_once_with(
        live_autonomy_level=3,
    )


def test_escalation_wire_submits_candidates_and_emits_requested(tmp_dir):
    """Returned candidates must be routed through submit_and_emit.

    The wire is the missing seam from Finding #1: get_escalation_candidates
    existed, EscalationStore.submit existed, but nothing called them
    together. This test locks in that connection.
    """
    from autonomy.escalation import EscalationRequest, METRIC_ESCALATION_POLICY

    orch = _build_orchestrator_at_l2(tmp_dir)
    orch._autonomy_level = 3

    pol = METRIC_ESCALATION_POLICY["confidence_volatility"]
    fake_req = EscalationRequest(
        metric="confidence_volatility",
        metric_context_summary="deficit exceeded 600s with veto",
        severity="high",
        target_module=pol["target_module"],
        declared_scope=list(pol["declared_scope"]),
        submitted_autonomy_level=3,
    )
    orch._metric_triggers.get_escalation_candidates = mock.MagicMock(
        return_value=[fake_req],
    )

    # Use a tmp-scoped EscalationStore so we don't touch ~/.jarvis.
    from autonomy.escalation import EscalationStore
    store = EscalationStore(
        pending_path=tmp_dir / "pending_escalations.json",
        activity_path=tmp_dir / "escalation_activity.jsonl",
    )
    orch._escalation_store = store

    recorder = _EventRecorder()
    with mock.patch.object(orch, "_emit_event", recorder.emit):
        orch._evaluate_l3_escalations()

    # The submit call landed in the store and is observable as pending.
    pending = store.list_pending()
    assert len(pending) == 1
    assert pending[0].request.metric == "confidence_volatility"
    assert list(pending[0].request.declared_scope) == list(pol["declared_scope"])
    # submit_and_emit fired the requested event through orch._emit_event.
    requested = recorder.only("autonomy:escalation_requested")
    assert len(requested) == 1
    assert requested[0]["metric"] == "confidence_volatility"


def test_escalation_wire_swallows_trigger_exceptions(tmp_dir):
    """A raising trigger must not crash the autonomy tick."""
    orch = _build_orchestrator_at_l2(tmp_dir)
    orch._autonomy_level = 3
    orch._metric_triggers.get_escalation_candidates = mock.MagicMock(
        side_effect=RuntimeError("trigger blew up"),
    )

    orch._evaluate_l3_escalations()

    assert orch._escalation_wire_last_error_log_ts > 0.0
