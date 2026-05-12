"""Tests for gated autonomy auto-restore on boot (reconcile_on_boot).

Verifies: L2 auto-restore, L3 warn-only, 5 safety gates,
structured report shape, malformed input fail-closed behavior.
"""

from __future__ import annotations

import json
import os
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

    def to_dict(self) -> dict[str, Any]:
        return {"net_delta": self.net_delta, "worked": self.worked}

    @classmethod
    def from_dict(cls, d: dict) -> "FakePolicyOutcome":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class FakePolicyMemory:
    """Minimal stub for AutonomyPolicyMemory."""

    def __init__(self, outcomes: list[FakePolicyOutcome] | None = None,
                 wins: int = 20, total: int = 30, win_rate: float = 0.85):
        self._outcomes = outcomes or []
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


class FakeQuarantinePressureState:
    def __init__(self, high: bool = False):
        self.high = high
        self.elevated = high
        self.composite = 0.8 if high else 0.0


class FakeQuarantinePressure:
    def __init__(self, high: bool = False):
        self.current = FakeQuarantinePressureState(high=high)


class FakeContradictionEngine:
    def __init__(self, debt: float = 0.05):
        self._debt = debt

    @property
    def contradiction_debt(self) -> float:
        return self._debt


def _write_persisted_state(path: Path, level: int = 2,
                            promoted_at: float | None = None,
                            win_rate: float = 0.85, outcomes: int = 30) -> None:
    state = {
        "autonomy_level": level,
        "promoted_at": promoted_at or time.time() - 300,
        "restored_from_policy_win_rate": win_rate,
        "restored_from_policy_outcomes": outcomes,
        "restored_from_delta_improvement_rate": 0.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def _build_orchestrator(
    tmp_dir: Path,
    persisted_level: int | None = None,
    policy_wins: int = 20,
    policy_total: int = 30,
    policy_win_rate: float = 0.85,
    outcomes: list[FakePolicyOutcome] | None = None,
    quarantine_high: bool = False,
    contradiction_debt: float = 0.05,
    promoted_at: float | None = None,
    persisted_win_rate: float | None = None,
    persisted_outcomes: int | None = None,
):
    """Build an AutonomyOrchestrator with controlled dependencies."""
    state_path = tmp_dir / "autonomy_state.json"

    if persisted_level is not None:
        _write_persisted_state(
            state_path, level=persisted_level,
            promoted_at=promoted_at,
            win_rate=persisted_win_rate if persisted_win_rate is not None else policy_win_rate,
            outcomes=persisted_outcomes if persisted_outcomes is not None else policy_total,
        )

    if outcomes is None:
        outcomes = [FakePolicyOutcome(net_delta=0.05, worked=True) for _ in range(5)]

    fake_policy = FakePolicyMemory(
        outcomes=outcomes, wins=policy_wins,
        total=policy_total, win_rate=policy_win_rate,
    )
    fake_quarantine = FakeQuarantinePressure(high=quarantine_high)
    fake_contradiction = FakeContradictionEngine(debt=contradiction_debt)

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
        orch._policy_memory = fake_policy
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
        orch._autonomy_level = 1
        orch._level_restored_from_disk = False
        orch._persisted_autonomy_data = None
        orch._current_mode = ""
        orch._goal_callback = None
        orch._goal_manager = None
        orch._last_promotion_check = 0.0
        orch._last_eval_replay_time = 0.0
        orch._boot_time = time.time()

        orch._restore_autonomy_level(config_level=1)

    def mock_quarantine_pressure():
        return fake_quarantine

    def mock_contradiction_instance():
        return fake_contradiction

    return orch, state_path, mock_quarantine_pressure, mock_contradiction_instance


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestGatedAutoRestore:
    """Tests for reconcile_on_boot() gated auto-restore."""

    def _reconcile(self, orch, state_path, mock_qp, mock_ce):
        with mock.patch("autonomy.orchestrator._AUTONOMY_STATE_PATH", state_path), \
             mock.patch(
                 "autonomy.orchestrator.AutonomyOrchestrator._check_quarantine_high",
                 side_effect=lambda: mock_qp().current.high,
             ), \
             mock.patch(
                 "autonomy.orchestrator.AutonomyOrchestrator._check_contradiction_debt",
                 side_effect=lambda: mock_ce().contradiction_debt,
             ):
            return orch.reconcile_on_boot()

    def test_l2_happy_path_restores(self, tmp_dir):
        """Persisted L2, all gates pass -> auto-restores to L2."""
        orch, sp, qp, ce = _build_orchestrator(tmp_dir, persisted_level=2)

        report = self._reconcile(orch, sp, qp, ce)

        assert report["auto_restored"] is True
        assert report["restored_level"] == 2
        assert report["current_level_after"] == 2
        assert orch._autonomy_level == 2
        assert orch._level_restored_from_disk is True
        assert not report["vetoed_by"]
        assert report["gate_results"]["persisted_state_valid"] is True
        assert report["gate_results"]["policy_qualifies"] is True
        assert report["gate_results"]["regression_clear"] is True
        assert report["gate_results"]["quarantine_clear"] is True
        assert report["gate_results"]["debt_clear"] is True

    def test_l3_happy_path_does_not_auto_restore(self, tmp_dir):
        """Persisted L3, all gates pass -> stays at L1, logs eligible-but-manual."""
        outcomes = [FakePolicyOutcome(net_delta=0.05, worked=True) for _ in range(5)]
        orch, sp, qp, ce = _build_orchestrator(
            tmp_dir, persisted_level=3,
            policy_wins=35, policy_total=50, policy_win_rate=0.70,
            outcomes=outcomes,
        )

        report = self._reconcile(orch, sp, qp, ce)

        assert report["auto_restored"] is False
        assert report["l3_eligible_but_manual"] is True
        assert orch._autonomy_level == 1
        assert report["current_level_after"] == 1

    def test_veto_policy_no_longer_qualifies(self, tmp_dir):
        """Persisted L2 but policy memory doesn't qualify -> vetoed."""
        orch, sp, qp, ce = _build_orchestrator(
            tmp_dir, persisted_level=2,
            policy_wins=3, policy_total=10, policy_win_rate=0.30,
        )

        report = self._reconcile(orch, sp, qp, ce)

        assert report["auto_restored"] is False
        assert orch._autonomy_level == 1
        assert "policy_no_longer_qualifies" in report["vetoed_by"]
        assert report["gate_results"]["policy_qualifies"] is False

    def test_veto_two_regressions(self, tmp_dir):
        """Persisted L2, 2+ regressions in last 5 -> vetoed."""
        outcomes = [
            FakePolicyOutcome(net_delta=0.05, worked=True),
            FakePolicyOutcome(net_delta=-0.05, worked=False),
            FakePolicyOutcome(net_delta=0.03, worked=True),
            FakePolicyOutcome(net_delta=-0.04, worked=False),
            FakePolicyOutcome(net_delta=0.02, worked=True),
        ]
        orch, sp, qp, ce = _build_orchestrator(
            tmp_dir, persisted_level=2, outcomes=outcomes,
        )

        report = self._reconcile(orch, sp, qp, ce)

        assert report["auto_restored"] is False
        assert orch._autonomy_level == 1
        assert report["recent_regression_count"] == 2
        any_regression_veto = any("recent_regressions" in v for v in report["vetoed_by"])
        assert any_regression_veto
        assert report["gate_results"]["regression_clear"] is False

    def test_veto_strong_recent_regression(self, tmp_dir):
        """Most recent outcome is a strong regression -> vetoed."""
        strong_delta = -(2.5 * MIN_MEANINGFUL_DELTA)
        outcomes = [
            FakePolicyOutcome(net_delta=0.05, worked=True),
            FakePolicyOutcome(net_delta=0.05, worked=True),
            FakePolicyOutcome(net_delta=0.05, worked=True),
            FakePolicyOutcome(net_delta=0.05, worked=True),
            FakePolicyOutcome(net_delta=strong_delta, worked=False),
        ]
        orch, sp, qp, ce = _build_orchestrator(
            tmp_dir, persisted_level=2, outcomes=outcomes,
        )

        report = self._reconcile(orch, sp, qp, ce)

        assert report["auto_restored"] is False
        assert orch._autonomy_level == 1
        any_regression_veto = any("recent_regressions" in v for v in report["vetoed_by"])
        assert any_regression_veto

    def test_single_mild_regression_does_not_veto(self, tmp_dir):
        """One mild regression in last 5 -> still restores L2."""
        outcomes = [
            FakePolicyOutcome(net_delta=0.05, worked=True),
            FakePolicyOutcome(net_delta=0.05, worked=True),
            FakePolicyOutcome(net_delta=-0.03, worked=False),
            FakePolicyOutcome(net_delta=0.05, worked=True),
            FakePolicyOutcome(net_delta=0.05, worked=True),
        ]
        orch, sp, qp, ce = _build_orchestrator(
            tmp_dir, persisted_level=2, outcomes=outcomes,
        )

        report = self._reconcile(orch, sp, qp, ce)

        assert report["auto_restored"] is True
        assert orch._autonomy_level == 2
        assert report["recent_regression_count"] == 1
        assert report["gate_results"]["regression_clear"] is True

    def test_veto_quarantine_high(self, tmp_dir):
        """Quarantine pressure high -> vetoed."""
        orch, sp, qp, ce = _build_orchestrator(
            tmp_dir, persisted_level=2, quarantine_high=True,
        )

        report = self._reconcile(orch, sp, qp, ce)

        assert report["auto_restored"] is False
        assert orch._autonomy_level == 1
        assert "quarantine_pressure_high" in report["vetoed_by"]
        assert report["quarantine_high"] is True
        assert report["gate_results"]["quarantine_clear"] is False

    def test_veto_contradiction_debt_high(self, tmp_dir):
        """Contradiction debt above threshold -> vetoed."""
        orch, sp, qp, ce = _build_orchestrator(
            tmp_dir, persisted_level=2, contradiction_debt=0.25,
        )

        report = self._reconcile(orch, sp, qp, ce)

        assert report["auto_restored"] is False
        assert orch._autonomy_level == 1
        any_debt_veto = any("contradiction_debt_high" in v for v in report["vetoed_by"])
        assert any_debt_veto
        assert report["gate_results"]["debt_clear"] is False

    def test_no_persisted_state(self, tmp_dir):
        """No persisted file -> no restore, no warning, clean report."""
        orch, sp, qp, ce = _build_orchestrator(tmp_dir, persisted_level=None)

        report = self._reconcile(orch, sp, qp, ce)

        assert report["auto_restored"] is False
        assert report["requested_level"] is None
        assert report["persisted_level"] is None
        assert orch._autonomy_level == 1
        assert not report["vetoed_by"]

    def test_malformed_json_fails_closed(self, tmp_dir):
        """Corrupt autonomy_state.json -> fail closed, no crash, no restore."""
        state_path = tmp_dir / "autonomy_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{ corrupt json !!!")

        orch, _, qp, ce = _build_orchestrator(tmp_dir, persisted_level=None)

        with mock.patch("autonomy.orchestrator._AUTONOMY_STATE_PATH", state_path):
            orch._restore_autonomy_level(config_level=1)

        assert orch._persisted_autonomy_data is None

        report = self._reconcile(orch, state_path, qp, ce)

        assert report["auto_restored"] is False
        assert orch._autonomy_level == 1

    def test_report_includes_persisted_age(self, tmp_dir):
        """Report includes persisted_state_age_s computed from promoted_at."""
        promoted_at = time.time() - 600
        orch, sp, qp, ce = _build_orchestrator(
            tmp_dir, persisted_level=2, promoted_at=promoted_at,
        )

        report = self._reconcile(orch, sp, qp, ce)

        assert report["persisted_state_age_s"] is not None
        assert report["persisted_state_age_s"] >= 590

    def test_report_snapshot_matches_policy(self, tmp_dir):
        """Snapshot match is True when persisted and current policy are close."""
        orch, sp, qp, ce = _build_orchestrator(
            tmp_dir, persisted_level=2,
            policy_wins=20, policy_total=30, policy_win_rate=0.85,
            persisted_win_rate=0.84, persisted_outcomes=28,
        )

        report = self._reconcile(orch, sp, qp, ce)

        assert report["persisted_snapshot_matches_policy"] is True

    def test_report_snapshot_mismatch_when_diverged(self, tmp_dir):
        """Snapshot match is False when persisted and current policy diverge."""
        orch, sp, qp, ce = _build_orchestrator(
            tmp_dir, persisted_level=2,
            policy_wins=20, policy_total=30, policy_win_rate=0.85,
            persisted_win_rate=0.50, persisted_outcomes=10,
        )

        report = self._reconcile(orch, sp, qp, ce)

        assert report["persisted_snapshot_matches_policy"] is False

    def test_report_structure_complete(self, tmp_dir):
        """All expected keys are present in the report."""
        orch, sp, qp, ce = _build_orchestrator(tmp_dir, persisted_level=2)

        report = self._reconcile(orch, sp, qp, ce)

        expected_keys = {
            "auto_restored", "restored_level", "requested_level",
            "vetoed_by", "l3_eligible_but_manual", "gate_results",
            "current_level_before", "current_level_after",
            "persisted_level", "policy_eligible_level",
            "persisted_state_age_s", "persisted_snapshot_matches_policy",
            "recent_regression_count", "contradiction_debt",
            "quarantine_high", "disagreements",
        }
        assert expected_keys.issubset(set(report.keys()))

        gate_keys = {
            "persisted_state_valid", "policy_qualifies",
            "regression_clear", "quarantine_clear", "debt_clear",
        }
        assert gate_keys == set(report["gate_results"].keys())
