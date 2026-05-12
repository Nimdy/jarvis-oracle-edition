"""Tests for Phase 6.5 snapshot caches.

Covers:
- ``_build_l3_escalation_cache`` produces a safe-default when no engine
  / orchestrator is attached.
- ``current_ok`` is sourced strictly from the live orchestrator's
  :meth:`check_promotion_eligibility` — never backfilled from a
  persisted file. This is the Pillar 10 "current-ok live-only" guard.
- ``prior_attested_ok`` is sourced from the attestation ledger and is
  a distinct field; it is never folded into the live ``current_ok``.
- Attestation cache loads records with correct
  ``artifact_status`` / ``attestation_strength`` and aggregates a
  top-level strength for the UI.
- Empty ledgers return empty-but-well-formed structures.

These tests complete the "three-field separation + attestation boundary
+ no-backfill regression guard" part of the Phase A checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from dashboard.snapshot import (
    _build_attestation_cache,
    _build_l3_escalation_cache,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


class _StubOrchestrator:
    """Minimal orchestrator stub matching the two fields the cache reads."""

    def __init__(
        self,
        *,
        autonomy_level: int,
        eligible_for_l3: bool,
        wins: int = 0,
        win_rate: float = 0.0,
        recent_regressions: int = 0,
        l3_reason: str = "",
    ) -> None:
        self.autonomy_level = autonomy_level
        self._eligible = eligible_for_l3
        self._wins = wins
        self._win_rate = win_rate
        self._recent_regressions = recent_regressions
        self._reason = l3_reason

    def check_promotion_eligibility(self) -> dict:
        return {
            "current_level": self.autonomy_level,
            "eligible_for_l3": self._eligible,
            "wins": self._wins,
            "win_rate": self._win_rate,
            "recent_regressions": self._recent_regressions,
            "l3_reason": self._reason,
        }


class _StubEngine:
    def __init__(self, orch: _StubOrchestrator | None) -> None:
        self._autonomy_orchestrator = orch


def _seed_verified_attestation(ledger_path: Path, artifact_path: Path) -> None:
    """Write a hash-verified autonomy.l3 record directly as JSON."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_body = b"proof report body"
    artifact_path.write_bytes(artifact_body)
    report_hash = "sha256:" + hashlib.sha256(artifact_body).hexdigest()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps([
            {
                "capability_id": "autonomy.l3",
                "evidence_source": str(artifact_path),
                "evidence_window_start": "2026-03-21T00:00:00Z",
                "evidence_window_end": "2026-04-17T07:00:00Z",
                "measured_values": {
                    "oracle_composite": 95.1,
                    "autonomy_domain_score": "10.0/10",
                    "autonomy_level_reached": 3,
                    "win_rate": 0.79,
                    "total_outcomes": 208,
                },
                "acceptance_reason": "test attestation for snapshot cache",
                "accepted_by": "operator:test",
                "accepted_at": "2026-04-22T00:00:00Z",
                "report_hash": report_hash,
                "artifact_refs": [str(artifact_path)],
                "schema_version": 1,
                "measured_source": "parsed",
            }
        ]),
        encoding="utf-8",
    )


def _seed_archived_missing_attestation(ledger_path: Path) -> None:
    """Write a record whose artifact is missing (archived_missing strength)."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    fake_body = b"archived proof"
    ledger_path.write_text(
        json.dumps([
            {
                "capability_id": "autonomy.l3",
                "evidence_source": "/tmp/does-not-exist.md",
                "evidence_window_start": "2026-03-21T00:00:00Z",
                "evidence_window_end": "2026-04-17T07:00:00Z",
                "measured_values": {
                    "oracle_composite": 95.1,
                    "autonomy_domain_score": "10.0/10",
                    "autonomy_level_reached": 3,
                    "win_rate": 0.79,
                    "total_outcomes": 208,
                },
                "acceptance_reason": "archived artifact test",
                "accepted_by": "operator:test",
                "accepted_at": "2026-04-22T00:00:00Z",
                "report_hash": "sha256:" + hashlib.sha256(fake_body).hexdigest(),
                "artifact_refs": ["/tmp/does-not-exist.md"],
                "schema_version": 1,
                "measured_source": "parsed",
            }
        ]),
        encoding="utf-8",
    )


@pytest.fixture
def isolated_ledger(monkeypatch, tmp_path):
    """Redirect attestation + escalation persistence paths into tmp_path."""
    from autonomy import attestation as att_mod
    from autonomy import escalation as esc_mod

    monkeypatch.setattr(att_mod, "_ATTESTATION_PATH", tmp_path / "attest.json")
    monkeypatch.setattr(esc_mod, "_PENDING_PATH", tmp_path / "pending.json")
    monkeypatch.setattr(esc_mod, "_ACTIVITY_PATH", tmp_path / "activity.jsonl")
    yield tmp_path


# --------------------------------------------------------------------------
# _build_l3_escalation_cache
# --------------------------------------------------------------------------


class TestL3EscalationCache:
    def test_no_engine_returns_safe_defaults(self, isolated_ledger):
        cache = _build_l3_escalation_cache(None)
        assert cache["available"] is False
        assert cache["current_ok"] is False
        assert cache["prior_attested_ok"] is False
        assert cache["attestation_strength"] == "none"
        assert cache["request_ok"] is False
        assert cache["approval_required"] is True
        assert cache["activation_ok"] is False
        assert cache["pending"] == []
        assert cache["recent_lifecycle"] == []

    def test_engine_without_orchestrator_returns_safe_defaults(self, isolated_ledger):
        cache = _build_l3_escalation_cache(_StubEngine(orch=None))
        assert cache["available"] is False
        assert cache["current_ok"] is False

    def test_current_ok_reflects_live_eligibility_true(self, isolated_ledger):
        orch = _StubOrchestrator(
            autonomy_level=2,
            eligible_for_l3=True,
            wins=30,
            win_rate=0.71,
            recent_regressions=0,
            l3_reason="Earned: 30/25 wins, 71%/50% rate, 0 regressions in last 10",
        )
        cache = _build_l3_escalation_cache(_StubEngine(orch))

        assert cache["available"] is True
        assert cache["live_autonomy_level"] == 2
        assert cache["current_ok"] is True
        assert cache["current_detail"]["wins"] == 30
        assert cache["current_detail"]["win_rate"] == pytest.approx(0.71)
        assert cache["current_detail"]["recent_regressions"] == 0
        assert "Earned" in cache["current_detail"]["reason"]
        # Not yet at L3, so activation_ok must remain False.
        assert cache["activation_ok"] is False
        assert cache["approval_required"] is True
        assert cache["request_ok"] is True  # via current_ok

    def test_current_ok_reflects_live_eligibility_false(self, isolated_ledger):
        orch = _StubOrchestrator(
            autonomy_level=1,
            eligible_for_l3=False,
            wins=3,
            win_rate=0.40,
            l3_reason="Need 25 wins (have 3) and 50% win rate (have 40%)",
        )
        cache = _build_l3_escalation_cache(_StubEngine(orch))
        assert cache["current_ok"] is False
        assert cache["request_ok"] is False
        assert cache["current_detail"]["wins"] == 3

    def test_current_ok_never_backfilled_from_persisted_file(
        self, isolated_ledger, tmp_path, monkeypatch
    ):
        """Regression guard: even if a stale autonomy-state file claims
        L3-eligible=True, the cache must still read from the live
        orchestrator and report the live answer. This enforces the
        Pillar 10 "current_ok is live-only" invariant."""
        stale_state = tmp_path / "autonomy_state.json"
        stale_state.write_text(
            json.dumps(
                {
                    "autonomy_level": 3,
                    "eligible_for_l3": True,
                    "win_rate": 0.95,
                    "wins": 500,
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("HOME", str(tmp_path))

        orch = _StubOrchestrator(
            autonomy_level=1,
            eligible_for_l3=False,
            wins=0,
            win_rate=0.0,
            l3_reason="Need 25 wins (have 0) and 50% win rate (have 0%)",
        )
        cache = _build_l3_escalation_cache(_StubEngine(orch))

        assert cache["current_ok"] is False
        assert cache["live_autonomy_level"] == 1
        assert cache["current_detail"]["wins"] == 0
        assert cache["activation_ok"] is False

    def test_prior_attested_ok_is_distinct_from_current_ok(self, isolated_ledger):
        artifact = isolated_ledger / "artifact.md"
        _seed_verified_attestation(isolated_ledger / "attest.json", artifact)

        orch = _StubOrchestrator(
            autonomy_level=1,
            eligible_for_l3=False,
            wins=0,
            win_rate=0.0,
        )
        cache = _build_l3_escalation_cache(_StubEngine(orch))

        # Live health is still False — attestation must not contaminate it.
        assert cache["current_ok"] is False
        # But attestation DOES unlock requestability.
        assert cache["prior_attested_ok"] is True
        assert cache["attestation_strength"] == "verified"
        assert cache["request_ok"] is True
        # Activation still requires live L3.
        assert cache["activation_ok"] is False
        assert cache["approval_required"] is True

    def test_archived_missing_strength_distinguished_from_verified(self, isolated_ledger):
        _seed_archived_missing_attestation(isolated_ledger / "attest.json")
        orch = _StubOrchestrator(autonomy_level=1, eligible_for_l3=False)
        cache = _build_l3_escalation_cache(_StubEngine(orch))
        assert cache["prior_attested_ok"] is True
        assert cache["attestation_strength"] == "archived_missing"

    def test_activation_ok_only_true_when_live_level_is_three(self, isolated_ledger):
        orch = _StubOrchestrator(autonomy_level=3, eligible_for_l3=True)
        cache = _build_l3_escalation_cache(_StubEngine(orch))
        assert cache["live_autonomy_level"] == 3
        assert cache["activation_ok"] is True
        assert cache["approval_required"] is False


# --------------------------------------------------------------------------
# _build_attestation_cache
# --------------------------------------------------------------------------


class TestAttestationCache:
    def test_empty_ledger_returns_empty_structure(self, isolated_ledger):
        cache = _build_attestation_cache()
        assert cache["records"] == []
        assert cache["prior_attested_ok"] is False
        assert cache["attestation_strength"] == "none"
        assert "updated_at" in cache

    def test_verified_record_surfaces_in_cache(self, isolated_ledger):
        artifact = isolated_ledger / "artifact.md"
        _seed_verified_attestation(isolated_ledger / "attest.json", artifact)
        cache = _build_attestation_cache()
        assert cache["prior_attested_ok"] is True
        assert cache["attestation_strength"] == "verified"
        assert len(cache["records"]) == 1
        rec = cache["records"][0]
        assert rec["capability_id"] == "autonomy.l3"
        assert rec["artifact_status"] == "hash_verified"
        assert rec["attestation_strength"] == "verified"
        assert rec["measured_values"]["oracle_composite"] == 95.1

    def test_archived_missing_record_surfaces_lower_strength(self, isolated_ledger):
        _seed_archived_missing_attestation(isolated_ledger / "attest.json")
        cache = _build_attestation_cache()
        assert cache["prior_attested_ok"] is True
        assert cache["attestation_strength"] == "archived_missing"
        assert cache["records"][0]["artifact_status"] == "missing"
        assert cache["records"][0]["attestation_strength"] == "archived_missing"

    def test_cache_never_writes_to_ledger(self, isolated_ledger):
        ledger_path = isolated_ledger / "attest.json"
        assert not ledger_path.exists()
        _build_attestation_cache()
        # Reading the empty ledger must not create the file.
        assert not ledger_path.exists()
