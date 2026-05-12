"""Phase 6.5 — end-to-end smoke integration test.

This is the permanent regression-coverage version of the "seed → request_ok
flips → manual promote → durable audit" operator ritual. It exercises the
full Phase 6.5 contract in-process with zero risk to the live brain:

    1. Fresh-brain start: attestation ledger empty, no escalations, L2.
    2. _build_l3_escalation_cache reports all-false state.
    3. Seed a hash-verified ``autonomy.l3`` attestation record.
    4. Cache recompute: ``prior_attested_ok`` flips True,
       ``attestation_strength=="verified"``, ``request_ok`` lights up,
       but ``current_ok`` / ``activation_ok`` stay False (still L2 with
       no live eligibility).
    5. Validation pack confirms ``prior_attested_ok`` is a *separate*
       field on the L3 check and does not contaminate ``ever_ok`` on
       any non-L3 check.
    6. Manual L3 promotion through the orchestrator's public API emits
       exactly one ``AUTONOMY_LEVEL_CHANGED`` (2→3) and exactly one
       ``AUTONOMY_L3_PROMOTED`` with ``outcome="clean"`` — no denial,
       no rollback, no parked/expired events.
    7. The durable audit ledger contains both events on disk in the
       correct order, and a fresh ``AutonomyAuditLedger`` instance
       re-reading the same file surfaces the same sequence (persistence
       survives restart).

All state is isolated under ``tmp_path``:
 - attestation ledger path
 - escalation store pending/activity paths
 - autonomy audit ledger path
 - a fresh ``EventBus`` with the boot barrier open, injected in place of
   the module-level singleton inside ``autonomy.orchestrator``
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# --------------------------------------------------------------------------
# Orchestrator-at-L2 builder (mirrors test_l3_promotion_invariant.py)
# --------------------------------------------------------------------------


class _FakePolicyMemory:
    """Minimal policy memory stub used by the orchestrator during tests."""

    def get_l2_readiness(self) -> dict[str, Any]:
        return {
            "cycles_since_last_regression": 0,
            "warmup_remaining_s": 0.0,
            "avoid_patterns": [],
            "unique_tools": [],
            "unique_types": [],
        }


def _build_orchestrator_at_l2(tmp_dir: Path):
    """Construct an AutonomyOrchestrator pre-seeded at L2, warmup bypassed."""
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
        orch._policy_memory = _FakePolicyMemory()
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


class _StubEngine:
    """Minimal engine surface the snapshot cache reads."""

    def __init__(self, orch: Any) -> None:
        self._autonomy_orchestrator = orch


# --------------------------------------------------------------------------
# Attestation seeding (hash-verified autonomy.l3 record)
# --------------------------------------------------------------------------


def _seed_verified_attestation(ledger_path: Path, artifact_path: Path) -> str:
    """Write a hash-verified autonomy.l3 record straight to the ledger JSON.

    Returns the ``report_hash`` so callers can reference it in audit trail
    assertions. Bypasses ``AttestationLedger.add`` deliberately: this mirrors
    the shape the seed CLI would produce and keeps the smoke free of parser
    dependencies.
    """
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_body = b"phase_6_5 smoke proof body\n"
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
                "acceptance_reason": "phase_6_5 smoke seed",
                "accepted_by": "operator:phase_6_5_smoke",
                "accepted_at": "2026-04-22T00:00:00Z",
                "report_hash": report_hash,
                "artifact_refs": [str(artifact_path)],
                "schema_version": 1,
                "measured_source": "parsed",
            }
        ]),
        encoding="utf-8",
    )
    return report_hash


# --------------------------------------------------------------------------
# Fixture: isolate every persistence path + redirect the orchestrator's bus
# --------------------------------------------------------------------------


@pytest.fixture()
def smoke_env(monkeypatch, tmp_path):
    """Redirect every Phase 6.5 persistence surface into ``tmp_path``.

    Also substitutes a fresh, barrier-open ``EventBus`` for the module
    singleton the orchestrator imports, so emitted events are captured
    by the audit ledger wired in the same test — not by any globally
    shared bus.
    """
    from autonomy import attestation as att_mod
    from autonomy import escalation as esc_mod
    from autonomy import audit_ledger as audit_mod
    from consciousness.events import EventBus

    monkeypatch.setattr(att_mod, "_ATTESTATION_PATH", tmp_path / "ever_proven_attestation.json")
    monkeypatch.setattr(esc_mod, "_PENDING_PATH", tmp_path / "pending.json")
    monkeypatch.setattr(esc_mod, "_ACTIVITY_PATH", tmp_path / "activity.jsonl")
    monkeypatch.setattr(audit_mod, "_AUDIT_PATH", tmp_path / "autonomy_audit.jsonl")

    bus = EventBus()
    bus.open_barrier()
    monkeypatch.setattr("autonomy.orchestrator.event_bus", bus)

    return {
        "tmp_path": tmp_path,
        "bus": bus,
        "attestation_path": tmp_path / "ever_proven_attestation.json",
        "artifact_path": tmp_path / "artifact.md",
        "audit_path": tmp_path / "autonomy_audit.jsonl",
    }


# --------------------------------------------------------------------------
# The smoke
# --------------------------------------------------------------------------


def test_phase_6_5_end_to_end_smoke(smoke_env):
    """Full Phase 6.5 contract: attestation → request_ok → manual promote → audit."""
    from autonomy.attestation import AttestationLedger
    from autonomy.audit_ledger import AutonomyAuditLedger
    from dashboard.snapshot import _build_l3_escalation_cache
    from jarvis_eval.validation_pack import build_validation_pack

    tmp_path: Path = smoke_env["tmp_path"]
    bus = smoke_env["bus"]
    audit_path: Path = smoke_env["audit_path"]

    # ---- 1. Fresh brain: no attestation, no escalations, orchestrator at L2.
    orch = _build_orchestrator_at_l2(tmp_path)
    engine = _StubEngine(orch)

    ledger = AttestationLedger()
    assert ledger.load() == [], "fresh-brain attestation ledger must be empty"
    assert ledger.prior_attested_ok("autonomy.l3") is False

    # ---- 2. Wire the durable audit subscriber to THIS bus.
    audit = AutonomyAuditLedger(path=audit_path)
    audit.wire(bus=bus)
    assert audit.wired is True
    try:
        # ---- 3. Cache reflects fresh-brain state: every gate closed.
        cache_before = _build_l3_escalation_cache(engine)
        assert cache_before["available"] is True
        assert cache_before["live_autonomy_level"] == 2
        assert cache_before["current_ok"] is False
        assert cache_before["prior_attested_ok"] is False
        assert cache_before["attestation_strength"] == "none"
        assert cache_before["request_ok"] is False
        assert cache_before["activation_ok"] is False
        assert cache_before["approval_required"] is True

        # Non-L3 ever_ok baseline (must remain invariant across attestation seed).
        vp_before = build_validation_pack({}, {}, {}, {"l3": cache_before}, {})
        non_l3_ever_before = {
            c["id"]: c["ever_ok"]
            for c in vp_before["checks"]
            if c.get("kind") != "phase6_5"
        }
        assert non_l3_ever_before, "expected at least one non-L3 check in the baseline pack"

        # ---- 4. Seed a verified attestation.
        report_hash = _seed_verified_attestation(
            smoke_env["attestation_path"], smoke_env["artifact_path"]
        )

        # ---- 5. Cache recompute: prior_attested_ok flips, current_ok stays False.
        cache_after_seed = _build_l3_escalation_cache(engine)
        assert cache_after_seed["available"] is True
        assert cache_after_seed["live_autonomy_level"] == 2
        # current_ok is strictly live-sourced and must NOT be backfilled
        # from the attestation ledger.
        assert cache_after_seed["current_ok"] is False, (
            "current_ok must never be backfilled from prior attestation"
        )
        assert cache_after_seed["prior_attested_ok"] is True
        assert cache_after_seed["attestation_strength"] == "verified"
        # request_ok = current_ok OR prior_attested_ok → True via attestation.
        assert cache_after_seed["request_ok"] is True
        # Still L2, so activation is not yet authorised.
        assert cache_after_seed["activation_ok"] is False
        assert cache_after_seed["approval_required"] is True

        # ---- 6. Validation-pack invariants: attestation does NOT contaminate
        #        ever_ok on non-L3 checks; prior_attested_ok is a separate
        #        explicit field on the L3 request check.
        vp_after_seed = build_validation_pack({}, {}, {}, {"l3": cache_after_seed}, {})
        non_l3_ever_after_seed = {
            c["id"]: c["ever_ok"]
            for c in vp_after_seed["checks"]
            if c.get("kind") != "phase6_5"
        }
        assert non_l3_ever_after_seed == non_l3_ever_before, (
            "attestation seed must not flip any non-L3 ever_ok field "
            f"(diff: { {k: (non_l3_ever_before.get(k), v) for k, v in non_l3_ever_after_seed.items() if non_l3_ever_before.get(k) != v} })"
        )

        checks_by_id = {c["id"]: c for c in vp_after_seed["checks"]}
        l3_req = checks_by_id["l3_escalation_requestable"]
        assert l3_req["prior_attested_ok"] is True
        assert l3_req["attestation_strength"] == "verified"
        assert l3_req["current_ok"] is True, "request_ok should drive the L3 check's current_ok"

        # Spot-check: non-L3 checks do NOT expose prior_attested_ok at all
        # (evidence-class separation is structural, not a coincidence).
        for cid, c in checks_by_id.items():
            if c.get("kind") == "phase6_5":
                continue
            assert "prior_attested_ok" not in c, (
                f"non-L3 check {cid!r} leaked prior_attested_ok field"
            )

        # ---- 7. Pre-promotion: durable audit ledger must still be empty.
        assert audit.load_recent(limit=50) == []

        # ---- 8. Manual L3 promotion via the orchestrator's public API.
        #        Evidence path uses the ``prior_attested`` class (the same
        #        class the dashboard API selects when current_ok is False
        #        but prior_attested_ok is True).
        evidence_path = f"prior_attested:autonomy.l3:{report_hash}"
        orch.set_autonomy_level(
            3,
            evidence_path=evidence_path,
            approval_source="operator:phase_6_5_smoke",
            caller_id="phase_6_5_smoke",
        )
        assert orch._autonomy_level == 3

        # ---- 9. Cache recompute: activation_ok now True, approval not required.
        cache_after_promote = _build_l3_escalation_cache(engine)
        assert cache_after_promote["live_autonomy_level"] == 3
        assert cache_after_promote["activation_ok"] is True
        assert cache_after_promote["approval_required"] is False
        # current_ok is still False — the orchestrator's
        # check_promotion_eligibility() reports no live wins for this
        # session. Activation flowed from attestation, not live earning.
        assert cache_after_promote["current_ok"] is False
        assert cache_after_promote["prior_attested_ok"] is True

        # ---- 10. Event sequence on the durable audit ledger.
        entries = audit.load_recent(limit=50)
        event_names = [e["event"] for e in entries]
        assert event_names == [
            "autonomy:level_changed",
            "autonomy:l3_promoted",
        ], f"unexpected event sequence: {event_names}"

        lvl = entries[0]
        assert lvl["event"] == "autonomy:level_changed"
        assert lvl["payload"]["old_level"] == 2
        assert lvl["payload"]["new_level"] == 3

        promoted = entries[1]
        assert promoted["event"] == "autonomy:l3_promoted"
        p = promoted["payload"]
        assert p["outcome"] == "clean", (
            "AUTONOMY_L3_PROMOTED must carry outcome='clean'; "
            "denials belong on AUTONOMY_L3_ACTIVATION_DENIED"
        )
        assert p["prior_level"] == 2
        assert p["evidence_path"] == evidence_path
        assert p["approval_source"] == "operator:phase_6_5_smoke"
        assert p["caller_id"] == "phase_6_5_smoke"

        # ---- 11. Negative guards: no denial, no rollback, no parked, no expired.
        for forbidden in (
            "autonomy:l3_activation_denied",
            "autonomy:escalation_rolled_back",
            "autonomy:escalation_parked",
            "autonomy:escalation_expired",
            "autonomy:escalation_rejected",
        ):
            assert forbidden not in event_names, (
                f"clean manual promotion must not emit {forbidden}"
            )

        # ---- 12. Persistence: a brand-new ledger instance re-reading the
        #        same file must surface the identical sequence.
        persisted = AutonomyAuditLedger(path=audit_path).load_recent(limit=50)
        assert [e["event"] for e in persisted] == event_names
        assert persisted[1]["payload"]["outcome"] == "clean"
        assert persisted[1]["payload"]["evidence_path"] == evidence_path
    finally:
        audit.unwire()
