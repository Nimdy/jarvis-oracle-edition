"""Skeleton regression tests for Phase E Language Kernel artifact
identity (P1.5).

Covers:
  - ``register_current_checkpoint`` returns ``None`` when no Phase C
    checkpoint exists (PRE-MATURE on fresh brain).
  - Registration produces a stable artifact id of the form
    ``phasec-v{version}-{hash[:12]}``, and persists the registry +
    snapshot.
  - Re-registering the same bytes is a no-op (returns the existing
    live artifact rather than double-registering).
  - Registration after real byte changes assigns a new version and
    updates ``live_artifact_id``.
  - ``rollback_to`` restores the snapshot bytes on disk and updates
    ``live_artifact_id`` + appends to ``rollback_history``.
  - ``get_state`` reports ``pre_mature`` / ``registered`` correctly,
    surfaces on-disk hash + drift flag, and respects the 10-element
    truncation on ``artifacts`` + ``rollback_history``.
  - ``LanguagePromotionGovernor.get_live_artifact_id`` reads through
    to the registry without activating anything.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from language.kernel import (
    LanguageKernelRegistry,
    get_language_kernel_registry,
    set_language_kernel_registry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_registry():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        registry_path = base / "registry.json"
        snapshot_dir = base / "snapshots"
        checkpoint = base / "student_checkpoint.json"
        reg = LanguageKernelRegistry(
            registry_path=registry_path,
            snapshot_dir=snapshot_dir,
            checkpoint_path=checkpoint,
        )
        yield reg, checkpoint, registry_path, snapshot_dir


def _write_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Pre-mature / registration
# ---------------------------------------------------------------------------


class TestPreMatureAndRegistration:

    def test_register_returns_none_when_no_checkpoint(self, tmp_registry):
        reg, checkpoint, _, _ = tmp_registry
        assert not checkpoint.exists()
        assert reg.register_current_checkpoint() is None
        state = reg.get_state()
        assert state["status"] == "pre_mature"
        assert state["live_artifact"] is None
        assert state["total_artifacts"] == 0
        assert state["checkpoint_exists"] is False

    def test_register_creates_artifact(self, tmp_registry):
        reg, checkpoint, registry_path, snapshot_dir = tmp_registry
        _write_checkpoint(checkpoint, {"schema_version": 1, "loss": 0.42})
        artifact = reg.register_current_checkpoint(notes="initial")
        assert artifact is not None
        assert artifact.version == 1
        assert artifact.artifact_id.startswith("phasec-v1-")
        assert len(artifact.hash) == 64
        assert artifact.checkpoint_schema_version == 1
        assert artifact.notes == "initial"
        # Persistence invariants.
        assert registry_path.exists()
        assert (snapshot_dir / f"{artifact.artifact_id}.json").exists()
        # Snapshot is byte-for-byte identical to the source at time of
        # registration.
        assert (
            (snapshot_dir / f"{artifact.artifact_id}.json").read_bytes()
            == checkpoint.read_bytes()
        )

    def test_reregister_same_bytes_is_idempotent(self, tmp_registry):
        reg, checkpoint, _, _ = tmp_registry
        _write_checkpoint(checkpoint, {"schema_version": 1, "loss": 0.42})
        first = reg.register_current_checkpoint()
        second = reg.register_current_checkpoint()
        assert first is not None and second is not None
        assert first.artifact_id == second.artifact_id
        assert reg.get_state()["total_artifacts"] == 1

    def test_register_after_change_assigns_new_version(self, tmp_registry):
        reg, checkpoint, _, _ = tmp_registry
        _write_checkpoint(checkpoint, {"schema_version": 1, "loss": 0.42})
        v1 = reg.register_current_checkpoint()
        _write_checkpoint(checkpoint, {"schema_version": 1, "loss": 0.37})
        v2 = reg.register_current_checkpoint()
        assert v1 is not None and v2 is not None
        assert v1.version == 1 and v2.version == 2
        assert v1.hash != v2.hash
        state = reg.get_state()
        assert state["total_artifacts"] == 2
        assert state["live_artifact"]["artifact_id"] == v2.artifact_id

    def test_state_drift_flag_when_checkpoint_edited_outside(self, tmp_registry):
        reg, checkpoint, _, _ = tmp_registry
        _write_checkpoint(checkpoint, {"loss": 0.42})
        reg.register_current_checkpoint()
        # Someone (or a training run) overwrote the checkpoint without
        # re-registering. The registry should flag drift.
        _write_checkpoint(checkpoint, {"loss": 0.10})
        state = reg.get_state()
        assert state["matches_live_artifact"] is False
        assert state["on_disk_hash"] is not None


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


class TestRollback:

    def test_rollback_restores_prior_bytes(self, tmp_registry):
        reg, checkpoint, _, _ = tmp_registry
        _write_checkpoint(checkpoint, {"schema_version": 1, "loss": 0.42})
        v1 = reg.register_current_checkpoint()
        _write_checkpoint(checkpoint, {"schema_version": 1, "loss": 0.10})
        v2 = reg.register_current_checkpoint()
        assert v1 is not None and v2 is not None
        restored = reg.rollback_to(v1.artifact_id, reason="regression_detected")
        assert restored is not None
        assert restored.artifact_id == v1.artifact_id
        # Bytes on disk now match v1's hash.
        assert reg.get_state()["on_disk_hash"] == v1.hash
        assert reg.get_state()["live_artifact"]["artifact_id"] == v1.artifact_id
        rb = reg.get_state()["rollback_history"]
        assert rb and rb[-1]["from_artifact_id"] == v2.artifact_id
        assert rb[-1]["reason"] == "regression_detected"

    def test_rollback_unknown_id_returns_none(self, tmp_registry):
        reg, checkpoint, _, _ = tmp_registry
        _write_checkpoint(checkpoint, {"loss": 0.42})
        reg.register_current_checkpoint()
        assert reg.rollback_to("phasec-vX-deadbeef") is None

    def test_rollback_missing_snapshot_returns_none(self, tmp_registry):
        reg, checkpoint, _, snapshot_dir = tmp_registry
        _write_checkpoint(checkpoint, {"loss": 0.42})
        a1 = reg.register_current_checkpoint()
        assert a1 is not None
        # Manually delete the snapshot to simulate disk corruption.
        (snapshot_dir / f"{a1.artifact_id}.json").unlink()
        assert reg.rollback_to(a1.artifact_id) is None

    def test_rollback_does_not_modify_phase_d_governor(self, tmp_registry):
        """Phase E artifact rollback must NOT bypass the Phase D guard."""
        reg, checkpoint, _, _ = tmp_registry
        _write_checkpoint(checkpoint, {"loss": 0.42})
        v1 = reg.register_current_checkpoint()
        _write_checkpoint(checkpoint, {"loss": 0.10})
        reg.register_current_checkpoint()

        # Capture governor state before rollback.
        from jarvis_eval.language_promotion import LanguagePromotionGovernor

        gov = LanguagePromotionGovernor()
        levels_before = {
            rc: gov.get_level(rc) for rc in gov._states.keys()
        }
        reg.rollback_to(v1.artifact_id)
        levels_after = {
            rc: gov.get_level(rc) for rc in gov._states.keys()
        }
        # Levels are unchanged — artifact rollback is scope-isolated.
        assert levels_before == levels_after


# ---------------------------------------------------------------------------
# History truncation
# ---------------------------------------------------------------------------


class TestHistoryTruncation:

    def test_artifacts_list_trims_to_last_10(self, tmp_registry):
        reg, checkpoint, _, _ = tmp_registry
        for i in range(15):
            _write_checkpoint(checkpoint, {"loss": 0.5 - i * 0.01, "i": i})
            reg.register_current_checkpoint()
        state = reg.get_state()
        assert state["total_artifacts"] == 15
        assert len(state["artifacts"]) == 10
        assert state["artifacts"][-1]["version"] == 15

    def test_rollback_history_caps_at_50(self, tmp_registry):
        reg, checkpoint, _, _ = tmp_registry
        _write_checkpoint(checkpoint, {"loss": 0.42})
        first = reg.register_current_checkpoint()
        _write_checkpoint(checkpoint, {"loss": 0.10})
        second = reg.register_current_checkpoint()
        assert first is not None and second is not None
        # Ping-pong rollbacks 60 times; internal cap is 50.
        for i in range(60):
            target = first.artifact_id if i % 2 == 0 else second.artifact_id
            reg.rollback_to(target)
        # Internal state should be capped (peek at get_state which trims
        # to last 10 for dashboard, but internal cap is 50).
        state_len = len(reg._state.rollback_history)  # noqa: SLF001
        assert state_len == 50


# ---------------------------------------------------------------------------
# Governor wiring
# ---------------------------------------------------------------------------


class TestGovernorWiring:

    def teardown_method(self, method):
        set_language_kernel_registry(None)

    def test_governor_reads_live_artifact_id(self, tmp_registry):
        reg, checkpoint, _, _ = tmp_registry
        _write_checkpoint(checkpoint, {"loss": 0.42})
        art = reg.register_current_checkpoint()
        set_language_kernel_registry(reg)

        from jarvis_eval.language_promotion import LanguagePromotionGovernor

        gov = LanguagePromotionGovernor()
        assert art is not None
        assert gov.get_live_artifact_id() == art.artifact_id

    def test_governor_returns_none_when_no_registry(self, tmp_registry):
        reg, checkpoint, _, _ = tmp_registry
        # No artifacts registered; checkpoint does not exist.
        set_language_kernel_registry(reg)

        from jarvis_eval.language_promotion import LanguagePromotionGovernor

        gov = LanguagePromotionGovernor()
        assert gov.get_live_artifact_id() is None
