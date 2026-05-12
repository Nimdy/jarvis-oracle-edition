"""P3.4 rollback round-trip regression test.

Covers the full seed-register -> mutate -> roll back -> forward-register
cycle used by the Phase E Language Kernel artifact identity surface.

Specifically guards the following invariants (see TODO_V2.md P3.4):

  * ``get_live_artifact()`` returns ``None`` before any seed.
  * After seeding, ``get_live_artifact()`` returns the newly registered
    artifact and ``get_state()["status"] == "registered"``.
  * A mutate-and-register step creates a new version whose
    ``live_artifact_id`` is the newer one.
  * ``rollback_to(v1.artifact_id)`` restores v1's bytes on disk,
    re-points ``live_artifact_id`` to v1, and appends to
    ``rollback_history``.
  * A forward re-register of v2's bytes restores v2 as live without
    introducing a v3 (hash-dedup).
  * The operator seed script (``brain/scripts/seed_language_kernel.py``)
    is importable and exposes a ``main`` entry point.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from language.kernel import LanguageKernelRegistry


@pytest.fixture
def seeded_registry():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        checkpoint = base / "student_checkpoint.json"
        reg = LanguageKernelRegistry(
            registry_path=base / "registry.json",
            snapshot_dir=base / "snapshots",
            checkpoint_path=checkpoint,
        )
        yield reg, checkpoint


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


class TestSeedRollbackRoundtrip:

    def test_full_roundtrip(self, seeded_registry):
        reg, checkpoint = seeded_registry

        assert reg.get_live_artifact() is None
        assert reg.get_state()["status"] == "pre_mature"

        _write(checkpoint, {"schema_version": 1, "loss": 0.42, "stage": "seed"})
        v1 = reg.register_current_checkpoint(notes="phase_c_seed")
        assert v1 is not None
        assert v1.version == 1
        assert reg.get_state()["status"] == "registered"
        assert reg.get_live_artifact().artifact_id == v1.artifact_id
        v1_bytes = checkpoint.read_bytes()

        # Forward progress: write new bytes + register v2.
        _write(checkpoint, {"schema_version": 1, "loss": 0.10, "stage": "forward"})
        v2 = reg.register_current_checkpoint(notes="forward_step")
        assert v2 is not None
        assert v2.version == 2
        assert v2.artifact_id != v1.artifact_id
        assert reg.get_live_artifact().artifact_id == v2.artifact_id
        v2_bytes = checkpoint.read_bytes()
        assert v2_bytes != v1_bytes

        # Rollback v2 -> v1: on-disk bytes restored, live pointer moved.
        restored = reg.rollback_to(v1.artifact_id, reason="roundtrip_test")
        assert restored is not None and restored.artifact_id == v1.artifact_id
        assert checkpoint.read_bytes() == v1_bytes
        assert reg.get_live_artifact().artifact_id == v1.artifact_id
        rb_history = reg.get_state()["rollback_history"]
        assert rb_history[-1]["from_artifact_id"] == v2.artifact_id
        assert rb_history[-1]["to_artifact_id"] == v1.artifact_id
        assert rb_history[-1]["reason"] == "roundtrip_test"

        # Forward-advance via rollback_to(v2): operator path for moving
        # back to a previously-registered identity without creating a
        # new version number.
        advanced = reg.rollback_to(v2.artifact_id, reason="forward_recovery")
        assert advanced is not None and advanced.artifact_id == v2.artifact_id
        assert checkpoint.read_bytes() == v2_bytes
        assert reg.get_live_artifact().artifact_id == v2.artifact_id
        assert reg.get_state()["total_artifacts"] == 2

        # Sanity: re-registering the current live bytes is a no-op
        # (hash-dedup against the live artifact).
        noop = reg.register_current_checkpoint(notes="noop")
        assert noop is not None
        assert noop.artifact_id == v2.artifact_id
        assert reg.get_state()["total_artifacts"] == 2


class TestSeedScriptImportable:

    def test_seed_script_exposes_main(self):
        """The operator seed tool must be importable for CI + docs."""
        import importlib

        # Importing as a module via the brain/scripts package.
        module = importlib.import_module("scripts.seed_language_kernel")
        assert hasattr(module, "main")
        assert callable(module.main)

    def test_seed_script_dry_run_works_with_no_checkpoint(self, seeded_registry, capsys, monkeypatch):
        """Dry-run against a fresh registry should not raise and should
        print the pre-mature state."""
        reg, _ = seeded_registry
        import importlib

        module = importlib.import_module("scripts.seed_language_kernel")

        # Monkey-patch the registry accessor so the script reads our
        # temporary registry rather than the real ``~/.jarvis`` one.
        monkeypatch.setattr(
            module,
            "get_language_kernel_registry",
            lambda: reg,
        )
        rc = module.main(["--dry-run"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "dry-run" in captured.out
        assert "\"status\"" in captured.out
