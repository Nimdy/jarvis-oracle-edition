"""Specialist weight persistence — boot-load of last-known-good weights.

Contract: WEIGHTS persist (fast rebuild, no retrain-from-zero) but AUTHORITY
re-earns. P0 restores only shadow/Tier-1 specialists (no broadcast authority);
authority-bearing Matrix Tier-2 foci are skipped (re-earn on live reps). A
corrupt/incompatible checkpoint is discarded, never breaks boot.
"""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hemisphere.registry import (
    HemisphereRegistry, ModelVersion, _topology_to_dict, _dict_to_topology,
)
from hemisphere.types import NetworkTopology, LayerDefinition


def _topo() -> NetworkTopology:
    layers = (
        LayerDefinition(id="in", layer_type="input", node_count=8, activation="linear"),
        LayerDefinition(id="h1", layer_type="hidden", node_count=16, activation="relu", dropout=0.1),
        LayerDefinition(id="out", layer_type="output", node_count=4, activation="softmax"),
    )
    return NetworkTopology(input_size=8, layers=layers, output_size=4,
                           total_parameters=123,
                           activation_functions=tuple(l.activation for l in layers))


def _mv(focus, ver, acc, path, active=False, topo=True, created=0.0):
    return ModelVersion(
        version=ver, focus=focus, network_id=f"{focus}-{ver}", name=focus,
        accuracy=acc, loss=0.1, total_parameters=1, is_active=active,
        created_at=created, path=path,
        topology_json=_topology_to_dict(_topo()) if topo else {},
    )


class TestTopologyRoundtrip:
    def test_roundtrip(self):
        t = _topo()
        t2 = _dict_to_topology(_topology_to_dict(t))
        assert t2.input_size == 8 and t2.output_size == 4 and t2.total_parameters == 123
        assert len(t2.layers) == 3
        assert t2.layers[1].node_count == 16 and t2.layers[1].dropout == 0.1
        assert t2.activation_functions == ("linear", "relu", "softmax")

    def test_malformed_raises(self):
        # caller relies on this to discard an incompatible checkpoint
        with pytest.raises((KeyError, TypeError)):
            _dict_to_topology({"input_size": 8})  # no layers/output_size


class TestRegistryBestVersion:
    def _reg(self, tmp_path):
        reg = HemisphereRegistry(base_dir=Path(tmp_path))
        reg._versions = {}
        return reg

    def test_foci_only_with_versions(self, tmp_path):
        reg = self._reg(tmp_path)
        reg._versions = {"diagnostic": [_mv("diagnostic", 1, 0.5, str(tmp_path / "d.pt"))],
                         "empty": []}
        assert reg.foci() == ["diagnostic"]

    def test_active_wins(self, tmp_path):
        p = tmp_path / "x.pt"; p.write_text("w")
        reg = self._reg(tmp_path)
        reg._versions = {"diagnostic": [
            _mv("diagnostic", 1, 0.9, str(p), active=False),
            _mv("diagnostic", 2, 0.4, str(p), active=True),  # lower acc but ACTIVE
        ]}
        assert reg.best_version("diagnostic").version == 2

    def test_highest_accuracy_when_none_active(self, tmp_path):
        p = tmp_path / "x.pt"; p.write_text("w")
        reg = self._reg(tmp_path)
        reg._versions = {"diagnostic": [
            _mv("diagnostic", 1, 0.4, str(p)),
            _mv("diagnostic", 2, 0.8, str(p)),
        ]}
        assert reg.best_version("diagnostic").version == 2

    def test_missing_weights_or_topology_excluded(self, tmp_path):
        present = tmp_path / "x.pt"; present.write_text("w")
        reg = self._reg(tmp_path)
        reg._versions = {"diagnostic": [
            _mv("diagnostic", 1, 0.9, str(tmp_path / "gone.pt")),     # file missing
            _mv("diagnostic", 2, 0.8, str(present), topo=False),      # no topology
        ]}
        assert reg.best_version("diagnostic") is None


_HAS_ORCH = True
try:
    from hemisphere.orchestrator import HemisphereOrchestrator
except Exception:  # pragma: no cover - heavy deps absent
    _HAS_ORCH = False


@pytest.mark.skipif(not _HAS_ORCH, reason="orchestrator import unavailable")
class TestRestore:
    def _stub(self, fake_registry, fake_engine):
        return SimpleNamespace(_registry=fake_registry, _engine=fake_engine,
                               _networks={}, _networks_lock=threading.Lock())

    def test_restores_shadow_and_tier2_with_firewall(self, tmp_path):
        from hemisphere.types import SpecialistLifecycleStage
        p = tmp_path / "w.pt"; p.write_text("w")
        loaded = []
        fake_engine = SimpleNamespace(load_model=lambda nid, topo, path: loaded.append(nid))
        best = {
            "diagnostic": _mv("diagnostic", 3, 0.7, str(p)),            # shadow
            "positive_memory": _mv("positive_memory", 2, 0.9, str(p)),  # Tier-2
        }
        fake_registry = SimpleNamespace(
            foci=lambda: list(best.keys()),
            best_version=lambda f: best[f],
            discard_version=lambda *a, **k: None,
        )
        stub = self._stub(fake_registry, fake_engine)
        HemisphereOrchestrator._restore_persisted_specialists(stub)
        # BOTH restored, weights loaded for both
        assert set(loaded) == {"diagnostic-3", "positive_memory-2"}
        # shadow: keeps training accuracy, no lifecycle (no authority to leak)
        shadow = stub._networks["diagnostic-3"]
        assert shadow.status.value == "ready"
        assert shadow.specialist_lifecycle is None
        assert shadow.performance.accuracy == 0.7
        # Tier-2 FIREWALL: weights back, but ALL live standing reset
        t2 = stub._networks["positive_memory-2"]
        assert t2.specialist_lifecycle == SpecialistLifecycleStage.PROBATIONARY_TRAINING
        assert t2.specialist_impact_score == 0.0
        assert t2.specialist_verification_ts == 0.0
        assert t2.performance.accuracy == 0.0  # NOT the persisted 0.9 — re-measure live

    def test_tier2_restore_respects_probationary_cap(self, tmp_path):
        from hemisphere.types import MAX_PROBATIONARY_SPECIALISTS, MATRIX_ELIGIBLE_FOCUSES
        p = tmp_path / "w.pt"; p.write_text("w")
        loaded = []
        fake_engine = SimpleNamespace(load_model=lambda nid, topo, path: loaded.append(nid))
        # every Tier-2 focus has persisted weights (more than the cap)
        best = {f.value: _mv(f.value, 1, 0.8, str(p)) for f in MATRIX_ELIGIBLE_FOCUSES}
        assert len(best) > MAX_PROBATIONARY_SPECIALISTS
        fake_registry = SimpleNamespace(
            foci=lambda: list(best.keys()),
            best_version=lambda f: best[f],
            discard_version=lambda *a, **k: None,
        )
        stub = self._stub(fake_registry, fake_engine)
        HemisphereOrchestrator._restore_persisted_specialists(stub)
        assert len(stub._networks) == MAX_PROBATIONARY_SPECIALISTS
        assert len(loaded) == MAX_PROBATIONARY_SPECIALISTS

    def test_incompatible_checkpoint_discarded_not_fatal(self, tmp_path):
        p = tmp_path / "w.pt"; p.write_text("w")
        def _boom(nid, topo, path):
            raise RuntimeError("state_dict mismatch")
        discarded = []
        fake_engine = SimpleNamespace(load_model=_boom)
        fake_registry = SimpleNamespace(
            foci=lambda: ["diagnostic"],
            best_version=lambda f: _mv("diagnostic", 5, 0.7, str(p)),
            discard_version=lambda focus, ver, **k: discarded.append((focus, ver)),
        )
        stub = self._stub(fake_registry, fake_engine)
        # must NOT raise
        HemisphereOrchestrator._restore_persisted_specialists(stub)
        assert stub._networks == {}
        assert discarded == [("diagnostic", 5)]
