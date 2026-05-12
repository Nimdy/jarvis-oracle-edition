"""HemisphereRegistry: model versioning and persistence.

Persists hemisphere models to ~/.jarvis/hemispheres/{focus}/{version}.pt
with metadata in a JSON sidecar. Pattern mirrors brain/policy/registry.py.
"""

from __future__ import annotations

import json
import logging
import os
import time as _time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hemisphere.types import (
    HemisphereFocus,
    HemisphereSnapshot,
    HemisphereState,
    NetworkArchitecture,
    NetworkTopology,
    PerformanceMetrics,
    SubstrateType,
)

logger = logging.getLogger(__name__)

MAX_VERSIONS_PER_FOCUS = 3
BASE_DIR = Path.home() / ".jarvis" / "hemispheres"


@dataclass
class ModelVersion:
    version: int
    focus: str
    network_id: str
    name: str
    accuracy: float
    loss: float
    total_parameters: int
    is_active: bool
    created_at: float
    path: str
    topology_json: dict[str, Any] = field(default_factory=dict)


class HemisphereRegistry:
    """Versioned model storage for hemisphere networks."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base = base_dir or BASE_DIR
        self._versions: dict[str, list[ModelVersion]] = {}
        self._next_version: dict[str, int] = {}
        self._load_state()

    # ------------------------------------------------------------------
    # Register / promote
    # ------------------------------------------------------------------

    def register(
        self,
        network: NetworkArchitecture,
        save_fn: Any = None,
    ) -> ModelVersion:
        """Register a new model version and optionally persist weights."""
        focus_key = network.focus.value
        ver = self._next_version.get(focus_key, 1)
        self._next_version[focus_key] = ver + 1

        focus_dir = self._base / focus_key
        focus_dir.mkdir(parents=True, exist_ok=True)
        weight_path = str(focus_dir / f"v{ver:04d}.pt")

        if save_fn:
            try:
                save_fn(network.id, weight_path)
            except Exception:
                logger.exception("Failed to save weights for %s v%d", focus_key, ver)

        mv = ModelVersion(
            version=ver,
            focus=focus_key,
            network_id=network.id,
            name=network.name,
            accuracy=network.performance.accuracy,
            loss=network.performance.loss,
            total_parameters=network.topology.total_parameters,
            is_active=False,
            created_at=_time.time(),
            path=weight_path,
            topology_json=_topology_to_dict(network.topology),
        )

        self._versions.setdefault(focus_key, []).append(mv)
        self._prune(focus_key)
        self._save_state()
        return mv

    def promote(self, focus: str, version: int) -> bool:
        """Mark a version as the active model for its focus area."""
        versions = self._versions.get(focus, [])
        found = False
        for v in versions:
            if v.version == version:
                v.is_active = True
                found = True
            else:
                v.is_active = False
        if found:
            self._save_state()
        return found

    def get_active(self, focus: str) -> ModelVersion | None:
        for v in self._versions.get(focus, []):
            if v.is_active:
                return v
        return None

    def deactivate(self, focus: str, network_id: str | None = None) -> bool:
        """Clear the active flag for a focus, optionally only for one network."""
        versions = self._versions.get(focus, [])
        changed = False
        for v in versions:
            if not v.is_active:
                continue
            if network_id is not None and v.network_id != network_id:
                continue
            v.is_active = False
            changed = True
        if changed:
            self._save_state()
        return changed

    def discard_version(
        self,
        focus: str,
        version: int,
        *,
        delete_weights: bool = False,
        reason: str = "",
    ) -> bool:
        """Remove a persisted version from the registry, optionally deleting weights.

        Used when a checkpoint is irrecoverably stale or incompatible with the
        current topology builder and should stop poisoning future boots.
        """
        versions = self._versions.get(focus, [])
        target = next((v for v in versions if v.version == version), None)
        if target is None:
            return False

        if delete_weights:
            try:
                if os.path.exists(target.path):
                    os.remove(target.path)
            except OSError:
                logger.warning("Failed to delete hemisphere weights for %s v%d", focus, version)

        versions.remove(target)

        # If the removed version was active, promote the best remaining version
        # so restore can still recover something meaningful on next boot.
        if target.is_active and versions:
            best = max(versions, key=lambda v: (v.accuracy, -v.loss, v.created_at))
            best.is_active = True

        self._save_state()
        logger.info(
            "Discarded %s hemisphere version v%d%s%s",
            focus,
            version,
            " and deleted weights" if delete_weights else "",
            f" ({reason})" if reason else "",
        )
        return True

    def get_versions(self, focus: str) -> list[ModelVersion]:
        return list(self._versions.get(focus, []))

    def get_all_active(self) -> dict[str, ModelVersion]:
        result: dict[str, ModelVersion] = {}
        for focus, versions in self._versions.items():
            for v in versions:
                if v.is_active:
                    result[focus] = v
        return result

    # ------------------------------------------------------------------
    # Snapshot for dashboard
    # ------------------------------------------------------------------

    def get_state(self, evolution_generations: dict[str, int] | None = None) -> HemisphereState:
        gens = evolution_generations or {}
        snapshots: list[HemisphereSnapshot] = []
        total_active = 0
        total_params = 0

        for focus_enum in HemisphereFocus:
            fk = focus_enum.value
            versions = self._versions.get(fk, [])
            active = self.get_active(fk)
            best_acc = max((v.accuracy for v in versions), default=0.0)
            best_loss = min((v.loss for v in versions), default=float("inf"))
            best_train = max((getattr(v, "training_accuracy", v.accuracy) for v in versions), default=0.0)
            best_val = max((getattr(v, "validation_accuracy", v.accuracy) for v in versions), default=0.0)
            active_count = 1 if active else 0
            total_active += active_count
            total_params += active.total_parameters if active else 0

            snapshots.append(HemisphereSnapshot(
                focus=focus_enum,
                network_count=active_count,
                active_network_id=active.network_id if active else None,
                best_accuracy=best_acc,
                best_loss=best_loss,
                best_training_accuracy=best_train,
                best_validation_accuracy=best_val,
                total_attempts=len(versions),
                evolution_generations=gens.get(fk, 0),
                migration_readiness=active.accuracy if active else 0.0,
                status="active" if active else ("idle" if not versions else "ready"),
            ))

        overall_mr = sum(s.migration_readiness for s in snapshots) / max(len(snapshots), 1)

        return HemisphereState(
            hemispheres=tuple(snapshots),
            total_networks=total_active,
            total_parameters=total_params,
            active_substrate=SubstrateType.RULE_BASED,
            overall_migration_readiness=overall_mr,
            evolution_active=total_active > 0,
            last_cycle_time=_time.time(),
            timestamp=_time.time(),
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _prune(self, focus: str) -> None:
        versions = self._versions.get(focus, [])
        while len(versions) > MAX_VERSIONS_PER_FOCUS:
            oldest = next((v for v in versions if not v.is_active), None)
            if oldest is None:
                break
            try:
                if os.path.exists(oldest.path):
                    os.remove(oldest.path)
            except OSError:
                pass
            versions.remove(oldest)

    def _save_state(self) -> None:
        from memory.persistence import atomic_write_json
        state_path = self._base / "registry_state.json"
        data = {
            "versions": {
                k: [_mv_to_dict(v) for v in vs]
                for k, vs in self._versions.items()
            },
            "next_version": self._next_version,
        }
        try:
            atomic_write_json(state_path, data, indent=2)
        except Exception:
            logger.exception("Failed to save hemisphere registry state")

    def _load_state(self) -> None:
        state_path = self._base / "registry_state.json"
        if not state_path.exists():
            return
        try:
            data = json.loads(state_path.read_text())
            self._next_version = data.get("next_version", {})
            for focus, items in data.get("versions", {}).items():
                self._versions[focus] = [_dict_to_mv(d) for d in items]
            pruned = False
            for focus in list(self._versions):
                before = len(self._versions[focus])
                self._prune(focus)
                if len(self._versions[focus]) < before:
                    pruned = True
            if pruned:
                self._save_state()
                logger.info("Pruned stale hemisphere versions on load (cap=%d per focus)", MAX_VERSIONS_PER_FOCUS)
            total_versions = sum(len(v) for v in self._versions.values())
            active_count = sum(
                1 for vs in self._versions.values()
                for v in vs if v.is_active
            )
            has_topo = sum(
                1 for vs in self._versions.values()
                for v in vs if v.is_active and v.topology_json
            )
            logger.info(
                "Hemisphere registry loaded: %d versions across %d focuses, "
                "%d active, %d with topology_json",
                total_versions, len(self._versions), active_count, has_topo,
            )
        except Exception:
            logger.exception("Failed to load hemisphere registry state")


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _mv_to_dict(mv: ModelVersion) -> dict[str, Any]:
    return {
        "version": mv.version,
        "focus": mv.focus,
        "network_id": mv.network_id,
        "name": mv.name,
        "accuracy": mv.accuracy,
        "loss": mv.loss,
        "total_parameters": mv.total_parameters,
        "is_active": mv.is_active,
        "created_at": mv.created_at,
        "path": mv.path,
        "topology_json": mv.topology_json,
    }


def _dict_to_mv(d: dict[str, Any]) -> ModelVersion:
    return ModelVersion(
        version=d["version"],
        focus=d["focus"],
        network_id=d["network_id"],
        name=d["name"],
        accuracy=d["accuracy"],
        loss=d["loss"],
        total_parameters=d["total_parameters"],
        is_active=d["is_active"],
        created_at=d["created_at"],
        path=d["path"],
        topology_json=d.get("topology_json", {}),
    )


def _topology_to_dict(t: NetworkTopology) -> dict[str, Any]:
    return {
        "input_size": t.input_size,
        "output_size": t.output_size,
        "total_parameters": t.total_parameters,
        "layers": [
            {"id": la.id, "layer_type": la.layer_type,
             "node_count": la.node_count, "activation": la.activation,
             "dropout": la.dropout}
            for la in t.layers
        ],
    }
