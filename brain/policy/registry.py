"""Model Registry — versioned model storage, promotion, rollback.

Models stored at ~/.jarvis/policy_models/policy_vNNNN.pt.
Active model tracked in policy_state.json.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
MODEL_DIR = JARVIS_DIR / "policy_models"
STATE_FILE = MODEL_DIR / "policy_state.json"
MAX_VERSIONS = 20


@dataclass
class ModelVersion:
    version: int
    arch: str
    created_at: float
    validation_loss: float
    shadow_win_rate: float
    is_active: bool = False
    path: str = ""


@dataclass
class RegistryState:
    active_version: int = 0
    total_versions: int = 0
    versions: list[ModelVersion] = field(default_factory=list)


class ModelRegistry:
    def __init__(self) -> None:
        self._state = RegistryState()
        self._load_state()

    def register(
        self,
        arch: str,
        validation_loss: float,
        shadow_win_rate: float,
        model_saver: Any = None,
    ) -> ModelVersion:
        """Register a new model version. Optionally save weights via model_saver(path)."""
        self._state.total_versions += 1
        version = self._state.total_versions

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = str(MODEL_DIR / f"policy_v{version:04d}.pt")

        entry = ModelVersion(
            version=version,
            arch=arch,
            created_at=time.time(),
            validation_loss=validation_loss,
            shadow_win_rate=shadow_win_rate,
            path=model_path,
        )

        if model_saver:
            try:
                model_saver(model_path)
            except Exception:
                logger.exception("Failed to save model weights for v%d", version)

        self._state.versions.append(entry)
        self._prune_old_versions()
        self._save_state()

        from policy.telemetry import policy_telemetry
        policy_telemetry.update_registry(
            total=self._state.total_versions,
            active_version=self._state.active_version,
            active_arch=(self.get_active() or entry).arch,
        )
        policy_telemetry.log_event("model_registered",
                                   f"v{version:04d} arch={arch} loss={validation_loss:.4f}")

        logger.info("Registered model v%04d (arch=%s, loss=%.4f, win_rate=%.2f)",
                     version, arch, validation_loss, shadow_win_rate)
        return entry

    def promote(self, version: int) -> bool:
        """Make a version the active model."""
        for v in self._state.versions:
            v.is_active = (v.version == version)

        self._state.active_version = version
        self._save_state()

        promoted = self.get_active()
        if promoted:
            from policy.telemetry import policy_telemetry
            policy_telemetry.record_promotion(version, promoted.arch)
            policy_telemetry.update_registry(
                total=self._state.total_versions,
                active_version=version,
                active_arch=promoted.arch,
            )

        logger.info("Promoted model v%04d to active", version)
        return True

    def get_active(self) -> ModelVersion | None:
        for v in self._state.versions:
            if v.is_active:
                return v
        return None

    def get_version(self, version: int) -> ModelVersion | None:
        for v in self._state.versions:
            if v.version == version:
                return v
        return None

    def should_promote(self, candidate: ModelVersion, margin: float = 0.02) -> bool:
        """Returns True if candidate beats current active model by margin."""
        active = self.get_active()
        if active is None:
            return True
        return candidate.validation_loss < active.validation_loss - margin

    def get_status(self) -> dict[str, Any]:
        active = self.get_active()
        return {
            "active_version": self._state.active_version,
            "total_versions": self._state.total_versions,
            "active_arch": active.arch if active else "none",
            "recent_versions": [
                {"version": v.version, "arch": v.arch,
                 "loss": v.validation_loss, "active": v.is_active}
                for v in self._state.versions[-5:]
            ],
        }

    # -- persistence ---------------------------------------------------------

    def _save_state(self) -> None:
        try:
            from memory.persistence import atomic_write_json
            data = {
                "active_version": self._state.active_version,
                "total_versions": self._state.total_versions,
                "versions": [asdict(v) for v in self._state.versions],
            }
            atomic_write_json(STATE_FILE, data, indent=2)
        except Exception:
            logger.exception("Failed to save registry state")

    def _load_state(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text())
            self._state.active_version = data.get("active_version", 0)
            self._state.total_versions = data.get("total_versions", 0)
            self._state.versions = [
                ModelVersion(**v) for v in data.get("versions", [])
            ]
        except Exception:
            logger.exception("Failed to load registry state")

    def _prune_old_versions(self) -> None:
        while len(self._state.versions) > MAX_VERSIONS:
            for i, v in enumerate(self._state.versions):
                if not v.is_active:
                    removed = self._state.versions.pop(i)
                    Path(removed.path).unlink(missing_ok=True)
                    break
            else:
                break
