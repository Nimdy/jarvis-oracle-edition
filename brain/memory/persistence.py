"""Memory + consciousness persistence — auto-save/load to disk."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from memory.storage import memory_storage

logger = logging.getLogger(__name__)

PERSISTENCE_SCHEMA_VERSION = 2


def atomic_write_json(path: Path | str, data: Any, **json_kwargs) -> None:
    """Write JSON atomically: write to temp file then os.replace (POSIX-atomic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, **json_kwargs)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

JARVIS_DIR = Path.home() / ".jarvis"
MEMORIES_PATH = JARVIS_DIR / "memories.json"
CONSCIOUSNESS_STATE_PATH = JARVIS_DIR / "consciousness_state.json"


class MemoryPersistence:
    def __init__(self, path: str = "", interval_s: float = 60.0) -> None:
        self._path = path or str(MEMORIES_PATH)
        self._interval_s = interval_s
        self._task: asyncio.Task | None = None

    def load(self) -> int:
        if not os.path.exists(self._path):
            logger.info("No persisted memories at %s", self._path)
            return 0
        try:
            with open(self._path) as f:
                data = json.load(f)
            loaded = memory_storage.load_from_json(data)
            logger.info("Loaded %d memories from %s", loaded, self._path)
            return loaded
        except Exception:
            logger.exception("Failed to load memories from %s", self._path)
            return 0

    def save(self) -> bool:
        try:
            data = memory_storage.to_json()
            atomic_write_json(self._path, data, default=str)
            logger.debug("Saved %d memories to %s", len(data), self._path)
            return True
        except Exception:
            logger.exception("Failed to save memories to %s", self._path)
            return False

    def start_auto_save(self) -> None:
        if self._task:
            return
        self._task = asyncio.get_event_loop().create_task(self._auto_save_loop())

    def stop_auto_save(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None
        self.save()

    async def _auto_save_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._interval_s)
                self.save()
        except asyncio.CancelledError:
            pass


class ConsciousnessPersistence:
    """Auto-save/load consciousness state (evolution, observer, mutations) to ~/.jarvis/."""

    _SUBSYSTEMS = ("evolution", "observer", "driven_evolution", "kernel_config", "governor", "analytics")
    _GESTATION_KEYS = ("gestation_in_progress", "gestation_complete", "gestation_completed_at")

    def __init__(self) -> None:
        self._path = CONSCIOUSNESS_STATE_PATH
        self._instance_id = self._get_or_create_instance_id()
        self._boot_id = uuid.uuid4().hex[:12]
        self._auto_save_task: asyncio.Task | None = None
        self._sticky_merge_state: dict[str, Any] = {}

    @staticmethod
    def _get_or_create_instance_id() -> str:
        path = JARVIS_DIR / "instance_id"
        if path.exists():
            return path.read_text().strip()
        iid = uuid.uuid4().hex[:16]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(iid)
        return iid

    def _capture_sticky_merge_state(self, data: dict[str, Any]) -> None:
        for key in self._GESTATION_KEYS:
            if key in data:
                self._sticky_merge_state[key] = data[key]
            else:
                self._sticky_merge_state.pop(key, None)

    def update_gestation_sticky(self, gestation_keys: dict[str, Any]) -> None:
        """Update sticky gestation keys from an external writer (e.g. GestationManager).

        This must be called whenever gestation state is written directly to
        the consciousness_state file, so the next periodic save_from_system()
        doesn't overwrite it with stale in-memory state.
        """
        for key in self._GESTATION_KEYS:
            if key in gestation_keys:
                self._sticky_merge_state[key] = gestation_keys[key]
            else:
                self._sticky_merge_state.pop(key, None)

    def _ensure_sticky_merge_state_loaded(self) -> None:
        if self._sticky_merge_state or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            if isinstance(data, dict):
                self._capture_sticky_merge_state(data)
        except Exception:
            logger.debug("Failed to warm sticky merge state from %s", self._path, exc_info=True)

    def load(self, *, log_success: bool = True) -> dict[str, Any]:
        if not self._path.exists():
            logger.info("No persisted consciousness state at %s", self._path)
            return {}
        try:
            data = json.loads(self._path.read_text())
            if isinstance(data, dict):
                self._capture_sticky_merge_state(data)
            if log_success:
                logger.info("Loaded consciousness state from %s", self._path)
            return data
        except Exception:
            logger.exception("Failed to load consciousness state from %s", self._path)
            return {}

    def save(self, state: dict[str, Any]) -> bool:
        try:
            atomic_write_json(self._path, state, indent=2, default=str)
            logger.debug("Saved consciousness state to %s", self._path)
            return True
        except Exception:
            logger.exception("Failed to save consciousness state to %s", self._path)
            return False

    def save_from_system(self, consciousness_system: Any, engine: Any = None) -> bool:
        """Save directly from a ConsciousnessSystem instance."""
        if engine and not getattr(engine, "_restore_complete", True):
            logger.debug("Consciousness auto-save skipped: restore not yet complete")
            return False
        try:
            evo_state = consciousness_system.evolution.get_state()
            observer_state = consciousness_system.observer.get_state()
            driven_state = consciousness_system.driven_evolution.get_state()
            config_data = consciousness_system.config.to_dict()

            epistemic_block: dict[str, Any] = {}
            try:
                from epistemic.contradiction_engine import ContradictionEngine
                ce = ContradictionEngine.get_instance()
                if ce is not None:
                    epistemic_block["contradiction_debt"] = round(ce.contradiction_debt, 6)
            except Exception:
                pass

            state = {
                "_provenance": {
                    "schema_version": PERSISTENCE_SCHEMA_VERSION,
                    "instance_id": self._instance_id,
                    "boot_id": self._boot_id,
                    "pid": os.getpid(),
                    "saved_at_ts": time.time(),
                },
                "evolution": evo_state.to_dict(),
                "observer": observer_state.to_dict(),
                "driven_evolution": driven_state,
                "kernel_config": config_data,
                "governor": {
                    "mutation_count": consciousness_system.governor.mutation_count,
                    "rollback_count": consciousness_system.governor.rollback_count,
                    "last_mutation_time": consciousness_system.governor.last_mutation_time,
                    "mutation_timestamps": list(consciousness_system.governor._mutation_timestamps),
                },
                "analytics": consciousness_system.analytics.get_full_state(),
                "epistemic": epistemic_block,
            }

            self._ensure_sticky_merge_state_loaded()
            for gk in self._GESTATION_KEYS:
                if gk in self._sticky_merge_state and gk not in state:
                    state[gk] = self._sticky_merge_state[gk]

            result = self.save(state)
            if result:
                self._capture_sticky_merge_state(state)

            try:
                extended_persistence.save_all()
            except Exception:
                logger.debug("Extended persistence save failed (non-critical)")

            return result
        except Exception:
            logger.exception("Failed to save consciousness state from system")
            return False

    def restore_to_system(self, consciousness_system: Any) -> bool:
        """Restore persisted state into a ConsciousnessSystem instance.

        Restores all 6 subsystems: evolution, observer, driven_evolution,
        kernel_config, governor, analytics. Logs which keys were present/missing.
        """
        data = self.load()
        if not data:
            return False

        prov = data.get("_provenance", {})
        if prov:
            logger.info(
                "Consciousness state provenance: schema=%s instance=%s boot=%s pid=%s saved=%.0f",
                prov.get("schema_version", "?"), prov.get("instance_id", "?"),
                prov.get("boot_id", "?"), prov.get("pid", "?"),
                prov.get("saved_at_ts", 0),
            )

        present = [k for k in self._SUBSYSTEMS if data.get(k) is not None]
        missing = [k for k in self._SUBSYSTEMS if data.get(k) is None]
        logger.info("Consciousness restore: present=%s missing=%s", present, missing)

        restored_count = 0
        try:
            observer_data = data.get("observer")
            gov_data = data.get("governor")

            persisted_observation_count = 0
            persisted_awareness_level = 0.0
            persisted_mutation_count = 0
            if observer_data is not None:
                persisted_observation_count = observer_data.get("observation_count", 0)
                persisted_awareness_level = observer_data.get("awareness_level", 0.0)
            if gov_data is not None:
                persisted_mutation_count = gov_data.get("mutation_count", 0)

            evo_data = data.get("evolution")
            if evo_data is not None:
                consciousness_system.evolution.load_state(
                    evo_data,
                    observation_count=persisted_observation_count,
                    mutation_count=persisted_mutation_count,
                    awareness_level=persisted_awareness_level,
                )
                restored_count += 1

                trust_info = consciousness_system.evolution.get_restore_trust()
                logger.info("Evolution restore trust: %s (anomalies=%d)",
                            trust_info.get("trust", "?"), trust_info.get("anomaly_count", 0))

            if observer_data is not None:
                consciousness_system.observer.load_state(observer_data)
                restored_count += 1

            driven_data = data.get("driven_evolution")
            if driven_data is not None:
                consciousness_system.driven_evolution.load_state(driven_data)
                restored_count += 1

            config_data = data.get("kernel_config")
            if config_data is not None:
                from consciousness.kernel_config import KernelConfig
                consciousness_system.config = KernelConfig.from_dict(config_data)
                restored_count += 1

            if gov_data is not None:
                consciousness_system.governor._mutation_count = gov_data.get("mutation_count", 0)
                consciousness_system.governor._rollback_count = gov_data.get("rollback_count", 0)
                consciousness_system.governor._last_mutation_time = gov_data.get("last_mutation_time", 0.0)
                now = time.time()
                raw_timestamps = gov_data.get("mutation_timestamps", [])
                one_hour_ago = now - 3600.0
                consciousness_system.governor._mutation_timestamps = [
                    ts for ts in raw_timestamps
                    if isinstance(ts, (int, float)) and ts > one_hour_ago
                ]
                restored_count += 1

            analytics_data = data.get("analytics")
            if analytics_data is not None:
                consciousness_system.analytics.load_state(analytics_data)
                restored_count += 1

            self._persisted_epistemic = data.get("epistemic", {})

            logger.info("Consciousness state restored: %d/%d subsystems from persistence",
                        restored_count, len(self._SUBSYSTEMS))
            return True
        except Exception:
            logger.exception("Failed to restore consciousness state")
            return False

    def get_boot_provenance(self) -> dict[str, Any]:
        """Return provenance info for boot banner."""
        info: dict[str, Any] = {
            "path": str(self._path),
            "instance_id": self._instance_id,
            "boot_id": self._boot_id,
        }
        if self._path.exists():
            stat = self._path.stat()
            info["mtime"] = stat.st_mtime
            info["size_bytes"] = stat.st_size
            try:
                data = json.loads(self._path.read_text())
                prov = data.get("_provenance", {})
                info["file_instance_id"] = prov.get("instance_id", "?")
                info["file_boot_id"] = prov.get("boot_id", "?")
                info["file_pid"] = prov.get("pid", "?")
                info["file_saved_at"] = prov.get("saved_at_ts", 0)
                info["schema_version"] = prov.get("schema_version", 0)
            except Exception:
                pass
        return info

    def start_auto_save(self, consciousness_system: Any, engine: Any, interval_s: float = 60.0) -> None:
        if self._auto_save_task:
            return
        self._auto_save_task = asyncio.get_event_loop().create_task(
            self._auto_save_loop(consciousness_system, engine, interval_s)
        )

    def stop_auto_save(self) -> None:
        if self._auto_save_task:
            self._auto_save_task.cancel()
            self._auto_save_task = None

    async def _auto_save_loop(self, consciousness_system: Any, engine: Any, interval_s: float) -> None:
        try:
            while True:
                await asyncio.sleep(interval_s)
                self.save_from_system(consciousness_system, engine=engine)
                try:
                    if hasattr(engine, '_autonomy_orchestrator') and engine._autonomy_orchestrator:
                        engine._autonomy_orchestrator.save_calibration()
                except Exception:
                    pass
                try:
                    if hasattr(engine, '_hemisphere_orchestrator') and engine._hemisphere_orchestrator:
                        engine._hemisphere_orchestrator._gap_detector.save_state()
                except Exception:
                    pass
                try:
                    save_intention_registry()
                except Exception:
                    logger.debug("intention_registry auto-save failed", exc_info=True)
        except asyncio.CancelledError:
            pass


class ExtendedPersistence:
    """Save/load for new soul kernel systems: causal models, personality snapshots, clusters, reports."""

    def __init__(self) -> None:
        self._causal_path = JARVIS_DIR / "causal_models.json"
        self._personality_path = JARVIS_DIR / "personality_snapshots.json"
        self._clusters_path = JARVIS_DIR / "memory_clusters.json"
        self._reports_path = JARVIS_DIR / "consciousness_reports.json"

    def save_causal_models(self) -> bool:
        try:
            from consciousness.epistemic_reasoning import epistemic_engine
            data = epistemic_engine.get_models()
            return self._write_json(self._causal_path, data)
        except Exception:
            logger.exception("Failed to save causal models")
            return False

    def load_causal_models(self) -> bool:
        try:
            data = self._read_json(self._causal_path)
            if not data:
                return False
            from consciousness.epistemic_reasoning import epistemic_engine
            for model_data in data:
                model_id = model_data.get("id", "")
                if model_id and model_id in epistemic_engine._models:
                    m = epistemic_engine._models[model_id]
                    m.confidence = model_data.get("confidence", m.confidence)
                    m.evidence_count = model_data.get("evidence_count", m.evidence_count)
            logger.info("Loaded causal models from %s", self._causal_path)
            return True
        except Exception:
            logger.exception("Failed to load causal models")
            return False

    def save_personality_snapshots(self) -> bool:
        try:
            from personality.rollback import personality_rollback
            state = personality_rollback.get_state()
            return self._write_json(self._personality_path, state)
        except Exception:
            logger.exception("Failed to save personality snapshots")
            return False

    def load_personality_snapshots(self) -> bool:
        try:
            data = self._read_json(self._personality_path)
            if not data:
                return False
            from personality.rollback import personality_rollback
            traits = data.get("current_traits", {})
            if traits:
                personality_rollback.update_traits(traits)
            logger.info("Loaded personality snapshots from %s", self._personality_path)
            return True
        except Exception:
            logger.exception("Failed to load personality snapshots")
            return False

    def save_clusters(self) -> bool:
        try:
            from memory.clustering import memory_cluster_engine
            data = memory_cluster_engine.get_clusters_full()
            return self._write_json(self._clusters_path, data)
        except Exception:
            logger.exception("Failed to save memory clusters")
            return False

    def load_clusters(self) -> bool:
        try:
            data = self._read_json(self._clusters_path)
            if not data:
                return False
            from memory.clustering import memory_cluster_engine
            from memory.storage import memory_storage
            valid_ids: set[str] | None = None
            try:
                valid_ids = {m.id for m in memory_storage.get_all()}
            except Exception:
                logger.debug("Could not get memory IDs for cluster pruning")
            restored = memory_cluster_engine.restore_clusters(data, valid_ids)
            logger.info("Restored %d memory clusters from %s", restored, self._clusters_path)
            return restored > 0
        except Exception:
            logger.exception("Failed to load memory clusters")
            return False

    def save_reports(self) -> bool:
        try:
            from consciousness.communication import consciousness_communicator
            reports = consciousness_communicator.get_recent_reports(10)
            return self._write_json(self._reports_path, reports)
        except Exception:
            logger.exception("Failed to save consciousness reports")
            return False

    def load_reports(self) -> bool:
        try:
            data = self._read_json(self._reports_path)
            if data:
                logger.info("Loaded %d consciousness reports from %s", len(data), self._reports_path)
            return bool(data)
        except Exception:
            logger.exception("Failed to load consciousness reports")
            return False

    def save_all(self) -> dict[str, bool]:
        return {
            "causal_models": self.save_causal_models(),
            "personality_snapshots": self.save_personality_snapshots(),
            "clusters": self.save_clusters(),
            "reports": self.save_reports(),
        }

    def load_all(self) -> dict[str, bool]:
        return {
            "causal_models": self.load_causal_models(),
            "personality_snapshots": self.load_personality_snapshots(),
            "clusters": self.load_clusters(),
            "reports": self.load_reports(),
        }

    def _write_json(self, path: Path, data: Any) -> bool:
        try:
            atomic_write_json(path, data, indent=2, default=str)
            logger.debug("Saved to %s", path)
            return True
        except Exception:
            logger.exception("Failed to write %s", path)
            return False

    def _read_json(self, path: Path) -> Any:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            logger.exception("Failed to read %s", path)
            return None


memory_persistence = MemoryPersistence()
consciousness_persistence = ConsciousnessPersistence()
extended_persistence = ExtendedPersistence()


def load_intention_registry() -> int:
    """Load persisted intention registry on boot. Safe no-op on failure.

    Returns the number of open intentions restored, or 0 on any failure.
    """
    try:
        from cognition.intention_registry import intention_registry
        return intention_registry.load()
    except Exception:
        logger.debug("intention_registry load skipped (import or load failed)", exc_info=True)
        return 0


def save_intention_registry() -> bool:
    """Save current intention registry state. Safe no-op on failure."""
    try:
        from cognition.intention_registry import intention_registry
        return intention_registry.save()
    except Exception:
        logger.debug("intention_registry save skipped (import or save failed)", exc_info=True)
        return False
