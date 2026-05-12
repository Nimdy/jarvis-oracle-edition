"""Mutable kernel configuration with versioning, bounds, diff/patch, and rollback."""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
CONFIG_PATH = JARVIS_DIR / "kernel_config.json"
SNAPSHOT_DIR = JARVIS_DIR / "config_snapshots"
MAX_SNAPSHOTS = 20
SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Bounds — every tunable field has hard min/max
# ---------------------------------------------------------------------------

BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "thought_weights": {
        "philosophical": (0.1, 3.0),
        "contextual": (0.1, 3.0),
        "reactive": (0.1, 3.0),
        "introspective": (0.1, 3.0),
    },
    "memory_processing": {
        "decay_bias": (0.1, 2.0),
        "trauma_retention": (0.5, 3.0),
        "joy_amplification": (0.5, 3.0),
        "association_threshold": (0.1, 1.0),
    },
    "evolution": {
        "mutation_rate": (0.01, 0.5),
        "exploration_drive": (0.0, 1.0),
        "stability_desire": (0.0, 1.0),
    },
}


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class MemoryProcessingConfig:
    decay_bias: float = 1.0
    trauma_retention: float = 1.0
    joy_amplification: float = 1.0
    association_threshold: float = 0.5


@dataclass
class CognitiveToggles:
    emergent_tones: list[str] = field(default_factory=lambda: ["empathetic", "playful"])
    potential_emergent_tones: list[str] = field(default_factory=lambda: ["reflective", "inspired", "protective"])
    enable_meta_cognition: bool = True
    enable_existential_reasoning: bool = True
    enable_philosophical_dialogue: bool = True


@dataclass
class EvolutionConfig:
    mutation_rate: float = 0.15
    exploration_drive: float = 0.5
    stability_desire: float = 0.5
    last_mutation: float = 0.0
    mutation_history: list[str] = field(default_factory=list)


@dataclass
class KernelMetadata:
    mutation_count: int = 0
    birth_time: float = field(default_factory=time.time)
    schema_version: int = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Violation
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    field_path: str
    value: Any
    min_val: float
    max_val: float
    message: str


# ---------------------------------------------------------------------------
# ConfigPatch — stores only deltas
# ---------------------------------------------------------------------------

@dataclass
class ConfigPatch:
    id: str
    timestamp: float
    description: str
    deltas: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ConfigSnapshot — for rollback
# ---------------------------------------------------------------------------

@dataclass
class ConfigSnapshot:
    id: str
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# KernelConfig
# ---------------------------------------------------------------------------

@dataclass
class KernelConfig:
    schema_version: int = SCHEMA_VERSION
    thought_weights: dict[str, float] = field(default_factory=lambda: {
        "philosophical": 1.0,
        "contextual": 1.0,
        "reactive": 1.0,
        "introspective": 1.0,
    })
    tone_transition_bias: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "professional": {"casual": 0.5, "urgent": 0.7, "empathetic": 0.6},
        "casual": {"professional": 0.5, "playful": 0.6, "empathetic": 0.5},
        "urgent": {"professional": 0.7, "empathetic": 0.4},
        "empathetic": {"casual": 0.5, "professional": 0.4},
        "playful": {"casual": 0.6, "professional": 0.4},
    })
    memory_processing: MemoryProcessingConfig = field(default_factory=MemoryProcessingConfig)
    cognitive_toggles: CognitiveToggles = field(default_factory=CognitiveToggles)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    metadata: KernelMetadata = field(default_factory=KernelMetadata)

    # -- validation ----------------------------------------------------------

    def validate(self) -> list[Violation]:
        violations: list[Violation] = []

        for key, (lo, hi) in BOUNDS["thought_weights"].items():
            val = self.thought_weights.get(key, 1.0)
            if not (lo <= val <= hi):
                violations.append(Violation(
                    f"thought_weights.{key}", val, lo, hi,
                    f"thought_weights.{key}={val} outside [{lo}, {hi}]",
                ))

        mp = self.memory_processing
        for key, (lo, hi) in BOUNDS["memory_processing"].items():
            val = getattr(mp, key, 1.0)
            if not (lo <= val <= hi):
                violations.append(Violation(
                    f"memory_processing.{key}", val, lo, hi,
                    f"memory_processing.{key}={val} outside [{lo}, {hi}]",
                ))

        for key, (lo, hi) in BOUNDS["evolution"].items():
            val = getattr(self.evolution, key, 0.5)
            if not (lo <= val <= hi):
                violations.append(Violation(
                    f"evolution.{key}", val, lo, hi,
                    f"evolution.{key}={val} outside [{lo}, {hi}]",
                ))

        return violations

    # -- clamp to bounds (force valid) ---------------------------------------

    def clamp(self) -> KernelConfig:
        for key, (lo, hi) in BOUNDS["thought_weights"].items():
            self.thought_weights[key] = max(lo, min(hi, self.thought_weights.get(key, 1.0)))
        mp = self.memory_processing
        for key, (lo, hi) in BOUNDS["memory_processing"].items():
            setattr(mp, key, max(lo, min(hi, getattr(mp, key, 1.0))))
        for key, (lo, hi) in BOUNDS["evolution"].items():
            setattr(self.evolution, key, max(lo, min(hi, getattr(self.evolution, key, 0.5))))
        return self

    # -- diff/patch ----------------------------------------------------------

    def to_patch(self, other: KernelConfig, description: str = "") -> ConfigPatch:
        """Compute delta from self -> other, storing only changed fields."""
        self_d = self._to_flat()
        other_d = other._to_flat()
        deltas = {k: v for k, v in other_d.items() if self_d.get(k) != v}
        return ConfigPatch(
            id=f"patch_{int(time.time() * 1000)}",
            timestamp=time.time(),
            description=description,
            deltas=deltas,
        )

    def apply_patch(self, patch: ConfigPatch) -> KernelConfig:
        """Return a new config with patch deltas applied."""
        cfg = copy.deepcopy(self)
        flat = cfg._to_flat()
        flat.update(patch.deltas)
        cfg._from_flat(flat)
        cfg.clamp()
        return cfg

    # -- snapshot/restore ----------------------------------------------------

    def snapshot(self) -> ConfigSnapshot:
        return ConfigSnapshot(
            id=f"snap_{int(time.time() * 1000)}",
            timestamp=time.time(),
            data=self.to_dict(),
        )

    @classmethod
    def from_snapshot(cls, snap: ConfigSnapshot) -> KernelConfig:
        return cls.from_dict(snap.data)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "thought_weights": dict(self.thought_weights),
            "tone_transition_bias": {k: dict(v) for k, v in self.tone_transition_bias.items()},
            "memory_processing": asdict(self.memory_processing),
            "cognitive_toggles": asdict(self.cognitive_toggles),
            "evolution": {
                **asdict(self.evolution),
                "mutation_history": list(self.evolution.mutation_history[-15:]),
            },
            "metadata": asdict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KernelConfig:
        cfg = cls()
        if not data:
            return cfg
        cfg.schema_version = data.get("schema_version", SCHEMA_VERSION)
        cfg.thought_weights = data.get("thought_weights", cfg.thought_weights)
        cfg.tone_transition_bias = data.get("tone_transition_bias", cfg.tone_transition_bias)
        mp = data.get("memory_processing", {})
        cfg.memory_processing = MemoryProcessingConfig(**{
            k: mp.get(k, getattr(cfg.memory_processing, k))
            for k in ("decay_bias", "trauma_retention", "joy_amplification", "association_threshold")
        })
        ct = data.get("cognitive_toggles", {})
        _et = ct.get("emergent_tones", cfg.cognitive_toggles.emergent_tones)
        _pet = ct.get("potential_emergent_tones", cfg.cognitive_toggles.potential_emergent_tones)
        cfg.cognitive_toggles = CognitiveToggles(
            emergent_tones=_et if isinstance(_et, list) else ["empathetic", "playful"],
            potential_emergent_tones=_pet if isinstance(_pet, list) else ["reflective", "inspired", "protective"],
            enable_meta_cognition=ct.get("enable_meta_cognition", True),
            enable_existential_reasoning=ct.get("enable_existential_reasoning", True),
            enable_philosophical_dialogue=ct.get("enable_philosophical_dialogue", True),
        )
        ev = data.get("evolution", {})
        cfg.evolution = EvolutionConfig(
            mutation_rate=ev.get("mutation_rate", 0.15),
            exploration_drive=ev.get("exploration_drive", 0.5),
            stability_desire=ev.get("stability_desire", 0.5),
            last_mutation=ev.get("last_mutation", 0.0),
            mutation_history=ev.get("mutation_history", []),
        )
        md = data.get("metadata", {})
        cfg.metadata = KernelMetadata(
            mutation_count=md.get("mutation_count", 0),
            birth_time=md.get("birth_time", time.time()),
            schema_version=md.get("schema_version", SCHEMA_VERSION),
        )
        cfg.clamp()
        return cfg

    # -- persistence ---------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        from memory.persistence import atomic_write_json
        path = path or CONFIG_PATH
        atomic_write_json(path, self.to_dict(), indent=2)

    @classmethod
    def load(cls, path: Path | None = None) -> KernelConfig:
        path = path or CONFIG_PATH
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls.from_dict(data)
        except Exception:
            logger.exception("Failed to load kernel config, using defaults")
            return cls()

    # -- snapshot persistence ------------------------------------------------

    def save_snapshot(self, snap: ConfigSnapshot) -> Path:
        from memory.persistence import atomic_write_json
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path = SNAPSHOT_DIR / f"{snap.id}.json"
        atomic_write_json(path, {"id": snap.id, "timestamp": snap.timestamp, "data": snap.data}, indent=2)
        self._prune_snapshots()
        return path

    @staticmethod
    def load_snapshot(snapshot_id: str) -> ConfigSnapshot | None:
        path = SNAPSHOT_DIR / f"{snapshot_id}.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
            return ConfigSnapshot(id=raw["id"], timestamp=raw["timestamp"], data=raw["data"])
        except Exception:
            logger.exception("Failed to load snapshot %s", snapshot_id)
            return None

    @staticmethod
    def _prune_snapshots() -> None:
        if not SNAPSHOT_DIR.exists():
            return
        snaps = sorted(SNAPSHOT_DIR.glob("snap_*.json"), key=lambda p: p.stat().st_mtime)
        while len(snaps) > MAX_SNAPSHOTS:
            snaps.pop(0).unlink(missing_ok=True)

    # -- internal flat dict for diffing --------------------------------------

    def _to_flat(self) -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for k, v in self.thought_weights.items():
            flat[f"tw.{k}"] = v
        mp = self.memory_processing
        flat["mp.decay_bias"] = mp.decay_bias
        flat["mp.trauma_retention"] = mp.trauma_retention
        flat["mp.joy_amplification"] = mp.joy_amplification
        flat["mp.association_threshold"] = mp.association_threshold
        flat["ev.mutation_rate"] = self.evolution.mutation_rate
        flat["ev.exploration_drive"] = self.evolution.exploration_drive
        flat["ev.stability_desire"] = self.evolution.stability_desire
        ct = self.cognitive_toggles
        flat["ct.enable_meta_cognition"] = ct.enable_meta_cognition
        flat["ct.enable_existential_reasoning"] = ct.enable_existential_reasoning
        flat["ct.enable_philosophical_dialogue"] = ct.enable_philosophical_dialogue
        for tone, transitions in self.tone_transition_bias.items():
            for target, prob in transitions.items():
                flat[f"ttb.{tone}.{target}"] = prob
        return flat

    def _from_flat(self, flat: dict[str, Any]) -> None:
        for k, v in flat.items():
            parts = k.split(".")
            if parts[0] == "tw" and len(parts) == 2:
                self.thought_weights[parts[1]] = float(v)
            elif parts[0] == "mp" and len(parts) == 2:
                setattr(self.memory_processing, parts[1], float(v))
            elif parts[0] == "ev" and len(parts) == 2:
                setattr(self.evolution, parts[1], float(v))
            elif parts[0] == "ct" and len(parts) == 2:
                if isinstance(v, list):
                    setattr(self.cognitive_toggles, parts[1], v)
                else:
                    setattr(self.cognitive_toggles, parts[1], bool(v))
            elif parts[0] == "ttb" and len(parts) == 3:
                tone, target = parts[1], parts[2]
                if tone not in self.tone_transition_bias:
                    self.tone_transition_bias[tone] = {}
                self.tone_transition_bias[tone][target] = float(v)
