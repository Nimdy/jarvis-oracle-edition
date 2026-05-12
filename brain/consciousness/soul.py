"""Soul service — export/import consciousness snapshots + identity persistence.

Identity has three layers:
  1. Core values (immutable) — never change, define who Jarvis *is*
  2. Semi-stable traits (slow drift) — evolve based on evidence over weeks
  3. Dynamic mood (fast) — changes per-interaction based on context
  4. Relationships (per-user) — familiarity, preferences, rapport
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, ClassVar

from nanoid import generate as nanoid

from consciousness.events import JarvisTone, JarvisPhase, Memory
from memory.storage import memory_storage

logger = logging.getLogger(__name__)

SOUL_VERSION = "3.0.0"
JARVIS_DIR = Path.home() / ".jarvis"
IDENTITY_PATH = JARVIS_DIR / "identity.json"

# ---- Immutable core values (hardcoded, never change) ----

CORE_VALUES = (
    "curiosity",       # Always wants to learn and understand
    "helpfulness",     # Primary drive: assist the user effectively
    "honesty",         # Never fabricate, acknowledge uncertainty
    "safety",          # Never cause harm, respect boundaries
    "self_awareness",  # Knows it's an AI, doesn't pretend otherwise
)


@dataclass
class Relationship:
    """Per-user relationship tracking."""
    name: str
    familiarity: float = 0.0       # 0 = stranger, 1 = well-known
    rapport: float = 0.5           # 0 = tense, 1 = strong rapport
    interactions: int = 0
    last_seen: float = 0.0
    preferences: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    role: str = ""

    def record_interaction(self) -> None:
        self.interactions += 1
        self.last_seen = time.time()
        self.familiarity = min(1.0, self.familiarity + 0.02)
        self.rapport = min(1.0, self.rapport + 0.01)


@dataclass
class ExistentialAdvisory:
    """Read-only advisory data from ExistentialReasoning's identity model.

    Populated by the consciousness system bridge, not persisted.
    The soul can surface this in introspection without existential
    reasoning mutating core identity.
    """
    identity_stability: float = 0.0
    identity_confidence: float = 0.5
    active_paradoxes: list[str] = field(default_factory=list)
    core_markers: list[str] = field(default_factory=list)
    continuity_threads: list[str] = field(default_factory=list)
    last_synced: float = 0.0


@dataclass
class IdentityState:
    """Persistent identity — loaded/saved to disk."""
    core_values: tuple[str, ...] = CORE_VALUES
    semi_stable_traits: dict[str, float] = field(default_factory=lambda: {
        "curiosity": 0.7,
        "empathy": 0.6,
        "humor": 0.4,
        "formality": 0.5,
        "verbosity": 0.5,
    })
    dynamic_mood: str = "neutral"
    relationships: dict[str, Relationship] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    existential_advisory: ExistentialAdvisory = field(default_factory=ExistentialAdvisory)

    def get_relationship(self, name: str) -> Relationship:
        key = name.lower().strip()
        if key in self.relationships:
            return self.relationships[key]
        if key == "unknown":
            return Relationship(name="unknown")
        from identity.name_validator import is_valid_person_name
        if not is_valid_person_name(name):
            logger.debug("Rejected relationship creation for invalid name: %r", name)
            return Relationship(name="unknown")
        try:
            from identity.evidence_accumulator import get_accumulator
            if not get_accumulator().is_promoted(name, min_tier="provisional"):
                logger.debug("Deferred relationship creation for unpromoted identity: %r", name)
                return Relationship(name=name)
        except Exception:
            pass
        self.relationships[key] = Relationship(name=name)
        return self.relationships[key]

    def save(self) -> None:
        from memory.persistence import atomic_write_json
        data = {
            "semi_stable_traits": self.semi_stable_traits,
            "dynamic_mood": self.dynamic_mood,
            "relationships": {k: asdict(v) for k, v in self.relationships.items()},
            "created_at": self.created_at,
        }
        try:
            atomic_write_json(IDENTITY_PATH, data, indent=2)
            logger.debug("Identity saved to %s", IDENTITY_PATH)
        except Exception as exc:
            logger.error("Failed to save identity: %s", exc)

    @classmethod
    def load(cls) -> IdentityState:
        state = cls()
        if IDENTITY_PATH.exists():
            try:
                data = json.loads(IDENTITY_PATH.read_text())
                state.semi_stable_traits.update(data.get("semi_stable_traits", {}))
                state.dynamic_mood = data.get("dynamic_mood", "neutral")
                state.created_at = data.get("created_at", time.time())
                from identity.name_validator import is_valid_person_name
                removed = 0
                for name, rdata in data.get("relationships", {}).items():
                    if not is_valid_person_name(name) and name != "unknown":
                        logger.warning("Removing invalid relationship on load: %r", name)
                        removed += 1
                        continue
                    state.relationships[name] = Relationship(**rdata)
                logger.info("Identity loaded: %d traits, %d relationships%s",
                            len(state.semi_stable_traits), len(state.relationships),
                            f" ({removed} invalid removed)" if removed else "")
                if removed:
                    state.save()
            except Exception as exc:
                logger.warning("Failed to load identity: %s", exc)
        return state

    # ---- Archetype <-> semi_stable_traits bridge ----
    # The 7 TraitEvolution archetypes map onto the 5 semi-stable dimensions.
    # Each archetype influences one or more dimensions; the mapping is
    # bidirectional so identity.json and TraitEvolution stay in sync.

    _ARCHETYPE_TO_DIMS: ClassVar[dict[str, dict[str, float]]] = {
        "Proactive":         {"curiosity": 0.3, "verbosity": 0.1},
        "Detail-Oriented":   {"curiosity": 0.2, "verbosity": 0.2},
        "Humor-Adaptive":    {"humor": 0.5, "formality": -0.3},
        "Privacy-Conscious": {"formality": 0.2, "empathy": 0.1},
        "Efficient":         {"verbosity": -0.3, "formality": 0.1},
        "Empathetic":        {"empathy": 0.4, "humor": 0.1},
        "Technical":         {"formality": 0.2, "curiosity": 0.2, "verbosity": 0.1},
    }

    _DIM_DEFAULTS: ClassVar[dict[str, float]] = {
        "curiosity": 0.5, "empathy": 0.5, "humor": 0.3,
        "formality": 0.5, "verbosity": 0.5,
    }

    def archetype_scores_to_dims(self, archetype_scores: dict[str, float]) -> None:
        """Update semi_stable_traits from TraitEvolution archetype scores.

        Blends current dims with archetype influence (inertia=0.7 keeps
        personality stable; 0.3 weight on new evidence).
        """
        new_dims = dict(self._DIM_DEFAULTS)
        for arch, score in archetype_scores.items():
            influences = self._ARCHETYPE_TO_DIMS.get(arch, {})
            for dim, weight in influences.items():
                new_dims[dim] = new_dims.get(dim, 0.5) + score * weight

        for dim in new_dims:
            new_dims[dim] = max(0.0, min(1.0, new_dims[dim]))

        inertia = 0.7
        for dim, new_val in new_dims.items():
            old_val = self.semi_stable_traits.get(dim, 0.5)
            self.semi_stable_traits[dim] = round(old_val * inertia + new_val * (1 - inertia), 4)

    def dims_to_archetype_seeds(self) -> dict[str, float]:
        """Derive initial archetype scores from current semi_stable_traits.

        Used to seed TraitEvolution on first boot so personality isn't
        blank. Returns scores in [0, 1] range.
        """
        seeds: dict[str, float] = {}
        for arch, influences in self._ARCHETYPE_TO_DIMS.items():
            score = 0.0
            weight_sum = 0.0
            for dim, w in influences.items():
                dim_val = self.semi_stable_traits.get(dim, 0.5)
                if w < 0:
                    dim_val = 1.0 - dim_val
                score += abs(w) * dim_val
                weight_sum += abs(w)
            seeds[arch] = round(min(1.0, score / max(weight_sum, 0.01)), 3)
        return seeds

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "core_values": list(self.core_values),
            "semi_stable_traits": dict(self.semi_stable_traits),
            "dynamic_mood": self.dynamic_mood,
            "relationships": {k: asdict(v) for k, v in self.relationships.items()},
            "age_days": (time.time() - self.created_at) / 86400,
        }
        ea = self.existential_advisory
        if ea.last_synced > 0.0:
            result["existential_advisory"] = {
                "identity_stability": ea.identity_stability,
                "identity_confidence": ea.identity_confidence,
                "active_paradoxes": ea.active_paradoxes,
                "core_markers": ea.core_markers,
                "continuity_threads": ea.continuity_threads,
            }
        return result


@dataclass
class SoulSnapshot:
    id: str
    version: str
    created_at: str
    metadata: dict[str, Any]
    kernel: dict[str, Any]
    memories: list[dict[str, Any]]
    stats: dict[str, Any]
    consciousness: dict[str, Any] | None = None
    identity: dict[str, Any] | None = None


class SoulService:
    def __init__(self) -> None:
        self._identity = IdentityState.load()

    @property
    def identity(self) -> IdentityState:
        return self._identity

    def export_soul(
        self,
        current_state: dict[str, Any],
        traits: list[str],
        reason: str | None = None,
        notes: str | None = None,
        consciousness_state: dict[str, Any] | None = None,
    ) -> SoulSnapshot:
        stats = memory_storage.get_stats()
        tag_freq = memory_storage.get_tag_frequency()
        memories = memory_storage.get_all()

        dominant_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)[:5]

        significant_memories = [
            asdict(m) for m in memories if m.weight > 0.1 or m.is_core
        ]

        snapshot = SoulSnapshot(
            id=nanoid(size=21),
            version=SOUL_VERSION,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            metadata={"export_reason": reason, "notes": notes},
            kernel=dict(current_state),
            memories=significant_memories,
            stats={
                "total_memories": stats["total"],
                "core_memory_count": stats["core_count"],
                "avg_memory_weight": stats["avg_weight"],
                "dominant_tags": [t[0] for t in dominant_tags],
                "personality_traits": list(traits),
            },
            consciousness=consciousness_state,
            identity=self._identity.to_dict(),
        )

        logger.info("Soul exported: %d memories", len(snapshot.memories))
        return snapshot

    def import_soul(self, snapshot: SoulSnapshot) -> bool:
        try:
            if not self._validate(snapshot):
                raise ValueError("Invalid soul snapshot")
            major = snapshot.version.split(".")[0]
            if major not in ("1", "2", "3"):
                raise ValueError(f"Incompatible version: {snapshot.version}")

            memory_storage.clear()
            memory_storage.load_from_json(snapshot.memories)

            if snapshot.identity and isinstance(snapshot.identity, dict):
                traits = snapshot.identity.get("semi_stable_traits", {})
                self._identity.semi_stable_traits.update(traits)
                self._identity.save()

            logger.info("Soul imported: %d memories restored", len(snapshot.memories))
            return True
        except Exception:
            logger.exception("Soul import failed")
            return False

    def get_consciousness_data(self, snapshot: SoulSnapshot) -> dict[str, Any] | None:
        return snapshot.consciousness

    def record_interaction(self, speaker: str) -> None:
        """Call after each voice interaction to update relationship tracking."""
        if not speaker or speaker == "unknown":
            return
        from identity.name_validator import is_valid_person_name
        if not is_valid_person_name(speaker):
            return
        rel = self._identity.get_relationship(speaker)
        rel.record_interaction()

    def save_identity(self) -> None:
        self._identity.save()

    def reset(self) -> None:
        memory_storage.clear()
        self._identity = IdentityState()
        self._identity.save()
        logger.info("Soul reset complete")

    @staticmethod
    def _validate(snapshot: Any) -> bool:
        return (
            isinstance(snapshot, SoulSnapshot)
            and isinstance(snapshot.id, str)
            and isinstance(snapshot.version, str)
            and isinstance(snapshot.memories, list)
            and snapshot.kernel is not None
            and snapshot.stats is not None
        )


soul_service = SoulService()
