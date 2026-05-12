"""Dream Artifact Pipeline — provisional dream outputs with validation lifecycle.

Dream artifacts are the ontological layer between dream cognition and canonical memory.
They are NEVER written to MemoryStorage directly, NEVER emitted as MEMORY_WRITE, and
NEVER fed to belief extraction. Only the ReflectiveValidator can promote them to
memories, and only through engine.remember().

Architectural rule:
    Dreaming may generate structure.
    Waking validation determines ontology.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)

ArtifactType = Literal[
    "bridge_candidate",
    "symbolic_summary",
    "tension_flag",
    "consolidation_proposal",
    "waking_question",
    "shadow_scenario",
]

ValidationState = Literal[
    "pending", "promoted", "held", "discarded", "quarantined",
]

MAX_ARTIFACT_BUFFER = 200
PROMOTION_MIN_COHERENCE = 0.45
PROMOTION_MIN_CONFIDENCE = 0.35
MAX_PROMOTIONS_PER_VALIDATION = 10


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DreamArtifact:
    artifact_id: str
    artifact_type: ArtifactType
    source_memory_ids: tuple[str, ...]
    content: str
    confidence: float
    cluster_coherence: float
    stance: str = "dreaming"
    provenance: str = "dream_observer"
    validation_state: ValidationState = "pending"
    timestamp: float = field(default_factory=time.time)
    promoted_at: float | None = None
    discarded_at: float | None = None
    validator_notes: str = ""

    def with_validation(
        self,
        new_state: ValidationState,
        notes: str = "",
    ) -> DreamArtifact:
        """Return a copy with updated validation state."""
        now = time.time()
        return DreamArtifact(
            artifact_id=self.artifact_id,
            artifact_type=self.artifact_type,
            source_memory_ids=self.source_memory_ids,
            content=self.content,
            confidence=self.confidence,
            cluster_coherence=self.cluster_coherence,
            stance=self.stance,
            provenance=self.provenance,
            validation_state=new_state,
            timestamp=self.timestamp,
            promoted_at=now if new_state == "promoted" else self.promoted_at,
            discarded_at=now if new_state in ("discarded", "quarantined") else self.discarded_at,
            validator_notes=notes or self.validator_notes,
        )


def create_artifact(
    artifact_type: ArtifactType,
    source_memory_ids: list[str] | tuple[str, ...],
    content: str,
    confidence: float,
    cluster_coherence: float,
) -> DreamArtifact:
    return DreamArtifact(
        artifact_id=f"dart_{uuid.uuid4().hex[:12]}",
        artifact_type=artifact_type,
        source_memory_ids=tuple(source_memory_ids),
        content=content[:500],
        confidence=max(0.0, min(1.0, confidence)),
        cluster_coherence=max(0.0, min(1.0, cluster_coherence)),
    )


# ---------------------------------------------------------------------------
# Bounded artifact buffer
# ---------------------------------------------------------------------------

_DREAM_STATS_PATH = Path(os.environ.get("JARVIS_HOME", os.path.expanduser("~/.jarvis"))) / "dream_stats.json"


def _load_cumulative_stats() -> dict[str, int]:
    """Load persisted cumulative dream artifact counts."""
    try:
        if _DREAM_STATS_PATH.exists():
            data = json.loads(_DREAM_STATS_PATH.read_text())
            if isinstance(data, dict):
                return {k: int(v) for k, v in data.items() if isinstance(v, (int, float))}
    except Exception as exc:
        logger.warning("Failed to load dream stats from %s: %s", _DREAM_STATS_PATH, exc)
    return {}


def _save_cumulative_stats(stats: dict[str, int]) -> None:
    """Atomically persist cumulative dream artifact counts."""
    try:
        _DREAM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=_DREAM_STATS_PATH.parent, suffix=".tmp")
        try:
            os.write(fd, json.dumps(stats).encode())
            os.close(fd)
            fd = -1
            os.replace(tmp, _DREAM_STATS_PATH)
        except Exception:
            if fd >= 0:
                os.close(fd)
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    except Exception as exc:
        logger.warning("Failed to save dream stats: %s", exc)


class ArtifactBuffer:
    """In-memory ring buffer for dream artifacts.

    The artifact buffer itself is ephemeral (ring buffer resets on restart), but
    cumulative counters (total_created, total_promoted, etc.) are persisted to
    ``~/.jarvis/dream_stats.json`` so maturity gates reflect lifetime totals.
    """

    def __init__(self, maxlen: int = MAX_ARTIFACT_BUFFER) -> None:
        self._buffer: deque[DreamArtifact] = deque(maxlen=maxlen)
        saved = _load_cumulative_stats()
        self._stats = {
            "total_created": saved.get("total_created", 0),
            "total_promoted": saved.get("total_promoted", 0),
            "total_discarded": saved.get("total_discarded", 0),
            "total_quarantined": saved.get("total_quarantined", 0),
            "total_held": saved.get("total_held", 0),
            "last_validation": saved.get("last_validation", 0.0),
            "validation_count": saved.get("validation_count", 0),
        }
        if any(v > 0 for v in self._stats.values()):
            logger.info(
                "Restored cumulative dream stats: created=%d promoted=%d validations=%d",
                self._stats["total_created"], self._stats["total_promoted"],
                self._stats["validation_count"],
            )

    def add(self, artifact: DreamArtifact) -> None:
        self._buffer.append(artifact)
        self._stats["total_created"] += 1
        self._persist()

    def get_pending(self) -> list[DreamArtifact]:
        return [a for a in self._buffer if a.validation_state == "pending"]

    def get_recent(self, limit: int = 20) -> list[DreamArtifact]:
        return list(self._buffer)[-limit:]

    def update(self, artifact_id: str, new_state: ValidationState, notes: str = "") -> bool:
        for i, a in enumerate(self._buffer):
            if a.artifact_id == artifact_id:
                prev_state = a.validation_state
                self._buffer[i] = a.with_validation(new_state, notes)
                stat_key = f"total_{new_state}"
                if prev_state != new_state and stat_key in self._stats:
                    self._stats[stat_key] += 1
                    self._persist()
                return True
        return False

    def _persist(self) -> None:
        _save_cumulative_stats(self._stats)

    def get_stats(self) -> dict[str, Any]:
        type_counts: dict[str, int] = {}
        state_counts: dict[str, int] = {}
        total_confidence = 0.0
        total_coherence = 0.0
        count = len(self._buffer)
        for a in self._buffer:
            type_counts[a.artifact_type] = type_counts.get(a.artifact_type, 0) + 1
            state_counts[a.validation_state] = state_counts.get(a.validation_state, 0) + 1
            total_confidence += a.confidence
            total_coherence += a.cluster_coherence

        return {
            "buffer_size": count,
            "buffer_capacity": self._buffer.maxlen,
            "by_type": type_counts,
            "by_state": state_counts,
            "avg_confidence": total_confidence / count if count else 0.0,
            "avg_coherence": total_coherence / count if count else 0.0,
            **self._stats,
        }


# ---------------------------------------------------------------------------
# Reflective Validator
# ---------------------------------------------------------------------------

class ReflectiveValidator:
    """Validates pending dream artifacts for promotion to canonical memory.

    Runs on exit from dreaming mode, on reflective mode entry, or periodically
    during reflective mode. Only promotes artifacts that meet quality thresholds
    and have grounded support from existing memories.
    """

    def __init__(
        self,
        buffer: ArtifactBuffer,
        remember_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._buffer = buffer
        self._remember_fn = remember_fn
        self._last_validation: float = buffer._stats.get("last_validation", 0.0)
        self._validation_count: int = buffer._stats.get("validation_count", 0)

    def set_remember_fn(self, fn: Callable[..., Any]) -> None:
        """Set the promotion callback (engine.remember)."""
        self._remember_fn = fn

    def validate_pending(
        self,
        system_context: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        """Review all pending artifacts and assign outcomes.

        Args:
            system_context: Optional dict of system state at validation time,
                used by the dream-artifact encoder for Tier-1 distillation
                signal recording.  Keys: memory_density, dream_cycle_count,
                awareness, belief_count, contradiction_debt, soul_integrity,
                quarantine_pressure, promotion_rate_session.
        """
        pending = self._buffer.get_pending()
        if not pending:
            return {"reviewed": 0, "promoted": 0, "held": 0, "discarded": 0, "quarantined": 0}

        results = {"reviewed": 0, "promoted": 0, "held": 0, "discarded": 0, "quarantined": 0}
        ctx = system_context or {}

        for artifact in pending:
            outcome = self._evaluate(artifact)

            if outcome.state == "promoted" and results["promoted"] >= MAX_PROMOTIONS_PER_VALIDATION:
                outcome = _ValidationOutcome("held", "promotion cap reached, deferring")

            self._buffer.update(artifact.artifact_id, outcome.state, outcome.notes)
            results["reviewed"] += 1
            results[outcome.state] += 1

            self._record_distillation_signal(artifact, outcome, ctx)

            if outcome.state == "promoted" and self._remember_fn:
                self._promote(artifact)

        self._last_validation = time.time()
        self._validation_count += 1
        self._buffer._stats["last_validation"] = self._last_validation
        self._buffer._stats["validation_count"] = self._validation_count
        self._buffer._persist()

        if results["reviewed"]:
            logger.info(
                "Reflective validation: %d reviewed, %d promoted, %d held, %d discarded, %d quarantined",
                results["reviewed"], results["promoted"], results["held"],
                results["discarded"], results["quarantined"],
            )

        return results

    def _record_distillation_signal(
        self,
        artifact: DreamArtifact,
        outcome: _ValidationOutcome,
        system_context: dict[str, Any],
    ) -> None:
        """Record feature + label teacher signals for the DREAM_SYNTHESIS specialist."""
        try:
            from hemisphere.dream_artifact_encoder import DreamArtifactEncoder
            from hemisphere.distillation import DistillationCollector

            artifact_dict = {
                "artifact_id": artifact.artifact_id,
                "artifact_type": artifact.artifact_type,
                "confidence": artifact.confidence,
                "cluster_coherence": artifact.cluster_coherence,
                "source_memory_ids": artifact.source_memory_ids,
                "content": artifact.content,
            }

            features = DreamArtifactEncoder.encode(artifact_dict, system_context)
            label, label_meta = DreamArtifactEncoder.encode_label(
                outcome.state, artifact_dict, outcome.notes,
            )

            collector = DistillationCollector.instance()
            collector.record(
                teacher="dream_features",
                signal_type="artifact_snapshot",
                data=features,
                metadata={"artifact_id": artifact.artifact_id},
                origin="dream_observer",
                fidelity=1.0,
            )
            collector.record(
                teacher="dream_validator",
                signal_type="validation_outcome",
                data=label,
                metadata=label_meta,
                origin="dream_observer",
                fidelity=1.0,
            )
        except Exception:
            logger.debug("Failed to record dream distillation signal", exc_info=True)

    _DREAM_SELF_REF_TAGS = frozenset({"dream_artifact", "dream_consolidation_proposal"})

    def _evaluate(self, artifact: DreamArtifact) -> _ValidationOutcome:
        """Heuristic evaluation of a single artifact."""
        if not artifact.source_memory_ids:
            return _ValidationOutcome("discarded", "no source memories")

        # Layer 1: content string check for known self-referential pattern
        if artifact.artifact_type == "consolidation_proposal":
            content_lower = artifact.content.lower()
            if "consolidation: dream_artifact" in content_lower or \
               "dream_consolidation_proposal" in content_lower:
                return _ValidationOutcome("discarded", "self-referential: consolidating dream artifacts")

        # Layer 2: source-memory tag dominance check (structural guard)
        if self._source_memories_dominated_by_dream(artifact):
            return _ValidationOutcome("discarded", "source memories are predominantly dream artifacts")

        if self._has_active_contradiction(artifact):
            return _ValidationOutcome("quarantined", "contradicts active beliefs")

        if artifact.artifact_type in ("tension_flag", "waking_question"):
            return _ValidationOutcome("held", "informational artifact, hold for context")

        if artifact.cluster_coherence < PROMOTION_MIN_COHERENCE:
            return _ValidationOutcome("discarded", f"low coherence: {artifact.cluster_coherence:.2f}")

        if artifact.confidence < PROMOTION_MIN_CONFIDENCE:
            return _ValidationOutcome("discarded", f"low confidence: {artifact.confidence:.2f}")

        if artifact.cluster_coherence >= 0.65 and artifact.confidence >= 0.5:
            return _ValidationOutcome("promoted", "meets promotion thresholds")

        return _ValidationOutcome("held", "borderline quality, holding for further review")

    def _source_memories_dominated_by_dream(self, artifact: DreamArtifact) -> bool:
        """Return True if >= 50% of resolvable source memories carry dream-self-ref tags."""
        if not artifact.source_memory_ids:
            return False
        try:
            from memory.storage import memory_storage
            dream_count = 0
            resolved = 0
            for mid in artifact.source_memory_ids:
                mem = memory_storage.get(mid)
                if mem is None:
                    continue
                resolved += 1
                if set(getattr(mem, "tags", ())) & self._DREAM_SELF_REF_TAGS:
                    dream_count += 1
            if resolved == 0:
                return False
            return dream_count >= resolved * 0.5
        except Exception:
            return False

    def _has_active_contradiction(self, artifact: DreamArtifact) -> bool:
        """Check if artifact content conflicts with active beliefs."""
        try:
            from epistemic.contradiction_engine import ContradictionEngine
            engine = ContradictionEngine.get_instance()
            active = engine._belief_store.get_active_beliefs()
            content_lower = artifact.content.lower()
            for b in active[:50]:
                if b.canonical_subject and b.canonical_subject.lower() in content_lower:
                    if b.polarity < 0:
                        return True
            return False
        except Exception:
            return False

    def _promote(self, artifact: DreamArtifact) -> None:
        """Promote artifact to canonical memory via engine.remember()."""
        if not self._remember_fn:
            return
        try:
            self._remember_fn(
                text=f"[Dream artifact: {artifact.artifact_type}] {artifact.content}",
                memory_type="observation",
                tags=("dream_artifact", f"dream_{artifact.artifact_type}"),
                weight=min(0.4, artifact.confidence * 0.5),
                provenance="dream_observer",
            )
            logger.info("Promoted dream artifact %s (%s)", artifact.artifact_id, artifact.artifact_type)
        except Exception:
            logger.debug("Failed to promote artifact %s", artifact.artifact_id, exc_info=True)

    def get_stats(self) -> dict[str, Any]:
        return {
            "last_validation": self._last_validation,
            "validation_count": self._validation_count,
        }


@dataclass
class _ValidationOutcome:
    state: ValidationState
    notes: str


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

artifact_buffer = ArtifactBuffer()
reflective_validator = ReflectiveValidator(artifact_buffer)
