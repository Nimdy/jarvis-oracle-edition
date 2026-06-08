"""Event Bridge — wires consciousness events to the autonomy pipeline.

Listens to KERNEL_THOUGHT, EXISTENTIAL_INQUIRY_COMPLETED,
CONSCIOUSNESS_EMERGENT_BEHAVIOR, META_THOUGHT_GENERATED, and cognitive
gap events. For each, asks the CuriosityDetector if there's an actionable
research question, then enqueues it in the AutonomyOrchestrator.

This is the loop closer: it turns "dashboard shows thoughts" into
"thoughts cause bounded action."
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from consciousness.events import (
    event_bus,
    KERNEL_THOUGHT,
    META_THOUGHT_GENERATED,
    EXISTENTIAL_INQUIRY_COMPLETED,
    CONSCIOUSNESS_EMERGENT_BEHAVIOR,
    AUTONOMY_RESEARCH_COMPLETED,
    AUTONOMY_RESEARCH_FAILED,
)
from autonomy.curiosity_detector import CuriosityDetector
from autonomy.research_intent import (
    ResearchIntent,
    ResearchResult,
    emit_thought_validation_outcome,
)

logger = logging.getLogger(__name__)

# Cap on remembered tension-seeded intents (id → intent) so a long uptime can't
# leak memory. Tension-seeded intents are rare (gated, shadow-first), so this is
# generous. Oldest entries are evicted FIFO.
_MAX_TRACKED_TENSION_INTENTS = 256


class AutonomyEventBridge:
    """Bridges consciousness events → curiosity detection → research queue."""

    def __init__(
        self,
        detector: CuriosityDetector,
        enqueue_cb: Callable[[ResearchIntent], bool],
    ) -> None:
        self._detector = detector
        self._enqueue = enqueue_cb
        self._cleanups: list[Callable[[], None]] = []
        self._events_processed: int = 0
        self._intents_generated: int = 0
        self._wired: bool = False
        # SPARK §3 component 2 — id → tension-seeded ResearchIntent. Used to emit
        # the external-only THOUGHT_VALIDATION_OUTCOME teacher signal when the
        # intent completes (so a degraded completion-event fallback can still
        # cite the belief_id even if the caller didn't hand us the result).
        self._tension_intents: dict[str, ResearchIntent] = {}
        self._validation_outcomes_emitted: int = 0

    def wire(self) -> None:
        """Subscribe to consciousness events. Safe to call multiple times."""
        if self._wired:
            return

        self._cleanups.append(event_bus.on(KERNEL_THOUGHT, self._on_thought))
        self._cleanups.append(event_bus.on(META_THOUGHT_GENERATED, self._on_meta_thought))
        self._cleanups.append(event_bus.on(EXISTENTIAL_INQUIRY_COMPLETED, self._on_existential))
        self._cleanups.append(event_bus.on(CONSCIOUSNESS_EMERGENT_BEHAVIOR, self._on_emergence))
        # SPARK §3 component 2 — emit THOUGHT_VALIDATION_OUTCOME on the completion
        # of a tension-seeded intent (degraded fallback; the orchestrator may also
        # call note_research_outcome() directly with the full result object).
        self._cleanups.append(event_bus.on(AUTONOMY_RESEARCH_COMPLETED, self._on_research_done))
        self._cleanups.append(event_bus.on(AUTONOMY_RESEARCH_FAILED, self._on_research_done))

        try:
            from consciousness.events import CONSCIOUSNESS_LEARNING_PROTOCOL
            self._cleanups.append(event_bus.on(CONSCIOUSNESS_LEARNING_PROTOCOL, self._on_learning_protocol))
        except ImportError:
            pass

        self._wired = True
        logger.info("Autonomy event bridge wired (%d listeners)", len(self._cleanups))

    def unwire(self) -> None:
        for cleanup in self._cleanups:
            try:
                cleanup()
            except Exception:
                pass
        self._cleanups.clear()
        self._wired = False

    def get_stats(self) -> dict[str, Any]:
        return {
            "wired": self._wired,
            "events_processed": self._events_processed,
            "intents_generated": self._intents_generated,
            "tension_intents_tracked": len(self._tension_intents),
            "validation_outcomes_emitted": self._validation_outcomes_emitted,
        }

    # -- event handlers -------------------------------------------------------

    def _on_thought(self, **kwargs: Any) -> None:
        """Handle KERNEL_THOUGHT events."""
        self._events_processed += 1
        thought_type = kwargs.get("thought_type", "")
        text = kwargs.get("text", "")
        depth = kwargs.get("depth", "surface")

        if thought_type == "research_finding":
            return

        tags = kwargs.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        confidence = kwargs.get("confidence", 0.5)
        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = 0.5

        # SPARK §3 component 2 — carry grounding provenance from a
        # belief_validation_curiosity tension-thought (default "" otherwise).
        belief_id = str(kwargs.get("belief_id", "") or "")
        validation_target = str(kwargs.get("validation_target", "") or "")

        intent = self._detector.evaluate_thought(
            thought_type=thought_type,
            text=text,
            depth=depth,
            tags=tags,
            confidence=confidence,
            belief_id=belief_id,
            validation_target=validation_target,
        )
        if intent:
            self._submit(intent)

    def _on_meta_thought(self, **kwargs: Any) -> None:
        """Handle META_THOUGHT_GENERATED — same data shape as KERNEL_THOUGHT."""
        self._on_thought(**kwargs)

    def _on_existential(self, **kwargs: Any) -> None:
        """Handle EXISTENTIAL_INQUIRY_COMPLETED events."""
        self._events_processed += 1
        category = kwargs.get("category", "")
        question = kwargs.get("question", "")
        depth = kwargs.get("depth", "surface")

        intent = self._detector.evaluate_existential_inquiry(
            category=category,
            question=question,
            depth=depth,
        )
        if intent:
            self._submit(intent)

    def _on_emergence(self, **kwargs: Any) -> None:
        """Handle CONSCIOUSNESS_EMERGENT_BEHAVIOR events."""
        self._events_processed += 1
        behavior_name = kwargs.get("behavior", kwargs.get("name", "unknown"))
        description = kwargs.get("description", str(kwargs))

        intent = self._detector.evaluate_emergence(
            behavior_name=behavior_name,
            description=description,
        )
        if intent:
            self._submit(intent)

    def _on_learning_protocol(self, **kwargs: Any) -> None:
        """Handle CONSCIOUSNESS_LEARNING_PROTOCOL — capability unlocked."""
        self._events_processed += 1
        protocol = kwargs.get("protocol", "")
        tier = kwargs.get("tier", 0)

        if tier >= 3:
            intent = ResearchIntent(
                question=f"What advanced learning techniques apply to {protocol} at tier {tier}?",
                source_event=f"learning_protocol:{protocol}",
                source_hint="web",
                priority=0.5,
                scope="external_ok",
                tag_cluster=("learning", protocol),
                reason=f"Learning protocol {protocol} activated at tier {tier}",
            )
            self._submit(intent)

    def _submit(self, intent: ResearchIntent) -> None:
        if intent.source_event.startswith("metric:"):
            logger.warning(
                "Bridge dropped metric-sourced intent — metric_triggers "
                "is the sole source for metric intents: %s",
                intent.question[:60],
            )
            return

        success = self._enqueue(intent)
        if success:
            self._intents_generated += 1
            # Remember tension-seeded intents so we can emit the teacher signal
            # when they complete (SPARK §3 component 2).
            if getattr(intent, "belief_id", ""):
                self._track_tension_intent(intent)
            logger.debug("Intent enqueued: %s (priority=%.2f)", intent.question[:50], intent.priority)

    def _track_tension_intent(self, intent: ResearchIntent) -> None:
        self._tension_intents[intent.id] = intent
        # FIFO-evict oldest if we exceed the cap (rare; tension intents are gated).
        while len(self._tension_intents) > _MAX_TRACKED_TENSION_INTENTS:
            oldest = next(iter(self._tension_intents))
            self._tension_intents.pop(oldest, None)

    # -- teacher signal (SPARK §3 component 2 / §8 P3) ------------------------

    def note_research_outcome(
        self,
        intent: ResearchIntent,
        result: ResearchResult | None,
        *,
        refuted: bool | None = None,
    ) -> dict[str, Any] | None:
        """Emit THOUGHT_VALIDATION_OUTCOME for a completed tension-seeded intent.

        Preferred entry point: the caller (orchestrator) hands us the full intent
        + result so the external-grounding decision sees the per-finding
        provenance. No-op (returns None) for non-tension intents. Never raises.
        """
        try:
            payload = emit_thought_validation_outcome(intent, result, refuted=refuted)
        except Exception:
            logger.debug("note_research_outcome failed", exc_info=True)
            return None
        if payload is not None:
            self._validation_outcomes_emitted += 1
            self._tension_intents.pop(getattr(intent, "id", ""), None)
            logger.info(
                "THOUGHT_VALIDATION_OUTCOME emitted (belief_id=%s grounded=%s refuted=%s)",
                payload.get("belief_id", "?"),
                payload.get("grounded"),
                payload.get("refuted"),
            )
        return payload

    def _on_research_done(self, **kwargs: Any) -> None:
        """Degraded fallback: emit the teacher signal from a completion EVENT.

        Used only for tension intents we remembered whose outcome wasn't already
        reported via note_research_outcome(). The completion event does not carry
        per-finding provenance, so result is None here — grounded resolves
        external-only and conservatively (False) unless note_research_outcome was
        called with the real result. Idempotent: pops the tracked intent.
        """
        intent_id = str(kwargs.get("intent_id", "") or "")
        if not intent_id:
            return
        intent = self._tension_intents.pop(intent_id, None)
        if intent is None:
            return  # not tension-seeded (or already reported) — nothing to do.
        # No result object available from the event; pass None (conservative,
        # external-only grounding decision in the helper handles it safely).
        self.note_research_outcome(intent, None)
