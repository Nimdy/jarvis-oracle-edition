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
)
from autonomy.curiosity_detector import CuriosityDetector
from autonomy.research_intent import ResearchIntent

logger = logging.getLogger(__name__)


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

    def wire(self) -> None:
        """Subscribe to consciousness events. Safe to call multiple times."""
        if self._wired:
            return

        self._cleanups.append(event_bus.on(KERNEL_THOUGHT, self._on_thought))
        self._cleanups.append(event_bus.on(META_THOUGHT_GENERATED, self._on_meta_thought))
        self._cleanups.append(event_bus.on(EXISTENTIAL_INQUIRY_COMPLETED, self._on_existential))
        self._cleanups.append(event_bus.on(CONSCIOUSNESS_EMERGENT_BEHAVIOR, self._on_emergence))

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

        intent = self._detector.evaluate_thought(
            thought_type=thought_type,
            text=text,
            depth=depth,
            tags=tags,
            confidence=confidence,
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
            logger.debug("Intent enqueued: %s (priority=%.2f)", intent.question[:50], intent.priority)
