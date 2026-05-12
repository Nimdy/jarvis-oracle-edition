"""Vision processor — summarises vision perception events from the Pi."""

from __future__ import annotations

import time
from typing import Callable

from consciousness.events import PerceptionEvent, event_bus, PERCEPTION_EVENT
from consciousness.engine import ConsciousnessEngine
from memory.core import CreateMemoryData


class VisionProcessor:
    def __init__(self, engine: ConsciousnessEngine) -> None:
        self._engine = engine
        self._persons_present = 0
        self._last_expression: str | None = None
        self._last_gesture: str | None = None
        self._last_update = 0.0
        self._cleanups: list[Callable[[], None]] = []

    def start(self) -> None:
        self._cleanups.append(
            event_bus.on(PERCEPTION_EVENT, self._on_event)
        )

    def stop(self) -> None:
        for cleanup in self._cleanups:
            cleanup()
        self._cleanups.clear()

    def _on_event(self, event: PerceptionEvent, **_) -> None:
        if event.source != "vision":
            return
        self._last_update = time.time()

        match event.type:
            case "person_detected":
                self._persons_present += 1
            case "person_lost":
                self._persons_present = max(0, self._persons_present - 1)
            case "gesture":
                self._last_gesture = event.data.get("gesture")
                if event.confidence > 0.5 and self._last_gesture:
                    self._engine.remember(CreateMemoryData(
                        type="observation",
                        payload=f"User made a {self._last_gesture} gesture",
                        weight=0.3,
                        tags=["gesture", "vision", self._last_gesture],
                        provenance="observed",
                    ))
            case "face_expression":
                self._last_expression = event.data.get("expression")
                if event.confidence > 0.5 and self._last_expression:
                    self._engine.remember(CreateMemoryData(
                        type="observation",
                        payload=f"User appears {self._last_expression}",
                        weight=0.35,
                        tags=["expression", "emotion", "vision", self._last_expression],
                        provenance="observed",
                    ))

    def get_summary(self) -> dict:
        return {
            "persons_present": self._persons_present,
            "last_expression": self._last_expression,
            "last_gesture": self._last_gesture,
            "last_update": self._last_update,
        }

    def get_context_string(self) -> str:
        parts: list[str] = []
        if self._persons_present > 0:
            parts.append(f"{self._persons_present} person(s) present")
        else:
            parts.append("No one visible")
        if self._last_expression:
            parts.append(f"Expression: {self._last_expression}")
        if self._last_gesture:
            parts.append(f"Recent gesture: {self._last_gesture}")
        return ". ".join(parts)
