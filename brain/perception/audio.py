"""Audio processor — summarises audio perception events from the Pi."""

from __future__ import annotations

import time
from typing import Callable

from consciousness.events import PerceptionEvent, event_bus, PERCEPTION_EVENT
from consciousness.engine import ConsciousnessEngine
from memory.core import CreateMemoryData


class AudioProcessor:
    def __init__(self, engine: ConsciousnessEngine) -> None:
        self._engine = engine
        self._ambient_sound = "silence"
        self._voice_active = False
        self._last_wake_word = 0.0
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
        if event.source != "audio":
            return
        self._last_update = time.time()

        match event.type:
            case "voice_activity":
                self._voice_active = event.data.get("active", False)
            case "ambient_sound":
                classification = event.data.get("classification", "silence")
                self._ambient_sound = classification
                if classification != "silence" and event.confidence > 0.5:
                    self._engine.remember(CreateMemoryData(
                        type="observation",
                        payload=f"Ambient sound: {classification}",
                        weight=0.15,
                        tags=["audio", "ambient", classification],
                        provenance="observed",
                    ))
            case "wake_word":
                self._last_wake_word = event.timestamp

    def get_summary(self) -> dict:
        return {
            "ambient_sound": self._ambient_sound,
            "voice_active": self._voice_active,
            "last_wake_word": self._last_wake_word,
            "last_update": self._last_update,
        }

    def get_context_string(self) -> str:
        parts: list[str] = []
        if self._ambient_sound != "silence":
            parts.append(f"Ambient: {self._ambient_sound}")
        if self._voice_active:
            parts.append("Voice activity detected")
        return ". ".join(parts) or "Quiet environment"
