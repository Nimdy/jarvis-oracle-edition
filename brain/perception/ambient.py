"""Ambient audio processor — meeting detection and conversation context."""

from __future__ import annotations

import asyncio
import time
from typing import Callable

from consciousness.events import event_bus, PERCEPTION_SCREEN_CONTEXT
from consciousness.engine import ConsciousnessEngine
from memory.core import CreateMemoryData

_MEETING_APPS = ("zoom", "teams", "meet", "discord", "slack")


class AmbientAudioProcessor:
    def __init__(self, engine: ConsciousnessEngine, analyze_interval_s: float = 10.0) -> None:
        self._engine = engine
        self._analyze_interval = analyze_interval_s
        self._in_meeting = False
        self._speech_detected = False
        self._conversation_context = ""
        self._last_analysis = 0.0
        self._cleanups: list[Callable[[], None]] = []
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._cleanups.append(
            event_bus.on(PERCEPTION_SCREEN_CONTEXT, self._on_screen_context)
        )
        self._task = asyncio.get_event_loop().create_task(self._analyze_loop())

    def stop(self) -> None:
        for cleanup in self._cleanups:
            cleanup()
        self._cleanups.clear()
        if self._task:
            self._task.cancel()
            self._task = None

    def _on_screen_context(self, app: str, **_) -> None:
        lower_app = app.lower()
        in_meeting = any(m in lower_app for m in _MEETING_APPS)
        if in_meeting != self._in_meeting:
            self._in_meeting = in_meeting
            if in_meeting:
                self._engine.remember(CreateMemoryData(
                    type="observation",
                    payload=f"User appears to be in a meeting ({app})",
                    weight=0.4,
                    tags=["meeting", "context", "audio"],
                    provenance="observed",
                ))
            else:
                self._engine.remember(CreateMemoryData(
                    type="observation",
                    payload="Meeting appears to have ended",
                    weight=0.3,
                    tags=["meeting", "ended", "context"],
                    provenance="observed",
                ))

    def _on_ambient_sound(self, classification: str, confidence: float, **_) -> None:
        if classification == "speech" and confidence > 0.5:
            self._speech_detected = True
        elif classification == "silence":
            self._speech_detected = False

    async def _analyze_loop(self) -> None:
        try:
            while True:
                self._analyze_context()
                await asyncio.sleep(self._analyze_interval)
        except asyncio.CancelledError:
            pass

    def _analyze_context(self) -> None:
        parts: list[str] = []
        if self._in_meeting:
            parts.append("In a meeting")
            parts.append("conversation ongoing" if self._speech_detected else "meeting is quiet")
        elif self._speech_detected:
            parts.append("Conversation detected nearby")

        context_str = ", ".join(parts) if parts else ""
        if context_str and context_str != self._conversation_context:
            self._conversation_context = context_str
            self._last_analysis = time.time()
            self._engine.remember(CreateMemoryData(
                type="observation",
                payload=f"Audio context: {context_str}",
                weight=0.2,
                tags=["audio_context", "ambient"],
                provenance="observed",
            ))

    def get_state(self) -> dict:
        return {
            "in_meeting": self._in_meeting,
            "speech_detected": self._speech_detected,
            "conversation_context": self._conversation_context,
            "last_analysis": self._last_analysis,
        }

    def get_context_string(self) -> str:
        if self._in_meeting:
            return "User is in a meeting"
        if self._speech_detected:
            return "Nearby conversation detected"
        return "Quiet environment"
