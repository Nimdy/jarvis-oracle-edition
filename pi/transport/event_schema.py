"""Shared event schema — Pi <-> Laptop bidirectional messages."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

EventSource = Literal["vision", "audio", "screen", "system"]

CRITICAL_EVENT_TYPES: frozenset[str] = frozenset({
    "person_detected",
    "person_lost",
    "face_expression",
    "playback_complete",
})


def new_conversation_id() -> str:
    return str(uuid.uuid4())


class PerceptionEvent(BaseModel):
    """Event sent from Pi senses to laptop brain."""
    source: EventSource
    type: str
    timestamp: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    conversation_id: str = ""

    def to_json(self) -> str:
        return self.model_dump_json()


class BrainMessage(BaseModel):
    """Message received from laptop brain."""
    type: Literal["response", "response_chunk", "response_end", "state_update", "command", "consciousness_feed"]
    text: str = ""
    tone: str = ""
    phase: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str = ""


# --- Vision event factories ---

def person_detected(confidence: float, bbox: tuple[int, int, int, int] | None = None) -> PerceptionEvent:
    data: dict[str, Any] = {}
    if bbox:
        data["bbox"] = list(bbox)
    return PerceptionEvent(source="vision", type="person_detected", confidence=confidence, data=data)


def person_lost(confidence: float) -> PerceptionEvent:
    return PerceptionEvent(source="vision", type="person_lost", confidence=confidence)


def gesture_detected(gesture: str, confidence: float) -> PerceptionEvent:
    return PerceptionEvent(source="vision", type="gesture", confidence=confidence, data={"gesture": gesture})


def face_expression(expression: str, confidence: float) -> PerceptionEvent:
    return PerceptionEvent(source="vision", type="face_expression", confidence=confidence, data={"expression": expression})


def pose_detected(keypoints: list, gesture: str, confidence: float) -> PerceptionEvent:
    """Body pose detected with 17-point skeleton and gesture classification."""
    return PerceptionEvent(
        source="vision", type="pose_detected", confidence=confidence,
        data={"keypoints": keypoints, "gesture": gesture},
    )


def face_crop_event(crop_b64: str, track_id: int, confidence: float) -> PerceptionEvent:
    """Face crop extracted for identity matching on brain."""
    return PerceptionEvent(
        source="vision", type="face_crop", confidence=confidence,
        data={"crop_b64": crop_b64, "track_id": track_id},
    )


def scene_summary(detections: list[dict], frame_size: tuple[int, int],
                  scene_change_score: float) -> PerceptionEvent:
    """Compact scene object summary from Pi aggregator (non-person COCO detections)."""
    return PerceptionEvent(
        source="vision", type="scene_summary",
        data={"detections": detections, "frame_size": list(frame_size),
              "scene_change_score": scene_change_score})


# --- System event factories ---

def sensor_status(
    mic_device: str = "",
    mic_rms: float = 0.0,
    mic_muted: bool = False,
    vision_fps: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> PerceptionEvent:
    """Periodic sensor health report sent to brain for dashboard/diagnostics."""
    data: dict[str, Any] = {
        "mic_device": mic_device,
        "mic_rms": round(mic_rms, 6),
        "mic_muted": mic_muted,
        "vision_fps": round(vision_fps, 1),
    }
    if extra:
        data.update(extra)
    return PerceptionEvent(source="system", type="sensor_status", data=data)


def sensor_health(
    cpu_temp_c: float = 0.0,
    cpu_percent: float = 0.0,
    mem_used_mb: float = 0.0,
    mem_total_mb: float = 0.0,
    camera_fps: float = 0.0,
    throttled: str = "",
    uptime_s: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> PerceptionEvent:
    """Periodic hardware health telemetry sent to brain."""
    data: dict[str, Any] = {
        "cpu_temp_c": round(cpu_temp_c, 1),
        "cpu_percent": round(cpu_percent, 1),
        "mem_used_mb": round(mem_used_mb, 0),
        "mem_total_mb": round(mem_total_mb, 0),
        "camera_fps": round(camera_fps, 1),
        "throttled": throttled,
        "uptime_s": round(uptime_s, 0),
    }
    if extra:
        data.update(extra)
    return PerceptionEvent(source="system", type="sensor_health", data=data)
