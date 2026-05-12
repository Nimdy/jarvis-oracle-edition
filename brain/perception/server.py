"""Perception bus — WebSocket server that Pi senses connect to."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import socket
import time
from collections import deque
from typing import Any, Callable

import websockets
from websockets.asyncio.server import Server, ServerConnection, serve

from consciousness.events import (
    PerceptionEvent, event_bus,
    PERCEPTION_EVENT, PERCEPTION_USER_PRESENT,
    PERCEPTION_SPEAKER_IDENTIFIED, PERCEPTION_USER_EMOTION,
    PERCEPTION_POSE_DETECTED,
    PERCEPTION_RAW_AUDIO,
    PERCEPTION_PLAYBACK_COMPLETE,
    PERCEPTION_SENSOR_DISCONNECTED,
    SYNTHETIC_EXERCISE_STATE,
)

logger = logging.getLogger(__name__)


import struct
import threading


class _ObsAudioForwarder:
    """TCP forwarder that sends length-prefixed WAV chunks to the OBS receiver.

    Uses TCP instead of UDP to guarantee in-order delivery of large WAV files.
    Each WAV is sent as: [4-byte big-endian length][WAV bytes].
    Auto-reconnects on failure. Pure outbound — no event bus, no memory, no locks.
    """

    def __init__(self) -> None:
        self._target = os.environ.get("OBS_AUDIO_TARGET", "").strip()
        self._addr: tuple[str, int] | None = None
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        if not self._target:
            return
        try:
            host, port_str = self._target.rsplit(":", 1)
            self._addr = (host, int(port_str))
            logger.info("OBS audio forwarder enabled → %s:%d (TCP)", host, int(port_str))
        except Exception as exc:
            logger.warning("OBS_AUDIO_TARGET=%r invalid, disabling: %s", self._target, exc)

    @property
    def enabled(self) -> bool:
        return self._addr is not None

    def _connect(self) -> bool:
        if self._addr is None:
            return False
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(2.0)
            self._sock.connect(self._addr)
            self._sock.settimeout(5.0)
            logger.info("OBS audio forwarder connected to %s:%d", *self._addr)
            return True
        except Exception:
            self._sock = None
            return False

    def send(self, message: dict[str, Any]) -> None:
        if self._addr is None:
            return
        audio_b64 = (message.get("data") or {}).get("audio_b64")
        if not audio_b64:
            return
        try:
            wav_bytes = base64.b64decode(audio_b64)
        except Exception:
            return

        with self._lock:
            if self._sock is None:
                if not self._connect():
                    return
            try:
                header = struct.pack(">I", len(wav_bytes))
                self._sock.sendall(header + wav_bytes)
            except Exception:
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None


_obs_forwarder = _ObsAudioForwarder()


def _forward_audio_to_obs(message: dict[str, Any]) -> None:
    """Send WAV audio to OBS receiver over TCP. Zero contamination."""
    _obs_forwarder.send(message)


class PerceptionServer:
    """WebSocket server on the laptop that receives perception events from the Pi.

    Also supports broadcasting messages back to connected sensors (responses,
    state updates, commands).
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9100) -> None:
        self._host = host
        self._port = port
        self._server: Server | None = None
        self._connections: dict[str, ServerConnection] = {}
        self._event_buffer: deque[PerceptionEvent] = deque(maxlen=1000)
        self._last_sensor_disconnect: float = 0.0
        self._last_sensor_connect: float = 0.0
        self._had_sensor: bool = False
        self._face_identifier: Any = None
        self._synthetic_sensors: set[str] = set()
        self._sensor_health: dict[str, dict] = {}

    def set_face_identifier(self, identifier: Any) -> None:
        self._face_identifier = identifier

    def get_sensor_health(self) -> dict[str, dict]:
        """Return latest health telemetry from all sensors."""
        return dict(self._sensor_health)

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._audio_queue: asyncio.Queue[tuple[str, float]] = asyncio.Queue()
        self._audio_sender_task = asyncio.create_task(self._audio_sender())
        self._server = await serve(
            self._handler,
            self._host,
            self._port,
            reuse_address=True,
            max_size=10 * 1024 * 1024,
            ping_interval=60,
            ping_timeout=120,
            compression=None,
            write_limit=2 ** 22,
        )
        logger.info("Perception bus on ws://%s:%d (write_limit=4MB)", self._host, self._port)

    async def stop(self) -> None:
        if hasattr(self, "_audio_sender_task"):
            self._audio_sender_task.cancel()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._connections.clear()
        logger.info("Perception bus stopped")

    # -- Audio priority send path -----------------------------------------------

    async def _audio_sender(self) -> None:
        """Dedicated sender that drains the audio queue with minimal latency.

        Runs in a tight loop so audio sends are never interleaved with kernel
        ticks or background consciousness work.
        """
        while True:
            try:
                payload, enqueue_t = await self._audio_queue.get()
                _send_t0 = time.monotonic()
                _queue_wait_ms = (_send_t0 - enqueue_t) * 1000
                _depth = self._audio_queue.qsize()

                for sensor_id, ws in list(self._connections.items()):
                    try:
                        await ws.send(payload)
                    except Exception:
                        logger.debug("Audio send failed to %s", sensor_id)

                _send_ms = (time.monotonic() - _send_t0) * 1000
                _kb = len(payload) / 1024
                logger.info(
                    "[AUDIO-SEND] %.0fKB sent in %.0fms (queue_wait=%.0fms depth=%d)",
                    _kb, _send_ms, _queue_wait_ms, _depth,
                )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Audio sender error")

    def broadcast_audio(self, message: dict[str, Any]) -> None:
        """Enqueue an audio message for priority delivery.

        Thread-safe. Audio messages bypass the general broadcast path and are
        sent by _audio_sender with minimal event-loop competition.
        """
        payload = json.dumps(message)
        loop = getattr(self, "_loop", None)
        if loop and loop.is_running():
            loop.call_soon_threadsafe(
                self._audio_queue.put_nowait, (payload, time.monotonic()),
            )
        else:
            self.broadcast(message)

        _forward_audio_to_obs(message)

    # -- General broadcast (non-audio) ------------------------------------------

    def broadcast(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message)
        loop = getattr(self, "_loop", None)
        for sensor_id, ws in list(self._connections.items()):
            try:
                if loop and loop.is_running():
                    loop.call_soon_threadsafe(loop.create_task, ws.send(payload))
                else:
                    asyncio.get_event_loop().create_task(ws.send(payload))
            except Exception:
                logger.debug("Failed to send to %s", sensor_id)

    def send_to_sensor(self, sensor_id: str, message: dict[str, Any]) -> bool:
        ws = self._connections.get(sensor_id)
        if not ws:
            return False
        try:
            loop = getattr(self, "_loop", None)
            if loop and loop.is_running():
                loop.call_soon_threadsafe(loop.create_task, ws.send(json.dumps(message)))
            else:
                asyncio.get_event_loop().create_task(ws.send(json.dumps(message)))
            return True
        except Exception:
            return False

    def send_camera_control(self, control: str, **kwargs) -> None:
        """Send a camera control command to the Pi."""
        cmd = {
            "type": "command",
            "data": {"action": "camera_control", "control": control, **kwargs},
        }
        self.broadcast(cmd)

    def get_connected_sensors(self) -> list[str]:
        return list(self._connections.keys())

    def get_sensor_absent_duration(self) -> float:
        """Seconds since all sensors disconnected, or 0 if any sensor is connected."""
        if self._connections:
            return 0.0
        if self._last_sensor_disconnect > 0:
            return time.time() - self._last_sensor_disconnect
        return 0.0

    @property
    def any_sensor_connected(self) -> bool:
        return bool(self._connections)

    def get_recent_events(self, count: int = 10) -> list[PerceptionEvent]:
        return list(self._event_buffer)[-count:]

    async def _handler(self, websocket: ServerConnection) -> None:
        sensor_id = websocket.request.headers.get("x-sensor-id", f"sensor-{id(websocket)}")
        logger.info("Sensor connected: %s", sensor_id)
        self._connections[sensor_id] = websocket
        self._last_sensor_connect = time.time()
        self._had_sensor = True

        try:
            async for raw in websocket:
                if isinstance(raw, bytes):
                    event_bus.emit(PERCEPTION_RAW_AUDIO, pcm_bytes=raw, sensor_id=sensor_id)
                    continue
                try:
                    data = json.loads(raw)
                    event = PerceptionEvent(
                        source=data.get("source", "system"),
                        type=data.get("type", "unknown"),
                        timestamp=data.get("timestamp", 0),
                        data=data.get("data", {}),
                        confidence=data.get("confidence", 1.0),
                        conversation_id=data.get("conversation_id", ""),
                    )
                    self._process_event(event, sensor_id)
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Invalid event from %s: %s", sensor_id, exc)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            logger.info("Sensor disconnected: %s", sensor_id)
            self._connections.pop(sensor_id, None)
            if sensor_id in self._synthetic_sensors:
                self._synthetic_sensors.discard(sensor_id)
                event_bus.emit(SYNTHETIC_EXERCISE_STATE, active=False, sensor_id=sensor_id)
            if not self._connections:
                self._last_sensor_disconnect = time.time()
                logger.info("All sensors disconnected — brain is solo")
                event_bus.emit(PERCEPTION_SENSOR_DISCONNECTED)

    def _process_event(self, event: PerceptionEvent, sensor_id: str) -> None:
        self._event_buffer.append(event)
        event_bus.emit(PERCEPTION_EVENT, event=event)

        match event.type:
            case "person_detected":
                event_bus.emit(PERCEPTION_USER_PRESENT, present=True, confidence=event.confidence)
            case "person_lost":
                event_bus.emit(PERCEPTION_USER_PRESENT, present=False, confidence=event.confidence)
            case "face_expression":
                expression = event.data.get("expression", "neutral")
                _EXPR_TO_EMOTION = {
                    "happy": "happy", "smile": "happy", "surprise": "excited",
                    "sad": "sad", "angry": "angry", "fear": "frustrated",
                    "disgust": "frustrated",
                }
                mapped = _EXPR_TO_EMOTION.get(expression, "")
                if mapped:
                    event_bus.emit(PERCEPTION_USER_EMOTION,
                                   emotion=mapped, confidence=event.confidence * 0.7,
                                   text_sentiment="neutral",
                                   source="face_expression",
                                   trust="medium")
            case "pose_detected":
                event_bus.emit(PERCEPTION_POSE_DETECTED,
                               keypoints=event.data.get("keypoints", []),
                               gesture=event.data.get("gesture", "neutral"),
                               confidence=event.confidence)
            case "face_crop":
                from consciousness.events import PERCEPTION_FACE_IDENTIFIED
                crop_b64 = event.data.get("crop_b64", "")
                track_id = event.data.get("track_id", -1)
                if crop_b64 and self._face_identifier:
                    try:
                        from memory.gate import memory_gate as _mg
                        _synthetic = _mg.synthetic_session_active()
                    except Exception:
                        _synthetic = False
                    if _synthetic:
                        # Truth-boundary guard: real camera face events must not
                        # mutate identity state or emit identity events during a
                        # synthetic perception exercise.
                        pass
                    else:
                        result = self._face_identifier.identify_b64(crop_b64)
                        if result.get("confidence", 0) > 0:
                            event_bus.emit(PERCEPTION_FACE_IDENTIFIED,
                                           name=result["name"],
                                           confidence=result["confidence"],
                                           is_known=result["is_known"],
                                           track_id=track_id,
                                           closest_match=result.get("closest_match", ""))
            case "playback_complete":
                event_bus.emit(PERCEPTION_PLAYBACK_COMPLETE,
                               conversation_id=event.conversation_id or "")
            case "scene_summary":
                from consciousness.events import PERCEPTION_SCENE_SUMMARY
                event_bus.emit(PERCEPTION_SCENE_SUMMARY,
                               detections=event.data.get("detections", []),
                               frame_size=event.data.get("frame_size", [640, 480]),
                               scene_change_score=event.data.get("scene_change_score", 0.0))
            case "sensor_health":
                health_data = {
                    "cpu_temp_c": event.data.get("cpu_temp_c", 0),
                    "cpu_percent": event.data.get("cpu_percent", 0),
                    "mem_used_mb": event.data.get("mem_used_mb", 0),
                    "mem_total_mb": event.data.get("mem_total_mb", 0),
                    "camera_fps": event.data.get("camera_fps", 0),
                    "throttled": event.data.get("throttled", ""),
                    "uptime_s": event.data.get("uptime_s", 0),
                    "audio_playing": event.data.get("audio_playing", False),
                    "phase": event.data.get("phase", ""),
                    "last_update": time.time(),
                }
                self._sensor_health[sensor_id] = health_data
                throttle_hex = health_data["throttled"]
                if throttle_hex and throttle_hex != "0x0":
                    logger.warning("Pi %s throttle detected: %s (temp=%.1f°C)",
                                   sensor_id, throttle_hex, health_data["cpu_temp_c"])
            case "synthetic_exercise_start":
                logger.info("Synthetic exercise started from sensor %s", sensor_id)
                self._synthetic_sensors.add(sensor_id)
                event_bus.emit(SYNTHETIC_EXERCISE_STATE, active=True, sensor_id=sensor_id)
            case "synthetic_exercise_end":
                logger.info("Synthetic exercise ended from sensor %s", sensor_id)
                self._synthetic_sensors.discard(sensor_id)
                event_bus.emit(SYNTHETIC_EXERCISE_STATE, active=False, sensor_id=sensor_id)

