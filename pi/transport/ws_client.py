"""Bidirectional WebSocket client — streams events to laptop, receives responses.

Supports two frame types:
  - JSON text frames for PerceptionEvent / BrainMessage
  - Binary frames for raw PCM audio streaming
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from collections import deque
from typing import Callable

import websockets
from websockets.asyncio.client import connect

from .event_schema import PerceptionEvent, BrainMessage, CRITICAL_EVENT_TYPES

logger = logging.getLogger(__name__)

_AUDIO_QUEUE_MAX = 50  # ~1.5s of 30ms chunks — audio is perishable


class TransportClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9100,
        sensor_id: str = "pi5-senses",
        reconnect_interval: float = 3.0,
        buffer_max: int = 500,
    ):
        self._uri = f"ws://{host}:{port}"
        self._sensor_id = sensor_id
        self._reconnect_interval = reconnect_interval
        self._critical_buffer: deque[str] = deque(maxlen=200)
        self._telemetry_buffer: deque[str] = deque(maxlen=buffer_max)
        self._ws = None
        self._running = False
        self._connected = False
        self._send_queue: asyncio.Queue[str] = asyncio.Queue()
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=_AUDIO_QUEUE_MAX)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_brain_message: Callable[[BrainMessage], None] | None = None
        self._on_disconnect: Callable[[], None] | None = None
        self._brain_msg_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="brain-msg",
        )
        self._brain_audio_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="brain-audio",
        )

    def on_brain_message(self, callback: Callable[[BrainMessage], None]):
        """Register callback for messages from the laptop brain."""
        self._on_brain_message = callback

    def on_disconnect(self, callback: Callable[[], None]):
        """Register callback fired when the brain WebSocket connection drops."""
        self._on_disconnect = callback

    async def start(self) -> None:
        self._running = True
        self._loop = asyncio.get_running_loop()
        asyncio.create_task(self._connection_loop())
        asyncio.create_task(self._send_loop())
        asyncio.create_task(self._audio_send_loop())
        logger.info("Transport started, target: %s", self._uri)

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._connected = False

    def send_event(self, event: PerceptionEvent) -> None:
        """Thread-safe: can be called from any thread (audio callbacks, etc.)."""
        payload = event.to_json()
        if self._connected and self._loop:
            self._loop.call_soon_threadsafe(self._send_queue.put_nowait, payload)
        else:
            if event.type in CRITICAL_EVENT_TYPES:
                self._critical_buffer.append(payload)
            else:
                self._telemetry_buffer.append(payload)

    def send_audio(self, pcm_bytes: bytes) -> None:
        """Thread-safe: push raw PCM audio for binary WebSocket streaming.

        Audio is perishable — if the queue is full the oldest chunk is dropped
        to make room. Silently does nothing when disconnected.
        """
        if not self._connected or not self._loop:
            return

        def _enqueue():
            if self._audio_queue.full():
                try:
                    self._audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                self._audio_queue.put_nowait(pcm_bytes)
            except asyncio.QueueFull:
                pass

        self._loop.call_soon_threadsafe(_enqueue)

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def _connection_loop(self) -> None:
        while self._running:
            try:
                extra_headers = {"X-Sensor-Id": self._sensor_id}
                self._ws = await connect(
                    self._uri,
                    additional_headers=extra_headers,
                    max_size=10 * 1024 * 1024,
                    compression=None,
                )
                self._connected = True
                logger.info("Connected to brain at %s", self._uri)

                await self._flush_buffer()

                async for raw in self._ws:
                    self._handle_incoming(raw)

            except (OSError, websockets.WebSocketException) as exc:
                logger.warning("Connection failed: %s. Retrying in %.1fs", exc, self._reconnect_interval)
            finally:
                was_connected = self._connected
                self._connected = False
                self._ws = None
                if was_connected and self._on_disconnect:
                    try:
                        self._on_disconnect()
                    except Exception:
                        logger.exception("on_disconnect callback failed")

            if self._running:
                await asyncio.sleep(self._reconnect_interval)

    _AUDIO_MSG_TYPES = frozenset({"response_chunk", "response_end", "response"})

    def _handle_incoming(self, raw: str | bytes) -> None:
        """Process messages from the laptop brain.

        Audio-related messages (response_chunk, response_end, response) are
        dispatched to a dedicated audio executor so they are never blocked by
        state_update / consciousness_feed processing in the regular executor.
        """
        try:
            data = json.loads(raw)
            msg = BrainMessage(**data)
            logger.debug("Brain message: type=%s", msg.type)
            if self._on_brain_message and self._loop:
                is_audio = msg.type in self._AUDIO_MSG_TYPES
                executor = self._brain_audio_executor if is_audio else self._brain_msg_executor
                if is_audio:
                    import time as _t
                    object.__setattr__(msg, "_ws_arrival", _t.monotonic())
                self._loop.run_in_executor(executor, self._on_brain_message, msg)
        except Exception as exc:
            logger.warning("Invalid brain message: %s", exc)

    async def _send_loop(self) -> None:
        while self._running:
            try:
                payload = await asyncio.wait_for(self._send_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if self._ws and self._connected:
                try:
                    await self._ws.send(payload)
                except websockets.ConnectionClosed:
                    self._telemetry_buffer.append(payload)
                    self._connected = False
            else:
                self._telemetry_buffer.append(payload)

    async def _audio_send_loop(self) -> None:
        """Send raw PCM binary frames. Audio is perishable — dropped on disconnect."""
        sent_count = 0
        drop_count = 0
        last_log = 0.0
        import time as _time

        while self._running:
            try:
                pcm_bytes = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if self._ws and self._connected:
                try:
                    await self._ws.send(pcm_bytes)
                    sent_count += 1
                except websockets.ConnectionClosed:
                    self._connected = False
                except Exception as exc:
                    logger.error("Audio send error: %s", exc)
            else:
                drop_count += 1

            now = _time.monotonic()
            if now - last_log >= 10.0:
                last_log = now
                qsz = self._audio_queue.qsize()
                logger.info("Audio stream: sent=%d dropped=%d qsize=%d connected=%s",
                            sent_count, drop_count, qsz, self._connected)

    async def _flush_buffer(self) -> None:
        critical_count = 0
        telemetry_count = 0
        while self._critical_buffer and self._ws and self._connected:
            payload = self._critical_buffer.popleft()
            try:
                await self._ws.send(payload)
                critical_count += 1
            except websockets.ConnectionClosed:
                self._critical_buffer.appendleft(payload)
                return
        while self._telemetry_buffer and self._ws and self._connected:
            payload = self._telemetry_buffer.popleft()
            try:
                await self._ws.send(payload)
                telemetry_count += 1
            except websockets.ConnectionClosed:
                self._telemetry_buffer.appendleft(payload)
                break
        total = critical_count + telemetry_count
        if total:
            logger.info("Flushed %d buffered events (%d critical)", total, critical_count)
