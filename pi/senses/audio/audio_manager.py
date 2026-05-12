"""Audio I/O manager — microphone capture and speaker playback.

Thin sensor architecture: the Pi captures raw mic audio and streams it to
the brain. All processing (wake word, VAD, STT) happens on the brain.

Architecture:
  - Single shared InputStream captures mic audio at native rate
  - Audio callback fires a user-provided function (for streaming to brain)
  - PlaybackWorker thread owns ALSA output (brain-synthesized audio)
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import subprocess
import threading
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    logger.warning("sounddevice not available — audio I/O disabled")


def _refresh_device_list() -> None:
    if not HAS_SOUNDDEVICE:
        return
    try:
        sd._terminate()
        sd._initialize()
        logger.info("PortAudio device list refreshed")
    except Exception:
        pass


def _log_all_devices() -> None:
    if not HAS_SOUNDDEVICE:
        return
    for i, d in enumerate(sd.query_devices()):
        logger.info("  Device %d: %s (in=%d, out=%d, rate=%d)",
                     i, d["name"], d["max_input_channels"],
                     d["max_output_channels"], int(d["default_samplerate"]))


def _find_device(name: str, kind: str) -> int | None:
    if not HAS_SOUNDDEVICE:
        return None
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower() and d[key] > 0:
            return i
    if kind == "input":
        for i, d in enumerate(sd.query_devices()):
            if name.lower() in d["name"].lower():
                try:
                    s = sd.InputStream(device=i, samplerate=int(d["default_samplerate"]),
                                       channels=1, dtype="float32", blocksize=1024)
                    s.close()
                    return i
                except Exception:
                    continue
    return None


def _find_alsa_card(name: str) -> str:
    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if line.startswith("card ") and name.lower() in line.lower():
                card_num = line.split(":")[0].replace("card ", "").strip()
                return f"plughw:{card_num},0"
    except Exception:
        pass
    return "plughw:2,0"


def _find_first_usb_speaker() -> str:
    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if line.startswith("card ") and "usb" in line.lower():
                card_num = line.split(":")[0].replace("card ", "").strip()
                return f"plughw:{card_num},0"
    except Exception:
        pass
    return "plughw:2,0"


_SENTINEL = object()


class _PlaybackItem:
    __slots__ = ("wav_path", "delete_after", "generation")

    def __init__(self, wav_path: str, delete_after: bool = False, generation: int = 0) -> None:
        self.wav_path = wav_path
        self.delete_after = delete_after
        self.generation = generation


class PlaybackWorker:
    """Single-threaded audio output. Only this thread calls aplay or mute/unmute."""

    def __init__(self, speaker_alsa: str, mute_fn: Callable, unmute_fn: Callable) -> None:
        self._speaker_alsa = speaker_alsa
        self._mute = mute_fn
        self._unmute = unmute_fn
        self._queue: queue.Queue[_PlaybackItem | object] = queue.Queue()
        self._current_proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._playing = False
        self._generation: int = 0

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="playback-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._queue.put(_SENTINEL)
        if self._thread:
            self._thread.join(timeout=5.0)

    def enqueue(self, wav_path: str, delete_after: bool = False) -> None:
        self._queue.put(_PlaybackItem(wav_path, delete_after, self._generation))

    @property
    def is_playing(self) -> bool:
        return self._playing

    def hard_stop(self) -> None:
        with self._proc_lock:
            if self._current_proc and self._current_proc.poll() is None:
                try:
                    self._current_proc.send_signal(signal.SIGKILL)
                except OSError:
                    pass

    def cancel(self) -> None:
        self._generation += 1
        discarded: list[_PlaybackItem] = []
        while True:
            try:
                item = self._queue.get_nowait()
                if isinstance(item, _PlaybackItem):
                    discarded.append(item)
            except queue.Empty:
                break
        self.hard_stop()
        for item in discarded:
            if item.delete_after:
                try:
                    os.unlink(item.wav_path)
                except OSError:
                    pass

    def _run(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                break
            self._playing = True
            self._mute()
            self._play_item(item)
            self._drain_remaining()
            self._unmute()
            self._playing = False

    def _drain_remaining(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is _SENTINEL:
                self._running = False
                return
            self._play_item(item)

    def _play_item(self, item: _PlaybackItem) -> None:
        if item.generation < self._generation:
            if item.delete_after:
                try:
                    os.unlink(item.wav_path)
                except OSError:
                    pass
            return
        try:
            with self._proc_lock:
                self._current_proc = subprocess.Popen(
                    ["aplay", "-D", self._speaker_alsa, item.wav_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            self._current_proc.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            logger.error("aplay hung for 30s — killing")
            self.hard_stop()
        except FileNotFoundError:
            logger.error("aplay not found or WAV missing: %s", item.wav_path)
        except Exception as exc:
            logger.error("Playback error: %s", exc)
        finally:
            with self._proc_lock:
                self._current_proc = None
            if item.delete_after:
                try:
                    os.unlink(item.wav_path)
                except OSError:
                    pass


class AudioManager:
    """Manages microphone capture and speaker output for the thin-sensor Pi."""

    def __init__(
        self,
        mic_sample_rate: int = 48000,
        channels: int = 1,
        mic_name: str = "",
        speaker_name: str = "",
    ):
        self.mic_sample_rate = mic_sample_rate
        self.channels = channels
        self.is_muted = False
        self._mute_lock = threading.Lock()

        self._mic_name = mic_name
        if mic_name:
            self.mic_device = _find_device(mic_name, "input")
        else:
            self.mic_device = self._find_any_input()

        if self.mic_device is None and HAS_SOUNDDEVICE:
            import time as _bt
            for attempt in range(1, 4):
                wait = 2 * attempt
                logger.warning("No mic found (attempt %d) — waiting %ds for USB device...", attempt, wait)
                _bt.sleep(wait)
                _refresh_device_list()
                if mic_name:
                    self.mic_device = _find_device(mic_name, "input")
                else:
                    self.mic_device = self._find_any_input()
                if self.mic_device is not None:
                    logger.info("Mic found on retry %d: device=%d", attempt, self.mic_device)
                    break

        self.speaker_alsa = _find_alsa_card(speaker_name) if speaker_name else _find_first_usb_speaker()

        self._shared_stream: sd.InputStream | None = None
        self._stream_callback: Callable | None = None

        # Stream health diagnostics
        self._callback_count: int = 0
        self._overflow_count: int = 0
        self._restart_lock = threading.Lock()
        self._restarting = False
        self._consecutive_stalls: int = 0

        self._on_speaking_change: Callable[[bool], None] | None = None

        self._playback = PlaybackWorker(self.speaker_alsa, self._mute, self._unmute)
        self._playback.start()

        logger.info("Audio: mic_device=%s, speaker=%s", self.mic_device, self.speaker_alsa)

    @staticmethod
    def _find_any_input() -> int | None:
        if not HAS_SOUNDDEVICE:
            return None
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                return i
        for i, d in enumerate(sd.query_devices()):
            try:
                s = sd.InputStream(device=i, samplerate=int(d["default_samplerate"]),
                                   channels=1, dtype="float32", blocksize=1024)
                s.close()
                return i
            except Exception:
                continue
        return None

    # -- Stream callback for brain audio streaming --------------------------

    def set_stream_callback(self, callback: Callable) -> None:
        """Set the callback that receives raw audio from the mic.

        The callback signature should match sounddevice:
            callback(indata: np.ndarray, frames: int, time_info, status)
        """
        self._stream_callback = callback

    # -- Shared mic stream --------------------------------------------------

    def start_shared_stream(self, blocksize: int = 4096) -> None:
        if not HAS_SOUNDDEVICE or self._shared_stream is not None:
            return
        if self.mic_device is None:
            self.mic_device = self._find_any_input()
        if self.mic_device is None:
            logger.warning("No microphone found — audio input disabled")
            _log_all_devices()
            return

        try:
            dev_info = sd.query_devices(self.mic_device)
            max_ch = dev_info["max_input_channels"]
            if max_ch < 1:
                logger.warning("Device %d (%s) reports 0 input channels",
                               self.mic_device, dev_info["name"])
                _log_all_devices()
                return
            channels = min(self.channels, max_ch)
        except Exception:
            channels = self.channels

        rate = self.mic_sample_rate
        try:
            self._shared_stream = sd.InputStream(
                device=self.mic_device,
                samplerate=rate,
                channels=channels,
                blocksize=blocksize,
                dtype="float32",
                callback=self._shared_callback,
            )
        except sd.PortAudioError:
            try:
                dev = sd.query_devices(self.mic_device)
                rate = int(dev["default_samplerate"])
                logger.warning("Mic doesn't support %dHz, using default %dHz", self.mic_sample_rate, rate)
                self.mic_sample_rate = rate
                self._shared_stream = sd.InputStream(
                    device=self.mic_device,
                    samplerate=rate,
                    channels=channels,
                    blocksize=blocksize,
                    dtype="float32",
                    callback=self._shared_callback,
                )
            except Exception as exc:
                logger.error("Failed to open mic stream: %s", exc)
                _log_all_devices()
                return
        self._shared_stream.start()
        self._unmute_alsa_capture()
        logger.info("Shared mic stream started: %dHz, %dch, device=%d",
                     rate, channels, self.mic_device)

    @property
    def is_stream_active(self) -> bool:
        return self._shared_stream is not None and self._shared_stream.active

    def restart_shared_stream(self, max_attempts: int = 5) -> None:
        import time as _t
        if not self._restart_lock.acquire(blocking=False):
            logger.debug("Restart already in progress — skipping")
            return
        try:
            self._restarting = True
            self._callback_count = 0
            logger.warning("Restarting shared mic stream (stalls=%d)", self._consecutive_stalls)
            self.stop_shared_stream()
            _t.sleep(0.5)

            if self._consecutive_stalls >= 2:
                logger.warning("Multiple consecutive stalls — full PortAudio reset")
                try:
                    sd._terminate()
                    _t.sleep(1.0)
                    sd._initialize()
                except Exception as exc:
                    logger.warning("PortAudio reset error: %s", exc)

            old_dev = self.mic_device
            mic_name = self._mic_name

            for attempt in range(1, max_attempts + 1):
                _refresh_device_list()
                self.mic_device = None
                if mic_name:
                    self.mic_device = _find_device(mic_name, "input")
                if self.mic_device is None:
                    self.mic_device = self._find_any_input()
                self.start_shared_stream()
                if self._shared_stream and self._shared_stream.active:
                    self._callback_count = 0
                    for _ in range(20):
                        _t.sleep(0.1)
                        if self._callback_count > 0:
                            self._consecutive_stalls = 0
                            logger.info("Mic stream recovered on attempt %d (device=%s)",
                                        attempt, self.mic_device)
                            return
                    logger.warning("Stream opened but no callbacks — retrying")
                    self.stop_shared_stream()
                backoff = min(2 ** attempt, 15)
                logger.warning("Mic restart attempt %d/%d failed — retrying in %ds",
                               attempt, max_attempts, backoff)
                _t.sleep(backoff)
            self._consecutive_stalls += 1
            logger.error("Mic stream could not be recovered after %d attempts", max_attempts)
            _log_all_devices()
        except Exception:
            logger.exception("Unexpected error during mic restart")
        finally:
            self._restarting = False
            self._restart_lock.release()

    def stop_shared_stream(self) -> None:
        s = self._shared_stream
        self._shared_stream = None
        if s:
            try:
                s.stop()
            except Exception as exc:
                logger.warning("Error stopping mic stream: %s", exc)
            try:
                s.close()
            except Exception as exc:
                logger.warning("Error closing mic stream: %s", exc)

    def shutdown(self) -> None:
        self.stop_shared_stream()
        self._playback.stop()

    def _shared_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        try:
            self._callback_count += 1
            if status:
                self._overflow_count += 1
            if self._stream_callback:
                self._stream_callback(indata, frames, time_info, status)
        except Exception as exc:
            logger.error("Shared callback error: %s", exc)

    # -- Muting (called by PlaybackWorker) ----------------------------------

    def set_speaking_callback(self, callback: Callable[[bool], None]) -> None:
        self._on_speaking_change = callback

    def _mute(self) -> None:
        with self._mute_lock:
            self.is_muted = True
        if self._on_speaking_change:
            try:
                self._on_speaking_change(True)
            except Exception:
                pass

    def _unmute(self) -> None:
        with self._mute_lock:
            self.is_muted = False
        if self._on_speaking_change:
            try:
                self._on_speaking_change(False)
            except Exception:
                pass

    # -- Public playback API ------------------------------------------------

    def play_wav(self, filepath: str, delete_after: bool = False) -> None:
        self._playback.enqueue(filepath, delete_after=delete_after)

    def hard_stop(self) -> None:
        self._playback.hard_stop()

    def cancel_playback(self) -> None:
        self._playback.cancel()

    @property
    def is_playing(self) -> bool:
        return self._playback.is_playing

    # -- ALSA capture unmute ------------------------------------------------

    @staticmethod
    def _unmute_alsa_capture() -> None:
        try:
            result = subprocess.run(
                ["arecord", "-l"], capture_output=True, text=True, check=False,
            )
            for line in result.stdout.splitlines():
                if not line.startswith("card "):
                    continue
                card_num = line.split(":")[0].replace("card ", "").strip()
                _unmute_card_by_name(card_num)
        except Exception:
            pass


def _unmute_card_by_name(card_num: str) -> None:
    try:
        result = subprocess.run(
            ["amixer", "-c", card_num, "contents"],
            capture_output=True, text=True, check=False,
        )
    except Exception:
        return
    lines = result.stdout.splitlines()
    for i, line in enumerate(lines):
        lower = line.lower()
        if "capture" in lower and "switch" in lower and "type=BOOLEAN" in line:
            if "numid=" not in line:
                continue
            numid = line.split("numid=")[1].split(",")[0]
            subprocess.run(
                ["amixer", "-c", card_num, "cset", f"numid={numid}", "on"],
                capture_output=True, check=False,
            )
            logger.info("Unmuted capture on card %s (numid=%s)", card_num, numid)
        if "capture" in lower and "volume" in lower and "type=INTEGER" in line:
            if "numid=" not in line:
                continue
            numid = line.split("numid=")[1].split(",")[0]
            subprocess.run(
                ["amixer", "-c", card_num, "cset", f"numid={numid}", "100%"],
                capture_output=True, check=False,
            )
            logger.info("Capture volume maxed on card %s (numid=%s)", card_num, numid)
