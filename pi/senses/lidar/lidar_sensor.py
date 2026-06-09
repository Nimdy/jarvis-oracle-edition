"""RPLIDAR S2 reader — emits a 2D sector summary to the brain (telemetry-only).

The S2 is an S-series ToF lidar: 1 Mbaud, model 113. The legacy A-series scan
descriptor does NOT match it (rplidar's iter_scans → "descriptor length mismatch"),
and the device's typical mode is DenseBoost/DENSE_CAPSULED which older python libs
don't parse. The robust path is pyrplidar's Standard scan (NORMAL ans_type), which
reads cleanly at 1 Mbaud. We aggregate raw points into N angular sectors (nearest
distance per sector) so the brain gets a compact, telemetry-only shape — never
beliefs, never object identity (matches the brain's lidar sink contract).

Runs as a daemon thread; pyrplidar is blocking, so it lives off the asyncio loop and
hands finished summaries back via a thread-safe callback.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 1_000_000          # S2 = 1 Mbaud
SECTORS = 12                       # 30° each
OPEN_SECTOR_MM = 1500              # a sector with nearest return > this (or empty) is "open"
EMIT_HZ = 5.0                      # summaries per second sent to the brain
MIN_POINTS_GOOD = 120              # below this in a window → scan_quality "sparse"
_SECTOR_DEG = 360.0 / SECTORS


class LidarSensor:
    """Reads the S2 and pushes 2D sector summaries via ``emit(summary: dict)``."""

    def __init__(
        self,
        emit: Callable[[dict], None],
        port: str = DEFAULT_PORT,
        baud: int = DEFAULT_BAUD,
        emit_hz: float = EMIT_HZ,
    ) -> None:
        self._emit = emit
        self._port = port
        self._baud = baud
        self._emit_interval = 1.0 / max(0.5, emit_hz)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lidar = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="lidar", daemon=True)
        self._thread.start()
        logger.info("LidarSensor started (port=%s baud=%d)", self._port, self._baud)

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._lidar is not None:
                self._lidar.stop()
                self._lidar.set_motor_pwm(0)
                self._lidar.disconnect()
        except Exception:
            pass
        self._lidar = None

    # -- internals ---------------------------------------------------------

    def _raw_recover(self) -> None:
        """A reader that died without STOP leaves the S2 streaming a scan, so a fresh
        connection lands mid-stream → "sync bytes mismatched" on the first command.
        Send STOP + flush + RESET below pyrplidar (raw serial) to guarantee a clean
        protocol state. Idempotent + best-effort."""
        try:
            import serial
            s = serial.Serial(self._port, self._baud, timeout=1)
            try:
                s.write(b"\xA5\x25"); time.sleep(0.4)   # STOP scan
                s.reset_input_buffer()                   # discard stale scan bytes
                s.write(b"\xA5\x40"); time.sleep(3.0)    # RESET (reboot device)
                s.reset_input_buffer()                   # discard boot banner
            finally:
                s.close()
            time.sleep(0.6)
        except Exception:
            logger.debug("lidar raw-recovery skipped", exc_info=True)

    def _connect(self):
        from pyrplidar import PyRPlidar
        self._raw_recover()            # guarantee a clean protocol state first
        lidar = PyRPlidar()
        lidar.connect(port=self._port, baudrate=self._baud, timeout=3)
        try:
            lidar.set_motor_pwm(600)   # harmless if the motor runs on external power
        except Exception:
            pass
        time.sleep(2.0)                # let the motor reach speed
        return lidar

    def _run(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            try:
                self._lidar = self._connect()
                self._scan_loop()
                backoff = 2.0
            except Exception as exc:
                logger.warning("Lidar loop error (%s) — retrying in %.0fs", exc, backoff)
                try:
                    if self._lidar is not None:
                        self._lidar.disconnect()
                except Exception:
                    pass
                self._lidar = None
                self._stop.wait(backoff)
                backoff = min(15.0, backoff * 1.5)

    def _scan_loop(self) -> None:
        gen = self._lidar.start_scan()           # Standard / NORMAL mode
        sector_min: list[float | None] = [None] * SECTORS
        points = 0
        revolutions = 0
        range_max_mm = 0.0
        window_start = time.time()
        last_emit = window_start

        for m in gen():
            if self._stop.is_set():
                break
            if getattr(m, "start_flag", False):
                revolutions += 1
            dist = float(getattr(m, "distance", 0) or 0)
            qual = float(getattr(m, "quality", 0) or 0)
            ang = float(getattr(m, "angle", 0) or 0) % 360.0
            if dist > 0 and qual > 0:
                s = int(ang // _SECTOR_DEG) % SECTORS
                if sector_min[s] is None or dist < sector_min[s]:
                    sector_min[s] = dist
                if dist > range_max_mm:
                    range_max_mm = dist
                points += 1

            now = time.time()
            if now - last_emit >= self._emit_interval:
                self._emit_summary(sector_min, points, revolutions, range_max_mm,
                                   now - window_start)
                sector_min = [None] * SECTORS
                points = 0
                revolutions = 0
                range_max_mm = 0.0
                window_start = now
                last_emit = now

    def _emit_summary(self, sector_min, points, revolutions, range_max_mm, dt) -> None:
        sectors = {}
        open_sectors = []
        for i, d in enumerate(sector_min):
            if d is None:
                open_sectors.append(i)            # no return in this sector → open
                continue
            sectors[str(i)] = round(d / 1000.0, 2)  # nearest distance in metres
            if d > OPEN_SECTOR_MM:
                open_sectors.append(i)
        scan_hz = round(revolutions / dt, 1) if dt > 0 else 0.0
        quality = "good" if points >= MIN_POINTS_GOOD else "sparse"
        summary = {
            "sensor": "rplidar_s2",
            "scan_hz": scan_hz,
            "points": points,
            "range_max_m": round(range_max_mm / 1000.0, 2),
            "sectors": sectors,                   # {sector_idx: nearest_metres}
            "open_sectors": sorted(open_sectors),
            "scan_quality": quality,
        }
        try:
            self._emit(summary)
        except Exception:
            logger.debug("lidar emit callback failed", exc_info=True)


# Standalone smoke-test: prints sector summaries from the real device.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def _print(s):
        occ = {k: v for k, v in s["sectors"].items()}
        print(f"hz={s['scan_hz']} pts={s['points']} max={s['range_max_m']}m "
              f"open={s['open_sectors']} quality={s['scan_quality']} sectors={occ}")

    sensor = LidarSensor(_print)
    sensor.start()
    try:
        time.sleep(8)
    finally:
        sensor.stop()
        print("done")
