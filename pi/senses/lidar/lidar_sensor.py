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
RAW_BINS = 360                     # raw per-1° nearest range, streamed for the brain room model
_RAW_DEG = 360.0 / RAW_BINS


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
        raw_min: list[float | None] = [None] * RAW_BINS   # raw per-1° nearest (brain room model)
        raw_qual: list[float | None] = [None] * RAW_BINS  # reflectivity (S2 quality) of that nearest pt
        points = 0
        revolutions = 0
        range_max_mm = 0.0
        raw_n = 0; dropped_quality = 0; dropped_zero = 0   # truth-layer drop telemetry
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
            raw_n += 1
            if qual <= 0:                         # reject qual <= 0
                dropped_quality += 1
            elif dist <= 0:                       # reject dist <= 0
                dropped_zero += 1
            else:
                s = int(ang // _SECTOR_DEG) % SECTORS
                if sector_min[s] is None or dist < sector_min[s]:
                    sector_min[s] = dist
                rb = int(ang // _RAW_DEG) % RAW_BINS
                if raw_min[rb] is None or dist < raw_min[rb]:
                    raw_min[rb] = dist
                    raw_qual[rb] = qual          # keep the reflectivity of the nearest return
                if dist > range_max_mm:
                    range_max_mm = dist
                points += 1

            now = time.time()
            if now - last_emit >= self._emit_interval:
                self._emit_summary(sector_min, raw_min, raw_qual, points, revolutions, range_max_mm,
                                   now - window_start, raw_n, dropped_quality, dropped_zero)
                sector_min = [None] * SECTORS
                raw_min = [None] * RAW_BINS
                raw_qual = [None] * RAW_BINS
                points = 0
                revolutions = 0
                range_max_mm = 0.0
                raw_n = 0; dropped_quality = 0; dropped_zero = 0
                window_start = now
                last_emit = now

    def _emit_summary(self, sector_min, raw_min, raw_qual, points, revolutions, range_max_mm, dt,
                      raw_n=0, dropped_quality=0, dropped_zero=0) -> None:
        sectors = {}
        open_sectors = []
        for i, d in enumerate(sector_min):
            if d is None:
                open_sectors.append(i)            # no return in this sector → open
                continue
            sectors[str(i)] = round(d / 1000.0, 2)  # nearest distance in metres
            if d > OPEN_SECTOR_MM:
                open_sectors.append(i)
        # Raw per-1°-bin nearest points in CANONICAL units for the brain room model:
        # deg→rad + mm→m converted ONCE here, bin-center bearing (the Pi stays thin —
        # just a 360-slot nearest array it already derives, no histogram/denoise/SLAM).
        points_polar = [
            [round(math.radians(i * _RAW_DEG + 0.5 * _RAW_DEG), 5), round(d / 1000.0, 3),
             round(raw_qual[i] or 0.0, 1)]                # reflectivity (S2 quality, ~0-255)
            for i, d in enumerate(raw_min) if d is not None
        ]
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
            "points_polar": points_polar,         # [[bearing_rad, range_m], ...]
            # Pi truth-layer drop telemetry (the brain reports its own range-gating drops)
            "raw_points": raw_n,
            "dropped_quality": dropped_quality,
            "dropped_zero": dropped_zero,
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
