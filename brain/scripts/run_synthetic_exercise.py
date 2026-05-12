#!/usr/bin/env python3
"""Synthetic perception exercise runner.

Generates TTS utterances and sends them as binary PCM audio over WebSocket
to the brain's perception server. The perception pipeline processes them
through wake word, VAD, STT, speaker ID, emotion, and tool router — then
hard-stops before conversation handler / LLM / memory / identity.

Usage:
    python -m scripts.run_synthetic_exercise --profile smoke
    python -m scripts.run_synthetic_exercise --profile route_coverage
    python -m scripts.run_synthetic_exercise --profile idle_soak --duration 3600
    python -m scripts.run_synthetic_exercise --count 20 --category command
    python -m scripts.run_synthetic_exercise --target-route ACADEMIC_SEARCH -n 30

Requires:
    - Brain server running on localhost:9100 (perception WebSocket)
    - Kokoro TTS model available (brain/models/kokoro-v1.0.onnx)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [synth] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("synthetic_exercise")

DEFAULT_REPORT_DIR = os.path.expanduser("~/.jarvis/synthetic_exercise/reports")

ROUTE_TO_CATEGORY: dict[str, str] = {
    "TIME": "time",
    "SYSTEM_STATUS": "command",
    "STATUS": "status",
    "INTROSPECTION": "introspection",
    "MEMORY": "memory",
    "WEB_SEARCH": "web_search",
    "ACADEMIC_SEARCH": "academic",
    "CODEBASE": "codebase",
    "SKILL": "skill",
    "IDENTITY": "identity",
    "VISION": "vision",
    "LIBRARY_INGEST": "library",
}


def resample_24k_to_16k(samples_24k: np.ndarray) -> np.ndarray:
    """Resample 24kHz float32 audio to 16kHz using linear interpolation."""
    n_out = int(len(samples_24k) * 16000 / 24000)
    x_old = np.linspace(0, 1, len(samples_24k))
    x_new = np.linspace(0, 1, n_out)
    return np.interp(x_new, x_old, samples_24k)


def float32_to_int16_bytes(audio_f32: np.ndarray) -> bytes:
    """Convert float32 audio [-1, 1] to int16 PCM bytes."""
    int16 = (audio_f32 * 32768.0).clip(-32768, 32767).astype(np.int16)
    return int16.tobytes()


def write_report(stats, report_dir: str) -> str | None:
    """Write JSON report and return the file path."""
    try:
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        profile = stats.profile_name or "manual"
        filename = f"{ts}_{profile}.json"
        path = os.path.join(report_dir, filename)

        report = {
            "version": "1.0",
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            **stats.to_dict(),
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Report written: %s", path)
        return path
    except Exception as exc:
        logger.warning("Failed to write report: %s", exc)
        return None


async def fetch_brain_stats(host: str, port: int) -> dict | None:
    """Fetch synthetic exercise stats from the brain dashboard API."""
    dashboard_port = port + 100 if port == 9100 else 9200
    url = f"http://{host}:{dashboard_port}/api/synthetic-exercise"
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    return await resp.json()
    except ImportError:
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception:
            pass
    except Exception:
        pass
    return None


async def run_exercise(
    host: str,
    port: int,
    count: int,
    delay_s: float,
    drain_s: float,
    voice: str,
    speed: float,
    category: str | None,
    weights: dict[str, float] | None,
    profile_name: str,
    report_dir: str,
) -> None:
    from synthetic.exercise import ExerciseStats, pick_utterance

    stats = ExerciseStats(
        utterances_requested=count,
        profile_name=profile_name,
    )

    # --- Init TTS ---
    logger.info("Initializing Kokoro TTS (voice=%s, speed=%.1f)...", voice, speed)
    try:
        from reasoning.tts import BrainTTS
        tts = BrainTTS(voice=voice, speed=speed, device="cpu")
        if not tts.available:
            logger.error("Kokoro TTS not available — cannot run exercise")
            return
    except Exception as exc:
        logger.error("TTS initialization failed: %s", exc)
        return
    logger.info("TTS ready")

    # --- Connect WebSocket ---
    import base64
    import io
    import wave
    import websockets

    uri = f"ws://{host}:{port}"
    headers = {"x-sensor-id": "synthetic-exercise"}
    loop = asyncio.get_running_loop()
    MAX_RECONNECTS = 10

    async def _connect():
        ws = await websockets.connect(
            uri, additional_headers=headers, max_size=10 * 1024 * 1024,
            ping_interval=None, ping_timeout=None,
        )
        await ws.send(json.dumps({"type": "synthetic_exercise_start", "source": "synthetic"}))
        return ws

    logger.info("Connecting to %s ...", uri)
    try:
        ws = await _connect()
    except ConnectionRefusedError:
        logger.error("Cannot connect to %s — is the brain server running?", uri)
        return
    except Exception as exc:
        logger.error("Initial connection failed: %s", exc)
        return

    logger.info(
        "Connected — starting %d utterances (profile=%s, delay=%.1fs)",
        count, profile_name or "manual", delay_s,
    )

    for i in range(count):
        text, cat = pick_utterance(category, weights=weights)
        stats.categories_exercised[cat] += 1

        try:
            audio_b64 = await loop.run_in_executor(
                None, tts.synthesize_b64, text,
            )
            if not audio_b64:
                stats.utterances_failed += 1
                stats.errors.append(f"TTS returned None for: {text[:40]}")
                continue

            wav_bytes = base64.b64decode(audio_b64)
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                pcm_24k = np.frombuffer(
                    wf.readframes(wf.getnframes()), dtype=np.int16,
                ).astype(np.float32) / 32768.0

            pcm_16k = resample_24k_to_16k(pcm_24k)
            pcm_bytes = float32_to_int16_bytes(pcm_16k)

            CHUNK_SIZE = 32000
            for offset in range(0, len(pcm_bytes), CHUNK_SIZE):
                chunk = pcm_bytes[offset : offset + CHUNK_SIZE]
                await ws.send(chunk)
                await asyncio.sleep(0.02)

            stats.utterances_sent += 1
            logger.info(
                "[%d/%d] %s | cat=%s | %.1fs audio | %.1f/min",
                i + 1, count, text[:50], cat,
                len(pcm_16k) / 16000, stats.rate_per_min,
            )

        except Exception as exc:
            if stats.reconnect_count >= MAX_RECONNECTS:
                stats.utterances_failed += 1
                stats.errors.append(f"Max reconnects reached: {exc}")
                logger.error("Max reconnects (%d) reached — aborting", MAX_RECONNECTS)
                break

            logger.warning("Send failed (utterance %d): %s — reconnecting...", i + 1, exc)
            stats.utterances_failed += 1
            stats.errors.append(f"reconnect after: {exc}")

            try:
                try:
                    await ws.close()
                except Exception:
                    pass
                await asyncio.sleep(2.0)
                ws = await _connect()
                stats.reconnect_count += 1
                logger.info("Reconnected (#%d) — resuming at utterance %d", stats.reconnect_count, i + 1)
            except Exception as reconn_exc:
                stats.reconnect_failures += 1
                logger.error("Reconnect failed: %s — aborting", reconn_exc)
                stats.errors.append(f"reconnect failed: {reconn_exc}")
                break

        if i < count - 1:
            await asyncio.sleep(delay_s)

    # --- Drain and close ---
    logger.info("All utterances sent — draining audio pipeline (%.0fs)...", drain_s)
    await asyncio.sleep(drain_s)
    try:
        await ws.send(json.dumps({"type": "synthetic_exercise_end", "source": "synthetic"}))
        await ws.close()
    except Exception:
        pass

    stats.end_time = time.time()

    # --- Fetch brain-side stats ---
    logger.info("Fetching brain-side stats...")
    await asyncio.sleep(1.0)
    brain_stats = await fetch_brain_stats(host, port)
    if brain_stats:
        stats.brain_utterances_stt = brain_stats.get("utterances_stt", 0)
        stats.brain_hard_stopped = brain_stats.get("utterances_stt", 0)
        stats.brain_routes_produced = brain_stats.get("routes_produced", 0)
        stats.brain_distillation_records = brain_stats.get("distillation_records", 0)
        stats.brain_blocked_side_effects = brain_stats.get("blocked_side_effects", 0)
        stats.brain_route_histogram = brain_stats.get("route_histogram", {})
        stats.brain_recent_examples = brain_stats.get("recent_route_examples", [])
        stats.brain_llm_leaks = brain_stats.get("llm_leaks", 0)
        stats.brain_tts_leaks = brain_stats.get("tts_leaks", 0)
        stats.brain_transcription_emit_leaks = brain_stats.get("transcription_emit_leaks", 0)
        stats.brain_memory_side_effects = brain_stats.get("memory_side_effects", 0)
        stats.brain_identity_side_effects = brain_stats.get("identity_side_effects", 0)
        logger.info("Brain stats retrieved: STT=%d routes=%d",
                     stats.brain_utterances_stt, stats.brain_routes_produced)
    else:
        logger.warning("Could not fetch brain-side stats (dashboard may not expose full ledger yet)")

    # --- Report ---
    print()
    print("=" * 70)
    print(stats.summary())
    print("=" * 70)

    report_path = write_report(stats, report_dir)
    if report_path:
        print(f"\nReport: {report_path}")


def main() -> None:
    from synthetic.exercise import PROFILES

    profile_names = ", ".join(PROFILES.keys())
    parser = argparse.ArgumentParser(
        description="Run synthetic perception exercise against the brain server",
    )
    parser.add_argument("--host", default="localhost", help="Brain server host")
    parser.add_argument("--port", type=int, default=9100, help="Perception WebSocket port")
    parser.add_argument("--count", "-n", type=int, default=None, help="Number of utterances (overrides profile)")
    parser.add_argument("--delay", "-d", type=float, default=None, help="Seconds between utterances (overrides profile)")
    parser.add_argument("--voice", default="af_bella", help="Kokoro TTS voice")
    parser.add_argument("--speed", type=float, default=1.0, help="TTS speech speed")
    parser.add_argument(
        "--category", "-c", default=None,
        help="Restrict to utterance category (command, question, status, etc.)",
    )
    parser.add_argument(
        "--profile", "-p", default=None,
        help=f"Named soak profile ({profile_names})",
    )
    parser.add_argument(
        "--duration", type=int, default=None,
        help="Run duration in seconds (for idle_soak, computes count from duration/delay)",
    )
    parser.add_argument(
        "--target-route", default=None,
        help="Target a specific ToolType route (TIME, SYSTEM_STATUS, MEMORY, etc.)",
    )
    parser.add_argument(
        "--report-dir", default=DEFAULT_REPORT_DIR,
        help=f"Report output directory (default: {DEFAULT_REPORT_DIR})",
    )
    args = parser.parse_args()

    # Resolve profile
    profile = None
    if args.profile:
        if args.profile not in PROFILES:
            parser.error(f"Unknown profile '{args.profile}'. Available: {profile_names}")
        profile = PROFILES[args.profile]

    # Resolve target-route to category
    category = args.category
    if args.target_route:
        category = ROUTE_TO_CATEGORY.get(args.target_route.upper())
        if not category:
            parser.error(
                f"Unknown target route '{args.target_route}'. "
                f"Available: {', '.join(ROUTE_TO_CATEGORY.keys())}"
            )

    # Determine final parameters
    if profile:
        count = args.count if args.count is not None else profile.effective_count(args.duration)
        delay_s = args.delay if args.delay is not None else profile.delay_s
        drain_s = profile.drain_s
        weights = profile.category_weights
        profile_name = profile.name
    else:
        count = args.count or 20
        delay_s = args.delay if args.delay is not None else 3.0
        drain_s = 8.0
        weights = None
        profile_name = ""

    if args.duration and not profile:
        count = max(1, int(args.duration / delay_s))

    asyncio.run(run_exercise(
        host=args.host,
        port=args.port,
        count=count,
        delay_s=delay_s,
        drain_s=drain_s,
        voice=args.voice,
        speed=args.speed,
        category=category,
        weights=weights,
        profile_name=profile_name,
        report_dir=args.report_dir,
    ))


if __name__ == "__main__":
    main()
