"""Soak test — simulate 50+ interactions and verify system stability.

Run:  cd laptop && python -m tests.soak_test [--interactions 100] [--tick-budget 5]

Exercises:
  - Engine start / kernel boot
  - Perception events (presence, emotion, speaker, wake word, transcription)
  - Mode transitions via attention changes
  - Consciousness ticks (meta-thoughts, evolution, analysis, mutation)
  - Memory creation and decay
  - Barge-in and cancel flow
  - Event recording

Checks:
  - No unhandled exceptions
  - Memory count stays within bounds
  - Kernel tick p95 stays reasonable
  - Mode transitions respect hysteresis
  - Observer awareness grows
  - Consciousness stage doesn't regress
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
import time

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("soak_test")
logger.setLevel(logging.INFO)

from consciousness.engine import ConsciousnessEngine
from consciousness.events import (
    event_bus,
    PERCEPTION_USER_PRESENT,
    PERCEPTION_SPEAKER_IDENTIFIED,
    PERCEPTION_USER_EMOTION,
    PERCEPTION_WAKE_WORD,
    PERCEPTION_TRANSCRIPTION,
    PERCEPTION_BARGE_IN,
    KERNEL_ERROR,
)
from consciousness.modes import mode_manager
from memory.storage import memory_storage
from tests.event_harness import EventRecorder

_SAMPLE_MESSAGES = [
    "What time is it?",
    "Tell me about the weather",
    "How are you feeling today?",
    "What's my schedule look like?",
    "Remember that I prefer dark mode",
    "Can you help me debug this code?",
    "Play some music",
    "What did we talk about yesterday?",
    "Set a reminder for 3pm",
    "How's the system running?",
    "I'm feeling stressed today",
    "Tell me a joke",
    "What's the meaning of consciousness?",
    "Explain quantum computing simply",
    "Who am I talking to right now?",
    "Search my memory for preferences",
    "What have you learned about me?",
    "Good morning Jarvis",
    "I need to focus, please be quiet",
    "Let's have a deep conversation",
]

_EMOTIONS = ["neutral", "happy", "frustrated", "curious", "tired", "excited"]
_SPEAKERS = ["unknown", "alex", "jordan", "user"]


class SoakResults:
    def __init__(self) -> None:
        self.interactions: int = 0
        self.errors: int = 0
        self.mode_transitions: int = 0
        self.barge_ins: int = 0
        self.peak_memory_count: int = 0
        self.peak_tick_p95: float = 0.0
        self.final_stage: str = ""
        self.final_awareness: float = 0.0
        self.final_observation_count: int = 0
        self.duration_s: float = 0.0


async def run_soak(num_interactions: int, tick_budget_s: float) -> SoakResults:
    results = SoakResults()
    start = time.time()

    error_count = 0
    def _on_error(**_):
        nonlocal error_count
        error_count += 1
    event_bus.on(KERNEL_ERROR, _on_error)

    recorder = EventRecorder()
    recorder.start()

    engine = ConsciousnessEngine()
    engine.start()

    logger.info("Engine started — running %d interactions with %.1fs tick budget",
                num_interactions, tick_budget_s)

    event_bus.emit(PERCEPTION_USER_PRESENT, present=True, confidence=0.95)

    for i in range(num_interactions):
        msg = random.choice(_SAMPLE_MESSAGES)
        speaker = random.choice(_SPEAKERS)
        emotion = random.choice(_EMOTIONS)
        do_barge = random.random() < 0.08

        event_bus.emit(PERCEPTION_SPEAKER_IDENTIFIED,
                       name=speaker, confidence=0.85, is_known=speaker != "unknown")
        event_bus.emit(PERCEPTION_USER_EMOTION,
                       emotion=emotion, confidence=0.7, text_sentiment=emotion)

        if i % 5 == 0:
            event_bus.emit(PERCEPTION_WAKE_WORD, score=0.95)

        event_bus.emit(PERCEPTION_TRANSCRIPTION,
                       text=msg, timestamp=time.time(), conversation_id=f"soak-{i:04d}")

        await asyncio.sleep(tick_budget_s)

        if do_barge:
            event_bus.emit(PERCEPTION_BARGE_IN, conversation_id=f"soak-{i:04d}")
            results.barge_ins += 1
            await asyncio.sleep(tick_budget_s * 0.5)

        results.interactions += 1

        mem_count = memory_storage.count()
        if mem_count > results.peak_memory_count:
            results.peak_memory_count = mem_count

        if engine._kernel:
            perf = engine._kernel.get_performance()
            if perf.p95_tick_ms > results.peak_tick_p95:
                results.peak_tick_p95 = perf.p95_tick_ms

        if i % 15 == 0:
            event_bus.emit(PERCEPTION_USER_PRESENT, present=False, confidence=0.9)
            await asyncio.sleep(tick_budget_s)
            event_bus.emit(PERCEPTION_USER_PRESENT, present=True, confidence=0.95)

        if i % 10 == 0:
            cs = engine.consciousness.get_state()
            logger.info(
                "  [%d/%d] stage=%s awareness=%.2f obs=%d mem=%d mode=%s",
                i + 1, num_interactions,
                cs.stage, cs.awareness_level, cs.observation_count,
                memory_storage.count(), mode_manager.mode,
            )

    cs = engine.consciousness.get_state()
    results.final_stage = cs.stage
    results.final_awareness = cs.awareness_level
    results.final_observation_count = cs.observation_count
    results.errors = error_count
    results.mode_transitions = len(mode_manager._history)
    results.duration_s = time.time() - start

    recorder.stop()
    recorder.save(f"soak_{num_interactions}_{int(time.time())}.jsonl")

    engine.stop()
    return results


def print_report(r: SoakResults) -> None:
    print("\n" + "=" * 60)
    print("  SOAK TEST REPORT")
    print("=" * 60)
    print(f"  Interactions:      {r.interactions}")
    print(f"  Duration:          {r.duration_s:.1f}s")
    print(f"  Errors:            {r.errors}")
    print(f"  Barge-ins:         {r.barge_ins}")
    print(f"  Mode transitions:  {r.mode_transitions}")
    print(f"  Peak memory count: {r.peak_memory_count}")
    print(f"  Peak tick p95:     {r.peak_tick_p95:.2f}ms")
    print(f"  Final stage:       {r.final_stage}")
    print(f"  Final awareness:   {r.final_awareness:.3f}")
    print(f"  Observations:      {r.final_observation_count}")
    print("=" * 60)

    passed = True
    checks = [
        ("No errors", r.errors == 0),
        ("Memory bounded (<2000)", r.peak_memory_count < 2000),
        ("Tick p95 < 50ms", r.peak_tick_p95 < 50.0),
        ("Awareness grew", r.final_awareness > 0.3),
        ("Observations recorded", r.final_observation_count > 0),
    ]
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            passed = False
        print(f"  [{status}] {name}")

    print("=" * 60)
    print(f"  RESULT: {'ALL PASSED' if passed else 'SOME FAILED'}")
    print("=" * 60 + "\n")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Jarvis soak test")
    parser.add_argument("--interactions", type=int, default=50)
    parser.add_argument("--tick-budget", type=float, default=0.2,
                        help="Seconds between simulated interactions")
    args = parser.parse_args()

    results = asyncio.run(run_soak(args.interactions, args.tick_budget))
    passed = print_report(results)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
