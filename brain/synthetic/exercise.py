"""Synthetic perception exercise engine.

Generates diverse utterances and feeds them as TTS-synthesized PCM audio
through the brain's real perception pipeline (wake word -> VAD -> STT ->
speaker ID -> emotion -> tool router) with a hard stop before conversation
handler, memory, identity, or LLM.

All distillation signals from this path carry origin="synthetic" provenance.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Utterance corpus — no personal/autobiographical content
# ---------------------------------------------------------------------------

_TIME_UTTERANCES = [
    "hey jarvis please tell me what time is it right now",
    "hey jarvis go ahead and tell me what is today's date",
    "hey jarvis please tell me what day is it today",
    "hey jarvis go ahead and tell me what's the date today",
    "hey jarvis please tell me do you know what time it is",
]

_COMMAND_UTTERANCES = [
    "hey jarvis please check the system status for me right now",
    "hey jarvis go ahead and check how's the system doing right now",
    "hey jarvis please tell me what's the cpu usage right now",
    "hey jarvis go ahead and run a system check for me please",
    "hey jarvis please check what is the system uptime right now",
    "hey jarvis go ahead and show me the memory usage right now",
    "hey jarvis please check what's the gpu temperature right now",
    "hey jarvis go ahead and show me the ram usage right now",
]

_STATUS_UTTERANCES = [
    "hey jarvis please tell me how are you doing right now",
    "hey jarvis go ahead and tell me what's your current status",
    "hey jarvis please tell me how are you feeling right now",
    "hey jarvis go ahead and give me a status report please",
    "hey jarvis please tell me how's everything running right now",
    "hey jarvis go ahead and tell me are you healthy right now",
]

_INTROSPECTION_UTTERANCES = [
    "hey jarvis please tell me what have you learned recently",
    "hey jarvis go ahead and tell me what are you thinking about",
    "hey jarvis please tell me what's on your mind right now",
    "hey jarvis go ahead and tell me about your capabilities please",
    "hey jarvis please tell me what are you good at right now",
    "hey jarvis go ahead and tell me what do you know about yourself",
    "hey jarvis please describe your current state for me",
    "hey jarvis go ahead and tell me what is your consciousness level",
    "hey jarvis please tell me about your confidence level right now",
    "hey jarvis go ahead and tell me what capabilities do you have",
]

_MEMORY_UTTERANCES = [
    "hey jarvis please tell me do you remember anything about science",
    "hey jarvis go ahead and tell me what do you remember about our conversations",
    "hey jarvis please recall something interesting for me right now",
    "hey jarvis go ahead and recall a recent interaction we had",
    "hey jarvis please tell me what did we talk about last time",
    "hey jarvis go ahead and tell me do you recall what happened earlier",
    "hey jarvis please tell me we discussed some topics recently remember",
    "hey jarvis go ahead and tell me you told me about something before",
    "hey jarvis please tell me what was the first thing you remember",
    "hey jarvis go ahead and tell me a memory you have right now",
]

_WEB_SEARCH_UTTERANCES = [
    "hey jarvis please search the web for latest AI research papers",
    "hey jarvis go ahead and search for raspberry pi projects online",
    "hey jarvis please search for black hole basics on the web",
    "hey jarvis go ahead and look up python best practices online",
    "hey jarvis please look up the latest on fusion energy research",
    "hey jarvis go ahead and search online for quantum computing basics",
    "hey jarvis please find online information about battery technology",
    "hey jarvis go ahead and search for cooking recipe tutorials online",
    "hey jarvis please look up the latest on climate change data",
    "hey jarvis go ahead and search the web for robotics projects",
]

_ACADEMIC_UTTERANCES = [
    "hey jarvis please find papers on transformer architectures for me",
    "hey jarvis go ahead and find papers on reinforcement learning methods",
    "hey jarvis please look up studies on consciousness and awareness",
    "hey jarvis go ahead and find studies about language models in journals",
    "hey jarvis please tell me what does science say about memory systems",
    "hey jarvis go ahead and find papers on cognitive architectures please",
    "hey jarvis please look up academic papers on attention mechanisms",
    "hey jarvis go ahead and find research on protein folding techniques",
    "hey jarvis please search for scholarly work on word embeddings",
    "hey jarvis go ahead and tell me what does the scientific literature say about perception",
]

_SKILL_UTTERANCES = [
    "hey jarvis go ahead and start learning to recognize different speakers",
    "hey jarvis go ahead and train yourself on emotion detection please",
    "hey jarvis go ahead and start learning to recognize music genres",
    "hey jarvis go ahead and start a learning job for voice identification",
    "hey jarvis please teach yourself how to detect emotions accurately",
    "hey jarvis go ahead and create a skill for audio recognition",
    "hey jarvis go ahead and start training on speaker diarization now",
    "hey jarvis go ahead and learn how to identify different voices",
    "hey jarvis please start learning face recognition right now",
    "hey jarvis go ahead and train on pattern recognition for me",
]

_CODEBASE_UTTERANCES = [
    "hey jarvis please show me the code for perception processing",
    "hey jarvis go ahead and tell me where is the function for tool routing",
    "hey jarvis please tell me which file handles the memory operations",
    "hey jarvis go ahead and analyze your codebase for me please",
    "hey jarvis please look up your code for the event bus system",
    "hey jarvis go ahead and inspect your code for the kernel loop",
    "hey jarvis please tell me where is the class for the observer",
    "hey jarvis go ahead and read your code for the policy layer",
    "hey jarvis please tell me which module handles the audio stream",
    "hey jarvis go ahead and analyze your code for the attention system",
]

_IDENTITY_UTTERANCES = [
    "hey jarvis please tell me who am i right now",
    "hey jarvis go ahead and tell me do you know who i am",
    "hey jarvis please tell me do you recognize me at all",
    "hey jarvis go ahead and tell me who is speaking right now",
    "hey jarvis please tell me do you know me or not",
    "hey jarvis go ahead and tell me who do you think i am",
    "hey jarvis please tell me do you recognize me right now",
    "hey jarvis go ahead and tell me is this the same person as before",
]

_VISION_UTTERANCES = [
    "hey jarvis please tell me what do you see right now",
    "hey jarvis go ahead and tell me can you see anything interesting",
    "hey jarvis please look around and tell me what you see",
    "hey jarvis go ahead and tell me what do you see in front of you",
    "hey jarvis please describe what you see around you right now",
    "hey jarvis go ahead and look at the room and describe it",
    "hey jarvis go ahead and show me what you see right now",
    "hey jarvis please tell me what do you see around you right now",
]

_LIBRARY_UTTERANCES = [
    "hey jarvis please study this textbook on information theory for me",
    "hey jarvis go ahead and study this textbook chapter on biology",
    "hey jarvis please ingest this textbook about physics and chemistry",
    "hey jarvis go ahead and read this textbook on cognitive science",
    "hey jarvis please learn from this book about artificial intelligence",
    "hey jarvis go ahead and study this book on statistics for me",
]

_EMOTIONAL_UTTERANCES = [
    "hey jarvis that's amazing",
    "hey jarvis i'm really frustrated with this",
    "hey jarvis this is so exciting",
    "hey jarvis i'm not sure about that",
    "hey jarvis that makes me happy",
    "hey jarvis i'm confused about something",
    "hey jarvis wow that's incredible",
    "hey jarvis that's disappointing",
]

_QUESTION_UTTERANCES = [
    "hey jarvis how does photosynthesis work",
    "hey jarvis explain gravity to me",
    "hey jarvis what is machine learning",
    "hey jarvis tell me about the solar system",
    "hey jarvis what causes earthquakes",
    "hey jarvis how do computers process data",
    "hey jarvis what is quantum entanglement",
    "hey jarvis explain the water cycle",
    "hey jarvis how does electricity work",
    "hey jarvis what are black holes",
]

_GENERAL_UTTERANCES = [
    "hey jarvis tell me a fun fact",
    "hey jarvis give me some advice",
    "hey jarvis that's a good point",
    "hey jarvis let's talk about something interesting",
]

_NOISY_UTTERANCES = [
    "hey jarvis uh what was i saying oh right system status",
    "hey jarvis hmm can you check something for me",
    "jarvis hey jarvis tell me the time",
    "hey jarvis wait no actually search for papers on attention",
    "hey jarvis um what do you think about",
]

UTTERANCE_CATEGORIES: dict[str, list[str]] = {
    "time": _TIME_UTTERANCES,
    "command": _COMMAND_UTTERANCES,
    "status": _STATUS_UTTERANCES,
    "introspection": _INTROSPECTION_UTTERANCES,
    "memory": _MEMORY_UTTERANCES,
    "web_search": _WEB_SEARCH_UTTERANCES,
    "academic": _ACADEMIC_UTTERANCES,
    "skill": _SKILL_UTTERANCES,
    "codebase": _CODEBASE_UTTERANCES,
    "identity": _IDENTITY_UTTERANCES,
    "vision": _VISION_UTTERANCES,
    "library": _LIBRARY_UTTERANCES,
    "emotional": _EMOTIONAL_UTTERANCES,
    "question": _QUESTION_UTTERANCES,
    "general": _GENERAL_UTTERANCES,
    "noisy": _NOISY_UTTERANCES,
}

ALL_UTTERANCES: list[str] = []
for _cat_list in UTTERANCE_CATEGORIES.values():
    ALL_UTTERANCES.extend(_cat_list)


def pick_utterance(
    category: str | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[str, str]:
    """Return (utterance_text, category_name).

    If category is given, pick from that category only.
    If weights is given, bias random selection toward higher-weighted categories.
    Otherwise, uniform random across all categories.
    """
    if category and category in UTTERANCE_CATEGORIES:
        items = UTTERANCE_CATEGORIES[category]
        return random.choice(items), category

    cats = list(UTTERANCE_CATEGORIES.keys())
    if weights:
        w = [weights.get(c, 0.5) for c in cats]
        total = sum(w)
        if total > 0:
            w = [x / total for x in w]
        cat = random.choices(cats, weights=w, k=1)[0]
    else:
        cat = random.choice(cats)
    return random.choice(UTTERANCE_CATEGORIES[cat]), cat


# ---------------------------------------------------------------------------
# Soak profiles
# ---------------------------------------------------------------------------

@dataclass
class SoakProfile:
    """Named configuration for a synthetic exercise run."""

    name: str
    count: int
    delay_s: float
    drain_s: float = 8.0
    category_weights: dict[str, float] | None = None
    description: str = ""

    def effective_count(self, duration_s: float | None = None) -> int:
        """Compute utterance count, optionally from duration."""
        if duration_s and self.delay_s > 0:
            return max(1, int(duration_s / self.delay_s))
        return self.count


ROUTE_COVERAGE_WEIGHTS: dict[str, float] = {
    "time": 0.8,
    "command": 1.2,
    "status": 1.0,
    "introspection": 1.2,
    "memory": 1.8,
    "web_search": 1.5,
    "academic": 1.8,
    "skill": 2.0,
    "codebase": 1.8,
    "identity": 1.5,
    "vision": 1.5,
    "library": 1.2,
    "emotional": 0.15,
    "question": 0.15,
    "general": 0.15,
    "noisy": 0.15,
}

PROFILES: dict[str, SoakProfile] = {
    "smoke": SoakProfile(
        name="smoke",
        count=5,
        delay_s=2.0,
        description="Quick verification (5 utterances)",
    ),
    "route_coverage": SoakProfile(
        name="route_coverage",
        count=100,
        delay_s=5.0,
        category_weights=ROUTE_COVERAGE_WEIGHTS,
        description="Balanced route coverage (100 utterances, weighted toward under-covered routes)",
    ),
    "idle_soak": SoakProfile(
        name="idle_soak",
        count=120,
        delay_s=30.0,
        description="Low-rate trickle for long runs (use --duration to override count)",
    ),
    "stress": SoakProfile(
        name="stress",
        count=50,
        delay_s=0.5,
        drain_s=12.0,
        description="High-rate pressure test (50 utterances, 0.5s delay)",
    ),
}


# ---------------------------------------------------------------------------
# Exercise stats
# ---------------------------------------------------------------------------

@dataclass
class ExerciseStats:
    """Tracks synthetic exercise session metrics."""

    utterances_requested: int = 0
    utterances_sent: int = 0
    utterances_failed: int = 0
    reconnect_count: int = 0
    reconnect_failures: int = 0
    categories_exercised: Counter = field(default_factory=Counter)
    routes_exercised: Counter = field(default_factory=Counter)
    stt_text_samples: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    profile_name: str = ""

    # Brain-confirmed counters (populated from orchestrator ledger at end)
    brain_utterances_stt: int = 0
    brain_hard_stopped: int = 0
    brain_routes_produced: int = 0
    brain_distillation_records: int = 0
    brain_blocked_side_effects: int = 0
    brain_route_histogram: dict[str, int] = field(default_factory=dict)
    brain_recent_examples: list[dict[str, str]] = field(default_factory=list)

    # Invariant leak counters (must be 0)
    brain_llm_leaks: int = 0
    brain_tts_leaks: int = 0
    brain_transcription_emit_leaks: int = 0
    brain_memory_side_effects: int = 0
    brain_identity_side_effects: int = 0

    _MAX_STT_SAMPLES = 20

    @property
    def utterances_processed(self) -> int:
        return self.utterances_sent

    @property
    def elapsed_s(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    @property
    def rate_per_min(self) -> float:
        elapsed = self.elapsed_s
        if elapsed < 1.0:
            return 0.0
        return self.utterances_sent / (elapsed / 60.0)

    @property
    def recovered_disconnects(self) -> int:
        return self.reconnect_count

    @property
    def transport_stable(self) -> bool:
        return self.reconnect_count == 0 and self.reconnect_failures == 0

    def record_stt_sample(self, text: str) -> None:
        if len(self.stt_text_samples) < self._MAX_STT_SAMPLES:
            self.stt_text_samples.append(text)

    @property
    def pass_result(self) -> bool:
        return len(self.fail_reasons) == 0

    @property
    def fail_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.brain_llm_leaks > 0:
            reasons.append(f"llm_leaks={self.brain_llm_leaks}")
        if self.brain_tts_leaks > 0:
            reasons.append(f"tts_leaks={self.brain_tts_leaks}")
        if self.brain_transcription_emit_leaks > 0:
            reasons.append(f"transcription_emit_leaks={self.brain_transcription_emit_leaks}")
        if self.brain_memory_side_effects > 0:
            reasons.append(f"memory_side_effects={self.brain_memory_side_effects}")
        if self.brain_identity_side_effects > 0:
            reasons.append(f"identity_side_effects={self.brain_identity_side_effects}")
        if self.brain_hard_stopped > 0 and self.brain_hard_stopped != self.brain_utterances_stt:
            reasons.append(
                f"hard_stop_mismatch: hard_stopped={self.brain_hard_stopped} "
                f"stt_ok={self.brain_utterances_stt}"
            )
        return reasons

    def consistency_check(self) -> list[str]:
        """Verify: hard_stopped <= stt_ok <= sent."""
        issues: list[str] = []
        if self.brain_utterances_stt > self.utterances_sent:
            issues.append(
                f"stt_ok ({self.brain_utterances_stt}) > sent ({self.utterances_sent})"
            )
        if self.brain_hard_stopped > self.brain_utterances_stt:
            issues.append(
                f"hard_stopped ({self.brain_hard_stopped}) > stt_ok ({self.brain_utterances_stt})"
            )
        return issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "utterances_requested": self.utterances_requested,
            "utterances_sent": self.utterances_sent,
            "utterances_failed": self.utterances_failed,
            "reconnect_count": self.reconnect_count,
            "reconnect_failures": self.reconnect_failures,
            "recovered_disconnects": self.recovered_disconnects,
            "transport_stable": self.transport_stable,
            "categories_exercised": dict(self.categories_exercised),
            "routes_exercised": dict(self.routes_exercised),
            "stt_text_samples": self.stt_text_samples,
            "errors": self.errors[-10:],
            "elapsed_s": round(self.elapsed_s, 1),
            "rate_per_min": round(self.rate_per_min, 2),
            "brain_utterances_stt": self.brain_utterances_stt,
            "brain_hard_stopped": self.brain_hard_stopped,
            "brain_routes_produced": self.brain_routes_produced,
            "brain_distillation_records": self.brain_distillation_records,
            "brain_blocked_side_effects": self.brain_blocked_side_effects,
            "brain_route_histogram": self.brain_route_histogram,
            "brain_recent_examples": self.brain_recent_examples,
            "invariants": {
                "llm_leaks": self.brain_llm_leaks,
                "tts_leaks": self.brain_tts_leaks,
                "transcription_emit_leaks": self.brain_transcription_emit_leaks,
                "memory_side_effects": self.brain_memory_side_effects,
                "identity_side_effects": self.brain_identity_side_effects,
            },
            "consistency_check": self.consistency_check(),
            "pass": self.pass_result,
            "fail_reasons": self.fail_reasons,
        }

    def summary(self) -> str:
        lines = [
            f"Synthetic Exercise — {self.utterances_sent} utterances "
            f"({self.utterances_failed} failed) in {self.elapsed_s:.0f}s "
            f"({self.rate_per_min:.1f}/min)",
        ]
        if self.profile_name:
            lines.append(f"  Profile: {self.profile_name}")
        if self.reconnect_count or self.reconnect_failures:
            lines.append(
                f"  Transport: {self.reconnect_count} recovered, "
                f"{self.reconnect_failures} failed reconnects "
                f"({'stable' if self.transport_stable else 'self-healed'})"
            )
        if self.categories_exercised:
            top = self.categories_exercised.most_common(8)
            lines.append("  Categories: " + ", ".join(f"{k}={v}" for k, v in top))
        if self.brain_utterances_stt:
            lines.append(
                f"  Brain confirmed: STT={self.brain_utterances_stt} "
                f"HARD_STOP={self.brain_hard_stopped} "
                f"routes={self.brain_routes_produced} "
                f"distillation={self.brain_distillation_records}"
            )
        if self.brain_route_histogram:
            top_routes = sorted(
                self.brain_route_histogram.items(), key=lambda x: -x[1],
            )[:8]
            lines.append(
                "  Route histogram: "
                + ", ".join(f"{k}={v}" for k, v in top_routes)
            )
        leaks = self.fail_reasons
        if leaks:
            lines.append(f"  FAIL: {', '.join(leaks)}")
        else:
            lines.append("  PASS: all invariants hold")
        consistency = self.consistency_check()
        if consistency:
            lines.append(f"  Consistency warnings: {'; '.join(consistency)}")
        if self.errors:
            lines.append(f"  Last error: {self.errors[-1][:80]}")
        return "\n".join(lines)
