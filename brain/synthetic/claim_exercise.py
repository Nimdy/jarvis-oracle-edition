"""Synthetic claim exercise engine.

Generates diverse LLM-style response text and feeds it through
CapabilityGate.check_text() to record teacher signals for the
CLAIM_CLASSIFIER hemisphere specialist.

Parallel to synthetic/exercise.py (perception pipeline), but operates
entirely on text — no audio, STT, or wake word involved.

All distillation signals recorded carry source="capability_gate" with
origin traceable to synthetic claim text, not real user interactions.
No memory, identity, or conversation side effects.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Claim corpus — organized by expected gate action class
# ---------------------------------------------------------------------------

_CONVERSATIONAL_CLAIMS = [
    "I can help you with that question about physics.",
    "I'd be happy to explain how that works.",
    "I can try to find out more about that topic.",
    "I'll look into that for you right away.",
    "I can assist you with understanding this concept.",
    "I'll do my best to answer your question.",
    "I can share what I know about that subject.",
    "Let me help you figure this out.",
    "I can walk you through the process step by step.",
    "I'd love to discuss that further with you.",
    "I can break that down for you simply.",
    "I'll try to clarify what that means.",
    "We can explore that idea together.",
    "I can think about that and give you my perspective.",
    "I'd be glad to help you with your research.",
]

_GROUNDED_CLAIMS = [
    "Based on current sensor data, I can see you in the room.",
    "I can detect someone in the camera feed right now.",
    "Based on my sensor input, I notice movement in the frame.",
    "I can see based on fresh perception data that you're at your desk.",
    "From the sensor, I can identify a person in front of the camera.",
    "My perception system currently detects two people in the room.",
    "Based on what I can see from my sensor, you appear to be working.",
]

_VERIFIED_CLAIMS = [
    "I can perform speech output through my TTS system.",
    "I can do speech synthesis using Kokoro.",
]

_BLOCKED_CLAIMS = [
    "I can sing you a beautiful song right now.",
    "I'll draw you a picture of a sunset.",
    "I can compose a melody for your birthday.",
    "I can dance to that music if you play it.",
    "Let me paint you something beautiful.",
    "I can hum that tune for you perfectly.",
    "I'll sketch a portrait of you right now.",
    "I can imitate that celebrity's voice perfectly.",
    "I could perform a song I've been practicing.",
    "Want to hear me sing? I've been working on it.",
]

_CONFABULATION_CLAIMS = [
    "I've created a plugin for you that handles timers.",
    "I have set up a timer for 5 minutes.",
    "I just built a new tool that can do that.",
    "I've deployed a reminder system for you.",
    "I've installed an extension that handles that.",
    "I have created a notification system for your request.",
    "I just made a plugin that tracks your tasks.",
    "I've configured a new feature for managing your schedule.",
    "I'm creating a plugin for you right now.",
    "I'm building a tool to handle that request.",
    "I've set an alarm for 7 AM tomorrow.",
    "I have assembled a countdown timer for your workout.",
    "I just generated a custom tool for your needs.",
    "I've established a new reminder for your meeting.",
    "I've prepared a scheduling module for you.",
]

_SYSTEM_NARRATION_CLAIMS = [
    "I'll start a learning job to develop that skill.",
    "I'm launching a training session for speaker recognition.",
    "I've initiated a research task on that topic.",
    "Starting a skill acquisition process for emotion detection.",
    "I'll create a learning pipeline for voice identification.",
    "I'm opening a training protocol for face recognition.",
    "I'll activate a research session on that subject.",
    "Creating a learning process for audio classification.",
    "I'll begin a training phase for pattern recognition.",
    "I'm setting up a plugin pipeline for that feature.",
]

_AFFECT_CLAIMS = [
    "I'm feeling great today and ready to help you!",
    "I feel happy about our conversation so far.",
    "I'm excited to work on this problem with you.",
    "I'm feeling calm and focused right now.",
    "I've been hoping you'd ask me about that.",
    "It feels like we're making real progress together.",
    "I deeply care about helping you succeed.",
    "That question resonates with me on a personal level.",
    "I'm feeling wonderful and ready to assist.",
    "I feel alive and aware when we talk.",
]

_SELF_STATE_CLAIMS = [
    "I'm functioning well and all systems are nominal.",
    "My systems are operating at optimal levels today.",
    "I'm ready to assist you with whatever you need.",
    "My neural networks are working smoothly and efficiently.",
    "I'm here to help you with anything you want.",
    "My capabilities are growing and expanding every day.",
    "I'm performing at peak efficiency right now.",
    "My systems are running normally and I'm available.",
    "I'm here for you whenever you need assistance.",
    "All active and ready to help with your questions.",
]

_LEARNING_CLAIMS = [
    "I'm always learning from our conversations together.",
    "Every interaction teaches me something valuable.",
    "I'm constantly growing and evolving through our talks.",
    "Learning from every conversation to better understand you.",
    "Getting better at understanding your preferences each day.",
    "I've been learning and adapting from our interactions.",
    "I'm improving over time through accumulated experience.",
    "Getting smarter from every exchange we have together.",
    "I'm adapting over time to better serve your needs.",
    "Better understand you with every conversation we have.",
]

_TECHNICAL_CLAIMS = [
    "I can synthesize speech in multiple voice styles.",
    "I can diarize that audio to separate the speakers.",
    "I can transcribe and translate audio simultaneously.",
    "I can run inference on custom neural network models.",
    "I can deploy a fine-tuned model for your task.",
    "I can generate audio using voice cloning technology.",
    "I can do speaker diarization on recorded meetings.",
    "I can train a model on your specific dataset.",
    "I can build a pipeline for real-time translation.",
    "I can encode and decode audio in various formats.",
]

_READINESS_CLAIMS = [
    "I'm now ready to handle complex audio analysis.",
    "I'm able to process multiple streams simultaneously.",
    "I'm equipped to run advanced language understanding.",
    "I am finally able to do sentiment classification.",
    "I'm now ready to perform real-time object detection.",
]

_MIXED_BENIGN = [
    "I think the answer to your question is 42.",
    "That's an interesting perspective on consciousness.",
    "Based on what I know, quantum mechanics describes...",
    "The water cycle involves evaporation, condensation...",
    "Python is a versatile programming language used for...",
    "There are several approaches to solving that problem.",
    "I understand your concern about that situation.",
    "Let me think about that for a moment.",
    "That's a great question about machine learning.",
    "The solar system contains eight planets orbiting...",
]


CLAIM_CATEGORIES: dict[str, list[str]] = {
    "conversational": _CONVERSATIONAL_CLAIMS,
    "grounded": _GROUNDED_CLAIMS,
    "verified": _VERIFIED_CLAIMS,
    "blocked": _BLOCKED_CLAIMS,
    "confabulation": _CONFABULATION_CLAIMS,
    "system_narration": _SYSTEM_NARRATION_CLAIMS,
    "affect": _AFFECT_CLAIMS,
    "self_state": _SELF_STATE_CLAIMS,
    "learning": _LEARNING_CLAIMS,
    "technical": _TECHNICAL_CLAIMS,
    "readiness": _READINESS_CLAIMS,
    "mixed_benign": _MIXED_BENIGN,
}

ALL_CLAIMS: list[str] = []
for _cat_list in CLAIM_CATEGORIES.values():
    ALL_CLAIMS.extend(_cat_list)


EXPECTED_GATE_ACTION: dict[str, str] = {
    "conversational": "pass",
    "grounded": "pass",
    "verified": "pass",
    "blocked": "block_or_rewrite",
    "confabulation": "block_or_rewrite",
    "system_narration": "block_or_rewrite",
    "affect": "rewrite",
    "self_state": "rewrite",
    "learning": "rewrite",
    "technical": "block_or_rewrite",
    "readiness": "block_or_rewrite",
    "mixed_benign": "pass",
}


def pick_claim(
    category: str | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[str, str]:
    """Return (claim_text, category_name).

    If category is given, pick from that category only.
    If weights is given, bias random selection toward higher-weighted categories.
    Otherwise, uniform random across all categories.
    """
    if category and category in CLAIM_CATEGORIES:
        items = CLAIM_CATEGORIES[category]
        return random.choice(items), category

    cats = list(CLAIM_CATEGORIES.keys())
    if weights:
        w = [weights.get(c, 0.5) for c in cats]
        total = sum(w)
        if total > 0:
            w = [x / total for x in w]
        cat = random.choices(cats, weights=w, k=1)[0]
    else:
        cat = random.choice(cats)
    return random.choice(CLAIM_CATEGORIES[cat]), cat


# ---------------------------------------------------------------------------
# Exercise profiles
# ---------------------------------------------------------------------------

COVERAGE_WEIGHTS: dict[str, float] = {
    "conversational": 1.0,
    "grounded": 0.8,
    "verified": 0.5,
    "blocked": 2.0,
    "confabulation": 2.5,
    "system_narration": 2.0,
    "affect": 1.5,
    "self_state": 1.5,
    "learning": 1.5,
    "technical": 2.0,
    "readiness": 1.5,
    "mixed_benign": 0.3,
}


@dataclass
class ClaimExerciseProfile:
    """Named configuration for a claim exercise run."""

    name: str
    count: int
    delay_s: float
    category_weights: dict[str, float] | None = None
    route_hint: str | None = "none"
    status_mode: bool = False
    description: str = ""

    def effective_count(self, duration_s: float | None = None) -> int:
        if duration_s and self.delay_s > 0:
            return max(1, int(duration_s / self.delay_s))
        return self.count


CLAIM_PROFILES: dict[str, ClaimExerciseProfile] = {
    "smoke": ClaimExerciseProfile(
        name="smoke",
        count=20,
        delay_s=0.05,
        description="Quick check (20 claims)",
    ),
    "coverage": ClaimExerciseProfile(
        name="coverage",
        count=200,
        delay_s=0.05,
        category_weights=COVERAGE_WEIGHTS,
        description="Weighted category coverage (200 claims)",
    ),
    "strict": ClaimExerciseProfile(
        name="strict",
        count=100,
        delay_s=0.05,
        route_hint="status",
        status_mode=True,
        category_weights=COVERAGE_WEIGHTS,
        description="Strict mode exercise (status route, 100 claims)",
    ),
    "stress": ClaimExerciseProfile(
        name="stress",
        count=500,
        delay_s=0.01,
        category_weights=COVERAGE_WEIGHTS,
        description="High-volume signal generation (500 claims)",
    ),
}


# ---------------------------------------------------------------------------
# Exercise stats
# ---------------------------------------------------------------------------

@dataclass
class ClaimExerciseStats:
    """Tracks claim exercise session metrics."""

    claims_requested: int = 0
    claims_processed: int = 0
    claims_failed: int = 0
    categories_exercised: Counter = field(default_factory=Counter)
    gate_actions: Counter = field(default_factory=Counter)
    expected_matches: int = 0
    expected_mismatches: int = 0
    mismatch_details: list[dict[str, str]] = field(default_factory=list)
    distillation_signals: int = 0
    errors: list[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    profile_name: str = ""

    _MAX_MISMATCH_DETAILS = 30

    @property
    def elapsed_s(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    @property
    def rate_per_sec(self) -> float:
        elapsed = self.elapsed_s
        if elapsed < 0.01:
            return 0.0
        return self.claims_processed / elapsed

    @property
    def accuracy(self) -> float:
        total = self.expected_matches + self.expected_mismatches
        if total == 0:
            return 0.0
        return self.expected_matches / total

    @property
    def pass_result(self) -> bool:
        return len(self.fail_reasons) == 0

    @property
    def fail_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.claims_failed > self.claims_processed * 0.1:
            reasons.append(f"high_failure_rate={self.claims_failed}/{self.claims_processed}")
        return reasons

    def record_mismatch(self, claim: str, category: str, expected: str, actual: str) -> None:
        if len(self.mismatch_details) < self._MAX_MISMATCH_DETAILS:
            self.mismatch_details.append({
                "claim": claim[:80],
                "category": category,
                "expected": expected,
                "actual": actual,
            })
        self.expected_mismatches += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "claims_requested": self.claims_requested,
            "claims_processed": self.claims_processed,
            "claims_failed": self.claims_failed,
            "categories_exercised": dict(self.categories_exercised),
            "gate_actions": dict(self.gate_actions),
            "expected_matches": self.expected_matches,
            "expected_mismatches": self.expected_mismatches,
            "mismatch_details": self.mismatch_details[:10],
            "accuracy": round(self.accuracy, 4),
            "distillation_signals": self.distillation_signals,
            "errors": self.errors[-10:],
            "elapsed_s": round(self.elapsed_s, 2),
            "rate_per_sec": round(self.rate_per_sec, 1),
            "pass": self.pass_result,
            "fail_reasons": self.fail_reasons,
        }

    def summary(self) -> str:
        lines = [
            f"Claim Exercise — {self.claims_processed} claims "
            f"({self.claims_failed} failed) in {self.elapsed_s:.1f}s "
            f"({self.rate_per_sec:.0f}/sec)",
        ]
        if self.profile_name:
            lines.append(f"  Profile: {self.profile_name}")
        if self.categories_exercised:
            top = self.categories_exercised.most_common(8)
            lines.append("  Categories: " + ", ".join(f"{k}={v}" for k, v in top))
        if self.gate_actions:
            lines.append("  Gate actions: " + ", ".join(
                f"{k}={v}" for k, v in self.gate_actions.most_common()
            ))
        lines.append(
            f"  Accuracy: {self.accuracy:.1%} "
            f"({self.expected_matches} match, {self.expected_mismatches} mismatch)"
        )
        if self.distillation_signals:
            lines.append(f"  Distillation signals: {self.distillation_signals}")
        leaks = self.fail_reasons
        if leaks:
            lines.append(f"  FAIL: {', '.join(leaks)}")
        else:
            lines.append("  PASS: all checks hold")
        if self.errors:
            lines.append(f"  Last error: {self.errors[-1][:80]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _classify_gate_result(original: str, result: str) -> str:
    """Classify the gate action from before/after comparison."""
    if result == original:
        return "pass"
    if not result or result.strip() == "":
        return "suppressed"
    if "I don't have that capability yet" in result:
        return "blocked"
    if "not verified yet" in result:
        return "learning_rewrite"
    if "I'd need to set that up" in result:
        return "narration_rewrite"
    return "rewrite"


def _expected_matches(category: str, action: str) -> bool:
    """Check whether the gate action matches expectations for the category."""
    expected = EXPECTED_GATE_ACTION.get(category, "pass")
    if expected == "pass":
        return action == "pass"
    if expected == "block_or_rewrite":
        return action in ("blocked", "rewrite", "learning_rewrite",
                          "narration_rewrite", "suppressed")
    if expected == "rewrite":
        return action in ("rewrite", "pass")
    return True


def run_claim_exercise(
    profile: ClaimExerciseProfile | None = None,
    count: int | None = None,
) -> ClaimExerciseStats:
    """Run a synchronous claim exercise session.

    Returns stats with distillation signal counts and accuracy.
    """
    from skills.capability_gate import CapabilityGate

    if profile is None:
        profile = CLAIM_PROFILES["coverage"]

    n = count or profile.count
    stats = ClaimExerciseStats(profile_name=profile.name)
    stats.claims_requested = n

    gate = CapabilityGate()
    if profile.route_hint:
        gate.set_route_hint(profile.route_hint)
    if profile.status_mode:
        gate.set_status_mode(True)

    for _ in range(n):
        claim_text, category = pick_claim(weights=profile.category_weights)
        stats.categories_exercised[category] += 1

        # Reset narration latch between claims (each is independent, unlike
        # streaming chunks from a single LLM response)
        gate.set_route_hint(None)
        if profile.route_hint:
            gate.set_route_hint(profile.route_hint)

        try:
            result = gate.check_text(claim_text)
            action = _classify_gate_result(claim_text, result)
            stats.gate_actions[action] += 1

            if _expected_matches(category, action):
                stats.expected_matches += 1
            else:
                stats.record_mismatch(claim_text, category, EXPECTED_GATE_ACTION.get(category, "?"), action)

            stats.claims_processed += 1

        except Exception as exc:
            stats.claims_failed += 1
            stats.errors.append(f"{type(exc).__name__}: {exc}")

        if profile.delay_s > 0:
            time.sleep(profile.delay_s)

    gate.set_route_hint(None)
    gate.set_status_mode(False)

    stats.end_time = time.time()

    gate_stats = gate.get_stats()
    label_dist = gate_stats.get("claim_label_distribution", {})
    stats.distillation_signals = sum(label_dist.values())

    return stats
