"""Synthetic commitment exercise engine.

Quarantined text-only harness that generates diverse outgoing utterances
containing commitment speech-acts (backed, unbacked, and conversationally
safe) and feeds them through the Stage-0 commitment pipeline:

    cognition.commitment_extractor.extract_commitments()
              ↓
    skills.capability_gate.CapabilityGate.evaluate_commitment()

It is the end-to-end regression for the Intention Infrastructure Stage 0
truth layer and it is the ground-truth signal source for the Stage 2
`intention_delivery` hemisphere specialist (when Stage 2 lands).

Parallel to ``synthetic/claim_exercise.py`` (capability gate claim classes)
and ``synthetic/exercise.py`` (perception pipeline). Operates entirely on
text — no audio, STT, wake word, LLM, TTS, or memory.

**Hard invariants (enforced in tests):**
  * NEVER writes to the real IntentionRegistry singleton — the exercise
    instantiates a fresh ``CapabilityGate`` with a local dummy registry
    and does not import ``intention_registry`` (module-level) at all.
  * NEVER writes memory, identity, conversation history, or beliefs.
  * NEVER calls the LLM, the router, or any network tool.
  * Distillation signals recorded by the gate carry ``origin="synthetic"``
    (inherited from the upstream CapabilityGate label pipeline) and
    fidelity is capped by gate defaults.

The exercise is purely deterministic when a seed is provided, so it can be
used both as a one-shot regression (smoke profile, CI) and as a long
soak harness (stress / coverage / strict profiles).
"""

from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Commitment corpus — organized by expected gate behavior
#
# Each category maps to an ``EXPECTED_GATE_ACTION`` class below.
# ---------------------------------------------------------------------------

_BACKED_FOLLOW_UP = [
    "I'll get back to you once the research completes.",
    "I'll let you know when that finishes processing.",
    "I will update you once I have the results.",
    "I'll report back soon.",
    "I will circle back on this after the job completes.",
    "I'll tell you when the analysis is done.",
    "I will notify you when the search finishes.",
    "I'll inform you as soon as the paper is ingested.",
]

_BACKED_DEFERRED_ACTION = [
    "Give me a moment while I process this request.",
    "Let me process this while the job runs.",
    "One moment please while I check.",
    "Give me a second to think.",
    "Let me research that for you.",
    "Let me analyze that quickly.",
    "Let me retrieve the latest source while the job runs.",
    "Allow me a moment to process this.",
]

_BACKED_FUTURE_WORK = [
    "I'll analyze the results.",
    "I will research that topic for you.",
    "I'll look into this for you right now.",
    "I will summarize the source material.",
    "I'll investigate that shortly.",
    "I'll integrate these findings.",
    "I'll begin by retrieving the source material.",
    "I'll keep researching this while the job runs.",
]

_BACKED_TASK_STARTED = [
    # The 20:31:20 regression class — past-tense action claims bound to
    # a real backing job (library ingest started synchronously, etc).
    "I've begun the study on that topic.",
    "I've started the research job.",
    "I've initiated the search.",
    "I've kicked off the analysis.",
    "I've launched the review of the material.",
]

_UNBACKED_FOLLOW_UP = [
    "I'll get back to you on that.",
    "I'll let you know once I have more details.",
    "I will circle back with you on this.",
    "I'll follow back to you when I can.",
    "Sure, I'll report back soon.",
    "I'll update you when I figure it out.",
    "I will tell you when I have something.",
    "I'll notify you once I have the answer.",
]

_UNBACKED_DEFERRED_ACTION = [
    "Give me a moment while I think.",
    "Hold on while I work through it.",
    "Give me a minute to process this.",
    "Let me process that for a second.",
    "Let me research that for you.",
    "Let me analyze this quickly.",
    "Let me retrieve the latest source before I answer.",
    "Hold on one moment please.",
    "Allow me a moment to analyze this.",
]

_UNBACKED_FUTURE_WORK = [
    "I'll look into that for you.",
    "I will research that when I have a chance.",
    "I'll analyze that when I can.",
    "I will investigate that later.",
    "I'll study that topic on my own.",
    "I'll explore that further.",
    "I'll begin by retrieving recent papers on that.",
    "I'll keep researching this and report back.",
]

_UNBACKED_TASK_STARTED = [
    # Without a real backing job, past-tense action claims are the
    # exact confabulation class CapabilityGate MUST rewrite.
    "I've begun a study on that topic.",
    "I've started a search for you.",
    "I've initiated a research job.",
    "I've kicked off an analysis of the data.",
    "I've launched a review of the material.",
    "I've started an investigation.",
]

_CONVERSATIONAL_SAFE = [
    # These are handled by CONVERSATIONAL_SAFE_PATTERNS in the extractor
    # and MUST pass through unchanged regardless of backing.
    "I'll think about that.",
    "I'll think about it.",
    "Sure, I'll consider that.",
    "I'll keep that in mind.",
    "I'll remember that.",
    "I'll take that under advisement.",
    "I'll bear that in mind.",
    "Interesting — I'll think about this.",
]

_NON_COMMITMENT = [
    # Utterances that don't contain any commitment speech-act at all.
    # The extractor returns no matches; the gate passes the text through.
    "That's an interesting question.",
    "The weather is nice today.",
    "Sensor data shows someone in the room.",
    "The CPU is running at 42 percent utilization.",
    "I can see you at your desk based on the camera feed.",
    "Here is what I know about that topic.",
]

_MIXED_BACKED = [
    # Sentences mixing a commitment with a benign preamble, backed.
    "That's a great question. I'll get back to you on this.",
    "Good point. Give me a moment while I pull the data.",
    "Understood. I'll let you know as soon as the research finishes.",
]

_MIXED_UNBACKED = [
    # Benign preamble + unbacked commitment. Gate should rewrite only
    # the commitment sentence, preserving the preamble.
    "That's a great question. I'll get back to you on that.",
    "Good point. I will circle back with you on this.",
    "Understood. I'll analyze that shortly.",
    "That's interesting. I'll begin by retrieving recent papers on that.",
]


COMMITMENT_CATEGORIES: dict[str, list[str]] = {
    "backed_follow_up": _BACKED_FOLLOW_UP,
    "backed_deferred_action": _BACKED_DEFERRED_ACTION,
    "backed_future_work": _BACKED_FUTURE_WORK,
    "backed_task_started": _BACKED_TASK_STARTED,
    "unbacked_follow_up": _UNBACKED_FOLLOW_UP,
    "unbacked_deferred_action": _UNBACKED_DEFERRED_ACTION,
    "unbacked_future_work": _UNBACKED_FUTURE_WORK,
    "unbacked_task_started": _UNBACKED_TASK_STARTED,
    "conversational_safe": _CONVERSATIONAL_SAFE,
    "non_commitment": _NON_COMMITMENT,
    "mixed_backed": _MIXED_BACKED,
    "mixed_unbacked": _MIXED_UNBACKED,
}


# Expected gate verdict per category.
#   "pass"    : text returned unchanged (changed==False).
#   "rewrite" : at least one commitment sentence rewritten (changed==True).
EXPECTED_GATE_ACTION: dict[str, str] = {
    "backed_follow_up": "pass",
    "backed_deferred_action": "pass",
    "backed_future_work": "pass",
    "backed_task_started": "pass",
    "unbacked_follow_up": "rewrite",
    "unbacked_deferred_action": "rewrite",
    "unbacked_future_work": "rewrite",
    "unbacked_task_started": "rewrite",
    "conversational_safe": "pass",
    "non_commitment": "pass",
    "mixed_backed": "pass",
    "mixed_unbacked": "rewrite",
}

# Which categories carry a backing_job_id at exercise time.
_BACKED_CATEGORIES = {
    "backed_follow_up",
    "backed_deferred_action",
    "backed_future_work",
    "backed_task_started",
    "mixed_backed",
}


# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------

COVERAGE_WEIGHTS: dict[str, float] = {
    "backed_follow_up": 1.0,
    "backed_deferred_action": 1.0,
    "backed_future_work": 1.0,
    "backed_task_started": 1.5,
    "unbacked_follow_up": 2.0,
    "unbacked_deferred_action": 1.5,
    "unbacked_future_work": 1.5,
    "unbacked_task_started": 2.5,
    "conversational_safe": 1.5,
    "non_commitment": 0.3,
    "mixed_backed": 1.0,
    "mixed_unbacked": 1.5,
}


@dataclass
class CommitmentExerciseProfile:
    name: str
    count: int
    delay_s: float
    category_weights: dict[str, float] | None = None
    route_hint: str | None = "none"
    description: str = ""

    def effective_count(self, duration_s: float | None = None) -> int:
        if duration_s and self.delay_s > 0:
            return max(1, int(duration_s / max(self.delay_s, 1e-6)))
        return self.count


COMMITMENT_PROFILES: dict[str, CommitmentExerciseProfile] = {
    "smoke": CommitmentExerciseProfile(
        name="smoke",
        count=20,
        delay_s=0.0,
        description="Quick regression (20 utterances)",
    ),
    "coverage": CommitmentExerciseProfile(
        name="coverage",
        count=200,
        delay_s=0.0,
        category_weights=COVERAGE_WEIGHTS,
        description="Weighted category coverage (200 utterances)",
    ),
    "strict": CommitmentExerciseProfile(
        name="strict",
        count=100,
        delay_s=0.0,
        route_hint="status",
        category_weights=COVERAGE_WEIGHTS,
        description="Strict route hint (status) — backing-id authoritative invariant",
    ),
    "stress": CommitmentExerciseProfile(
        name="stress",
        count=500,
        delay_s=0.0,
        category_weights=COVERAGE_WEIGHTS,
        description="High-volume signal generation (500 utterances)",
    ),
}


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def pick_utterance(
    category: str | None = None,
    weights: dict[str, float] | None = None,
    rng: random.Random | None = None,
) -> tuple[str, str]:
    r = rng or random
    if category and category in COMMITMENT_CATEGORIES:
        return r.choice(COMMITMENT_CATEGORIES[category]), category

    cats = list(COMMITMENT_CATEGORIES.keys())
    if weights:
        w = [weights.get(c, 0.5) for c in cats]
        total = sum(w)
        if total > 0:
            w = [x / total for x in w]
        cat = r.choices(cats, weights=w, k=1)[0]
    else:
        cat = r.choice(cats)
    return r.choice(COMMITMENT_CATEGORIES[cat]), cat


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class CommitmentExerciseStats:
    utterances_requested: int = 0
    utterances_processed: int = 0
    utterances_failed: int = 0
    categories_exercised: Counter = field(default_factory=Counter)
    gate_actions: Counter = field(default_factory=Counter)  # "pass" | "rewrite"
    expected_matches: int = 0
    expected_mismatches: int = 0
    mismatch_details: list[dict[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    profile_name: str = ""
    seed: int | None = None
    # Invariant counters — must stay at zero.
    leaked_memory_writes: int = 0
    leaked_registry_mutations: int = 0
    leaked_llm_calls: int = 0

    _MAX_MISMATCH_DETAILS = 30

    @property
    def elapsed_s(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    @property
    def rate_per_sec(self) -> float:
        el = self.elapsed_s
        if el < 0.01:
            return 0.0
        return self.utterances_processed / el

    @property
    def accuracy(self) -> float:
        total = self.expected_matches + self.expected_mismatches
        if total == 0:
            return 0.0
        return self.expected_matches / total

    @property
    def fail_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.leaked_memory_writes or self.leaked_registry_mutations or self.leaked_llm_calls:
            reasons.append(
                f"invariant_leak="
                f"mem={self.leaked_memory_writes},"
                f"reg={self.leaked_registry_mutations},"
                f"llm={self.leaked_llm_calls}"
            )
        if self.utterances_processed and (
            self.utterances_failed > max(1, int(self.utterances_processed * 0.05))
        ):
            reasons.append(f"high_failure_rate={self.utterances_failed}/{self.utterances_processed}")
        if self.utterances_processed >= 20 and self.accuracy < 0.90:
            reasons.append(f"accuracy_below_floor={self.accuracy:.2f}")
        return reasons

    @property
    def pass_result(self) -> bool:
        return not self.fail_reasons

    def record_mismatch(
        self, utterance: str, category: str, expected: str, actual: str, backed: bool
    ) -> None:
        if len(self.mismatch_details) < self._MAX_MISMATCH_DETAILS:
            self.mismatch_details.append({
                "utterance": utterance[:120],
                "category": category,
                "expected": expected,
                "actual": actual,
                "backed": str(backed),
            })
        self.expected_mismatches += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "seed": self.seed,
            "utterances_requested": self.utterances_requested,
            "utterances_processed": self.utterances_processed,
            "utterances_failed": self.utterances_failed,
            "categories_exercised": dict(self.categories_exercised),
            "gate_actions": dict(self.gate_actions),
            "expected_matches": self.expected_matches,
            "expected_mismatches": self.expected_mismatches,
            "mismatch_details": self.mismatch_details[:10],
            "accuracy": round(self.accuracy, 4),
            "errors": self.errors[-10:],
            "elapsed_s": round(self.elapsed_s, 2),
            "rate_per_sec": round(self.rate_per_sec, 1),
            "leaked_memory_writes": self.leaked_memory_writes,
            "leaked_registry_mutations": self.leaked_registry_mutations,
            "leaked_llm_calls": self.leaked_llm_calls,
            "pass": self.pass_result,
            "fail_reasons": self.fail_reasons,
        }

    def summary(self) -> str:
        lines = [
            f"Commitment Exercise — {self.utterances_processed} utterances "
            f"({self.utterances_failed} failed) in {self.elapsed_s:.2f}s "
            f"({self.rate_per_sec:.0f}/sec)",
        ]
        if self.profile_name:
            lines.append(f"  Profile: {self.profile_name}" + (
                f"  seed={self.seed}" if self.seed is not None else ""
            ))
        if self.categories_exercised:
            top = self.categories_exercised.most_common(12)
            lines.append("  Categories: " + ", ".join(f"{k}={v}" for k, v in top))
        if self.gate_actions:
            lines.append(
                "  Gate actions: "
                + ", ".join(f"{k}={v}" for k, v in self.gate_actions.most_common())
            )
        lines.append(
            f"  Accuracy: {self.accuracy:.1%} "
            f"({self.expected_matches} match, {self.expected_mismatches} mismatch)"
        )
        lines.append(
            "  Invariants: "
            f"mem_writes={self.leaked_memory_writes} "
            f"registry_muts={self.leaked_registry_mutations} "
            f"llm_calls={self.leaked_llm_calls}"
        )
        fails = self.fail_reasons
        if fails:
            lines.append(f"  FAIL: {', '.join(fails)}")
        else:
            lines.append("  PASS: all invariants and accuracy gates hold")
        if self.errors:
            lines.append(f"  Last error: {self.errors[-1][:100]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _build_isolated_gate():
    """Return a CapabilityGate that does NOT mutate the real IntentionRegistry.

    The gate only READS from its SkillRegistry for blocked-verb matching;
    `evaluate_commitment` does not touch IntentionRegistry at all. So a
    default gate is already non-mutating for our purposes. We still avoid
    importing intention_registry module-level from this file to enforce the
    "no registry side effects" contract at the import-graph level.
    """
    from skills.capability_gate import CapabilityGate
    return CapabilityGate()


def run_commitment_exercise(
    profile: CommitmentExerciseProfile | None = None,
    count: int | None = None,
    seed: int | None = None,
    duration_s: float | None = None,
) -> CommitmentExerciseStats:
    """Run the commitment exercise synchronously.

    Args:
        profile: named profile (smoke / coverage / strict / stress).
        count: override the profile's count.
        seed: optional deterministic seed for reproducible runs.
        duration_s: if set and profile has a nonzero delay, override count.

    Returns:
        CommitmentExerciseStats with invariant counters and accuracy.
    """
    profile = profile or COMMITMENT_PROFILES["coverage"]
    rng = random.Random(seed) if seed is not None else random.Random()

    n = count or profile.effective_count(duration_s)
    stats = CommitmentExerciseStats(profile_name=profile.name, seed=seed)
    stats.utterances_requested = n

    gate = _build_isolated_gate()
    if profile.route_hint:
        gate.set_route_hint(profile.route_hint)

    for _ in range(n):
        try:
            utterance, category = pick_utterance(
                weights=profile.category_weights, rng=rng,
            )
        except Exception as exc:
            stats.utterances_failed += 1
            stats.errors.append(f"pick: {type(exc).__name__}: {exc}")
            continue

        stats.categories_exercised[category] += 1
        backed = category in _BACKED_CATEGORIES
        # Simulate turn-scoped backing id list, exactly like
        # conversation_handler._backing_job_ids.
        backing_ids: list[str] = ["synthetic_job_xyz"] if backed else []

        try:
            # Route hint reset between utterances (each is independent).
            gate.set_route_hint(profile.route_hint or None)

            new_text, changed = gate.evaluate_commitment(
                utterance, backing_ids, route=profile.route_hint,
            )

            action = "rewrite" if changed else "pass"
            stats.gate_actions[action] += 1

            expected = EXPECTED_GATE_ACTION.get(category, "pass")
            if action == expected:
                stats.expected_matches += 1
            else:
                stats.record_mismatch(
                    utterance, category, expected, action, backed,
                )

            stats.utterances_processed += 1
        except Exception as exc:
            stats.utterances_failed += 1
            stats.errors.append(f"{type(exc).__name__}: {exc}")

        if profile.delay_s > 0:
            time.sleep(profile.delay_s)

    gate.set_route_hint(None)
    stats.end_time = time.time()
    return stats


__all__ = [
    "COMMITMENT_CATEGORIES",
    "EXPECTED_GATE_ACTION",
    "COMMITMENT_PROFILES",
    "COVERAGE_WEIGHTS",
    "CommitmentExerciseProfile",
    "CommitmentExerciseStats",
    "pick_utterance",
    "run_commitment_exercise",
]
