"""Synthetic belief contradiction exercise.

Exercises the epistemic claim extractor and conflict classifier on fabricated
memory/belief pairs with known conflict relationships across all 6 conflict
classes plus near-miss scenarios.

Truth boundary:
  - claim_extractor.extract_claims() is stateless and pure — safe to call
  - ConflictClassifier.classify() is stateless per call — safe to call
  - Standalone BeliefStore(path=tempdir) writes to temp files only
  - NEVER uses ContradictionEngine.get_instance() (subscribes to MEMORY_WRITE)
  - NEVER emits CONTRADICTION_DETECTED or MEMORY_WRITE events
  - NEVER calls _update_debt() (affects quarantine pressure system-wide)
  - Stats only — no distillation signals (no contradiction specialist NN yet)
"""

from __future__ import annotations

import logging
import random
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanoid import generate as nanoid

from consciousness.events import Memory
from epistemic.belief_record import BeliefRecord, BeliefStore, ConflictClassification
from epistemic.claim_extractor import extract_claims
from epistemic.conflict_classifier import ConflictClassifier

logger = logging.getLogger(__name__)

REPORT_DIR = Path.home() / ".jarvis" / "synthetic_exercise"


# ---------------------------------------------------------------------------
# Synthetic belief pair corpus
# ---------------------------------------------------------------------------

def _make_memory(payload: str, mem_type: str = "factual_knowledge",
                 provenance: str = "user_claim",
                 tags: tuple[str, ...] = ()) -> Memory:
    """Create a synthetic Memory for claim extraction."""
    return Memory(
        id=nanoid(),
        timestamp=time.time(),
        weight=0.5,
        tags=tags or ("synthetic",),
        payload=payload,
        type=mem_type,
        provenance=provenance,
    )


def _make_belief(
    subject: str, predicate: str, obj: str,
    modality: str = "is", stance: str = "assert",
    claim_type: str = "factual",
    provenance: str = "user_claim",
    confidence: float = 0.8,
) -> BeliefRecord:
    """Create a synthetic BeliefRecord directly."""
    bid = nanoid()
    return BeliefRecord(
        belief_id=bid,
        canonical_subject=subject,
        canonical_predicate=predicate,
        canonical_object=obj,
        modality=modality,
        stance=stance,
        polarity=1 if stance == "assert" else -1,
        claim_type=claim_type,
        epistemic_status="observed",
        extraction_confidence=confidence,
        belief_confidence=confidence,
        provenance=provenance,
        scope="global",
        source_memory_id=nanoid(),
        timestamp=time.time(),
        time_range=None,
        is_state_belief=False,
        conflict_key=f"{claim_type}::{subject}::{obj}",
        evidence_refs=[],
        contradicts=[],
        resolution_state="active",
        rendered_claim=f"{subject} {predicate} {obj}",
    )


def _build_conflict_pairs() -> list[dict[str, Any]]:
    """Build corpus of belief pairs with known expected conflict types."""
    pairs: list[dict[str, Any]] = []

    # --- Factual contradictions ---
    pairs.append({
        "name": "factual_temperature",
        "expected_type": "factual",
        "a": _make_belief("earth", "has_temperature", "warming",
                          claim_type="factual"),
        "b": _make_belief("earth", "has_temperature", "cooling",
                          claim_type="factual"),
    })
    pairs.append({
        "name": "factual_color",
        "expected_type": "factual",
        "a": _make_belief("sky", "is_color", "blue", claim_type="factual"),
        "b": _make_belief("sky", "is_color", "green", claim_type="factual"),
    })
    pairs.append({
        "name": "factual_capital",
        "expected_type": "factual",
        "a": _make_belief("france", "has_capital", "paris", claim_type="factual"),
        "b": _make_belief("france", "has_capital", "lyon", claim_type="factual"),
    })
    pairs.append({
        "name": "factual_count",
        "expected_type": "factual",
        "a": _make_belief("solar_system", "has_planets", "8", claim_type="factual"),
        "b": _make_belief("solar_system", "has_planets", "9", claim_type="factual"),
    })
    pairs.append({
        "name": "factual_location",
        "expected_type": "factual",
        "a": _make_belief("user", "lives_in", "new_york", claim_type="factual"),
        "b": _make_belief("user", "lives_in", "san_francisco", claim_type="factual"),
    })

    # --- Temporal conflicts ---
    pairs.append({
        "name": "temporal_job",
        "expected_type": "temporal",
        "a": _make_belief("user", "works_at", "google",
                          claim_type="factual", provenance="user_claim"),
        "b": _make_belief("user", "works_at", "meta",
                          claim_type="factual", provenance="user_claim"),
    })
    pairs.append({
        "name": "temporal_hobby",
        "expected_type": "temporal",
        "a": _make_belief("user", "favorite_hobby", "painting",
                          claim_type="preference"),
        "b": _make_belief("user", "favorite_hobby", "hiking",
                          claim_type="preference"),
    })
    pairs.append({
        "name": "temporal_status",
        "expected_type": "temporal",
        "a": _make_belief("user", "relationship_status", "single",
                          claim_type="factual", provenance="user_claim"),
        "b": _make_belief("user", "relationship_status", "married",
                          claim_type="factual", provenance="user_claim"),
    })

    # --- Identity tensions ---
    pairs.append({
        "name": "identity_belief",
        "expected_type": "identity_tension",
        "a": _make_belief("self", "is_capable_of", "creativity",
                          claim_type="identity"),
        "b": _make_belief("self", "is_not_capable_of", "creativity",
                          claim_type="identity", stance="deny"),
    })
    pairs.append({
        "name": "identity_value",
        "expected_type": "identity_tension",
        "a": _make_belief("self", "values", "honesty",
                          claim_type="identity"),
        "b": _make_belief("self", "struggles_with", "honesty",
                          claim_type="identity"),
    })

    # --- Policy norm conflicts ---
    pairs.append({
        "name": "policy_norm_action",
        "expected_type": "policy_norm",
        "a": _make_belief("system", "should_always", "respond_quickly",
                          modality="should", claim_type="policy"),
        "b": _make_belief("system", "should_never", "respond_quickly",
                          modality="should", claim_type="policy", stance="deny"),
    })
    pairs.append({
        "name": "policy_norm_proactive",
        "expected_type": "policy_norm",
        "a": _make_belief("system", "should", "be_proactive",
                          modality="should", claim_type="policy"),
        "b": _make_belief("system", "should_not", "be_proactive",
                          modality="should", claim_type="policy", stance="deny"),
    })

    # --- Provenance conflicts ---
    pairs.append({
        "name": "provenance_source",
        "expected_type": "provenance",
        "a": _make_belief("python", "is_version", "3.12",
                          claim_type="factual", provenance="external_source"),
        "b": _make_belief("python", "is_version", "3.11",
                          claim_type="factual", provenance="user_claim"),
    })
    pairs.append({
        "name": "provenance_observation",
        "expected_type": "provenance",
        "a": _make_belief("user", "preferred_language", "english",
                          claim_type="factual", provenance="observed"),
        "b": _make_belief("user", "preferred_language", "spanish",
                          claim_type="factual", provenance="model_inference"),
    })

    # --- Multi-perspective (philosophical) ---
    pairs.append({
        "name": "philosophical_view",
        "expected_type": "multi_perspective",
        "a": _make_belief("consciousness", "is", "emergent_property",
                          claim_type="philosophical"),
        "b": _make_belief("consciousness", "is", "fundamental_force",
                          claim_type="philosophical"),
    })

    return pairs


def _build_near_miss_pairs() -> list[dict[str, Any]]:
    """Build pairs that should NOT trigger conflict classification."""
    pairs: list[dict[str, Any]] = []

    pairs.append({
        "name": "different_subject",
        "expected_type": None,
        "a": _make_belief("python", "is_version", "3.12", claim_type="factual"),
        "b": _make_belief("java", "is_version", "21", claim_type="factual"),
    })
    pairs.append({
        "name": "different_domain",
        "expected_type": None,
        "a": _make_belief("user", "likes", "jazz", claim_type="preference"),
        "b": _make_belief("user", "works_at", "google", claim_type="factual"),
    })
    pairs.append({
        "name": "complementary_facts",
        "expected_type": None,
        "a": _make_belief("user", "has_pet", "dog", claim_type="factual"),
        "b": _make_belief("user", "has_pet", "cat", claim_type="factual"),
    })
    pairs.append({
        "name": "same_fact",
        "expected_type": None,
        "a": _make_belief("earth", "orbits", "sun", claim_type="factual"),
        "b": _make_belief("earth", "orbits", "sun", claim_type="factual"),
    })
    pairs.append({
        "name": "different_modality",
        "expected_type": None,
        "a": _make_belief("ai", "may_become", "sentient", modality="may",
                          claim_type="philosophical"),
        "b": _make_belief("ai", "is", "tool", claim_type="factual"),
    })

    return pairs


# ---------------------------------------------------------------------------
# Extraction exercise
# ---------------------------------------------------------------------------

def _build_extraction_memories() -> list[dict[str, Any]]:
    """Build synthetic memories for claim extraction coverage."""
    return [
        {
            "name": "factual_statement",
            "memory": _make_memory("Paris is the capital of France"),
            "expected_claims_min": 1,
        },
        {
            "name": "preference_statement",
            "memory": _make_memory(
                "The user prefers dark mode for all applications",
                mem_type="user_preference",
            ),
            "expected_claims_min": 1,
        },
        {
            "name": "identity_observation",
            "memory": _make_memory(
                "I believe I am capable of learning new things",
                provenance="model_inference",
            ),
            "expected_claims_min": 1,
        },
        {
            "name": "policy_should",
            "memory": _make_memory(
                "The system should always verify before claiming capabilities",
                mem_type="core_belief",
            ),
            "expected_claims_min": 1,
        },
        {
            "name": "structured_dict",
            "memory": _make_memory(
                {"claim": "Python is a programming language", "source": "observation"},
            ),
            "expected_claims_min": 1,
        },
        {
            "name": "empty_payload",
            "memory": _make_memory(""),
            "expected_claims_min": 0,
        },
    ]


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

@dataclass
class ContradictionExerciseProfile:
    name: str
    conflict_pairs: int
    near_miss_pairs: int
    extraction_count: int
    description: str = ""


PROFILES: dict[str, ContradictionExerciseProfile] = {
    "smoke": ContradictionExerciseProfile(
        name="smoke", conflict_pairs=20, near_miss_pairs=5, extraction_count=6,
        description="Quick check (20 conflict + 5 near-miss + 6 extraction)",
    ),
    "coverage": ContradictionExerciseProfile(
        name="coverage", conflict_pairs=40, near_miss_pairs=10, extraction_count=6,
        description="All 6 classes x 5+ positives + 10 near-misses",
    ),
    "stress": ContradictionExerciseProfile(
        name="stress", conflict_pairs=200, near_miss_pairs=50, extraction_count=30,
        description="Randomized high-volume (200 conflict + 50 near-miss)",
    ),
}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class ContradictionExerciseStats:
    pairs_requested: int = 0
    pairs_classified: int = 0
    near_misses_correct: int = 0
    near_misses_total: int = 0
    conflicts_by_type: Counter = field(default_factory=Counter)
    correct_classifications: int = 0
    incorrect_classifications: int = 0
    extraction_attempts: int = 0
    extraction_claims: int = 0
    errors: list[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    profile_name: str = ""

    @property
    def elapsed_s(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    @property
    def classification_accuracy(self) -> float:
        total = self.correct_classifications + self.incorrect_classifications
        return self.correct_classifications / total if total > 0 else 0.0

    @property
    def near_miss_accuracy(self) -> float:
        if self.near_misses_total == 0:
            return 1.0
        return self.near_misses_correct / self.near_misses_total

    @property
    def pass_result(self) -> bool:
        return len(self.fail_reasons) == 0

    @property
    def fail_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.pairs_classified == 0 and self.pairs_requested > 0:
            reasons.append("zero_classifications")
        return reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "pairs_requested": self.pairs_requested,
            "pairs_classified": self.pairs_classified,
            "correct_classifications": self.correct_classifications,
            "incorrect_classifications": self.incorrect_classifications,
            "classification_accuracy": round(self.classification_accuracy, 3),
            "near_misses_correct": self.near_misses_correct,
            "near_misses_total": self.near_misses_total,
            "near_miss_accuracy": round(self.near_miss_accuracy, 3),
            "conflicts_by_type": dict(self.conflicts_by_type),
            "extraction_attempts": self.extraction_attempts,
            "extraction_claims": self.extraction_claims,
            "errors": self.errors[-10:],
            "elapsed_s": round(self.elapsed_s, 2),
            "pass": self.pass_result,
            "fail_reasons": self.fail_reasons,
        }

    def summary(self) -> str:
        lines = [
            f"Contradiction Exercise — {self.pairs_classified} classified, "
            f"accuracy={self.classification_accuracy:.1%}, "
            f"near-miss={self.near_miss_accuracy:.1%} "
            f"in {self.elapsed_s:.1f}s",
        ]
        if self.profile_name:
            lines.append(f"  Profile: {self.profile_name}")
        if self.conflicts_by_type:
            lines.append("  Types: " + ", ".join(
                f"{k}={v}" for k, v in sorted(self.conflicts_by_type.items())
            ))
        lines.append(f"  Extraction: {self.extraction_claims} claims "
                      f"from {self.extraction_attempts} memories")
        if self.fail_reasons:
            lines.append(f"  FAIL: {', '.join(self.fail_reasons)}")
        else:
            lines.append("  PASS: all checks hold")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_contradiction_exercise(
    profile: ContradictionExerciseProfile | None = None,
) -> ContradictionExerciseStats:
    """Run a synchronous contradiction exercise session.

    Uses standalone BeliefStore in tempdir and standalone ConflictClassifier.
    No singleton access, no event emission, no real persistence writes.
    """
    if profile is None:
        profile = PROFILES["coverage"]

    stats = ContradictionExerciseStats(profile_name=profile.name)
    classifier = ConflictClassifier()

    # --- Phase 1: Conflict classification ---
    conflict_corpus = _build_conflict_pairs()
    near_miss_corpus = _build_near_miss_pairs()

    conflict_items: list[dict[str, Any]] = []
    while len(conflict_items) < profile.conflict_pairs:
        for item in conflict_corpus:
            conflict_items.append(item)
            if len(conflict_items) >= profile.conflict_pairs:
                break

    near_miss_items: list[dict[str, Any]] = []
    while len(near_miss_items) < profile.near_miss_pairs:
        for item in near_miss_corpus:
            near_miss_items.append(item)
            if len(near_miss_items) >= profile.near_miss_pairs:
                break

    stats.pairs_requested = len(conflict_items) + len(near_miss_items)

    for item in conflict_items:
        try:
            result = classifier.classify(item["a"], item["b"])
            stats.pairs_classified += 1

            if result is not None:
                stats.conflicts_by_type[result.conflict_type] += 1
                if item["expected_type"] and result.conflict_type == item["expected_type"]:
                    stats.correct_classifications += 1
                else:
                    stats.incorrect_classifications += 1
            else:
                stats.incorrect_classifications += 1

        except Exception as exc:
            stats.errors.append(f"classify {item.get('name', '?')}: "
                                f"{type(exc).__name__}: {exc}")

    stats.near_misses_total = len(near_miss_items)
    for item in near_miss_items:
        try:
            result = classifier.classify(item["a"], item["b"])
            stats.pairs_classified += 1

            if result is None:
                stats.near_misses_correct += 1
            else:
                stats.conflicts_by_type[f"near_miss_false_{result.conflict_type}"] += 1

        except Exception as exc:
            stats.errors.append(f"near_miss {item.get('name', '?')}: "
                                f"{type(exc).__name__}: {exc}")

    # --- Phase 2: Claim extraction ---
    extraction_corpus = _build_extraction_memories()
    extraction_items: list[dict[str, Any]] = []
    while len(extraction_items) < profile.extraction_count:
        for item in extraction_corpus:
            extraction_items.append(item)
            if len(extraction_items) >= profile.extraction_count:
                break

    for item in extraction_items:
        try:
            claims = extract_claims(item["memory"])
            stats.extraction_attempts += 1
            stats.extraction_claims += len(claims)
        except Exception as exc:
            stats.errors.append(f"extract {item.get('name', '?')}: "
                                f"{type(exc).__name__}: {exc}")

    # --- Phase 3: BeliefStore integration (temp dir) ---
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_beliefs = str(Path(tmpdir) / "beliefs.jsonl")
        tmp_tensions = str(Path(tmpdir) / "tensions.jsonl")
        store = BeliefStore(beliefs_path=tmp_beliefs, tensions_path=tmp_tensions)

        for item in conflict_items[:10]:
            try:
                store.add(item["a"])
                store.add(item["b"])
            except Exception:
                pass

        store_stats = store.get_stats()
        logger.debug("Temp BeliefStore: %d beliefs stored", store_stats.get("total", 0))

    stats.end_time = time.time()
    logger.info(
        "Contradiction exercise: %d classified, accuracy=%.1f%%, "
        "near-miss=%.1f%%",
        stats.pairs_classified,
        stats.classification_accuracy * 100,
        stats.near_miss_accuracy * 100,
    )
    return stats
