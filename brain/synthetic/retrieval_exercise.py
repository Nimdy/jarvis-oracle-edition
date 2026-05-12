"""Synthetic memory retrieval exercise.

Generates synthetic training pairs for the MemoryRanker and SalienceModel
by issuing read-only queries against the existing memory store and vector
index. Produces TrainingPair-compatible JSONL that cortex training can
load alongside real telemetry.

Truth boundary:
  - Read-only access to MemoryStorage and VectorStore
  - NEVER calls retrieval_log.log_retrieval() or log_outcome()
  - NEVER calls engine.remember() (no MEMORY_WRITE events)
  - NEVER writes to memory_retrieval_log.jsonl or memory_lifecycle_log.jsonl
  - Synthetic pairs written to a separate file with fidelity 0.7
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SYNTHETIC_FIDELITY = 0.7
SYNTHETIC_ORIGIN = "synthetic"
REPORT_DIR = Path.home() / ".jarvis" / "synthetic_exercise"
PAIRS_PATH = REPORT_DIR / "retrieval_pairs.jsonl"

# ---------------------------------------------------------------------------
# Query corpus — covers diverse retrieval patterns
# ---------------------------------------------------------------------------

_FACTUAL_QUERIES = [
    "neural network architecture",
    "memory system design",
    "how does consciousness work",
    "causal inference methods",
    "reinforcement learning techniques",
    "transformer attention mechanism",
    "speech recognition models",
    "episodic memory retrieval",
    "belief propagation algorithms",
    "sensor fusion techniques",
]

_IDENTITY_QUERIES = [
    "who is the primary user",
    "what is my name",
    "speaker identification",
    "face recognition profile",
    "identity fusion",
    "voice enrollment",
    "household members",
    "relationship graph",
]

_PREFERENCE_QUERIES = [
    "user preferences",
    "favorite topics",
    "communication style",
    "notification preferences",
    "proactivity settings",
    "tone preferences",
]

_TECHNICAL_QUERIES = [
    "tool routing system",
    "event bus architecture",
    "kernel tick loop",
    "hemisphere training pipeline",
    "self-improvement sandbox",
    "capability gate enforcement",
    "world model predictions",
    "truth calibration scoring",
]

_REFLECTIVE_QUERIES = [
    "what have I learned recently",
    "consciousness evolution",
    "existential reasoning",
    "self-observation patterns",
    "philosophical dialogue history",
    "dream consolidation results",
]

_RANDOM_QUERIES = [
    "quantum computing basics",
    "protein folding",
    "climate change data",
    "cooking recipes",
    "space exploration",
    "music theory",
    "history of mathematics",
    "modern art movements",
]

QUERY_CATEGORIES: dict[str, list[str]] = {
    "factual": _FACTUAL_QUERIES,
    "identity": _IDENTITY_QUERIES,
    "preference": _PREFERENCE_QUERIES,
    "technical": _TECHNICAL_QUERIES,
    "reflective": _REFLECTIVE_QUERIES,
    "random": _RANDOM_QUERIES,
}

ALL_QUERIES: list[str] = []
for _cat_list in QUERY_CATEGORIES.values():
    ALL_QUERIES.extend(_cat_list)

COVERAGE_WEIGHTS: dict[str, float] = {
    "factual": 1.5,
    "identity": 1.0,
    "preference": 1.0,
    "technical": 2.0,
    "reflective": 1.5,
    "random": 0.5,
}


def pick_query(
    category: str | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[str, str]:
    """Return (query_text, category_name)."""
    if category and category in QUERY_CATEGORIES:
        items = QUERY_CATEGORIES[category]
        return random.choice(items), category

    cats = list(QUERY_CATEGORIES.keys())
    if weights:
        w = [weights.get(c, 0.5) for c in cats]
        total = sum(w)
        if total > 0:
            w = [x / total for x in w]
        cat = random.choices(cats, weights=w, k=1)[0]
    else:
        cat = random.choice(cats)
    return random.choice(QUERY_CATEGORIES[cat]), cat


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

@dataclass
class RetrievalExerciseProfile:
    name: str
    count: int
    delay_s: float
    category_weights: dict[str, float] | None = None
    description: str = ""

    def effective_count(self, duration_s: float | None = None) -> int:
        if duration_s and self.delay_s > 0:
            return max(1, int(duration_s / self.delay_s))
        return self.count


PROFILES: dict[str, RetrievalExerciseProfile] = {
    "smoke": RetrievalExerciseProfile(
        name="smoke", count=20, delay_s=0.05,
        description="Quick check (20 queries)",
    ),
    "coverage": RetrievalExerciseProfile(
        name="coverage", count=100, delay_s=0.05,
        category_weights=COVERAGE_WEIGHTS,
        description="Weighted category coverage (100 queries)",
    ),
    "stress": RetrievalExerciseProfile(
        name="stress", count=500, delay_s=0.01,
        category_weights=COVERAGE_WEIGHTS,
        description="High-volume pair generation (500 queries)",
    ),
}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class RetrievalExerciseStats:
    queries_requested: int = 0
    queries_processed: int = 0
    queries_failed: int = 0
    pairs_generated: int = 0
    positive_pairs: int = 0
    negative_pairs: int = 0
    categories_exercised: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    profile_name: str = ""

    @property
    def elapsed_s(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    @property
    def rate_per_sec(self) -> float:
        elapsed = self.elapsed_s
        if elapsed < 0.01:
            return 0.0
        return self.queries_processed / elapsed

    @property
    def pass_result(self) -> bool:
        return len(self.fail_reasons) == 0

    @property
    def fail_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.queries_failed > self.queries_processed * 0.3:
            reasons.append(
                f"high_failure_rate={self.queries_failed}/{self.queries_processed}"
            )
        if self.pairs_generated == 0 and self.queries_processed > 0:
            reasons.append("zero_pairs_generated")
        return reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "queries_requested": self.queries_requested,
            "queries_processed": self.queries_processed,
            "queries_failed": self.queries_failed,
            "pairs_generated": self.pairs_generated,
            "positive_pairs": self.positive_pairs,
            "negative_pairs": self.negative_pairs,
            "categories_exercised": dict(self.categories_exercised),
            "errors": self.errors[-10:],
            "elapsed_s": round(self.elapsed_s, 2),
            "rate_per_sec": round(self.rate_per_sec, 1),
            "pass": self.pass_result,
            "fail_reasons": self.fail_reasons,
        }

    def summary(self) -> str:
        lines = [
            f"Retrieval Exercise — {self.queries_processed} queries, "
            f"{self.pairs_generated} pairs ({self.positive_pairs}+/{self.negative_pairs}-) "
            f"in {self.elapsed_s:.1f}s ({self.rate_per_sec:.0f}/sec)",
        ]
        if self.profile_name:
            lines.append(f"  Profile: {self.profile_name}")
        if self.categories_exercised:
            top = self.categories_exercised.most_common(6)
            lines.append("  Categories: " + ", ".join(f"{k}={v}" for k, v in top))
        if self.fail_reasons:
            lines.append(f"  FAIL: {', '.join(self.fail_reasons)}")
        else:
            lines.append("  PASS: all checks hold")
        if self.errors:
            lines.append(f"  Last error: {self.errors[-1][:80]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Heuristic label assignment
# ---------------------------------------------------------------------------

def _assign_label(similarity: float, tag_overlap: bool,
                  weight: float) -> float:
    """Heuristic label for a synthetic retrieval candidate.

    High similarity + tag overlap = highly relevant (1.0).
    Low similarity + no tag overlap = irrelevant (0.0).
    """
    score = 0.0
    if similarity > 0.6 and tag_overlap:
        score = 1.0
    elif similarity > 0.5:
        score = 0.8
    elif similarity > 0.35 and tag_overlap:
        score = 0.6
    elif similarity > 0.35:
        score = 0.4
    elif similarity > 0.2:
        score = 0.2
    score *= min(weight / 0.5, 1.0)
    return round(min(score, 1.0), 3)


def _query_tags_overlap(query: str, mem_tags: tuple[str, ...]) -> bool:
    """Check if any query token appears in the memory tags."""
    tokens = {t.lower() for t in query.split() if len(t) > 3}
    tag_str = " ".join(mem_tags).lower()
    return any(tok in tag_str for tok in tokens)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_retrieval_exercise(
    profile: RetrievalExerciseProfile | None = None,
    count: int | None = None,
    vector_store: Any | None = None,
    memory_storage: Any | None = None,
) -> RetrievalExerciseStats:
    """Run a synchronous retrieval exercise session.

    Requires a running brain with initialized vector store and memory storage.
    If not provided, attempts to import the global singletons (read-only).
    """
    if profile is None:
        profile = PROFILES["coverage"]

    n = count or profile.count
    stats = RetrievalExerciseStats(profile_name=profile.name)
    stats.queries_requested = n

    if vector_store is None:
        try:
            from memory.search import get_vector_store
            vector_store = get_vector_store()
        except Exception:
            pass
    if memory_storage is None:
        try:
            from memory.storage import memory_storage as _ms
            memory_storage = _ms
        except Exception:
            pass

    if not vector_store or not getattr(vector_store, "available", False):
        stats.errors.append("vector_store not available")
        stats.end_time = time.time()
        return stats

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    pairs_written = 0

    try:
        pairs_file = open(PAIRS_PATH, "a")
    except OSError as exc:
        stats.errors.append(f"cannot open pairs file: {exc}")
        stats.end_time = time.time()
        return stats

    try:
        for _ in range(n):
            query_text, category = pick_query(weights=profile.category_weights)
            stats.categories_exercised[category] += 1

            try:
                raw = vector_store.search(query_text, top_k=20, min_weight=0.0)
                if not raw:
                    stats.queries_processed += 1
                    continue

                for r in raw:
                    mem_id = r.get("memory_id", "")
                    mem = None
                    if memory_storage and hasattr(memory_storage, "get"):
                        mem = memory_storage.get(mem_id)
                    if not mem:
                        continue

                    similarity = r.get("similarity", 0.5)
                    age_hours = max(0.01, (now - mem.timestamp) / 3600.0)
                    recency_score = 1.0 / (1.0 + age_hours * 0.1)
                    tag_overlap = _query_tags_overlap(query_text, mem.tags)

                    label = _assign_label(similarity, tag_overlap, mem.weight)

                    features = [
                        similarity,
                        recency_score,
                        mem.weight,
                        0.0,  # heuristic_score placeholder
                        min(len(mem.tags), 10) / 10.0,
                        min(getattr(mem, "association_count", 0), 10) / 10.0,
                        getattr(mem, "priority", 0) / 1000.0,
                        0.0,  # provenance_boost placeholder
                        1.0 if tag_overlap else 0.0,
                        1.0 if getattr(mem, "is_core", False) else 0.0,
                        1.0 if mem.type == "conversation" else 0.0,
                        1.0 if mem.type == "factual_knowledge" else 0.0,
                    ]

                    pair = {
                        "features": features,
                        "label": label,
                        "fidelity": SYNTHETIC_FIDELITY,
                        "origin": SYNTHETIC_ORIGIN,
                        "query": query_text,
                        "memory_id": mem_id,
                        "timestamp": now,
                    }
                    pairs_file.write(json.dumps(pair) + "\n")
                    pairs_written += 1
                    stats.pairs_generated += 1
                    if label >= 0.5:
                        stats.positive_pairs += 1
                    else:
                        stats.negative_pairs += 1

                stats.queries_processed += 1

            except Exception as exc:
                stats.queries_failed += 1
                stats.errors.append(f"{type(exc).__name__}: {exc}")

            if profile.delay_s > 0:
                time.sleep(profile.delay_s)
    finally:
        pairs_file.close()

    stats.end_time = time.time()
    logger.info("Retrieval exercise: %d pairs written to %s", pairs_written, PAIRS_PATH)
    return stats


def load_synthetic_pairs(max_pairs: int = 500) -> list[dict[str, Any]]:
    """Load synthetic retrieval pairs for cortex training integration."""
    if not PAIRS_PATH.exists():
        return []
    pairs: list[dict[str, Any]] = []
    try:
        with open(PAIRS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(pairs) >= max_pairs:
                    break
    except OSError:
        pass
    return pairs
