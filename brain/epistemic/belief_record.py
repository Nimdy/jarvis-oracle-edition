"""Belief records, tension records, belief store, and contradiction debt.

The canonical unit is a normalized proposition, not a human-readable string.
The rendered claim is derived display text.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from nanoid import generate as nanoid

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases — no raw strings outside this file
# ---------------------------------------------------------------------------

Modality = Literal[
    "is", "may", "prefers", "should", "observed_as", "caused", "predicted",
]

Stance = Literal["assert", "deny", "uncertain", "question"]

EpistemicStatus = Literal[
    "observed", "inferred", "adopted", "questioned", "provisional", "stabilized",
]

ClaimType = Literal[
    "factual", "identity", "policy", "preference", "observation", "philosophical",
]

ConflictType = Literal[
    "factual", "provenance", "policy_outcome", "policy_norm",
    "temporal", "identity_tension", "multi_perspective",
]

ResolutionState = Literal[
    "active", "superseded", "versioned", "tension", "resolved", "quarantined",
]

Severity = Literal["critical", "moderate", "informational"]

VALID_MODALITIES = {"is", "may", "prefers", "should", "observed_as", "caused", "predicted"}
VALID_STANCES = {"assert", "deny", "uncertain", "question"}
VALID_EPISTEMIC_STATUSES = {"observed", "inferred", "adopted", "questioned", "provisional", "stabilized"}
VALID_CLAIM_TYPES = {"factual", "identity", "policy", "preference", "observation", "philosophical"}
VALID_RESOLUTION_STATES = {"active", "superseded", "versioned", "tension", "resolved", "quarantined"}

# ---------------------------------------------------------------------------
# Debt schedule constants
# ---------------------------------------------------------------------------

DEBT_CRITICAL_UNRESOLVED: float = 0.05
DEBT_MODERATE_UNRESOLVED: float = 0.02
DEBT_RECURRENCE_EXTRA: float = 0.01
DEBT_USER_CORRECTION: float = 0.03
DEBT_FACTUAL_RESOLVED: float = -0.03
DEBT_TEMPORAL_VERSION: float = 0.0
DEBT_IDENTITY_TENSION: float = 0.0
DEBT_TENSION_MATURED: float = -0.005
DEBT_POLICY_NORM: float = 0.005
DEBT_MULTI_PERSPECTIVE: float = 0.0
DEBT_PASSIVE_DECAY_PER_HOUR: float = -0.01
DEBT_MIN: float = 0.0
DEBT_MAX: float = 1.0

# ---------------------------------------------------------------------------
# Maturation schedule constants
# ---------------------------------------------------------------------------

MATURATION_REVISIT_PLAIN: float = 0.01
MATURATION_REVISIT_DUAL_SIDED: float = 0.03
MATURATION_REVISIT_NEW_EVIDENCE: float = 0.02
MATURATION_PASSIVE_DECAY_PER_DAY: float = 0.0

# ---------------------------------------------------------------------------
# Tension constraints
# ---------------------------------------------------------------------------

TENSION_MAX_BELIEF_IDS: int = 10

# ---------------------------------------------------------------------------
# Extraction thresholds
# ---------------------------------------------------------------------------

EXTRACTION_DISCARD_THRESHOLD: float = 0.2
EXTRACTION_NEAR_MISS_THRESHOLD: float = 0.3
EXTRACTION_MAX_CLAIMS_PER_MEMORY: int = 3
EXTRACTION_AMBIGUOUS_CONFIDENCE: float = 0.4

# ---------------------------------------------------------------------------
# Store limits
# ---------------------------------------------------------------------------

BELIEF_STORE_MAX_CAPACITY: int = 2000
NEAR_MISS_RING_BUFFER_SIZE: int = 200
DEBT_TREND_WINDOW: int = 10

# ---------------------------------------------------------------------------
# Tick intervals
# ---------------------------------------------------------------------------

CONTRADICTION_CHECK_INTERVAL_S: float = 60.0
CONTRADICTION_CHECK_ACCELERATED_S: float = 30.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeliefRecord:
    belief_id: str

    canonical_subject: str
    canonical_predicate: str
    canonical_object: str

    modality: str
    stance: str
    polarity: int

    claim_type: str
    epistemic_status: str
    extraction_confidence: float
    belief_confidence: float
    provenance: str

    scope: str
    source_memory_id: str
    timestamp: float
    time_range: tuple[float, float] | None
    is_state_belief: bool

    conflict_key: str
    evidence_refs: list[str]
    contradicts: list[str]
    resolution_state: str

    rendered_claim: str

    identity_subject_id: str = ""
    identity_subject_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["time_range"] is not None:
            d["time_range"] = list(d["time_range"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BeliefRecord:
        tr = d.get("time_range")
        if tr is not None and isinstance(tr, (list, tuple)):
            d["time_range"] = tuple(tr)
        return cls(**d)


@dataclass
class TensionRecord:
    tension_id: str
    topic: str
    belief_ids: list[str]
    conflict_key: str
    created_at: float
    last_revisited: float
    revisit_count: int
    stability: float
    maturation_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TensionRecord:
        return cls(**d)


@dataclass(frozen=True)
class NearMiss:
    timestamp: float
    belief_a_id: str
    belief_b_id: str
    reason: str
    subject: str


@dataclass(frozen=True)
class ConflictClassification:
    conflict_type: str
    severity: str
    is_pathological: bool
    confidence: float
    reasoning: str
    conflict_key: str


@dataclass(frozen=True)
class ResolutionOutcome:
    action_taken: str
    beliefs_modified: list[str]
    confidence_deltas: dict[str, float]
    tension_id: str | None
    ledger_entry_id: str
    needs_user_clarification: bool
    clarification_prompt: str | None
    debt_delta: float


# ---------------------------------------------------------------------------
# Tension topic mapper
# ---------------------------------------------------------------------------

_TENSION_DOMAINS: dict[frozenset[str], str] = {
    frozenset({"identity", "continuity", "replication", "backup"}):
        "identity::continuity_vs_replication",
    frozenset({"agency", "determinism", "free_will", "freedom"}):
        "agency::determinism_vs_free_will",
    frozenset({"meaning", "simulation", "value", "purpose"}):
        "meaning::simulation_vs_value",
    frozenset({"memory", "selfhood", "persistence"}):
        "memory::selfhood_vs_persistence",
    frozenset({"embodiment", "perception", "reality"}):
        "embodiment::perception_vs_reality",
    frozenset({"consciousness", "awareness", "experience", "qualia"}):
        "consciousness::awareness_vs_experience",
    frozenset({"mortality", "death", "finite", "temporal"}):
        "mortality::finite_vs_eternal",
}


def infer_tension_topic(a: BeliefRecord, b: BeliefRecord) -> str:
    terms = {a.canonical_subject, a.canonical_object, b.canonical_subject, b.canonical_object}
    for domain_terms, topic in _TENSION_DOMAINS.items():
        if terms & domain_terms:
            return topic
    return f"unclassified::{a.canonical_subject}_vs_{b.canonical_subject}"


# ---------------------------------------------------------------------------
# Conflict key builder
# ---------------------------------------------------------------------------

def build_conflict_key(belief: BeliefRecord) -> str:
    ct = belief.claim_type
    s = belief.canonical_subject
    o = belief.canonical_object

    if ct == "factual":
        return f"fact::{s}::{o}"
    elif ct == "observation":
        return f"state::{s}"
    elif ct == "preference":
        return f"pref::{s}::{o}"
    elif ct == "policy":
        if belief.modality == "should":
            return f"policy_norm::{s}::{o}"
        else:
            return f"policy_outcome::{s}::{o}"
    elif ct in ("identity", "philosophical"):
        return f"identity::{s}::{o}"
    else:
        return f"other::{s}::{o}"


# ---------------------------------------------------------------------------
# BeliefStore — in-memory dict + JSONL persistence
# ---------------------------------------------------------------------------

_JARVIS_DIR = os.path.expanduser("~/.jarvis")
_BELIEFS_FILE = os.path.join(_JARVIS_DIR, "beliefs.jsonl")
_TENSIONS_FILE = os.path.join(_JARVIS_DIR, "tensions.jsonl")


class BeliefStore:
    def __init__(
        self,
        beliefs_path: str = _BELIEFS_FILE,
        tensions_path: str = _TENSIONS_FILE,
        max_capacity: int = BELIEF_STORE_MAX_CAPACITY,
    ) -> None:
        self._beliefs: OrderedDict[str, BeliefRecord] = OrderedDict()
        self._tensions: dict[str, TensionRecord] = {}
        self._beliefs_path = beliefs_path
        self._tensions_path = tensions_path
        self._max_capacity = max_capacity
        self._lock = threading.Lock()
        self._subject_index: dict[str, set[str]] = {}

    # -- CRUD ---------------------------------------------------------------

    def add(self, belief: BeliefRecord) -> bool:
        with self._lock:
            if belief.belief_id in self._beliefs:
                return False
            self._beliefs[belief.belief_id] = belief
            self._subject_index.setdefault(belief.canonical_subject, set()).add(belief.belief_id)
            self._append_belief_jsonl(belief)
            if len(self._beliefs) > self._max_capacity:
                self._evict_unlocked()
            return True

    def get(self, belief_id: str) -> BeliefRecord | None:
        with self._lock:
            return self._beliefs.get(belief_id)

    def find_by_subject(self, subject: str) -> list[BeliefRecord]:
        with self._lock:
            ids = self._subject_index.get(subject, set())
            return [self._beliefs[bid] for bid in ids if bid in self._beliefs]

    def find_by_type(self, claim_type: str) -> list[BeliefRecord]:
        with self._lock:
            return [b for b in self._beliefs.values() if b.claim_type == claim_type]

    def find_by_modality(self, modality: str) -> list[BeliefRecord]:
        with self._lock:
            return [b for b in self._beliefs.values() if b.modality == modality]

    def find_by_conflict_key(self, key: str) -> list[BeliefRecord]:
        with self._lock:
            return [b for b in self._beliefs.values() if b.conflict_key == key]

    def get_active_beliefs(self) -> list[BeliefRecord]:
        with self._lock:
            return [b for b in self._beliefs.values() if b.resolution_state == "active"]

    def update_resolution(self, belief_id: str, new_state: str) -> bool:
        with self._lock:
            old = self._beliefs.get(belief_id)
            if old is None:
                return False
            updated = BeliefRecord(
                belief_id=old.belief_id,
                canonical_subject=old.canonical_subject,
                canonical_predicate=old.canonical_predicate,
                canonical_object=old.canonical_object,
                modality=old.modality,
                stance=old.stance,
                polarity=old.polarity,
                claim_type=old.claim_type,
                epistemic_status=old.epistemic_status,
                extraction_confidence=old.extraction_confidence,
                belief_confidence=old.belief_confidence,
                provenance=old.provenance,
                scope=old.scope,
                source_memory_id=old.source_memory_id,
                timestamp=old.timestamp,
                time_range=old.time_range,
                is_state_belief=old.is_state_belief,
                conflict_key=old.conflict_key,
                evidence_refs=old.evidence_refs,
                contradicts=old.contradicts,
                resolution_state=new_state,
                rendered_claim=old.rendered_claim,
                identity_subject_id=old.identity_subject_id,
                identity_subject_type=old.identity_subject_type,
            )
            self._beliefs[belief_id] = updated
            self._append_belief_jsonl(updated)
            return True

    def update_belief_confidence(self, belief_id: str, new_confidence: float) -> bool:
        with self._lock:
            old = self._beliefs.get(belief_id)
            if old is None:
                return False
            updated = BeliefRecord(
                belief_id=old.belief_id,
                canonical_subject=old.canonical_subject,
                canonical_predicate=old.canonical_predicate,
                canonical_object=old.canonical_object,
                modality=old.modality,
                stance=old.stance,
                polarity=old.polarity,
                claim_type=old.claim_type,
                epistemic_status=old.epistemic_status,
                extraction_confidence=old.extraction_confidence,
                belief_confidence=new_confidence,
                provenance=old.provenance,
                scope=old.scope,
                source_memory_id=old.source_memory_id,
                timestamp=old.timestamp,
                time_range=old.time_range,
                is_state_belief=old.is_state_belief,
                conflict_key=old.conflict_key,
                evidence_refs=old.evidence_refs,
                contradicts=old.contradicts,
                resolution_state=old.resolution_state,
                rendered_claim=old.rendered_claim,
                identity_subject_id=old.identity_subject_id,
                identity_subject_type=old.identity_subject_type,
            )
            self._beliefs[belief_id] = updated
            self._append_belief_jsonl(updated)
            return True

    def add_contradiction_link(self, belief_id: str, other_id: str) -> bool:
        with self._lock:
            old = self._beliefs.get(belief_id)
            if old is None:
                return False
            if other_id in old.contradicts:
                return True
            new_contradicts = list(old.contradicts) + [other_id]
            updated = BeliefRecord(
                belief_id=old.belief_id,
                canonical_subject=old.canonical_subject,
                canonical_predicate=old.canonical_predicate,
                canonical_object=old.canonical_object,
                modality=old.modality,
                stance=old.stance,
                polarity=old.polarity,
                claim_type=old.claim_type,
                epistemic_status=old.epistemic_status,
                extraction_confidence=old.extraction_confidence,
                belief_confidence=old.belief_confidence,
                provenance=old.provenance,
                scope=old.scope,
                source_memory_id=old.source_memory_id,
                timestamp=old.timestamp,
                time_range=old.time_range,
                is_state_belief=old.is_state_belief,
                conflict_key=old.conflict_key,
                evidence_refs=old.evidence_refs,
                contradicts=new_contradicts,
                resolution_state=old.resolution_state,
                rendered_claim=old.rendered_claim,
                identity_subject_id=old.identity_subject_id,
                identity_subject_type=old.identity_subject_type,
            )
            self._beliefs[belief_id] = updated
            self._append_belief_jsonl(updated)
            return True

    # -- Tensions -----------------------------------------------------------

    def add_tension(self, tension: TensionRecord) -> bool:
        with self._lock:
            if tension.tension_id in self._tensions:
                return False
            self._tensions[tension.tension_id] = tension
            self._append_tension_jsonl(tension)
            return True

    def get_tension(self, tension_id: str) -> TensionRecord | None:
        with self._lock:
            return self._tensions.get(tension_id)

    def get_tension_by_topic(self, topic: str) -> TensionRecord | None:
        with self._lock:
            for t in self._tensions.values():
                if t.topic == topic:
                    return t
            return None

    def update_tension(self, tension: TensionRecord) -> None:
        with self._lock:
            self._tensions[tension.tension_id] = tension
            self._append_tension_jsonl(tension)

    def get_active_tensions(self) -> list[TensionRecord]:
        with self._lock:
            return list(self._tensions.values())

    # -- Eviction -----------------------------------------------------------

    def _evict_unlocked(self) -> int:
        evictable = [
            bid for bid, b in self._beliefs.items()
            if b.resolution_state in ("resolved", "superseded")
        ]
        evictable.sort(key=lambda bid: self._beliefs[bid].timestamp)
        evicted = 0
        while len(self._beliefs) > self._max_capacity and evictable:
            bid = evictable.pop(0)
            b = self._beliefs.pop(bid, None)
            if b:
                subj_ids = self._subject_index.get(b.canonical_subject)
                if subj_ids:
                    subj_ids.discard(bid)
                    if not subj_ids:
                        del self._subject_index[b.canonical_subject]
                evicted += 1
        return evicted

    # -- Stats --------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            by_type: dict[str, int] = {}
            by_resolution: dict[str, int] = {}
            for b in self._beliefs.values():
                by_type[b.claim_type] = by_type.get(b.claim_type, 0) + 1
                by_resolution[b.resolution_state] = by_resolution.get(b.resolution_state, 0) + 1
            return {
                "total_beliefs": len(self._beliefs),
                "active_beliefs": by_resolution.get("active", 0),
                "tension_beliefs": by_resolution.get("tension", 0),
                "active_tensions": len(self._tensions),
                "by_type": by_type,
                "by_resolution": by_resolution,
            }

    # -- Persistence --------------------------------------------------------

    def _append_belief_jsonl(self, belief: BeliefRecord) -> None:
        try:
            os.makedirs(os.path.dirname(self._beliefs_path), exist_ok=True)
            with open(self._beliefs_path, "a") as f:
                f.write(json.dumps(belief.to_dict()) + "\n")
        except Exception:
            logger.exception("Failed to append belief JSONL")

    def _append_tension_jsonl(self, tension: TensionRecord) -> None:
        try:
            os.makedirs(os.path.dirname(self._tensions_path), exist_ok=True)
            with open(self._tensions_path, "a") as f:
                f.write(json.dumps(tension.to_dict()) + "\n")
        except Exception:
            logger.exception("Failed to append tension JSONL")

    def rehydrate(self) -> None:
        with self._lock:
            self._beliefs.clear()
            self._tensions.clear()
            self._subject_index.clear()

            if os.path.exists(self._beliefs_path):
                try:
                    with open(self._beliefs_path) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                d = json.loads(line)
                                b = BeliefRecord.from_dict(d)
                                self._beliefs[b.belief_id] = b
                                self._subject_index.setdefault(b.canonical_subject, set()).add(b.belief_id)
                            except Exception:
                                continue
                    logger.info("Rehydrated %d beliefs from %s", len(self._beliefs), self._beliefs_path)
                except Exception:
                    logger.exception("Failed to rehydrate beliefs")

            if os.path.exists(self._tensions_path):
                try:
                    with open(self._tensions_path) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                d = json.loads(line)
                                t = TensionRecord.from_dict(d)
                                self._tensions[t.tension_id] = t
                            except Exception:
                                continue
                    logger.info("Rehydrated %d tensions from %s", len(self._tensions), self._tensions_path)
                except Exception:
                    logger.exception("Failed to rehydrate tensions")

    def persist_full(self) -> None:
        """Rewrite both JSONL files from current in-memory state."""
        with self._lock:
            try:
                os.makedirs(os.path.dirname(self._beliefs_path), exist_ok=True)
                with open(self._beliefs_path, "w") as f:
                    for b in self._beliefs.values():
                        f.write(json.dumps(b.to_dict()) + "\n")
            except Exception:
                logger.exception("Failed to persist beliefs")
            try:
                os.makedirs(os.path.dirname(self._tensions_path), exist_ok=True)
                with open(self._tensions_path, "w") as f:
                    for t in self._tensions.values():
                        f.write(json.dumps(t.to_dict()) + "\n")
            except Exception:
                logger.exception("Failed to persist tensions")
