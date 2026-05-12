"""Edge store for the Belief Confidence Graph (Layer 7).

Provides EvidenceEdge — a weighted directed edge between two BeliefRecords —
and EdgeStore — the in-memory graph with bidirectional indices and JSONL
persistence with periodic compaction.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from nanoid import generate as nanoid

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGE_STRENGTH_DECAY_PER_DAY: float = 0.995
EDGE_STRENGTH_MIN: float = 0.01
EDGE_STORE_RECENT_BUFFER: int = 50
COMPACTION_RATIO_THRESHOLD: float = 2.0

VALID_EDGE_TYPES = frozenset({
    "supports", "contradicts", "refines", "depends_on", "derived_from",
})

VALID_EVIDENCE_BASES = frozenset({
    "shared_subject", "causal", "temporal_sequence", "belief_version",
    "extractor_link", "user_correction", "resolution_outcome",
    "memory_association", "orphan_fill",
})

_JARVIS_DIR = os.path.expanduser("~/.jarvis")
_EDGES_FILE = os.path.join(_JARVIS_DIR, "belief_edges.jsonl")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceEdge:
    """A weighted directed edge between two beliefs.

    ``depends_on`` direction: **dependent -> prerequisite**.
    The source_belief_id is the belief that *requires* the target; the
    target_belief_id is the prerequisite it depends on.  If the prerequisite
    is weakened, the dependent's effective confidence should decrease.
    """

    edge_id: str
    source_belief_id: str
    target_belief_id: str
    edge_type: str          # one of VALID_EDGE_TYPES
    strength: float         # [0.0, 1.0]
    provenance: str         # inherited from source belief's provenance
    created_at: float
    last_updated: float
    evidence_basis: str     # one of VALID_EVIDENCE_BASES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvidenceEdge:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def with_strength(self, new_strength: float) -> EvidenceEdge:
        return EvidenceEdge(
            edge_id=self.edge_id,
            source_belief_id=self.source_belief_id,
            target_belief_id=self.target_belief_id,
            edge_type=self.edge_type,
            strength=max(0.0, min(1.0, new_strength)),
            provenance=self.provenance,
            created_at=self.created_at,
            last_updated=time.time(),
            evidence_basis=self.evidence_basis,
        )


# ---------------------------------------------------------------------------
# EdgeStore — in-memory graph with bidirectional indices
# ---------------------------------------------------------------------------

class EdgeStore:
    """Thread-safe edge store with O(1) neighbour lookup and JSONL persistence."""

    def __init__(self, edges_path: str = _EDGES_FILE) -> None:
        self._edges: dict[str, EvidenceEdge] = {}
        self._outgoing: dict[str, set[str]] = {}    # belief_id -> edge_ids going out
        self._incoming: dict[str, set[str]] = {}     # belief_id -> edge_ids coming in
        self._by_type: dict[str, set[str]] = {}      # edge_type -> edge_ids
        self._dedup: dict[tuple[str, str, str, str], str] = {}  # merge key -> edge_id

        self._edges_path = edges_path
        self._lock = threading.Lock()
        self._recent: deque[EvidenceEdge] = deque(maxlen=EDGE_STORE_RECENT_BUFFER)
        self._jsonl_line_count: int = 0
        self._total_created: int = 0
        self._rehydrating: bool = False

    # -- CRUD ---------------------------------------------------------------

    def add(self, edge: EvidenceEdge) -> bool:
        """Add an edge.  Merges duplicates by (source, target, type, basis)."""
        if edge.edge_type not in VALID_EDGE_TYPES:
            logger.warning("Invalid edge type: %s", edge.edge_type)
            return False
        if edge.evidence_basis not in VALID_EVIDENCE_BASES:
            logger.warning("Invalid evidence basis: %s", edge.evidence_basis)
            return False
        if edge.source_belief_id == edge.target_belief_id:
            return False

        merge_key = (
            edge.source_belief_id,
            edge.target_belief_id,
            edge.edge_type,
            edge.evidence_basis,
        )

        with self._lock:
            existing_id = self._dedup.get(merge_key)
            if existing_id and existing_id in self._edges:
                old = self._edges[existing_id]
                merged = old.with_strength(max(old.strength, edge.strength))
                self._edges[existing_id] = merged
                self._recent.append(merged)
                self._append_jsonl(merged)
                return True

            self._edges[edge.edge_id] = edge
            self._outgoing.setdefault(edge.source_belief_id, set()).add(edge.edge_id)
            self._incoming.setdefault(edge.target_belief_id, set()).add(edge.edge_id)
            self._by_type.setdefault(edge.edge_type, set()).add(edge.edge_id)
            self._dedup[merge_key] = edge.edge_id
            self._recent.append(edge)
            self._total_created += 1
            self._append_jsonl(edge)
            return True

    def get(self, edge_id: str) -> EvidenceEdge | None:
        with self._lock:
            return self._edges.get(edge_id)

    def get_outgoing(self, belief_id: str) -> list[EvidenceEdge]:
        with self._lock:
            edge_ids = self._outgoing.get(belief_id, set())
            return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def get_incoming(self, belief_id: str) -> list[EvidenceEdge]:
        with self._lock:
            edge_ids = self._incoming.get(belief_id, set())
            return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def get_by_type(self, edge_type: str) -> list[EvidenceEdge]:
        with self._lock:
            edge_ids = self._by_type.get(edge_type, set())
            return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def remove(self, edge_id: str) -> bool:
        with self._lock:
            return self._remove_unlocked(edge_id)

    def _remove_unlocked(self, edge_id: str) -> bool:
        edge = self._edges.pop(edge_id, None)
        if edge is None:
            return False
        out_set = self._outgoing.get(edge.source_belief_id)
        if out_set:
            out_set.discard(edge_id)
            if not out_set:
                del self._outgoing[edge.source_belief_id]
        in_set = self._incoming.get(edge.target_belief_id)
        if in_set:
            in_set.discard(edge_id)
            if not in_set:
                del self._incoming[edge.target_belief_id]
        type_set = self._by_type.get(edge.edge_type)
        if type_set:
            type_set.discard(edge_id)
            if not type_set:
                del self._by_type[edge.edge_type]
        merge_key = (
            edge.source_belief_id,
            edge.target_belief_id,
            edge.edge_type,
            edge.evidence_basis,
        )
        self._dedup.pop(merge_key, None)
        return True

    # -- Belief eviction (cascade) ------------------------------------------

    def remove_edges_for_belief(self, belief_id: str) -> int:
        """Remove all edges referencing a belief.  Called when belief is evicted."""
        with self._lock:
            to_remove: set[str] = set()
            to_remove.update(self._outgoing.get(belief_id, set()))
            to_remove.update(self._incoming.get(belief_id, set()))
            removed = 0
            for eid in to_remove:
                if self._remove_unlocked(eid):
                    removed += 1
            return removed

    # -- Edge strength decay ------------------------------------------------

    def decay_strengths(self, days: float = 1.0) -> int:
        """Apply passive strength decay.  Returns count of edges removed."""
        factor = EDGE_STRENGTH_DECAY_PER_DAY ** days
        removed = 0
        with self._lock:
            to_remove: list[str] = []
            for eid, edge in self._edges.items():
                if edge.evidence_basis == "user_correction":
                    continue
                new_strength = edge.strength * factor
                if new_strength < EDGE_STRENGTH_MIN:
                    to_remove.append(eid)
                else:
                    self._edges[eid] = edge.with_strength(new_strength)
            for eid in to_remove:
                self._remove_unlocked(eid)
                removed += 1
        return removed

    # -- Stats --------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            by_type: dict[str, int] = {}
            by_basis: dict[str, int] = {}
            for edge in self._edges.values():
                by_type[edge.edge_type] = by_type.get(edge.edge_type, 0) + 1
                by_basis[edge.evidence_basis] = by_basis.get(edge.evidence_basis, 0) + 1

            involved_beliefs: set[str] = set()
            for edge in self._edges.values():
                involved_beliefs.add(edge.source_belief_id)
                involved_beliefs.add(edge.target_belief_id)

            return {
                "total_edges": len(self._edges),
                "total_created": self._total_created,
                "by_type": by_type,
                "by_basis": by_basis,
                "involved_belief_count": len(involved_beliefs),
                "recent_edges": [
                    {
                        "edge_id": e.edge_id,
                        "source": e.source_belief_id[:8],
                        "target": e.target_belief_id[:8],
                        "type": e.edge_type,
                        "strength": round(e.strength, 3),
                        "basis": e.evidence_basis,
                        "age_s": round(time.time() - e.created_at, 1),
                    }
                    for e in list(self._recent)[-10:]
                ],
            }

    # -- Persistence --------------------------------------------------------

    def _append_jsonl(self, edge: EvidenceEdge) -> None:
        if self._rehydrating:
            return
        try:
            os.makedirs(os.path.dirname(self._edges_path), exist_ok=True)
            with open(self._edges_path, "a") as f:
                f.write(json.dumps(edge.to_dict()) + "\n")
            self._jsonl_line_count += 1
        except Exception:
            logger.exception("Failed to append edge JSONL")

    def rehydrate(self) -> None:
        """Load edges from JSONL, applying duplicate merge."""
        with self._lock:
            self._edges.clear()
            self._outgoing.clear()
            self._incoming.clear()
            self._by_type.clear()
            self._dedup.clear()
            self._jsonl_line_count = 0

        if not os.path.exists(self._edges_path):
            return

        self._rehydrating = True
        count = 0
        try:
            with open(self._edges_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        edge = EvidenceEdge.from_dict(d)
                        self.add(edge)
                        count += 1
                    except Exception:
                        continue
        except Exception:
            logger.exception("Failed to rehydrate edges")
        finally:
            self._rehydrating = False

        with self._lock:
            self._jsonl_line_count = count
        logger.info("Rehydrated %d edge lines -> %d live edges from %s",
                     count, len(self._edges), self._edges_path)

    def compact(self) -> None:
        """Rewrite JSONL from live in-memory state, eliminating duplicates."""
        with self._lock:
            live_count = len(self._edges)
            if self._jsonl_line_count <= live_count * COMPACTION_RATIO_THRESHOLD:
                return

            try:
                os.makedirs(os.path.dirname(self._edges_path), exist_ok=True)
                tmp = self._edges_path + ".tmp"
                with open(tmp, "w") as f:
                    for edge in self._edges.values():
                        f.write(json.dumps(edge.to_dict()) + "\n")
                os.replace(tmp, self._edges_path)
                self._jsonl_line_count = live_count
                logger.info("Compacted belief edges: %d live edges", live_count)
            except Exception:
                logger.exception("Failed to compact edge JSONL")

    def needs_compaction(self) -> bool:
        with self._lock:
            return self._jsonl_line_count > len(self._edges) * COMPACTION_RATIO_THRESHOLD


def make_edge(
    source_belief_id: str,
    target_belief_id: str,
    edge_type: str,
    strength: float,
    provenance: str,
    evidence_basis: str,
) -> EvidenceEdge:
    """Factory helper for creating new edges with generated ID and timestamps."""
    now = time.time()
    return EvidenceEdge(
        edge_id=nanoid(size=12),
        source_belief_id=source_belief_id,
        target_belief_id=target_belief_id,
        edge_type=edge_type,
        strength=max(0.0, min(1.0, strength)),
        provenance=provenance,
        created_at=now,
        last_updated=now,
        evidence_basis=evidence_basis,
    )
