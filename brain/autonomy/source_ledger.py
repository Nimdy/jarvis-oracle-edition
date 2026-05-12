"""Source Usefulness Ledger — Phase 5.1b.

Per-source tracking of whether research-derived knowledge actually helped.
Every research source that produces memories gets a record. The record tracks:
  - how many memories were created
  - how many were retrieved in later conversations
  - how many retrievals had positive user signals
  - whether any candidate interventions were derived and promoted

Verdict system:
  pending                        — waiting for evidence (< min_age)
  provisional_useful             — early positive signal before final window
  useful                         — memories retrieved with positive outcomes
  interesting_but_non_actionable — stored but never retrieved
  redundant                      — duplicate of existing knowledge
  misleading                     — produced contradictions
  low_evidence                   — insufficient signal to judge
  failed_to_improve              — retrieved but no quality improvement

Wired back to policy_memory to boost/penalize research topics.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("autonomy.source_ledger")

_PERSISTENCE_PATH = os.path.expanduser("~/.jarvis/source_ledger.jsonl")
_MAX_RECORDS = 500
_MAX_FILE_BYTES = 10 * 1024 * 1024
_MIN_FINAL_VERDICT_AGE_S = 86400.0  # 24h for final verdict
_PROVISIONAL_WINDOW_S = 21600.0     # 6h for provisional signal

_VALID_VERDICTS = frozenset({
    "pending", "provisional_useful", "useful",
    "interesting_but_non_actionable", "redundant",
    "misleading", "low_evidence", "failed_to_improve",
})


@dataclass
class SourceUsefulnessRecord:
    source_id: str
    intent_id: str
    trigger_deficit: str = ""
    target_subsystem: str = ""
    memories_created: int = 0
    retrieved_count: int = 0
    useful_retrieval_count: int = 0
    supported_answers_count: int = 0
    intervention_count: int = 0
    intervention_promoted_count: int = 0
    verdict: str = "pending"
    created_at: float = field(default_factory=time.time)
    verdict_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceUsefulnessRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


class SourceUsefulnessLedger:
    """Tracks per-source usefulness for research-derived knowledge."""

    def __init__(self) -> None:
        self._records: dict[str, SourceUsefulnessRecord] = {}
        self._history: deque[SourceUsefulnessRecord] = deque(maxlen=_MAX_RECORDS)
        self._loaded = False

    def record_source(
        self,
        source_id: str,
        intent_id: str,
        trigger_deficit: str = "",
        target_subsystem: str = "",
        memories_created: int = 0,
    ) -> SourceUsefulnessRecord:
        """Register a new research source after knowledge integration."""
        rec = self._records.get(source_id)
        if rec:
            rec.memories_created += memories_created
            return rec

        rec = SourceUsefulnessRecord(
            source_id=source_id,
            intent_id=intent_id,
            trigger_deficit=trigger_deficit,
            target_subsystem=target_subsystem,
            memories_created=memories_created,
        )
        self._records[source_id] = rec
        self._history.append(rec)
        self._persist(rec)
        logger.info("Source registered: %s (intent=%s, memories=%d)",
                     source_id, intent_id, memories_created)
        return rec

    def record_retrieval(self, source_id: str, useful: bool) -> None:
        """Record that a source-lineage memory was retrieved in conversation."""
        rec = self._records.get(source_id)
        if not rec:
            return
        rec.retrieved_count += 1
        if useful:
            rec.useful_retrieval_count += 1

    def record_intervention(self, source_id: str, promoted: bool = False) -> None:
        """Record that a candidate intervention was derived from this source."""
        rec = self._records.get(source_id)
        if not rec:
            return
        rec.intervention_count += 1
        if promoted:
            rec.intervention_promoted_count += 1

    def get_provisional_verdict(self, source_id: str) -> str:
        """Early signal before the final 24h verdict window.

        Returns 'provisional_useful' if any retrieval with positive signal
        occurred within the first 6h. Otherwise returns current verdict.
        """
        rec = self._records.get(source_id)
        if not rec:
            return "pending"
        if rec.verdict not in ("pending", "provisional_useful"):
            return rec.verdict
        age = time.time() - rec.created_at
        if age < _PROVISIONAL_WINDOW_S and rec.useful_retrieval_count > 0:
            rec.verdict = "provisional_useful"
            return "provisional_useful"
        return rec.verdict

    def compute_verdicts(self, min_age_s: float = _MIN_FINAL_VERDICT_AGE_S) -> int:
        """Evaluate pending records older than min_age and assign final verdicts.

        Returns the number of verdicts assigned.
        """
        now = time.time()
        assigned = 0
        for rec in self._records.values():
            if rec.verdict not in ("pending", "provisional_useful"):
                continue
            if now - rec.created_at < min_age_s:
                continue

            verdict = self._compute_single_verdict(rec)
            rec.verdict = verdict
            rec.verdict_at = now
            assigned += 1
            logger.info("Verdict for %s: %s (retrieved=%d, useful=%d, interventions=%d)",
                        rec.source_id, verdict, rec.retrieved_count,
                        rec.useful_retrieval_count, rec.intervention_count)

        return assigned

    def _compute_single_verdict(self, rec: SourceUsefulnessRecord) -> str:
        if rec.intervention_promoted_count > 0:
            return "useful"

        if rec.useful_retrieval_count > 0:
            return "useful"

        if rec.retrieved_count > 0 and rec.useful_retrieval_count == 0:
            return "failed_to_improve"

        if rec.memories_created == 0:
            return "low_evidence"

        if rec.retrieved_count == 0:
            return "interesting_but_non_actionable"

        return "low_evidence"

    def get_topic_usefulness(self, tag_cluster: tuple[str, ...] | list[str]) -> float:
        """Compute aggregate usefulness score for a topic cluster.

        Used by policy_memory to boost/penalize research topics.
        Returns 0.0-1.0 where higher = more useful history.
        """
        if not tag_cluster:
            return 0.5
        tags = set(tag_cluster)
        matches = []
        for rec in self._records.values():
            if rec.verdict in ("pending", "provisional_useful"):
                continue
            intent_tags = set()
            if rec.trigger_deficit:
                intent_tags.add(rec.trigger_deficit)
            if rec.target_subsystem:
                intent_tags.add(rec.target_subsystem)
            if tags & intent_tags:
                matches.append(rec)

        if not matches:
            return 0.5

        useful = sum(1 for m in matches if m.verdict == "useful")
        total = len(matches)
        return useful / total if total > 0 else 0.5

    def get_stats(self) -> dict[str, Any]:
        by_verdict: dict[str, int] = {}
        for rec in self._records.values():
            by_verdict[rec.verdict] = by_verdict.get(rec.verdict, 0) + 1
        total_retrieved = sum(r.retrieved_count for r in self._records.values())
        total_useful = sum(r.useful_retrieval_count for r in self._records.values())
        return {
            "total_sources": len(self._records),
            "by_verdict": by_verdict,
            "total_retrievals": total_retrieved,
            "total_useful_retrievals": total_useful,
            "usefulness_rate": total_useful / total_retrieved if total_retrieved > 0 else 0.0,
        }

    def get_recent_records(self, limit: int = 20) -> list[dict[str, Any]]:
        records = list(self._history)[-limit:]
        return [r.to_dict() for r in reversed(records)]

    # -- persistence --------------------------------------------------------

    def _persist(self, rec: SourceUsefulnessRecord) -> None:
        try:
            os.makedirs(os.path.dirname(_PERSISTENCE_PATH), exist_ok=True)
            if os.path.exists(_PERSISTENCE_PATH):
                sz = os.path.getsize(_PERSISTENCE_PATH)
                if sz > _MAX_FILE_BYTES:
                    rotated = _PERSISTENCE_PATH + ".1"
                    if os.path.exists(rotated):
                        os.remove(rotated)
                    os.rename(_PERSISTENCE_PATH, rotated)
            with open(_PERSISTENCE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec.to_dict(), default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to persist source record: %s", exc)

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not os.path.exists(_PERSISTENCE_PATH):
            return
        try:
            with open(_PERSISTENCE_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        rec = SourceUsefulnessRecord.from_dict(d)
                        self._records[rec.source_id] = rec
                        self._history.append(rec)
                    except Exception:
                        continue
            logger.info("Loaded %d source usefulness records", len(self._records))
        except Exception as exc:
            logger.warning("Failed to load source ledger: %s", exc)


# Singleton
_instance: SourceUsefulnessLedger | None = None


def get_source_ledger() -> SourceUsefulnessLedger:
    global _instance
    if _instance is None:
        _instance = SourceUsefulnessLedger()
        _instance.load()
    return _instance
