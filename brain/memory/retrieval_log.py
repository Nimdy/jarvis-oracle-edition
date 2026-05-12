"""Memory retrieval telemetry for closed-loop conversational retrieval.

Captures the full retrieval pipeline per *conversation-scoped* query:
  1. log_retrieval() — called from _hybrid_search() with all candidates + features
  2. mark_injected() — called from _build_context() with memory IDs that made it into the prompt
  3. log_outcome() — called from conversation_handler with conversation result

Joined by event_id to produce (query, candidate_features, injected, outcome) triples
for training the MemoryRanker NN.

Important: background/internal semantic lookups must not enter this log. They do not
close the loop with injections, references, and outcomes, and would otherwise pollute
retrieval calibration and ranker telemetry windows.

Storage: append-only JSONL at ~/.jarvis/memory_retrieval_log.jsonl
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
LOG_PATH = JARVIS_DIR / "memory_retrieval_log.jsonl"
MAX_LOG_SIZE_MB = 10
_EVENT_BUFFER_CAP = 200
_MIN_LIFT_SAMPLE_EVENTS = 5
_ACK_QUERY_RE = re.compile(
    r"^(?:jarvis[\s,]+)?(?:thanks?|thank you|ok(?:ay)?|sure|got it|sounds good|"
    r"perfect|great|awesome|alright|all right|next time|check(?:point)?|help)"
    r"(?:[\s,]+jarvis)?[.!? ]*$",
    re.IGNORECASE,
)
_LOW_INFORMATION_QUERY_RE = re.compile(
    r"^(?:jarvis[\s,]+)?(?:day|date|today|tonight|morning|evening|afternoon|"
    r"birthday|bday|huh|hmm+|what|why|okay|ok|sure|right|check(?:point)?|help)"
    r"(?:[\s,]+jarvis)?[.!? ]*$",
    re.IGNORECASE,
)


def _is_reference_evaluable_event(
    event: RetrievalEvent,
    outcome: RetrievalOutcome | None,
) -> bool:
    """Return True only when a turn meaningfully tests memory-grounded articulation.

    We exclude acknowledgements and generic follow-up chatter because those turns
    do not require the assistant to surface retrieved memory in the reply text.
    """
    if outcome is None or outcome.outcome != "ok":
        return False
    if outcome.user_signal in ("follow_up", "positive"):
        return False
    query = (event.query_text or "").strip()
    if not query:
        return False
    if _ACK_QUERY_RE.match(query):
        return False
    if _LOW_INFORMATION_QUERY_RE.match(query):
        return False
    return True


def _tokenize_reference_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_']+", text.lower())


def _extract_number_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


@dataclass
class CandidateRecord:
    """Per-candidate feature snapshot captured during retrieval."""
    memory_id: str
    similarity: float
    recency_score: float
    weight: float
    memory_type: str
    tag_count: int
    association_count: int
    priority: int
    provenance_boost: float
    speaker_match: bool
    heuristic_score: float
    selected: bool
    injected: bool = False

    def to_feature_vector(self) -> list[float]:
        """12-dim feature vector matching MemoryRanker input spec."""
        return [
            self.similarity,
            self.recency_score,
            self.weight,
            self.heuristic_score,
            min(self.tag_count, 10) / 10.0,
            min(self.association_count, 10) / 10.0,
            self.priority / 1000.0,
            self.provenance_boost,
            1.0 if self.speaker_match else 0.0,
            0.0,  # is_core — filled by caller
            1.0 if self.memory_type == "conversation" else 0.0,
            1.0 if self.memory_type == "factual_knowledge" else 0.0,
        ]


@dataclass
class RetrievalEvent:
    """Full retrieval snapshot for a single query."""
    event_id: str
    conversation_id: str
    query_text: str
    candidates: list[CandidateRecord]
    selected_memory_ids: list[str]
    injected_memory_ids: list[str]
    timestamp: float
    ranker_used: bool = False


@dataclass
class RetrievalOutcome:
    """Conversation outcome tied to a retrieval event."""
    event_id: str
    conversation_id: str
    outcome: str
    latency_ms: float
    user_signal: str = ""
    timestamp: float = 0.0
    outcome_scope: str = "response_quality"


@dataclass
class TrainingPair:
    """Joined retrieval event + outcome ready for ranker training."""
    candidate: CandidateRecord
    outcome: str
    label: float
    features: list[float] = field(default_factory=list)


class MemoryRetrievalLog:
    """Thread-safe append-only log for memory retrieval events."""

    _instance: MemoryRetrievalLog | None = None

    def __init__(self, path: str | Path = "") -> None:
        self._path = Path(path) if path else LOG_PATH
        self._lock = threading.Lock()
        self._initialized = False
        self._total_events = 0
        self._total_outcomes = 0
        self._recent_events: OrderedDict[str, RetrievalEvent] = OrderedDict()
        self._recent_outcomes: OrderedDict[str, RetrievalOutcome] = OrderedDict()
        self._recent_references: OrderedDict[str, set[str]] = OrderedDict()
        self._outcome_stats: dict[str, int] = {"ok": 0, "error": 0, "barge_in": 0}
        self._boot_ts: float = time.time()
        self._rehydrated: bool = False
        self._rehydrated_count: int = 0
        self._reference_without_injection_count: int = 0
        self._feedback_stats: dict[str, int] = {
            "positive_applied": 0,
            "negative_applied": 0,
            "skipped_missing_event": 0,
            "skipped_no_injection": 0,
        }

    @classmethod
    def get_instance(cls) -> MemoryRetrievalLog:
        if cls._instance is None:
            cls._instance = MemoryRetrievalLog()
        return cls._instance

    def init(self) -> None:
        if self._initialized:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    def rehydrate(self, max_events: int = 200) -> int:
        """Replay last N events from JSONL into in-memory buffers for warm-start.

        Called once at boot. Rebuilds retrieval events, injections, and outcomes
        so eval_metrics and training pairs are immediately meaningful.
        Returns count of events rehydrated.
        """
        if self._rehydrated or not self._path.exists():
            return 0

        try:
            lines = self._path.read_text().strip().split("\n")
            tail = lines[-min(len(lines), max_events * 5):]
        except Exception:
            return 0

        events_by_id: dict[str, RetrievalEvent] = {}
        injections: dict[str, list[str]] = {}
        references: dict[str, set[str]] = {}
        outcomes: list[dict] = []

        for line in tail:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rtype = rec.get("type", "")
            if rtype == "retrieval":
                eid = rec.get("event_id", "")
                cid = rec.get("conversation_id", "")
                if not eid:
                    continue
                if not cid:
                    # Historical background/internal searches used the same log file,
                    # but they are not valid retrieval-training examples.
                    continue
                candidates = []
                for c in rec.get("candidates", []):
                    candidates.append(CandidateRecord(
                        memory_id=c.get("mid", ""),
                        similarity=c.get("sim", 0.0),
                        recency_score=c.get("rec", 0.0),
                        weight=c.get("w", 0.0),
                        memory_type=c.get("type", ""),
                        tag_count=c.get("tags", 0),
                        association_count=c.get("assoc", 0),
                        priority=c.get("pri", 0),
                        provenance_boost=c.get("prov", 0.0),
                        speaker_match=c.get("spk", False),
                        heuristic_score=c.get("hs", 0.0),
                        selected=c.get("sel", False),
                    ))
                events_by_id[eid] = RetrievalEvent(
                    event_id=eid,
                    conversation_id=cid,
                    query_text=rec.get("query", ""),
                    candidates=candidates,
                    selected_memory_ids=[c.memory_id for c in candidates if c.selected],
                    injected_memory_ids=[],
                    timestamp=rec.get("t", 0.0),
                    ranker_used=rec.get("ranker_used", False),
                )
            elif rtype == "injection":
                eid = rec.get("event_id", "")
                if eid:
                    injections[eid] = rec.get("injected_memory_ids", [])
            elif rtype == "reference":
                eid = rec.get("event_id", "")
                ref_ids = rec.get("referenced_memory_ids", [])
                if eid and ref_ids:
                    references.setdefault(eid, set()).update(ref_ids)
            elif rtype == "outcome":
                outcomes.append(rec)

        for eid, inj_ids in injections.items():
            ev = events_by_id.get(eid)
            if ev:
                ev.injected_memory_ids = inj_ids

        count = 0
        with self._lock:
            for ev in list(events_by_id.values())[-max_events:]:
                self._recent_events[ev.event_id] = ev
                if len(self._recent_events) > _EVENT_BUFFER_CAP:
                    self._recent_events.popitem(last=False)
                self._total_events += 1
                count += 1

            for oc_rec in outcomes:
                eid = oc_rec.get("event_id", "")
                if eid not in events_by_id:
                    continue
                oc = RetrievalOutcome(
                    event_id=eid,
                    conversation_id=oc_rec.get("conversation_id", ""),
                    outcome=oc_rec.get("outcome", ""),
                    latency_ms=oc_rec.get("latency_ms", 0.0),
                    user_signal=oc_rec.get("user_signal", ""),
                    timestamp=oc_rec.get("t", 0.0),
                )
                self._recent_outcomes[eid] = oc
                if len(self._recent_outcomes) > _EVENT_BUFFER_CAP:
                    self._recent_outcomes.popitem(last=False)
                self._total_outcomes += 1
                outcome = oc.outcome
                if outcome in self._outcome_stats:
                    self._outcome_stats[outcome] += 1

            for eid, ref_set in references.items():
                if eid in events_by_id:
                    self._recent_references[eid] = ref_set
            while len(self._recent_references) > _EVENT_BUFFER_CAP:
                self._recent_references.popitem(last=False)

        self._rehydrated = True
        self._rehydrated_count = count
        if count > 0:
            logger.info("Rehydrated %d retrieval events from JSONL warm-start", count)
        return count

    def log_retrieval(
        self,
        conversation_id: str,
        query_text: str,
        candidates: list[CandidateRecord],
        selected_memory_ids: list[str],
        ranker_used: bool = False,
    ) -> str:
        """Log a conversational retrieval event and return event_id.

        Background/internal semantic lookups are intentionally excluded because they
        never produce the closed-loop outcome signals used for training and calibration.
        """
        if not conversation_id:
            return ""
        event_id = f"ret_{uuid.uuid4().hex[:12]}"
        event = RetrievalEvent(
            event_id=event_id,
            conversation_id=conversation_id,
            query_text=query_text[:200],
            candidates=candidates,
            selected_memory_ids=selected_memory_ids,
            injected_memory_ids=[],
            timestamp=time.time(),
            ranker_used=ranker_used,
        )

        with self._lock:
            self._recent_events[event_id] = event
            self._recent_events.move_to_end(event_id)
            if len(self._recent_events) > _EVENT_BUFFER_CAP:
                self._recent_events.popitem(last=False)
            self._total_events += 1

        self._append({
            "type": "retrieval",
            "event_id": event_id,
            "conversation_id": conversation_id,
            "query": query_text[:200],
            "candidate_count": len(candidates),
            "selected_count": len(selected_memory_ids),
            "ranker_used": ranker_used,
            "candidates": [
                {
                    "mid": c.memory_id,
                    "sim": round(c.similarity, 4),
                    "rec": round(c.recency_score, 4),
                    "w": round(c.weight, 4),
                    "type": c.memory_type,
                    "tags": c.tag_count,
                    "assoc": c.association_count,
                    "pri": c.priority,
                    "prov": round(c.provenance_boost, 3),
                    "spk": c.speaker_match,
                    "hs": round(c.heuristic_score, 4),
                    "sel": c.selected,
                }
                for c in candidates
            ],
            "t": round(event.timestamp, 3),
        })

        try:
            from memory.storage import memory_storage
            for mid in selected_memory_ids:
                memory_storage.record_access(mid)
        except Exception:
            pass

        return event_id

    def mark_injected(self, event_id: str, injected_memory_ids: list[str]) -> None:
        """Called after context building to record which memories reached the prompt."""
        with self._lock:
            event = self._recent_events.get(event_id)
            if event:
                event.injected_memory_ids = injected_memory_ids

        if injected_memory_ids:
            self._append({
                "type": "injection",
                "event_id": event_id,
                "injected_memory_ids": injected_memory_ids,
                "t": round(time.time(), 3),
            })
            self._notify_source_ledger(injected_memory_ids)

    def _notify_source_ledger(self, memory_ids: list[str]) -> None:
        """Note that research-derived memories were injected.

        Retrieval counting is handled exclusively by ``log_outcome()`` which
        has the actual user signal (positive/negative/neutral). Recording here
        too would double-count every injection.
        """

    def record_references(self, event_id: str, referenced_memory_ids: list[str]) -> None:
        """Record which memories were actually referenced in the response text."""
        if not event_id or not referenced_memory_ids:
            return
        with self._lock:
            self._recent_references.setdefault(event_id, set()).update(referenced_memory_ids)
            self._recent_references.move_to_end(event_id)
            if len(self._recent_references) > _EVENT_BUFFER_CAP:
                self._recent_references.popitem(last=False)

    def log_outcome(
        self,
        conversation_id: str,
        outcome: str,
        latency_ms: float = 0.0,
        user_signal: str = "",
        outcome_scope: str = "response_quality",
    ) -> None:
        """Log the conversation outcome for correlation with retrieval events.

        outcome_scope prevents cross-layer label pollution: outcomes scoped to
        autonomy_policy should not penalize memory provenance in ranker training.
        """
        event_id = self._find_event_by_conversation(conversation_id)
        if not event_id:
            return

        oc = RetrievalOutcome(
            event_id=event_id,
            conversation_id=conversation_id,
            outcome=outcome,
            latency_ms=latency_ms,
            user_signal=user_signal,
            timestamp=time.time(),
            outcome_scope=outcome_scope,
        )

        with self._lock:
            self._recent_outcomes[event_id] = oc
            if len(self._recent_outcomes) > _EVENT_BUFFER_CAP:
                self._recent_outcomes.popitem(last=False)
            self._total_outcomes += 1
            if outcome in self._outcome_stats:
                self._outcome_stats[outcome] += 1

        self._append({
            "type": "outcome",
            "event_id": event_id,
            "conversation_id": conversation_id,
            "outcome": outcome,
            "latency_ms": round(latency_ms, 1),
            "user_signal": user_signal,
            "outcome_scope": outcome_scope,
            "t": round(oc.timestamp, 3),
        })

        self.apply_outcome_feedback(
            conversation_id=conversation_id,
            outcome=outcome,
            user_signal=user_signal,
            outcome_scope=outcome_scope,
        )

        # Phase 5.1b: upgrade source ledger with usefulness signal on outcome
        try:
            event_for_ledger = self.get_latest_event_for_conversation(conversation_id)
            if event_for_ledger and event_for_ledger.injected_memory_ids:
                from memory.storage import memory_storage
                from autonomy.source_ledger import get_source_ledger
                ledger = get_source_ledger()
                useful = user_signal in ("positive", "follow_up")
                for mid in event_for_ledger.injected_memory_ids:
                    mem = memory_storage.get(mid)
                    if not mem:
                        continue
                    payload = getattr(mem, "payload", None)
                    if isinstance(payload, dict) and "source_lineage" in payload:
                        sid = payload["source_lineage"].get("source_id", "")
                        if sid:
                            ledger.record_retrieval(sid, useful=useful)
        except Exception as exc:
            logger.debug("source_ledger outcome update failed: %s", exc)

    def apply_outcome_feedback(
        self,
        conversation_id: str,
        outcome: str,
        user_signal: str = "",
        outcome_scope: str = "response_quality",
    ) -> dict[str, int]:
        """Apply light weight updates to injected memories from retrieval outcomes.

        This closes the retrieval learning loop without replacing the ranker labels.
        We only touch injected memories from response-quality style outcomes.
        """
        if outcome_scope not in ("response_quality", "retrieval_selection", "general"):
            return {"reinforced": 0, "downweighted": 0}

        event = self.get_latest_event_for_conversation(conversation_id)
        if not event:
            self._feedback_stats["skipped_missing_event"] += 1
            return {"reinforced": 0, "downweighted": 0}

        injected_ids = set(event.injected_memory_ids)
        if not injected_ids:
            self._feedback_stats["skipped_no_injection"] += 1
            return {"reinforced": 0, "downweighted": 0}

        with self._lock:
            referenced_ids = set(self._recent_references.get(event.event_id, set()))

        reinforced = 0
        downweighted = 0
        try:
            from memory.storage import memory_storage
        except Exception:
            return {"reinforced": 0, "downweighted": 0}

        for candidate in event.candidates:
            if candidate.memory_id not in injected_ids:
                continue
            sim = max(0.0, min(1.0, candidate.similarity))
            was_referenced = candidate.memory_id in referenced_ids

            if outcome == "ok" and user_signal not in ("negative", "correction"):
                if was_referenced or user_signal == "positive":
                    boost = 0.008 + (sim * (0.022 if was_referenced else 0.012))
                    if user_signal == "positive":
                        boost += 0.01
                    if memory_storage.adjust_weight(candidate.memory_id, boost):
                        reinforced += 1
            elif outcome in ("error", "barge_in", "timeout", "cancelled") or user_signal == "negative":
                penalty = 0.01 + ((1.0 - sim) * 0.015)
                if user_signal == "negative":
                    penalty += 0.01
                if memory_storage.adjust_weight(candidate.memory_id, -penalty):
                    downweighted += 1

        if reinforced:
            self._feedback_stats["positive_applied"] += reinforced
        if downweighted:
            self._feedback_stats["negative_applied"] += downweighted
        return {"reinforced": reinforced, "downweighted": downweighted}

    def get_latest_event_for_conversation(self, conversation_id: str) -> RetrievalEvent | None:
        with self._lock:
            for event in reversed(list(self._recent_events.values())):
                if event.conversation_id == conversation_id:
                    return event
        return None

    def get_training_pairs(self, limit: int = 500) -> list[TrainingPair]:
        """Join recent events with outcomes to produce labeled training pairs.

        Label schema (three-tier with reference awareness):
          Base labels:
            injected + referenced + ok  -> 1.0  (strong positive, LLM actually used it)
            injected + NOT referenced + ok -> 0.8  (LLM ignored this memory)
            selected, not injected, ok  -> 0.5  (weak positive)
            injected + error/barge_in   -> 0.3  (weak negative)
            not selected                -> 0.0  (negative)

          User signal modifiers:
            "positive" on selected+ok:  +0.1
            "negative" on injected+referenced: floor at 0.8 (protected — LLM used it)
            "negative" on injected+not-referenced: -0.2 (bad retrieval)
            "follow_up" on not-selected: +0.05

          Provenance tie-breaker (max +/-0.05):
            positive outcome + high provenance (boost >= 0.06): +0.05
            negative outcome + low provenance (boost < 0.04):   -0.05
        """
        pairs: list[TrainingPair] = []

        with self._lock:
            events = list(self._recent_events.values())
            outcomes = dict(self._recent_outcomes)
            references = dict(self._recent_references)

        for event in events:
            if not event.conversation_id:
                continue
            oc = outcomes.get(event.event_id)
            if not oc:
                continue

            injected_set = set(event.injected_memory_ids)
            selected_set = set(event.selected_memory_ids)
            referenced_set = references.get(event.event_id, set())
            is_ok = oc.outcome == "ok"
            sig = oc.user_signal

            for candidate in event.candidates:
                mid = candidate.memory_id
                is_injected = mid in injected_set
                is_selected = mid in selected_set
                is_referenced = mid in referenced_set

                # Case 3: referenced but not injected — don't credit ranker
                if is_referenced and not is_injected:
                    self._reference_without_injection_count += 1
                    # Still include as a normal non-selected candidate
                    label = 0.0 if not is_selected else 0.5
                elif is_injected and is_ok:
                    label = 1.0 if is_referenced else 0.8
                elif is_selected and not is_injected and is_ok:
                    label = 0.5
                elif is_injected and not is_ok:
                    label = 0.3
                elif not is_selected:
                    label = 0.0
                else:
                    label = 0.2

                # User signal modifiers
                if sig == "positive" and is_selected and not is_injected:
                    label = min(1.0, label + 0.1)
                elif sig == "negative" and is_injected:
                    if is_referenced:
                        label = max(0.8, label)
                    else:
                        label = max(0.0, label - 0.2)
                elif sig == "follow_up" and not is_selected:
                    label = min(1.0, label + 0.05)

                # Provenance tie-breaker (intentionally tiny).
                # Only apply when outcome_scope is relevant to memory quality —
                # autonomy_policy regressions should never penalize provenance.
                _scope = oc.outcome_scope if hasattr(oc, 'outcome_scope') else "response_quality"
                _scope_relevant = _scope in ("response_quality", "retrieval_selection", "general")
                prov_boost = candidate.provenance_boost
                if _scope_relevant:
                    if is_ok and sig != "negative" and prov_boost >= 0.06:
                        label = min(1.0, label + 0.05)
                    elif not is_ok and prov_boost < 0.04:
                        label = max(0.0, label - 0.05)

                features = candidate.to_feature_vector()
                pairs.append(TrainingPair(
                    candidate=candidate,
                    outcome=oc.outcome,
                    label=round(label, 4),
                    features=features,
                ))

        return pairs[-limit:]

    def get_retrieval_success_rate(self, window: int = 100) -> float:
        """Rolling success rate: % of events with outcome=ok and >=1 injection."""
        with self._lock:
            events = [e for e in self._recent_events.values() if e.conversation_id][-window:]
            outcomes = dict(self._recent_outcomes)

        if not events:
            return 0.0

        successes = 0
        total = 0
        for event in events:
            oc = outcomes.get(event.event_id)
            if not oc:
                continue
            total += 1
            if oc.outcome == "ok" and event.injected_memory_ids:
                successes += 1

        return successes / total if total > 0 else 0.0

    def get_eval_metrics(self, window: int = 100) -> dict[str, Any]:
        """Compute ranker/heuristic split, lift, coverage, and reference metrics.

        Returns None for rates/lift when there's insufficient data to avoid
        misleading dashboard signals (e.g., lift=-1 when ranker has zero events).
        All computation happens outside the lock to avoid reentrancy.
        """
        with self._lock:
            events = [e for e in self._recent_events.values() if e.conversation_id][-window:]
            outcomes = dict(self._recent_outcomes)
            references = dict(self._recent_references)
            ref_no_inj_count = self._reference_without_injection_count

        ranker_events = [e for e in events if e.ranker_used]
        heuristic_events = [e for e in events if not e.ranker_used]

        def _success_rate(evts: list) -> float | None:
            total = 0
            ok = 0
            for ev in evts:
                oc = outcomes.get(ev.event_id)
                if not oc:
                    continue
                total += 1
                if oc.outcome == "ok" and ev.injected_memory_ids:
                    ok += 1
            return ok / total if total > 0 else None

        ranker_rate = _success_rate(ranker_events)
        heuristic_rate = _success_rate(heuristic_events)
        overall_rate = _success_rate(events)

        total_with_outcome = sum(1 for e in events if outcomes.get(e.event_id))
        ranker_with_outcome = sum(1 for e in ranker_events if outcomes.get(e.event_id))
        coverage = round(ranker_with_outcome / total_with_outcome, 4) if total_with_outcome > 0 else None
        heuristic_with_outcome = sum(1 for e in heuristic_events if outcomes.get(e.event_id))

        lift: float | None = None
        if (
            ranker_rate is not None
            and heuristic_rate is not None
            and heuristic_rate > 0
            and ranker_with_outcome >= _MIN_LIFT_SAMPLE_EVENTS
            and heuristic_with_outcome >= _MIN_LIFT_SAMPLE_EVENTS
        ):
            lift = round((ranker_rate - heuristic_rate) / heuristic_rate, 4)

        training_pairs = len(self.get_training_pairs())
        _MIN_TRAINING_PAIRS = 50

        # Reference match rate: fraction of injected memories that were referenced
        total_injected = 0
        total_referenced = 0
        reference_eval_events = 0
        reference_eval_excluded_count = 0
        prov_weighted_ok = 0.0
        prov_weighted_total = 0.0
        for ev in events:
            oc = outcomes.get(ev.event_id)
            if not oc:
                continue
            if _is_reference_evaluable_event(ev, oc):
                reference_eval_events += 1
                ref_set = references.get(ev.event_id, set())
                for mid in ev.injected_memory_ids:
                    total_injected += 1
                    if mid in ref_set:
                        total_referenced += 1
            else:
                reference_eval_excluded_count += 1
            # Provenance-weighted success rate
            for c in ev.candidates:
                if c.memory_id in set(ev.injected_memory_ids):
                    w = max(c.provenance_boost, 0.01)
                    prov_weighted_total += w
                    if oc.outcome == "ok":
                        prov_weighted_ok += w

        ref_match_rate = round(total_referenced / total_injected, 4) if total_injected > 0 else None
        prov_success = round(prov_weighted_ok / prov_weighted_total, 4) if prov_weighted_total > 0 else None

        return {
            "overall_success_rate": round(overall_rate, 4) if overall_rate is not None else None,
            "ranker_success_rate": round(ranker_rate, 4) if ranker_rate is not None else None,
            "heuristic_success_rate": round(heuristic_rate, 4) if heuristic_rate is not None else None,
            "lift": lift,
            "coverage": coverage,
            "ranker_events": len(ranker_events),
            "heuristic_events": len(heuristic_events),
            "total_with_outcome": total_with_outcome,
            "min_lift_sample_events": _MIN_LIFT_SAMPLE_EVENTS,
            "training_pairs_available": training_pairs,
            "training_pairs_min_required": _MIN_TRAINING_PAIRS,
            "training_ready": training_pairs >= _MIN_TRAINING_PAIRS,
            "reference_match_rate": ref_match_rate,
            "reference_eval_events": reference_eval_events,
            "reference_eval_injected_count": total_injected,
            "reference_eval_excluded_count": reference_eval_excluded_count,
            "reference_without_injection_count": ref_no_inj_count,
            "provenance_weighted_success_rate": prov_success,
        }

    def get_stats(self) -> dict[str, Any]:
        """Snapshot of log state for dashboard. Lock-safe (no nested calls)."""
        with self._lock:
            events = list(self._recent_events.values())
            outcomes = dict(self._recent_outcomes)
            stats = {
                "total_events": self._total_events,
                "total_outcomes": self._total_outcomes,
                "buffered_events": len(self._recent_events),
                "outcome_stats": dict(self._outcome_stats),
                "feedback_stats": dict(self._feedback_stats),
                "log_exists": self._path.exists(),
                "log_size_kb": round(self._path.stat().st_size / 1024, 1) if self._path.exists() else 0,
                "boot_ts": round(self._boot_ts, 3),
                "rehydrated": self._rehydrated,
                "rehydrated_count": self._rehydrated_count,
            }

        window_events = events[-100:]
        successes = 0
        total = 0
        for event in window_events:
            oc = outcomes.get(event.event_id)
            if not oc:
                continue
            total += 1
            if oc.outcome == "ok" and event.injected_memory_ids:
                successes += 1
        stats["retrieval_success_rate"] = round(successes / total if total > 0 else 0.0, 3)
        return stats

    def get_last_conversation_id(self) -> str:
        """Return the conversation_id of the most recent retrieval event."""
        with self._lock:
            if self._recent_events:
                return list(self._recent_events.values())[-1].conversation_id
        return ""

    def get_new_event_count(self) -> int:
        """How many events have been logged (for training gate checks)."""
        return self._total_events

    def _find_event_by_conversation(self, conversation_id: str) -> str | None:
        with self._lock:
            for event in reversed(list(self._recent_events.values())):
                if event.conversation_id == conversation_id:
                    return event.event_id
        return None

    def _append(self, entry: dict[str, Any]) -> None:
        if not self._initialized:
            self.init()
        with self._lock:
            try:
                if self._path.exists() and self._path.stat().st_size > MAX_LOG_SIZE_MB * 1024 * 1024:
                    self._rotate()
                with open(self._path, "a") as f:
                    f.write(json.dumps(entry, separators=(",", ":"), default=str) + "\n")
            except Exception as exc:
                logger.debug("Retrieval log write failed: %s", exc)

    def _rotate(self) -> None:
        try:
            lines = self._path.read_text().strip().split("\n")
            keep = lines[len(lines) // 2:]
            self._path.write_text("\n".join(keep) + "\n")
            logger.info("Rotated retrieval log: %d -> %d entries", len(lines), len(keep))
        except Exception:
            pass


def detect_memory_references(
    response_text: str,
    injected_memories: list[dict],
    min_overlap_words: int = 3,
) -> list[str]:
    """Detect which injected memories were actually used in the LLM response.

    Uses the formatted text (what the LLM actually saw) when available, falling
    back to raw payload. Two detection paths:
      1. Sliding n-gram match (min_overlap_words consecutive words)
      2. Bag-of-words overlap fallback (content words only)
    """
    if not response_text or not injected_memories:
        return []

    resp_lower = response_text.lower()
    resp_words = _tokenize_reference_text(response_text)
    resp_word_set = set(resp_words)
    resp_numbers = _extract_number_tokens(response_text)
    _stopwords = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "and", "but", "or", "not", "no",
        "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
        "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    })
    resp_content_words = resp_word_set - _stopwords
    referenced: list[str] = []

    for mem in injected_memories:
        mid = mem.get("id", "")
        if not mid:
            continue

        text = mem.get("formatted", "")
        if not text:
            payload = mem.get("payload", "")
            if isinstance(payload, dict):
                user_msg = payload.get("user_message", "")
                response = payload.get("response", "")
                if user_msg and response:
                    text = f"{user_msg} {response}"
                elif user_msg:
                    text = user_msg
                else:
                    text = " ".join(str(v) for v in payload.values() if isinstance(v, str))
            else:
                text = str(payload) if payload else ""
        if not text:
            continue

        mem_lower = text.lower()
        mem_words = _tokenize_reference_text(text)
        effective_overlap = min(min_overlap_words, len(mem_words))
        if effective_overlap < 2:
            continue

        matched = False
        for i in range(len(mem_words) - effective_overlap + 1):
            ngram = " ".join(mem_words[i:i + effective_overlap])
            if ngram in resp_lower:
                matched = True
                break

        if not matched:
            mem_content_words = set(mem_words) - _stopwords
            if mem_content_words:
                overlap = resp_content_words & mem_content_words
                ratio = len(overlap) / len(mem_content_words)
                if ratio >= 0.25 or len(overlap) >= 5:
                    matched = True

        if not matched:
            mem_numbers = _extract_number_tokens(text)
            shared_numbers = mem_numbers & resp_numbers
            if shared_numbers:
                mem_content_words = set(mem_words) - _stopwords
                overlap = resp_content_words & mem_content_words
                if len(shared_numbers) >= 2 or len(overlap) >= 2:
                    matched = True

        if matched:
            referenced.append(mid)

    return referenced


memory_retrieval_log = MemoryRetrievalLog.get_instance()
