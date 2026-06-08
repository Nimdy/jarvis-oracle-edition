"""Grounding Queue — durable async batched validation queue (SPARK_DESIGN §6, §8 P4).

The leverage point named in SPARK §6: an **async Grounding Queue** that converts
external validation from a synchronous-interrupt to an asynchronous-review.
Pending validation questions are ranked by ``tension × graph-leverage ×
staleness``; each carries the belief, its provenance, current confidence, and
which way it would move. The operator answers at leisure (typed, batched),
feeding the external-validation path — never auto-fired.

This module is the **durable batch** required by the SPARK §6 input-starvation
degradation: when the operator is absent AND no Pi signal AND web is exhausted,
the grounding loop must NOT manufacture validation. Pending questions instead
accumulate here and survive restart, to be reviewed at next operator presence.

CHANNEL-SELECTION ROUTER (SPARK §6): :func:`route_channel` picks the cheapest
external validator that *can* validate a facet (identity/self → operator;
scene/physical → Pi senses; factual/external → web). Never ask the operator what
the web can answer; never ask the web what only the operator knows.

INPUT-STARVATION STATE (SPARK §6/§10) is first-class, never hidden:
:class:`StarvationState` is surfaced on the dashboard. When starved, grounding
self-floors to ``local_only`` and high-tension ungrounded beliefs are
WEIGHT-REDUCED-NEVER-DELETED (the "memories always write" invariant) — that
quarantine is recorded here as a *pending recommendation*, executed only when the
quarantine immune system / active gate is wired (P5); advisory only flags it.

HONESTY GUARDRAILS (SPARK §7) enforced here:
  * VIEW-ONLY EPISTEMICS: this module reads belief provenance/confidence but
    NEVER mutates the frozen ``BeliefRecord`` or writes ``beliefs.jsonl``. The
    operator's answer is recorded as an *external-validation outcome* (a separate
    durable log); the actual belief mutation is the active-tier closure (P5).
  * Answering is OPERATOR-GATED: nothing here is auto-fired. ``answer()`` runs
    only on an explicit operator POST.
  * Being corrected counts as success: a "refuted" answer still records
    ``grounded=True`` (the belief moved from inferred to externally-anchored).

Backward-compatible: new fields default to safe values; ``to_dict`` emits them
and the JSONL reader tolerates their absence.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

QUEUE_PATH = os.path.join(os.path.expanduser("~"), ".jarvis", "grounding_queue.json")

# Max durable pending questions (oldest-lowest-rank evicted past this).
MAX_PENDING = 200
# A pending question expires (stale) after this many seconds without an answer.
QUESTION_TTL_S = 14 * 24 * 3600.0  # 14 days — a leisure-review queue, not urgent.
# Staleness reaches its ranking ceiling at this age (older = no extra weight).
STALENESS_CEILING_S = 7 * 24 * 3600.0


# ---------------------------------------------------------------------------
# Channel-selection router (SPARK §6).
# ---------------------------------------------------------------------------

# Cheapest external validator that CAN validate each facet. Mirrors
# provenance_scorer._FACET_CHANNEL but lives here too so the queue is
# self-contained (the router is the queue's, not the scorer's, responsibility).
_FACET_CHANNEL: dict[str, str] = {
    "identity": "operator",
    "self": "operator",
    "scene": "pi_senses",
    "factual": "web",
}

# Channels in trust/cost order (SPARK §6): operator highest trust/lowest BW,
# Pi senses narrow scope, web high bandwidth/no operator cost.
CHANNELS = ("operator", "pi_senses", "web")


def route_channel(facet: str, *, web_exhausted: bool = False) -> str:
    """Pick the cheapest validating channel for a facet (SPARK §6).

    ``web_exhausted`` degrades a web-routed facet to ``local_only`` (the honest
    starvation floor — never manufacture validation).
    """
    channel = _FACET_CHANNEL.get(facet, "web")
    if channel == "web" and web_exhausted:
        return "local_only"
    return channel


# ---------------------------------------------------------------------------
# Input-starvation state (SPARK §6/§10 — first-class, never hidden).
# ---------------------------------------------------------------------------

@dataclass
class StarvationState:
    """Honest degradation state. Starved when no validator can be reached.

    ``self_floored_local_only`` means grounding has degraded to internal-
    coherence-only (it does NOT manufacture validation). This is the negative
    control of SPARK §9: when the operator is absent, grounding throughput must
    DROP (input-starved), not climb.
    """

    operator_present: bool = False
    pi_signal_available: bool = False
    web_exhausted: bool = False
    starved: bool = False
    self_floored_local_only: bool = False
    pending_batch_size: int = 0
    last_operator_seen_ts: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_present": self.operator_present,
            "pi_signal_available": self.pi_signal_available,
            "web_exhausted": self.web_exhausted,
            "starved": self.starved,
            "self_floored_local_only": self.self_floored_local_only,
            "pending_batch_size": self.pending_batch_size,
            "last_operator_seen_ts": self.last_operator_seen_ts,
            "note": self.note,
        }


def assess_starvation(
    *,
    operator_present: bool,
    pi_signal_available: bool,
    web_exhausted: bool,
    pending_batch_size: int = 0,
    last_operator_seen_ts: float = 0.0,
) -> StarvationState:
    """Compute the starvation state (SPARK §6 degradation contract).

    Starved iff the operator is absent AND no Pi signal AND web is exhausted —
    the only honest condition under which grounding self-floors to local_only.
    """
    starved = (not operator_present) and (not pi_signal_available) and web_exhausted
    note = ""
    if starved:
        note = (
            "input-starved: operator absent, no Pi signal, web exhausted — "
            "grounding self-floored to local_only; pending questions batched "
            "for next operator presence (no validation manufactured)"
        )
    elif not operator_present and not pi_signal_available:
        note = (
            "operator absent, no Pi signal — web channel still open for factual "
            "facets; identity/self/scene questions batched for review"
        )
    return StarvationState(
        operator_present=operator_present,
        pi_signal_available=pi_signal_available,
        web_exhausted=web_exhausted,
        starved=starved,
        self_floored_local_only=starved,
        pending_batch_size=pending_batch_size,
        last_operator_seen_ts=last_operator_seen_ts,
        note=note,
    )


# ---------------------------------------------------------------------------
# Pending question record.
# ---------------------------------------------------------------------------

@dataclass
class PendingGroundingQuestion:
    """One durable, ranked validation question awaiting operator review.

    ``rank`` = tension × graph_leverage × staleness (SPARK §6). ``moves_toward``
    names which way the belief would move ("confirm → +confidence" /
    "refute → quarantine"), so the operator sees the consequence before answering.
    """

    question_id: str
    belief_id: str
    question_text: str
    facet: str = "factual"
    channel: str = "web"
    provenance: str = ""
    rendered_claim: str = ""
    grounding_tension: float = 0.0
    graph_leverage: float = 0.0
    base_confidence: float = 0.0
    effective_confidence: float = 0.0
    created_at: float = field(default_factory=time.time)
    asked_synchronously: bool = False
    answered: bool = False
    answered_at: float = 0.0
    answer_text: str = ""
    external_validation: str = ""  # "confirmed" | "refuted" | "" (unanswered)
    grounded: bool = False

    def staleness(self, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        age = max(0.0, now - self.created_at)
        return min(1.0, age / STALENESS_CEILING_S) if STALENESS_CEILING_S > 0 else 0.0

    def rank(self, now: float | None = None) -> float:
        """tension × graph_leverage × staleness (SPARK §6).

        Staleness is floored at a small positive so a fresh hub belief is not
        ranked at zero on creation (graph-leverage still dominates early).
        """
        stale = max(0.05, self.staleness(now))
        lev = max(0.0, min(1.0, self.graph_leverage))
        ten = max(0.0, min(1.0, self.grounding_tension))
        return round(ten * lev * stale, 6)

    def moves_toward(self) -> str:
        return (
            "confirm → +confidence/+salience · refute → quarantine "
            "(weight-reduce, never delete) — either way the belief gains "
            "external anchoring (grounded=True)"
        )

    def to_dict(self, now: float | None = None) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "belief_id": self.belief_id,
            "question_text": self.question_text,
            "facet": self.facet,
            "channel": self.channel,
            "provenance": self.provenance,
            "rendered_claim": self.rendered_claim[:200],
            "grounding_tension": round(self.grounding_tension, 4),
            "graph_leverage": round(self.graph_leverage, 4),
            "base_confidence": round(self.base_confidence, 4),
            "effective_confidence": round(self.effective_confidence, 4),
            "created_at": self.created_at,
            "staleness": round(self.staleness(now), 4),
            "rank": self.rank(now),
            "asked_synchronously": self.asked_synchronously,
            "answered": self.answered,
            "answered_at": self.answered_at,
            "answer_text": self.answer_text[:500],
            "external_validation": self.external_validation,
            "grounded": self.grounded,
            "moves_toward": self.moves_toward(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PendingGroundingQuestion":
        return cls(
            question_id=str(d.get("question_id", "")),
            belief_id=str(d.get("belief_id", "")),
            question_text=str(d.get("question_text", "")),
            facet=str(d.get("facet", "factual")),
            channel=str(d.get("channel", "web")),
            provenance=str(d.get("provenance", "")),
            rendered_claim=str(d.get("rendered_claim", "")),
            grounding_tension=float(d.get("grounding_tension", 0.0) or 0.0),
            graph_leverage=float(d.get("graph_leverage", 0.0) or 0.0),
            base_confidence=float(d.get("base_confidence", 0.0) or 0.0),
            effective_confidence=float(d.get("effective_confidence", 0.0) or 0.0),
            created_at=float(d.get("created_at", time.time()) or time.time()),
            asked_synchronously=bool(d.get("asked_synchronously", False)),
            answered=bool(d.get("answered", False)),
            answered_at=float(d.get("answered_at", 0.0) or 0.0),
            answer_text=str(d.get("answer_text", "")),
            external_validation=str(d.get("external_validation", "")),
            grounded=bool(d.get("grounded", False)),
        )


def classify_answer(answer_text: str) -> tuple[str, bool]:
    """Map an operator answer to (external_validation, grounded).

    SPARK §7: being corrected counts as success. A "no/wrong" still records
    ``grounded=True`` (the belief moved from inferred to externally-anchored);
    only the validation polarity differs. Empty/ignored answers do not validate.
    """
    text = (answer_text or "").strip().lower()
    if not text or len(text) < 2:
        return "", False
    _NEG = ("no", "wrong", "incorrect", "false", "not true", "nope", "refute", "disagree")
    _POS = ("yes", "correct", "right", "true", "confirmed", "confirm", "agree", "indeed")
    if any(text.startswith(n) or (" " + n + " ") in (" " + text + " ") for n in _NEG):
        return "refuted", True
    if any(text.startswith(p) or (" " + p + " ") in (" " + text + " ") for p in _POS):
        return "confirmed", True
    # A substantive freeform answer still anchors the belief externally.
    return "confirmed", True


# ---------------------------------------------------------------------------
# The durable queue.
# ---------------------------------------------------------------------------

class GroundingQueue:
    """Durable, ranked, operator-gated validation queue (singleton)."""

    _instance: "GroundingQueue | None" = None

    def __init__(self) -> None:
        self._pending: dict[str, PendingGroundingQuestion] = {}
        self._starvation = StarvationState()
        self._total_enqueued = 0
        self._total_answered = 0
        self._total_confirmed = 0
        self._total_refuted = 0
        self._total_expired = 0
        self._load()

    @classmethod
    def get_instance(cls) -> "GroundingQueue":
        if cls._instance is None:
            cls._instance = GroundingQueue()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    # -- enqueue (called from the advisory grounding path; never auto-answers) --

    def enqueue(
        self,
        *,
        belief_id: str,
        question_text: str,
        facet: str = "factual",
        channel: str = "web",
        provenance: str = "",
        rendered_claim: str = "",
        grounding_tension: float = 0.0,
        graph_leverage: float = 0.0,
        base_confidence: float = 0.0,
        effective_confidence: float = 0.0,
        asked_synchronously: bool = False,
    ) -> PendingGroundingQuestion | None:
        """Add (or refresh) a pending validation question. Dedup by belief_id.

        Returns the pending record, or None on failure. Never raises into caller.
        """
        try:
            bid = (belief_id or "").strip()
            if not bid:
                # Allow belief-less prompts but key them on the question text so
                # they still dedup and persist.
                bid = "q:" + str(abs(hash(question_text)) % (10 ** 10))
            existing = self._pending.get(bid)
            if existing is not None and not existing.answered:
                # Refresh the live tension/leverage so ranking stays current; do
                # not reset created_at (staleness must keep accruing).
                existing.grounding_tension = grounding_tension or existing.grounding_tension
                existing.graph_leverage = graph_leverage or existing.graph_leverage
                existing.base_confidence = base_confidence or existing.base_confidence
                existing.effective_confidence = (
                    effective_confidence or existing.effective_confidence
                )
                if asked_synchronously:
                    existing.asked_synchronously = True
                self.save()
                return existing

            qid = f"gq_{int(time.time() * 1000)}_{len(self._pending)}"
            rec = PendingGroundingQuestion(
                question_id=qid,
                belief_id=bid,
                question_text=question_text,
                facet=facet,
                channel=channel,
                provenance=provenance,
                rendered_claim=rendered_claim,
                grounding_tension=grounding_tension,
                graph_leverage=graph_leverage,
                base_confidence=base_confidence,
                effective_confidence=effective_confidence,
                asked_synchronously=asked_synchronously,
            )
            self._pending[bid] = rec
            self._total_enqueued += 1
            self._evict_if_needed()
            self.save()
            return rec
        except Exception:
            logger.debug("GroundingQueue.enqueue failed", exc_info=True)
            return None

    def _evict_if_needed(self) -> None:
        """Past MAX_PENDING, evict the lowest-ranked unanswered questions."""
        unanswered = [q for q in self._pending.values() if not q.answered]
        if len(unanswered) <= MAX_PENDING:
            return
        unanswered.sort(key=lambda q: q.rank())
        for q in unanswered[: len(unanswered) - MAX_PENDING]:
            self._pending.pop(q.belief_id, None)

    def expire_stale(self) -> int:
        """Remove unanswered questions older than the TTL. Returns count removed."""
        now = time.time()
        removed = 0
        for bid in list(self._pending.keys()):
            q = self._pending[bid]
            if not q.answered and (now - q.created_at) > QUESTION_TTL_S:
                self._pending.pop(bid, None)
                removed += 1
        if removed:
            self._total_expired += removed
            self.save()
        return removed

    # -- answer (OPERATOR-GATED only; never auto-fired) ------------------------

    def answer(self, question_id: str, answer_text: str) -> dict[str, Any]:
        """Record an operator answer (SPARK §6). OPERATOR-GATED — never auto-fired.

        Records the external-validation outcome on the queue and to the durable
        outcome log + the grounding/spark promotion gates (external-only). Does
        NOT mutate the belief here (view-only; belief mutation is the P5 active
        closure). Returns the updated record dict.
        """
        rec = None
        for q in self._pending.values():
            if q.question_id == question_id:
                rec = q
                break
        if rec is None:
            return {"ok": False, "error": "question_not_found", "question_id": question_id}

        validation, grounded = classify_answer(answer_text)
        rec.answered = True
        rec.answered_at = time.time()
        rec.answer_text = answer_text or ""
        rec.external_validation = validation
        rec.grounded = grounded
        self._total_answered += 1
        if validation == "confirmed":
            self._total_confirmed += 1
        elif validation == "refuted":
            self._total_refuted += 1

        # External-validation outcome → promotion gates (external-only, §7).
        # ``validated`` = the belief got an external touch at all (confirm OR
        # refute both count). Being corrected still counts as a grounding success.
        try:
            from autonomy.drives import GroundingDrivePromotion
            GroundingDrivePromotion.get_instance().record_external_validation(grounded)
        except Exception:
            logger.debug("answer: GroundingDrivePromotion record failed", exc_info=True)
        try:
            from autonomy.spark_metrics import SparkPromotion
            SparkPromotion.get_instance().record_external_validation(grounded)
        except Exception:
            logger.debug("answer: SparkPromotion record failed", exc_info=True)

        # Durable, append-only external-validation log (audit chain, §2 station 5).
        self._append_outcome_log(rec)
        self.save()
        return {"ok": True, "record": rec.to_dict()}

    def _append_outcome_log(self, rec: PendingGroundingQuestion) -> None:
        try:
            path = Path(QUEUE_PATH).with_name("grounding_outcomes.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps({
                "ts": rec.answered_at,
                "question_id": rec.question_id,
                "belief_id": rec.belief_id,
                "external_validation": rec.external_validation,
                "grounded": rec.grounded,
                "facet": rec.facet,
                "channel": rec.channel,
                "answer_excerpt": rec.answer_text[:200],
            })
            with open(path, "a") as f:
                f.write(line + "\n")
        except Exception:
            logger.debug("answer: outcome-log append failed", exc_info=True)

    # -- starvation -----------------------------------------------------------

    def set_starvation(self, state: StarvationState) -> None:
        state.pending_batch_size = self.pending_count()
        self._starvation = state

    def starvation(self) -> StarvationState:
        self._starvation.pending_batch_size = self.pending_count()
        return self._starvation

    # -- read accessors -------------------------------------------------------

    def pending_count(self) -> int:
        return sum(1 for q in self._pending.values() if not q.answered)

    def ranked_pending(self, limit: int = 50) -> list[PendingGroundingQuestion]:
        now = time.time()
        pending = [q for q in self._pending.values() if not q.answered]
        pending.sort(key=lambda q: q.rank(now), reverse=True)
        return pending[: max(0, limit)]

    def get_status(self, limit: int = 50) -> dict[str, Any]:
        now = time.time()
        ranked = self.ranked_pending(limit=limit)
        recent_answered = sorted(
            (q for q in self._pending.values() if q.answered),
            key=lambda q: q.answered_at, reverse=True,
        )[:10]
        return {
            "phase": "P4_advisory_async_queue",
            "authority": "operator_gated_review",
            "auto_fires": False,
            "pending_count": self.pending_count(),
            "totals": {
                "enqueued": self._total_enqueued,
                "answered": self._total_answered,
                "confirmed": self._total_confirmed,
                "refuted": self._total_refuted,
                "expired": self._total_expired,
            },
            "starvation": self.starvation().to_dict(),
            "pending": [q.to_dict(now) for q in ranked],
            "recent_answered": [q.to_dict(now) for q in recent_answered],
        }

    # -- persistence (atomic write, mirrors promotion.py) ---------------------

    def save(self) -> None:
        try:
            data = {
                "pending": {bid: q.to_dict() for bid, q in self._pending.items()},
                "starvation": self._starvation.to_dict(),
                "totals": {
                    "enqueued": self._total_enqueued,
                    "answered": self._total_answered,
                    "confirmed": self._total_confirmed,
                    "refuted": self._total_refuted,
                    "expired": self._total_expired,
                },
            }
            path = Path(QUEUE_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(path)
        except Exception:
            logger.debug("GroundingQueue.save failed", exc_info=True)

    def _load(self) -> None:
        try:
            path = Path(QUEUE_PATH)
            if not path.exists():
                return
            data = json.loads(path.read_text())
            for bid, qd in (data.get("pending", {}) or {}).items():
                try:
                    self._pending[bid] = PendingGroundingQuestion.from_dict(qd)
                except Exception:
                    continue
            totals = data.get("totals", {}) or {}
            self._total_enqueued = int(totals.get("enqueued", 0) or 0)
            self._total_answered = int(totals.get("answered", 0) or 0)
            self._total_confirmed = int(totals.get("confirmed", 0) or 0)
            self._total_refuted = int(totals.get("refuted", 0) or 0)
            self._total_expired = int(totals.get("expired", 0) or 0)
            st = data.get("starvation", {}) or {}
            self._starvation = StarvationState(
                operator_present=bool(st.get("operator_present", False)),
                pi_signal_available=bool(st.get("pi_signal_available", False)),
                web_exhausted=bool(st.get("web_exhausted", False)),
                starved=bool(st.get("starved", False)),
                self_floored_local_only=bool(st.get("self_floored_local_only", False)),
                pending_batch_size=int(st.get("pending_batch_size", 0) or 0),
                last_operator_seen_ts=float(st.get("last_operator_seen_ts", 0.0) or 0.0),
                note=str(st.get("note", "")),
            )
            logger.info("GroundingQueue restored: %d pending", self.pending_count())
        except Exception:
            logger.debug("GroundingQueue.load failed", exc_info=True)
