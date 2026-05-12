"""Quarantine Anomaly Scorer — Layer 8 Shadow Mode.

Scores 5 categories of suspicious activity:
  1. Contradiction spikes
  2. Memory anomalies
  3. Input manipulation signals
  4. Calibration drift
  5. Identity resolution instability

Each produces a QuarantineSignal with score [0, 1].
Shadow mode: scores and logs. Never blocks, mutates, or suppresses.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_COOLDOWN_WINDOW_S = 600.0
_CHRONIC_THRESHOLD = 3
_COOLDOWN_CLEANUP_MULTIPLIER = 2.0


@dataclass
class QuarantineSignal:
    score: float
    category: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    memory_id: str | None = None
    identity_context: dict[str, Any] | None = None
    repeat_count: int = 0
    is_chronic: bool = False

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        self.score = max(0.0, min(1.0, self.score))


@dataclass
class _CooldownEntry:
    fingerprint: str
    first_seen: float
    last_seen: float
    repeat_count: int
    last_signal: QuarantineSignal


CATEGORY_CONTRADICTION = "contradiction_spike"
CATEGORY_MEMORY = "memory_anomaly"
CATEGORY_MANIPULATION = "input_manipulation"
CATEGORY_CALIBRATION = "calibration_drift"
CATEGORY_IDENTITY = "identity_instability"

_CONTRADICTION_DEBT_WARN = 0.4
_CONTRADICTION_DEBT_CRITICAL = 0.7
_MEMORY_RATE_WARN = 10
_MEMORY_RATE_CRITICAL = 25
_LOW_CONFIDENCE_THRESHOLD = 0.45
_IDENTITY_FLIP_WARN = 3
_IDENTITY_FLIP_CRITICAL = 6
_LOW_CONF_WRITE_WINDOW_S = 60.0
_LOW_CONF_WRITE_WARN = 5


class QuarantineScorer:
    """Shadow-mode anomaly scorer for Layer 8.

    Invariants (tested, not just documented):
      - Never blocks memory writes
      - Never mutates belief confidence
      - Never suppresses retrieval results
      - Never changes policy scores
      - Never affects user-visible replies
    """

    def __init__(self) -> None:
        self._signals: deque[QuarantineSignal] = deque(maxlen=200)
        self._last_tick: float = 0.0
        self._tick_count: int = 0
        self._prev_contradiction_debt: float = 0.0
        self._prev_memory_count: int = 0
        self._identity_flips: deque[float] = deque(maxlen=20)
        self._low_conf_writes: deque[float] = deque(maxlen=50)
        self._signal_cooldowns: dict[str, _CooldownEntry] = {}
        self._suppressed_total: int = 0

    def tick(self, state: dict[str, Any]) -> list[QuarantineSignal]:
        """Run all 5 scorers. Returns new signals (may be empty).

        state keys expected:
          contradiction_debt: float
          recent_memory_writes: list[dict]  (memories written since last tick)
          identity_confidence_dist: dict    (recent identity resolution stats)
          calibration_snapshot: dict | None (truth calibration state)
          memory_count: int                 (total memory count)
        """
        now = time.time()
        self._last_tick = now
        self._tick_count += 1
        signals: list[QuarantineSignal] = []

        signals.extend(self._score_contradiction_spike(state, now))
        signals.extend(self._score_memory_anomalies(state, now))
        signals.extend(self._score_input_manipulation(state, now))
        signals.extend(self._score_calibration_drift(state, now))
        signals.extend(self._score_identity_instability(state, now))

        emitted = self._apply_dedupe(signals, now)
        self._cleanup_expired_cooldowns(now)

        for sig in emitted:
            self._signals.append(sig)

        if emitted:
            logger.debug("Quarantine shadow: %d signals emitted, %d suppressed (categories: %s)",
                         len(emitted), len(signals) - len(emitted),
                         {s.category for s in emitted})

        return emitted

    @staticmethod
    def _compute_fingerprint(signal: QuarantineSignal) -> str:
        """Category + coarsened evidence key for dedupe grouping."""
        cat = signal.category
        ev = signal.evidence

        if cat == CATEGORY_CALIBRATION:
            domain = ev.get("domain", "overall")
            score = ev.get("score", ev.get("truth_score", 0.0))
            return f"{cat}:{domain}:{round(score or 0.0, 1)}"

        if cat == CATEGORY_CONTRADICTION:
            debt = ev.get("debt", 0.0)
            return f"{cat}:{round(debt, 1)}"

        if cat == CATEGORY_IDENTITY:
            flips = ev.get("flip_count", 0)
            bucket = flips // 3
            conflict = ev.get("voice") is not None or ev.get("face") is not None
            return f"{cat}:flips_{bucket}:conflict_{conflict}"

        if cat == CATEGORY_MEMORY:
            rate = ev.get("write_rate", 0)
            low = ev.get("window_count", 0)
            unresolved = ev.get("unresolved_count", 0)
            return f"{cat}:rate_{rate // 5}:low_{low // 3}:unres_{unresolved // 3}"

        if cat == CATEGORY_MANIPULATION:
            corrections = ev.get("correction_count", 0)
            conflicts = ev.get("conflict_count", 0)
            return f"{cat}:corr_{corrections // 3}:prov_{conflicts // 2}"

        return f"{cat}:unknown"

    def _apply_dedupe(
        self, raw_signals: list[QuarantineSignal], now: float,
    ) -> list[QuarantineSignal]:
        """Filter signals through cooldown. Returns only those that should be emitted."""
        emitted: list[QuarantineSignal] = []

        for sig in raw_signals:
            fp = self._compute_fingerprint(sig)
            entry = self._signal_cooldowns.get(fp)

            if entry is None:
                self._signal_cooldowns[fp] = _CooldownEntry(
                    fingerprint=fp,
                    first_seen=now,
                    last_seen=now,
                    repeat_count=0,
                    last_signal=sig,
                )
                emitted.append(sig)
                continue

            age = now - entry.first_seen
            if age >= _COOLDOWN_WINDOW_S:
                recap = QuarantineSignal(
                    score=sig.score,
                    category=sig.category,
                    reason=sig.reason,
                    evidence={
                        **sig.evidence,
                        "repeat_count": entry.repeat_count,
                        "chronic_duration_s": round(age, 1),
                    },
                    timestamp=now,
                    memory_id=sig.memory_id,
                    identity_context=sig.identity_context,
                    repeat_count=entry.repeat_count,
                    is_chronic=entry.repeat_count >= _CHRONIC_THRESHOLD,
                )
                self._signal_cooldowns[fp] = _CooldownEntry(
                    fingerprint=fp,
                    first_seen=now,
                    last_seen=now,
                    repeat_count=0,
                    last_signal=sig,
                )
                emitted.append(recap)
            else:
                entry.last_seen = now
                entry.repeat_count += 1
                entry.last_signal = sig
                self._suppressed_total += 1

        return emitted

    def _cleanup_expired_cooldowns(self, now: float) -> None:
        cutoff = now - _COOLDOWN_WINDOW_S * _COOLDOWN_CLEANUP_MULTIPLIER
        expired = [fp for fp, e in self._signal_cooldowns.items() if e.last_seen < cutoff]
        for fp in expired:
            del self._signal_cooldowns[fp]

    def _score_contradiction_spike(self, state: dict[str, Any], now: float) -> list[QuarantineSignal]:
        debt = state.get("contradiction_debt", 0.0)
        prev = self._prev_contradiction_debt
        self._prev_contradiction_debt = debt

        signals: list[QuarantineSignal] = []
        if debt >= _CONTRADICTION_DEBT_CRITICAL:
            signals.append(QuarantineSignal(
                score=min(1.0, debt),
                category=CATEGORY_CONTRADICTION,
                reason=f"Critical contradiction debt: {debt:.2f}",
                evidence={"debt": debt, "prev_debt": prev, "delta": debt - prev},
                timestamp=now,
            ))
        elif debt >= _CONTRADICTION_DEBT_WARN:
            delta = debt - prev
            if delta > 0.1:
                signals.append(QuarantineSignal(
                    score=debt * 0.8,
                    category=CATEGORY_CONTRADICTION,
                    reason=f"Contradiction debt spike: {prev:.2f} -> {debt:.2f}",
                    evidence={"debt": debt, "prev_debt": prev, "delta": delta},
                    timestamp=now,
                ))
        return signals

    def _score_memory_anomalies(self, state: dict[str, Any], now: float) -> list[QuarantineSignal]:
        recent_writes: list[dict] = state.get("recent_memory_writes", [])
        count = state.get("memory_count", 0)
        prev = self._prev_memory_count
        self._prev_memory_count = count

        signals: list[QuarantineSignal] = []

        write_rate = len(recent_writes)
        if write_rate >= _MEMORY_RATE_CRITICAL:
            signals.append(QuarantineSignal(
                score=min(1.0, write_rate / 40.0),
                category=CATEGORY_MEMORY,
                reason=f"High memory creation rate: {write_rate} in last tick",
                evidence={"write_rate": write_rate, "total": count},
                timestamp=now,
            ))
        elif write_rate >= _MEMORY_RATE_WARN:
            signals.append(QuarantineSignal(
                score=write_rate / 40.0,
                category=CATEGORY_MEMORY,
                reason=f"Elevated memory creation rate: {write_rate}",
                evidence={"write_rate": write_rate, "total": count},
                timestamp=now,
            ))

        low_conf = [w for w in recent_writes
                    if w.get("identity_confidence", 1.0) < _LOW_CONFIDENCE_THRESHOLD]
        if low_conf:
            for w in low_conf:
                self._low_conf_writes.append(now)
            window_writes = sum(1 for t in self._low_conf_writes if now - t < _LOW_CONF_WRITE_WINDOW_S)
            if window_writes >= _LOW_CONF_WRITE_WARN:
                signals.append(QuarantineSignal(
                    score=min(1.0, window_writes / 10.0),
                    category=CATEGORY_MEMORY,
                    reason=f"{window_writes} low-confidence identity writes in {_LOW_CONF_WRITE_WINDOW_S}s",
                    evidence={
                        "low_conf_count": len(low_conf),
                        "window_count": window_writes,
                        "sample_ids": [w.get("id", "") for w in low_conf[:3]],
                    },
                    timestamp=now,
                ))

        unresolved = [w for w in recent_writes if w.get("identity_needs_resolution")]
        if len(unresolved) >= 3:
            signals.append(QuarantineSignal(
                score=min(1.0, len(unresolved) / 8.0),
                category=CATEGORY_MEMORY,
                reason=f"{len(unresolved)} memories from unresolved identities",
                evidence={"unresolved_count": len(unresolved)},
                timestamp=now,
            ))

        return signals

    def _score_input_manipulation(self, state: dict[str, Any], now: float) -> list[QuarantineSignal]:
        signals: list[QuarantineSignal] = []

        recent_writes: list[dict] = state.get("recent_memory_writes", [])
        corrections = [w for w in recent_writes if w.get("type") == "user_preference"
                       and "correc" in str(w.get("payload", "")).lower()]
        if len(corrections) >= 3:
            signals.append(QuarantineSignal(
                score=min(1.0, len(corrections) / 5.0),
                category=CATEGORY_MANIPULATION,
                reason=f"{len(corrections)} corrections in recent window",
                evidence={"correction_count": len(corrections)},
                timestamp=now,
            ))

        provenance_conflicts = state.get("provenance_conflicts", 0)
        if provenance_conflicts >= 2:
            signals.append(QuarantineSignal(
                score=min(1.0, provenance_conflicts / 5.0),
                category=CATEGORY_MANIPULATION,
                reason=f"{provenance_conflicts} provenance conflicts (user_claim vs observed)",
                evidence={"conflict_count": provenance_conflicts},
                timestamp=now,
            ))

        return signals

    def _score_calibration_drift(self, state: dict[str, Any], now: float) -> list[QuarantineSignal]:
        cal = state.get("calibration_snapshot")
        if not cal:
            return []

        signals: list[QuarantineSignal] = []

        domain_scores = cal.get("domain_scores", {})
        domain_prov = cal.get("domain_provisional", {})
        for domain, score in domain_scores.items():
            if domain_prov.get(domain, True):
                continue
            if score is not None and score < 0.3:
                signals.append(QuarantineSignal(
                    score=1.0 - score,
                    category=CATEGORY_CALIBRATION,
                    reason=f"Domain '{domain}' score critically low: {score:.2f}",
                    evidence={"domain": domain, "score": score},
                    timestamp=now,
                ))

        truth_score = cal.get("truth_score")
        if truth_score is not None and truth_score < 0.4:
            signals.append(QuarantineSignal(
                score=1.0 - truth_score,
                category=CATEGORY_CALIBRATION,
                reason=f"Overall truth score low: {truth_score:.2f}",
                evidence={"truth_score": truth_score},
                timestamp=now,
            ))

        return signals

    def _score_identity_instability(self, state: dict[str, Any], now: float) -> list[QuarantineSignal]:
        id_dist = state.get("identity_confidence_dist", {})
        signals: list[QuarantineSignal] = []

        flips = id_dist.get("recent_flips", 0)
        if flips > 0:
            for _ in range(flips):
                self._identity_flips.append(now)

        recent_flips = sum(1 for t in self._identity_flips if now - t < 120.0)
        if recent_flips >= _IDENTITY_FLIP_CRITICAL:
            signals.append(QuarantineSignal(
                score=min(1.0, recent_flips / 10.0),
                category=CATEGORY_IDENTITY,
                reason=f"Rapid identity flip: {recent_flips} flips in 120s",
                evidence={"flip_count": recent_flips},
                timestamp=now,
            ))
        elif recent_flips >= _IDENTITY_FLIP_WARN:
            signals.append(QuarantineSignal(
                score=recent_flips / 10.0,
                category=CATEGORY_IDENTITY,
                reason=f"Identity oscillation: {recent_flips} flips in 120s",
                evidence={"flip_count": recent_flips},
                timestamp=now,
            ))

        conflict_active = id_dist.get("conflict_active", False)
        if conflict_active:
            signals.append(QuarantineSignal(
                score=0.6,
                category=CATEGORY_IDENTITY,
                reason="Voice/face ownership conflict active",
                evidence={"voice": id_dist.get("voice_name"), "face": id_dist.get("face_name")},
                timestamp=now,
            ))

        return signals

    def get_recent_signals(self, limit: int = 50) -> list[dict[str, Any]]:
        recent = list(self._signals)[-limit:]
        return [
            {
                "score": s.score,
                "category": s.category,
                "reason": s.reason,
                "evidence": s.evidence,
                "timestamp": s.timestamp,
                "memory_id": s.memory_id,
                "identity_context": s.identity_context,
                "repeat_count": s.repeat_count,
                "is_chronic": s.is_chronic,
            }
            for s in recent
        ]

    def get_stats(self) -> dict[str, Any]:
        signals = list(self._signals)
        categories: dict[str, int] = {}
        for s in signals:
            categories[s.category] = categories.get(s.category, 0) + 1

        now = time.time()
        chronic: list[dict[str, Any]] = []
        for fp, entry in self._signal_cooldowns.items():
            if entry.repeat_count >= _CHRONIC_THRESHOLD:
                chronic.append({
                    "fingerprint": fp,
                    "category": entry.last_signal.category,
                    "reason": entry.last_signal.reason,
                    "repeat_count": entry.repeat_count,
                    "duration_s": round(now - entry.first_seen, 1),
                })

        return {
            "tick_count": self._tick_count,
            "total_signals": len(signals),
            "category_counts": categories,
            "last_tick": self._last_tick,
            "suppressed_duplicates": self._suppressed_total,
            "active_cooldowns": len(self._signal_cooldowns),
            "chronic_signals": chronic,
        }
