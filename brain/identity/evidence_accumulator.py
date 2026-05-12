"""Identity Evidence Accumulator — promotes identities through evidence tiers.

No identity becomes persistent until enough independent evidence accumulates.
All identity creation paths route through this single gate.

Promotion tiers:
  candidate   — seen once, not trusted, in-memory only
  provisional — evidence score >= 0.75, session-only (no durable persistence)
  persistent  — evidence score >= 1.5, at least 2 evidence types, written to disk

Evidence sources and weights:
  manual_enroll:      +1.0  (explicit dashboard/voice enrollment)
  voice_match:        +0.35 (repeated voice match over threshold)
  face_match:         +0.35 (repeated face match over threshold)
  textual_self_id:    +0.50 (conversational "my name is X")
  weak_regex:         +0.10 (regex extraction from casual speech)
  contradiction:      -0.40 (conflicting identity signals)
  environment_noise:  -0.30 (no face, no voice, desk/screen scene)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from identity.name_validator import is_valid_person_name, rejection_reason

logger = logging.getLogger(__name__)

CANDIDATES_PATH = Path.home() / ".jarvis" / "identity_candidates.json"

PROVISIONAL_THRESHOLD = 0.75
PERSISTENT_THRESHOLD = 1.5
MIN_EVIDENCE_TYPES_FOR_PERSISTENT = 2
MIN_OBSERVATIONS_FOR_PERSISTENT = 2
MAX_CONTRADICTION_FOR_PROMOTION = 0.5

CANDIDATE_EXPIRY_LOW_S = 86400.0       # 24h if low evidence
CANDIDATE_EXPIRY_MEDIUM_S = 604800.0   # 7d if nearing threshold

EVIDENCE_WEIGHTS: dict[str, float] = {
    "manual_enroll": 1.0,
    "voice_match": 0.35,
    "face_match": 0.35,
    "textual_self_id": 0.50,
    "weak_regex": 0.10,
    "contradiction": -0.40,
    "environment_noise": -0.30,
}


@dataclass
class EvidenceEvent:
    source: str
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.0
    details: str = ""


@dataclass
class IdentityCandidate:
    raw_name: str
    normalized_name: str
    first_seen_ts: float = field(default_factory=time.time)
    last_seen_ts: float = field(default_factory=time.time)
    evidence_events: list[EvidenceEvent] = field(default_factory=list)
    promoted: bool = False
    promotion_tier: str = "candidate"
    promotion_ts: float = 0.0

    @property
    def evidence_score(self) -> float:
        score = 0.0
        for ev in self.evidence_events:
            w = EVIDENCE_WEIGHTS.get(ev.source, 0.0)
            score += w * max(0.1, ev.confidence)
        return round(score, 4)

    @property
    def evidence_types(self) -> set[str]:
        return {ev.source for ev in self.evidence_events if EVIDENCE_WEIGHTS.get(ev.source, 0) > 0}

    @property
    def observation_count(self) -> int:
        return len([ev for ev in self.evidence_events if EVIDENCE_WEIGHTS.get(ev.source, 0) > 0])

    @property
    def contradiction_score(self) -> float:
        return sum(
            abs(EVIDENCE_WEIGHTS.get(ev.source, 0)) * ev.confidence
            for ev in self.evidence_events
            if EVIDENCE_WEIGHTS.get(ev.source, 0) < 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_name": self.raw_name,
            "normalized_name": self.normalized_name,
            "first_seen_ts": self.first_seen_ts,
            "last_seen_ts": self.last_seen_ts,
            "evidence_events": [
                {"source": e.source, "timestamp": e.timestamp,
                 "confidence": e.confidence, "details": e.details}
                for e in self.evidence_events
            ],
            "promoted": self.promoted,
            "promotion_tier": self.promotion_tier,
            "promotion_ts": self.promotion_ts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IdentityCandidate:
        events = [
            EvidenceEvent(
                source=e["source"], timestamp=e.get("timestamp", 0),
                confidence=e.get("confidence", 0), details=e.get("details", ""),
            )
            for e in d.get("evidence_events", [])
        ]
        return cls(
            raw_name=d["raw_name"],
            normalized_name=d["normalized_name"],
            first_seen_ts=d.get("first_seen_ts", 0),
            last_seen_ts=d.get("last_seen_ts", 0),
            evidence_events=events,
            promoted=d.get("promoted", False),
            promotion_tier=d.get("promotion_tier", "candidate"),
            promotion_ts=d.get("promotion_ts", 0),
        )


class EvidenceAccumulator:
    """Central gate for identity creation — all paths route through here.

    Thread-safe via simple dict operations (GIL-protected reads/writes).
    """

    def __init__(self, candidates_path: Path | None = None) -> None:
        self._candidates: dict[str, IdentityCandidate] = {}
        self._path = candidates_path or CANDIDATES_PATH
        self._promotion_callbacks: list = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                for key, cdata in data.get("candidates", {}).items():
                    self._candidates[key] = IdentityCandidate.from_dict(cdata)
                logger.info("Loaded %d identity candidates", len(self._candidates))
            except Exception as exc:
                logger.warning("Failed to load identity candidates: %s", exc)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {"candidates": {k: v.to_dict() for k, v in self._candidates.items()}}
            from memory.persistence import atomic_write_json
            atomic_write_json(self._path, data, indent=2)
        except Exception as exc:
            logger.warning("Failed to save identity candidates: %s", exc)

    def on_promotion(self, callback) -> None:
        """Register a callback(name, tier) for when a candidate is promoted."""
        self._promotion_callbacks.append(callback)

    def observe(
        self,
        name: str,
        source: str,
        confidence: float = 1.0,
        details: str = "",
    ) -> dict[str, Any]:
        """Record an identity evidence signal. Returns action taken.

        Returns dict with keys:
          - accepted: bool — whether the observation was recorded
          - tier: str — current tier after this observation
          - promoted: bool — whether this observation triggered a promotion
          - action: str — what happened ("rejected", "created", "updated", "promoted_provisional", "promoted_persistent")
        """
        result = {"accepted": False, "tier": "rejected", "promoted": False, "action": "rejected", "name": name}

        if not name or not isinstance(name, str):
            return result

        clean = name.strip().title()
        key = clean.lower()

        if source not in ("contradiction", "environment_noise"):
            if not is_valid_person_name(clean):
                reason = rejection_reason(clean)
                logger.debug("Evidence rejected for invalid name %r: %s", clean, reason)
                return result

        if source not in EVIDENCE_WEIGHTS:
            logger.warning("Unknown evidence source: %r", source)
            return result

        now = time.time()
        ev = EvidenceEvent(source=source, timestamp=now, confidence=confidence, details=details)

        if key not in self._candidates:
            self._candidates[key] = IdentityCandidate(
                raw_name=name, normalized_name=clean,
                first_seen_ts=now, last_seen_ts=now,
                evidence_events=[ev],
            )
            result.update(accepted=True, tier="candidate", action="created")
            logger.info("New identity candidate: %r (source=%s, conf=%.2f)", clean, source, confidence)
        else:
            cand = self._candidates[key]
            cand.last_seen_ts = now
            cand.evidence_events.append(ev)
            result.update(accepted=True, tier=cand.promotion_tier, action="updated")

        cand = self._candidates[key]
        old_tier = cand.promotion_tier

        new_tier = self._evaluate_tier(cand)
        if new_tier != old_tier:
            cand.promotion_tier = new_tier
            cand.promotion_ts = now
            if new_tier == "persistent":
                cand.promoted = True
            result["tier"] = new_tier
            result["promoted"] = True
            result["action"] = f"promoted_{new_tier}"
            logger.info(
                "Identity %r promoted: %s → %s (score=%.2f, types=%s, obs=%d)",
                clean, old_tier, new_tier, cand.evidence_score,
                cand.evidence_types, cand.observation_count,
            )
            for cb in self._promotion_callbacks:
                try:
                    cb(clean, new_tier)
                except Exception:
                    logger.debug("Promotion callback error", exc_info=True)
        else:
            result["tier"] = cand.promotion_tier

        self._save()
        return result

    def _evaluate_tier(self, cand: IdentityCandidate) -> str:
        """Determine what tier a candidate should be in based on current evidence.

        Promotion rules:
          persistent requires: score >= 1.5 AND (2+ evidence types OR 1 manual confirmation)
                               AND contradiction_score < 0.5
          provisional requires: score >= 0.75 AND contradiction_score < 0.5
        """
        score = cand.evidence_score
        types = cand.evidence_types
        contradictions = cand.contradiction_score
        has_manual = "manual_enroll" in types

        if contradictions >= MAX_CONTRADICTION_FOR_PROMOTION:
            return "candidate"

        if score >= PERSISTENT_THRESHOLD:
            if has_manual or len(types) >= MIN_EVIDENCE_TYPES_FOR_PERSISTENT:
                return "persistent"

        if score >= PROVISIONAL_THRESHOLD:
            return "provisional"

        return "candidate"

    def is_promoted(self, name: str, min_tier: str = "provisional") -> bool:
        """Check if a name has been promoted to at least the given tier."""
        key = name.lower().strip()
        cand = self._candidates.get(key)
        if not cand:
            return False
        tiers = {"candidate": 0, "provisional": 1, "persistent": 2}
        return tiers.get(cand.promotion_tier, 0) >= tiers.get(min_tier, 1)

    def is_persistent(self, name: str) -> bool:
        """Check if a name is a persistent (fully promoted) identity."""
        return self.is_promoted(name, min_tier="persistent")

    def get_candidate(self, name: str) -> IdentityCandidate | None:
        return self._candidates.get(name.lower().strip())

    def reject_candidate(self, name: str) -> bool:
        """Manually reject a candidate (dashboard action)."""
        key = name.lower().strip()
        if key in self._candidates:
            logger.info("Manually rejected identity candidate: %r", name)
            del self._candidates[key]
            self._save()
            return True
        return False

    def force_promote(self, name: str) -> bool:
        """Manually promote a candidate to persistent (dashboard action).

        Creates a candidate with manual_enroll evidence if none exists.
        """
        key = name.lower().strip()
        clean = name.strip().title()

        if not is_valid_person_name(clean):
            return False

        if key not in self._candidates:
            self.observe(clean, "manual_enroll", confidence=1.0, details="force_promote")
        else:
            self.observe(clean, "manual_enroll", confidence=1.0, details="force_promote")

        cand = self._candidates.get(key)
        if cand and cand.promotion_tier != "persistent":
            cand.promotion_tier = "persistent"
            cand.promoted = True
            cand.promotion_ts = time.time()
            self._save()
            for cb in self._promotion_callbacks:
                try:
                    cb(clean, "persistent")
                except Exception:
                    pass

        return True

    def merge_candidate(self, source_name: str, target_name: str) -> bool:
        """Merge source candidate's evidence into target. Returns True if merged."""
        src_key = source_name.lower().strip()
        tgt_key = target_name.lower().strip()

        src = self._candidates.get(src_key)
        tgt = self._candidates.get(tgt_key)
        if not src:
            logger.info("Evidence merge: source '%s' not found (nothing to merge)", source_name)
            return True
        if not tgt:
            tgt = IdentityCandidate(
                raw_name=target_name, normalized_name=target_name.strip().title(),
                first_seen_ts=src.first_seen_ts, last_seen_ts=src.last_seen_ts,
            )
            self._candidates[tgt_key] = tgt

        for ev in src.evidence_events:
            ev_copy = EvidenceEvent(
                source=ev.source, timestamp=ev.timestamp,
                confidence=ev.confidence,
                details=f"[merged from {source_name}] {ev.details}",
            )
            tgt.evidence_events.append(ev_copy)

        tgt.first_seen_ts = min(tgt.first_seen_ts, src.first_seen_ts)
        tgt.last_seen_ts = max(tgt.last_seen_ts, src.last_seen_ts)

        new_tier = self._evaluate_tier(tgt)
        if new_tier != tgt.promotion_tier:
            tgt.promotion_tier = new_tier
            tgt.promotion_ts = time.time()
            if new_tier == "persistent":
                tgt.promoted = True

        del self._candidates[src_key]
        self._save()
        logger.info("Evidence merge: '%s' → '%s' (score=%.2f, tier=%s)",
                     source_name, target_name, tgt.evidence_score, tgt.promotion_tier)
        return True

    def cleanup_expired(self) -> int:
        """Remove expired low-evidence candidates. Returns count removed."""
        now = time.time()
        to_remove = []
        for key, cand in self._candidates.items():
            if cand.promoted:
                continue
            age = now - cand.last_seen_ts
            score = cand.evidence_score
            if score < PROVISIONAL_THRESHOLD and age > CANDIDATE_EXPIRY_LOW_S:
                to_remove.append(key)
            elif score < PERSISTENT_THRESHOLD and age > CANDIDATE_EXPIRY_MEDIUM_S:
                to_remove.append(key)
        for key in to_remove:
            logger.info("Expiring identity candidate: %r (score=%.2f)", key, self._candidates[key].evidence_score)
            del self._candidates[key]
        if to_remove:
            self._save()
        return len(to_remove)

    def get_all_candidates(self) -> list[dict[str, Any]]:
        """Return summary of all candidates for dashboard."""
        result = []
        for key, cand in self._candidates.items():
            result.append({
                "name": cand.normalized_name,
                "tier": cand.promotion_tier,
                "score": cand.evidence_score,
                "evidence_types": sorted(cand.evidence_types),
                "observation_count": cand.observation_count,
                "contradiction_score": round(cand.contradiction_score, 3),
                "first_seen": cand.first_seen_ts,
                "last_seen": cand.last_seen_ts,
                "age_s": round(time.time() - cand.first_seen_ts, 1),
                "promoted": cand.promoted,
            })
        return sorted(result, key=lambda x: x["score"], reverse=True)

    def get_stats(self) -> dict[str, Any]:
        tiers = {"candidate": 0, "provisional": 0, "persistent": 0}
        for cand in self._candidates.values():
            tiers[cand.promotion_tier] = tiers.get(cand.promotion_tier, 0) + 1
        return {
            "total_candidates": len(self._candidates),
            "tier_counts": tiers,
            "persistent_names": sorted(
                c.normalized_name for c in self._candidates.values()
                if c.promotion_tier == "persistent"
            ),
        }


_instance: EvidenceAccumulator | None = None


def get_accumulator() -> EvidenceAccumulator:
    """Get or create the singleton EvidenceAccumulator."""
    global _instance
    if _instance is None:
        _instance = EvidenceAccumulator()
    return _instance
