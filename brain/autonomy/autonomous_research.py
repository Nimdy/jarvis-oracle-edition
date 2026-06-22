"""P5b — Autonomous Research (SHADOW execute) + its own zero-authority promotion gate.

The earned step from operator-pull research questions → JARVIS researching on her own. This module
is the EARNING LINK: when the grounding drive raises a factual belief-tension question, JARVIS may
auto-fire ``academic_search`` IN SHADOW, compute the conclusion she *would* reach, and log it —
**mutating nothing**. When the operator later answers that same belief, the shadow conclusion is
scored against the real answer; the match/mismatch is the only thing that earns this gate upward.

GOVERNANCE (docs/P5_AUTONOMOUS_RESEARCH.md — earn-don't-declare):
  * SEPARATE gate. ``AutonomousResearchPromotion`` is its OWN counter at level 0/shadow — it does NOT
    reuse ``GroundingDrivePromotion`` (already active, earned on a different capability). Reusing that
    would be a stealth authority grant.
  * ZERO AUTHORITY. P5b mutates no belief, drives no cadence/reward. It only fires research in shadow,
    logs the would-be conclusion, and scores it against independent operator answers.
  * EARNS on AUTONOMOUS-vs-OPERATOR ACCURACY (never self-scored): a shadow conclusion is matched only
    against a *later, independent* operator answer on the same belief.
  * HONORS the existing gates: the governor mode-gate + ``check_prior_knowledge`` dedup + the academic
    rate-limit are all checked BEFORE a shadow fire (routed through, never around).
  * DEFAULT-OFF firing. ``ENABLE_P5B_SHADOW`` gates the live academic fire (off until the operator
    flips it). The gate + scoring + telemetry are always present so the zero-authority guarantee is
    structural from day one.
  * HONEST / PRE-MATURE. The first-increment conclusion is a COARSE evidence-strength proxy (did the
    research surface peer-reviewed support?), not yet an LLM-judged confirm/refute — labelled as such;
    if the proxy can't earn the accuracy bar, the gate simply never promotes (that is the design working).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("jarvis.autonomy.p5b")

_JARVIS = Path.home() / ".jarvis"
ARP_PROMOTION_PATH = str(_JARVIS / "autonomous_research_promotion.json")
ARP_SHADOW_LOG = str(_JARVIS / "autonomous_research_shadow.jsonl")     # every shadow conclusion
ARP_MATCHED_LOG = str(_JARVIS / "autonomous_research_matched.jsonl")   # scored shadow↔operator pairs

# Earning thresholds. Accuracy here is PREDICTIVE (shadow conclusion vs a later operator answer), so the
# promote bar is well above chance — a weak coarse proxy SHOULD fail to earn (earn-don't-declare).
ARP_MIN_OUTCOMES = 20           # ≥20 scored shadow↔operator pairs before any promotion
ARP_MIN_SHADOW_HOURS = 4.0      # live-soak floor
ARP_PROMOTE_ACCURACY = 0.70     # shadow conclusions must match operator ≥70% to promote to advisory
ARP_DEMOTE_ACCURACY = 0.50      # below chance-ish → demote
ARP_DEMOTE_WINDOW = 20
ARP_TRANSITION_COOLDOWN_S = 300.0

# Live academic fire is OFF by default — the operator flips this when ready. The gate/scoring/telemetry
# run regardless, so the structure is auditable from day one with zero outward calls.
ENABLE_P5B_SHADOW = os.environ.get("JARVIS_ENABLE_P5B_SHADOW", "0") not in ("0", "", "false", "False")


# ---------------------------------------------------------------------------
# The separate, zero-authority promotion gate (mirrors SparkPromotion's shape,
# but earns on PREDICTIVE ACCURACY, and persists on every outcome from day one).
# ---------------------------------------------------------------------------
@dataclass
class _ARPState:
    level: int = 0  # 0=shadow, 1=advisory, 2=active — DEFAULTS TO SHADOW
    shadow_start_ts: float = field(default_factory=time.time)
    total_outcomes: int = 0
    accuracy_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    last_promoted_at: float = 0.0
    last_demoted_at: float = 0.0


class AutonomousResearchPromotion:
    """Zero-authority P5b gate. Earns ONLY on shadow-vs-operator predictive accuracy. NEVER reuses
    GroundingDrivePromotion (that gate is earned on operator-pull, a different capability)."""

    _instance: "AutonomousResearchPromotion | None" = None

    def __init__(self) -> None:
        self._state = _ARPState()
        self._load()

    @classmethod
    def get_instance(cls) -> "AutonomousResearchPromotion":
        if cls._instance is None:
            cls._instance = AutonomousResearchPromotion()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    @property
    def level(self) -> int:
        return self._state.level

    def is_shadow(self) -> bool:
        return self._state.level == 0

    def is_advisory(self) -> bool:
        return self._state.level == 1

    def is_active(self) -> bool:
        """True only when promoted to active (P5d). ALWAYS False in P5b/shadow."""
        return self._state.level >= 2

    def record_outcome(self, matched: bool) -> None:
        """Record one scored shadow↔operator pair. ``matched`` = the shadow conclusion agreed with the
        operator's later answer. Persists every time (no reset-on-restart deadlock — the SparkPromotion
        lesson baked in)."""
        self._state.total_outcomes += 1
        self._state.accuracy_history.append(1.0 if matched else 0.0)
        self._check_transitions()
        self.save()

    def _accuracy(self) -> float:
        h = self._state.accuracy_history
        return (sum(h) / len(h)) if h else 0.0

    def _promotion_eligible(self) -> bool:
        if self._state.level >= 2:
            return False
        if self._state.total_outcomes < ARP_MIN_OUTCOMES:
            return False
        if (time.time() - self._state.shadow_start_ts) / 3600.0 < ARP_MIN_SHADOW_HOURS:
            return False
        return self._accuracy() >= ARP_PROMOTE_ACCURACY

    def _check_transitions(self) -> None:
        now = time.time()
        if now - max(self._state.last_promoted_at, self._state.last_demoted_at) < ARP_TRANSITION_COOLDOWN_S:
            return
        if self._promotion_eligible():
            self._state.level = min(2, self._state.level + 1)
            self._state.last_promoted_at = now
            logger.info("P5b: AutonomousResearchPromotion → level %d (accuracy %.2f, n=%d)",
                        self._state.level, self._accuracy(), self._state.total_outcomes)
        elif self._state.level > 0 and len(self._state.accuracy_history) >= ARP_DEMOTE_WINDOW:
            recent = list(self._state.accuracy_history)[-ARP_DEMOTE_WINDOW:]
            if (sum(recent) / len(recent)) < ARP_DEMOTE_ACCURACY:
                self._state.level = max(0, self._state.level - 1)
                self._state.last_demoted_at = now
                logger.info("P5b: AutonomousResearchPromotion DEMOTED → level %d (accuracy dropped)",
                            self._state.level)

    def get_status(self) -> dict[str, Any]:
        hist = list(self._state.accuracy_history)
        hours = (time.time() - self._state.shadow_start_ts) / 3600.0
        return {
            "level": self._state.level,
            "level_name": {0: "shadow", 1: "advisory", 2: "active"}.get(self._state.level, "unknown"),
            "authority": "zero_authority_shadow" if self._state.level == 0 else (
                "advisory" if self._state.level == 1 else "active"),
            "total_outcomes": self._state.total_outcomes,
            "predictive_accuracy": round(self._accuracy(), 4),
            "window_size": len(hist),
            "hours_in_shadow": round(hours, 1),
            "promotion_ready": self._promotion_eligible(),
            "promote_accuracy_bar": ARP_PROMOTE_ACCURACY,
            "min_outcomes": ARP_MIN_OUTCOMES,
            "live_fire_enabled": ENABLE_P5B_SHADOW,
            "drives_levers": self.is_active(),
            "note": "PRE-MATURE — coarse evidence-strength proxy; earns on shadow-vs-operator accuracy only",
        }

    def save(self) -> None:
        data = {
            "level": self._state.level, "shadow_start_ts": self._state.shadow_start_ts,
            "total_outcomes": self._state.total_outcomes,
            "accuracy_history": list(self._state.accuracy_history),
            "last_promoted_at": self._state.last_promoted_at,
            "last_demoted_at": self._state.last_demoted_at,
        }
        try:
            p = Path(ARP_PROMOTION_PATH)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(p)
        except Exception:
            logger.debug("P5b: save failed", exc_info=True)

    def _load(self) -> None:
        try:
            p = Path(ARP_PROMOTION_PATH)
            if not p.exists():
                return
            d = json.loads(p.read_text())
            self._state.level = int(d.get("level", 0) or 0)
            self._state.shadow_start_ts = d.get("shadow_start_ts", time.time())
            self._state.total_outcomes = int(d.get("total_outcomes", 0) or 0)
            for v in d.get("accuracy_history", []):
                self._state.accuracy_history.append(float(v))
            self._state.last_promoted_at = d.get("last_promoted_at", 0.0)
            self._state.last_demoted_at = d.get("last_demoted_at", 0.0)
        except Exception:
            logger.debug("P5b: load failed", exc_info=True)


# ---------------------------------------------------------------------------
# Shadow conclusions: derive a coarse conclusion, store it pending, score it
# against the operator's later answer (the earning link). Mutates NO belief.
# ---------------------------------------------------------------------------
_VEC_KEYS = ("vector", "raw_vector", "composite_vector")
_pending: dict[str, dict[str, Any]] = {}   # belief_id → shadow conclusion awaiting an operator answer


def _append_jsonl(path: str, rec: dict[str, Any]) -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        logger.debug("P5b: append %s failed", path, exc_info=True)


def derive_conclusion(research_result: Any) -> dict[str, Any]:
    """Coarse, honest shadow conclusion from an academic ResearchResult. PRE-MATURE: this is an
    evidence-STRENGTH proxy (did the search surface peer-reviewed support?), NOT yet an LLM-judged
    confirm/refute of the belief — that polarity judgement is the documented P5b follow-on."""
    findings = list(getattr(research_result, "findings", []) or [])
    peer = [f for f in findings if getattr(f, "source_type", "") == "peer_reviewed"]
    max_conf = max((float(getattr(f, "confidence", 0.0) or 0.0) for f in findings), default=0.0)
    if peer and max_conf >= 0.6:
        evidence = "strong"
    elif findings:
        evidence = "weak"
    else:
        evidence = "none"
    return {
        "fired": bool(getattr(research_result, "success", False)),
        "n_findings": len(findings),
        "peer_reviewed": len(peer),
        "max_confidence": round(max_conf, 3),
        "evidence": evidence,
    }


def record_shadow_conclusion(belief_id: str, question: str, conclusion: dict[str, Any]) -> None:
    """Store a shadow conclusion (no mutation) and log it durably for later scoring."""
    if not belief_id:
        return
    rec = {"ts": time.time(), "belief_id": belief_id, "question": (question or "")[:300],
           "conclusion": conclusion, "scored": False}
    _pending[belief_id] = rec
    _append_jsonl(ARP_SHADOW_LOG, rec)


def record_operator_answer(belief_id: str, validation: str) -> None:
    """THE EARNING HOOK. When the operator answers a belief that had a prior shadow conclusion, score
    the match and record it into the (separate, zero-authority) gate. Coarse proxy: the research having
    surfaced evidence ↔ the operator confirming. Safe no-op when there is no pending shadow conclusion."""
    rec = _pending.pop(belief_id, None)
    if rec is None:
        return
    try:
        shadow_supports = rec["conclusion"].get("evidence") in ("strong", "weak")
        operator_supports = (validation == "confirmed")
        matched = (shadow_supports == operator_supports)
        AutonomousResearchPromotion.get_instance().record_outcome(matched)
        _append_jsonl(ARP_MATCHED_LOG, {
            "ts": time.time(), "belief_id": belief_id, "validation": validation,
            "shadow_evidence": rec["conclusion"].get("evidence"), "matched": matched,
        })
    except Exception:
        logger.debug("P5b: record_operator_answer failed", exc_info=True)


async def shadow_research(query_iface: Any, belief_id: str, question: str, facet: str) -> bool:
    """P5b SHADOW EXECUTE. Auto-fire academic_search IN SHADOW for a factual belief-tension question,
    derive the would-be conclusion, store + log it — MUTATE NOTHING. Honors the governor + dedup + the
    rate-limit BEFORE firing. Default-OFF (ENABLE_P5B_SHADOW). Never raises. Returns True if it fired."""
    if not ENABLE_P5B_SHADOW or facet != "factual" or not (belief_id and question):
        return False
    try:
        # Governor mode-gate (research only in the allowed consciousness modes).
        try:
            from autonomy.research_governor import ResearchGovernor
            gov = ResearchGovernor.get_instance()
            if hasattr(gov, "is_research_allowed") and not gov.is_research_allowed():
                return False
        except Exception:
            pass
        # Dedup: never re-research a settled topic.
        try:
            from autonomy.knowledge_integrator import KnowledgeIntegrator
            ki = KnowledgeIntegrator.get_instance()
            prior = ki.check_prior_knowledge(question) if hasattr(ki, "check_prior_knowledge") else None
            if isinstance(prior, dict) and prior.get("recommendation") == "skip":
                return False
        except Exception:
            pass
        # Fire academic_search IN SHADOW via the existing executor.
        from autonomy.research_intent import ResearchIntent
        intent = ResearchIntent(question=question, source_hint="academic", scope="external_ok",
                                 max_results=8)
        result = await query_iface._execute_academic(intent)
        conclusion = derive_conclusion(result)
        record_shadow_conclusion(belief_id, question, conclusion)
        logger.info("P5b shadow: belief=%s evidence=%s n=%d (mutated nothing)",
                    belief_id[:12], conclusion["evidence"], conclusion["n_findings"])
        return True
    except Exception:
        logger.debug("P5b: shadow_research failed (no-op)", exc_info=True)
        return False


def get_status() -> dict[str, Any]:
    """Telemetry surface — the separate gate + the pending/scored counts. PRE-MATURE."""
    st = AutonomousResearchPromotion.get_instance().get_status()
    st["pending_shadow_conclusions"] = len(_pending)
    st["phase"] = "P5b_shadow_execute"
    st["mutates_beliefs"] = False
    return st
