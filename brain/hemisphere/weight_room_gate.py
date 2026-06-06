"""Weight-Room P2: per-specialist lived-baseline registry + would-block evaluator (SHADOW).

Computes, per distilled specialist, whether it WOULD be blocked from gaining promotion
authority under the lived-before-synthetic rule (AWAKENING_PROTOCOL.md:116 — "lived
baseline first, synthetic amplification second, promotion third, never in reverse"),
and LOGS the decision. It enforces NOTHING — P3 is the first phase that denies
authority. See docs/WEIGHT_ROOM_DESIGN.md §3 components (3)+(5), §6 rollout P2.

Design constraints honoured here:
  - ASYMMETRIC gate: training + shadow inference are never touched; only a (future)
    AUTHORITY surface would be gated. This module only reads counts and writes a log.
  - FALSE-BLOCK guard: the floor is PER-SPECIALIST, not one-size-fits-all. Rare-event
    specialists use a low/hybrid floor; specialists with no live signal source are
    honestly marked blocked-by-design (a state, not a floor they can never meet), not
    silently stalled. A uniform high floor would permanently stall them — forbidden.
  - HONESTY: synthetic is the default; lived counts come from real origin telemetry
    (weight-room P0), never from a specialist grading itself. Every would-block AND
    every grandfather exemption is logged + surfaced (no silent carve-outs).
  - FAIL-CLOSED-TO-SHADOW: any error here only affects the shadow log.

Singleton: ``weight_room_gate``.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# -- Decision modes ---------------------------------------------------------
MODE_HYBRID = "rare_event_hybrid"      # lived_baseline_met = synthetic >= S AND lived >= L
MODE_LIVED = "lived"                   # lived_baseline_met = lived >= L
MODE_BLOCKED_BY_DESIGN = "blocked_by_design"  # no live signal source yet (correct to be shadow)
MODE_NOT_YET_GATABLE = "not_yet_gatable"      # no origin instrumentation — must be wired first
MODE_EXEMPT = "exempt"                 # grandfathered always-on Tier-1 (logged, never silent)

# Six always-on Tier-1 specialists already promoted with no lifecycle_stage — retro-
# validation is impossible and would regress working perception. Explicitly exempt +
# LOGGED (the honesty rule applies to the gate's own carve-outs). New specialists are
# NOT exempt. (docs/WEIGHT_ROOM_DESIGN.md §4.)
EXEMPT_SPECIALISTS = frozenset({
    "emotion_depth", "speaker_repr", "face_repr",
    "perception_fusion", "voice_intent", "speaker_diarize",
})

# Rare-event hybrid floor: lived ground truth is scarce for these, so a uniform high
# floor would never be met. Hybrid = "enough synthetic coverage AND a few real reps".
_HYBRID_MIN_SYNTHETIC = 50
_HYBRID_MIN_LIVED = 3
# Standard lived floor for specialists with a routine live signal.
_LIVED_MIN = 10


def classify(teacher: str) -> dict[str, Any]:
    """Map a teacher/specialist name to its lived-baseline regime.

    Substring-based so it is robust to the collector's signal-source suffixes
    (e.g. ``skill_acquisition_features`` / ``skill_acquisition_outcome``). Unknown
    specialists default to NOT_YET_GATABLE — the conservative honest state ("instrument
    its origin before it can be gated"), never a silent allow.
    """
    t = (teacher or "").lower()
    if t in EXEMPT_SPECIALISTS:
        return {"mode": MODE_EXEMPT, "reason": "grandfathered always-on Tier-1 (no lifecycle_stage; logged)"}
    # blocked-by-design: a real live-signal SOURCE does not exist yet
    if t.startswith("thought_trigger"):
        return {"mode": MODE_BLOCKED_BY_DESIGN,
                "reason": "no live signal until THOUGHT_VALIDATION_OUTCOME traffic flows (wired, input-starved)"}
    if t.startswith("code_quality"):
        return {"mode": MODE_BLOCKED_BY_DESIGN,
                "reason": "no live code-quality outcome signal wired"}
    # rare-event hybrid: scarce lived ground truth -> low/hybrid floor
    if (t.startswith("skill_acquisition") or t.startswith("plan")
            or t.startswith("claim_classifier") or t.startswith("diagnostic")):
        return {"mode": MODE_HYBRID, "min_synthetic": _HYBRID_MIN_SYNTHETIC, "min_lived": _HYBRID_MIN_LIVED}
    # routine lived signal available
    if t.startswith("dream"):
        return {"mode": MODE_LIVED, "min_lived": _LIVED_MIN}
    return {"mode": MODE_NOT_YET_GATABLE,
            "reason": "no origin instrumentation for this specialist — wire component (1) first"}


class WeightRoomGate:
    """Shadow would-block evaluator over the distilled specialists. Authority: NONE."""

    _instance: "WeightRoomGate | None" = None

    @classmethod
    def get_instance(cls) -> "WeightRoomGate":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._last_decisions: dict[str, str] = {}  # for change-only logging

    def _evaluate_one(self, teacher: str, lived: int, synthetic: int) -> dict[str, Any]:
        reg = classify(teacher)
        mode = reg["mode"]
        out: dict[str, Any] = {
            "specialist": teacher, "mode": mode,
            "lived": int(lived), "synthetic": int(synthetic),
        }

        if mode == MODE_EXEMPT:
            out.update(decision="exempt", lived_baseline_met=True, reason=reg["reason"])
            return out
        if mode in (MODE_BLOCKED_BY_DESIGN, MODE_NOT_YET_GATABLE):
            # not a would-block (it isn't even eligible to be measured yet) — a distinct,
            # honest state so it is never confused with "failed the floor".
            out.update(decision=mode, lived_baseline_met=False, reason=reg["reason"])
            return out

        if mode == MODE_HYBRID:
            s_min, l_min = reg["min_synthetic"], reg["min_lived"]
            met = synthetic >= s_min and lived >= l_min
            reason = (f"hybrid floor met (synthetic {synthetic}>={s_min} AND lived {lived}>={l_min})"
                      if met else
                      f"lived baseline NOT met (need synthetic>={s_min} AND lived>={l_min}; "
                      f"have synthetic={synthetic}, lived={lived})")
        else:  # MODE_LIVED
            l_min = reg["min_lived"]
            met = lived >= l_min
            reason = (f"lived floor met (lived {lived}>={l_min})" if met
                      else f"lived baseline NOT met (need lived>={l_min}; have {lived})")

        out.update(
            decision="would_allow" if met else "would_block",
            lived_baseline_met=met,
            reason=reason,
        )
        return out

    def evaluate_all(self) -> dict[str, Any]:
        """Compute would-block decisions for every specialist seen in the collector.

        Pure read of weight-room P0 per-teacher lived/synthetic counts. Logs only when a
        decision CHANGES (no per-poll spam). Never raises (fail-closed-to-shadow)."""
        decisions: dict[str, dict[str, Any]] = {}
        try:
            from hemisphere.distillation import DistillationCollector
            collector = DistillationCollector.instance()
            per_teacher = (collector.get_stats().get("teachers", {}) if collector else {}) or {}
        except Exception:
            logger.debug("WeightRoomGate: collector stats unavailable", exc_info=True)
            per_teacher = {}

        for teacher, st in per_teacher.items():
            if not isinstance(st, dict):
                continue
            d = self._evaluate_one(teacher, st.get("lived", 0) or 0, st.get("synthetic", 0) or 0)
            decisions[teacher] = d
            prev = self._last_decisions.get(teacher)
            if prev != d["decision"]:
                self._last_decisions[teacher] = d["decision"]
                if d["decision"] == "would_block":
                    logger.info("WeightRoom would-block (SHADOW, enforces nothing): %s — %s",
                                teacher, d["reason"])
                elif d["decision"] in (MODE_BLOCKED_BY_DESIGN, MODE_NOT_YET_GATABLE):
                    logger.info("WeightRoom %s: %s — %s", d["decision"], teacher, d["reason"])

        return decisions

    def get_status(self) -> dict[str, Any]:
        decisions = self.evaluate_all()
        summary: dict[str, int] = {}
        for d in decisions.values():
            summary[d["decision"]] = summary.get(d["decision"], 0) + 1
        return {
            "phase": "P2_lived_baseline_registry",
            "authority": "shadow_would_block_only",
            "enforces": False,
            "floors": {
                "hybrid": {"min_synthetic": _HYBRID_MIN_SYNTHETIC, "min_lived": _HYBRID_MIN_LIVED},
                "lived": {"min_lived": _LIVED_MIN},
            },
            "exempt_specialists": sorted(EXEMPT_SPECIALISTS),
            "summary": summary,
            "decisions": list(decisions.values()),
            "note": "would-block decisions are LOGGED + surfaced, never enforced (P3 enforces, advisory)",
        }


weight_room_gate = WeightRoomGate.get_instance()
