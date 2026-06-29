"""OSV Phase D — turn the self-view's first-class GAPS into curiosity targets.

The Operational Self-View (``cognition/self_view``) already knows, honestly, what JARVIS *cannot*
account for: a first-class list of ``gaps`` (code-grounded architecture gaps, comparator-less
scoreboard categories, absent performance facts, missing self-beliefs). Phase D points curiosity at
those gaps so JARVIS investigates its own blind spots — the prerequisite for *purposeful* self-
improvement rather than metric navel-gazing.

DISCIPLINE (earn-don't-declare + David's memory guardrail):

* **Shadow + gated.** At default autonomy this only *proposes* (computes + surfaces). It enqueues real
  research ONLY when the earned gate is met (``autonomy_level >= GAP_CURIOSITY_MIN_LEVEL`` — the same
  tier MetricTriggers use for auto-escalation). Default L1 → zero behavior, zero research.
* **Memory-safe.** This module NEVER writes a memory. When (earned) it enqueues, it rides the
  orchestrator's ``enqueue`` which dedups against queue + completions + persisted trace (so each gap is
  researched *at most once, ever* — never per-cycle) and is bounded (``MAX_QUEUE_SIZE``). Intents are
  ``scope="local_only"`` (read-only codebase research — no file mutation, no golden-auth).
* **Deduped + bounded** at the proposal layer too: one proposal per stable gap-key, capped at
  ``MAX_PROPOSALS``; the shadow surfacing buffer is a bounded deque.
* **Anti-theater.** Scoreboard "insufficient samples / no comparator" gaps are NOT proposed — those
  close with lived reps, not research. We only target genuinely investigable gaps.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from autonomy.research_intent import ResearchIntent

logger = logging.getLogger(__name__)

# Earned gate: the autonomy level at/above which gap-curiosity actually enqueues research.
# Matches MetricTriggers' auto-escalation tier. Below this it is pure shadow.
GAP_CURIOSITY_MIN_LEVEL = 3
MAX_PROPOSALS = 8          # bound on proposals produced per build
MAX_SHADOW_LOG = 20        # bound on the surfacing buffer


def _is_researchable(gap: dict[str, Any]) -> bool:
    """Scoreboard sample/comparator gaps need lived reps, not curiosity — skip them (anti-theater)."""
    area = str(gap.get("area", ""))
    return not area.startswith("scoreboard")


def _gap_key(gap: dict[str, Any]) -> str:
    """Stable key so the same gap maps to the same proposal/question every build (dedup)."""
    area = str(gap.get("area", "")).strip().lower()
    reason = str(gap.get("reason", "") or "")[:60].strip().lower()
    return f"{area}|{reason}"


def _gap_question(gap: dict[str, Any]) -> str:
    area = str(gap.get("area", ""))
    reason = str(gap.get("reason", "") or "").strip()
    if area == "architecture":
        return (f"My self-view flags an architecture gap: {reason}. "
                "What is the current state of this in my codebase, and what would close it?")
    return f"My self-view has a gap in {area}: {reason}. Can I investigate and close it?"


def _gap_priority(gap: dict[str, Any]) -> float:
    area = str(gap.get("area", ""))
    if area == "architecture":
        return 0.55
    if area.startswith("performance"):
        return 0.45
    if area.startswith("belief"):
        return 0.40
    return 0.35


class OSVGapCuriosity:
    """Proposes (shadow) and — only when earned — enqueues curiosity research for self-view gaps."""

    def __init__(self) -> None:
        self._shadow_log: deque[dict[str, Any]] = deque(maxlen=MAX_SHADOW_LOG)
        self._last_summary: dict[str, Any] = {}

    def proposals(self, model: dict[str, Any], max_n: int = MAX_PROPOSALS) -> list[dict[str, Any]]:
        """Pure: self-view model -> deduped, bounded, priority-sorted curiosity proposals."""
        gaps = (model or {}).get("gaps") or []
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for g in gaps:
            if not isinstance(g, dict) or not _is_researchable(g):
                continue
            key = _gap_key(g)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "key": key,
                "area": str(g.get("area", "")),
                "reason": str(g.get("reason", "") or "")[:200],
                "question": _gap_question(g),
                "priority": _gap_priority(g),
            })
        out.sort(key=lambda p: -p["priority"])
        return out[:max_n]

    def feed(self, orchestrator: Any, autonomy_level: int,
             model: dict[str, Any]) -> dict[str, Any]:
        """Compute proposals (always); enqueue research ONLY when earned (gated).

        Returns a summary dict. Always pushes a bounded shadow record. Never writes a memory.
        """
        props = self.proposals(model)
        gated = autonomy_level < GAP_CURIOSITY_MIN_LEVEL
        enqueued = 0
        if not gated and orchestrator is not None:
            for p in props:
                intent = ResearchIntent(
                    question=p["question"],
                    source_event="osv_gap_curiosity",
                    source_hint="codebase",
                    priority=p["priority"],
                    scope="local_only",          # read-only research; no mutation, no golden-auth
                    tag_cluster=("self_view", "gap", p["area"][:24]),
                )
                try:
                    if orchestrator.enqueue(intent):   # dedups (queue+completed+trace) + bounded
                        enqueued += 1
                except Exception:
                    logger.debug("gap-curiosity enqueue failed", exc_info=True)
        summary = {
            "proposed": len(props),
            "enqueued": enqueued,
            "gated": gated,
            "min_level": GAP_CURIOSITY_MIN_LEVEL,
            "autonomy_level": autonomy_level,
        }
        self._shadow_log.append({
            **summary,
            "top": [{"area": p["area"], "question": p["question"][:140]} for p in props[:5]],
        })
        self._last_summary = summary
        return summary

    def shadow_state(self) -> dict[str, Any]:
        """Bounded, observable shadow state for surfacing (read-only)."""
        return {"summary": dict(self._last_summary), "recent": list(self._shadow_log)}
