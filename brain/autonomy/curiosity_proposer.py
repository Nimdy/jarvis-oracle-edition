"""Shadow would-attend proposer — STEP 2 of the self-sensing -> curiosity bridge.

Consumes the self-sensing signal CARRIED into ``DriveSignals`` by STEP 1 and
logs what the system WOULD attend to: "she'd explore sector X because that's
where she's learning most" — plus every no-target tick. Pure observability, the
third link of the loop (notices -> gets curious -> WOULD attend):

  * authority = NONE. It sets no drive urgency, writes no belief and no memory,
    and returns nothing any lever reads. It only records would-have decisions.
  * non-fabrication: it proposes ONLY when self-sensing emitted a real
    ``curiosity_target`` — never on a STARVED / static scene. That firewall lives
    upstream in ``cognition/self_sensing._curiosity_target`` (best-LP > 0.01 on a
    sector with real recent motion); here we simply honour whatever it carried.
  * no silent gaps: no-target ticks are COUNTED, with the regime that caused
    them, so a quiet desk reads as an honest "nothing to attend", not silence.
  * bounded / memory-safe: recent proposals live in a fixed-size deque; nothing
    grows without bound and nothing touches canonical memory. Durable
    persistence + boot-rehydrate is STEP 6, deliberately NOT here (counters reset
    on restart until then — honest recovery, like the other shadow ledgers).

It earns a lever only later: STEP 3 (offline critic-test — does attending the
LP-target actually reduce next-frame error beyond persistence, shuffle/negative
controlled?) then STEP 4 (advisory nudge, only on an externally-attributed
causal win). Until then it is shadow-forever safe.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

_RECENT_MAXLEN = 200


class ShadowCuriosityProposer:
    """Read-only would-attend recorder over the carried self-sensing signal."""

    def __init__(self) -> None:
        self._ticks = 0
        self._proposals = 0
        self._no_target = 0
        self._regime_counts: dict[str, int] = {}
        self._sector_counts: dict[int, int] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=_RECENT_MAXLEN)
        self._last_proposal: dict[str, Any] | None = None

    def observe(self, signals: Any, *, now: float | None = None) -> None:
        """Record ONE shadow decision from the STEP-1 carried signal.

        Pure read of ``signals`` — never mutates it, never returns an actionable
        value. Fully exception-safe: a failure here must never perturb drive
        collection (this is a one-way tap, not inline in any lever)."""
        try:
            self._ticks += 1
            regime = str(getattr(signals, "self_sensing_regime", "unknown") or "unknown")
            self._regime_counts[regime] = self._regime_counts.get(regime, 0) + 1
            sector = getattr(signals, "self_sensing_target_sector", None)
            if sector is None:
                # honest no-target tick (quiet / plateaued scene) — counted, not hidden
                self._no_target += 1
                return
            sector = int(sector)
            deg = getattr(signals, "self_sensing_target_deg", None)
            prop: dict[str, Any] = {
                "sector": sector,
                "deg": deg,
                "learning_progress": round(float(getattr(signals, "self_sensing_target_lp", 0.0) or 0.0), 4),
                "skill": round(float(getattr(signals, "self_sensing_target_skill", 0.0) or 0.0), 4),
                "regime": regime,
                "lp_global": round(float(getattr(signals, "self_sensing_lp", 0.0) or 0.0), 5),
                "would_attend": ("sector %d (%s deg) — highest learning-progress right now; "
                                 "SHADOW, influences nothing" % (sector, str(deg))),
            }
            if now is not None:
                prop["ts"] = round(float(now), 3)
            self._proposals += 1
            self._sector_counts[sector] = self._sector_counts.get(sector, 0) + 1
            self._recent.append(prop)
            self._last_proposal = prop
        except Exception:
            logger.debug("curiosity proposer observe failed", exc_info=True)

    def get_status(self) -> dict[str, Any]:
        target_rate = (self._proposals / self._ticks) if self._ticks else 0.0
        return {
            "phase": "P0_would_attend_shadow",
            "authority": "zero_authority",
            "drives_levers": False,
            "ticks": self._ticks,
            "proposals": self._proposals,
            "no_target_ticks": self._no_target,
            "target_rate": round(target_rate, 4),
            "regime_counts": dict(self._regime_counts),
            "sector_counts": {str(k): v for k, v in sorted(self._sector_counts.items())},
            "last_proposal": self._last_proposal,
            "recent": list(self._recent)[-20:],
            "note": ("would-attend proposals off the carried self-sensing target; SHADOW / "
                     "authority=none — logs where attention COULD go plus every no-target tick "
                     "(honest, no silent gaps). Earns a lever only via the STEP 3 critic-test."),
        }


_proposer: ShadowCuriosityProposer | None = None


def get_curiosity_proposer() -> ShadowCuriosityProposer:
    """Module singleton — one shadow proposer for the process."""
    global _proposer
    if _proposer is None:
        _proposer = ShadowCuriosityProposer()
    return _proposer
