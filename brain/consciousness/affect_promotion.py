"""Shadow-to-active promotion controller for the affect layer (SPARK_DESIGN §3 #3, §8).

Mirrors :class:`cognition.promotion.WorldModelPromotion` (the proven
shadow→advisory→active, accuracy-gated, auto-demoting pattern) for the
dopamine/serotonin/cortisol affect readout.

Levels:
  0 — shadow:   compute scalars + propose levers, write to the attribution
                ledger ONLY.  Drives NO real lever (cadence, reward, curiosity
                weight, user-facing token, autonomy action).  **DEFAULT.**
  1 — advisory: affect readouts appear (gate-filtered, labelled) and lightly
                bias curiosity ranking (P4 wires the coupling; gated off here).
  2 — active:   cadence/reward coupling (P5; behind its own kill-switch).

This module ships at **level 0** and **never flips itself to active** — it
implements the mechanism and its gate; promotion is earned the §8 way and is
the operator's call.  Promotion criteria (§8 P1): backing rate ≥0.65 over ≥50
paired observations, ``MIN_SHADOW_HOURS`` (4.0, **live-tick**) elapsed, no
scalar pinned at ceiling/floor, provenance completeness 100%.

The §7 "success is never the mechanism's own dial" rule is honoured: the gate
freezes if the self-prompt proxy rises without ``grounded:inferred`` falling —
but P1 only *records* the paired observations; it does not act, so the controller
exposes the eligibility computation and auto-demotes on a governor freeze signal,
and otherwise leaves the gate shut.
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

logger = logging.getLogger(__name__)

AFFECT_PROMOTION_PATH = os.path.join(
    os.path.expanduser("~"), ".jarvis", "affect_promotion.json",
)

# §8 P1 advancement gate.
MIN_PAIRED_OBSERVATIONS = 50
MIN_SHADOW_HOURS = 4.0          # live-tick (not wall-clock) — anti reset-gaming
PROMOTE_BACKING_RATE = 0.65
DEMOTE_BACKING_RATE = 0.50      # §10.3 backing rate <0.50 over 20 ticks → demote
DEMOTE_WINDOW = 20
TRANSITION_COOLDOWN_S = 300.0   # mirrors promotion.py TRANSITION_COOLDOWN_S


@dataclass
class _AffectPromotionState:
    level: int = 0  # 0=shadow, 1=advisory, 2=active — DEFAULT 0
    shadow_start_ts: float = field(default_factory=time.time)
    # Live-tick clock: accrued seconds of measured ticks, persisted so a restart
    # can't reset-game MIN_SHADOW_HOURS (§7).
    live_tick_seconds: float = 0.0
    total_paired: int = 0
    backing_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    last_promoted_at: float = 0.0
    last_demoted_at: float = 0.0
    auto_demoted_for_oscillation: bool = False


class AffectPromotion:
    """Backing-rate-gated promotion controller for the affect layer.

    Defaults to shadow (level 0).  In shadow, :meth:`record_shadow_tick` writes
    the computed scalars + proposed levers to the attribution ledger and accrues
    a paired backing observation for the promotion gate — and nothing else.
    """

    def __init__(self) -> None:
        self._state = _AffectPromotionState()
        self._load()

    @property
    def level(self) -> int:
        return self._state.level

    @property
    def is_shadow(self) -> bool:
        return self._state.level == 0

    @property
    def is_advisory(self) -> bool:
        """True at level 1 (SPARK §8 P4): affect readouts APPEAR (gate-filtered,
        labelled) and lightly bias curiosity ranking — but drive NO cadence/reward
        lever (that coupling is P5/active, behind its own kill-switch)."""
        return self._state.level == 1

    @property
    def is_active(self) -> bool:
        """True only at level 2 (P5). Cadence/reward coupling lives here."""
        return self._state.level >= 2

    # ── ADVISORY (level 1) readout + light curiosity bias (SPARK §8 P4) ──────

    def advisory_readout(self, snapshot: Any | None = None) -> dict[str, Any]:
        """Gate-filtered, LABELLED affect readout for advisory display (SPARK §8 P4).

        Returns an honesty-safe payload: each scalar's regulated level, raw value,
        provenance, and a one-line summary that has been run through the
        CapabilityGate (so any "I feel X" nickname framing is rewritten to name the
        real signal, §7). Empty/neutral when not advisory or no snapshot — advisory
        only *surfaces* the readout; it never asserts a feeling.
        """
        snap = snapshot if snapshot is not None else self._last_snapshot()
        if snap is None or not self.is_advisory:
            return {
                "visible": False,
                "level_name": self._level_name(),
                "note": "affect readout hidden — layer not at advisory",
            }
        out: dict[str, Any] = {
            "visible": True,
            "level_name": self._level_name(),
            "labelled": True,
            "drives_levers": False,  # advisory NEVER drives cadence/reward
            "scalars": {},
        }
        try:
            from skills.capability_gate import capability_gate
        except Exception:
            capability_gate = None  # type: ignore[assignment]
        for ro in (snap.dopamine, snap.serotonin, snap.cortisol):
            try:
                raw_summary = (
                    f"{ro.nickname} is {('elevated' if ro.level > 0.55 else 'low' if ro.level < 0.45 else 'neutral')}"
                )
                # Run through the gate so the nickname is rewritten to its real
                # signal (cannot-lie / no-feeling, §7).
                summary = (
                    capability_gate.check_text(raw_summary)
                    if capability_gate is not None else raw_summary
                )
                out["scalars"][ro.nickname] = {
                    "level": round(float(ro.level), 4),
                    "raw": round(float(ro.raw), 4),
                    "all_sources_zero": bool(ro.all_sources_zero),
                    "cannot_lie_clamped": bool(ro.cannot_lie_clamped),
                    "provenance": ro.provenance,
                    "labelled_summary": summary,
                }
            except Exception:
                logger.debug("advisory_readout scalar render failed", exc_info=True)
        return out

    def curiosity_bias(self, snapshot: Any | None = None) -> float:
        """Light, bounded additive curiosity-ranking bias (SPARK §4/§8 P4).

        Returns 0.0 unless advisory (so DEFAULT runtime behaviour is unchanged).
        At advisory, dopamine above neutral nudges curiosity up; the magnitude is
        small (`0.20·(d−0.5)`, SPARK §4) and clamped, so it lightly biases ranking
        without conjuring a drive from zero. NEVER touches cadence/reward.
        """
        if not self.is_advisory:
            return 0.0
        snap = snapshot if snapshot is not None else self._last_snapshot()
        if snap is None:
            return 0.0
        try:
            d = float(snap.dopamine.level)
            bias = 0.20 * (d - 0.5)
            return max(-0.10, min(0.10, bias))
        except Exception:
            return 0.0

    def _last_snapshot(self) -> Any | None:
        try:
            from consciousness.affect_state import affect_state
            return affect_state.snapshot()
        except Exception:
            return None

    # ── Shadow tick: ledger-only write + paired-observation accrual ──────────

    def record_shadow_tick(self, snapshot: Any, *, tick_dt_s: float = 0.0) -> str:
        """Write the affect snapshot to the attribution ledger (shadow-only).

        ``snapshot`` is an :class:`consciousness.affect_state.AffectSnapshot`.
        Returns the ledger entry id (empty string on failure).  Accrues one
        paired backing observation (whether each elevated scalar is backed by a
        non-zero real source) and advances the live-tick clock.  Applies NO
        lever.  Auto-demotes if the governor signalled excessive oscillation.
        """
        # Advance the live-tick soak clock (anti reset-gaming).
        if tick_dt_s > 0:
            self._state.live_tick_seconds += tick_dt_s

        backing = self._paired_backing(snapshot)
        if backing is not None:
            self._state.total_paired += 1
            self._state.backing_history.append(1.0 if backing else 0.0)

        # Governor freeze → auto-demote to shadow (§5.5).
        if getattr(snapshot, "demote_signal", False) and self._state.level > 0:
            self._auto_demote("affect governor oscillation freeze")

        entry_id = self._write_ledger(snapshot)
        # Promotion is the operator's call and gated; we only *check* eligibility
        # and never flip ourselves to active in P1.  Auto-demote remains active.
        self._maybe_auto_demote()
        # Persist the EARNED soak clock + backing evidence every tick (the spark-bug
        # lesson, df7c8bb). record_shadow_tick is the ONLY accrual path for
        # live_tick_seconds / total_paired / backing_history; without a save here they
        # live only in RAM and reset to 0 every reboot, so the layer can NEVER clear
        # MIN_SHADOW_HOURS and is pinned in shadow forever. Small atomic write; this
        # makes the earned evidence durable only — authority stays operator-gated.
        self.save()
        return entry_id

    @staticmethod
    def _paired_backing(snapshot: Any) -> bool | None:
        """A scalar that is *elevated* (raw > 0) must be *backed* (a real source).

        Returns True if every elevated scalar this tick had a non-zero source
        reading (i.e. not confabulated), False if any elevated scalar was unbacked,
        or None if nothing was elevated this tick (no observation to score).
        """
        try:
            readouts = [snapshot.dopamine, snapshot.serotonin, snapshot.cortisol]
        except Exception:
            return None
        elevated = [r for r in readouts if r.raw > 0.0]
        if not elevated:
            return None
        # Backed iff the readout was NOT all-sources-zero (cannot-lie clamp would
        # have forced raw to 0 if it were, so an elevated+all_zero is impossible,
        # but we assert backing explicitly for the honesty contract).
        return all(not r.all_sources_zero for r in elevated)

    def _write_ledger(self, snapshot: Any) -> str:
        try:
            from consciousness.attribution_ledger import attribution_ledger
            data: dict[str, Any] = {
                "level": self._state.level,
                "level_name": self._level_name(),
            }
            try:
                data.update(snapshot.to_dict())
            except Exception:
                pass
            return attribution_ledger.record(
                subsystem="affect_state",
                event_type="affect_shadow_readout",
                source="affect_tick",
                data=data,
            )
        except Exception:
            logger.debug("Affect shadow ledger write failed", exc_info=True)
            return ""

    # ── Gate computation (eligibility only; never self-promotes in P1) ───────

    def _backing_rate(self) -> float:
        hist = list(self._state.backing_history)
        return sum(hist) / len(hist) if hist else 0.0

    def _provenance_complete(self, snapshot: Any | None) -> bool:
        if snapshot is None:
            return False
        try:
            for r in (snapshot.dopamine, snapshot.serotonin, snapshot.cortisol):
                if not r.provenance:
                    return False
        except Exception:
            return False
        return True

    def _any_scalar_pinned(self, snapshot: Any | None) -> bool:
        """No scalar may be pinned at the clamp ceiling/floor (§8 P1)."""
        if snapshot is None:
            return False
        try:
            from consciousness.affect_regulation import CLAMP_LO, CLAMP_HI
            for r in (snapshot.dopamine, snapshot.serotonin, snapshot.cortisol):
                if r.level <= CLAMP_LO + 1e-6 or r.level >= CLAMP_HI - 1e-6:
                    return True
        except Exception:
            return False
        return False

    def promotion_eligible(self, snapshot: Any | None = None) -> bool:
        """§8 P1 eligibility — does NOT promote; informs the operator/dashboard."""
        if self._state.level >= 2:
            return False
        if self._state.total_paired < MIN_PAIRED_OBSERVATIONS:
            return False
        if self._state.live_tick_seconds / 3600.0 < MIN_SHADOW_HOURS:
            return False
        if len(self._state.backing_history) < MIN_PAIRED_OBSERVATIONS:
            return False
        if self._backing_rate() < PROMOTE_BACKING_RATE:
            return False
        if self._any_scalar_pinned(snapshot):
            return False
        if not self._provenance_complete(snapshot):
            return False
        return True

    def _maybe_auto_demote(self) -> None:
        now = time.time()
        last_transition = max(self._state.last_promoted_at, self._state.last_demoted_at)
        if last_transition > 0 and (now - last_transition) < TRANSITION_COOLDOWN_S:
            return
        hist = list(self._state.backing_history)
        if len(hist) >= DEMOTE_WINDOW and self._state.level > 0:
            recent = hist[-DEMOTE_WINDOW:]
            if (sum(recent) / len(recent)) < DEMOTE_BACKING_RATE:
                self._auto_demote(
                    f"backing rate {sum(recent)/len(recent):.2f} < {DEMOTE_BACKING_RATE}"
                )

    def _auto_demote(self, reason: str) -> None:
        old = self._state.level
        self._state.level = max(self._state.level - 1, 0)
        if self._state.level != old:
            self._state.last_demoted_at = time.time()
            self._state.shadow_start_ts = time.time()
            self._state.auto_demoted_for_oscillation = "oscillation" in reason
            logger.warning("Affect layer demoted: level %d → %d (%s)",
                           old, self._state.level, reason)
            self.save()

    def _level_name(self) -> str:
        return {0: "shadow", 1: "advisory", 2: "active"}.get(self._state.level, "unknown")

    def get_status(self) -> dict[str, Any]:
        return {
            "level": self._state.level,
            "level_name": self._level_name(),
            "total_paired": self._state.total_paired,
            "backing_rate": round(self._backing_rate(), 3),
            "backing_window_size": len(self._state.backing_history),
            "live_tick_hours": round(self._state.live_tick_seconds / 3600.0, 2),
            "promotion_ready": self.promotion_eligible(),
            "auto_demoted_for_oscillation": self._state.auto_demoted_for_oscillation,
            "last_promoted_at": self._state.last_promoted_at,
            "last_demoted_at": self._state.last_demoted_at,
            # SPARK §8 P4 — advisory affect readout (gate-filtered, labelled).
            # Hidden / no-lever unless promoted to advisory; ALWAYS drives_levers=False.
            "advisory_readout": self.advisory_readout(),
            "curiosity_bias": round(self.curiosity_bias(), 4),
        }

    # ── Persistence (mirrors promotion.py atomic write) ──────────────────────

    def save(self) -> None:
        data = {
            "level": self._state.level,
            "shadow_start_ts": self._state.shadow_start_ts,
            "live_tick_seconds": self._state.live_tick_seconds,
            "total_paired": self._state.total_paired,
            "backing_history": list(self._state.backing_history),
            "last_promoted_at": self._state.last_promoted_at,
            "last_demoted_at": self._state.last_demoted_at,
            "auto_demoted_for_oscillation": self._state.auto_demoted_for_oscillation,
        }
        try:
            path = Path(AFFECT_PROMOTION_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(path)
        except Exception:
            logger.debug("Failed to save affect promotion state", exc_info=True)

    def _load(self) -> None:
        try:
            path = Path(AFFECT_PROMOTION_PATH)
            if not path.exists():
                return
            data = json.loads(path.read_text())
            self._state.level = int(data.get("level", 0))
            self._state.shadow_start_ts = float(data.get("shadow_start_ts", time.time()))
            self._state.live_tick_seconds = float(data.get("live_tick_seconds", 0.0))
            self._state.total_paired = int(data.get("total_paired", 0))
            for v in data.get("backing_history", []):
                self._state.backing_history.append(float(v))
            self._state.last_promoted_at = float(data.get("last_promoted_at", 0.0))
            self._state.last_demoted_at = float(data.get("last_demoted_at", 0.0))
            self._state.auto_demoted_for_oscillation = bool(
                data.get("auto_demoted_for_oscillation", False)
            )
            logger.info("Affect promotion restored: level=%d, paired=%d",
                        self._state.level, self._state.total_paired)
        except Exception:
            logger.debug("Failed to load affect promotion state", exc_info=True)


affect_promotion = AffectPromotion()
