"""Operational Modes — higher-level behavioral states for the consciousness kernel.

Modes are distinct from phases (which describe what Jarvis is currently doing).
A mode shapes *how* Jarvis behaves: tick cadence, proactivity frequency,
response depth, memory reinforcement, and interruption sensitivity.

Modes can be set by heuristics (AttentionCore output) or later by the
neural policy layer.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Literal

from consciousness.events import event_bus

logger = logging.getLogger(__name__)

MODE_CHANGE = "mode:change"

OperationalMode = Literal[
    "gestation",       # Fresh brain — self-discovery, no human interaction
    "passive",         # No user present — low cadence, background tasks only
    "conversational",  # Active dialogue — high cadence, fast response, proactive
    "reflective",      # User present but idle — medium cadence, introspection
    "focused",         # User engaged in work — low proactivity, minimal interruption
    "sleep",           # Extended absence — minimal tick, save power
    "dreaming",        # After 5min sleep — memory consolidation, cross-association, reflection
    "deep_learning",   # Pi absent >1hr — max cadence, self-improvement, evolution
]

ALL_MODES: tuple[OperationalMode, ...] = (
    "gestation", "passive", "conversational", "reflective", "focused",
    "sleep", "dreaming", "deep_learning",
)

SLEEP_TO_DREAM_S = 300.0


ALL_CYCLES = frozenset({
    "meta_thought", "analysis", "evolution", "mutation", "existential",
    "dialogue", "hemisphere", "self_improvement", "dream", "study", "truth_calibration",
    "learning_jobs", "memory_maintenance", "association_repair",
    "cortex_training", "contradiction", "belief_graph", "quarantine", "goals",
    "scene_continuity", "world_model", "reflective_audit", "soul_integrity",
    "capability_discovery", "curiosity_questions", "onboarding",
    "shadow_lang", "health_monitor", "fractal_recall", "acquisition",
    "intention_stale_sweep", "intention_resolver",
})


@dataclass(frozen=True)
class ModeProfile:
    """Behavioral parameters for a mode."""
    tick_cadence_multiplier: float
    proactivity_cooldown_s: float
    response_depth_hint: Literal["brief", "normal", "detailed"]
    memory_reinforcement_multiplier: float
    interruption_sensitivity: float  # 0 = ignore all, 1 = respond to everything
    allowed_cycles: frozenset[str] = ALL_CYCLES


_SLEEP_CYCLES = frozenset({"meta_thought", "analysis", "dream", "memory_maintenance",
                           "association_repair", "cortex_training", "quarantine",
                           "contradiction", "truth_calibration", "belief_graph",
                           "world_model", "reflective_audit", "soul_integrity", "study",
                           "health_monitor", "acquisition", "intention_stale_sweep"})
_DREAM_CYCLES = frozenset({"analysis", "dream", "memory_maintenance", "association_repair",
                            "hemisphere", "evolution", "meta_thought", "cortex_training",
                            "contradiction", "truth_calibration", "belief_graph", "quarantine",
                            "goals", "scene_continuity", "world_model",
                            "reflective_audit", "soul_integrity",
                            "existential", "dialogue", "study", "learning_jobs",
                            "mutation", "self_improvement", "capability_discovery",
                            "health_monitor", "acquisition", "intention_stale_sweep"})

DEFAULT_PROFILES: dict[OperationalMode, ModeProfile] = {
    "gestation": ModeProfile(
        tick_cadence_multiplier=1.5,
        proactivity_cooldown_s=9999.0,
        response_depth_hint="detailed",
        memory_reinforcement_multiplier=2.0,
        interruption_sensitivity=0.0,
        allowed_cycles=ALL_CYCLES - frozenset({"acquisition"}),
    ),
    "passive": ModeProfile(
        tick_cadence_multiplier=0.5,
        proactivity_cooldown_s=300.0,
        response_depth_hint="brief",
        memory_reinforcement_multiplier=0.5,
        interruption_sensitivity=0.3,
    ),
    "conversational": ModeProfile(
        tick_cadence_multiplier=1.5,
        proactivity_cooldown_s=300.0,
        response_depth_hint="normal",
        memory_reinforcement_multiplier=1.5,
        interruption_sensitivity=1.0,
    ),
    "reflective": ModeProfile(
        tick_cadence_multiplier=0.8,
        proactivity_cooldown_s=120.0,
        response_depth_hint="detailed",
        memory_reinforcement_multiplier=1.2,
        interruption_sensitivity=0.6,
    ),
    "focused": ModeProfile(
        tick_cadence_multiplier=0.7,
        proactivity_cooldown_s=600.0,
        response_depth_hint="brief",
        memory_reinforcement_multiplier=0.8,
        interruption_sensitivity=0.2,
    ),
    "sleep": ModeProfile(
        tick_cadence_multiplier=0.2,
        proactivity_cooldown_s=3600.0,
        response_depth_hint="brief",
        memory_reinforcement_multiplier=0.3,
        interruption_sensitivity=0.1,
        allowed_cycles=_SLEEP_CYCLES,
    ),
    "dreaming": ModeProfile(
        tick_cadence_multiplier=0.5,
        proactivity_cooldown_s=600.0,
        response_depth_hint="detailed",
        memory_reinforcement_multiplier=2.0,
        interruption_sensitivity=0.8,
        allowed_cycles=_DREAM_CYCLES,
    ),
    "deep_learning": ModeProfile(
        tick_cadence_multiplier=2.0,
        proactivity_cooldown_s=60.0,
        response_depth_hint="detailed",
        memory_reinforcement_multiplier=1.5,
        interruption_sensitivity=0.8,
    ),
}


_MIN_DWELL_S: dict[OperationalMode, float] = {
    "gestation": 300.0,
    "passive": 15.0,
    "conversational": 5.0,
    "reflective": 20.0,
    "focused": 30.0,
    "sleep": 60.0,
    "dreaming": 60.0,
    "deep_learning": 120.0,
}

_ENTER_ENGAGEMENT: dict[OperationalMode, float] = {
    "gestation": 0.0,
    "conversational": 0.6,
    "reflective": 0.35,
    "focused": 0.0,
    "passive": 0.0,
    "sleep": 0.0,
    "dreaming": 0.0,
    "deep_learning": 0.0,
}

_EXIT_ENGAGEMENT: dict[OperationalMode, float] = {
    "gestation": 0.0,
    "conversational": 0.4,
    "reflective": 0.2,
    "focused": 0.0,
    "passive": 0.0,
    "sleep": 0.0,
    "dreaming": 0.0,
    "deep_learning": 0.0,
}


_MODE_RANK: dict[OperationalMode, int] = {
    "sleep": 0, "passive": 1, "focused": 2,
    "reflective": 3, "conversational": 4,
    "deep_learning": 5, "dreaming": 1, "gestation": 5,
}

BOOT_GRACE_S = 60.0


class ModeManager:
    """Manages the current operational mode and exposes its profile."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mode: OperationalMode = "passive"
        self._profile: ModeProfile = DEFAULT_PROFILES["passive"]
        self._since: float = time.time()
        self._history: list[dict[str, Any]] = []
        self._boot_grace_until: float = time.time() + BOOT_GRACE_S
        self._boot_grace_logged: bool = False
        self._last_grace_block_log: float = 0.0
        self._matrix_deep_learning: bool = False
        self._subscribe_matrix_event()

    def _subscribe_matrix_event(self) -> None:
        """Listen for Matrix Protocol deep-learning requests and completions."""
        try:
            from consciousness.events import (
                MATRIX_DEEP_LEARNING_REQUESTED,
                SKILL_LEARNING_COMPLETED,
            )
            event_bus.on(MATRIX_DEEP_LEARNING_REQUESTED, self._on_matrix_deep_learning)
            event_bus.on(SKILL_LEARNING_COMPLETED, self._on_learning_completed)
        except Exception:
            pass

    def _on_matrix_deep_learning(self, **_kwargs: Any) -> None:
        """Transition to deep_learning when a Matrix Protocol job starts."""
        self._matrix_deep_learning = True
        allowed = ("passive", "sleep", "dreaming", "reflective", "focused")
        with self._lock:
            if self._mode in allowed:
                pass  # release lock before calling set_mode
            else:
                logger.info(
                    "Matrix deep_learning requested but mode=%s — deferring",
                    self._mode,
                )
                return
        self.set_mode("deep_learning", reason="matrix_protocol_activated", force=True)

    def _on_learning_completed(self, **kwargs: Any) -> None:
        """Release deep_learning hold if Matrix Protocol job completes."""
        if not self._matrix_deep_learning:
            return
        self._matrix_deep_learning = False
        with self._lock:
            if self._mode == "deep_learning":
                pass  # release lock before set_mode
            else:
                return
        self.set_mode("passive", reason="matrix_learning_completed", force=True)
        logger.info("Matrix deep_learning released — job completed")

    @property
    def mode(self) -> OperationalMode:
        return self._mode

    @property
    def profile(self) -> ModeProfile:
        return self._profile

    def set_mode(self, mode: OperationalMode, reason: str = "", force: bool = False) -> bool:
        with self._lock:
            if mode == self._mode:
                return False
            if mode not in DEFAULT_PROFILES:
                logger.warning("Unknown mode: %s", mode)
                return False

            if not force:
                dwell = time.time() - self._since
                min_dwell = _MIN_DWELL_S.get(self._mode, 10.0)
                if dwell < min_dwell:
                    return False

            old = self._mode
            self._mode = mode
            self._profile = DEFAULT_PROFILES[mode]
            self._since = time.time()
            self._history.append({
                "from": old, "to": mode, "reason": reason, "time": self._since,
            })
            if len(self._history) > 50:
                self._history = self._history[-50:]

        event_bus.emit(MODE_CHANGE, from_mode=old, to_mode=mode, reason=reason)
        logger.info("Mode: %s → %s (%s)", old, mode, reason or "manual")
        return True

    def suggest_mode_from_attention(self, attention_state: dict[str, Any]) -> OperationalMode:
        """Heuristic mode selection with hysteresis-aware thresholds."""
        with self._lock:
            cur = self._mode
            since = self._since

        if cur == "gestation":
            return "gestation"
        if cur == "deep_learning":
            return "deep_learning"

        present = attention_state.get("person_present", False)
        engagement = attention_state.get("engagement_level", 0.0)
        last_interaction = attention_state.get("last_interaction_time", 0.0)
        idle_s = time.time() - last_interaction if last_interaction else 9999

        if not present:
            if cur == "sleep" and (time.time() - since) >= SLEEP_TO_DREAM_S:
                suggested = "dreaming"
            elif cur == "dreaming":
                suggested = "dreaming"
            else:
                suggested = "sleep" if idle_s > 600 else "passive"
        else:
            enter_conv = _ENTER_ENGAGEMENT["conversational"]
            exit_conv = _EXIT_ENGAGEMENT["conversational"]
            enter_refl = _ENTER_ENGAGEMENT["reflective"]
            exit_refl = _EXIT_ENGAGEMENT["reflective"]

            if cur == "conversational" and engagement >= exit_conv and idle_s < 60:
                suggested = "conversational"
            elif engagement >= enter_conv and idle_s < 30:
                suggested = "conversational"
            elif cur == "reflective" and engagement >= exit_refl:
                suggested = "reflective"
            elif engagement >= enter_refl:
                suggested = "reflective"
            else:
                suggested = "focused" if idle_s < 300 else "passive"

        return self._apply_boot_grace(suggested)

    def _apply_boot_grace(self, suggested: OperationalMode) -> OperationalMode:
        """During boot grace: block downgrades (-> sleep), allow upgrades."""
        now = time.time()
        with self._lock:
            grace_until = self._boot_grace_until
            cur_mode = self._mode

        if now >= grace_until:
            return suggested

        if not self._boot_grace_logged:
            from datetime import datetime, timezone
            until_iso = datetime.fromtimestamp(grace_until, tz=timezone.utc).isoformat()
            logger.info("Mode boot grace active (until=%s)", until_iso)
            self._boot_grace_logged = True

        cur_rank = _MODE_RANK.get(cur_mode, 1)
        sug_rank = _MODE_RANK.get(suggested, 1)
        if sug_rank < cur_rank:
            if now - self._last_grace_block_log >= 10.0:
                self._last_grace_block_log = now
                logger.info("Mode grace: blocked %s → %s (suggested=%s, until=%.0fs)",
                            cur_mode, "passive", suggested,
                            grace_until - now)
            return "passive" if cur_mode == "passive" else cur_mode
        return suggested

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            m, p, s, h_len = self._mode, self._profile, self._since, len(self._history)
        return {
            "mode": m,
            "profile": asdict(p),
            "since": s,
            "duration_s": time.time() - s,
            "history_len": h_len,
        }


mode_manager = ModeManager()
