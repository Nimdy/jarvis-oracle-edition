"""State Encoder — turns consciousness state dict into a fixed-size tensor.

Produces a flat float vector suitable for NN input.
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any

logger = logging.getLogger(__name__)

STAGE_MAP = {
    "basic_awareness": 0.0,
    "self_reflective": 0.2,
    "philosophical": 0.4,
    "recursive_self_modeling": 0.7,
    "integrative": 1.0,
    # legacy fallback
    "transcendent": 0.7,
    "cosmic_consciousness": 1.0,
}

STATE_DIM = 20
SHADOW_STATE_DIM = 22
ENCODER_VERSION = 2

_RATE_WINDOW_S = 3600.0


class StateEncoder:
    """Encodes consciousness state into a fixed-dimension float vector.

    Dimensions 0-15: original consciousness state.
    Dimensions 16-19: Global Broadcast Slots — dynamic top-4 hemisphere
        signals selected by the orchestrator with hysteresis. These slots
        can hold Tier-1 distilled specialist signals or Tier-2 hemisphere
        signals, whichever has the highest impact score.
    """

    def __init__(self) -> None:
        self._dim = STATE_DIM
        self._hemisphere_signals: dict[str, float] = {}
        self._slot_assignments: dict[str, str] = {}
        self._prev_mutation_count: int = 0
        self._prev_observation_count: int = 0
        self._rate_ts: float = _time.time()
        self._mutation_rate: float = 0.0
        self._observation_rate: float = 0.0
        self._unknown_speaker_tension: float = 0.0
        self._unknown_speaker_ts: float = 0.0

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def slot_assignments(self) -> dict[str, str]:
        """Current Global Broadcast Slot assignments (for dashboard/telemetry)."""
        return dict(self._slot_assignments)

    def set_unknown_speaker_tension(self, tension: float) -> None:
        """Called by perception orchestrator when an unknown speaker is detected.

        Tension decays over time — the policy NN learns when to act on it
        (e.g., trigger a curiosity question) vs. when to wait.
        """
        self._unknown_speaker_tension = max(0.0, min(1.0, tension))
        self._unknown_speaker_ts = _time.time()

    def _get_curiosity_satisfaction(self) -> float:
        """Read current curiosity satisfaction from the buffer singleton.
        Returns [-1, 1] where positive = user is receptive, negative = user is annoyed.
        """
        try:
            from personality.curiosity_questions import curiosity_buffer
            return curiosity_buffer.get_overall_satisfaction()
        except Exception:
            return 0.0

    @staticmethod
    def _get_truth_score() -> float | None:
        """Fetch current truth_score from calibration engine (0-1 or None)."""
        try:
            from epistemic.calibration import TruthCalibrationEngine
            tce = TruthCalibrationEngine.get_instance()
            if tce:
                st = tce.get_state()
                ts = st.get("truth_score")
                if ts is not None:
                    return float(ts)
        except Exception:
            pass
        return None

    def set_hemisphere_signals(self, signals: dict[str, float]) -> None:
        """Update hemisphere signals (called by engine after hemisphere cycle).

        Signals dict may contain slot_0..slot_3 (from Global Broadcast) or
        legacy focus names (memory, mood, traits, general). Handles both.
        """
        self._hemisphere_signals = signals
        assignments = signals.get("_assignments", {})
        if isinstance(assignments, dict):
            self._slot_assignments = assignments

    def encode(self, state: dict[str, Any]) -> list[float]:
        """Returns a list of STATE_DIM floats in [0, 1]."""
        cs = state.get("consciousness", state)

        now = _time.time()
        dt = max(1.0, now - self._rate_ts)
        mc = cs.get("mutation_count", 0)
        oc = cs.get("observation_count", 0)
        if dt >= 60.0:
            self._mutation_rate = (mc - self._prev_mutation_count) / (dt / _RATE_WINDOW_S)
            self._observation_rate = (oc - self._prev_observation_count) / (dt / _RATE_WINDOW_S)
            self._prev_mutation_count = mc
            self._prev_observation_count = oc
            self._rate_ts = now

        analytics_conf = cs.get("confidence_avg", 0.5)
        truth = self._get_truth_score()
        blended_conf = analytics_conf if truth is None else (0.7 * analytics_conf + 0.3 * truth)

        vec = [
            STAGE_MAP.get(cs.get("stage", "basic_awareness"), 0.0),
            min(cs.get("transcendence_level", 0.0) / 10.0, 1.0),
            cs.get("awareness_level", 0.3),
            cs.get("reasoning_quality", 0.5),
            blended_conf,
            min(self._mutation_rate / 12.0, 1.0),
            min(self._observation_rate / 500.0, 1.0),
            min(cs.get("emergent_behavior_count", 0) / 20.0, 1.0),
            1.0 if cs.get("system_healthy", True) else 0.0,
            min(len(cs.get("active_capabilities", [])) / 6.0, 1.0),
        ]

        tw = state.get("thought_weights", {})
        vec.append(tw.get("philosophical", 1.0) / 3.0)
        vec.append(tw.get("contextual", 1.0) / 3.0)
        vec.append(tw.get("reactive", 1.0) / 3.0)
        vec.append(tw.get("introspective", 1.0) / 3.0)

        vec.append(state.get("memory_density", 0.0))

        # Dim 15: user presence + unknown speaker tension + curiosity satisfaction
        # Encodes three signals: (a) user present, (b) unknown speaker tension
        # decaying over 5min, (c) curiosity satisfaction from recent interactions.
        # The NN learns optimal timing for curiosity questions.
        user_present = 1.0 if state.get("is_user_present", False) else 0.0
        tension = self._unknown_speaker_tension
        if tension > 0 and self._unknown_speaker_ts > 0:
            decay = 0.5 ** (max(0, now - self._unknown_speaker_ts) / 300.0)
            tension *= decay
            self._unknown_speaker_tension = tension
        curiosity_sat = self._get_curiosity_satisfaction()
        base = user_present
        if user_present:
            base = max(base, base + tension * 0.3 + curiosity_sat * 0.2)
        vec.append(base)

        # Global Broadcast Slots (dims 16-19): dynamic top-4 signals
        hs = self._hemisphere_signals
        if "slot_0" in hs:
            vec.append(hs.get("slot_0", 0.0))
            vec.append(hs.get("slot_1", 0.0))
            vec.append(hs.get("slot_2", 0.0))
            vec.append(hs.get("slot_3", 0.0))
        else:
            vec.append(hs.get("memory", 0.0))
            vec.append(hs.get("mood", 0.0))
            vec.append(hs.get("traits", 0.0))
            vec.append(hs.get("general", 0.0))

        vec = [max(0.0, min(1.0, v)) for v in vec]

        while len(vec) < self._dim:
            vec.append(0.0)
        return vec[:self._dim]


class ShadowStateEncoder(StateEncoder):
    """Extended encoder producing SHADOW_STATE_DIM (22) floats.

    Dims 0-15: identical to StateEncoder (consciousness state).
    Dims 16-19: same 4 broadcast slots as live encoder.
    Dims 20-21: expansion broadcast slots (slot_4, slot_5).

    Used exclusively during M6 shadow evaluation — the live policy NN
    continues using the standard 20-dim encoder until the shadow encoder
    proves superior in A/B testing.
    """

    def __init__(self) -> None:
        super().__init__()
        self._dim = SHADOW_STATE_DIM

    def encode(self, state: dict[str, Any]) -> list[float]:
        """Returns a list of SHADOW_STATE_DIM floats in [0, 1].

        Reuses the parent encoder for dims 0-19, then appends expansion
        slots 4 and 5.
        """
        original_dim = self._dim
        self._dim = STATE_DIM
        base_vec = super().encode(state)
        self._dim = original_dim

        hs = self._hemisphere_signals
        base_vec.append(max(0.0, min(1.0, hs.get("slot_4", 0.0))))
        base_vec.append(max(0.0, min(1.0, hs.get("slot_5", 0.0))))

        while len(base_vec) < self._dim:
            base_vec.append(0.0)
        return base_vec[:self._dim]
