"""Self-Sensing Loop — shadow learning-progress over the lidar world.

The seed of the autonomous-growth engine. EVIDENCE (2026-06-21, see
docs/AUTONOMOUS_GROWTH_STRATEGY.md): the live lidar 12-sector range vector is
predictable BEYOND persistence — a learned model beat the "nothing changed"
baseline by +26% overall and +45.6% on dynamic transitions, shuffle-confirmed.
Unlike the policy NN (signal ABSENT, Spearman ~0.06), the SENSES carry a real,
NON-OPERATOR learning signal. This module harvests it, honestly:

    predict the next sensor frame -> check it against ground truth (the real next
    frame) -> reward the error REDUCTION (learning progress), where the signal lives
    (the dynamic moments).

INTEGRITY (the whole point — built in, not bolted on):
- SHADOW / authority = none. It observes, predicts, scores, reports. It does NOT
  influence behavior, write beliefs, or write memory. No gate is flipped here.
- Negative control is structural: skill is the MSE-ratio vs the PERSISTENCE baseline
  ("predict no change") over a rolling window. If the predictor is not beating
  persistence, skill<=0 and there is NO real signal — reported honestly, never gamed.
- Ground truth is the REAL next sensor reading (external), never a self-score.
- Learning-progress is the *reduction* of prediction error over time, not a rising dial.

Pure numpy (closed-form ridge); no sklearn on the hot path. ~2s tick cadence.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time  # noqa: F401  (engine passes timestamps; kept for callers)
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
STATE_FILE = JARVIS_DIR / "self_sensing_state.json"

N_SECTORS = 12
DYNAMIC_THRESH_M = 0.02      # a transition is "dynamic" if ||delta|| > 2cm (above lidar noise)
REFIT_EVERY = 20             # refit the predictor every N observations
MIN_FIT_SAMPLES = 30         # need this many (features,delta) pairs before fitting
BUF_MAX = 500               # rolling training buffer
SKILL_WINDOW = 250          # rolling window for the MSE-ratio skill metric
RIDGE_LAMBDA = 1.0
EMA_ALPHA = 0.05            # slow EMA for stable error-magnitude reporting + LP
MAX_GAP_S = 10.0           # sensor gap longer than this resets velocity continuity
LP_WINDOW = 8              # learning-progress over the last N refit snapshots


def _rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(v * v))) if v.size else 0.0


class SelfSensingLoop:
    """Shadow predictor of the next lidar frame + learning-progress signal."""

    def __init__(self) -> None:
        self._prev_vec: np.ndarray | None = None     # X[t-1] for velocity + persistence
        self._last_t: float = 0.0
        self._pending: dict[str, Any] | None = None  # {pred, feats, base, t}
        self._buf: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=BUF_MAX)
        self._W: np.ndarray | None = None            # (2*N_SECTORS, N_SECTORS) ridge weights

        # running, persisted stats
        self._n_obs = 0
        self._n_scored = 0
        self._n_dynamic = 0
        self._n_refits = 0
        self._n_gaps = 0
        self._err_ema: float | None = None           # model prediction error (RMS, m) — for LP + reporting
        self._persist_ema: float | None = None       # persistence baseline error (RMS, m)
        # rolling window of (model_sq_err, persist_sq_err, is_dynamic) -> honest MSE-ratio skill
        self._recent: deque[tuple[float, float, bool]] = deque(maxlen=SKILL_WINDOW)
        self._lp_hist: deque[float] = deque(maxlen=LP_WINDOW)  # err_ema snapshots at each refit

    # -- core tick -----------------------------------------------------------

    def observe(self, sectors: list[float] | None, t: float) -> None:
        """Feed the current 12-sector range vector (meters). Called ~every 2s."""
        vec = self._clean(sectors)
        if vec is None:
            self._n_gaps += 1
            self._reset_continuity()   # don't train across a missing/garbage frame
            return

        # gap detection: a long pause makes velocity meaningless -> reset continuity
        if self._last_t and (t - self._last_t) > MAX_GAP_S:
            self._n_gaps += 1
            self._reset_continuity()
        self._last_t = t
        self._n_obs += 1

        # 1) SCORE the pending prediction against the now-observed actual (ground truth)
        if self._pending is not None:
            base = self._pending["base"]
            pred = self._pending["pred"]
            actual = vec
            move = _rms(actual - base) * math.sqrt(N_SECTORS)  # ||delta|| (m)
            err = _rms(actual - pred)            # model error (RMS over sectors)
            persist_err = _rms(actual - base)    # persistence baseline error
            is_dynamic = move > DYNAMIC_THRESH_M
            self._err_ema = self._ema(self._err_ema, err)
            self._persist_ema = self._ema(self._persist_ema, persist_err)
            self._recent.append((err * err, persist_err * persist_err, is_dynamic))
            self._n_scored += 1
            if is_dynamic:
                self._n_dynamic += 1
            # 2) TRAIN: the features we predicted from -> the actual delta that occurred
            self._buf.append((self._pending["feats"], actual - base))
            if self._n_obs % REFIT_EVERY == 0 and len(self._buf) >= MIN_FIT_SAMPLES:
                self._fit()

        # 3) PREDICT the next frame from [position, velocity]
        feats = self._features(vec, self._prev_vec)
        delta_hat = (feats @ self._W) if self._W is not None else np.zeros(N_SECTORS)
        self._pending = {"pred": vec + delta_hat, "feats": feats, "base": vec, "t": t}
        self._prev_vec = vec

    # -- predictor -----------------------------------------------------------

    @staticmethod
    def _features(cur: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
        """[position(12), velocity(12)] = 24-dim feature vector."""
        vel = (cur - prev) if prev is not None else np.zeros(N_SECTORS)
        return np.concatenate([cur, vel])

    def _fit(self) -> None:
        """Closed-form multi-output ridge: W = (F'F + lambda I)^-1 F'D."""
        try:
            F = np.array([f for f, _ in self._buf])      # (n, 24)
            D = np.array([d for _, d in self._buf])      # (n, 12)
            k = F.shape[1]
            A = F.T @ F + RIDGE_LAMBDA * np.eye(k)
            self._W = np.linalg.solve(A, F.T @ D)        # (24, 12)
            self._n_refits += 1
            sk = self._skill(True)  # snapshot dynamic-skill: learning-progress = skill rising over time
            if sk is not None:
                self._lp_hist.append(sk)
        except Exception:
            logger.debug("self-sensing fit failed", exc_info=True)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _clean(sectors: list[float] | None) -> np.ndarray | None:
        if not sectors or len(sectors) != N_SECTORS:
            return None
        try:
            v = np.array([float(x) for x in sectors], dtype=float)
        except (TypeError, ValueError):
            return None
        if not np.all(np.isfinite(v)) or np.any(v < 0):
            return None
        return v

    def _reset_continuity(self) -> None:
        self._prev_vec = None
        self._pending = None

    @staticmethod
    def _ema(cur: float | None, x: float) -> float:
        return x if cur is None else (EMA_ALPHA * x + (1.0 - EMA_ALPHA) * cur)

    def _skill(self, dynamic_only: bool) -> float | None:
        """MSE-ratio skill vs persistence over the rolling window. >0 = beats 'no change'."""
        rows = [(m, p) for m, p, d in self._recent if (d or not dynamic_only)]
        if len(rows) < 5:
            return None
        sm = sum(m for m, _ in rows)
        sp = sum(p for _, p in rows)
        return (1.0 - sm / sp) if sp > 1e-12 else None

    def _learning_progress(self) -> float:
        """Rise in prediction skill over the recent refit window (positive = the model is learning)."""
        if len(self._lp_hist) < 2:
            return 0.0
        return float(self._lp_hist[-1] - self._lp_hist[0])

    # -- telemetry (shadow, read-only) --------------------------------------

    def get_status(self) -> dict[str, Any]:
        scored = max(1, self._n_scored)
        skill_all = self._skill(False)
        skill_dyn = self._skill(True)
        return {
            "phase": "P0_shadow_observe",
            "authority": {
                "influences_behavior": False,
                "writes_beliefs": False,
                "writes_memory": False,
            },
            "observations": self._n_obs,
            "scored": self._n_scored,
            "gaps": self._n_gaps,
            "dynamic_fraction": round(self._n_dynamic / scored, 3),
            "skill_vs_persistence": round(skill_all, 4) if skill_all is not None else None,
            "skill_vs_persistence_dynamic": round(skill_dyn, 4) if skill_dyn is not None else None,
            "prediction_error_ema_m": round(self._err_ema, 4) if self._err_ema is not None else None,
            "persistence_error_ema_m": round(self._persist_ema, 4) if self._persist_ema is not None else None,
            "learning_progress": round(self._learning_progress(), 5),
            "predictor": {
                "fitted": self._W is not None,
                "train_samples": len(self._buf),
                "refits": self._n_refits,
            },
            # honest negative control: are we ACTUALLY beating "nothing changed"
            # on the moments that matter? <=~0 means no real signal yet.
            "beating_persistence": bool(skill_dyn is not None and skill_dyn > 0.05),
            "note": ("shadow / observe-only; ground truth = the real next lidar frame; "
                     "skill is the MSE-ratio vs the persistence baseline (built-in negative "
                     "control); no behavior authority, nothing written"),
        }

    # -- persistence (survive reboot) ---------------------------------------

    def to_state(self) -> dict[str, Any]:
        return {
            "n_obs": self._n_obs, "n_scored": self._n_scored, "n_dynamic": self._n_dynamic,
            "n_refits": self._n_refits, "n_gaps": self._n_gaps,
            "err_ema": self._err_ema, "persist_ema": self._persist_ema,
            "recent": [[m, p, bool(d)] for m, p, d in self._recent],
            "lp_hist": list(self._lp_hist),
            "W": self._W.tolist() if self._W is not None else None,
        }

    def load_state(self, s: dict[str, Any]) -> None:
        try:
            self._n_obs = int(s.get("n_obs", 0)); self._n_scored = int(s.get("n_scored", 0))
            self._n_dynamic = int(s.get("n_dynamic", 0)); self._n_refits = int(s.get("n_refits", 0))
            self._n_gaps = int(s.get("n_gaps", 0))
            self._err_ema = s.get("err_ema"); self._persist_ema = s.get("persist_ema")
            self._recent = deque(
                [(float(m), float(p), bool(d)) for m, p, d in s.get("recent", [])],
                maxlen=SKILL_WINDOW,
            )
            self._lp_hist = deque(s.get("lp_hist", []), maxlen=LP_WINDOW)
            W = s.get("W")
            if W is not None:
                self._W = np.array(W, dtype=float)
        except Exception:
            logger.warning("self-sensing load_state failed", exc_info=True)

    def save(self) -> None:
        try:
            JARVIS_DIR.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(JARVIS_DIR), suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_state(), f)
            os.replace(tmp, STATE_FILE)
        except Exception:
            logger.debug("self-sensing save failed", exc_info=True)

    def restore(self) -> None:
        try:
            if STATE_FILE.exists():
                self.load_state(json.loads(STATE_FILE.read_text()))
                logger.info("Self-sensing restored: %d obs, skill_dyn=%s",
                            self._n_obs, self._skill(True))
        except Exception:
            logger.debug("self-sensing restore failed", exc_info=True)
