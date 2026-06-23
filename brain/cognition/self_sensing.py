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
SECTOR_DECAY = 0.99        # per-sector decayed-sum window (~100 scored steps) for curiosity targeting


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
        self._lp_hist: deque[float] = deque(maxlen=LP_WINDOW)  # global skill snapshots at each refit
        # per-sector decayed squared-error sums -> per-sector skill + learning-progress (Step 2: curiosity)
        self._ss_model = np.zeros(N_SECTORS)
        self._ss_persist = np.zeros(N_SECTORS)
        self._sector_lp_hist: list[deque[float]] = [deque(maxlen=LP_WINDOW) for _ in range(N_SECTORS)]

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
            diff_m = actual - pred               # per-sector model error
            diff_p = actual - base               # per-sector persistence error
            move = _rms(diff_p) * math.sqrt(N_SECTORS)  # ||delta|| (m)
            err = _rms(diff_m)                   # model error (RMS over sectors)
            persist_err = _rms(diff_p)           # persistence baseline error
            is_dynamic = move > DYNAMIC_THRESH_M
            # per-sector decayed squared-error sums (for curiosity targeting)
            self._ss_model = SECTOR_DECAY * self._ss_model + diff_m * diff_m
            self._ss_persist = SECTOR_DECAY * self._ss_persist + diff_p * diff_p
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
            # per-sector skill snapshots -> per-sector learning-progress (curiosity)
            ssk, _slp, _ssa = self._per_sector()
            for j in range(N_SECTORS):
                if not math.isnan(ssk[j]):
                    self._sector_lp_hist[j].append(float(ssk[j]))
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

    def _per_sector(self):
        """Per-sector skill (MSE-ratio vs persistence), learning-progress, and recent activity."""
        ssp = self._ss_persist
        with np.errstate(divide="ignore", invalid="ignore"):
            skill = np.where(ssp > 1e-9, 1.0 - self._ss_model / np.maximum(ssp, 1e-12), np.nan)
        lp = np.array([(h[-1] - h[0]) if len(h) >= 2 else 0.0 for h in self._sector_lp_hist])
        return skill, lp, ssp

    def _curiosity_target(self) -> dict[str, Any] | None:
        """SHADOW suggestion: the sector where the model is learning the MOST right now.

        Authority = none — this only reports where attention COULD go, never acts. Returns
        None when nothing is meaningfully learnable (a static scene, or pure noise) — it does
        not fabricate a target. This is the Step-2 curiosity signal: learning-progress points
        at the region with novel-but-learnable structure (not too easy, not unpredictable).
        """
        skill, lp, ssp = self._per_sector()
        if float(ssp.max()) < 1e-9:
            return None
        active = ssp > 0.1 * float(ssp.max())   # only sectors with meaningful recent movement
        cand = [(j, float(lp[j])) for j in range(N_SECTORS) if active[j]]
        if not cand:
            return None
        j, best = max(cand, key=lambda x: x[1])
        if best <= 0.01:                        # nothing is meaningfully improving -> no target
            return None
        return {"sector": j, "deg": j * 30,
                "learning_progress": round(best, 4),
                "skill": round(float(skill[j]), 4) if not math.isnan(skill[j]) else None}

    # -- telemetry (shadow, read-only) --------------------------------------

    def _regime(self, skill_dyn: float | None, lp: float) -> tuple[str, str, int, bool]:
        """Read-only LABEL over the existing numbers (authority unchanged): distinguish a quiet
        world (STARVED — expected at a still desk, NOT a regression) from a genuine predictor
        failure (FAILED). This exists because the +0.463 headline was a first-30-min reading;
        live steady-state is volatile + event-bandwidth-gated. attention=True ONLY for FAILED."""
        recent_dyn = sum(1 for _m, _p, d in self._recent if d)
        floor = max(5, SKILL_WINDOW // 10)
        if self._W is None or self._n_scored < SKILL_WINDOW // 4:
            return "WARMING", "predictor not yet fitted / too few scored frames", recent_dyn, False
        if skill_dyn is None or recent_dyn < floor:
            return ("STARVED", "quiet world: only %d dynamic frames in the recent window — too few "
                    "to score skill (expected at a still desk, NOT a regression)" % recent_dyn,
                    recent_dyn, False)
        if skill_dyn < -0.05:
            return ("FAILED", "predictor below persistence on moving frames (skill_dyn %.3f, %d recent "
                    "dynamic frames) — genuinely worse than 'no change'" % (skill_dyn, recent_dyn),
                    recent_dyn, True)
        if skill_dyn > 0.05 and lp >= 0:
            return "EARNING", "beating persistence on moving frames + still learning", recent_dyn, False
        return ("STARVED", "marginal: skill_dyn %.3f near zero / LP %.3f — signal thin, not failing"
                % (skill_dyn, lp), recent_dyn, False)

    def get_status(self) -> dict[str, Any]:
        scored = max(1, self._n_scored)
        skill_all = self._skill(False)
        skill_dyn = self._skill(True)
        lp_val = self._learning_progress()
        regime, regime_reason, event_bw, attention = self._regime(skill_dyn, lp_val)
        ps_skill, ps_lp, ps_act = self._per_sector()
        ja = int(np.argmax(ps_act)) if float(ps_act.max()) > 1e-9 else None
        most_active = None
        if ja is not None:
            most_active = {"sector": ja, "deg": ja * 30,
                           "activity": round(float(ps_act[ja]), 4),
                           "skill": round(float(ps_skill[ja]), 4) if not math.isnan(ps_skill[ja]) else None}
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
            "learning_progress": round(lp_val, 5),
            # interpretive regime label (read-only; authority unchanged) so a quiet desk reads
            # as STARVED (expected) not FAILED — the honest steady-state the +0.463 headline missed.
            "health": {
                "regime": regime,
                "reason": regime_reason,
                "attention": attention,   # True ONLY for FAILED; STARVED/WARMING are not regressions
                "event_bandwidth_recent": event_bw,
                "note": "label over the numbers above; STARVED=quiet world (expected), "
                        "FAILED=predictor worse than persistence on moving frames",
            },
            "predictor": {
                "fitted": self._W is not None,
                "train_samples": len(self._buf),
                "refits": self._n_refits,
            },
            "per_sector": [
                {"sector": j, "deg": j * 30,
                 "skill": round(float(ps_skill[j]), 3) if not math.isnan(ps_skill[j]) else None,
                 "lp": round(float(ps_lp[j]), 4),
                 "activity": round(float(ps_act[j]), 4)}
                for j in range(N_SECTORS)
            ],
            "curiosity_target": self._curiosity_target(),
            "most_active_sector": most_active,
            "curiosity_note": ("shadow suggestion only — the sector where the model is learning "
                               "most right now; influences nothing (authority=none); None when "
                               "nothing is meaningfully learnable"),
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
            "ss_model": self._ss_model.tolist(),
            "ss_persist": self._ss_persist.tolist(),
            "sector_lp_hist": [list(h) for h in self._sector_lp_hist],
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
            sm = s.get("ss_model"); sp = s.get("ss_persist")
            if sm is not None:
                self._ss_model = np.array(sm, dtype=float)
            if sp is not None:
                self._ss_persist = np.array(sp, dtype=float)
            slh = s.get("sector_lp_hist")
            if slh:
                self._sector_lp_hist = [deque(h, maxlen=LP_WINDOW) for h in slh]
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
