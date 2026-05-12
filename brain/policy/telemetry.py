"""Policy Telemetry — single source of truth for NN observability.

Hot-path contract:
  - Only update counters, EMAs, last-known values, ring-buffer appends
  - Never allocate except rare event log_event() calls
  - Never scan buffers, compute aggregates, or touch model weights

Cold-path contract (snapshot):
  - Returns a pre-built dict of primitives + list(events)
  - O(1) to build — just copy fields
  - Never triggers inference, training, or evaluation
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

EMA_ALPHA = 0.12
EMA_ALPHA_SLOW = 0.05
MAX_EVENTS = 80
RATE_WINDOW_S = 5.0


def _ema(prev: float, x: float, alpha: float = EMA_ALPHA) -> float:
    return x if prev == 0.0 else (alpha * x + (1.0 - alpha) * prev)


def _ema_upper(prev: float, x: float, alpha: float = EMA_ALPHA) -> float:
    """EMA that tracks the upper envelope — cheap p95 proxy."""
    peak = max(x, prev)
    return alpha * peak + (1.0 - alpha) * prev if prev > 0.0 else x


@dataclass
class PolicyTelemetry:
    # ── Status ──────────────────────────────────────────────────────────
    active: bool = False
    mode: str = "shadow"
    arch: str = "none"
    model_id: str = "none"
    model_version: int = 0
    device: str = "cpu"
    train_enabled: bool = False
    feature_flags: dict[str, bool] = field(default_factory=dict)

    # ── Performance (EMA-updated on hot path) ───────────────────────────
    decisions_total: int = 0
    decisions_per_s_ema: float = 0.0
    decision_ms_p50_ema: float = 0.0
    decision_ms_p95_ema: float = 0.0
    encode_ms_ema: float = 0.0
    train_step_ms_ema: float = 0.0
    last_step_ts: float = 0.0

    # ── Safety / Governor ───────────────────────────────────────────────
    blocks_total: int = 0
    passes_total: int = 0
    overruns_total: int = 0
    last_block_reason: str = ""
    last_block_ts: float = 0.0
    auto_disabled: bool = False
    regressions: int = 0

    # ── Learning / Eval ─────────────────────────────────────────────────
    shadow_ab_total: int = 0
    shadow_nn_wins: int = 0
    shadow_kernel_wins: int = 0
    shadow_ties: int = 0
    last_eval_score: float = 0.0
    eligible_for_control: bool = False
    promotion_candidates: int = 0
    last_promotion_ts: float = 0.0
    win_margin_ema: float = 0.0
    noop_count: int = 0
    nn_decisive_win_rate: float = 0.0

    # ── Training ────────────────────────────────────────────────────────
    train_runs_total: int = 0
    last_train_loss: float = 0.0
    last_train_epochs: int = 0
    last_train_ts: float = 0.0
    last_train_duration_s: float = 0.0

    # ── Training history (for dashboard charts) ──────────────────────
    training_loss_history: deque = field(default_factory=lambda: deque(maxlen=50))
    reward_history: deque = field(default_factory=lambda: deque(maxlen=200))
    win_rate_history: deque = field(default_factory=lambda: deque(maxlen=100))

    # ── Registry ────────────────────────────────────────────────────────
    registry_total_versions: int = 0
    registry_active_version: int = 0
    registry_active_arch: str = "none"

    # ── Rate tracking (internal, not exported) ──────────────────────────
    _rate_count: int = 0
    _rate_window_start: float = 0.0
    _windowed_nn_win_rate: float = 0.0

    # ── Recent activity ring buffer ─────────────────────────────────────
    events: deque = field(default_factory=lambda: deque(maxlen=MAX_EVENTS))

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    # ═══════════════════════════════════════════════════════════════════
    # Hot-path update methods — O(1), no alloc (except log_event)
    # ═══════════════════════════════════════════════════════════════════

    def record_decision(self, latency_ms: float, encode_ms: float) -> None:
        with self._lock:
            self.decisions_total += 1
            self.decision_ms_p50_ema = _ema(self.decision_ms_p50_ema, latency_ms)
            self.decision_ms_p95_ema = _ema_upper(self.decision_ms_p95_ema, latency_ms)
            self.encode_ms_ema = _ema(self.encode_ms_ema, encode_ms)
            self.last_step_ts = time.time()
            self._update_rate()

    def record_block(self, reason: str) -> None:
        with self._lock:
            self.blocks_total += 1
            self.last_block_reason = reason
            self.last_block_ts = time.time()

    def record_pass(self) -> None:
        with self._lock:
            self.passes_total += 1

    def record_overrun(self) -> None:
        with self._lock:
            self.overruns_total += 1

    def record_shadow(self, nn_won: bool) -> None:
        with self._lock:
            self.shadow_ab_total += 1
            if nn_won:
                self.shadow_nn_wins += 1
            else:
                self.shadow_kernel_wins += 1

    def record_shadow_tie(self) -> None:
        with self._lock:
            self.shadow_ab_total += 1
            self.shadow_ties += 1

    def record_win_margin(self, margin: float) -> None:
        self.win_margin_ema = _ema(self.win_margin_ema, margin, EMA_ALPHA_SLOW)

    def record_train_step(self, step_ms: float) -> None:
        self.train_step_ms_ema = _ema(self.train_step_ms_ema, step_ms, EMA_ALPHA_SLOW)

    def record_train_complete(
        self, epochs: int, loss: float, duration_s: float,
        epoch_losses: list[float] | None = None,
    ) -> None:
        self.train_runs_total += 1
        self.last_train_loss = loss
        self.last_train_epochs = epochs
        self.last_train_ts = time.time()
        self.last_train_duration_s = duration_s
        self.training_loss_history.append({
            "timestamp": time.time(),
            "final_loss": round(loss, 6),
            "epoch_losses": [round(l, 6) for l in (epoch_losses or [loss])],
            "epochs": epochs,
        })
        self.log_event("train_complete",
                       f"epochs={epochs} loss={loss:.4f} time={duration_s:.1f}s")

    def record_reward(self, reward: float) -> None:
        """Record a reward value from the experience buffer."""
        self.reward_history.append({"timestamp": time.time(), "value": round(reward, 4)})

    def record_win_rate_snapshot(self, win_rate_override: float | None = None) -> None:
        """Snapshot the current rolling win rate for the dashboard chart."""
        with self._lock:
            if win_rate_override is not None:
                rate = win_rate_override
            else:
                if self.shadow_ab_total == 0:
                    return
                rate = self.shadow_nn_wins / self.shadow_ab_total
            self.win_rate_history.append({"timestamp": time.time(), "value": round(rate, 4)})

    def record_promotion(self, version: int, arch: str) -> None:
        self.model_version = version
        self.arch = arch
        self.registry_active_version = version
        self.registry_active_arch = arch
        self.last_promotion_ts = time.time()
        self.log_event("promotion", f"v{version} ({arch}) promoted to active")

    def update_registry(self, total: int, active_version: int, active_arch: str) -> None:
        self.registry_total_versions = total
        self.registry_active_version = active_version
        self.registry_active_arch = active_arch

    def update_eval(self, score: float, eligible: bool, candidates: int) -> None:
        self.last_eval_score = score
        self.eligible_for_control = eligible
        self.promotion_candidates = candidates

    # ── Event log (only on state transitions, not every tick) ───────────

    def log_event(self, etype: str, msg: str, meta: dict[str, Any] | None = None) -> None:
        self.events.append({
            "ts": time.time(),
            "type": etype,
            "msg": msg,
            "meta": meta or {},
        })

    # ═══════════════════════════════════════════════════════════════════
    # Cold-path snapshot — O(1), returns a dict of primitives
    # ═══════════════════════════════════════════════════════════════════

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            ab_total = self.shadow_ab_total
            nn_wins = self.shadow_nn_wins
            return {
                "active": self.active,
                "mode": self.mode,
                "arch": self.arch,
                "model_id": self.model_id,
                "model_version": self.model_version,
                "device": self.device,
                "train_enabled": self.train_enabled,
                "feature_flags": dict(self.feature_flags),

                "decisions_total": self.decisions_total,
                "decisions_per_s_ema": round(self.decisions_per_s_ema, 2),
                "decision_ms_p50_ema": round(self.decision_ms_p50_ema, 2),
                "decision_ms_p95_ema": round(self.decision_ms_p95_ema, 2),
                "encode_ms_ema": round(self.encode_ms_ema, 2),
                "train_step_ms_ema": round(self.train_step_ms_ema, 2),
                "last_step_ts": self.last_step_ts,

                "blocks_total": self.blocks_total,
                "passes_total": self.passes_total,
                "overruns_total": self.overruns_total,
                "last_block_reason": self.last_block_reason,
                "last_block_ts": self.last_block_ts,
                "auto_disabled": self.auto_disabled,
                "regressions": self.regressions,

                "shadow_ab_total": ab_total,
                "shadow_nn_wins": nn_wins,
                "shadow_kernel_wins": self.shadow_kernel_wins,
                "shadow_ties": self.shadow_ties,
                "nn_win_rate": round(self._windowed_nn_win_rate, 3) if self._windowed_nn_win_rate > 0 else (round(nn_wins / ab_total, 3) if ab_total > 0 else 0.0),
                "nn_decisive_win_rate": round(self.nn_decisive_win_rate, 3),
                "last_eval_score": round(self.last_eval_score, 3),
                "eligible_for_control": self.eligible_for_control,
                "promotion_candidates": self.promotion_candidates,
                "last_promotion_ts": self.last_promotion_ts,
                "win_margin_ema": round(self.win_margin_ema, 4),
                "noop_count": self.noop_count,

                "train_runs_total": self.train_runs_total,
                "last_train_loss": round(self.last_train_loss, 4),
                "last_train_epochs": self.last_train_epochs,
                "last_train_ts": self.last_train_ts,
                "last_train_duration_s": round(self.last_train_duration_s, 1),

                "registry_total_versions": self.registry_total_versions,
                "registry_active_version": self.registry_active_version,
                "registry_active_arch": self.registry_active_arch,

                "recent_events": list(self.events),

                "training_loss_history": list(self.training_loss_history),
                "reward_history": list(self.reward_history),
                "win_rate_history": list(self.win_rate_history),
            }

    # ── internal rate tracking ──────────────────────────────────────────

    def _update_rate(self) -> None:
        now = time.time()
        self._rate_count += 1
        elapsed = now - self._rate_window_start
        if elapsed >= RATE_WINDOW_S:
            rate = self._rate_count / elapsed if elapsed > 0 else 0.0
            self.decisions_per_s_ema = _ema(self.decisions_per_s_ema, rate, EMA_ALPHA_SLOW)
            self._rate_count = 0
            self._rate_window_start = now


# Module-level singleton
policy_telemetry = PolicyTelemetry()
