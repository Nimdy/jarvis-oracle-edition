"""Goal Continuity Layer — constants, thresholds, rate limits, and scoring weights."""

from __future__ import annotations

# ── Capacity caps ──
MAX_ACTIVE_GOALS: int = 5
MAX_CANDIDATES: int = 20
MAX_COMPLETED_RETAINED: int = 50
MAX_TASKS_PER_GOAL: int = 8
MAX_NEW_GOALS_PER_HOUR: int = 10
MAX_PROMOTIONS_PER_HOUR: int = 3

# ── Promotion thresholds ──
PROMOTION_SCORE_THRESHOLD: float = 0.6
PROMOTION_RECURRENCE_MIN: int = 2
PROMOTION_RECURRENCE_WINDOW_S: float = 600.0
PROMOTION_DEFICIT_CYCLES_MIN: int = 2

# ── Promotion scoring weights (documented formula) ──
#
# score = 0.0
# if explicit_user_requested: score += SCORE_USER_REQUEST
# score += (recurrence_count - 1) * SCORE_DRIVE_RECURRENCE   (drive signals)
# score += (recurrence_count - 1) * SCORE_THOUGHT_CLUSTER     (thought/existential/emergence)
# score += sustained_deficit_cycles * SCORE_METRIC_DEFICIT_CYCLE
# score += merge_count * SCORE_DEDUP_MERGE
# score += stale_windows * SCORE_STALE_DECAY
# clamped to [0.0, 2.0]
SCORE_USER_REQUEST: float = 1.0
SCORE_DRIVE_RECURRENCE: float = 0.25
SCORE_THOUGHT_CLUSTER: float = 0.20
SCORE_METRIC_DEFICIT_CYCLE: float = 0.30
SCORE_DEDUP_MERGE: float = 0.15
SCORE_STALE_DECAY: float = -0.10
STALE_DECAY_INTERVAL_S: float = 3600.0

# ── Lifecycle windows ──
CANDIDATE_EXPIRY_S: float = 86400.0
BLOCKED_REVIEW_THRESHOLD_S: float = 7200.0
ABANDON_ACTIVE_S: float = 28800.0
STALE_WINDOW_S: float = 1800.0
COOLDOWN_AFTER_ABANDON_S: float = 3600.0
COOLDOWN_AFTER_BLOCK_S: float = 1800.0

# ── Active metric goal refresh ──
REFRESH_MATERIALITY_BAND: float = 0.05  # ignore live changes smaller than this
REFRESH_RESOLUTION_THRESHOLD: float = 0.65  # live score above this = "resolved"
REFRESH_PAUSE_THRESHOLD: float = 0.50  # live score above this = "no longer critical"
REFRESH_PAUSE_CYCLES_REQUIRED: int = 3  # consecutive non-critical cycles before auto-pause
REFRESH_COMPLETE_CYCLES_REQUIRED: int = 3  # consecutive resolved cycles before completion

# ── Dedup ──
DEDUP_JACCARD_THRESHOLD: float = 0.6

# ── Tick cadence ──
GOALS_INTERVAL_S: float = 120.0

# ── Dispatch (Phase 2) ──
DISPATCH_COOLDOWN_S: float = 120.0
STALLED_PROGRESS_THRESHOLD: float = 0.1
