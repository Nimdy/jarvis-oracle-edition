"""#9.3-A — goal grounding: metric-deficit churn dampening.

Proves the sustained-deficit gate stops single-sample metric goals (the navel-gazing
churn: create -> can't act -> auto-abandon). A deficit must persist across >= N
consecutive ticks before it becomes a goal signal; recovery resets the streak; genuinely
sustained deficits still create. No behavior authority involved.
"""
from __future__ import annotations

import goals.signal_producers as sp
from goals.signal_producers import detect_metric_deficits


def _reset(warm: bool = True):
    sp._deficit_streaks.clear()
    # pre-warm so the warmup gate (uptime>180 + 2 ticks) isn't what's being tested
    sp._warmup_state["ticks_seen"] = 5 if warm else 0
    sp._warmup_state["first_tick_time"] = 1.0 if warm else 0.0
    for k in sp._producer_stats:
        sp._producer_stats[k] = 0


# memory_health threshold is 0.40 in _DEFICIT_CONFIGS
_BAD = {"components": {"memory_health": 0.30}}
_GOOD = {"components": {"memory_health": 0.95}}


def test_single_sample_is_suppressed():
    _reset()
    s1 = detect_metric_deficits(_BAD, None, None, uptime_s=300)
    assert s1 == []  # one weak sample must NOT manufacture a goal
    assert sp._producer_stats["metric_sustained_skipped"] == 1
    assert sp._producer_stats["metric_created"] == 0


def test_sustained_health_deficit_is_unactionable():
    # Health-monitor components have no goal-layer executor (#26).
    _reset()
    detect_metric_deficits(_BAD, None, None, uptime_s=300)        # streak 1
    s2 = detect_metric_deficits(_BAD, None, None, uptime_s=300)   # streak 2 — still no mint
    assert s2 == []
    assert sp._producer_stats["metric_created"] == 0
    assert sp._producer_stats["metric_unactionable_skipped"] >= 1


def test_recovery_resets_streak():
    _reset()
    detect_metric_deficits(_BAD, None, None, uptime_s=300)   # streak 1
    detect_metric_deficits(_GOOD, None, None, uptime_s=300)  # recover -> reset
    s = detect_metric_deficits(_BAD, None, None, uptime_s=300)  # streak 1 again
    assert s == []  # flapping at the threshold never qualifies


def test_healthy_metrics_never_create():
    _reset()
    for _ in range(5):
        assert detect_metric_deficits(_GOOD, None, None, uptime_s=300) == []
    assert sp._producer_stats["metric_created"] == 0
    assert sp._producer_stats["metric_sustained_skipped"] == 0


def test_warmup_gate_still_applies():
    _reset(warm=False)
    assert detect_metric_deficits(_BAD, None, None, uptime_s=10) == []
    assert sp._producer_stats["metric_warmup_skipped"] >= 1
    assert sp._producer_stats["metric_created"] == 0


def test_calibration_deficit_also_gated():
    _reset()
    cal = {"domain_scores": {"weather": 0.10}, "domain_provisional": {"weather": False}}
    assert detect_metric_deficits(None, cal, None, uptime_s=300) == []   # streak 1
    s2 = detect_metric_deficits(None, cal, None, uptime_s=300)           # streak 2
    assert len(s2) == 1 and s2[0].source == "truth_calibration"


def test_active_deficits_are_unactionable():
    # metric_triggers have no goal-layer repair path. Lived leftover: reasoning_coherence
    # merged hundreds of times with 0 task successes. Do not mint.
    _reset()
    ad = {"tick_overrun": {"duration_s": 600, "severity": "high"}}
    s = detect_metric_deficits(None, None, ad, uptime_s=300)
    assert s == []
    assert sp._producer_stats["metric_unactionable_skipped"] >= 1
    assert sp._producer_stats["metric_created"] == 0


# ── #9.3-A.2: per-kind active cap ──

import tempfile  # noqa: E402
import time as _time  # noqa: E402
from pathlib import Path  # noqa: E402

from goals.goal import Goal, GoalSignal  # noqa: E402
from goals.goal_registry import GoalRegistry  # noqa: E402
from goals.goal_manager import GoalManager  # noqa: E402


def _mgr() -> GoalManager:
    p = Path(tempfile.mkdtemp()) / "goals.json"
    return GoalManager(registry=GoalRegistry(path=p))


def _add_active_metric(reg: GoalRegistry, i: int, now: float) -> None:
    reg.add(Goal(
        title=f"metric goal {i}", kind="system_health", status="active",
        source_scope="metric", promotion_score=0.9, sustained_deficit_cycles=2,
        tag_cluster=(f"m{i}", "health"), created_at=now, updated_at=now,
        last_observed_at=now,
    ))


class TestPerKindCap:
    def test_metric_cap_holds_back_excess_metric_goals(self):
        m = _mgr(); reg = m._registry; now = _time.time()
        _add_active_metric(reg, 0, now)
        _add_active_metric(reg, 1, now)  # 2 active metric goals == cap
        cand = Goal(
            title="metric candidate extra", kind="system_health", status="candidate",
            source_scope="metric", promotion_score=0.9, sustained_deficit_cycles=2,
            tag_cluster=("extra", "health"), created_at=now, updated_at=now,
        )
        reg.add(cand)
        m._evaluate_promotions(now)
        assert reg.get(cand.goal_id).status == "candidate"  # held back by per-kind cap

    def test_user_goal_bypasses_metric_cap(self):
        m = _mgr(); reg = m._registry; now = _time.time()
        _add_active_metric(reg, 0, now)
        _add_active_metric(reg, 1, now)  # metric at cap
        user = Goal(
            title="improve memory recall", kind="user_goal", status="candidate",
            source_scope="user", explicit_user_requested=True, promotion_score=1.0,
            tag_cluster=("memory", "recall"), created_at=now, updated_at=now,
        )
        reg.add(user)
        m._evaluate_promotions(now)
        assert reg.get(user.goal_id).status == "active"  # user goals are never metric-capped


# ── #9.3-A.3: source-lifecycle telemetry ──

class TestSourceLifecycle:
    def test_created_and_abandoned_tracked_by_source(self):
        m = _mgr()
        sig = GoalSignal(
            signal_type="user_request", source="conversation", source_scope="user",
            content="please improve memory recall", tag_cluster=("memory",),
        )
        m.observe_signal(sig)
        created = [g for g in m._registry.get_all() if g.source_scope == "user"]
        assert created, "user goal should have been created"

        lc = m.get_source_lifecycle()
        assert lc.get("user", {}).get("created", 0) >= 1

        m.abandon_goal(created[0].goal_id, "test abandon")
        lc2 = m.get_source_lifecycle()
        assert lc2["user"]["abandoned"] >= 1
        assert lc2["user"]["abandon_rate"] > 0.0

    def test_lifecycle_exposed_in_status(self):
        m = _mgr()
        status = m.get_status()
        assert "source_lifecycle" in status

    def test_lifecycle_rebuilds_from_persisted_registry(self):
        m = _mgr()
        now = _time.time()
        m._registry.add(Goal(
            title="old metric", kind="system_health", status="abandoned",
            source_scope="system", created_at=now, updated_at=now,
        ))
        m._rebuild_source_lifecycle()
        lc = m.get_source_lifecycle()
        assert lc.get("system", {}).get("created", 0) >= 1
        assert lc["system"]["abandoned"] >= 1


class TestActionabilityConsume:
    def test_review_abandons_sticky_metric_trigger_goal(self):
        from goals.review import GoalReview
        g = Goal(
            title="Sustained metric deficit: reasoning_coherence (high, 302s)",
            kind="system_health",
            status="active",
            evidence_types=["metric_deficit"],
            source_event="metric_triggers",
            source_scope="system",
        )
        update = GoalReview().review_goal(g)
        assert update.should_abandon is True
        assert "actionable" in update.reason

    def test_calibration_operator_review_is_kept(self):
        from goals.review import GoalReview
        g = Goal(
            title="Calibration domain 'weather' critically low: 0.10 (repair: operator review)",
            kind="system_health",
            status="active",
            evidence_types=["metric_deficit"],
            source_event="truth_calibration",
            source_scope="metric",
        )
        update = GoalReview().review_goal(g)
        assert update.should_abandon is False

    def test_user_requested_system_health_is_kept(self):
        from goals.review import GoalReview
        g = Goal(
            title="Fix the disk fill",
            kind="system_health",
            status="active",
            explicit_user_requested=True,
            source_scope="user",
        )
        update = GoalReview().review_goal(g)
        assert update.should_abandon is False
