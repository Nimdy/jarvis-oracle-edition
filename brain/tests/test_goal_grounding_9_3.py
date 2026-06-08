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


def test_sustained_deficit_creates_after_streak():
    _reset()
    detect_metric_deficits(_BAD, None, None, uptime_s=300)        # streak 1 -> nothing
    s2 = detect_metric_deficits(_BAD, None, None, uptime_s=300)   # streak 2 -> create
    assert len(s2) == 1
    assert s2[0].source == "health_monitor"
    assert s2[0].source_scope == "metric"
    assert sp._producer_stats["metric_created"] == 1


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


def test_active_deficits_source_unaffected_by_gate():
    # active_deficits already carries its own duration>=300s gate; it should still
    # create immediately (it is not routed through the consecutive-tick streak).
    _reset()
    ad = {"tick_overrun": {"duration_s": 600, "severity": "high"}}
    s = detect_metric_deficits(None, None, ad, uptime_s=300)
    assert len(s) == 1 and s[0].source == "metric_triggers"


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
