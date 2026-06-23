"""Tests for the Fidelity keystone (#9): route maturity reporting onto the
MEASURED scoreboard composite, and label self-scored numbers.

Pins:
  - the composite only enables on real coverage (>=2 categories, >=5 samples each);
  - comparator-less categories stay visibly empty (never painted as measurements);
  - the banner headline reports the MEASURED composite + coverage, not the legacy
    COMPOSITE_ENABLED flag;
  - the oracle benchmark (a self-grade) is explicitly labeled self_scored.
"""
from __future__ import annotations

from jarvis_eval.dashboard_adapter import _build_scoreboard, build_dashboard_snapshot
from jarvis_eval.config import SCOREBOARD_MIN_SAMPLES, SCOREBOARD_MIN_CATEGORIES, COMPOSITE_ENABLED


# comparator -> epistemic class gates the composite (>=1 world_judged required before it enables)
_WORLD = "world_model_causal.predictive_accuracy_live"   # world_judged
_SELF = "pvl.process_contracts"                          # self_verified


def _score(cat, score, n, comparator=None):
    e = {"category": cat, "score": score, "sample_size": n}
    if comparator:
        e["raw_metrics"] = {"comparator": comparator}
    return e


def _two_measured():
    # epistemic_integrity is WORLD-JUDGED (satisfies the >=1 world_judged gate); self_report_honesty
    # is self_verified (real PVL comparator); capability below floor.
    return [
        _score("epistemic_integrity", 0.83, SCOREBOARD_MIN_SAMPLES + 1, _WORLD),
        _score("self_report_honesty", 0.77, 108, _SELF),
        _score("capability", 1.0, 1),  # below floor -> not measured
    ]


class TestBuildScoreboard:
    def test_enables_on_two_measured(self):
        sb = _build_scoreboard(_two_measured())
        assert sb["composite_enabled"] is True
        assert sb["composite"] is not None
        assert sb["coverage"]["measured"] == 2
        assert sb["coverage"]["total"] == 7

    def test_one_measured_stays_disabled(self):
        sb = _build_scoreboard([_score("self_report_honesty", 0.77, 108)])
        assert sb["composite_enabled"] is False
        assert sb["composite"] is None
        assert sb["coverage"]["measured"] == 1

    def test_below_floor_not_measured(self):
        sb = _build_scoreboard([_score("capability", 1.0, SCOREBOARD_MIN_SAMPLES - 1)])
        cap = next(b for b in sb["bars"] if b["category"] == "capability")
        assert cap["measured"] is False

    def test_comparatorless_categories_stay_empty(self):
        sb = _build_scoreboard(_two_measured())
        for cat in ("memory_integrity", "safety_immunity", "autonomy_long_horizon", "human_value"):
            bar = next(b for b in sb["bars"] if b["category"] == cat)
            assert bar["measured"] is False
            assert bar["score"] is None

    def test_composite_is_mean_of_measured(self):
        sb = _build_scoreboard([
            _score("epistemic_integrity", 0.80, 10, _WORLD),
            _score("self_report_honesty", 0.90, 10, _SELF),
        ])
        assert sb["composite"] == 0.85

    def test_self_verified_only_stays_disabled(self):
        # Two MEASURED categories but BOTH self_verified -> composite must NOT enable (no world judge).
        sb = _build_scoreboard([
            _score("self_report_honesty", 0.90, 10, _SELF),
            _score("capability", 0.80, 10, _SELF),
        ])
        assert sb["composite_enabled"] is False
        assert sb["coverage"]["world_judged_measured"] == 0

    def test_unknown_comparator_fails_closed(self):
        # A measured category with no/unknown comparator is NOT counted world_judged.
        sb = _build_scoreboard([
            _score("epistemic_integrity", 0.80, 10),          # no comparator -> unknown
            _score("self_report_honesty", 0.90, 10, _SELF),
        ])
        assert sb["coverage"]["world_judged_measured"] == 0
        assert sb["composite_enabled"] is False


class TestBannerRouting:
    def _snapshot(self, scores):
        return build_dashboard_snapshot({}, {}, [], recent_scores=scores)

    def test_banner_reports_measured_composite(self):
        snap = self._snapshot(_two_measured())
        banner = snap["banner"]
        sb = snap["scoreboard"]
        # headline mirrors the MEASURED scoreboard, not the legacy flag
        assert banner["composite_enabled"] == sb["composite_enabled"] is True
        assert banner["composite"] == sb["composite"]
        assert banner["composite_coverage"] == sb["coverage"]
        assert banner["composite_source"] == "scoreboard_measured"
        assert banner["legacy_composite_flag"] == COMPOSITE_ENABLED  # unchanged, clearly named

    def test_banner_honest_when_uncovered(self):
        snap = self._snapshot([_score("self_report_honesty", 0.77, 108)])  # only 1 measured
        assert snap["banner"]["composite_enabled"] is False
        assert snap["banner"]["composite"] is None
        assert snap["banner"]["composite_coverage"]["measured"] == 1

    def test_oracle_benchmark_labeled_self_scored(self):
        snap = self._snapshot(_two_measured())
        ob = snap["oracle_benchmark"]
        assert ob.get("self_scored") is True
        assert ob.get("is_measurement") is False
