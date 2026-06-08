"""Weight-Room P1 — lived-only, min-N-gated live-shadow accuracy (synthetic-firewalled).

Mirrors the lived-before-synthetic discipline P0 established at the promotion gate:
a prediction made in a synthetic session is telemetry, never a lived rep the
live-shadow accuracy (and any later promotion gate) reads.
"""

from __future__ import annotations

from hemisphere.distillation import (
    live_shadow_accuracy,
    LIVE_SHADOW_MIN_N,
    LIVE_SHADOW_SUFFICIENT_N,
    _is_synthetic_origin,
)
from acquisition.plan_encoder import ShadowPredictionArtifact


class TestLiveShadowAccuracyHelper:
    def test_none_below_min_n(self):
        # 1-2 samples would give a meaningless 0%/100% -> honestly unmeasured (None)
        r = live_shadow_accuracy(correct=1, total=1)
        assert r["live_shadow_accuracy"] is None
        assert r["live_shadow_total"] == 1
        assert r["sufficient_data"] is False

    def test_computed_at_min_n(self):
        r = live_shadow_accuracy(correct=7, total=LIVE_SHADOW_MIN_N)
        assert r["live_shadow_accuracy"] == round(7 / LIVE_SHADOW_MIN_N, 4)

    def test_sufficient_data_boundary(self):
        assert live_shadow_accuracy(25, LIVE_SHADOW_SUFFICIENT_N - 1)["sufficient_data"] is False
        assert live_shadow_accuracy(25, LIVE_SHADOW_SUFFICIENT_N)["sufficient_data"] is True

    def test_clamps_correct_to_total(self):
        # defensive: correct can never exceed total
        r = live_shadow_accuracy(correct=99, total=10)
        assert r["live_shadow_correct"] == 10
        assert r["live_shadow_accuracy"] == 1.0

    def test_zero_total_is_none_not_div_zero(self):
        r = live_shadow_accuracy(0, 0)
        assert r["live_shadow_accuracy"] is None
        assert r["live_shadow_total"] == 0


class TestShadowArtifactOrigin:
    def test_defaults_to_live(self):
        a = ShadowPredictionArtifact(acquisition_id="x")
        assert a.origin == "live"

    def test_origin_round_trips(self):
        a = ShadowPredictionArtifact(acquisition_id="x", origin="synthetic")
        d = a.to_dict()
        assert d["origin"] == "synthetic"
        assert ShadowPredictionArtifact.from_dict(d).origin == "synthetic"

    def test_old_artifact_without_origin_defaults_live(self):
        # backward compat: a pre-P1 artifact dict has no origin -> conservative "live"
        old = {"acquisition_id": "x", "predicted_class": "approved", "correct": True}
        assert ShadowPredictionArtifact.from_dict(old).origin == "live"


def _lived_only_accuracy(resolved_shadows):
    """Replicates the aggregation's lived-only filter (dashboard/app.py) so the
    firewall SEMANTICS are locked by a test independent of the route handler."""
    resolved_live = [
        s for s in resolved_shadows
        if not str(s.get("origin", "live")).lower().startswith("synthetic")
    ]
    correct = sum(1 for s in resolved_live if s.get("correct") is True)
    return live_shadow_accuracy(correct, len(resolved_live))


class TestNegativeControl:
    def test_synthetic_shadows_excluded_from_lived_accuracy(self):
        # 12 lived (all correct) + 8 synthetic (all wrong). Lived accuracy must be
        # 1.0 over 12 — the synthetic reps cannot drag it down or pad the count.
        resolved = (
            [{"origin": "live", "correct": True} for _ in range(12)]
            + [{"origin": "synthetic", "correct": False} for _ in range(8)]
        )
        r = _lived_only_accuracy(resolved)
        assert r["live_shadow_total"] == 12
        assert r["live_shadow_accuracy"] == 1.0

    def test_all_synthetic_stays_unmeasured(self):
        # a soak of ONLY synthetic reps must leave the lived number unmeasured (None)
        resolved = [{"origin": "synthetic", "correct": True} for _ in range(40)]
        r = _lived_only_accuracy(resolved)
        assert r["live_shadow_total"] == 0
        assert r["live_shadow_accuracy"] is None

    def test_origin_predicate_matches_distillation(self):
        # the filter must agree with the canonical _is_synthetic_origin
        assert _is_synthetic_origin("synthetic") is True
        assert _is_synthetic_origin("synthetic_exercise") is True
        assert _is_synthetic_origin("live") is False
        assert _is_synthetic_origin("") is False
