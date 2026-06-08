"""#9 labeling sweep: policy + world-model marquee numbers must carry explicit
provenance so they can never render as unlabeled external measurements.

Pins (no scores changed — labels only):
  - world-model causal: predictive_accuracy is a measurement; persistence_accuracy
    and the pooled overall_accuracy are NOT.
  - mental simulator: avg_confidence is internally scored, not a measurement.
  - policy evaluator: nn_win_rate / nn_decisive_win_rate are shadow-only, not
    measurements.
"""
from __future__ import annotations

from cognition.causal_engine import CausalEngine
from cognition.simulator import MentalSimulator
from policy.evaluator import PolicyEvaluator


def _assert_labeled(prov, key, *, is_measurement):
    assert key in prov, f"{key} missing provenance"
    entry = prov[key]
    assert entry.get("is_measurement") is is_measurement
    assert entry.get("kind"), f"{key} provenance missing kind"
    assert entry.get("note"), f"{key} provenance missing note"


class TestWorldModelCausalProvenance:
    def test_causal_accuracy_carries_provenance(self):
        acc = CausalEngine().get_accuracy()
        prov = acc.get("provenance", {})
        _assert_labeled(prov, "predictive_accuracy", is_measurement=True)
        _assert_labeled(prov, "predictive_accuracy_live", is_measurement=True)
        _assert_labeled(prov, "persistence_accuracy", is_measurement=False)
        _assert_labeled(prov, "overall_accuracy", is_measurement=False)

    def test_pooled_overall_is_not_a_measurement(self):
        prov = CausalEngine().get_accuracy()["provenance"]
        assert prov["overall_accuracy"]["kind"] == "pooled_internally_scored"

    def test_every_accuracy_headline_has_a_label(self):
        acc = CausalEngine().get_accuracy()
        prov = acc["provenance"]
        for key in ("predictive_accuracy", "persistence_accuracy", "overall_accuracy"):
            assert key in acc          # the number is present
            assert key in prov         # ...and it cannot render unlabeled


class TestSimulatorProvenance:
    def test_avg_confidence_labeled_internally_scored(self):
        stats = MentalSimulator(CausalEngine()).get_stats()
        prov = stats.get("provenance", {})
        _assert_labeled(prov, "avg_confidence", is_measurement=False)
        assert prov["avg_confidence"]["kind"] == "internally_scored"


class TestPolicyProvenance:
    def test_policy_win_rates_labeled_shadow_only(self):
        status = PolicyEvaluator().get_status()
        prov = status.get("provenance", {})
        _assert_labeled(prov, "nn_win_rate", is_measurement=False)
        _assert_labeled(prov, "nn_decisive_win_rate", is_measurement=False)
        assert prov["nn_win_rate"]["kind"] == "shadow_only"

    def test_win_rate_headline_present_and_labeled(self):
        status = PolicyEvaluator().get_status()
        # the headline number exists AND has provenance — never unlabeled
        assert "nn_win_rate" in status
        assert "nn_win_rate" in status["provenance"]
