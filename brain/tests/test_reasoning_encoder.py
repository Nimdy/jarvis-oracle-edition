"""Native Cognition #3, Phase 0 — the reasoning-substrate encoder (SHADOW).

The encoder reads the live belief field's grounding state and produces a single
``[0, 1]`` grounding-coherence signal — "how grounded is the substrate I would
reason from". These tests pin the firewall properties: it writes nothing, it
gates honestly (no field ⇒ 0.0, never an optimistic prior), it clamps pathological
inputs, and the shadow stance observer cites a REAL belief without emitting it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hemisphere.reasoning_encoder as re_enc
from hemisphere.reasoning_encoder import ReasoningEncoder


# --------------------------------------------------------------------------
# Context builders
# --------------------------------------------------------------------------

def _grounded_ctx():
    """A well-grounded belief field: mostly grounded provenance, calm, anchored,
    validations landing — the encoder should report high coherence."""
    return {
        "sources_available": True,
        "sampled_beliefs": 60,
        "inferred_count": 6,          # 90% grounded
        "mean_tension": 0.08,
        "orphan_rate": 0.05,
        "quarantine_pressure": 0.0,
        "high_tension_count": 2,
        "validation_total_outcomes": 30,
        "validation_grounded_rate": 0.8,
    }


def _ungrounded_ctx():
    """A poorly-grounded field: mostly inferred, orphaned, hot, validations
    failing — the encoder should report low coherence."""
    return {
        "sources_available": True,
        "sampled_beliefs": 40,
        "inferred_count": 38,         # ~5% grounded
        "mean_tension": 0.85,
        "orphan_rate": 0.86,
        "quarantine_pressure": 0.7,
        "high_tension_count": 34,
        "validation_total_outcomes": 25,
        "validation_grounded_rate": 0.08,
    }


# --------------------------------------------------------------------------
# Pure encoder
# --------------------------------------------------------------------------

class TestPureEncoder:
    def test_encode_dims_and_bounds(self):
        vec = ReasoningEncoder.encode(_grounded_ctx())
        assert len(vec) == ReasoningEncoder.FEATURE_DIM == 8
        assert all(0.0 <= v <= 1.0 for v in vec)

    def test_empty_brain_signal_zero(self):
        # No field, no validations — honest 0.0 (no grounded substrate to reason from).
        empty = {}
        assert ReasoningEncoder.compute_signal_value(empty) == 0.0
        assert ReasoningEncoder.encode(empty) == [0.0] * 8

    def test_grounded_field_high_signal(self):
        sig = ReasoningEncoder.compute_signal_value(_grounded_ctx())
        assert sig > 0.6, sig

    def test_ungrounded_field_low_signal(self):
        sig = ReasoningEncoder.compute_signal_value(_ungrounded_ctx())
        assert sig < 0.3, sig

    def test_grounded_beats_ungrounded(self):
        assert (ReasoningEncoder.compute_signal_value(_grounded_ctx())
                > ReasoningEncoder.compute_signal_value(_ungrounded_ctx()))

    def test_grounding_block_gated_on_sources(self):
        # Even with great-looking numbers, no readable field ⇒ block A all zero.
        ctx = _grounded_ctx()
        ctx["sources_available"] = False
        assert ReasoningEncoder.encode_grounding_block(ctx) == [0.0, 0.0, 0.0, 0.0]

    def test_grounding_block_gated_on_sampled(self):
        ctx = _grounded_ctx()
        ctx["sampled_beliefs"] = 0
        assert ReasoningEncoder.encode_grounding_block(ctx) == [0.0, 0.0, 0.0, 0.0]

    def test_validation_block_gated_on_outcomes(self):
        ctx = _grounded_ctx()
        ctx["validation_total_outcomes"] = 0
        assert ReasoningEncoder.encode_validation_block(ctx) == [0.0, 0.0]

    def test_pathological_inputs_clamped(self):
        bad = {
            "sources_available": True,
            "sampled_beliefs": 10,
            "inferred_count": 9999,           # inferred > sampled
            "mean_tension": float("nan"),
            "orphan_rate": -5.0,
            "quarantine_pressure": 12.0,
            "high_tension_count": 9999,
            "validation_total_outcomes": -3,  # coerced to 0 ⇒ block B gated off
            "validation_grounded_rate": float("inf"),
        }
        vec = ReasoningEncoder.encode(bad)       # assertions inside must not fire
        assert all(0.0 <= v <= 1.0 for v in vec)
        # inferred > sampled ⇒ grounded fraction floors at 0.0
        assert vec[0] == 0.0
        # high > sampled ⇒ not-dominated-by-hot floors at 0.0
        assert vec[7] == 0.0

    def test_signal_monotone_in_grounded_fraction(self):
        lo = _grounded_ctx(); lo["inferred_count"] = 55  # mostly inferred
        hi = _grounded_ctx(); hi["inferred_count"] = 0   # fully grounded
        assert (ReasoningEncoder.compute_signal_value(hi)
                > ReasoningEncoder.compute_signal_value(lo))


# --------------------------------------------------------------------------
# Shadow grounded-stance observer (cites a real belief, never emits)
# --------------------------------------------------------------------------

class TestShadowStance:
    def _ctx_with_top(self, **over):
        bt = SimpleNamespace(
            belief_id="bel_abc123",
            rendered_claim="the kitchen light is usually left on at night",
            provenance="model_inference",
            base_confidence=0.42,
            grounding_tension=0.71,
        )
        ctx = _grounded_ctx()
        ctx["_top_tensions"] = [bt]
        ctx.update(over)
        return ctx

    def test_cites_real_belief_and_counts(self):
        before = re_enc._grounded_stances_shadowed
        stance = re_enc.observe_grounded_stance(self._ctx_with_top())
        assert stance is not None
        assert stance["belief_id"] == "bel_abc123"
        assert "kitchen light" in stance["claim"]
        assert stance["authority"] == "shadow_observe_only"
        assert "bel_abc123" in stance["text"]
        # telemetry advanced, but nothing was emitted (function never touches the bus)
        assert re_enc._grounded_stances_shadowed == before + 1

    def test_none_when_no_top_tension(self):
        ctx = _grounded_ctx()
        ctx["_top_tensions"] = []
        assert re_enc.observe_grounded_stance(ctx) is None

    def test_posture_confident_when_grounded(self):
        stance = re_enc.observe_grounded_stance(self._ctx_with_top())
        assert "grounded confidence" in stance["posture"]

    def test_posture_tentative_when_under_grounded(self):
        ctx = self._ctx_with_top(**_ungrounded_ctx())
        # restore the fake top (the _ungrounded_ctx() update dropped it)
        ctx["_top_tensions"] = [SimpleNamespace(
            belief_id="bel_x", rendered_claim="x", provenance="model_inference",
            base_confidence=0.1, grounding_tension=0.9)]
        stance = re_enc.observe_grounded_stance(ctx)
        assert "tentatively" in stance["posture"]


# --------------------------------------------------------------------------
# Live boundary — defensive, default-safe, never raises
# --------------------------------------------------------------------------

@pytest.fixture
def _isolate_tension_promotion(tmp_path, monkeypatch):
    """Keep the tension-thought promotion singleton off the live ~/.jarvis file."""
    try:
        import consciousness.meta_cognitive_thoughts as mct
        monkeypatch.setattr(
            mct, "TENSION_THOUGHT_PROMOTION_PATH",
            tmp_path / "tension_thought_promotion.json", raising=False)
        mct.TensionThoughtPromotion.reset_instance()
        yield
        mct.TensionThoughtPromotion.reset_instance()
    except Exception:
        yield


class TestLiveGather:
    def test_gather_never_raises_and_default_safe(self, _isolate_tension_promotion):
        ctx = re_enc.gather_reasoning_context()
        # all expected keys present; with no live belief graph everything reads
        # ungrounded (the honest default), never an optimistic prior.
        for k in ("sources_available", "sampled_beliefs", "inferred_count",
                  "mean_tension", "orphan_rate", "quarantine_pressure",
                  "high_tension_count", "validation_total_outcomes",
                  "validation_grounded_rate", "_top_tensions"):
            assert k in ctx
        assert ctx["validation_total_outcomes"] == 0

    def test_compute_live_signal_never_raises(self, _isolate_tension_promotion):
        sig = re_enc.compute_live_signal()
        assert 0.0 <= sig <= 1.0

    def test_get_status_shape(self, _isolate_tension_promotion):
        st = re_enc.get_status()
        assert st["focus"] == "native_reasoning"
        assert st["phase"] == "P0_shadow"
        assert st["authority"] == "shadow_observe_only"
        assert 0.0 <= st["reasoning_signal"] <= 1.0
        assert len(st["feature_vector"]) == 8
