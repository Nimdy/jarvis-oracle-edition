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


@pytest.fixture(autouse=True)
def _reset_encoder_module_state(monkeypatch):
    """Reset the module-level context cache + stance cooldown so the cache/cooldown
    tests are deterministic and never leak state across the suite (these globals only
    affect _cached_context / get_status / maybe_observe_grounded_stance)."""
    monkeypatch.setattr(re_enc, "_ctx_cache", None, raising=False)
    monkeypatch.setattr(re_enc, "_ctx_cache_ts", 0.0, raising=False)
    monkeypatch.setattr(re_enc, "_last_stance_ts", 0.0, raising=False)
    yield


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


# --------------------------------------------------------------------------
# Shared context cache (keeps ProvenanceScorer.compute off the 2s snapshot hot path)
# --------------------------------------------------------------------------

class TestContextCache:
    def test_memoizes_within_ttl(self, monkeypatch):
        clock = {"t": 100.0}
        monkeypatch.setattr(re_enc.time, "monotonic", lambda: clock["t"])
        gathers = {"n": 0}
        monkeypatch.setattr(re_enc, "gather_reasoning_context",
                            lambda engine=None: {"g": (gathers.__setitem__("n", gathers["n"] + 1) or gathers["n"])})
        a = re_enc._cached_context()
        b = re_enc._cached_context()                 # within TTL -> no second gather
        assert a is b and gathers["n"] == 1
        clock["t"] += re_enc._CTX_CACHE_TTL_S + 1.0
        re_enc._cached_context()                     # past TTL -> re-gather
        assert gathers["n"] == 2

    def test_get_status_uses_cache(self, monkeypatch):
        clock = {"t": 500.0}
        monkeypatch.setattr(re_enc.time, "monotonic", lambda: clock["t"])
        gathers = {"n": 0}
        monkeypatch.setattr(re_enc, "gather_reasoning_context",
                            lambda engine=None: (gathers.__setitem__("n", gathers["n"] + 1) or dict(_grounded_ctx())))
        re_enc.get_status(); re_enc.get_status()      # two polls within TTL -> one gather
        assert gathers["n"] == 1


# --------------------------------------------------------------------------
# Cooldown-gated cadence wrapper (fires the shadow stance ~once / _STANCE_COOLDOWN_S)
# --------------------------------------------------------------------------

class TestMaybeObserveCooldown:
    def test_fires_then_gated_then_fires(self, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr(re_enc.time, "monotonic", lambda: clock["t"])
        calls = {"n": 0}
        monkeypatch.setattr(re_enc, "observe_grounded_stance",
                            lambda ctx=None: (calls.__setitem__("n", calls["n"] + 1) or {"ok": True}))
        assert re_enc.maybe_observe_grounded_stance({"x": 1}) is not None   # last=0 -> fires
        assert calls["n"] == 1
        assert re_enc.maybe_observe_grounded_stance({"x": 1}) is None        # within cooldown -> gated
        assert calls["n"] == 1
        clock["t"] += re_enc._STANCE_COOLDOWN_S + 0.01
        assert re_enc.maybe_observe_grounded_stance({"x": 1}) is not None    # cooldown elapsed -> fires
        assert calls["n"] == 2

    def test_attempt_advances_timestamp_even_when_observe_none(self, monkeypatch):
        # Empty/ungrounded field (observe returns None) must STILL advance the cooldown
        # so we never re-gather/compute every tick on a cold brain.
        clock = {"t": 5000.0}
        monkeypatch.setattr(re_enc.time, "monotonic", lambda: clock["t"])
        calls = {"n": 0}
        monkeypatch.setattr(re_enc, "observe_grounded_stance",
                            lambda ctx=None: (calls.__setitem__("n", calls["n"] + 1) or None))
        assert re_enc.maybe_observe_grounded_stance({"x": 1}) is None        # fired (delegated) -> None
        assert calls["n"] == 1
        assert re_enc.maybe_observe_grounded_stance({"x": 1}) is None        # gated, no delegate
        assert calls["n"] == 1

    def test_uses_cached_context_when_ctx_none(self, monkeypatch):
        clock = {"t": 9000.0}
        monkeypatch.setattr(re_enc.time, "monotonic", lambda: clock["t"])
        sentinel = {"_top_tensions": [], "sentinel": True}
        monkeypatch.setattr(re_enc, "_cached_context", lambda engine=None: sentinel)
        seen = {}
        monkeypatch.setattr(re_enc, "observe_grounded_stance",
                            lambda ctx=None: (seen.__setitem__("ctx", ctx) or {"ok": True}))
        re_enc.maybe_observe_grounded_stance(None)
        assert seen["ctx"] is sentinel                                       # reused cache, no fresh gather


# --------------------------------------------------------------------------
# Phase 1 WITNESS — read-only legibility of the reasoning_validation earning stream.
# Asserts the firewall + earn-don't-declare invariants (the whole point of the slice).
# --------------------------------------------------------------------------

class TestReasoningStreamWitness:
    def _sig(self, grounded=None, signal=None, origin="live"):
        return SimpleNamespace(
            data={"grounded": grounded, "reasoning_signal": signal},
            origin=origin, fidelity=1.0, timestamp=0.0)

    def _patch_batch(self, monkeypatch, batch):
        """Replace the live ring-buffer reader with a spy returning a fake batch.
        Returns a dict capturing the args the witness asked for."""
        seen = {}
        def _fake(teacher, limit=200, min_fidelity=0.0, lived_only=False):
            seen["teacher"] = teacher
            seen["lived_only"] = lived_only
            return list(batch)
        import hemisphere.distillation as dist
        monkeypatch.setattr(dist.distillation_collector, "get_training_batch", _fake)
        # the witness must NEVER write — make record() blow up if it's ever called
        monkeypatch.setattr(dist.distillation_collector, "record",
                            lambda *a, **k: pytest.fail("witness must not write (record called)"))
        return seen

    def test_zero_reps_honestly_unmeasured(self, monkeypatch):
        self._patch_batch(monkeypatch, [])
        st = re_enc.reasoning_stream_status()
        assert st["total_reps"] == 0
        assert st["grounded_rate"] is None                      # never a fake 0.0
        assert re_enc.REASONING_STREAM_PHASE1_THRESHOLD == 30
        assert st["reps_to_phase1_threshold"] == 30
        assert st["phase1_prototype_ready"] is False
        assert st["authority"] == "shadow_observe_only"

    def test_below_min_n_rate_stays_none(self, monkeypatch):
        # N=9 < LIVE_SHADOW_MIN_N (10) — the core anti-fabrication assertion.
        batch = [self._sig(grounded=True)] * 5 + [self._sig(grounded=False)] * 4
        self._patch_batch(monkeypatch, batch)
        st = re_enc.reasoning_stream_status()
        assert st["total_reps"] == 9
        assert st["grounded_count"] == 5 and st["ungrounded_count"] == 4
        assert st["grounded_rate"] is None

    def test_at_min_n_rate_matches_honesty_floor(self, monkeypatch):
        from hemisphere.distillation import live_shadow_accuracy
        batch = [self._sig(grounded=True)] * 7 + [self._sig(grounded=False)] * 5  # N=12
        self._patch_batch(monkeypatch, batch)
        st = re_enc.reasoning_stream_status()
        assert st["total_reps"] == 12 and st["grounded_count"] == 7
        # reuses the shared floor verbatim (not a hand-rolled rate)
        assert st["grounded_rate"] == live_shadow_accuracy(7, 12)["live_shadow_accuracy"] == round(7 / 12, 4)

    def test_external_only_lived_filter_requested(self, monkeypatch):
        seen = self._patch_batch(monkeypatch, [self._sig(grounded=True)])
        re_enc.reasoning_stream_status()
        assert seen["teacher"] == "reasoning_validation"
        assert seen["lived_only"] is True                       # synthetic excluded at source

    def test_grounded_counts_only_true(self, monkeypatch):
        # grounded must be exactly True; None / False are NOT grounded.
        batch = [self._sig(grounded=True), self._sig(grounded=False), self._sig(grounded=None)]
        self._patch_batch(monkeypatch, batch)
        st = re_enc.reasoning_stream_status()
        assert st["grounded_count"] == 1 and st["total_reps"] == 3

    def test_phase1_ready_flips_no_authority(self, monkeypatch):
        batch = [self._sig(grounded=True)] * re_enc.REASONING_STREAM_PHASE1_THRESHOLD
        self._patch_batch(monkeypatch, batch)
        st = re_enc.reasoning_stream_status()
        assert st["phase1_prototype_ready"] is True
        assert st["authority"] == "shadow_observe_only"         # ready is purely informational
        assert not hasattr(re_enc, "NativeReasoningPromotion")  # no gate exists

    def test_never_raises_returns_unavailable(self, monkeypatch):
        import hemisphere.distillation as dist
        def _boom(*a, **k):
            raise RuntimeError("buffer down")
        monkeypatch.setattr(dist.distillation_collector, "get_training_batch", _boom)
        st = re_enc.reasoning_stream_status()
        assert st["error"] == "unavailable"
        assert st["authority"] == "shadow_observe_only"

    def test_mean_signal_skips_none(self, monkeypatch):
        self._patch_batch(monkeypatch, [self._sig(signal=None), self._sig(signal=None)])
        assert re_enc.reasoning_stream_status()["mean_reasoning_signal"] is None
        self._patch_batch(monkeypatch, [self._sig(signal=0.6), self._sig(signal=None), self._sig(signal=0.8)])
        assert re_enc.reasoning_stream_status()["mean_reasoning_signal"] == round((0.6 + 0.8) / 2, 4)

    def test_get_status_carries_stream_and_no_qwen_import(self, monkeypatch, _isolate_tension_promotion):
        self._patch_batch(monkeypatch, [self._sig(grounded=True)])
        st = re_enc.get_status()
        assert "reasoning_validation_stream" in st
        assert st["reasoning_validation_stream"]["authority"] == "shadow_observe_only"
        # NO QWEN TOUCH: the witness module must not reference the replacement targets.
        src = open(re_enc.__file__).read()
        for forbidden in ("existential_reasoning", "philosophical_dialogue", "_try_llm_enrich"):
            assert forbidden not in src
