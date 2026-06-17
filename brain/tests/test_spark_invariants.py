"""Import-light invariant tests for the Spark / Grounding-Ring build.

These tests deliberately avoid torch (and any heavy ML import) so they run in a
CPU/torch-less sandbox.  They cover the SPARK_DESIGN honesty + safety invariants
that must hold regardless of runtime:

  * HomeostaticGovernor (``affect_regulation.regulate``): clamp band,
    mean-reversion toward baseline, bounded absorption, the refractory window,
    and oscillation freeze/auto-demote.
  * The affect cannot-lie clamp: a scalar whose every source reads 0 is forced
    to exactly 0.0 (no confabulated arousal), both at the source
    (``AffectState.compute``) and through the CapabilityGate nickname rewrite.
  * ProvenanceScorer is VIEW-ONLY: scoring a fake belief/edge store performs no
    mutation (no setattr, no mutator call, no write).
  * The new MetricSnapshot grounding-ring fields are default-safe / backward
    compatible (construct with no kwargs, round-trip via to_dict).

Run directly (``python3 brain/tests/test_spark_invariants.py``) or via
``python3 -m pytest brain/tests/test_spark_invariants.py -q`` — both supported;
the module guards its own ``sys.path`` so a direct run finds the ``brain`` pkgs.
"""

from __future__ import annotations

import os
import sys

# Make ``brain/`` importable whether run via pytest (rootdir=repo) or directly.
_BRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BRAIN not in sys.path:
    sys.path.insert(0, _BRAIN)

from consciousness import affect_regulation as AR
from consciousness import affect_state as AS
from consciousness.affect_state import AffectState


# ──────────────────────────────────────────────────────────────────────────────
# 1. HomeostaticGovernor — clamps / mean-reversion / bounded absorption / refractory
# ──────────────────────────────────────────────────────────────────────────────

def test_governor_clamp_band_never_saturates():
    """A relentless max-arousal drive can never push the level out of [0.05,0.95]."""
    st = AR.ScalarRegulatorState()
    t = 0.0
    for _ in range(200):
        st = AR.regulate(st, raw=1.0, dt=1.0, now_ts=t)
        t += 1.0
        assert AR.CLAMP_LO <= st.level <= AR.CLAMP_HI
    # Saturation = dead lever; the band stops strictly below the hard ceiling.
    assert st.level <= AR.CLAMP_HI
    assert st.level > AR.BASELINE  # it did move upward

    st2 = AR.ScalarRegulatorState()
    t = 0.0
    for _ in range(200):
        st2 = AR.regulate(st2, raw=0.0, dt=1.0, now_ts=t)
        t += 1.0
        assert AR.CLAMP_LO <= st2.level <= AR.CLAMP_HI
    assert st2.level < AR.BASELINE


def test_governor_mean_reversion_decays_toward_baseline():
    """With input at baseline, arousal bleeds back toward 0.5 (3600s half-life).

    §5 order: mean-reversion runs FIRST (old arousal bleeds off), THEN bounded
    absorption pulls toward the new raw reading.  Feeding ``raw = BASELINE`` keeps
    the absorption pulling the same direction as the decay, so the level provably
    moves toward baseline; over many ticks it converges on it.
    """
    st = AR.ScalarRegulatorState(level=0.9, last_update_ts=0.0)
    out = AR.regulate(st, raw=AR.BASELINE, dt=AR.MEAN_REVERSION_HALFLIFE_S, now_ts=AR.MEAN_REVERSION_HALFLIFE_S)
    # After one half-life the decay alone gives 0.70; absorption toward 0.5 pulls
    # it a touch further down — so it is strictly below 0.70 and strictly above 0.5.
    assert AR.BASELINE < out.level < 0.70
    # And it is strictly closer to baseline than it started.
    assert abs(out.level - AR.BASELINE) < abs(0.9 - AR.BASELINE)

    # Over a long quiet stretch it converges essentially onto baseline.
    s = AR.ScalarRegulatorState(level=0.9, last_update_ts=0.0)
    t = 0.0
    for _ in range(50):
        t += AR.MEAN_REVERSION_HALFLIFE_S
        s = AR.regulate(s, raw=AR.BASELINE, dt=AR.MEAN_REVERSION_HALFLIFE_S, now_ts=t)
    assert abs(s.level - AR.BASELINE) < 1e-3


def test_governor_pure_decay_function_is_monotonic_and_noop_on_nonpositive_dt():
    assert AR._half_life_decay(0.9, 0.0) == 0.9          # dt<=0 is a no-op
    assert AR._half_life_decay(0.9, -5.0) == 0.9
    half = AR._half_life_decay(0.9, AR.MEAN_REVERSION_HALFLIFE_S)
    assert abs(half - 0.70) < 1e-6


def test_governor_bounded_absorption_caps_a_single_step():
    """A huge incoming swing is absorbed by at most MAX_STEP*GAIN in one tick."""
    st = AR.ScalarRegulatorState(level=0.5, last_update_ts=0.0)
    out = AR.regulate(st, raw=1.0, dt=0.0, now_ts=0.0)  # dt=0 → no decay this tick
    # applied = GAIN * clamp(raw-level, -MAX_STEP, +MAX_STEP) = 0.5 * 0.15
    assert abs(out.level - (0.5 + AR.GAIN * AR.MAX_STEP)) < 1e-9
    assert out.level <= 0.5 + AR.MAX_STEP  # never absorbs more than the cap


def test_governor_refractory_window_opens_and_halves_step():
    """A single incoming swing >= REFRACTORY_SWING opens the 180s window."""
    st = AR.ScalarRegulatorState(level=0.5, last_update_ts=0.0)
    out = AR.regulate(st, raw=0.9, dt=0.0, now_ts=100.0)  # raw_delta 0.4 >= 0.25
    assert out.refractory_until_ts == 100.0 + AR.REFRACTORY_WINDOW_S
    # Inside the window the effective max step halves: a second big swing moves
    # the level by at most MAX_STEP/2 * GAIN/2.
    level_before = out.level
    out2 = AR.regulate(out, raw=1.0, dt=0.0, now_ts=150.0)  # still inside window
    moved = out2.level - level_before
    assert 0.0 < moved <= (AR.MAX_STEP * 0.5) * (AR.GAIN * 0.5) + 1e-9


def test_governor_oscillation_freezes_and_signals_demote():
    """Sustained sign-flipping freezes the lever and raises the demote signal."""
    st = AR.ScalarRegulatorState(level=0.5, last_update_ts=0.0)
    t = 0.0
    saw_demote = False
    # Alternate hard high / hard low with dt=0 (no decay) to force sign flips.
    for i in range(12):
        raw = 1.0 if i % 2 == 0 else 0.0
        st = AR.regulate(st, raw=raw, dt=0.0, now_ts=t)
        t += 1.0
        if st.demote_signal:
            saw_demote = True
            break
    assert saw_demote, "excessive oscillation must raise demote_signal"
    assert st.frozen
    # A frozen lever stays frozen but keeps decaying honestly toward baseline.
    frozen_next = AR.regulate(st, raw=1.0, dt=AR.MEAN_REVERSION_HALFLIFE_S, now_ts=t + 5)
    assert frozen_next.frozen
    assert abs(frozen_next.level - AR.BASELINE) <= abs(st.level - AR.BASELINE) + 1e-9


def test_governor_reset_freeze_rearms():
    st = AR.ScalarRegulatorState(frozen=True, demote_signal=True, recent_steps=[0.1, -0.1])
    out = AR.reset_freeze(st)
    assert not out.frozen and not out.demote_signal and out.recent_steps == []


def test_governor_state_roundtrip_backward_compatible():
    """from_dict tolerates a missing/empty payload (older persisted state)."""
    assert AR.ScalarRegulatorState.from_dict(None).level == AR.BASELINE
    assert AR.ScalarRegulatorState.from_dict({}).level == AR.BASELINE
    full = AR.ScalarRegulatorState(level=0.7, update_count=3).to_dict()
    again = AR.ScalarRegulatorState.from_dict(full)
    assert again.level == 0.7 and again.update_count == 3


def test_clamp_levers_keeps_everything_in_native_band():
    out = AR.clamp_levers({
        "cadence_multiplier": 99.0,
        "memory_reinforcement": -5.0,
        "interval_multipliers": {"curiosity": 50.0, "belief_graph": 0.0},
        "urgency_bias": {"grounding": 7.0},
        "unknown_key": "passthrough",
    })
    assert out["cadence_multiplier"] == AR.CADENCE_HI
    assert out["memory_reinforcement"] == AR.REINFORCEMENT_LO
    assert out["interval_multipliers"]["curiosity"] == AR.INTERVAL_HI
    assert out["interval_multipliers"]["belief_graph"] == AR.INTERVAL_LO
    assert out["urgency_bias"]["grounding"] == AR.URGENCY_HI
    assert out["unknown_key"] == "passthrough"  # additive / backward compatible


def test_neutral_levers_are_identity():
    n = AR.neutral_levers()
    assert n["cadence_multiplier"] == 1.0
    assert n["memory_reinforcement"] == 1.0
    assert n["interval_multipliers"] == {} and n["urgency_bias"] == {}


# ──────────────────────────────────────────────────────────────────────────────
# 2. Affect cannot-lie clamp — a scalar with all-zero sources is forced to 0.0
# ──────────────────────────────────────────────────────────────────────────────

def test_affect_cannot_lie_clamp_at_source():
    """No signals at all → dopamine & cortisol forced to exactly 0.0 + flagged."""
    a = AffectState()  # fresh instance, not the module singleton
    snap = a.compute(signals=None, delta_tracker=None, world_model=None, now_ts=1000.0)
    assert snap.dopamine.raw == 0.0
    assert snap.dopamine.cannot_lie_clamped is True
    assert snap.dopamine.all_sources_zero is True
    assert snap.cortisol.raw == 0.0
    assert snap.cortisol.cannot_lie_clamped is True
    assert snap.cortisol.all_sources_zero is True
    # Provenance is always present (auditable, never "felt").
    assert snap.cortisol.provenance and snap.dopamine.provenance


def test_affect_cortisol_clamp_exact_zero_with_all_zero_sources():
    """_compute_cortisol(None) reads 0/0/0 in a torch-less env → exactly 0.0."""
    ro = AffectState._compute_cortisol(None)
    assert ro.raw == 0.0
    assert ro.cannot_lie_clamped is True
    for _sig, pair in ro.provenance.items():
        assert pair[1] == 0.0  # every source read 0


def test_affect_backed_scalar_is_not_clamped():
    """A genuinely-backed scalar (serotonin high-when-calm) is NOT cannot-lie clamped."""
    a = AffectState()
    snap = a.compute(now_ts=1.0)
    # serotonin's coherence term is 1.0 when contradiction_debt is 0 (calm) — real
    # backing, so it must NOT be clamped to zero.
    assert snap.serotonin.raw > 0.0
    assert snap.serotonin.cannot_lie_clamped is False


def test_affect_cannot_lie_clamp_through_capability_gate():
    """The gate rewrites an affect-nickname claim, never asserting a feeling, and
    the confabulation ledger flags an unbacked claim when the scalar is clamped."""
    from skills.capability_gate import CapabilityGate

    g = CapabilityGate()

    # No snapshot computed yet → honest "no current reading" + classified unbacked.
    AS.affect_state._last = None
    out = g.check_text("My cortisol is high right now.")
    assert "cortisol" not in out.lower()          # the nickname never survives
    assert "unresolved-tension" in out.lower()    # named real signal phrase
    assert g._confab_unbacked >= 1

    # A backed, correctly-directed reading → cite the real source + value.
    ro_c = AS.AffectReadout(
        nickname="cortisol", raw=0.8, level=0.7,
        provenance={
            "contradiction_debt": ["contradiction_debt", 0.62],
            "quarantine_pressure": ["QuarantinePressure.composite", 0.1],
            "friction_rate": ["DriveSignals.gate_blocks_recent", 0.0],
        },
        all_sources_zero=False,
    )
    ro_d = AS.AffectReadout(nickname="dopamine", raw=0.0, all_sources_zero=True, cannot_lie_clamped=True)
    ro_s = AS.AffectReadout(nickname="serotonin", raw=0.5)
    AS.affect_state._last = AS.AffectSnapshot(timestamp=1.0, dopamine=ro_d, serotonin=ro_s, cortisol=ro_c)
    out2 = g.check_text("My cortisol is high right now.")
    assert "cortisol" not in out2.lower()
    assert "contradiction_debt=0.62" in out2
    assert g._confab_backed_anthropomorphized >= 1


# ──────────────────────────────────────────────────────────────────────────────
# 3. ProvenanceScorer is VIEW-ONLY — scoring mutates nothing
# ──────────────────────────────────────────────────────────────────────────────

class _FrozenBelief:
    """A stand-in BeliefRecord that records any attempted mutation as a failure."""

    __slots__ = (
        "belief_id", "provenance", "claim_type", "canonical_subject",
        "identity_subject_id", "belief_confidence", "rendered_claim", "_mutations",
    )

    def __init__(self, bid, provenance, conf):
        object.__setattr__(self, "belief_id", bid)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "claim_type", "fact")
        object.__setattr__(self, "canonical_subject", "the sky")
        object.__setattr__(self, "identity_subject_id", "")
        object.__setattr__(self, "belief_confidence", conf)
        object.__setattr__(self, "rendered_claim", "the sky is blue")
        object.__setattr__(self, "_mutations", [])

    def __setattr__(self, name, value):  # any write is recorded
        object.__getattribute__(self, "_mutations").append((name, value))
        object.__setattr__(self, name, value)

    # Mutators a scorer must never call.
    def add_evidence(self, *a, **k):
        raise AssertionError("ProvenanceScorer called a belief mutator (add_evidence)")

    def update(self, *a, **k):
        raise AssertionError("ProvenanceScorer called a belief mutator (update)")


class _FakeBeliefStore:
    def __init__(self, beliefs):
        self._beliefs = beliefs
        self.saved = 0

    def get_active_beliefs(self):
        return list(self._beliefs)  # defensive copy — reads never expose internals

    # Mutators that must never be invoked by a view-only scorer.
    def save(self, *a, **k):
        raise AssertionError("ProvenanceScorer called belief_store.save")

    def add_belief(self, *a, **k):
        raise AssertionError("ProvenanceScorer called belief_store.add_belief")


def test_provenance_scorer_is_view_only():
    from epistemic.provenance_scorer import ProvenanceScorer

    beliefs = [
        _FrozenBelief("b1", "model_inference", 0.3),
        _FrozenBelief("b2", "observed", 0.9),
        _FrozenBelief("b3", "model_inference", 0.2),
    ]
    store = _FakeBeliefStore(beliefs)

    scorer = ProvenanceScorer()
    # Force the scorer to use our fake store + no edge store (read-only path).
    scorer._resolve_stores = lambda: (store, None)  # type: ignore[method-assign]

    report = scorer.compute(top_n=5)

    # It produced a real reading (sources available, sampled all 3).
    assert report.sources_available is True
    assert report.sampled_beliefs == 3
    assert report.inferred_count == 2
    # ... and mutated NOTHING on any belief or the store.
    for b in beliefs:
        assert b._mutations == [], f"belief {b.belief_id} was mutated: {b._mutations}"
    assert store.saved == 0
    # An inferred, orphaned belief reads hot; a GROUNDED belief (b2=observed) reads
    # 0 tension and drops out of the report entirely — it's been externally validated,
    # so it is never re-queued for grounding (regardless of hub leverage).
    by_id = {t.belief_id: t for t in report.top_tensions}
    assert by_id["b1"].grounding_tension > 0.0 and by_id["b1"].is_inferred
    assert "b2" not in by_id


def test_provenance_scorer_default_safe_without_store():
    from epistemic.provenance_scorer import ProvenanceScorer

    scorer = ProvenanceScorer()
    scorer._resolve_stores = lambda: (None, None)  # type: ignore[method-assign]
    report = scorer.compute()
    assert report.sources_available is False
    assert report.aggregate_tension == 0.0
    assert report.top_tensions == []


# ──────────────────────────────────────────────────────────────────────────────
# 4. New MetricSnapshot grounding-ring fields are default-safe / backward compatible
# ──────────────────────────────────────────────────────────────────────────────

def test_metric_snapshot_new_fields_default_safe():
    from autonomy.delta_tracker import MetricSnapshot

    # Construct with NO kwargs — every grounding-ring field must default safely.
    snap = MetricSnapshot()
    assert snap.orphan_rate == 0.0
    assert snap.inference_validation_gap == 0.0
    assert snap.external_validation_rate == 0.0
    assert snap.grounded_inferred_ratio == 0.0
    assert snap.avg_chain_length == 0.0

    # to_dict emits them (so persisted JSONL carries them going forward) ...
    d = snap.to_dict()
    for k in (
        "orphan_rate", "inference_validation_gap", "external_validation_rate",
        "grounded_inferred_ratio", "avg_chain_length",
    ):
        assert k in d and d[k] == 0.0

    # ... and a snapshot built the legacy way (only the original fields) still
    # constructs — i.e. older callers/readers that omit the new fields are fine.
    legacy = MetricSnapshot(timestamp=123.0, confidence_avg=0.5, memory_count=10)
    assert legacy.orphan_rate == 0.0          # defaulted, not required
    assert legacy.to_dict()["external_validation_rate"] == 0.0


def test_spark_metrics_dataclass_default_safe():
    from autonomy.spark_metrics import SparkMetrics

    m = SparkMetrics()
    d = m.to_dict()
    for k in (
        "orphan_rate", "inference_validation_gap", "external_validation_rate",
        "grounded_inferred_ratio", "avg_chain_length",
    ):
        assert k in d and d[k] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Direct-run harness (no pytest required; torch-free).
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed of {len(tests)}")
    sys.exit(1 if failed else 0)
