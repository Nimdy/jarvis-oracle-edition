"""OSV P0 acceptance tests (GitHub #23).

Proves the substrate is honest + read-only:
  - synthesizer is a pure read-only function (input not mutated, no canonical writes);
  - every reported value carries provenance;
  - self-scored / internal / shadow / synthetic values can NEVER render as measurements;
  - gaps are first-class and explicit;
  - dormant/gated capabilities render as dormant, not available;
  - /api/self-view model has a stable schema;
  - missing subsystem data degrades to gap/unknown — never a fabricated default.
"""
from __future__ import annotations

import copy

import pytest

import cognition.self_view as sv
from cognition.self_view.provenance import (
    ALL_PROVENANCE,
    MEASUREMENT_LEVELS,
    Fact,
    Provenance,
    gap,
    unknown,
)
from cognition.self_view.synthesizer import SCHEMA_VERSION, SelfModel, SelfViewSynthesizer


def _full_sources():
    return {
        "skills": {"by_status": {"verified": 14}, "skills": [
            {"skill_id": "web_scraping_v1", "status": "verified",
             "learning_job_id": "job_x", "updated_at": 100},
            {"skill_id": "speech_output", "status": "verified", "updated_at": 50},
        ]},
        "scoreboard": {
            "composite": 0.83, "composite_enabled": True,
            "coverage": {"measured": 2, "total": 7},
            "bars": [
                {"category": "epistemic_integrity", "score": 0.83, "sample_size": 6, "measured": True},
                {"category": "self_report_honesty", "score": 0.77, "sample_size": 108, "measured": True},
                {"category": "memory_integrity", "score": None, "sample_size": 0, "measured": False},
                {"category": "safety_immunity", "score": None, "sample_size": 0, "measured": False},
            ],
        },
        "causal": {"predictive_accuracy": 0.8, "predictive_total": 10, "persistence_accuracy": 0.9},
        "simulator": {"avg_confidence": 0.72},
        "simulator_promotion": {"level": 0, "level_name": "shadow", "total_validated": 12000},
        "world_model_promotion": {"level": 1, "level_name": "advisory", "total_validated": 80},
        "policy": {"nn_win_rate": 0.55, "mode": "shadow"},
        "cognitive_planner": {"active": False, "reason": "simulator_not_advisory (12000/100)"},
        "counterfactual": {"active": False, "reason": "data_gated"},
        "recent_changes": [{"name": "web_scraping_v1", "kind": "skill", "when": 100}],
    }


def _fake_snapshot():
    """Mimics the dashboard build_cache output (verified shapes)."""
    return {
        "consciousness": {"stage": "integrative", "awareness_level": 0.98,
                          "transcendence_level": 10.0},
        "evolution": {"stage": "integrative", "transcendence_level": 10.0},
        "policy": {"mode": "shadow", "nn_win_rate": 0.009},
        "self_improve": {"active": True, "stage": 2, "effective_dry_run": True},
        "world_model": {"level": 2, "level_name": "active", "total_validated": 111465},
        "simulator": {"level": 0, "level_name": "shadow", "total_validated": 12236},
        "observer": {"awareness_level": 0.98, "observation_count": 40},
        "mutations": {"count": 5, "rollback_count": 0},
        "memory": {"opaque": "shape"},          # unknown lifecycle -> present/unknown
        "quarantine": {},                        # CURATED + empty -> gap (first-class)
        "summary": {"skip": "me"},               # must be skipped
        "_internal": {"skip": "me"},             # must be skipped
        "empty_thing": {},                       # non-curated + empty -> omitted
    }


def _all_facts(model: SelfModel):
    for dim in (model.structural, model.performance, model.maturity, model.belief,
                model.change, model.subsystems):
        for v in dim.values():
            if isinstance(v, Fact):
                yield v
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, Fact):
                        yield x


# ---------------------------------------------------------------------------
# Provenance primitive
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_only_measured_is_a_measurement(self):
        assert Fact(0.8, Provenance.MEASURED).is_measurement is True
        for lvl in (ALL_PROVENANCE - MEASUREMENT_LEVELS):
            assert Fact(1, lvl).is_measurement is False

    def test_invalid_provenance_coerces_to_unknown(self):
        f = Fact(1, "totally_made_up")
        assert f.provenance == Provenance.UNKNOWN
        assert f.is_measurement is False

    def test_is_measurement_cannot_be_set_independently(self):
        # there is no constructor arg for is_measurement — it derives from provenance only
        f = Fact(0.9, Provenance.SELF_SCORED)
        assert f.to_dict()["is_measurement"] is False

    def test_gap_and_unknown_helpers(self):
        assert gap("x").is_absent and not gap("x").is_measurement
        assert unknown("y").is_absent and not unknown("y").is_measurement


# ---------------------------------------------------------------------------
# Synthesizer: honesty
# ---------------------------------------------------------------------------

class TestSynthesizerHonesty:
    def test_predictive_is_measurement_others_are_not(self):
        m = SelfViewSynthesizer().synthesize(_full_sources(), now=1.0)
        assert m.performance["predictive_accuracy"].is_measurement is True
        assert m.performance["scoreboard_composite"].is_measurement is True
        assert m.performance["persistence_accuracy"].is_measurement is False
        assert m.performance["simulator_avg_confidence"].is_measurement is False
        assert m.performance["policy_nn_win_rate"].is_measurement is False
        assert m.performance["policy_nn_win_rate"].provenance == Provenance.SHADOW_ONLY

    def test_no_measurement_leak_anywhere(self):
        m = SelfViewSynthesizer().synthesize(_full_sources(), now=1.0)
        for f in _all_facts(m):
            if f.provenance not in MEASUREMENT_LEVELS:
                assert f.is_measurement is False

    def test_dormant_renders_dormant_not_available(self):
        m = SelfViewSynthesizer().synthesize(_full_sources(), now=1.0)
        cp = m.maturity["cognitive_planner"]
        assert cp.value == "dormant"
        assert cp.provenance == Provenance.DORMANT
        assert cp.is_measurement is False
        assert m.maturity["counterfactual"].provenance == Provenance.DORMANT

    def test_gaps_are_explicit(self):
        m = SelfViewSynthesizer().synthesize(_full_sources(), now=1.0)
        areas = {g["area"] for g in m.gaps}
        assert "scoreboard:memory_integrity" in areas
        assert "scoreboard:safety_immunity" in areas
        assert m.coverage["gap_count"] == len(m.gaps)


# ---------------------------------------------------------------------------
# Synthesizer: read-only + degrade-not-fake
# ---------------------------------------------------------------------------

class TestReadOnlyAndDegradation:
    def test_input_not_mutated(self):
        src = _full_sources()
        before = copy.deepcopy(src)
        SelfViewSynthesizer().synthesize(src, now=1.0)
        assert src == before  # pure read — no mutation of canonical-ish input

    def test_empty_sources_degrade_to_gaps_not_fakes(self):
        m = SelfViewSynthesizer().synthesize({}, now=1.0)
        # nothing fabricated: every performance fact is absent with value None
        for f in m.performance.values():
            assert isinstance(f, Fact)
            assert f.is_absent
            assert f.value is None
        assert m.performance == {} or all(f.is_absent for f in m.performance.values())
        assert m.coverage["measured_performance_facts"] == 0
        assert len(m.gaps) >= 1
        # structural + belief + change degrade to gaps, not invented content
        assert m.structural["capabilities"].provenance == Provenance.GAP
        assert m.belief["self_beliefs"].provenance == Provenance.GAP
        assert m.change["recent"].provenance == Provenance.GAP

    def test_synthesize_writes_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sv, "STATE_PATH", str(tmp_path / "sv.json"))
        SelfViewSynthesizer().synthesize(_full_sources(), now=1.0)
        assert not (tmp_path / "sv.json").exists()  # synthesize never persists


# ---------------------------------------------------------------------------
# Schema stability + serialization
# ---------------------------------------------------------------------------

class TestSchema:
    def test_stable_top_level_schema(self):
        d = SelfViewSynthesizer().synthesize(_full_sources(), now=1.0).to_dict()
        assert set(d.keys()) == {
            "schema_version", "generated_at", "structural", "performance",
            "maturity", "belief", "change", "subsystems", "gaps", "coverage",
        }
        assert d["schema_version"] == SCHEMA_VERSION

    def test_every_serialized_fact_has_provenance_and_is_measurement(self):
        d = SelfViewSynthesizer().synthesize(_full_sources(), now=1.0).to_dict()
        for dim in ("structural", "performance", "maturity", "change"):
            for v in d[dim].values():
                facts = v if isinstance(v, list) else [v]
                for f in facts:
                    if isinstance(f, dict) and "value" in f:
                        assert "provenance" in f
                        assert "is_measurement" in f


# ---------------------------------------------------------------------------
# Live build path (engine=None) degrades gracefully + persists to OSV file only
# ---------------------------------------------------------------------------

class TestBuildAndPersist:
    def test_build_with_no_engine_degrades(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sv, "STATE_PATH", str(tmp_path / "sv.json"))
        src = _full_sources()
        model = sv.build_self_view(
            engine=None,
            eval_snapshot={"scoreboard": src["scoreboard"]},
            skills_summary=src["skills"],
            now=1.0,
        )
        # scoreboard + skills present; world-model sources (engine=None) -> gaps
        assert model["performance"]["scoreboard_composite"]["is_measurement"] is True
        assert model["performance"]["predictive_accuracy"]["provenance"] == Provenance.GAP

    def test_persist_roundtrip_writes_only_osv_file(self, tmp_path, monkeypatch):
        p = tmp_path / "sv.json"
        monkeypatch.setattr(sv, "STATE_PATH", str(p))
        model = SelfViewSynthesizer().synthesize(_full_sources(), now=1.0).to_dict()
        sv.save_self_view(model)
        assert p.exists()
        loaded = sv.load_self_view()
        assert loaded["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# P0.5: subsystem inventory sourced from the build_cache snapshot
# ---------------------------------------------------------------------------

class TestSubsystemInventory:
    def _model(self):
        return sv.build_self_view(engine=None, eval_snapshot={}, skills_summary={},
                                  snapshot=_fake_snapshot(), now=1.0)

    def test_consciousness_is_self_scored_not_measurement(self):
        sub = self._model()["subsystems"]["consciousness"]
        assert sub["provenance"] == Provenance.SELF_SCORED
        assert sub["is_measurement"] is False
        assert "awareness_level" in sub["value"]  # self-reported value carried, not as proof

    def test_policy_is_shadow_only(self):
        sub = self._model()["subsystems"]["policy"]
        assert sub["provenance"] == Provenance.SHADOW_ONLY
        assert sub["is_measurement"] is False
        assert "nn_win_rate" in sub["note"]

    def test_promotion_levels_render_by_real_level(self):
        subs = self._model()["subsystems"]
        assert subs["world_model"]["value"] == "active"
        assert subs["world_model"]["provenance"] == Provenance.MEASURED
        assert subs["simulator"]["value"] == "shadow"
        assert subs["simulator"]["provenance"] == Provenance.SHADOW_ONLY

    def test_self_improve_active_gated(self):
        sub = self._model()["subsystems"]["self_improve"]
        assert sub["value"].startswith("active")
        assert sub["provenance"] == Provenance.MEASURED

    def test_unknown_shape_degrades_to_present_not_fake(self):
        sub = self._model()["subsystems"]["memory"]
        assert sub["value"] == "present"
        assert sub["provenance"] == Provenance.UNKNOWN

    def test_curated_empty_subsystem_is_gap(self):
        sub = self._model()["subsystems"]["quarantine"]  # curated + empty -> gap
        assert sub["provenance"] == Provenance.GAP

    def test_noncurated_empty_subsystem_omitted(self):
        subs = self._model()["subsystems"]
        assert "empty_thing" not in subs  # non-curated + empty -> not tracked

    def test_meta_keys_skipped(self):
        subs = self._model()["subsystems"]
        assert "summary" not in subs
        assert "_internal" not in subs

    def test_no_measurement_leak_in_subsystems(self):
        subs = self._model()["subsystems"]
        for name, f in subs.items():
            if isinstance(f, dict) and f.get("provenance") not in (Provenance.MEASURED,):
                assert f.get("is_measurement") is False

    def test_coverage_tallies_subsystems(self):
        cov = self._model()["coverage"]
        assert cov["subsystem_count"] >= 6
        bp = cov["subsystems_by_provenance"]
        assert bp.get(Provenance.SELF_SCORED, 0) >= 1
        assert bp.get(Provenance.SHADOW_ONLY, 0) >= 1

    def test_no_snapshot_degrades_to_gap(self):
        m = sv.build_self_view(engine=None, eval_snapshot={}, skills_summary={},
                               snapshot=None, now=1.0)
        assert "_meta" in m["subsystems"]
        assert m["subsystems"]["_meta"]["provenance"] == Provenance.GAP
