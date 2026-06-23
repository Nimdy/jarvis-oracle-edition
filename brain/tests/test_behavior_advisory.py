"""Companion Cognition P3 — read->behavior ADVISORY (shadow / narrate-only)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import consciousness.behavior_advisory as ba


def _read(*, tripped=True, sentiment="negative", engagement="disengaging",
          self_check="reply length proportionate", confidence=0.6,
          evidence=None, salience=0.7, speaker="david"):
    return SimpleNamespace(
        salience_tripped=tripped, user_sentiment=sentiment, engagement=engagement,
        self_check=self_check, confidence=confidence, evidence=evidence or [],
        salience=salience, speaker=speaker,
    )


def _pm(*, disposition="", disp_conf=0.0, responsiveness="",
        verbosity_pref="forming — not enough reads yet", verbosity_confidence=0.0):
    return SimpleNamespace(
        disposition=disposition, disposition_confidence=disp_conf,
        responsiveness=responsiveness,
        verbosity_pref=verbosity_pref, verbosity_confidence=verbosity_confidence,
    )


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(ba, "_STATE_PATH", str(tmp_path / "adv.json"))
    return ba.BehaviorAdvisoryEngine()


class TestBehaviorAdvisory:
    def test_no_advisory_when_salience_not_tripped(self, engine):
        # anti-chatterbox spine: nothing notable -> no advisory at all
        assert engine.propose(_read(tripped=False), _pm()) is None
        assert engine._total == 0

    def test_emits_suggestions_when_tripped(self, engine):
        adv = engine.propose(_read(), _pm())
        assert adv is not None
        assert adv.applied is False          # P3 never applies
        assert adv.primary is not None
        assert any(s["adjustment"] == "give_space" for s in adv.suggestions)

    def test_never_applies_anything(self, engine):
        adv = engine.propose(_read(sentiment="negative", engagement="neutral"), _pm())
        assert adv.applied is False
        assert all("apply" not in s for s in adv.suggestions)  # no apply flag exists

    def test_person_model_corroboration_raises_confidence_and_flags(self, engine):
        # Same live read; with a learned "tends to pull back" disposition the
        # disengagement suggestion becomes person_aware and gains confidence.
        bare = engine.propose(_read(), _pm())
        bare_gs = next(s for s in bare.suggestions if s["adjustment"] == "give_space")
        assert bare_gs["person_aware"] is False

        aware = engine.propose(
            _read(),
            _pm(disposition="tends to pull back / disengage", disp_conf=0.6),
        )
        aware_gs = next(s for s in aware.suggestions if s["adjustment"] == "give_space")
        assert aware_gs["person_aware"] is True
        assert aware_gs["confidence"] > bare_gs["confidence"]
        assert "learned pattern" in aware_gs["reason"]

    def test_weak_disposition_does_not_corroborate(self, engine):
        # Below the disposition-confidence floor -> not person-aware (no fake boost).
        adv = engine.propose(
            _read(),
            _pm(disposition="tends to pull back / disengage", disp_conf=0.1),
        )
        gs = next(s for s in adv.suggestions if s["adjustment"] == "give_space")
        assert gs["person_aware"] is False

    def test_overexplain_and_delay_signals(self, engine):
        adv = engine.propose(
            _read(sentiment="neutral", engagement="neutral",
                  self_check="may be overexplaining (long reply to a simple turn)",
                  evidence=[["latency_ms", 9000]]),
            _pm(),
        )
        kinds = {s["adjustment"] for s in adv.suggestions}
        assert "be_concise" in kinds
        assert "acknowledge_delay" in kinds

    def test_promotion_readiness_is_structural_only(self, engine):
        pr = engine.get_status()["promotion_readiness"]
        assert pr["gate"] == "P3->P4"
        assert pr["structural_ready"] is False          # fresh -> not enough advisories
        assert "EARNED" in pr["note"]

    def test_promotion_readiness_requires_person_aware(self, engine):
        # Enough advisories but ZERO person-aware (the live 0/155 case) -> still NOT ready. The raw
        # count must not light the gate; the learned person-model has to have actually informed them.
        engine._total = ba._P3P4_MIN_ADVISORIES + 120
        engine._person_aware = 0
        pr = engine.get_status()["promotion_readiness"]
        assert pr["structural_ready"] is False
        assert any("person-aware" in b for b in pr["blocking"])
        assert pr["min_person_aware_fraction"] == ba._P3P4_MIN_PERSON_AWARE_FRACTION

    # ── brevity axis (uncages corroboration for a steady/engaged companion) ──
    def _oe_read(self):
        # over-explain on a steady/engaged turn — the case that was previously un-corroborable
        return _read(sentiment="neutral", engagement="engaged",
                     self_check="very long reply — watch for overexplaining")

    def test_be_concise_person_aware_when_brevity_axis_earned(self, engine):
        bare = engine.propose(self._oe_read(), _pm())
        bare_bc = next(s for s in bare.suggestions if s["adjustment"] == "be_concise")
        assert bare_bc["person_aware"] is False          # no learned brevity pref yet

        aware = engine.propose(
            self._oe_read(),
            _pm(verbosity_pref="prefers concise replies", verbosity_confidence=0.6),
        )
        aware_bc = next(s for s in aware.suggestions if s["adjustment"] == "be_concise")
        assert aware_bc["person_aware"] is True
        assert aware_bc["confidence"] > bare_bc["confidence"]
        assert "learned pattern" in aware_bc["reason"]

    def test_be_concise_weak_brevity_axis_does_not_corroborate(self, engine):
        adv = engine.propose(
            self._oe_read(),
            _pm(verbosity_pref="prefers concise replies", verbosity_confidence=0.1),
        )
        bc = next(s for s in adv.suggestions if s["adjustment"] == "be_concise")
        assert bc["person_aware"] is False               # below the 0.30 floor -> no fake boost

    def test_be_concise_ignores_unrelated_brevity_label(self, engine):
        # "no clear length preference" must NOT corroborate even at high confidence
        adv = engine.propose(
            self._oe_read(),
            _pm(verbosity_pref="no clear length preference", verbosity_confidence=0.9),
        )
        bc = next(s for s in adv.suggestions if s["adjustment"] == "be_concise")
        assert bc["person_aware"] is False
