"""Companion Cognition P1 — theory-of-mind: the LEARNED verbosity/brevity axis.

The brevity axis is a 2nd disposition the person-model EARNS from lived reads (a
within-person contrast: does this person react worse when JARVIS over-explains, vs
their own baseline?). It must stay forming below the sample floor, earn confidence
only on real contrast, and never assert a fact. This is what uncages the be_concise
advisory's person-awareness for a steady/engaged companion.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import consciousness.theory_of_mind as tom


def _read(*, engagement="engaged", sentiment="neutral", confidence=0.6, overexplain=False):
    sc = ("very long reply — watch for overexplaining" if overexplain
          else "reply length proportionate")
    return SimpleNamespace(engagement=engagement, user_sentiment=sentiment,
                           confidence=confidence, self_check=sc)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(tom, "_TOM_STATE_PATH", str(tmp_path / "tom.json"))
    return tom.TheoryOfMindEngine()


class TestVerbosityAxis:
    def test_forming_below_overexplain_sample_floor(self, engine):
        pm = None
        for _ in range(3):  # < _VERBOSITY_MIN_OE_SAMPLES (4)
            pm = engine.observe("david", _read(overexplain=True,
                                               engagement="disengaging", sentiment="negative"))
        assert pm.verbosity_pref.startswith("forming")
        assert pm.verbosity_confidence == 0.0

    def test_earns_prefers_concise_when_overexplain_reacts_worse(self, engine):
        # baseline good (engaged/positive); worse (disengaging/negative) when over-explained
        for _ in range(8):
            engine.observe("david", _read(engagement="engaged", sentiment="positive"))
        pm = None
        for _ in range(6):
            pm = engine.observe("david", _read(overexplain=True,
                                               engagement="disengaging", sentiment="negative"))
        assert pm.verbosity_pref == "prefers concise replies"
        assert pm.verbosity_confidence >= 0.30  # earns above the corroboration floor

    def test_no_preference_when_overexplain_reacts_the_same(self, engine):
        for _ in range(8):
            engine.observe("david", _read(engagement="engaged", sentiment="positive"))
        pm = None
        for _ in range(6):  # over-explained but SAME good reaction -> no concision pref
            pm = engine.observe("david", _read(overexplain=True,
                                               engagement="engaged", sentiment="positive"))
        assert pm.verbosity_pref == "no clear length preference"
        assert pm.verbosity_confidence == 0.0

    def test_engagement_axis_still_works(self, engine):
        pm = None
        for _ in range(6):
            pm = engine.observe("david", _read(engagement="engaged", sentiment="positive"))
        assert "engaged" in pm.disposition
        assert pm.disposition_confidence > 0.0

    def test_back_compat_load_without_verbosity_fields(self, tmp_path, monkeypatch):
        old = {"observations": 5, "people": {"david": {
            "name": "david", "sample_count": 5, "disposition": "tends to be engaged",
            "disposition_confidence": 0.5, "current_feeling": "positive",
            "feeling_confidence": 0.6, "responsiveness": "responsive",
            "consistency": 0.8, "last_updated": 1.0,
            "recent": [["engaged", "positive"]],
        }}}
        p = tmp_path / "tom_old.json"
        p.write_text(json.dumps(old))
        monkeypatch.setattr(tom, "_TOM_STATE_PATH", str(p))
        e2 = tom.TheoryOfMindEngine()           # must load old format without crashing
        m = e2.get_model("david")
        assert m is not None
        assert m["verbosity_pref"].startswith("forming")   # defaulted
        assert m["verbosity_confidence"] == 0.0

    def test_hypothesis_only_authority(self, engine):
        pm = engine.observe("david", _read())
        d = engine.get_model("david")
        assert d["hypothesis"] is True
        assert d["authority"] == "shadow_logged_only"
