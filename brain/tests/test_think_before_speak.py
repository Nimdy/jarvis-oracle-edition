"""Tests for Think-Before-Speak TBS-0 (shadow pre-speech read) — glass-box, injects nothing."""

import consciousness.think_before_speak as tbs


def _reset(tmp_path, monkeypatch):
    monkeypatch.setattr(tbs, "_SHADOW_LOG", str(tmp_path / "pss.jsonl"))
    tbs.PreSpeechReader.reset_instance()


def test_shadow_injects_nothing(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    st = tbs.PreSpeechReader.get_instance().get_status()
    assert st["injects_prompt"] is False          # glass-box auditability: structurally False
    assert st["shapes_reply"] is False
    assert st["authority"] == "shadow_observe_only"
    assert st["phase"] == "TBS-0_shadow_observe"


def test_lean_concise_from_learned_pref(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    s = tbs.PreSpeechReader.get_instance().read_before_speak(
        speaker="David", user_text="how does X work",
        person_model={"verbosity_pref": "prefers concise replies", "verbosity_confidence": 0.6})
    assert s.stance == "lean_concise"
    assert s.would_inject and "concise" in s.would_inject
    assert s.injected is False                    # never injected in TBS-0


def test_give_space_on_negative(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    s = tbs.PreSpeechReader.get_instance().read_before_speak(
        speaker="David", user_text="ugh", user_emotion="frustrated", person_model={})
    assert s.stance == "give_space"               # distress takes priority


def test_match_warmth(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    s = tbs.PreSpeechReader.get_instance().read_before_speak(
        speaker="David", user_text="haha nice", user_emotion="happy",
        person_model={"humor_reception": "lands well", "humor_confidence": 0.5})
    assert s.stance == "match_warmth"


def test_default_none(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    s = tbs.PreSpeechReader.get_instance().read_before_speak(
        speaker="David", user_text="hello", person_model={})
    assert s.stance == "none" and s.would_inject is None


def test_unearned_pref_does_not_trip(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    # verbosity preference below the confidence floor must NOT produce a stance (not yet earned)
    s = tbs.PreSpeechReader.get_instance().read_before_speak(
        speaker="David", user_text="x",
        person_model={"verbosity_pref": "prefers concise replies", "verbosity_confidence": 0.1})
    assert s.stance == "none"


def test_glassbox_persists_across_restart(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    r = tbs.PreSpeechReader.get_instance()
    for _ in range(3):
        r.read_before_speak(speaker="David", user_text="x", person_model={})
    assert r.get_status()["total_reads"] == 3
    tbs.PreSpeechReader.reset_instance()
    r2 = tbs.PreSpeechReader.get_instance()        # reloads from the durable log
    assert r2.get_status()["total_reads"] == 3     # glass box survives reboot
