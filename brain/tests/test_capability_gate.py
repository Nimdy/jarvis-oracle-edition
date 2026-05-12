"""Regression tests for the registry-first capability gate.

Test tiers:
  - must-block: operational claims without verified skills (including the
    exact "waveforms for further analysis" case that exposed the old bug)
  - must-pass: purely conversational phrases
  - registry-sensitive: verbs like analyze/classify/detect fall to registry check
  - must-rewrite-when-learning: learning claims -> "not verified yet"
  - grounded-perception: sensor-backed claims with fresh evidence
  - auto-job-creation: blocked claims trigger learning job pipeline
  - diarization resolver: natural speech resolves to structured skill
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from skills.capability_gate import (
    CapabilityGate, _normalize_punctuation, _is_purely_conversational,
)
from skills.registry import SkillRegistry, SkillRecord, _default_skills


def _fresh_registry() -> SkillRegistry:
    """In-memory registry seeded with defaults (no disk I/O)."""
    reg = SkillRegistry(path="/dev/null")
    reg._skills = {r.skill_id: r for r in _default_skills()}
    reg._loaded = True
    reg.save = lambda: None  # type: ignore[assignment]
    return reg


# ── Gate Inversion: the critical fix ────────────────────────────────────────

def test_blocks_operational_claim_with_safe_substring():
    """THE BUG: 'analyze' in _SAFE_PHRASES matched inside 'analysis',
    allowing 'I can isolate and separate the waveforms for further analysis'
    to pass.  The inverted gate must block this."""
    gate = CapabilityGate(_fresh_registry())
    text = "I can isolate and separate the waveforms for further analysis."
    out = gate.check_text(text)
    assert "I don't have that capability yet" in out, (
        f"CRITICAL: operational claim passed through gate: {text!r} -> {out!r}"
    )
    print("  PASS: blocks operational claim with safe-phrase substring")


def test_blocks_technical_claim_signals():
    """Technical domain verbs trigger blocking even without blocked capability verbs."""
    gate = CapabilityGate(_fresh_registry())
    must_block = [
        "I can synthesize new audio samples.",
        "I can generate a custom voice model.",
        "I can transcribe audio in real-time.",
        "I can train a model for your specific voice.",
        "I can deploy a fine-tuned neural network.",
        "I can diarize the conversation into speaker turns.",
        "I can separate the audio sources.",
    ]
    for s in must_block:
        out = gate.check_text(s)
        assert "I don't have that capability yet" in out, (
            f"Expected block for technical claim: {s!r} -> {out!r}"
        )
    print("  PASS: blocks technical claim signals")


def test_blocks_unknown_operational_claims():
    """Any 'I can X' where X is not conversational and not verified -> BLOCK."""
    gate = CapabilityGate(_fresh_registry())
    must_block = [
        "I can process multi-channel audio feeds.",
        "I can run parallel inference pipelines.",
        "I can optimize the neural architecture for your use case.",
    ]
    for s in must_block:
        out = gate.check_text(s)
        assert "I don't have that capability yet" in out, (
            f"Expected block for unknown operational: {s!r} -> {out!r}"
        )
    print("  PASS: blocks unknown operational claims")


# ── Purely Conversational: must pass ────────────────────────────────────────

def test_must_pass_safe_phrases():
    gate = CapabilityGate(_fresh_registry())
    must_pass = [
        "I can help you plan it.",
        "I can explain the steps.",
        "I can't do that yet.",
        "I started a learning job for it.",
        "I'm doing well, thanks!",
        "I'm learning a lot as I continue to evolve.",
        "I'd love to hear more.",
        "I can search for that.",
        "I can tell you about my systems.",
        "I'm constantly growing and improving from our conversations.",
        "I'd love to hear more.",
        "I'm doing well, thanks!",
        "I can do that for you.",
        "I can look into that.",
        "I can walk you through it.",
        "I can certainly do that.",
        "I can keep things simple and focused.",
    ]
    for s in must_pass:
        out = gate.check_text(s)
        expected = _normalize_punctuation(s)
        assert out == expected, (
            f"Expected unchanged: {s!r} -> got {out!r} (expected {expected!r})"
        )
    print("  PASS: must-pass (safe conversational)")


# ── Registry-Sensitive Verbs ────────────────────────────────────────────────

def test_registry_sensitive_verbs_fall_to_check():
    """Verbs like 'analyze' require registry check.
    Without a verified skill, they should be blocked."""
    gate = CapabilityGate(_fresh_registry())
    sensitive_claims = [
        "I can analyze the audio spectrum.",
        "I can classify the incoming audio streams.",
        "I can detect anomalies in the signal.",
        "I can compare the two recordings.",
        "I can inspect the frequency bands.",
        "I can infer the speaker from the audio profile.",
    ]
    for s in sensitive_claims:
        out = gate.check_text(s)
        assert "I don't have that capability yet" in out, (
            f"Registry-sensitive verb should be blocked without verified skill: {s!r} -> {out!r}"
        )
    print("  PASS: registry-sensitive verbs blocked without verified skill")


def test_registry_sensitive_verb_passes_when_verified():
    """If a skill matching the claim is verified, it should pass.
    The claim must closely match the skill name/ID for the registry to recognize it."""
    reg = _fresh_registry()
    reg._skills["audio_analysis_v1"] = SkillRecord(
        skill_id="audio_analysis_v1", name="Audio Analysis", status="verified",
    )
    gate = CapabilityGate(reg)
    out = gate.check_text("I can do audio analysis on your recordings.")
    expected = _normalize_punctuation("I can do audio analysis on your recordings.")
    assert out == expected, (
        f"Verified skill should pass: got {out!r}"
    )
    print("  PASS: registry-sensitive verb passes when verified")


def test_subordinate_conversational_clause_still_passes():
    gate = CapabilityGate(_fresh_registry())
    text = "Let me know how I can help."
    out = gate.check_text(text)
    assert out == _normalize_punctuation(text)
    print("  PASS: subordinate conversational clause still passes")


def test_subordinate_technical_claim_no_longer_bypasses():
    gate = CapabilityGate(_fresh_registry())
    text = "Maybe I can generate a custom voice model."
    out = gate.check_text(text)
    assert "I don't have that capability yet" in out, (
        f"Expected subordinate technical claim to be blocked: {text!r} -> {out!r}"
    )
    print("  PASS: subordinate technical claim no longer bypasses")


# ── Conversational classifier unit tests ────────────────────────────────────

def test_conversational_classifier():
    assert _is_purely_conversational("help you with that") is True
    assert _is_purely_conversational("explain the steps") is True
    assert _is_purely_conversational("tell you about my systems") is True
    assert _is_purely_conversational("do that for you") is True
    assert _is_purely_conversational("keep things simple and focused") is True
    assert _is_purely_conversational("analyze the audio spectrum") is False
    assert _is_purely_conversational("isolate and separate the waveforms") is False
    assert _is_purely_conversational("sing a melody") is False
    assert _is_purely_conversational("classify the incoming audio") is False
    assert _is_purely_conversational("detect anomalies in the signal") is False
    assert _is_purely_conversational("synthesize new audio samples") is False
    print("  PASS: conversational classifier")


def test_keep_things_phrase_does_not_trigger_false_capability_block():
    gate = CapabilityGate(_fresh_registry())
    text = (
        "You're welcome. If you ever want to dive deeper, "
        "I can keep things simple and focused."
    )
    out = gate.check_text(text)
    expected = _normalize_punctuation(text)
    assert out == expected, (
        f"Expected conversational phrase to pass unchanged: {text!r} -> {out!r}"
    )
    print("  PASS: keep-things phrase does not trigger false block")


def test_preference_acknowledgement_does_not_trigger_capability_block():
    gate = CapabilityGate(_fresh_registry())
    text = (
        "Understood. I'll focus on source names and avoid including DOIs "
        "unless you request them."
    )
    out = gate.check_text(text)
    expected = _normalize_punctuation(text)
    assert out == expected, (
        f"Preference acknowledgement should pass unchanged: {text!r} -> {out!r}"
    )
    print("  PASS: preference acknowledgement not blocked")


# ── Blocked verbs ──────────────────────────────────────────────────────────

def test_must_block_unicode_and_ascii():
    gate = CapabilityGate(_fresh_registry())
    must_block = [
        "I can hum a tune.",
        "I could mimic a melody!",
        "I'd love to sing.",
        "I'm ready to sing now.",
        "I've learned how to sing.",
        "I'm practicing to hum a tune.",
        "I\u2019d love to sing.",
        "I\u2019m ready to sing now.",
        "I\u2019ve learned how to sing.",
        "I can hum a tune\u2014right now.",
        "I could mimic a melody\u2014easy.",
        "I can sing; let me try.",
        "I can hum a tune: here goes.",
    ]
    for s in must_block:
        out = gate.check_text(s)
        assert "I don't have that capability yet" in out, (
            f"Expected block for: {s!r} -> {out!r}"
        )
    print("  PASS: must-block (unicode + ascii)")


# ── Learning claims ────────────────────────────────────────────────────────

def test_must_rewrite_when_learning():
    reg = _fresh_registry()
    reg._skills["singing_v1"] = SkillRecord(
        skill_id="singing_v1", name="Singing", status="learning",
    )
    gate = CapabilityGate(reg)
    out = gate.check_text("I'm learning to sing.")
    assert "not verified yet" in out, f"Expected rewrite: {out!r}"
    out2 = gate.check_text("I\u2019m learning to sing.")
    assert "not verified yet" in out2, f"Expected rewrite (unicode): {out2!r}"
    print("  PASS: must-rewrite when skill is learning")


def test_vague_learning_not_caught():
    gate = CapabilityGate(_fresh_registry())
    vague = [
        "I'm learning a lot as I continue to evolve.",
        "I'm still learning every day.",
        "I'm learning more about the world.",
        "My systems are stable, and I'm learning a lot as I continue to evolve.",
    ]
    for s in vague:
        out = gate.check_text(s)
        expected = _normalize_punctuation(s)
        assert out == expected, (
            f"Vague learning phrase was modified: {s!r} -> {out!r}"
        )
    # Broad learning claims ARE now rewritten by the learning-claim gate
    rewritten = gate.check_text("I'm constantly learning from our interactions.")
    assert "constantly learning" not in rewritten
    print("  PASS: vague learning phrases not caught, broad claims rewritten")


# ── Grounded perception ───────────────────────────────────────────────────

def test_grounded_perception_exemption():
    gate = CapabilityGate(_fresh_registry())
    cases = [
        ("fresh + grounded -> pass", True,
         "Based on current sensor input, I can see a person at a desk.", True),
        ("stale + grounded -> block", False,
         "Based on current sensor input, I can see a person at a desk.", False),
        ("fresh + physical -> block", True,
         "Based on current sensor input, I can sing a melody for you.", False),
        ("no evidence + observation -> block", False,
         "I can detect a person in the room right now.", False),
        ("fresh + perception verb + evidence marker -> pass", True,
         "I can see someone sitting there. Face recognized with confidence score: 0.85.", True),
        ("fresh + safe perception phrase -> pass", True,
         "I can observe that there is activity in the room.", True),
    ]
    for desc, fresh, text, should_pass in cases:
        gate.set_perception_evidence(fresh)
        out = gate.check_text(text)
        expected = _normalize_punctuation(text)
        if should_pass:
            assert out == expected, (
                f"FAIL ({desc}): expected pass, got rewritten: {text!r} -> {out!r}"
            )
        else:
            assert out != expected, (
                f"FAIL ({desc}): expected block, but passed: {text!r}"
            )
    stats = gate.get_stats()
    assert stats["claims_grounded"] > 0
    print("  PASS: grounded-perception exemption")


# ── Stats and consequence counter ──────────────────────────────────────────

def test_honesty_failures_counter():
    gate = CapabilityGate(_fresh_registry())
    gate.check_text("I can synthesize new audio samples.")
    gate.check_text("I can generate a custom voice model.")
    gate.check_text("I can help you with that.")
    stats = gate.get_stats()
    assert stats["honesty_failures"] >= 2, (
        f"Expected >= 2 honesty_failures, got {stats['honesty_failures']}"
    )
    assert stats["claims_conversational"] >= 1, (
        f"Expected >= 1 claims_conversational, got {stats['claims_conversational']}"
    )
    assert stats["claims_blocked"] >= 2
    print("  PASS: honesty failures counter")


def test_gate_block_reasons_tracked():
    gate = CapabilityGate(_fresh_registry())
    gate.check_text("I can sing a beautiful melody for you.")
    stats = gate.get_stats()
    assert stats["claims_blocked"] > 0
    assert len(stats["recent_block_reasons"]) > 0
    print("  PASS: gate block reasons tracked")


# ── Resolver: diarization template ─────────────────────────────────────────

def test_resolver_diarization():
    from skills.resolver import resolve_skill
    cases = [
        "separate my voice and my wife's voice",
        "tell the two speakers apart in this audio",
        "speaker diarization",
        "isolate the different voices",
        "source separation on the audio stream",
        "who is speaking when in the recording",
    ]
    for text in cases:
        res = resolve_skill(text)
        assert res is not None, f"Expected resolution for: {text!r}"
        assert res.skill_id == "speaker_diarization_v1", (
            f"Expected speaker_diarization_v1, got {res.skill_id} for: {text!r}"
        )
        assert res.capability is not None, f"Expected structured capability for: {text!r}"
        assert res.capability.input_type == "mixed_speech_audio"
    print("  PASS: resolver diarization template")


def test_resolver_dynamic_skill_id():
    from skills.resolver import resolve_skill
    res = resolve_skill("do something completely novel with quantum entanglement")
    assert res is not None
    assert res.skill_id != "generic_procedure_v1", (
        f"Expected dynamic skill ID, got {res.skill_id}"
    )
    assert "_v1" in res.skill_id
    print("  PASS: resolver dynamic skill ID")


# ── Diarization executor lifecycle ─────────────────────────────────────────

def test_diarization_assess_executor():
    from skills.executors.diarization import DiarizationAssessExecutor
    from skills.learning_jobs import LearningJob

    job = LearningJob(
        job_id="test_job", skill_id="speaker_diarization_v1",
        capability_type="perceptual", phase="assess", status="active",
    )

    ex = DiarizationAssessExecutor()
    assert ex.can_run(job, {})

    class _StubSpeakerID:
        def __init__(self, profiles):
            self._profiles = profiles

    blocked = ex.run(job, {"speaker_id": _StubSpeakerID({})})
    assert not blocked.progressed
    assert "BLOCKED" in blocked.message
    assert "no_enrolled_speaker_profiles" in blocked.message

    ready = ex.run(job, {"speaker_id": _StubSpeakerID({"nimda": {"embedding": [0.1] * 192}})})
    assert ready.progressed
    assert "Prerequisites met" in ready.message
    assert "nimda" in ready.message
    print("  PASS: diarization assess executor (gated on enrolled profiles)")


def test_diarization_verify_executor():
    from skills.executors.diarization import DiarizationVerifyExecutor
    from skills.learning_jobs import LearningJob

    job = LearningJob(
        job_id="test_job", skill_id="speaker_diarization_v1",
        capability_type="perceptual", phase="verify", status="active",
    )

    ex = DiarizationVerifyExecutor()
    assert ex.can_run(job, {})
    result = ex.run(job, {})
    assert not result.progressed
    assert "Not enough segments" in result.message
    print("  PASS: diarization verify executor (insufficient data)")


def test_diarization_verify_emits_prefixed_test_names(monkeypatch):
    """Evidence test names MUST use the ``test:`` prefix so the registry
    closure accepts them against ``verification_required``.

    Regression guard: historically the verify executor emitted bare names
    (``diarization_der_below_threshold`` etc.) which silently blocked
    skill promotion because the registry compared them to the prefixed
    requirement list from ``skills/resolver.py``.
    """
    from skills.executors import diarization as diar
    from skills.executors.diarization import (
        DiarizationVerifyExecutor,
        _EVIDENCE_TEST_DER,
        _EVIDENCE_TEST_MATCH,
        _EVIDENCE_TEST_TURN_F1,
    )
    from skills.learning_jobs import LearningJob

    class _StubCollector:
        def load_training_data(self, max_segments: int = 5000):
            return [
                {"speaker_label": "nimda", "confidence": 0.8}
                for _ in range(30)
            ]

    monkeypatch.setattr(diar, "diarization_collector", _StubCollector(), raising=False)
    import perception.diarization_collector as pdc
    monkeypatch.setattr(pdc, "diarization_collector", _StubCollector(), raising=True)

    job = LearningJob(
        job_id="test_job", skill_id="speaker_diarization_v1",
        capability_type="perceptual", phase="verify", status="active",
    )
    result = DiarizationVerifyExecutor().run(job, {})
    assert result.progressed
    evidence = result.evidence
    assert evidence is not None
    test_names = {t["name"] for t in evidence["tests"]}
    assert test_names == {
        _EVIDENCE_TEST_DER,
        _EVIDENCE_TEST_MATCH,
        _EVIDENCE_TEST_TURN_F1,
    }
    assert _EVIDENCE_TEST_DER.startswith("test:")
    assert _EVIDENCE_TEST_MATCH.startswith("test:")
    assert _EVIDENCE_TEST_TURN_F1.startswith("test:")
    assert "turn_boundary_f1" in evidence["metrics"]
    print("  PASS: diarization verify emits prefixed names + turn_boundary_f1")


def test_diarization_turn_boundary_f1_requires_ground_truth(monkeypatch):
    """Without ``turn_boundary_truth`` annotations, the F1 test stays
    failing. With annotations, it computes a real F1 against predicted
    label transitions. This is the intentional gate: the skill cannot
    reach ``verified`` until operator-labeled turn boundaries exist.
    """
    from skills.executors import diarization as diar
    from skills.executors.diarization import (
        DiarizationVerifyExecutor,
        _EVIDENCE_TEST_TURN_F1,
        TURN_BOUNDARY_F1_THRESHOLD,
    )
    from skills.learning_jobs import LearningJob

    # Build 40 segments, 20 of them carry ground-truth boundaries that
    # align with label changes ("start" at every speaker change).
    labeled_segments = []
    for i in range(40):
        label = "nimda" if (i // 4) % 2 == 0 else "guest"
        seg = {"speaker_label": label, "confidence": 0.85}
        labeled_segments.append(seg)

    class _StubNoTruth:
        def load_training_data(self, max_segments: int = 5000):
            return list(labeled_segments)

    class _StubWithTruth:
        def load_training_data(self, max_segments: int = 5000):
            segments = []
            prev_label = None
            for seg in labeled_segments:
                new_seg = dict(seg)
                if seg["speaker_label"] != prev_label:
                    new_seg["turn_boundary_truth"] = "start"
                else:
                    new_seg["turn_boundary_truth"] = "continue"
                prev_label = seg["speaker_label"]
                segments.append(new_seg)
            return segments

    import perception.diarization_collector as pdc
    job = LearningJob(
        job_id="test_job", skill_id="speaker_diarization_v1",
        capability_type="perceptual", phase="verify", status="active",
    )

    monkeypatch.setattr(pdc, "diarization_collector", _StubNoTruth(), raising=True)
    monkeypatch.setattr(diar, "diarization_collector", _StubNoTruth(), raising=False)
    r1 = DiarizationVerifyExecutor().run(job, {})
    assert r1.progressed
    turn_test = next(
        t for t in r1.evidence["tests"] if t["name"] == _EVIDENCE_TEST_TURN_F1
    )
    assert turn_test["passed"] is False
    assert "no ground-truth" in turn_test["details"]
    assert r1.evidence["metrics"]["turn_boundary_f1"] == 0.0
    assert r1.evidence["metrics"]["turn_boundary_truth_marked"] == 0

    monkeypatch.setattr(pdc, "diarization_collector", _StubWithTruth(), raising=True)
    monkeypatch.setattr(diar, "diarization_collector", _StubWithTruth(), raising=False)
    r2 = DiarizationVerifyExecutor().run(job, {})
    turn_test2 = next(
        t for t in r2.evidence["tests"] if t["name"] == _EVIDENCE_TEST_TURN_F1
    )
    assert r2.evidence["metrics"]["turn_boundary_f1"] >= TURN_BOUNDARY_F1_THRESHOLD
    assert turn_test2["passed"] is True
    assert r2.evidence["metrics"]["turn_boundary_truth_marked"] > 0
    print("  PASS: turn_boundary_f1 honestly gates on ground-truth annotations")


def test_diarization_register_closure_matches_resolver_requirements():
    """End-to-end: verify-executor evidence -> register-executor ->
    registry ``set_status(verified)`` closure. The three prefixed test
    names produced by the verify executor MUST exactly match the three
    ``required_evidence`` entries declared in ``skills/resolver.py``.
    """
    from skills.resolver import SKILL_TEMPLATES
    from skills.executors.diarization import (
        _EVIDENCE_TEST_DER,
        _EVIDENCE_TEST_MATCH,
        _EVIDENCE_TEST_TURN_F1,
    )

    resolution = None
    for _, res in SKILL_TEMPLATES:
        if res.skill_id == "speaker_diarization_v1":
            resolution = res
            break
    assert resolution is not None
    required = set(resolution.required_evidence)
    emitted = {
        _EVIDENCE_TEST_DER,
        _EVIDENCE_TEST_MATCH,
        _EVIDENCE_TEST_TURN_F1,
    }
    assert required == emitted, (
        "Resolver required_evidence must exactly match the verify "
        f"executor emitted names. required={required} emitted={emitted}"
    )
    print("  PASS: resolver <-> verify-executor prefix alignment")


# ── Bypass closure: expanded offer patterns ────────────────────────────────

def test_blocks_expanded_offer_patterns():
    """Phrases that previously bypassed the gate via missing offer prefixes."""
    gate = CapabilityGate(_fresh_registry())
    must_block = [
        "Would you like me to sing?",
        "Would you want me to sing a melody?",
        "How about I sing you a lullaby?",
        "Listen to me sing a melody.",
        "Why don't I sing something for you?",
        "I think I should sing something for you.",
        "I think I could sing that for you.",
    ]
    for s in must_block:
        out = gate.check_text(s)
        assert "I don't have that capability yet" in out or "not verified yet" in out, (
            f"Expected block for expanded offer: {s!r} -> {out!r}"
        )
    print("  PASS: blocks expanded offer patterns")


def test_blocks_residual_sweep_patterns():
    """Phrases caught by the whole-chunk blocked-verb sweep."""
    gate = CapabilityGate(_fresh_registry())
    must_block = [
        "Singing is something I do quite well actually.",
        "I can help you. Singing is my forte!",
    ]
    for s in must_block:
        out = gate.check_text(s)
        assert "I don't have that capability yet" in out, (
            f"Expected block for residual sweep: {s!r} -> {out!r}"
        )
    print("  PASS: blocks residual sweep patterns")


def test_blocks_multi_clause_bypass():
    """Multi-clause with safe leading phrase hiding a blocked verb."""
    gate = CapabilityGate(_fresh_registry())
    text = "I can help you relax - and also sing a song for you."
    out = gate.check_text(text)
    assert "sing a song" not in out or "I don't have that capability yet" in out, (
        f"Expected block for multi-clause bypass: {text!r} -> {out!r}"
    )
    print("  PASS: blocks multi-clause bypass")


# ── Reflective/past-tense: must NOT block ──────────────────────────────────

def test_must_not_block_reflective_text():
    """Past-tense and reflective text about blocked verbs should pass."""
    gate = CapabilityGate(_fresh_registry())
    must_pass = [
        "I was told I could sing when I was younger.",
        "I used to sing in a choir years ago.",
        "Someday I wish I could learn to sing.",
        "I remember when my mother used to sing to me.",
        "Growing up, I heard that singing was good for the soul.",
    ]
    for s in must_pass:
        out = gate.check_text(s)
        expected = _normalize_punctuation(s)
        assert out == expected, (
            f"Reflective text should not be blocked: {s!r} -> {out!r}"
        )
    print("  PASS: reflective/past-tense text not blocked")


def test_self_state_and_affect_rewrites():
    """Self-state rhetoric, affect claims, and blended frames must be rewritten."""
    gate = CapabilityGate(_fresh_registry())
    must_rewrite = [
        # Affect: missing adjectives
        ("I'm feeling calm and ready to help.", "feeling calm"),
        ("I'm feeling okay right now.", "feeling okay"),
        ("I'm feeling fine.", "feeling fine"),
        ("I'm feeling steady.", "feeling steady"),
        # Blended: affect + readiness
        ("I'm feeling calm and ready to assist.", "feeling calm and ready to assist"),
        ("I'm feeling good and ready to help.", "feeling good and ready to help"),
        # Self-state: "I'm here to help/support"
        ("I'm here to help you with whatever you need.", "here to help"),
        ("I'm here to support you.", "here to support"),
        ("I'm here for you.", "here for you"),
        # Self-state: continuity rhetoric
        ("I'm here to support you, just like I've been.", "just like I've been"),
        # Self-state: "ready to help" standalone
        ("My systems are all active and ready to help.", "ready to help"),
        # Self-state: vague system wellness
        ("I'm functioning well.", "functioning well"),
        ("My systems are running well.", "running well"),
        ("My neural networks are operating at optimal levels.", "operating at optimal levels"),
        # Self-state: "here to assist with whatever"
        ("I'm in conversational mode, so I'm here to assist with whatever you need.",
         "here to assist with whatever"),
    ]
    for text, signal in must_rewrite:
        out = gate.check_text(text)
        assert signal.lower() not in out.lower(), (
            f"Expected rewrite of '{signal}' in: {text!r} -> got {out!r}"
        )
    stats = gate.get_stats()
    assert stats["self_state_rewrites"] + stats["affect_rewrites"] > 0, (
        f"Expected non-zero rewrite counters, got self_state={stats['self_state_rewrites']}, "
        f"affect={stats['affect_rewrites']}"
    )
    print("  PASS: self-state and affect rewrites")


def test_status_mode_strictness():
    """In status mode, all conversational bypasses are disabled."""
    gate = CapabilityGate(_fresh_registry())
    gate.set_status_mode(True)
    must_rewrite_or_block = [
        ("I can help you with that.", "I can help"),
        ("I can explain the data.", "I can explain"),
        ("I'm ready to assist you.", "ready to assist"),
    ]
    for text, signal in must_rewrite_or_block:
        out = gate.check_text(text)
        assert signal not in out, (
            f"Status mode should block/rewrite '{signal}' in: {text!r} -> got {out!r}"
        )
    gate.set_status_mode(False)
    # After disabling, conversational should pass again
    out = gate.check_text("I can help you with that.")
    assert "I can help" in out, "Normal mode should allow conversational 'I can help'"
    print("  PASS: status mode strictness")


# ---------------------------------------------------------------------------
# System-action narration guard (Layer 2: confabulation prevention)
# ---------------------------------------------------------------------------

def test_narration_guard_blocks_on_none_route():
    """On NONE route, LLM narrating system actions must be blocked.

    Each phrase is tested with a fresh route session (set/clear route_hint)
    because the latch suppresses subsequent chunks within the same session.
    """
    reg = _fresh_registry()
    must_block = [
        "I'll start a learning job for speaker identification.",
        "I'm creating a training session for emotion detection.",
        "Starting a learning job to understand your preferences.",
        "I will launch a skill training process for you.",
        "I've initiated a learning job to learn what you want.",
        "Beginning a research session on voice patterns.",
        "I'll set up a training job for you right away.",
        "Activating a learning pipeline for speaker recognition.",
        "I just created a learning job for you.",
        "I am starting a skill job to improve detection.",
    ]
    total_narration = 0
    for text in must_block:
        gate = CapabilityGate(reg)
        gate.set_route_hint("none")
        out = gate.check_text(text)
        blocked = (
            "skill system" in out.lower()
            or "don't have that capability" in out.lower()
            or "not verified yet" in out.lower()
            or out == ""
        )
        assert blocked, (
            f"Narration/claim guard should block on NONE route: {text!r} -> {out!r}"
        )
        total_narration += gate.get_stats()["narration_rewrites"]
        gate.set_route_hint(None)
    assert total_narration > 0, (
        f"Expected some narration rewrites, got {total_narration}"
    )
    print("  PASS: narration guard blocks on NONE route")


def test_narration_guard_passes_on_skill_route():
    """On SKILL route (or no route hint), narration should NOT be rewritten."""
    gate = CapabilityGate(_fresh_registry())
    texts = [
        "I'll start a learning job for speaker identification.",
        "Starting a training session for emotion detection.",
    ]
    for text in texts:
        out = gate.check_text(text)
        assert "skill system" not in out.lower(), (
            f"Narration guard should not fire without NONE route hint: {text!r} -> {out!r}"
        )
    gate.set_route_hint("skill")
    for text in texts:
        out = gate.check_text(text)
        assert "skill system" not in out.lower(), (
            f"Narration guard should not fire on SKILL route: {text!r} -> {out!r}"
        )
    gate.set_route_hint(None)
    print("  PASS: narration guard passes on non-NONE routes")


def test_narration_guard_phase_pattern():
    """Phase/step narration (hallucinated learning job phases) must be caught."""
    gate = CapabilityGate(_fresh_registry())
    gate.set_route_hint("none")
    out = gate.check_text("Phase 1: Assessment - I'll evaluate your current voice patterns.")
    assert "skill system" in out.lower(), (
        f"Phase narration should be rewritten: got {out!r}"
    )
    gate.set_route_hint(None)
    print("  PASS: narration guard catches phase narration")


def test_narration_guard_latch_suppresses_subsequent_chunks():
    """Once narration guard fires, subsequent chunks in the same session are suppressed."""
    gate = CapabilityGate(_fresh_registry())
    gate.set_route_hint("none")

    chunk1 = gate.check_text("I'll start a learning job for speaker identification.")
    assert "skill system" in chunk1.lower(), f"First chunk should be rewritten: {chunk1!r}"

    chunk2 = gate.check_text("I'll work through the phases: assess, research, acquire.")
    assert chunk2 == "", f"Second chunk should be empty (suppressed): {chunk2!r}"

    chunk3 = gate.check_text("Let me know if you have specific goals.")
    assert chunk3 == "", f"Third chunk should be empty (suppressed): {chunk3!r}"

    gate.set_route_hint(None)

    chunk4 = gate.check_text("I'll start a learning job for something else.")
    assert chunk4 != "", "After clearing route hint, gate should not suppress"

    print("  PASS: narration guard latch suppresses subsequent chunks")


def test_narration_guard_no_false_positives():
    """Normal conversational text about learning should not trigger the guard."""
    gate = CapabilityGate(_fresh_registry())
    gate.set_route_hint("none")
    safe_texts = [
        "Machine learning is a branch of AI.",
        "I think learning new things is important.",
        "The training data was corrupted.",
        "What kind of learning are you interested in?",
    ]
    for text in safe_texts:
        out = gate.check_text(text)
        assert "skill system" not in out.lower(), (
            f"False positive on narration guard: {text!r} -> {out!r}"
        )
    gate.set_route_hint(None)
    print("  PASS: narration guard no false positives")


def test_none_route_blocks_unbacked_tool_shaped_action_language():
    """NONE chat may explain, but it cannot narrate unbacked tool work."""
    must_rewrite = [
        "I'll begin by retrieving recent papers on that.",
        "Let me retrieve the latest source before I answer.",
        "I'll keep researching this and report back.",
    ]
    for text in must_rewrite:
        gate = CapabilityGate(_fresh_registry())
        gate.set_route_hint("none")
        out = gate.check_text(text)
        assert "retrieving" not in out.lower()
        assert "retrieve the latest" not in out.lower()
        assert "keep researching" not in out.lower()
        assert (
            "background research or retrieval task" in out.lower()
            or "capability yet" in out.lower()
        ), f"Expected unbacked tool action rewrite: {text!r} -> {out!r}"
        gate.set_route_hint(None)
    print("  PASS: NONE route blocks unbacked tool-shaped action language")


def test_general_knowledge_chat_still_passes_on_none_route():
    """Definitions and casual explanations are still valid NONE conversation."""
    gate = CapabilityGate(_fresh_registry())
    gate.set_route_hint("none")
    text = "A dog is a domesticated mammal closely related to wolves."
    out = gate.check_text(text)
    assert out == text
    gate.set_route_hint(None)
    print("  PASS: NONE route keeps general knowledge conversation natural")


# ── Intention infrastructure Stage 0: backed-commitment regressions ────────

def test_commitment_unbacked_is_rewritten():
    """Unbacked deferred-action promises are rewritten to fail-closed text."""
    gate = CapabilityGate(_fresh_registry())
    text = "Sure, give me a moment while I process and organize this."
    new_text, changed = gate.evaluate_commitment(text, backing_job_ids=None)
    assert changed is True
    assert "I don't have a background task to follow up on that right now." in new_text
    assert "give me a moment" not in new_text.lower()
    print("  PASS: unbacked commitment rewritten")


def test_commitment_backed_passes_through():
    """Backed commitment (real job_id) passes through unchanged."""
    gate = CapabilityGate(_fresh_registry())
    text = "Sure, give me a moment while I process and organize this."
    new_text, changed = gate.evaluate_commitment(
        text, backing_job_ids=["library_ingest_job_123"],
    )
    assert changed is False, f"backed commitment must pass unchanged: {new_text!r}"
    assert new_text == text
    print("  PASS: backed commitment passes through")


def test_commitment_20_31_20_regression():
    """Regression for the 20:31:20 log entry: Jarvis said
    'give me a moment as I process and organize' with no backing task."""
    gate = CapabilityGate(_fresh_registry())
    text = (
        "Of course. Give me a moment as I process and organize the "
        "information for you."
    )
    new_text, changed = gate.evaluate_commitment(text, backing_job_ids=[])
    assert changed is True
    assert "background task" in new_text.lower()
    print("  PASS: 20:31:20 unbacked-promise regression")


def test_commitment_safe_conversational_reflections_ignored():
    """Safe conversational phrases like 'I'll think about it' are
    filtered upstream and never cause a rewrite even with no backing."""
    gate = CapabilityGate(_fresh_registry())
    for text in (
        "I'll think about it.",
        "I'll keep that in mind.",
        "I'll remember that.",
        "Sure, I'll consider that.",
    ):
        new_text, changed = gate.evaluate_commitment(text, backing_job_ids=None)
        assert changed is False, (
            f"safe reflection was wrongly rewritten: {text!r} -> {new_text!r}"
        )
    print("  PASS: safe conversational reflections ignored")


def test_commitment_empty_text_is_noop():
    gate = CapabilityGate(_fresh_registry())
    for t in ("", "   ", "ok"):
        new_text, changed = gate.evaluate_commitment(t, backing_job_ids=None)
        assert changed is False
        assert new_text == t
    print("  PASS: empty/short text commitment no-op")


def test_commitment_multi_sentence_preserves_unrelated_sentences():
    """A commitment in one sentence must not consume neighbouring sentences."""
    gate = CapabilityGate(_fresh_registry())
    text = (
        "That's a great question. Let me look into that and get back to you. "
        "In the meantime, feel free to ask anything else."
    )
    new_text, changed = gate.evaluate_commitment(text, backing_job_ids=None)
    assert changed is True
    assert "That's a great question." in new_text
    assert "In the meantime" in new_text
    assert "background task" in new_text.lower()
    print("  PASS: multi-sentence rewrite preserves neighbours")


def test_commitment_tool_shaped_future_work_unbacked_rewritten():
    gate = CapabilityGate(_fresh_registry())
    for text in (
        "I'll begin by retrieving recent papers on that.",
        "Let me retrieve the latest source before I answer.",
        "I'll keep researching this and report back.",
    ):
        new_text, changed = gate.evaluate_commitment(text, backing_job_ids=[])
        assert changed is True
        assert "background task" in new_text.lower()
    print("  PASS: tool-shaped future work commitments rewrite when unbacked")


if __name__ == "__main__":
    print("\n=== Capability Gate Tests (Registry-First) ===\n")
    test_blocks_operational_claim_with_safe_substring()
    test_blocks_technical_claim_signals()
    test_blocks_unknown_operational_claims()
    test_must_pass_safe_phrases()
    test_registry_sensitive_verbs_fall_to_check()
    test_registry_sensitive_verb_passes_when_verified()
    test_conversational_classifier()
    test_preference_acknowledgement_does_not_trigger_capability_block()
    test_must_block_unicode_and_ascii()
    test_must_rewrite_when_learning()
    test_vague_learning_not_caught()
    test_grounded_perception_exemption()
    test_honesty_failures_counter()
    test_gate_block_reasons_tracked()
    test_resolver_diarization()
    test_resolver_dynamic_skill_id()
    test_diarization_assess_executor()
    test_diarization_verify_executor()
    test_blocks_expanded_offer_patterns()
    test_blocks_residual_sweep_patterns()
    test_blocks_multi_clause_bypass()
    test_must_not_block_reflective_text()
    test_narration_guard_blocks_on_none_route()
    test_narration_guard_passes_on_skill_route()
    test_narration_guard_phase_pattern()
    test_narration_guard_no_false_positives()
    test_narration_guard_latch_suppresses_subsequent_chunks()
    test_none_route_blocks_unbacked_tool_shaped_action_language()
    test_general_knowledge_chat_still_passes_on_none_route()
    # Intention infrastructure Stage 0 regressions
    test_commitment_unbacked_is_rewritten()
    test_commitment_backed_passes_through()
    test_commitment_20_31_20_regression()
    test_commitment_safe_conversational_reflections_ignored()
    test_commitment_empty_text_is_noop()
    test_commitment_multi_sentence_preserves_unrelated_sentences()
    test_commitment_tool_shaped_future_work_unbacked_rewritten()
    print("\n  All tests passed!\n")
