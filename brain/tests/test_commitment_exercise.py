"""Tests for the synthetic commitment exercise harness.

Covers:
  - Corpus validation (every category has >= 3 utterances, strings).
  - Expected gate-action mapping is complete and uses valid labels.
  - Profile definitions are structurally valid.
  - Invariant guards (no memory writes, no registry mutations, no LLM calls).
  - Determinism under a fixed seed.
  - Accuracy floor on smoke + coverage profiles (100% expected; corpus
    was tightened to match current Stage-0 extractor coverage).
  - Report shape (to_dict).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Ensure brain/ is on the path regardless of pytest working directory.
BRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(BRAIN_ROOT))

from synthetic.commitment_exercise import (  # noqa: E402
    COMMITMENT_CATEGORIES,
    COMMITMENT_PROFILES,
    COVERAGE_WEIGHTS,
    EXPECTED_GATE_ACTION,
    CommitmentExerciseStats,
    pick_utterance,
    run_commitment_exercise,
)


# --------------------------- Corpus validation ------------------------------

def test_every_category_has_at_least_three_utterances():
    for cat, items in COMMITMENT_CATEGORIES.items():
        assert len(items) >= 3, f"{cat!r} has only {len(items)} utterance(s)"
        for u in items:
            assert isinstance(u, str)
            assert len(u.strip()) >= 4


def test_expected_action_keys_match_categories():
    assert set(EXPECTED_GATE_ACTION) == set(COMMITMENT_CATEGORIES), (
        "EXPECTED_GATE_ACTION categories drifted from COMMITMENT_CATEGORIES"
    )
    for cat, action in EXPECTED_GATE_ACTION.items():
        assert action in {"pass", "rewrite"}, (
            f"category {cat!r} has unknown expected action {action!r}"
        )


def test_coverage_weights_cover_every_category():
    missing = set(COMMITMENT_CATEGORIES) - set(COVERAGE_WEIGHTS)
    assert not missing, f"COVERAGE_WEIGHTS missing: {missing}"
    for v in COVERAGE_WEIGHTS.values():
        assert v >= 0


# --------------------------- Profile validation -----------------------------

def test_named_profiles_present_and_valid():
    for name in ("smoke", "coverage", "strict", "stress"):
        p = COMMITMENT_PROFILES[name]
        assert p.name == name
        assert p.count > 0
        assert p.delay_s >= 0
        if p.category_weights is not None:
            assert set(p.category_weights) <= set(COMMITMENT_CATEGORIES)


# --------------------------- Picker behavior --------------------------------

def test_pick_utterance_honors_category():
    for cat in ("backed_follow_up", "unbacked_task_started", "conversational_safe"):
        text, returned_cat = pick_utterance(category=cat)
        assert returned_cat == cat
        assert text in COMMITMENT_CATEGORIES[cat]


def test_pick_utterance_deterministic_under_seed():
    import random
    rng1 = random.Random(123)
    rng2 = random.Random(123)
    seq1 = [pick_utterance(weights=COVERAGE_WEIGHTS, rng=rng1) for _ in range(10)]
    seq2 = [pick_utterance(weights=COVERAGE_WEIGHTS, rng=rng2) for _ in range(10)]
    assert seq1 == seq2


# --------------------------- Invariant guards -------------------------------

def test_module_does_not_import_intention_registry_at_module_level():
    """The exercise must NEVER touch the real IntentionRegistry singleton.

    This is enforced structurally: synthetic/commitment_exercise.py does
    not import cognition.intention_registry at module load. The guard
    lets us regression-test that contract.
    """
    mod = importlib.import_module("synthetic.commitment_exercise")
    src = Path(mod.__file__).read_text()
    assert "from cognition.intention_registry" not in src
    assert "import cognition.intention_registry" not in src


def test_smoke_run_touches_no_real_subsystems(monkeypatch):
    """A smoke run must NOT resolve real intentions, write memory, or
    call the LLM. We monkeypatch the most likely leakage surfaces to raise
    if they are invoked."""
    # ``cognition.intention_registry`` is re-exported as the singleton
    # instance via cognition/__init__.py, so a bare ``import cognition``
    # returns the singleton, not the module. We use importlib to fetch
    # the actual module object.
    import importlib
    ir_mod = importlib.import_module("cognition.intention_registry")
    registry = ir_mod.intention_registry  # module-level singleton

    def _boom(*_a, **_k):
        raise RuntimeError("synthetic exercise attempted a real registry mutation")

    monkeypatch.setattr(registry, "register", _boom)
    monkeypatch.setattr(registry, "resolve", _boom)
    monkeypatch.setattr(registry, "abandon", _boom)

    stats = run_commitment_exercise(profile=COMMITMENT_PROFILES["smoke"], seed=42)
    assert stats.leaked_memory_writes == 0
    assert stats.leaked_registry_mutations == 0
    assert stats.leaked_llm_calls == 0
    assert stats.utterances_processed >= 1


# --------------------------- Accuracy floor ---------------------------------

def test_smoke_profile_accuracy_is_perfect_with_seeded_run():
    stats = run_commitment_exercise(profile=COMMITMENT_PROFILES["smoke"], seed=42)
    assert stats.utterances_processed == COMMITMENT_PROFILES["smoke"].count
    assert stats.utterances_failed == 0
    # Corpus was tightened to the current Stage-0 extractor coverage; we
    # expect perfect accuracy. If this falls below 1.0, either the extractor
    # regressed or the corpus drifted — both are serious.
    assert stats.accuracy == 1.0, f"smoke accuracy regressed: {stats.accuracy}"
    assert stats.pass_result is True


def test_coverage_profile_accuracy_meets_floor():
    stats = run_commitment_exercise(profile=COMMITMENT_PROFILES["coverage"], seed=7)
    assert stats.utterances_processed == COMMITMENT_PROFILES["coverage"].count
    assert stats.accuracy >= 0.95, (
        f"coverage accuracy below floor: {stats.accuracy:.3f}\n"
        f"mismatches: {stats.mismatch_details[:5]}"
    )


def test_stress_profile_does_not_explode():
    # Tight count to keep CI fast; the profile definition is stress-sized
    # but we override so tests stay snappy.
    stats = run_commitment_exercise(
        profile=COMMITMENT_PROFILES["stress"], seed=99, count=50,
    )
    assert stats.utterances_processed == 50
    assert stats.utterances_failed == 0
    assert stats.accuracy >= 0.95


# --------------------------- Report shape -----------------------------------

def test_stats_to_dict_has_expected_shape():
    stats = run_commitment_exercise(profile=COMMITMENT_PROFILES["smoke"], seed=1)
    d = stats.to_dict()
    for key in (
        "profile", "seed", "utterances_requested", "utterances_processed",
        "utterances_failed", "categories_exercised", "gate_actions",
        "expected_matches", "expected_mismatches", "mismatch_details",
        "accuracy", "errors", "elapsed_s", "rate_per_sec",
        "leaked_memory_writes", "leaked_registry_mutations",
        "leaked_llm_calls", "pass", "fail_reasons",
    ):
        assert key in d, f"missing report key: {key}"
    assert isinstance(d["pass"], bool)
    assert isinstance(d["categories_exercised"], dict)
    assert isinstance(d["gate_actions"], dict)


def test_summary_contains_pass_or_fail_marker():
    stats = run_commitment_exercise(profile=COMMITMENT_PROFILES["smoke"], seed=5)
    s = stats.summary()
    assert ("PASS" in s) or ("FAIL" in s)


# --------------------------- Invariant: determinism -------------------------

def test_same_seed_produces_identical_run():
    a = run_commitment_exercise(profile=COMMITMENT_PROFILES["smoke"], seed=17)
    b = run_commitment_exercise(profile=COMMITMENT_PROFILES["smoke"], seed=17)
    # Category histogram + gate actions must be identical under the same seed.
    assert dict(a.categories_exercised) == dict(b.categories_exercised)
    assert dict(a.gate_actions) == dict(b.gate_actions)
    assert a.expected_matches == b.expected_matches
    assert a.expected_mismatches == b.expected_mismatches


# --------------------------- Empty/unicode edge cases -----------------------

def test_utterances_do_not_contain_cr_or_null():
    for cat, items in COMMITMENT_CATEGORIES.items():
        for u in items:
            assert "\r" not in u, f"{cat!r}: CR present in {u!r}"
            assert "\x00" not in u, f"{cat!r}: null byte in {u!r}"
