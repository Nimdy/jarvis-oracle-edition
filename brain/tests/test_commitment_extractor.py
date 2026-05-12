"""Tests for cognition.commitment_extractor.

Stage 0 infrastructure: verify that the regex bootstrap correctly classifies
outgoing commitment speech acts, respects the conversational safe list, and
catches the 20:31:20 regression utterance that triggered this build.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cognition.commitment_extractor import (
    CommitmentMatch,
    extract_commitments,
    COMMITMENT_PATTERNS,
    CONVERSATIONAL_SAFE_PATTERNS,
)


# ── Positive matches ────────────────────────────────────────────────────────

def test_follow_up_get_back_to_you():
    out = extract_commitments("Sure thing. I'll get back to you once I have a result.")
    assert out, "expected follow_up match on 'get back to you'"
    assert out[0].commitment_type == "follow_up"


def test_follow_up_let_you_know_when():
    out = extract_commitments("I'll let you know when it finishes.")
    assert out
    assert out[0].commitment_type == "follow_up"


def test_deferred_give_me_a_moment():
    out = extract_commitments("Give me a moment while I pull that up.")
    assert out
    assert out[0].commitment_type == "deferred_action"


def test_deferred_allow_me_a_moment_process():
    # 20:31:20 regression — "allow me a moment to process/organize"
    text = "Allow me a moment to process and organize the research I was asked about."
    out = extract_commitments(text)
    assert out, f"20:31:20 regression: no commitment match on {text!r}"
    ctypes = {m.commitment_type for m in out}
    assert "deferred_action" in ctypes or "future_work" in ctypes


def test_deferred_let_me_process():
    out = extract_commitments("Let me process that for you.")
    assert out
    assert out[0].commitment_type == "deferred_action"


def test_future_work_i_will_analyze():
    out = extract_commitments("I'll analyze the findings and come back to you.")
    assert out
    ctypes = {m.commitment_type for m in out}
    assert "future_work" in ctypes or "follow_up" in ctypes


def test_future_work_begin_by_retrieving():
    out = extract_commitments("I'll begin by retrieving recent papers on that.")
    assert out, "retrieval-shaped future work must be detected"
    assert out[0].commitment_type == "future_work"


def test_deferred_let_me_retrieve():
    out = extract_commitments("Let me retrieve the latest source before I answer.")
    assert out, "retrieval-shaped deferred work must be detected"
    assert out[0].commitment_type == "deferred_action"


def test_future_work_keep_researching():
    out = extract_commitments("I'll keep researching this and report back.")
    assert out, "continued research promise must be detected"
    ctypes = {m.commitment_type for m in out}
    assert "future_work" in ctypes or "follow_up" in ctypes


def test_task_started_past_tense():
    # 20:31:20 regression style: "I've initiated a study"
    out = extract_commitments("I've initiated a study into that topic.")
    assert out, "expected task_started match on past-tense action claim"
    assert out[0].commitment_type == "task_started"


def test_task_started_present_progressive():
    out = extract_commitments("I am currently researching that for you.")
    assert out
    assert out[0].commitment_type == "task_started"


# ── Negative matches (conversational safe list) ────────────────────────────

def test_safe_think_about_it():
    out = extract_commitments("Interesting question. I'll think about it.")
    assert out == [], "'think about it' must be conversationally safe"


def test_safe_consider_that():
    out = extract_commitments("I'll consider that carefully.")
    assert out == []


def test_safe_keep_that_in_mind():
    out = extract_commitments("I'll keep that in mind going forward.")
    assert out == []


def test_safe_remember_that():
    out = extract_commitments("I'll remember that for next time.")
    assert out == []


def test_safe_purely_informational():
    out = extract_commitments("That sounds like an interesting idea.")
    assert out == []


def test_empty_input_is_empty():
    assert extract_commitments("") == []
    assert extract_commitments(None) == []  # type: ignore[arg-type]


# ── Match shape + metadata ──────────────────────────────────────────────────

def test_match_provides_sentence_and_span():
    text = "All set. I'll get back to you shortly. Let me know if anything else."
    out = extract_commitments(text)
    assert out
    m = out[0]
    assert isinstance(m, CommitmentMatch)
    assert m.full_sentence
    assert m.span[0] < m.span[1]
    assert m.phrase


def test_iterable_unpacking_supported():
    text = "I'll let you know when ready."
    out = extract_commitments(text)
    assert out
    phrase, ctype, kind = out[0]
    assert phrase
    assert ctype in {"follow_up", "deferred_action", "future_work",
                     "task_started", "generic"}
    assert kind in {"follow_up_any", "any_background", "unknown"} or kind


def test_pattern_table_size_bootstrap():
    """We expect ~15 bootstrap patterns per the plan."""
    assert 10 <= len(COMMITMENT_PATTERNS) <= 30
    assert len(CONVERSATIONAL_SAFE_PATTERNS) >= 4


if __name__ == "__main__":
    import traceback
    fns = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    passed = failed = 0
    for f in fns:
        try:
            f()
            passed += 1
            print(f"  PASS: {f.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {f.__name__}: {e}")
        except Exception:
            failed += 1
            print(f"  ERROR: {f.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
