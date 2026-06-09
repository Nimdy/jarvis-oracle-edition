"""Test beliefs must never hit the live ~/.jarvis/beliefs.jsonl (GitHub #50).

Regression for the bel_test_98 leak: the autouse `_isolate_belief_store` fixture
(conftest) redirects the belief-store path constants to a tmp dir, and BeliefStore
resolves its path at call-time, so a default-constructed store writes to tmp.
"""
from __future__ import annotations

import os

import pytest

try:
    from epistemic.belief_record import BeliefStore
except Exception:  # pragma: no cover - heavy deps absent
    pytest.skip("epistemic.belief_record import unavailable", allow_module_level=True)


def test_default_belief_store_is_isolated_to_tmp(tmp_path):
    # autouse fixture patched _BELIEFS_FILE -> tmp_path/beliefs.jsonl
    s = BeliefStore()
    assert s._beliefs_path == str(tmp_path / "beliefs.jsonl")
    assert s._tensions_path == str(tmp_path / "tensions.jsonl")


def test_belief_store_never_writes_live_home():
    s = BeliefStore()
    live = os.path.expanduser("~/.jarvis")
    assert not s._beliefs_path.startswith(live)


def test_explicit_path_still_honored(tmp_path):
    # passing a path explicitly must still work (regression on the None-default change)
    p = str(tmp_path / "explicit_beliefs.jsonl")
    s = BeliefStore(beliefs_path=p)
    assert s._beliefs_path == p
