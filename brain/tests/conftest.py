"""Shared pytest fixtures for the brain test-suite.

Keep the counterfactual engine's persisted state out of the real ~/.jarvis.
The Layer-9 reflective audit (test_reflective_audit.py::run_audit) now invokes
the counterfactual engine, which persists to STATE_PATH. Without isolation that
write lands in the live runtime's ~/.jarvis. Redirect it to a tmp file for
every test. Wrapped defensively so a missing import can never break collection.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_counterfactual_state(tmp_path, monkeypatch):
    try:
        import epistemic.counterfactual.engine as cfmod
        monkeypatch.setattr(cfmod, "STATE_PATH", str(tmp_path / "cf_state.json"), raising=False)
        cfmod.CounterfactualEngine._instance = None
        yield
        cfmod.CounterfactualEngine._instance = None
    except Exception:
        yield
