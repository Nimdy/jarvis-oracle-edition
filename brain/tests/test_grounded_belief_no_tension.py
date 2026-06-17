"""Regression: an already-grounded belief must carry ~0 grounding-tension so it
drops out of the queue regardless of hub leverage. Before the fix, a grounded HUB
belief still got the global pressure term × leverage and was re-asked after the
operator had already confirmed it.
"""
from __future__ import annotations

import types

from epistemic.provenance_scorer import ProvenanceScorer, _GROUNDED_PROVENANCE


def _tension(provenance, is_orphan=True, pressure=0.9):
    # (belief, is_orphan, base_conf, effective_conf, pressure)
    t, detail = ProvenanceScorer._belief_tension(
        types.SimpleNamespace(provenance=provenance), is_orphan, 0.5, 0.5, pressure)
    return t, detail


def test_every_grounded_provenance_gets_zero_tension():
    # operator/sensor/cited-source beliefs are externally validated -> no re-grounding
    for prov in _GROUNDED_PROVENANCE:
        t, d = _tension(prov)
        assert t == 0.0, f"{prov} should be 0 tension, got {t}"
        assert d.get("grounded") == 1.0


def test_inferred_belief_still_carries_tension():
    t, _ = _tension("model_inference")
    assert t > 0.0


def test_web_scrap_is_not_treated_as_grounded():
    # untrusted scraped web is NOT grounded (firewall) -> still carries tension
    t, _ = _tension("web_scrap")
    assert t > 0.0


def test_grounded_hub_beats_pressure_and_leverage():
    # the actual bug: grounded belief, even as an orphan under high system pressure,
    # must be 0 (so tension*leverage can't re-queue it)
    t, _ = _tension("user_claim", is_orphan=True, pressure=1.0)
    assert t == 0.0
