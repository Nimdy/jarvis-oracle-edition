"""Fidelity #8.1 — self-status must not silently drop resolver verdict stats.

The self_status block accumulates into `lines` (-> MeaningFrame.facts), but it used
`facts.append` for the resolver-verdict enrichment — an undefined name in that scope,
so it raised NameError, swallowed by a bare except, silently dropping the stats from
JARVIS's report of its own cognition. This locks the fix.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from reasoning.bounded_response import build_meaning_frame


def test_self_status_surfaces_resolver_verdicts_not_dropped():
    fake_resolver = MagicMock()
    fake_resolver.get_status.return_value = {
        "total_evaluated": 12,
        "shadow_metrics": {"sufficient_data": True, "shadow_accuracy": 0.83},
    }
    with patch("cognition.intention_resolver.get_intention_resolver", return_value=fake_resolver):
        frame = build_meaning_frame(
            response_class="self_status",
            grounding_payload="Status: nominal\nUptime: 1h",
        )
    joined = " ".join(frame.facts)
    # Before the fix these lines were NameError-dropped -> absent from facts.
    assert "Resolver shadow verdicts evaluated: 12" in joined
    assert "Resolver shadow accuracy: 83.0%" in joined


def test_self_status_resolver_enrichment_failure_is_nonfatal():
    # If the resolver itself raises, self-status must still build (degrade, not crash).
    fake_resolver = MagicMock()
    fake_resolver.get_status.side_effect = RuntimeError("resolver down")
    with patch("cognition.intention_resolver.get_intention_resolver", return_value=fake_resolver):
        frame = build_meaning_frame(
            response_class="self_status",
            grounding_payload="Status: nominal",
        )
    assert frame.response_class == "self_status"
    assert isinstance(frame.facts, list)
