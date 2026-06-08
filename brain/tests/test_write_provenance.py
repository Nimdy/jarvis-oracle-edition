"""Banter firewall — write-provenance rule (golden-command authority model).

Casual chatter must never become an asserted fact. A golden command is the
write-authority; otherwise playful/casual banter is downgraded to the 0.0-trust
``casual_conversation`` class; serious/neutral mentions keep their base provenance.
"""
from __future__ import annotations

import pytest

try:
    from consciousness.events import (
        PROVENANCE_BOOST, PROVENANCE_ORDINAL, resolve_write_provenance,
    )
except Exception:  # pragma: no cover - heavy deps absent
    pytest.skip("consciousness.events import unavailable", allow_module_level=True)


class TestCasualProvenanceTier:
    def test_registered_at_zero_trust(self):
        assert PROVENANCE_BOOST["casual_conversation"] == 0.0
        assert "casual_conversation" in PROVENANCE_ORDINAL

    def test_strictly_below_user_claim_and_conversation(self):
        assert PROVENANCE_BOOST["casual_conversation"] < PROVENANCE_BOOST["user_claim"]
        assert PROVENANCE_BOOST["casual_conversation"] < PROVENANCE_BOOST["conversation"]
        # least authoritative source in the ordinal ranking
        assert PROVENANCE_ORDINAL["casual_conversation"] == max(PROVENANCE_ORDINAL.values())


class TestWriteProvenanceRule:
    def test_golden_command_is_authority_even_when_playful(self):
        # the user explicitly meant it, mid-banter
        assert resolve_write_provenance("user_claim", is_golden_command=True, tone="playful") == "user_claim"
        assert resolve_write_provenance("user_claim", is_golden_command=True, tone="casual") == "user_claim"

    def test_banter_downgraded_to_casual(self):
        assert resolve_write_provenance("user_claim", tone="playful") == "casual_conversation"
        assert resolve_write_provenance("user_claim", tone="casual") == "casual_conversation"

    def test_serious_neutral_keeps_base(self):
        assert resolve_write_provenance("user_claim", tone="professional") == "user_claim"
        assert resolve_write_provenance("user_claim", tone="empathetic") == "user_claim"
        assert resolve_write_provenance("user_claim", tone="") == "user_claim"

    def test_never_elevates_trust(self):
        # whatever the rule returns, it can't be MORE trusted than the base
        base = "user_claim"
        for golden in (True, False):
            for tone in ("playful", "casual", "professional", "urgent", "empathetic", ""):
                out = resolve_write_provenance(base, is_golden_command=golden, tone=tone)
                assert PROVENANCE_BOOST.get(out, 0.0) <= PROVENANCE_BOOST[base]

    def test_dirt_on_pizza_scenario(self):
        # "I love dirt on pizza" said while bsing -> protected, never a fact
        prov = resolve_write_provenance("user_claim", is_golden_command=False, tone="playful")
        assert prov == "casual_conversation"
        assert PROVENANCE_BOOST[prov] == 0.0
        # ...but "Jarvis, remember I like extra garlic" (golden) IS authoritative
        prov2 = resolve_write_provenance("user_claim", is_golden_command=True, tone="playful")
        assert prov2 == "user_claim"
