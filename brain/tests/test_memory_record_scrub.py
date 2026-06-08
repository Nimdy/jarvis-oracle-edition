"""Live bug: raw memory-record fields (type=/source_id=/chunk_ids=[...]) leaked into
a SPOKEN reply ('Here's what I remember about that. type=study_claim, source_id=...,
chunk_ids=[...]') — corrupting the answer and choking the phonemizer on hash tokens.
"""
from __future__ import annotations

from conversation_handler import _scrub_record_artifacts, _format_grounded_fallback

# the actual leaked preview from David's live session
LEAK = ("type=study_claim, source_id=src_80b5bf73311d2e3c, claim=Concept: Memory, "
        "claim_type=definition, chunk_ids=['chk_dbbf80d2604e28f2', 'chk_47f1cb22ec167d65']")

FORBIDDEN = ("type=study_claim", "source_id=", "chunk_ids", "claim_type=", "src_80b5", "chk_dbbf")


class TestScrub:
    def test_extracts_claim_drops_metadata_and_hashes(self):
        out = _scrub_record_artifacts(LEAK)
        assert "Concept: Memory" in out
        for bad in FORBIDDEN:
            assert bad not in out, bad

    def test_fallback_never_speaks_raw_record(self):
        reply = _format_grounded_fallback("Memory recall", LEAK)
        for bad in FORBIDDEN:
            assert bad not in reply, bad
        # no bare id-hash token survives (phonemizer choke source)
        assert "chk_" not in reply and "src_" not in reply

    def test_plain_text_is_untouched(self):
        # a normal memory must pass through cleanly (no over-scrubbing)
        s = "You are David and we have talked many times."
        assert _scrub_record_artifacts(s) == s

    def test_empty_safe(self):
        assert _scrub_record_artifacts("") == ""
        assert _scrub_record_artifacts(None) == ""
