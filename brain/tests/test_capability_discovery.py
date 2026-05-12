"""Tests for Capability Discovery: normalization, tracking, gap analysis, proposals."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from skills.discovery import (
    BUILTIN_FAMILIES,
    BlockFrequencyTracker,
    BlockPattern,
    CapabilityFamily,
    CapabilityFamilyNormalizer,
    CapabilityGap,
    GapAnalyzer,
    LearningProposer,
    PendingProposal,
    _MIN_BLOCKS_FOR_ACTION,
    _MIN_BLOCKS_FOR_PROPOSAL,
)


# ---------------------------------------------------------------------------
# Capability Family Normalization
# ---------------------------------------------------------------------------

class TestCapabilityFamilyNormalizer:
    def test_exact_alias_match(self):
        n = CapabilityFamilyNormalizer()
        fam = n.normalize("camera_control", "")
        assert fam.family_id == "camera_control"
        assert fam.builtin is True

    def test_partial_alias_match(self):
        n = CapabilityFamilyNormalizer()
        fam = n.normalize("adjust_zoom", "set camera zoom")
        assert fam.family_id == "camera_control"

    def test_singing_family(self):
        n = CapabilityFamilyNormalizer()
        fam = n.normalize("singing_v1", "")
        assert fam.family_id == "singing"

    def test_version_suffix_stripped(self):
        n = CapabilityFamilyNormalizer()
        fam = n.normalize("singing_v1", "")
        assert fam.family_id == "singing"

    def test_keyword_overlap_from_text(self):
        n = CapabilityFamilyNormalizer()
        fam = n.normalize(None, "can you sing a song")
        assert fam.family_id == "singing"

    def test_dynamic_family_creation(self):
        n = CapabilityFamilyNormalizer()
        fam = n.normalize("quantum_teleportation_v1", "teleport me")
        assert fam.builtin is False
        assert fam.family_id is not None

    def test_dynamic_family_reuse(self):
        n = CapabilityFamilyNormalizer()
        fam1 = n.normalize("quantum_teleportation_v1", "")
        fam2 = n.normalize("quantum_teleportation_v1", "")
        assert fam1.family_id == fam2.family_id

    def test_domain_inferred_from_verbs(self):
        n = CapabilityFamilyNormalizer()
        fam = n.normalize("zoom_something_new", "zoom in close")
        if fam.family_id == "camera_control":
            assert fam.domain == "actuator"
        else:
            assert fam.domain in ("actuator", "unknown")

    def test_all_builtin_families_have_aliases(self):
        for fid, fam in BUILTIN_FAMILIES.items():
            assert len(fam.aliases) > 0
            assert fam.family_id == fid


# ---------------------------------------------------------------------------
# Block Frequency Tracker
# ---------------------------------------------------------------------------

class TestBlockFrequencyTracker:
    def test_record_increments(self):
        n = CapabilityFamilyNormalizer()
        t = BlockFrequencyTracker(n)
        t._patterns.clear()  # reset for test isolation
        t.record_block("singing_v1", "I can sing")
        t.record_block("singing_v1", "I'd love to sing")
        patterns = t.get_all_patterns()
        assert "singing" in patterns
        assert patterns["singing"].block_count == 2
        assert patterns["singing"].session_blocks == 2

    def test_surface_phrases_capped(self):
        n = CapabilityFamilyNormalizer()
        t = BlockFrequencyTracker(n)
        t._patterns.clear()
        for i in range(15):
            t.record_block("singing_v1", f"phrase {i}")
        p = t.get_all_patterns()["singing"]
        assert len(p.surface_phrases) <= 10

    def test_family_normalization_groups_aliases(self):
        n = CapabilityFamilyNormalizer()
        t = BlockFrequencyTracker(n)
        t._patterns.clear()
        t.record_block("camera_control", "zoom in")
        t.record_block("adjust_zoom", "set zoom to zero")
        t.record_block("camera_zoom", "zoom out")
        patterns = t.get_all_patterns()
        assert "camera_control" in patterns
        assert patterns["camera_control"].block_count == 3

    def test_actionable_patterns_threshold(self):
        n = CapabilityFamilyNormalizer()
        t = BlockFrequencyTracker(n)
        t._patterns.clear()
        for _ in range(_MIN_BLOCKS_FOR_ACTION):
            t.record_block("singing_v1", "sing")
        actionable = t.get_actionable_patterns()
        assert len(actionable) == 1

    def test_snapshot_format(self):
        n = CapabilityFamilyNormalizer()
        t = BlockFrequencyTracker(n)
        t._patterns.clear()
        t.record_block("singing_v1", "sing")
        snap = t.get_snapshot()
        assert len(snap) == 1
        assert "family_id" in snap[0]
        assert "domain" in snap[0]


# ---------------------------------------------------------------------------
# Gap Analyzer
# ---------------------------------------------------------------------------

class TestGapAnalyzer:
    def _make_tracker_with_blocks(self, count: int = 5) -> BlockFrequencyTracker:
        n = CapabilityFamilyNormalizer()
        t = BlockFrequencyTracker(n)
        t._patterns.clear()
        for _ in range(count):
            t.record_block("singing_v1", "I can sing")
        return t

    def test_analyze_returns_gaps(self):
        t = self._make_tracker_with_blocks(5)
        a = GapAnalyzer()
        gaps = a.analyze(t)
        assert len(gaps) == 1
        assert gaps[0].family.family_id == "singing"

    def test_defer_below_threshold(self):
        t = self._make_tracker_with_blocks(2)
        a = GapAnalyzer()
        gaps = a.analyze(t)
        assert len(gaps) == 0

    def test_propose_to_user_at_5_blocks(self):
        t = self._make_tracker_with_blocks(5)
        a = GapAnalyzer()
        gaps = a.analyze(t)
        assert gaps[0].suggested_action == "propose_to_user"

    def test_retry_with_knowledge_on_prior_failure(self):
        t = self._make_tracker_with_blocks(5)
        p = t.get_all_patterns()["singing"]
        p.job_status = "blocked"
        p.job_failure_reason = "no TTS melody support"
        a = GapAnalyzer()
        gaps = a.analyze(t)
        assert gaps[0].suggested_action == "retry_with_knowledge"

    def test_evidence_strength_bounded(self):
        t = self._make_tracker_with_blocks(20)
        a = GapAnalyzer()
        gaps = a.analyze(t)
        assert gaps[0].evidence_strength <= 1.0

    def test_priority_bounded(self):
        t = self._make_tracker_with_blocks(20)
        a = GapAnalyzer()
        gaps = a.analyze(t)
        assert 0.0 <= gaps[0].priority <= 1.0


# ---------------------------------------------------------------------------
# Learning Proposer
# ---------------------------------------------------------------------------

class TestLearningProposer:
    def _make_gap(self, action: str = "propose_to_user", family_id: str = "singing") -> CapabilityGap:
        family = BUILTIN_FAMILIES.get(family_id) or CapabilityFamily(
            family_id=family_id, domain="creative", canonical_name="Test",
            aliases=frozenset({family_id}), builtin=False,
        )
        return CapabilityGap(
            family=family,
            evidence_strength=0.7,
            evidence_sources=("block_frequency",),
            block_count=5,
            surface_phrases=("I can sing",),
            has_prior_attempt=False,
            prior_failure_reason=None,
            suggested_action=action,
            priority=0.6,
        )

    def test_proposal_queued(self):
        p = LearningProposer()
        gap = self._make_gap()
        actions = p.process_gaps([gap])
        assert len(actions) == 1
        assert actions[0]["action"] == "queued_proposal"

    def test_24h_cooldown(self):
        p = LearningProposer()
        gap = self._make_gap()
        p.process_gaps([gap])
        p.mark_proposed(p.get_next_proposal())
        gap2 = self._make_gap(family_id="drawing")
        actions = p.process_gaps([gap2])
        assert all(a["action"] != "queued_proposal" for a in actions)

    def test_rejection_suppresses_family(self):
        p = LearningProposer()
        gap = self._make_gap()
        p.process_gaps([gap])
        p.record_user_response("singing", accepted=False)
        gap2 = self._make_gap()
        actions = p.process_gaps([gap2])
        assert len(actions) == 0

    def test_research_dispatched(self):
        p = LearningProposer()
        gap = self._make_gap(action="research")
        mock_enqueue = MagicMock()
        actions = p.process_gaps([gap], enqueue_research=mock_enqueue)
        assert any(a["action"] == "research" for a in actions)

    def test_defer_produces_no_action(self):
        p = LearningProposer()
        gap = self._make_gap(action="defer")
        actions = p.process_gaps([gap])
        assert len(actions) == 0

    def test_max_pending_cap(self):
        p = LearningProposer()
        for i in range(10):
            gap = self._make_gap(family_id=f"fam_{i}")
            p.process_gaps([gap])
        assert len(p._pending) <= 5

    def test_bypass_cooldown(self):
        p = LearningProposer()
        gap = self._make_gap()
        p.process_gaps([gap])
        p.mark_proposed(p.get_next_proposal())
        gap2 = self._make_gap(family_id="drawing")
        p.process_gaps([gap2])
        result = p.bypass_cooldown_for_user_ask()
        assert result is not None or len(p._pending) == 0

    def test_snapshot_format(self):
        p = LearningProposer()
        snap = p.get_snapshot()
        assert "pending_count" in snap
        assert "proposals_made" in snap
        assert "next_proposal_eligible_s" in snap
