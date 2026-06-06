"""#9.2: predictive world-model rule cleanup.

Pins:
  - the 4 near-duplicate user.present persistence rules are dropped (rule set +
    _PERSISTENCE_RULE_IDS);
  - the 2 new transition rules exist and are classified PREDICTIVE (not persistence);
  - the honesty invariant: persistence-rule hits feed persistence_accuracy ONLY and
    can never inflate predictive_accuracy / predictive_total.
"""
from __future__ import annotations

from cognition.causal_engine import (
    CausalEngine,
    _build_default_rules,
    _PERSISTENCE_RULE_IDS,
)

DROPPED = {
    "idle_user_no_conversation",
    "stable_scene_persists",
    "display_zone_mode_stable",
    "multi_entity_scene_stable",
}
ADDED = {
    "mode_changed_to_sleep_absence",
    "topic_changed_conversation_continues",
}


def _rule_ids():
    return {r.rule_id for r in _build_default_rules()}


class TestRuleSetCleanup:
    def test_dropped_rules_removed_from_ruleset(self):
        ids = _rule_ids()
        for r in DROPPED:
            assert r not in ids, f"{r} should be dropped"

    def test_dropped_rules_removed_from_persistence_set(self):
        for r in DROPPED:
            assert r not in _PERSISTENCE_RULE_IDS

    def test_canonical_presence_persistence_kept(self):
        ids = _rule_ids()
        assert "present_user_stays" in ids
        assert "present_user_stays" in _PERSISTENCE_RULE_IDS

    def test_added_rules_present(self):
        ids = _rule_ids()
        for r in ADDED:
            assert r in ids, f"{r} should be added"

    def test_added_rules_classified_predictive_not_persistence(self):
        # New transition rules MUST NOT be persistence — else they couldn't count as
        # foresight, defeating the point.
        for r in ADDED:
            assert r not in _PERSISTENCE_RULE_IDS

    def test_no_duplicate_rule_ids(self):
        rules = _build_default_rules()
        ids = [r.rule_id for r in rules]
        assert len(ids) == len(set(ids))


class TestPersistenceCannotInflatePredictive:
    def test_persistence_hits_excluded_from_predictive_total(self):
        eng = CausalEngine()
        # 100 persistence hits, 10 genuine predictive hits.
        eng._rule_hits["present_user_stays"] = 100          # persistence
        eng._rule_hits["mode_changed_to_sleep_absence"] = 10  # predictive (new)
        acc = eng.get_accuracy()
        assert acc["persistence_total"] == 100
        assert acc["predictive_total"] == 10  # persistence NOT pooled in
        assert acc["predictive_accuracy"] == 1.0
        assert acc["per_rule"]["present_user_stays"]["kind"] == "persistence"
        assert acc["per_rule"]["mode_changed_to_sleep_absence"]["kind"] == "predictive"

    def test_pure_persistence_yields_zero_predictive_coverage(self):
        eng = CausalEngine()
        eng._rule_hits["present_user_stays"] = 50
        eng._rule_hits["quiet_desk_stays_quiet"] = 50
        eng._rule_misses["present_user_stays"] = 5
        acc = eng.get_accuracy()
        # All hits are persistence -> predictive coverage is empty, not inflated.
        assert acc["predictive_total"] == 0
        assert acc["predictive_accuracy"] == 0.0
        assert acc["persistence_total"] == 105

    def test_predictive_accuracy_live_excludes_persistence(self):
        eng = CausalEngine()
        eng._rule_hits_live["present_user_stays"] = 80      # persistence (live)
        eng._rule_hits_live["topic_changed_conversation_continues"] = 8  # predictive (live)
        acc = eng.get_accuracy()
        assert acc["predictive_total_live"] == 8
        assert acc["predictive_accuracy_live"] == 1.0
