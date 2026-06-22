"""Tests for P5b autonomous-research (shadow-execute) — the separate zero-authority gate + earning link."""

import autonomy.autonomous_research as ar


def _reset(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "ARP_PROMOTION_PATH", str(tmp_path / "arp.json"))
    monkeypatch.setattr(ar, "ARP_SHADOW_LOG", str(tmp_path / "shadow.jsonl"))
    monkeypatch.setattr(ar, "ARP_MATCHED_LOG", str(tmp_path / "matched.jsonl"))
    ar.AutonomousResearchPromotion.reset_instance()
    ar._pending.clear()


class _F:
    def __init__(self, st, c):
        self.source_type = st
        self.confidence = c


class _R:
    def __init__(self, findings, success=True):
        self.findings = findings
        self.success = success


def test_gate_starts_shadow_zero_authority(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    g = ar.AutonomousResearchPromotion.get_instance()
    assert g.level == 0 and g.is_shadow() and not g.is_active()
    st = g.get_status()
    assert st["authority"] == "zero_authority_shadow"
    assert st["drives_levers"] is False
    assert st["live_fire_enabled"] in (True, False)


def test_gate_persists_across_restart(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    g = ar.AutonomousResearchPromotion.get_instance()
    for _ in range(5):
        g.record_outcome(True)
    assert g._state.total_outcomes == 5
    ar.AutonomousResearchPromotion.reset_instance()          # simulate restart
    g2 = ar.AutonomousResearchPromotion.get_instance()
    assert g2._state.total_outcomes == 5                     # persisted, NOT reset (the spark lesson)


def test_derive_conclusion():
    assert ar.derive_conclusion(_R([_F("peer_reviewed", 0.8)]))["evidence"] == "strong"
    assert ar.derive_conclusion(_R([_F("unverified", 0.3)]))["evidence"] == "weak"
    assert ar.derive_conclusion(_R([]))["evidence"] == "none"
    assert ar.derive_conclusion(_R([_F("peer_reviewed", 0.4)]))["evidence"] == "weak"  # low conf


def test_earning_link_scores_match_and_mismatch(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    g = ar.AutonomousResearchPromotion.get_instance()
    # shadow found strong evidence; operator confirms → MATCH
    ar.record_shadow_conclusion("bel_1", "is X true?", {"evidence": "strong", "n_findings": 3})
    ar.record_operator_answer("bel_1", "confirmed")
    assert g._state.total_outcomes == 1 and g._accuracy() == 1.0
    # shadow found nothing; operator confirms → MISMATCH
    ar.record_shadow_conclusion("bel_2", "q", {"evidence": "none", "n_findings": 0})
    ar.record_operator_answer("bel_2", "confirmed")
    assert g._state.total_outcomes == 2 and g._accuracy() == 0.5


def test_no_pending_is_noop(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    ar.record_operator_answer("unknown_belief", "confirmed")   # no pending shadow conclusion
    assert ar.AutonomousResearchPromotion.get_instance()._state.total_outcomes == 0


def test_no_premature_promotion_zero_authority_held(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    g = ar.AutonomousResearchPromotion.get_instance()
    for _ in range(25):
        g.record_outcome(True)            # 25 matches, 100% accuracy — exceeds outcome + accuracy bars
    assert g.level == 0                    # but hours_in_shadow ~0 < 4h → NOT promoted (earn-don't-declare)
    assert g.is_active() is False
