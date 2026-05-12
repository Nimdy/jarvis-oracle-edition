import unittest

from jarvis_eval.validation_pack import (
    build_runtime_validation_report,
    build_validation_pack,
    render_validation_markdown,
)


def _contract(contract_id: str, status: str, evidence: str, ever: bool = False, last: str = ""):
    return {
        "contract_id": contract_id,
        "status": status,
        "evidence": evidence,
        "ever_passed": ever,
        "last_pass_evidence": last,
    }


class ValidationPackTests(unittest.TestCase):
    def _base_pvl(self):
        return {
            "coverage_pct": 90.0,
            "applicable_contracts": 20,
            "ever_passing_contracts": 18,
            "groups": [
                {
                    "group_id": "hemisphere_distillation",
                    "contracts": [
                        _contract("hemisphere_ready", "pass", "total_networks=8", True, "total_networks=8"),
                    ],
                },
                {
                    "group_id": "skill_learning",
                    "contracts": [
                        _contract("skill_registered", "pass", "total=14", True, "total=14"),
                        _contract("learning_job_started", "pass", "total_count=1", True, "total_count=1"),
                        _contract("job_phase_advanced", "pass", "phase_transition_count=4", True, "phase_transition_count=4"),
                        _contract("skill_learning_completed", "pass", "completed_count=1", True, "completed_count=1"),
                    ],
                },
                {
                    "group_id": "consciousness_tick",
                    "contracts": [
                        _contract("evolution_analyzed", "pass", "emergent_behavior_count=100", True, "emergent_behavior_count=100"),
                    ],
                },
            ],
        }

    def _base_maturity(self):
        return {
            "active_gates": 30,
            "total_gates": 35,
            "ever_active_gates": 32,
            "categories": [
                {
                    "id": "epistemic",
                    "label": "Epistemic",
                    "gates": [
                        {
                            "id": "soul_integrity",
                            "label": "Soul Integrity Index",
                            "status": "active",
                            "display": "0.890 / 0.870",
                            "current": 0.89,
                            "threshold": 0.87,
                            "ever_met": True,
                            "best_current": 0.91,
                        }
                    ],
                }
            ],
        }

    def _base_language(self):
        classes = [
            "self_status",
            "self_introspection",
            "recent_learning",
            "recent_research",
            "memory_recall",
            "identity_answer",
            "capability_status",
        ]
        return {
            "corpus_total_examples": 220,
            "corpus_response_classes": {c: 40 for c in classes},
            "corpus_route_class_pairs": {
                "status|self_status": 40,
                "introspection|self_introspection": 40,
                "introspection|recent_learning": 40,
                "introspection|recent_research": 40,
                "memory|memory_recall": 40,
                "identity|identity_answer": 40,
                "introspection|capability_status": 40,
            },
            "quality_total_events": 64,
            "gate_color": "green",
            "gate_color_code": 2,
            "gate_scores_by_class": {c: {"color": "green", "scores": {"sample_count": 1.0}} for c in classes},
            "promotion_summary": {
                c: {
                    "level": "canary" if i == 0 else "shadow",
                    "color": "green",
                    "consecutive_green": 2,
                    "consecutive_red": 0,
                    "promotion_history_len": 1 if i == 0 else 0,
                    "last_transition_at": 123.0 if i == 0 else 0.0,
                }
                for i, c in enumerate(classes)
            },
            "promotion_shadow_count": 6,
            "promotion_canary_count": 1,
            "promotion_live_count": 0,
            "promotion_red_classes": 0,
            "promotion_total_evaluations": 21,
            "promotion_max_consecutive_red": 0,
            "runtime_bridge_enabled": False,
            "runtime_rollout_mode": "off",
            "runtime_guard_total": 0,
            "runtime_live_total": 0,
            "runtime_blocked_by_guard_count": 0,
            "runtime_unpromoted_live_attempts": 0,
            "runtime_live_red_classes": 0,
            "phase_c": {"student_available": True},
        }

    def _base_autonomy(self):
        return {
            "completed_total": 121,
            "delta_tracker": {"total_measured": 275, "total_improved": 174},
            "policy_memory": {
                "total_wins": 95,
                "overall_win_rate": 0.693,
                "avoid_patterns": [
                    {"tags": ["conversation_quality", "friction"], "total": 3, "win_rate": 0.0}
                ],
            },
        }

    def test_validation_pack_ready_when_all_checks_pass(self) -> None:
        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            self._base_language(),
            self._base_autonomy(),
        )
        self.assertEqual(pack["status"], "ready")
        self.assertTrue(pack["ready_for_next_items"])
        self.assertTrue(pack["ready_for_continuation"])
        self.assertEqual(pack["continuation_mode"], "current_green")
        self.assertEqual(pack["checks_regressed"], 0)

    def test_validation_pack_marks_regressed_critical_contract(self) -> None:
        pvl = self._base_pvl()
        pvl["groups"][0]["contracts"][0] = _contract(
            "hemisphere_ready",
            "fail",
            "total_networks below threshold (1.0)",
            True,
            "total_networks=8",
        )

        pack = build_validation_pack(
            pvl,
            self._base_maturity(),
            self._base_language(),
            self._base_autonomy(),
        )

        self.assertEqual(pack["status"], "blocked")
        self.assertFalse(pack["ready_for_next_items"])
        self.assertTrue(pack["ready_for_continuation"])
        self.assertEqual(pack["continuation_mode"], "historically_proven_recovering")
        self.assertIn("hemisphere_ready", pack["regressed_check_ids"])
        self.assertIn("hemisphere_ready", pack["blocked_check_ids"])
        hemi_check = next(c for c in pack["checks"] if c["id"] == "hemisphere_ready")
        self.assertFalse(hemi_check["current_ok"])
        self.assertTrue(hemi_check["ever_ok"])
        self.assertIn("total_networks=8", hemi_check["ever_detail"])

    def test_validation_pack_blocks_on_released_without_validation(self) -> None:
        release_validation = {
            "released_total": 12,
            "released_validated": 11,
            "released_without_validation": 1,
            "validation_failed": 1,
        }
        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            self._base_language(),
            self._base_autonomy(),
            release_validation,
        )
        self.assertEqual(pack["status"], "blocked")
        self.assertIn("output_release_validation", pack["blocked_check_ids"])
        check = next(c for c in pack["checks"] if c["id"] == "output_release_validation")
        self.assertTrue(check["critical"])
        self.assertFalse(check["current_ok"])

    def test_validation_pack_blocks_continuation_when_critical_never_met(self) -> None:
        pvl = self._base_pvl()
        pvl["groups"][0]["contracts"][0] = _contract(
            "hemisphere_ready",
            "fail",
            "total_networks below threshold (1.0)",
            False,
            "",
        )

        pack = build_validation_pack(
            pvl,
            self._base_maturity(),
            self._base_language(),
            self._base_autonomy(),
        )

        self.assertEqual(pack["status"], "blocked")
        self.assertFalse(pack["ready_for_next_items"])
        self.assertFalse(pack["ready_for_continuation"])
        self.assertEqual(pack["continuation_mode"], "not_yet_proven")
        self.assertIn("hemisphere_ready", pack["ever_blocked_check_ids"])

    def test_runtime_report_uses_existing_validation_payload(self) -> None:
        snapshot = {
            "_ts": 123.0,
            "eval": {
                "validation_pack": {
                    "status": "caution",
                    "ready_for_next_items": True,
                    "ready_for_continuation": True,
                    "continuation_mode": "historically_proven_recovering",
                    "continuation_action": "continue with monitoring",
                    "next_action": "watch regressions",
                    "checks": [{"id": "x", "label": "X", "current_ok": True, "ever_ok": True, "critical": False, "current_detail": "ok", "ever_detail": "ok"}],
                    "checks_total": 1,
                    "checks_passing": 1,
                    "checks_ever_met": 1,
                    "checks_regressed": 0,
                    "critical_total": 0,
                    "critical_passing": 0,
                },
                "pvl": {"coverage_pct": 88.0},
                "maturity_tracker": {"active_gates": 1, "total_gates": 1, "ever_active_gates": 1},
            },
        }
        report = build_runtime_validation_report(snapshot)
        self.assertEqual(report["status"], "caution")
        self.assertTrue(report["ready_for_next_items"])
        self.assertTrue(report["ready_for_continuation"])
        self.assertEqual(report["continuation_mode"], "historically_proven_recovering")
        self.assertEqual(report["validation"]["checks_total"], 1)

    def test_runtime_report_backfills_continuation_fields_for_legacy_payload(self) -> None:
        snapshot = {
            "_ts": 124.0,
            "eval": {
                "validation_pack": {
                    "status": "ready",
                    "ready_for_next_items": True,
                    "next_action": "all good",
                    "checks": [{"id": "legacy", "label": "Legacy", "current_ok": True, "ever_ok": True}],
                    "checks_total": 1,
                    "checks_passing": 1,
                    "checks_ever_met": 1,
                    "checks_regressed": 0,
                    "critical_total": 0,
                    "critical_passing": 0,
                },
                "pvl": {"coverage_pct": 90.0},
                "maturity_tracker": {"active_gates": 1, "total_gates": 1, "ever_active_gates": 1},
            },
        }
        report = build_runtime_validation_report(snapshot)
        self.assertTrue(report["ready_for_next_items"])
        self.assertTrue(report["ready_for_continuation"])
        self.assertEqual(report["continuation_mode"], "current_green")
        self.assertEqual(report["continuation_action"], "all good")

    def test_validation_pack_detects_language_promotion_regression(self) -> None:
        language = self._base_language()
        language["promotion_canary_count"] = 0
        language["promotion_live_count"] = 0

        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            language,
            self._base_autonomy(),
        )

        self.assertEqual(pack["status"], "caution")
        self.assertIn("language_promotion_progress", pack["regressed_check_ids"])
        check = next(c for c in pack["checks"] if c["id"] == "language_promotion_progress")
        self.assertFalse(check["current_ok"])
        self.assertTrue(check["ever_ok"])

    def test_validation_pack_surfaces_route_class_baseline_gaps(self) -> None:
        language = self._base_language()
        language["corpus_route_class_pairs"]["introspection|capability_status"] = 0

        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            language,
            self._base_autonomy(),
        )

        contract = next(c for c in pack["checks"] if c["id"] == "language_route_class_contract")
        self.assertFalse(contract["current_ok"])
        self.assertIn("INTROSPECTION->capability_status", contract["current_detail"])

        rows = pack.get("language_route_class_baselines", [])
        self.assertEqual(len(rows), 4)
        cap = next(r for r in rows if r["response_class"] == "capability_status")
        self.assertFalse(cap["current_ok"])
        self.assertEqual(cap["count"], 0)

    def test_validation_pack_surfaces_language_evidence_target_gaps(self) -> None:
        language = self._base_language()
        targets = ("recent_learning", "recent_research", "identity_answer", "capability_status")
        for rc in targets:
            language["corpus_response_classes"][rc] = 6
            language["gate_scores_by_class"][rc] = {
                "color": "red",
                "gate_reason": "insufficient_samples",
                "scores": {"sample_count": 0.2},
            }
            language["promotion_summary"][rc]["color"] = "red"
            language["promotion_summary"][rc]["gate_reason"] = "insufficient_samples"

        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            language,
            self._base_autonomy(),
        )

        target_rows = pack.get("language_evidence_targets", [])
        self.assertEqual(len(target_rows), 4)
        rl = next(r for r in target_rows if r["response_class"] == "recent_learning")
        self.assertFalse(rl["current_ok"])
        self.assertEqual(rl["gap"], 24)
        self.assertEqual(rl["gate_reason"], "insufficient_samples")

        target_check = next(c for c in pack["checks"] if c["id"] == "language_evidence_recent_learning")
        self.assertFalse(target_check["current_ok"])
        self.assertIn("gap=24", target_check["current_detail"])

    def test_language_evidence_prefers_live_gate_reason_over_promotion_reason(self) -> None:
        language = self._base_language()
        language["gate_scores_by_class"]["capability_status"] = {
            "color": "green",
            "gate_reason": "ok",
            "scores": {"sample_count": 1.0},
        }
        language["promotion_summary"]["capability_status"]["color"] = "red"
        language["promotion_summary"]["capability_status"]["gate_reason"] = "hallucination_ceiling"

        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            language,
            self._base_autonomy(),
        )

        row = next(r for r in pack["language_evidence_targets"] if r["response_class"] == "capability_status")
        self.assertTrue(row["current_ok"])
        self.assertEqual(row["color"], "green")
        self.assertEqual(row["gate_reason"], "ok")

    def test_language_evidence_uses_live_red_reason_when_present(self) -> None:
        language = self._base_language()
        language["gate_scores_by_class"]["capability_status"] = {
            "color": "red",
            "gate_reason": "hallucination_ceiling",
            "scores": {"sample_count": 1.0},
        }
        language["promotion_summary"]["capability_status"]["color"] = "green"
        language["promotion_summary"]["capability_status"]["gate_reason"] = "ok"

        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            language,
            self._base_autonomy(),
        )

        row = next(r for r in pack["language_evidence_targets"] if r["response_class"] == "capability_status")
        self.assertFalse(row["current_ok"])
        self.assertEqual(row["color"], "red")
        self.assertEqual(row["gate_reason"], "hallucination_ceiling")

    def test_language_evidence_normalizes_stale_live_reason_when_color_not_red(self) -> None:
        language = self._base_language()
        language["gate_scores_by_class"]["capability_status"] = {
            "color": "green",
            "gate_reason": "hallucination_ceiling",
            "scores": {"sample_count": 1.0},
        }
        language["promotion_summary"]["capability_status"]["color"] = "red"
        language["promotion_summary"]["capability_status"]["gate_reason"] = "hallucination_ceiling"

        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            language,
            self._base_autonomy(),
        )

        row = next(r for r in pack["language_evidence_targets"] if r["response_class"] == "capability_status")
        self.assertTrue(row["current_ok"])
        self.assertEqual(row["color"], "green")
        self.assertEqual(row["gate_reason"], "ok")

    def test_runtime_guardrails_block_when_rollout_enabled_and_unpromoted_live_attempts(self) -> None:
        language = self._base_language()
        language["runtime_bridge_enabled"] = True
        language["runtime_rollout_mode"] = "full"
        language["runtime_guard_total"] = 8
        language["runtime_unpromoted_live_attempts"] = 2

        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            language,
            self._base_autonomy(),
        )

        self.assertEqual(pack["status"], "blocked")
        self.assertIn("language_runtime_guardrails", pack["blocked_check_ids"])
        check = next(c for c in pack["checks"] if c["id"] == "language_runtime_guardrails")
        self.assertTrue(check["critical"])
        self.assertFalse(check["current_ok"])

    def test_runtime_guardrails_not_critical_when_rollout_disabled(self) -> None:
        language = self._base_language()
        language["runtime_bridge_enabled"] = False
        language["runtime_rollout_mode"] = "off"
        language["runtime_unpromoted_live_attempts"] = 3

        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            language,
            self._base_autonomy(),
        )

        self.assertNotEqual(pack["status"], "blocked")
        check = next(c for c in pack["checks"] if c["id"] == "language_runtime_guardrails")
        self.assertFalse(check["critical"])
        self.assertFalse(check["current_ok"])

    def test_markdown_contains_checks_table(self) -> None:
        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            self._base_language(),
            self._base_autonomy(),
        )
        report = {
            "generated_at": "2026-01-01T00:00:00Z",
            "snapshot_ts": 1.0,
            "status": pack["status"],
            "ready_for_next_items": pack["ready_for_next_items"],
            "next_action": pack["next_action"],
            "validation": pack,
            "summary": {
                "pvl_coverage_pct": 90.0,
                "pvl_passing_contracts": 10,
                "pvl_failing_contracts": 0,
                "pvl_awaiting_contracts": 0,
                "pvl_ever_passing_contracts": 10,
                "maturity_active_gates": 30,
                "maturity_total_gates": 35,
                "maturity_ever_active_gates": 32,
            },
        }
        md = render_validation_markdown(report)
        self.assertIn("# Runtime Validation Pack", md)
        self.assertIn("| Check | Current | Ever Met | Critical |", md)
        self.assertIn("## Language Evidence Targets", md)
        self.assertIn("Hemisphere Ready", md)

    def test_phase5_proof_chain_present_and_passing(self) -> None:
        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            self._base_language(),
            self._base_autonomy(),
        )
        proof = next(c for c in pack["checks"] if c["id"] == "phase5_proof_chain")
        self.assertTrue(proof["current_ok"])
        self.assertTrue(proof["ever_ok"])

    # -- Phase 6.5: L3 escalation validation checks -----------------------

    def _autonomy_with_l3(
        self,
        *,
        current_ok: bool = False,
        prior_attested_ok: bool = False,
        attestation_strength: str = "none",
        live_level: int = 1,
        pending: list | None = None,
        recent_lifecycle: list | None = None,
        wins: int = 0,
        win_rate: float = 0.0,
        reason: str = "",
    ) -> dict:
        base = self._base_autonomy()
        base["l3"] = {
            "available": True,
            "live_autonomy_level": live_level,
            "current_ok": current_ok,
            "current_detail": {
                "wins": wins,
                "win_rate": win_rate,
                "recent_regressions": 0,
                "reason": reason,
            },
            "prior_attested_ok": prior_attested_ok,
            "attestation_strength": attestation_strength,
            "request_ok": bool(current_ok or prior_attested_ok),
            "approval_required": live_level < 3,
            "activation_ok": live_level >= 3,
            "pending": pending or [],
            "recent_lifecycle": recent_lifecycle or [],
            "policy": {},
        }
        return base

    def test_l3_requestable_via_current_live(self) -> None:
        autonomy = self._autonomy_with_l3(
            current_ok=True,
            prior_attested_ok=False,
            live_level=2,
            wins=30,
            win_rate=0.71,
            reason="Earned: 30/25 wins, 71%/50% rate, 0 regressions",
        )
        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            self._base_language(),
            autonomy,
        )
        check = next(c for c in pack["checks"] if c["id"] == "l3_escalation_requestable")
        self.assertTrue(check["current_ok"])
        self.assertTrue(check["ever_ok"])
        # Attestation remains a separate field even when current_ok drives the check.
        self.assertIn("prior_attested_ok", check)
        self.assertFalse(check["prior_attested_ok"])
        self.assertEqual(check["attestation_strength"], "none")
        self.assertIn("no accepted attestation", check["prior_attested_detail"])

    def test_l3_requestable_via_prior_attestation(self) -> None:
        autonomy = self._autonomy_with_l3(
            current_ok=False,
            prior_attested_ok=True,
            attestation_strength="verified",
            live_level=1,
        )
        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            self._base_language(),
            autonomy,
        )
        check = next(c for c in pack["checks"] if c["id"] == "l3_escalation_requestable")
        self.assertTrue(check["current_ok"])  # request_ok-driven
        self.assertTrue(check["ever_ok"])
        self.assertTrue(check["prior_attested_ok"])
        self.assertEqual(check["attestation_strength"], "verified")

    def test_l3_attestation_strength_archived_missing_distinguished(self) -> None:
        autonomy = self._autonomy_with_l3(
            current_ok=False,
            prior_attested_ok=True,
            attestation_strength="archived_missing",
            live_level=1,
        )
        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            self._base_language(),
            autonomy,
        )
        check = next(c for c in pack["checks"] if c["id"] == "l3_escalation_requestable")
        self.assertTrue(check["prior_attested_ok"])
        self.assertEqual(check["attestation_strength"], "archived_missing")

    def test_l3_not_requestable_when_neither_current_nor_prior(self) -> None:
        autonomy = self._autonomy_with_l3(
            current_ok=False,
            prior_attested_ok=False,
            live_level=1,
            wins=0,
            win_rate=0.0,
            reason="Need 25 wins (have 0) and 50% win rate",
        )
        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            self._base_language(),
            autonomy,
        )
        check = next(c for c in pack["checks"] if c["id"] == "l3_escalation_requestable")
        self.assertFalse(check["current_ok"])
        self.assertFalse(check["ever_ok"])
        self.assertFalse(check["prior_attested_ok"])
        self.assertEqual(check["attestation_strength"], "none")

    def test_l3_ever_ok_is_not_backfilled_from_attestation_into_standard_field(
        self,
    ) -> None:
        """Regression guard: ``prior_attested_ok`` must be emitted as its
        own field. It must NOT be smuggled into ``ever_ok`` while leaving
        attestation fields absent — that would let consumers mistake
        attestation for auto-observed ever-met state."""
        autonomy = self._autonomy_with_l3(
            current_ok=False,
            prior_attested_ok=True,
            attestation_strength="verified",
            live_level=1,
        )
        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            self._base_language(),
            autonomy,
        )
        check = next(c for c in pack["checks"] if c["id"] == "l3_escalation_requestable")
        # The attestation boundary: a check that reads from the ledger
        # MUST surface prior_attested_ok / attestation_strength so the UI
        # can render a distinct badge. If those keys disappear, a future
        # refactor likely collapsed attestation into ever_ok.
        self.assertIn("prior_attested_ok", check)
        self.assertIn("attestation_strength", check)
        self.assertEqual(check["attestation_strength"], "verified")
        self.assertTrue(check["prior_attested_ok"])

    def test_l3_lifecycle_ever_counts_terminals(self) -> None:
        autonomy = self._autonomy_with_l3(
            current_ok=True,
            live_level=3,
            pending=[{"id": "esc_pending_1", "metric": "reasoning_coherence", "severity": "high"}],
            recent_lifecycle=[
                {"id": "esc_a", "status": "approved", "outcome": "clean"},
                {"id": "esc_b", "status": "rolled_back", "outcome": "rolled_back"},
                {"id": "esc_c", "status": "parked", "outcome": "parked"},
            ],
        )
        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            self._base_language(),
            autonomy,
        )
        lifecycle = next(c for c in pack["checks"] if c["id"] == "l3_escalation_lifecycle_ever")
        self.assertTrue(lifecycle["current_ok"])
        self.assertTrue(lifecycle["ever_ok"])
        self.assertIn("approved=1", lifecycle["current_detail"])
        self.assertIn("rolled_back=1", lifecycle["current_detail"])
        self.assertIn("parked=1", lifecycle["current_detail"])
        self.assertIn("pending=1", lifecycle["current_detail"])

    def test_l3_lifecycle_ever_false_on_fresh_brain(self) -> None:
        autonomy = self._autonomy_with_l3(current_ok=False, live_level=1)
        pack = build_validation_pack(
            self._base_pvl(),
            self._base_maturity(),
            self._base_language(),
            autonomy,
        )
        lifecycle = next(c for c in pack["checks"] if c["id"] == "l3_escalation_lifecycle_ever")
        self.assertFalse(lifecycle["current_ok"])
        self.assertFalse(lifecycle["ever_ok"])
        self.assertIn("no escalation has reached a terminal status", lifecycle["ever_detail"])


if __name__ == "__main__":
    unittest.main()
