from __future__ import annotations

import json
import time


def test_weight_room_smoke_allowed_without_distillation_recording(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from synthetic.skill_acquisition_dashboard import SkillAcquisitionWeightRoomRunner

    runner = SkillAcquisitionWeightRoomRunner()
    status = runner.status(engine=None, startup_ts=0.0)

    assert status["gates"]["smoke"]["allowed"] is True
    assert status["gates"]["smoke"]["record_signals"] is False
    assert status["authority"] == "telemetry_only"
    assert status["promotion_eligible"] is False
    assert status["live_influence"] is False


def test_weight_room_training_profiles_block_without_runtime_maturity(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ENABLE_SYNTHETIC_SKILL_ACQUISITION_HEAVY", raising=False)
    from synthetic.skill_acquisition_dashboard import evaluate_gates

    coverage = evaluate_gates("coverage", engine=None, startup_ts=time.time())
    strict = evaluate_gates("strict", engine=None, startup_ts=time.time())

    assert coverage["allowed"] is False
    assert "engine_not_ready" in coverage["blocked_reasons"]
    assert "startup_grace_active" in coverage["blocked_reasons"]
    assert strict["allowed"] is False
    assert "heavy_profile_operator_flag_disabled" in strict["blocked_reasons"]


def test_weight_room_refuses_concurrent_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    import synthetic.skill_acquisition_dashboard as dashboard
    from synthetic.skill_acquisition_dashboard import SkillAcquisitionWeightRoomRunner

    def _slow_run(*_args, progress_callback=None, **_kwargs):
        from synthetic.skill_acquisition_exercise import SkillAcquisitionExerciseStats

        stats = SkillAcquisitionExerciseStats(profile_name="smoke")
        for i in range(3):
            stats.episodes += 1
            if progress_callback:
                progress_callback(stats, i + 1, 3)
            time.sleep(0.05)
        stats.ended_at = time.time()
        return stats

    monkeypatch.setattr(dashboard, "run_skill_acquisition_exercise", _slow_run)
    runner = SkillAcquisitionWeightRoomRunner()

    first = runner.start("smoke", engine=None, startup_ts=0.0, count=3)
    second = runner.start("smoke", engine=None, startup_ts=0.0, count=3)

    assert first["started"] is True
    assert second["started"] is False
    assert "run_already_active" in second["blocked_reasons"]


def test_weight_room_status_reads_reports_without_claiming_authority(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from synthetic import skill_acquisition_dashboard as dashboard
    from synthetic.skill_acquisition_dashboard import SkillAcquisitionWeightRoomRunner

    report_dir = tmp_path / ".jarvis" / "synthetic_exercise" / "skill_acquisition_reports"
    report_dir.mkdir(parents=True)
    (report_dir / "123_coverage.json").write_text(
        json.dumps(
            {
                "profile_name": "coverage",
                "episodes": 12,
                "features_recorded": 12,
                "labels_recorded": 12,
                "scenarios": {"contract_mismatch": 4},
                "outcomes": {"contract_failed": 4},
                "invariant_failures": [],
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "REPORT_DIR", report_dir)

    status = SkillAcquisitionWeightRoomRunner().status(engine=None, startup_ts=0.0)

    assert status["synthetic_episodes_total"] == 12
    assert status["synthetic_features_total"] == 12
    assert status["synthetic_labels_total"] == 12
    assert status["latest_report"]["profile_name"] == "coverage"
    assert status["can_verify_skills"] is False
    assert status["can_promote_plugins"] is False
    assert status["can_unlock_capabilities"] is False

