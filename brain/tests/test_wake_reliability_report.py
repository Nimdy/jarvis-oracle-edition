import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.wake_reliability_report import build_wake_report, render_wake_markdown


def test_build_wake_report_parses_samples() -> None:
    log_text = "\n".join(
        [
            "2026-01-01 00:00:01 [perception.audio_stream] INFO: Wake listen: chunks=150 max_score=0.120 threshold=0.50 speaking=False",
            "2026-01-01 00:00:16 [perception.audio_stream] INFO: Wake listen: chunks=160 max_score=0.490 threshold=0.50 speaking=False",
            "2026-01-01 00:00:47 [perception.audio_stream] INFO: Wake word triggered: score=0.620",
            "2026-01-01 00:00:49 [perception.audio_stream] INFO: Dispatching 2.4s speech (conv=abc12345)",
            "2026-01-01 00:00:50 [jarvis.conversation] INFO: [LATENCY] route_complete=6ms route=STATUS (conv=abc12345)",
        ]
    )

    report = build_wake_report(log_text, window_lines=500, min_samples=2, min_attempts=1)

    assert report["sample_count"] == 2
    assert report["trigger_count"] == 1
    assert report["overall"]["count"] == 2
    assert report["non_speaking"]["count"] == 2
    assert report["trigger_pipeline"]["trigger_to_dispatch_rate"] == 1.0
    assert report["trigger_pipeline"]["trigger_to_route_rate"] == 1.0
    assert report["trigger_pipeline"]["trigger_to_user_handled_rate"] == 1.0
    assert report["trigger_pipeline"]["actionable_route_miss_count"] == 0
    assert report["assessment"]["status"] in {"healthy_interaction", "healthy_interaction_low_activity"}


def test_build_wake_report_empty_window() -> None:
    report = build_wake_report("", window_lines=500, min_samples=10)
    assert report["sample_count"] == 0
    assert report["overall"]["count"] == 0
    assert report["assessment"]["status"] == "insufficient_data"


def test_build_wake_report_detects_insufficient_interaction_samples() -> None:
    lines = [
        "Wake listen: chunks=100 max_score=0.010 threshold=0.50 speaking=False"
        for _ in range(120)
    ]
    report = build_wake_report("\n".join(lines), window_lines=1000, min_samples=80)
    assert report["sample_count"] == 120
    assert report["overall"]["hit_rate"] == 0.0
    assert report["assessment"]["status"] == "insufficient_interaction_samples"


def test_since_last_restart_scope_reduces_old_noise() -> None:
    old_lines = [
        "2026-01-01 00:00:01 [perception.audio_stream] INFO: Wake listen: chunks=100 max_score=0.010 threshold=0.50 speaking=False"
        for _ in range(80)
    ]
    new_lines = [
        "2026-01-01 01:00:00 [consciousness.engine] INFO: Jarvis consciousness awakened",
        "2026-01-01 01:00:15 [perception.audio_stream] INFO: Wake listen: chunks=100 max_score=0.980 threshold=0.50 speaking=False",
        "2026-01-01 01:00:16 [perception.audio_stream] INFO: Wake word triggered: score=0.980",
        "2026-01-01 01:00:17 [perception.audio_stream] INFO: Dispatching 3.1s speech (conv=ff99aa11)",
        "2026-01-01 01:00:18 [jarvis.conversation] INFO: [LATENCY] route_complete=8ms route=STATUS (conv=ff99aa11)",
    ]
    report_all = build_wake_report(
        "\n".join(old_lines + new_lines),
        window_lines=500,
        min_samples=20,
        min_attempts=1,
        since_last_restart=False,
    )
    report_restart = build_wake_report(
        "\n".join(old_lines + new_lines),
        window_lines=500,
        min_samples=20,
        min_attempts=1,
        since_last_restart=True,
    )
    assert report_all["sample_count"] > report_restart["sample_count"]
    assert report_restart["trigger_count"] == 1
    assert report_restart["trigger_pipeline"]["trigger_to_route_rate"] == 1.0


def test_passive_windows_do_not_force_false_failure_with_successful_attempts() -> None:
    lines = []
    for i in range(200):
        lines.append(
            f"2026-01-01 02:00:{i % 60:02d} [perception.audio_stream] INFO: Wake listen: chunks=100 max_score=0.010 threshold=0.50 speaking=False"
        )
    lines += [
        "2026-01-01 02:10:00 [perception.audio_stream] INFO: Wake listen: chunks=120 max_score=0.970 threshold=0.50 speaking=False",
        "2026-01-01 02:10:01 [perception.audio_stream] INFO: Wake word triggered: score=0.970",
        "2026-01-01 02:10:02 [perception.audio_stream] INFO: Dispatching 2.0s speech (conv=a1)",
        "2026-01-01 02:10:03 [jarvis.conversation] INFO: [LATENCY] route_complete=6ms route=STATUS (conv=a1)",
        "2026-01-01 02:12:00 [perception.audio_stream] INFO: Wake listen: chunks=120 max_score=0.990 threshold=0.50 speaking=False",
        "2026-01-01 02:12:01 [perception.audio_stream] INFO: Wake word triggered: score=0.990",
        "2026-01-01 02:12:02 [perception.audio_stream] INFO: Dispatching 1.8s speech (conv=b2)",
        "2026-01-01 02:12:03 [jarvis.conversation] INFO: [LATENCY] route_complete=7ms route=IDENTITY (conv=b2)",
    ]
    report = build_wake_report(
        "\n".join(lines),
        window_lines=10000,
        min_samples=80,
        min_attempts=2,
    )
    assert report["passive_filter"]["passive_rate"] > 0.9
    assert report["trigger_pipeline"]["trigger_to_route_rate"] == 1.0
    assert report["trigger_pipeline"]["trigger_to_user_handled_rate"] == 1.0
    assert report["assessment"]["status"] in {"healthy_interaction", "healthy_interaction_low_activity"}


def test_empty_stt_nonroute_is_classified_benign() -> None:
    log_text = "\n".join(
        [
            "2026-01-01 00:00:01 [perception.audio_stream] INFO: Wake listen: chunks=100 max_score=0.900 threshold=0.50 speaking=False",
            "2026-01-01 00:00:02 [perception.audio_stream] INFO: Wake word triggered: score=0.900",
            "2026-01-01 00:00:03 [perception.audio_stream] INFO: Dispatching 2.0s speech (conv=a1)",
            "2026-01-01 00:00:03 [perception.stt] INFO: STT: 2.0s audio -> '' (0.00s, lang=en p=1.00)",
            "2026-01-01 00:00:03 [jarvis.perception] INFO: STT returned empty (conv=a1)",
            "2026-01-01 00:00:10 [perception.audio_stream] INFO: Wake word triggered: score=0.910",
            "2026-01-01 00:00:11 [perception.audio_stream] INFO: Dispatching 2.1s speech (conv=b2)",
            "2026-01-01 00:00:12 [jarvis.conversation] INFO: [LATENCY] route_complete=7ms route=STATUS (conv=b2)",
        ]
    )
    report = build_wake_report(log_text, window_lines=500, min_samples=1, min_attempts=2)
    pipeline = report["trigger_pipeline"]
    assert pipeline["trigger_to_route_rate"] == 0.5
    assert pipeline["trigger_to_user_handled_rate"] == 1.0
    assert pipeline["benign_nonroute_count"] == 1
    assert pipeline["actionable_route_miss_count"] == 0
    assert report["assessment"]["status"] == "healthy_interaction_benign_nonroutes"


def test_echo_discard_nonroute_is_classified_benign() -> None:
    log_text = "\n".join(
        [
            "2026-01-01 00:00:01 [perception.audio_stream] INFO: Wake word triggered: score=0.870",
            "2026-01-01 00:00:02 [perception.audio_stream] INFO: Dispatching 3.0s speech (conv=e1)",
            "2026-01-01 00:00:03 [jarvis.perception] WARNING: Echo detected (95% similar, threshold=70%, gap=5.2s) — discarding: Good morning. Ready to start the day.",
        ]
    )
    report = build_wake_report(log_text, window_lines=200, min_samples=1, min_attempts=1)
    pipeline = report["trigger_pipeline"]
    assert pipeline["benign_nonroute_count"] == 1
    assert pipeline["actionable_route_miss_count"] == 0
    assert pipeline["trigger_to_user_handled_rate"] == 1.0
    assert report["assessment"]["status"] == "healthy_interaction_benign_nonroutes"
    attempt = pipeline["recent_attempts"][-1]
    assert attempt["attempt_outcome"] == "benign_echo_discard"


def test_render_wake_markdown_includes_sections() -> None:
    report = build_wake_report(
        "Wake listen: chunks=100 max_score=0.50 threshold=0.50 speaking=False",
        window_lines=100,
        min_samples=1,
    )
    md = render_wake_markdown(report)
    assert "# Wake Reliability Report" in md
    assert "## Overall (Raw Always-Listening)" in md
    assert "## Passive Listening Filter" in md
    assert "## Trigger Pipeline (Attempt-Focused)" in md
    assert "Trigger -> user-handled rate (route + benign)" in md
