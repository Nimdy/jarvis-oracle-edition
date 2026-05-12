"""Regression tests for intervention runner backlog and persistence hygiene."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from autonomy.intervention_runner import InterventionRunner
from autonomy.interventions import CandidateIntervention


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_uses_latest_record_per_intervention_id(monkeypatch, tmp_path):
    from autonomy import intervention_runner as ir_mod

    file_path = tmp_path / "interventions.jsonl"
    monkeypatch.setattr(ir_mod, "_PERSISTENCE_PATH", str(file_path))

    now = time.time()
    rows = [
        {
            "intervention_id": "iv_same",
            "change_type": "routing_rule",
            "target_subsystem": "routing",
            "status": "proposed",
            "created_at": now - 20,
        },
        {
            "intervention_id": "iv_same",
            "change_type": "routing_rule",
            "target_subsystem": "routing",
            "status": "measured",
            "created_at": now - 20,
            "shadow_start": now - 10,
            "shadow_end": now - 5,
        },
        {
            "intervention_id": "iv_stale",
            "change_type": "prompt_frame",
            "target_subsystem": "conversation",
            "status": "proposed",
            "created_at": now - (ir_mod._MAX_PROPOSED_AGE_S + 60),
        },
    ]
    _write_jsonl(file_path, rows)

    runner = InterventionRunner()
    runner.load()
    stats = runner.get_stats()
    assert stats["proposed_count"] == 0
    assert stats["shadow_active_count"] == 0
    completed = runner.get_recent(10)
    statuses = {row["intervention_id"]: row["status"] for row in completed}
    assert statuses["iv_same"] == "measured"
    assert statuses["iv_stale"] == "expired"


def test_status_transitions_are_persisted(monkeypatch, tmp_path):
    from autonomy import intervention_runner as ir_mod

    file_path = tmp_path / "interventions.jsonl"
    monkeypatch.setattr(ir_mod, "_PERSISTENCE_PATH", str(file_path))

    runner = InterventionRunner()
    iv = CandidateIntervention(
        intervention_id="iv_flow",
        change_type="routing_rule",
        target_subsystem="routing",
        trigger_deficit="metric",
        status="proposed",
    )
    assert runner.propose(iv)
    assert runner.activate_shadow("iv_flow")

    # Simulate elapsed shadow window and positive result.
    shadow_iv = runner._shadow_active[0]
    shadow_iv.shadow_end = time.time() - 1
    measured = runner.check_shadow_results()
    assert len(measured) == 1
    measured[0].measured_delta = 0.05
    assert runner.promote("iv_flow")

    rows = [json.loads(line) for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    statuses = [r["status"] for r in rows if r.get("intervention_id") == "iv_flow"]
    assert statuses == ["proposed", "shadow", "measured", "promoted"]


def test_load_trims_proposed_backlog_to_global_cap(monkeypatch, tmp_path):
    from autonomy import intervention_runner as ir_mod

    file_path = tmp_path / "interventions.jsonl"
    monkeypatch.setattr(ir_mod, "_PERSISTENCE_PATH", str(file_path))

    now = time.time()
    rows = []
    for i in range(ir_mod._MAX_UNRESOLVED_GLOBAL + 5):
        rows.append(
            {
                "intervention_id": f"iv_{i}",
                "change_type": "prompt_frame",
                "target_subsystem": "conversation",
                "status": "proposed",
                "created_at": now - i,
            }
        )
    _write_jsonl(file_path, rows)

    runner = InterventionRunner()
    runner.load()
    stats = runner.get_stats()
    assert stats["proposed_count"] == ir_mod._MAX_UNRESOLVED_GLOBAL
    recent = runner.get_recent(50)
    expired = [r for r in recent if r.get("status") == "expired"]
    assert len(expired) >= 1
