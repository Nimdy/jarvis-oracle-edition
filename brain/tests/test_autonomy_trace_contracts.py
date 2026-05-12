from pathlib import Path


_SRC_PATH = Path(__file__).resolve().parent.parent / "autonomy" / "orchestrator.py"


def _source() -> str:
    return _SRC_PATH.read_text(encoding="utf-8")


def test_intent_trace_helpers_exist() -> None:
    src = _source()
    assert "def _intent_trace_fields(intent: Any) -> dict[str, str]:" in src
    assert "def _intent_evidence_refs(cls, intent: Any) -> list[dict[str, str]]:" in src
    assert "\"goal_id\": str(getattr(intent, \"goal_id\", \"\"))" in src
    assert "\"task_id\": str(getattr(intent, \"task_id\", \"\"))" in src
    assert "\"golden_trace_id\": str(getattr(intent, \"golden_trace_id\", \"\"))" in src


def test_autonomy_ledger_records_use_trace_data_and_evidence_refs() -> None:
    src = _source()
    assert "data={" in src
    assert "**_intent_trace" in src
    assert "evidence_refs=self._intent_evidence_refs(intent)" in src
    assert "evidence_refs=_intent_evidence_refs" in src


def test_autonomy_metadata_and_outcomes_include_goal_task_golden_fields() -> None:
    src = _source()
    assert "\"goal_id\": intent.goal_id or \"\"" in src
    assert "\"task_id\": intent.task_id or \"\"" in src
    assert "\"golden_trace_id\": intent.golden_trace_id or \"\"" in src
    assert "goal_id=goal_id" in src
    assert "task_id=task_id" in src
    assert "golden_trace_id=golden_trace_id" in src

