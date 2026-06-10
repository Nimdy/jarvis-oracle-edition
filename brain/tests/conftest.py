"""Shared pytest fixtures for the brain test-suite.

Keep the counterfactual engine's persisted state out of the real ~/.jarvis.
The Layer-9 reflective audit (test_reflective_audit.py::run_audit) now invokes
the counterfactual engine, which persists to STATE_PATH. Without isolation that
write lands in the live runtime's ~/.jarvis. Redirect it to a tmp file for
every test. Wrapped defensively so a missing import can never break collection.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_counterfactual_state(tmp_path, monkeypatch):
    try:
        import epistemic.counterfactual.engine as cfmod
        monkeypatch.setattr(cfmod, "STATE_PATH", str(tmp_path / "cf_state.json"), raising=False)
        cfmod.CounterfactualEngine._instance = None
        yield
        cfmod.CounterfactualEngine._instance = None
    except Exception:
        yield


@pytest.fixture(autouse=True)
def _isolate_lidar_extrinsic(tmp_path, monkeypatch):
    """Keep load_extrinsic() from reading the live rig's ~/.jarvis/lidar_extrinsic.json.

    On a CALIBRATED machine (David's brain ships ty_m=1.092) the 'no mount → identity'
    tests would otherwise fail because the per-instance config is present. Point the
    module constant at a non-existent tmp path so tests see the product default.
    """
    try:
        import cognition.lidar_calibration as lc
        monkeypatch.setattr(lc, "_EXTRINSIC_FILE", str(tmp_path / "no_lidar_extrinsic.json"), raising=False)
        for k in ("JARVIS_LIDAR_YAW_RAD", "JARVIS_LIDAR_TX_M", "JARVIS_LIDAR_MOUNT_HEIGHT_M", "JARVIS_LIDAR_TZ_M"):
            monkeypatch.delenv(k, raising=False)
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True)
def _isolate_belief_store(tmp_path, monkeypatch):
    """Keep test-created beliefs/edges OUT of the live ~/.jarvis (GitHub #50).

    A stray test-belief (`bel_test_98 "test is discard"`) leaked into the live
    `~/.jarvis/beliefs.jsonl` and surfaced in the grounding queue. The BeliefStore
    now resolves its path at call-time, so redirecting the module constants here
    sends all test belief/tension/edge writes to a tmp dir. Defensive — a missing
    import or signature change can never break collection.
    """
    try:
        import epistemic.belief_record as br
        monkeypatch.setattr(br, "_BELIEFS_FILE", str(tmp_path / "beliefs.jsonl"), raising=False)
        monkeypatch.setattr(br, "_TENSIONS_FILE", str(tmp_path / "tensions.jsonl"), raising=False)
    except Exception:
        pass
    try:
        import epistemic.belief_graph.edges as eg
        monkeypatch.setattr(eg, "_EDGES_FILE", str(tmp_path / "belief_edges.jsonl"), raising=False)
    except Exception:
        pass
    # drop any cached engine/store so the next access rebuilds against the tmp paths
    try:
        import epistemic.contradiction_engine as ce
        if hasattr(ce, "ContradictionEngine"):
            for attr in ("_instance", "_singleton"):
                if hasattr(ce.ContradictionEngine, attr):
                    setattr(ce.ContradictionEngine, attr, None)
    except Exception:
        pass
    # grounding queue persists to ~/.jarvis/grounding_queue.json too — isolate it +
    # reset the singleton so test answers/enqueues don't leak or accumulate across tests
    try:
        import autonomy.grounding_queue as gq
        monkeypatch.setattr(gq, "QUEUE_PATH", str(tmp_path / "grounding_queue.json"), raising=False)
        if hasattr(gq, "GroundingQueue") and hasattr(gq.GroundingQueue, "reset_instance"):
            gq.GroundingQueue.reset_instance()
    except Exception:
        pass
    yield
