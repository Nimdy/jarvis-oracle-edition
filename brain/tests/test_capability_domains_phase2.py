"""Matrix v2 Phase 2 — document ingest + topic-scoped recall (the tracer core).

Proves: ingest into an isolated store; recall is domain-scoped (no leakage to
sibling domains or core); topic-trigger routes to the right domain; no-match
returns None (no confabulation); deletion still clean.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from cognition.capability_domains import (
    CapabilityDomainRegistry,
    ingest_text,
    recall,
    recall_answer,
    chunk_text,
)

ARM_DOC = (
    "The xArm6 robot arm has six degrees of freedom. Its maximum reach is 700 "
    "millimeters and its payload is 5 kilograms.\n\n"
    "The arm communicates over Modbus TCP. Each joint has a servo with a torque "
    "limit. The gripper is pneumatic.\n\n"
    "Calibration requires homing each joint to its zero position before operation."
)
MEDS_DOC = (
    "Metformin is taken twice daily with meals to manage blood glucose.\n\n"
    "Lisinopril is taken once each morning for blood pressure. Do not skip doses.\n\n"
    "Vitamin D is a weekly supplement taken on Sundays."
)


def _reg():
    return CapabilityDomainRegistry(root=Path(tempfile.mkdtemp()) / "domains")


def test_chunk_text_splits_paragraphs():
    chunks = chunk_text(ARM_DOC)
    assert len(chunks) >= 3
    assert all(c.strip() for c in chunks)


def test_ingest_then_recall_from_isolated_store():
    r = _reg()
    arm = r.create("Robot Arm xArm6", kind="physical")
    n = ingest_text(r, arm, "xarm6_manual", ARM_DOC)
    assert n >= 3
    # tallies updated on the domain record
    arm = r.get(arm.domain_id)
    assert arm.chunk_count == n and arm.source_count == 1
    assert arm.provenance["ingested"] == n
    assert arm.status == "ingesting"

    hits = recall(arm, "what is the reach of the arm?", k=3)
    assert hits, "recall should find the reach chunk"
    assert any("700" in h["text"] for h in hits)


def test_recall_is_domain_scoped_no_leakage():
    r = _reg()
    arm = r.create("Robot Arm", kind="physical")
    meds = r.create("Medication Regimen", kind="document")
    ingest_text(r, arm, "arm", ARM_DOC)
    ingest_text(r, meds, "meds", MEDS_DOC)

    # recall from the meds domain must NOT surface arm content
    meds_hits = recall(meds, "metformin glucose", k=5)
    assert meds_hits and all("xArm6" not in h["text"] and "servo" not in h["text"]
                             for h in meds_hits)
    # and the arm domain must not surface meds content
    arm_hits = recall(arm, "modbus servo torque", k=5)
    assert arm_hits and all("Metformin" not in h["text"] for h in arm_hits)


def test_topic_trigger_routes_to_right_domain():
    r = _reg()
    arm = r.create("Robot Arm xArm6", kind="physical")
    meds = r.create("Medication Regimen", kind="document")
    ingest_text(r, arm, "arm", ARM_DOC)
    ingest_text(r, meds, "meds", MEDS_DOC)

    ans = recall_answer(r, "tell me about the xarm6 servo torque and reach")
    assert ans is not None
    assert ans["domain_id"] == arm.domain_id
    assert ans["claim_scope"] == "know_about"   # never "can_do"
    assert ans["chunks"]


def test_no_match_returns_none_no_confabulation():
    r = _reg()
    arm = r.create("Robot Arm", kind="physical")
    ingest_text(r, arm, "arm", ARM_DOC)
    # an unrelated query matches no domain -> None (caller falls back, no made-up answer)
    assert recall_answer(r, "what is the capital of France?") is None


def test_deletion_clears_domain_knowledge():
    r = _reg()
    arm = r.create("Robot Arm", kind="physical")
    meds = r.create("Medication Regimen", kind="document")
    ingest_text(r, arm, "arm", ARM_DOC)
    ingest_text(r, meds, "meds", MEDS_DOC)

    assert r.delete(arm.domain_id) is True
    # arm knowledge is gone (store file removed with the domain dir)...
    assert r.get(arm.domain_id) is None
    assert not Path(arm.root_dir).exists()
    # ...meds domain still recalls fine (clean ablation, no collateral)
    meds = r.get(meds.domain_id)
    assert recall(meds, "lisinopril blood pressure", k=3)
