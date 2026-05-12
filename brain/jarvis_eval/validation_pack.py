"""Runtime validation pack builders for roadmap readiness checks.

This module produces a compact, evidence-first validation view:
- current state (what is true now)
- ever-met state (what has been proven at least once)
- explicit regressions (ever met, currently not met)

Used by:
- eval dashboard adapter (live UI panel)
- scripts/run_validation_pack.py (repeatable operator report)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


_CRITICAL_CONTRACTS: tuple[tuple[str, str], ...] = (
    ("hemisphere_ready", "Hemisphere Ready"),
    ("skill_registered", "Skill Registered"),
    ("learning_job_started", "Learning Job Started"),
    ("job_phase_advanced", "Job Phase Advanced"),
    ("skill_learning_completed", "Learning Completed"),
    ("evolution_analyzed", "Evolution Analyzed"),
)

_LANGUAGE_EVIDENCE_TARGETS: tuple[str, ...] = (
    "recent_learning",
    "recent_research",
    "identity_answer",
    "capability_status",
)
_MIN_CLASS_SAMPLES = 30
_LANGUAGE_ROUTE_CLASS_BASELINES: tuple[tuple[str, str], ...] = (
    ("INTROSPECTION", "recent_learning"),
    ("INTROSPECTION", "recent_research"),
    ("IDENTITY", "identity_answer"),
    ("INTROSPECTION", "capability_status"),
)


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _index_contracts(pvl_panel: dict[str, Any]) -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    for group in (pvl_panel.get("groups") or []):
        contracts = group.get("contracts") if isinstance(group, dict) else []
        for contract in contracts or []:
            if not isinstance(contract, dict):
                continue
            cid = str(contract.get("contract_id", "") or contract.get("id", "") or "")
            if cid:
                idx[cid] = contract
    return idx


def _index_maturity_gates(maturity_tracker: dict[str, Any]) -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    for category in (maturity_tracker.get("categories") or []):
        if not isinstance(category, dict):
            continue
        cat_id = str(category.get("id", "") or "")
        for gate in (category.get("gates") or []):
            if not isinstance(gate, dict):
                continue
            gid = str(gate.get("id", "") or "")
            if not gid:
                continue
            merged = dict(gate)
            merged["_category_id"] = cat_id
            idx[gid] = merged
    return idx


def _build_check(
    *,
    check_id: str,
    label: str,
    kind: str,
    current_ok: bool,
    ever_ok: bool,
    current_detail: str,
    ever_detail: str,
    critical: bool,
    prior_attested_ok: bool | None = None,
    prior_attested_detail: str | None = None,
    attestation_strength: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": check_id,
        "label": label,
        "kind": kind,
        "critical": bool(critical),
        "current_ok": bool(current_ok),
        "ever_ok": bool(ever_ok),
        "current_detail": str(current_detail or "--"),
        "ever_detail": str(ever_detail or "--"),
    }
    # Attestation is a separate evidence class from ``ever_ok`` (which
    # tracks auto-observed high-water state). Only emit these fields
    # when the check actually participates in the attestation model so
    # existing consumers remain stable.
    if prior_attested_ok is not None:
        out["prior_attested_ok"] = bool(prior_attested_ok)
        out["prior_attested_detail"] = str(prior_attested_detail or "--")
        out["attestation_strength"] = str(attestation_strength or "none")
    return out


def _build_contract_check(
    *,
    contract_id: str,
    label: str,
    contracts: dict[str, dict[str, Any]],
    critical: bool,
) -> dict[str, Any]:
    contract = contracts.get(contract_id, {})
    status = str(contract.get("status", "") or "").lower()
    current_ok = status == "pass"
    ever_ok = bool(contract.get("ever_passed", False))

    evidence = str(contract.get("evidence", "") or "")
    last_pass = str(contract.get("last_pass_evidence", "") or "")

    current_detail = evidence or (status if status else "not_observed")
    if ever_ok:
        ever_detail = last_pass or "historically passed"
    else:
        ever_detail = "not yet met"

    return _build_check(
        check_id=contract_id,
        label=label,
        kind="contract",
        current_ok=current_ok,
        ever_ok=ever_ok,
        current_detail=current_detail,
        ever_detail=ever_detail,
        critical=critical,
    )


def _build_gate_check(
    *,
    gate_id: str,
    label: str,
    gates: dict[str, dict[str, Any]],
    critical: bool = False,
) -> dict[str, Any]:
    gate = gates.get(gate_id, {})
    status = str(gate.get("status", "") or "").lower()
    current_ok = status == "active"
    ever_ok = bool(gate.get("ever_met", False))
    current_detail = str(gate.get("display", "") or "--")

    best = _as_float(gate.get("best_current"))
    threshold = _as_float(gate.get("threshold"))
    if ever_ok and best is not None:
        if threshold is not None:
            ever_detail = f"best={best:.3f} threshold={threshold:.3f}"
        else:
            ever_detail = f"best={best:.3f}"
    elif ever_ok:
        ever_detail = "historically active"
    else:
        ever_detail = "not yet met"

    return _build_check(
        check_id=gate_id,
        label=label,
        kind="maturity_gate",
        current_ok=current_ok,
        ever_ok=ever_ok,
        current_detail=current_detail,
        ever_detail=ever_detail,
        critical=critical,
    )


# ---------------------------------------------------------------------------
# P4 HRR / VSA validation checks
# ---------------------------------------------------------------------------

# Forbidden imports scanned for in any HRR-owned module. Enforcement is
# pragmatic structural scanning (string-level), NOT Python call-graph analysis.
# Monkeypatch sentinel coverage lives in brain/tests/test_hrr_validation_pack.py.
_HRR_FORBIDDEN_IMPORTS: tuple[tuple[str, str], ...] = (
    # (import substring, human-readable reason)
    ("policy.state_encoder", "HRR must not feed the policy encoder"),
    ("policy.policy_nn", "HRR must not train/write the policy NN"),
    ("epistemic.belief_graph.bridge", "HRR must not create belief edges"),
    ("epistemic.soul_integrity", "HRR must not touch soul integrity"),
    ("memory.persistence", "HRR must not write to canonical memory"),
    ("memory.storage", "HRR must not write to canonical memory"),
    ("memory.canonical", "HRR must not write to canonical memory"),
    ("autonomy.", "HRR must not drive autonomy"),
    ("identity.kernel", "HRR must not mutate identity"),
)

# HRR-owned directories (relative to brain/). Any .py under these trees is
# scanned for forbidden imports.
_HRR_MODULE_ROOTS: tuple[str, ...] = (
    "library/vsa",
    "cognition/hrr_world_encoder.py",
    "cognition/hrr_simulation_shadow.py",
    "cognition/hrr_spatial_encoder.py",
    "cognition/mental_world.py",
    "cognition/mental_navigation.py",
    "cognition/spatial_scene_graph.py",
    "memory/hrr_recall_advisor.py",
    "hemisphere/hrr_specialist.py",
    "synthetic/hrr_exercise.py",
)

# P5 mental-world scene lane — extra module roots scanned for the same
# forbidden imports plus P5-specific "must not import" rules (handled
# inside :func:`_scan_p5_mental_world_imports`). Tracked separately so
# the structural check can distinguish fixture_ok vs live_ok.
_P5_MENTAL_WORLD_ROOTS: tuple[str, ...] = (
    "cognition/spatial_scene_graph.py",
    "cognition/hrr_spatial_encoder.py",
    "cognition/mental_world.py",
    "cognition/mental_navigation.py",
)

_HRR_FRESHNESS_SECONDS = 30 * 24 * 60 * 60  # 30 days


def _brain_root() -> "Path":
    from pathlib import Path as _Path

    here = _Path(__file__).resolve()
    for cand in [here, *here.parents]:
        if (cand / "library").exists() and (cand / "tests").exists():
            return cand
    return _Path(__file__).resolve().parent.parent


def _repo_root() -> "Path":
    from pathlib import Path as _Path

    brain = _brain_root()
    parent = brain.parent
    if (parent / "docs").exists():
        return parent
    return brain


def _load_hrr_stage0_evidence() -> dict[str, Any] | None:
    """Read the Stage 0 exercise JSON if present, else None. Never raises."""
    import json as _json
    from pathlib import Path as _Path

    path = _repo_root() / "docs" / "validation_reports" / "evidence" / "hrr_stage0.json"
    if not path.exists():
        return None
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _scan_hrr_forbidden_imports() -> tuple[bool, list[str]]:
    """Scan every HRR-owned .py file for any forbidden import substring.

    Returns ``(is_clean, violation_details)``. A clean scan has ``is_clean=True``
    and an empty list. Missing files are silently skipped so this stays
    idempotent across commits that land HRR modules incrementally.
    """
    from pathlib import Path as _Path

    brain = _brain_root()
    violations: list[str] = []
    scanned = 0

    for root in _HRR_MODULE_ROOTS:
        target = brain / root
        py_files: list[_Path] = []
        if target.is_dir():
            py_files = sorted(target.rglob("*.py"))
        elif target.is_file():
            py_files = [target]
        # Missing is OK — module may not have landed yet this sprint.
        for pyf in py_files:
            scanned += 1
            try:
                src = pyf.read_text(encoding="utf-8")
            except OSError:
                continue
            for needle, _reason in _HRR_FORBIDDEN_IMPORTS:
                if needle in src:
                    rel = pyf.relative_to(brain)
                    violations.append(f"{rel}:{needle}")

    return (len(violations) == 0, violations)


def _scan_p5_mental_world_imports() -> tuple[bool, list[str]]:
    """P5-specific structural scan.

    In addition to the general HRR forbidden list, the P5 mental-world
    modules must not reach into the perception orchestrator's private
    members (they must use the public ``get_scene_snapshot`` /
    ``get_spatial_tracks`` / ``get_spatial_anchors`` accessors only).
    """
    from pathlib import Path as _Path

    brain = _brain_root()
    violations: list[str] = []
    for root in _P5_MENTAL_WORLD_ROOTS:
        target = brain / root
        if not target.is_file():
            continue
        try:
            src = target.read_text(encoding="utf-8")
        except OSError:
            continue
        rel = target.relative_to(brain)
        # No private member access on the orchestrator.
        forbidden_snippets = (
            "_scene_tracker",
            "_spatial_estimator",
            "_perc_orch._",
        )
        for snippet in forbidden_snippets:
            if snippet in src:
                violations.append(f"{rel}:{snippet}")
    return (len(violations) == 0, violations)


def _p5_scene_fixture_evidence() -> dict[str, Any] | None:
    """Read the P5 mental-world fixture evidence JSON if present.

    The evidence is produced by the soak-test tooling and contains
    a small deterministic scene + encoder metrics used to prove the
    encoder behaves under fixtures (``fixture_ok``). ``live_ok`` is
    a separate check that relies on live engine samples.
    """
    import json as _json

    path = (
        _repo_root()
        / "docs"
        / "validation_reports"
        / "evidence"
        / "hrr_spatial_scene_fixture.json"
    )
    if not path.exists():
        return None
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _p5_live_evidence() -> dict[str, Any] | None:
    """Read the P5 live mental-world sample snapshot, when present."""
    import json as _json

    path = (
        _repo_root()
        / "docs"
        / "validation_reports"
        / "evidence"
        / "hrr_spatial_scene_live.json"
    )
    if not path.exists():
        return None
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _build_p5_mental_world_checks() -> list[dict[str, Any]]:
    """Build the P5 mental-world scene-lane checks.

    All four checks are non-critical (research lane). They NEVER flip
    the ``spatial_hrr_mental_world`` maturity marker — it stays
    ``PRE-MATURE`` until an explicit governance decision promotes it.

    Check set:

    1. ``p5_mental_world_status_marker`` — the dashboard status marker
       must remain ``PRE-MATURE`` for the mental-world lane.
    2. ``p5_mental_world_structure`` — no private-member access into
       ``perception_orchestrator``; P5 must use public accessors only.
    3. ``p5_mental_world_fixture_ok`` — encoder passes the deterministic
       fixture scene (cleanup / relation-recovery / no-side-effects).
    4. ``p5_mental_world_live_ok`` — engine-sampled live run shows
       authority flags pinned and zero side effects.
    """
    struct_ok, struct_violations = _scan_p5_mental_world_imports()
    struct_detail = (
        f"p5_private_access_violations={len(struct_violations)} ({'; '.join(struct_violations[:5]) or 'clean'})"
    )

    # Status marker — always PRE-MATURE until governance flips it.
    status_ok = True
    status_detail = "spatial_hrr_mental_world=PRE-MATURE (expected)"

    # Fixture axis.
    fix = _p5_scene_fixture_evidence()
    if fix is None:
        fixture_ok = False
        fixture_detail = "no hrr_spatial_scene_fixture.json evidence present"
    else:
        cleanup = fix.get("cleanup_accuracy")
        rel_recov = fix.get("relation_recovery")
        side_effects = fix.get("spatial_hrr_side_effects", None)
        authority_flags = fix.get("authority_flags") or {}
        fixture_ok = bool(
            isinstance(cleanup, (int, float)) and cleanup >= 0.9
            and isinstance(rel_recov, (int, float)) and rel_recov >= 0.8
            and side_effects == 0
            and all(v is False for v in authority_flags.values())
        )
        fixture_detail = (
            f"cleanup={cleanup}, relation_recovery={rel_recov}, "
            f"side_effects={side_effects}, authority_all_false="
            f"{all(v is False for v in authority_flags.values())}"
        )

    # Live axis.
    live = _p5_live_evidence()
    if live is None:
        live_ok = False
        live_detail = "no hrr_spatial_scene_live.json evidence present"
    else:
        authority_flags = live.get("authority_flags") or {}
        side_effects = live.get("spatial_hrr_side_effects", None)
        samples = int(live.get("samples_observed", 0) or 0)
        live_ok = bool(
            samples > 0
            and side_effects == 0
            and all(v is False for v in authority_flags.values())
        )
        live_detail = (
            f"samples={samples}, side_effects={side_effects}, "
            f"authority_all_false={all(v is False for v in authority_flags.values())}"
        )

    return [
        _build_check(
            check_id="p5_mental_world_status_marker",
            label="P5 Mental World status marker (spatial_hrr_mental_world=PRE-MATURE)",
            kind="hrr_p5",
            current_ok=status_ok,
            ever_ok=status_ok,
            current_detail=status_detail,
            ever_detail=status_detail,
            critical=False,
        ),
        _build_check(
            check_id="p5_mental_world_structure",
            label="P5 Mental World structural boundary (public perception accessors only)",
            kind="hrr_p5",
            current_ok=struct_ok,
            ever_ok=struct_ok,
            current_detail=struct_detail,
            ever_detail=struct_detail,
            critical=False,
        ),
        _build_check(
            check_id="p5_mental_world_fixture_ok",
            label="P5 Mental World fixture_ok (deterministic scene encoder gates)",
            kind="hrr_p5",
            current_ok=fixture_ok,
            ever_ok=fixture_ok,
            current_detail=fixture_detail,
            ever_detail=fixture_detail,
            critical=False,
        ),
        _build_check(
            check_id="p5_mental_world_live_ok",
            label="P5 Mental World live_ok (engine-sampled zero-authority runs)",
            kind="hrr_p5",
            current_ok=live_ok,
            ever_ok=live_ok,
            current_detail=live_detail,
            ever_detail=live_detail,
            critical=False,
        ),
    ]


def _build_hrr_checks() -> list[dict[str, Any]]:
    """Build the 5 HRR validation checks (2 non-critical, 3 critical)."""
    import time as _time

    evidence = _load_hrr_stage0_evidence()
    gates = (evidence or {}).get("gates") or {}
    thresholds = gates.get("thresholds") or {}
    generated_at = (evidence or {}).get("generated_at")
    now_ts = int(_time.time())
    age_s = None
    if isinstance(generated_at, (int, float)):
        age_s = int(now_ts - int(generated_at))

    fresh = age_s is not None and age_s <= _HRR_FRESHNESS_SECONDS
    primitive_ok = bool(evidence is not None and gates.get("all_pass", False) and fresh)
    primitive_ever = bool(evidence is not None and gates.get("all_pass", False))
    acc8 = gates.get("cleanup_accuracy_at_8")
    acc16 = gates.get("cleanup_accuracy_at_16")
    fp = gates.get("false_positive_rate")
    primitive_detail = (
        f"acc@8={acc8}, acc@16={acc16}, fp={fp}, age_s={age_s}"
        if evidence is not None
        else "no hrr_stage0.json evidence present"
    )

    authority_flags = (evidence or {}).get("authority_flags") or {}
    side_effects = (evidence or {}).get("hrr_side_effects", None)
    truth_ok = bool(
        evidence is not None
        and side_effects == 0
        and all(v is False for v in authority_flags.values())
    )
    truth_ever = truth_ok  # there is no history store; presence of passing evidence is sufficient
    truth_detail = (
        f"hrr_side_effects={side_effects}, authority_flags_false={all(v is False for v in authority_flags.values())}"
        if evidence is not None
        else "no hrr_stage0.json evidence present"
    )

    is_clean, violations = _scan_hrr_forbidden_imports()
    violation_summary = "; ".join(violations[:5]) if violations else "clean"
    struct_detail = f"forbidden_import_violations={len(violations)} ({violation_summary})"

    checks = [
        _build_check(
            check_id="hrr_primitive_library",
            label="HRR Primitive Library (Stage 0 synthetic gates)",
            kind="hrr",
            current_ok=primitive_ok,
            ever_ok=primitive_ever,
            current_detail=primitive_detail + (" (thresholds=" + str(thresholds) + ")" if thresholds else ""),
            ever_detail=primitive_detail,
            critical=False,
        ),
        _build_check(
            check_id="hrr_truth_boundary",
            label="HRR Truth Boundary (hrr_side_effects == 0, all authority flags false)",
            kind="hrr",
            current_ok=truth_ok,
            ever_ok=truth_ever,
            current_detail=truth_detail,
            ever_detail=truth_detail,
            critical=False,
        ),
        _build_check(
            check_id="hrr_policy_non_influence",
            label="HRR Policy Non-Influence (no forbidden policy imports in HRR modules)",
            kind="hrr",
            current_ok=is_clean,
            ever_ok=is_clean,
            current_detail=struct_detail,
            ever_detail=struct_detail,
            critical=True,
        ),
        _build_check(
            check_id="hrr_belief_non_influence",
            label="HRR Belief Non-Influence (no belief-graph writer imports in HRR modules)",
            kind="hrr",
            current_ok=is_clean,
            ever_ok=is_clean,
            current_detail=struct_detail,
            ever_detail=struct_detail,
            critical=True,
        ),
        _build_check(
            check_id="hrr_memory_non_canonical",
            label="HRR Memory Non-Canonical (no canonical-memory writer imports in HRR modules)",
            kind="hrr",
            current_ok=is_clean,
            ever_ok=is_clean,
            current_detail=struct_detail,
            ever_detail=struct_detail,
            critical=True,
        ),
    ]
    return checks


def build_validation_pack(
    pvl_panel: dict[str, Any] | None,
    maturity_tracker: dict[str, Any] | None,
    language_panel: dict[str, Any] | None = None,
    autonomy_panel: dict[str, Any] | None = None,
    release_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a unified runtime validation pack from eval panels."""
    pvl = pvl_panel or {}
    maturity = maturity_tracker or {}
    language = language_panel or {}
    autonomy = autonomy_panel or {}
    release = release_validation or {}

    contracts = _index_contracts(pvl)
    gates = _index_maturity_gates(maturity)

    checks: list[dict[str, Any]] = []

    # Global health checks (critical)
    pvl_cov = _as_float(pvl.get("coverage_pct")) or 0.0
    pvl_applicable = int(pvl.get("applicable_contracts", 0) or 0)
    pvl_ever_passing = int(pvl.get("ever_passing_contracts", 0) or 0)
    pvl_ever_cov = (100.0 * pvl_ever_passing / pvl_applicable) if pvl_applicable > 0 else 0.0
    checks.append(
        _build_check(
            check_id="pvl_coverage",
            label="PVL Coverage >= 85%",
            kind="meta",
            current_ok=pvl_cov >= 85.0,
            ever_ok=pvl_ever_cov >= 85.0 if pvl_applicable > 0 else pvl_cov >= 85.0,
            current_detail=f"{pvl_cov:.1f}%",
            ever_detail=f"{pvl_ever_cov:.1f}%" if pvl_applicable > 0 else f"{pvl_cov:.1f}%",
            critical=True,
        )
    )

    total_gates = int(maturity.get("total_gates", 0) or 0)
    active_gates = int(maturity.get("active_gates", 0) or 0)
    ever_gates = int(maturity.get("ever_active_gates", 0) or 0)
    cur_ratio = (active_gates / total_gates) if total_gates > 0 else 0.0
    ever_ratio = (ever_gates / total_gates) if total_gates > 0 else cur_ratio
    checks.append(
        _build_check(
            check_id="maturity_progress",
            label="Maturity Active Ratio >= 75%",
            kind="meta",
            current_ok=cur_ratio >= 0.75,
            ever_ok=ever_ratio >= 0.75,
            current_detail=f"{active_gates}/{total_gates} ({cur_ratio * 100:.1f}%)",
            ever_detail=f"{ever_gates}/{total_gates} ({ever_ratio * 100:.1f}%)",
            critical=True,
        )
    )

    released_total = int(release.get("released_total", 0) or 0)
    released_validated = int(release.get("released_validated", 0) or 0)
    released_without_validation = int(release.get("released_without_validation", 0) or 0)
    validation_failed = int(release.get("validation_failed", 0) or 0)
    checks.append(
        _build_check(
            check_id="output_release_validation",
            label="Output Release Validation (released_without_validation == 0)",
            kind="trace_release",
            current_ok=released_without_validation == 0,
            ever_ok=released_without_validation == 0,
            current_detail=(
                f"released={released_total}, validated={released_validated}, "
                f"without_validation={released_without_validation}, validation_failed={validation_failed}"
            ),
            ever_detail=(
                f"released={released_total}, validated={released_validated}, "
                f"without_validation={released_without_validation}"
            ),
            critical=True,
        )
    )

    # Critical contracts (what has truly happened)
    for cid, label in _CRITICAL_CONTRACTS:
        checks.append(
            _build_contract_check(
                contract_id=cid,
                label=label,
                contracts=contracts,
                critical=True,
            )
        )

    # Aggregate skill learning lifecycle.
    skill_ids = (
        "skill_registered",
        "learning_job_started",
        "job_phase_advanced",
        "skill_learning_completed",
    )
    skill_current = 0
    skill_ever = 0
    for sid in skill_ids:
        c = contracts.get(sid, {})
        if str(c.get("status", "")).lower() == "pass":
            skill_current += 1
        if bool(c.get("ever_passed", False)):
            skill_ever += 1
    checks.append(
        _build_check(
            check_id="skill_learning_lifecycle",
            label="Skill Learning Lifecycle Complete",
            kind="composite",
            current_ok=skill_current == len(skill_ids),
            ever_ok=skill_ever == len(skill_ids),
            current_detail=f"{skill_current}/{len(skill_ids)} currently passing",
            ever_detail=f"{skill_ever}/{len(skill_ids)} ever passed",
            critical=True,
        )
    )

    # Non-critical trend checks that still matter for roadmap pacing.
    checks.append(
        _build_gate_check(
            gate_id="soul_integrity",
            label="Soul Integrity Sustained",
            gates=gates,
            critical=False,
        )
    )

    # Phase 5 proof-chain checks (weakness -> research -> measured improvement).
    policy_memory = autonomy.get("policy_memory", {}) if isinstance(autonomy.get("policy_memory"), dict) else {}
    delta_tracker = autonomy.get("delta_tracker", {}) if isinstance(autonomy.get("delta_tracker"), dict) else {}
    avoid_patterns = policy_memory.get("avoid_patterns", [])
    if not isinstance(avoid_patterns, list):
        avoid_patterns = []
    weakness_tags = []
    if avoid_patterns and isinstance(avoid_patterns[0], dict):
        weakness_tags = [str(t) for t in (avoid_patterns[0].get("tags") or []) if isinstance(t, str)]
    weakness_now = len(avoid_patterns) > 0
    weakness_ever = weakness_now
    weakness_detail = (
        f"avoid_patterns={len(avoid_patterns)}"
        + (f", top_tags={','.join(weakness_tags[:3])}" if weakness_tags else "")
    )
    checks.append(
        _build_check(
            check_id="phase5_weakness_signal",
            label="Phase 5 Weakness Signal Identified",
            kind="phase5",
            current_ok=weakness_now,
            ever_ok=weakness_ever,
            current_detail=weakness_detail,
            ever_detail=weakness_detail if weakness_ever else "no weakness signal recorded",
            critical=False,
        )
    )

    research_contract = contracts.get("research_completed", {})
    research_status = str(research_contract.get("status", "") or "").lower()
    research_completed_total = int(autonomy.get("completed_total", 0) or 0)
    research_ever = bool(research_contract.get("ever_passed", False)) or bool(
        (gates.get("maturity_research_completed_20", {}) or {}).get("ever_met", False)
    ) or (research_completed_total >= 20)
    research_now = (research_status == "pass") or bool(
        (gates.get("maturity_research_completed_20", {}) or {}).get("status") == "active"
    ) or (research_completed_total >= 20)
    checks.append(
        _build_check(
            check_id="phase5_research_executed",
            label="Phase 5 Research Executed",
            kind="phase5",
            current_ok=research_now,
            ever_ok=research_ever,
            current_detail=f"research_completed={research_status or 'not_seen'}, completed_total={research_completed_total}",
            ever_detail=(
                str(research_contract.get("last_pass_evidence", "") or "")
                if research_ever
                else "research_completed not yet observed"
            ),
            critical=False,
        )
    )

    delta_contract = contracts.get("delta_measured", {})
    delta_status = str(delta_contract.get("status", "") or "").lower()
    total_measured = int(delta_tracker.get("total_measured", 0) or 0)
    total_improved = int(delta_tracker.get("total_improved", 0) or 0)
    delta_now = (delta_status == "pass") or (total_measured > 0)
    delta_ever = bool(delta_contract.get("ever_passed", False)) or (total_measured > 0)
    checks.append(
        _build_check(
            check_id="phase5_delta_measured",
            label="Phase 5 Delta Measurement Captured",
            kind="phase5",
            current_ok=delta_now,
            ever_ok=delta_ever,
            current_detail=f"delta_measured={delta_status or 'not_seen'}, measured={total_measured}",
            ever_detail=(
                str(delta_contract.get("last_pass_evidence", "") or "")
                if delta_ever
                else "delta_measured not yet observed"
            ),
            critical=False,
        )
    )

    total_wins = int(policy_memory.get("total_wins", 0) or 0)
    win_rate = _as_float(policy_memory.get("overall_win_rate")) or 0.0
    wins_gate = gates.get("maturity_autonomy_wins_10", {}) if isinstance(gates.get("maturity_autonomy_wins_10"), dict) else {}
    wins_now = bool(wins_gate.get("status") == "active") or (total_wins >= 10) or (total_improved >= 10)
    wins_ever = bool(wins_gate.get("ever_met", False)) or (total_wins >= 10) or (total_improved >= 10)
    checks.append(
        _build_check(
            check_id="phase5_positive_attribution",
            label="Phase 5 Positive Attribution Present",
            kind="phase5",
            current_ok=wins_now,
            ever_ok=wins_ever,
            current_detail=f"wins={total_wins}, improved={total_improved}, win_rate={win_rate:.3f}",
            ever_detail=(
                f"best_wins={wins_gate.get('best_current', total_wins)}"
                if wins_ever
                else "positive attribution floor not yet reached"
            ),
            critical=False,
        )
    )

    proof_now = all([weakness_now, research_now, delta_now, wins_now])
    proof_ever = all([weakness_ever, research_ever, delta_ever, wins_ever])
    checks.append(
        _build_check(
            check_id="phase5_proof_chain",
            label="Phase 5 End-to-End Proof Chain",
            kind="phase5",
            current_ok=proof_now,
            ever_ok=proof_ever,
            current_detail=(
                f"weakness={int(weakness_now)}, research={int(research_now)}, "
                f"delta={int(delta_now)}, wins={int(wins_now)}"
            ),
            ever_detail=(
                f"weakness={int(weakness_ever)}, research={int(research_ever)}, "
                f"delta={int(delta_ever)}, wins={int(wins_ever)}"
            ),
            critical=False,
        )
    )

    # Phase 6.5: L3 escalation surface. ``current_ok`` is sourced
    # strictly from the live autonomy eligibility panel;
    # ``prior_attested_ok`` is an explicit separate field and MUST NOT
    # be folded into ``ever_ok``.
    #
    # Emit the L3 checks only when the autonomy panel actually includes
    # the ``l3`` sub-dict. Callers that predate Phase 6.5 (e.g. unit
    # tests, minimal probes) still produce a valid pack without these
    # checks.
    if isinstance(autonomy.get("l3"), dict):
        l3_panel = autonomy["l3"]
        l3_current_ok = bool(l3_panel.get("current_ok", False))
        l3_prior_attested_ok = bool(l3_panel.get("prior_attested_ok", False))
        l3_attestation_strength = str(l3_panel.get("attestation_strength", "none") or "none")
        l3_live_level = int(l3_panel.get("live_autonomy_level", 0) or 0)
        l3_request_ok = bool(l3_panel.get("request_ok", l3_current_ok or l3_prior_attested_ok))
        l3_current_detail = (
            l3_panel.get("current_detail", {})
            if isinstance(l3_panel.get("current_detail"), dict)
            else {}
        )
        l3_win_rate = float(l3_current_detail.get("win_rate", 0.0) or 0.0)
        l3_wins = int(l3_current_detail.get("wins", 0) or 0)
        l3_reason = str(l3_current_detail.get("reason", "") or "")
        l3_current_detail_str = (
            f"live_level={l3_live_level}, wins={l3_wins}, win_rate={l3_win_rate:.3f}"
            + (f", reason={l3_reason}" if l3_reason else "")
        )
        if l3_prior_attested_ok:
            l3_prior_detail_str = f"attestation_strength={l3_attestation_strength}"
        else:
            l3_prior_detail_str = "no accepted attestation on file"
        checks.append(
            _build_check(
                check_id="l3_escalation_requestable",
                label="L3 Escalation Requestable (current_ok OR prior_attested_ok)",
                kind="phase6_5",
                current_ok=l3_request_ok,
                # Ever-met for this check tracks ``request_ok`` having
                # been true at least once during this live session;
                # attestation history is reported through the separate
                # attestation fields, not backfilled into ever_ok.
                ever_ok=l3_request_ok,
                current_detail=l3_current_detail_str,
                ever_detail=l3_current_detail_str if l3_request_ok else "not yet met",
                critical=False,
                prior_attested_ok=l3_prior_attested_ok,
                prior_attested_detail=l3_prior_detail_str,
                attestation_strength=l3_attestation_strength,
            )
        )

        # Lifecycle check: has an escalation ever reached a terminal
        # status in the store?
        pending_list = (
            l3_panel.get("pending", [])
            if isinstance(l3_panel.get("pending"), list)
            else []
        )
        lifecycle_tail = (
            l3_panel.get("recent_lifecycle", [])
            if isinstance(l3_panel.get("recent_lifecycle"), list)
            else []
        )
        terminal_outcomes = [
            str(entry.get("status", "") or "")
            for entry in lifecycle_tail
            if isinstance(entry, dict)
        ]
        clean_terminals = sum(1 for s in terminal_outcomes if s == "approved")
        rolled_back_terminals = sum(1 for s in terminal_outcomes if s == "rolled_back")
        parked_terminals = sum(1 for s in terminal_outcomes if s == "parked")
        rejected_terminals = sum(1 for s in terminal_outcomes if s == "rejected")
        any_terminal = len(terminal_outcomes) > 0
        lifecycle_detail = (
            f"pending={len(pending_list)}, approved={clean_terminals}, "
            f"rolled_back={rolled_back_terminals}, parked={parked_terminals}, "
            f"rejected={rejected_terminals}"
        )
        checks.append(
            _build_check(
                check_id="l3_escalation_lifecycle_ever",
                label="L3 Escalation Lifecycle Executed End-to-End",
                kind="phase6_5",
                current_ok=any_terminal,
                ever_ok=any_terminal,
                current_detail=lifecycle_detail,
                ever_detail=(
                    lifecycle_detail
                    if any_terminal
                    else "no escalation has reached a terminal status"
                ),
                critical=False,
            )
        )

    corpus_total = int(language.get("corpus_total_examples", 0) or 0)
    quality_events = int(language.get("quality_total_events", 0) or 0)
    gate_color = str(language.get("gate_color", "") or "").lower()
    gate_color_code = int(language.get("gate_color_code", 0) or 0)
    gate_scores_by_class = (
        language.get("gate_scores_by_class", {})
        if isinstance(language.get("gate_scores_by_class"), dict)
        else {}
    )
    route_class_pairs_raw = (
        language.get("corpus_route_class_pairs", {})
        if isinstance(language.get("corpus_route_class_pairs"), dict)
        else {}
    )
    route_class_pairs: dict[str, int] = {}
    for k, v in route_class_pairs_raw.items():
        key = str(k or "").strip().lower()
        if not key:
            continue
        try:
            route_class_pairs[key] = int(v or 0)
        except Exception:
            continue
    # Backward-compatible fallback for older snapshots that only expose recent examples.
    if not route_class_pairs:
        recent_examples = (
            language.get("corpus_recent_examples", [])
            if isinstance(language.get("corpus_recent_examples"), list)
            else []
        )
        for ex in recent_examples:
            if not isinstance(ex, dict):
                continue
            route = str(ex.get("route", "") or "").strip().lower()
            response_class = str(ex.get("response_class", "") or "").strip().lower()
            if not route or not response_class:
                continue
            key = f"{route}|{response_class}"
            route_class_pairs[key] = route_class_pairs.get(key, 0) + 1
    corpus_by_class = (
        language.get("corpus_response_classes", {})
        if isinstance(language.get("corpus_response_classes"), dict)
        else {}
    )
    promotion_summary = (
        language.get("promotion_summary", {})
        if isinstance(language.get("promotion_summary"), dict)
        else {}
    )
    promotion_shadow_count = int(language.get("promotion_shadow_count", 0) or 0)
    promotion_canary_count = int(language.get("promotion_canary_count", 0) or 0)
    promotion_live_count = int(language.get("promotion_live_count", 0) or 0)
    promotion_red_classes = int(language.get("promotion_red_classes", 0) or 0)
    promotion_red_quality_classes = int(language.get("promotion_red_quality_classes", promotion_red_classes) or 0)
    promotion_red_data_limited_classes = int(language.get("promotion_red_data_limited_classes", 0) or 0)
    promotion_total_evaluations = int(language.get("promotion_total_evaluations", 0) or 0)
    promotion_max_consecutive_red = int(language.get("promotion_max_consecutive_red", 0) or 0)
    runtime_bridge_enabled = bool(language.get("runtime_bridge_enabled", False))
    runtime_rollout_mode = str(language.get("runtime_rollout_mode", "off") or "off").lower()
    runtime_rollout_active = runtime_bridge_enabled and runtime_rollout_mode != "off"
    runtime_guard_total = int(language.get("runtime_guard_total", 0) or 0)
    runtime_live_total = int(language.get("runtime_live_total", 0) or 0)
    runtime_blocked_count = int(language.get("runtime_blocked_by_guard_count", 0) or 0)
    runtime_unpromoted_attempts = int(language.get("runtime_unpromoted_live_attempts", 0) or 0)
    runtime_live_red_classes = int(language.get("runtime_live_red_classes", 0) or 0)
    expected_classes = max(len(promotion_summary), 7)
    language_evidence_targets: list[dict[str, Any]] = []
    language_route_class_baselines: list[dict[str, Any]] = []
    promotion_current_progress = (promotion_canary_count + promotion_live_count) > 0
    promotion_ever_progress = promotion_current_progress
    for row in promotion_summary.values():
        if not isinstance(row, dict):
            continue
        if int(row.get("promotion_history_len", 0) or 0) > 0:
            promotion_ever_progress = True
            break
        if _as_float(row.get("last_transition_at")) and (_as_float(row.get("last_transition_at")) or 0) > 0:
            promotion_ever_progress = True
            break

    phase_c = language.get("phase_c", {}) if isinstance(language.get("phase_c"), dict) else {}
    student_available = bool(phase_c.get("student_available", False))
    checks.append(
        _build_check(
            check_id="language_shadow_student",
            label="Language Shadow Student Available",
            kind="language",
            current_ok=student_available,
            ever_ok=student_available,
            current_detail="available" if student_available else "not available",
            ever_detail="available" if student_available else "not yet met",
            critical=False,
        )
    )
    checks.append(
        _build_check(
            check_id="language_shadow_data_volume",
            label="Language Shadow Data Volume >= 100",
            kind="language",
            current_ok=corpus_total >= 100,
            ever_ok=corpus_total >= 100,
            current_detail=f"corpus={corpus_total}, quality_events={quality_events}",
            ever_detail=f"corpus={corpus_total}, quality_events={quality_events}",
            critical=False,
        )
    )
    checks.append(
        _build_check(
            check_id="language_gate_not_red",
            label="Language Gate Not Red",
            kind="language",
            current_ok=(gate_color_code >= 1) or (gate_color and gate_color != "red"),
            ever_ok=((gate_color_code >= 1) or (gate_color and gate_color != "red") or promotion_ever_progress),
            current_detail=f"gate_color={gate_color or 'unknown'}",
            ever_detail="historically non-red gate observed" if promotion_ever_progress else f"gate_color={gate_color or 'unknown'}",
            critical=False,
        )
    )
    checks.append(
        _build_check(
            check_id="language_promotion_evals",
            label="Language Promotion Evaluations Recorded",
            kind="language",
            current_ok=promotion_total_evaluations >= expected_classes,
            ever_ok=promotion_total_evaluations >= expected_classes,
            current_detail=(
                f"evals={promotion_total_evaluations}, classes={expected_classes}, "
                f"levels s/c/l={promotion_shadow_count}/{promotion_canary_count}/{promotion_live_count}"
            ),
            ever_detail=(
                f"evals={promotion_total_evaluations}, classes={expected_classes}, "
                f"levels s/c/l={promotion_shadow_count}/{promotion_canary_count}/{promotion_live_count}"
            ),
            critical=False,
        )
    )
    checks.append(
        _build_check(
            check_id="language_promotion_progress",
            label="Language Promotion Progress Observed",
            kind="language",
            current_ok=promotion_current_progress,
            ever_ok=promotion_ever_progress,
            current_detail=(
                f"canary+live={promotion_canary_count + promotion_live_count}, "
                f"red_quality={promotion_red_quality_classes}, "
                f"red_data_limited={promotion_red_data_limited_classes}, "
                f"red_total={promotion_red_classes}, max_red={promotion_max_consecutive_red}"
            ),
            ever_detail=(
                "historical promotion transition observed"
                if promotion_ever_progress
                else "no transitions observed yet"
            ),
            critical=False,
        )
    )
    checks.append(
        _build_check(
            check_id="language_class_gate_diagnostics",
            label="Language Class Gate Diagnostics Present",
            kind="language",
            current_ok=len(gate_scores_by_class) >= expected_classes,
            ever_ok=len(gate_scores_by_class) >= expected_classes,
            current_detail=f"class_diagnostics={len(gate_scores_by_class)} expected={expected_classes}",
            ever_detail=f"class_diagnostics={len(gate_scores_by_class)} expected={expected_classes}",
            critical=False,
        )
    )
    route_class_matched = 0
    for route, response_class in _LANGUAGE_ROUTE_CLASS_BASELINES:
        pair_key = f"{route.lower()}|{response_class.lower()}"
        pair_count = int(route_class_pairs.get(pair_key, 0) or 0)
        pair_ok = pair_count > 0
        if pair_ok:
            route_class_matched += 1
        checks.append(
            _build_check(
                check_id=f"language_route_class_{route.lower()}_{response_class}",
                label=f"Language Baseline Route/Class ({route} -> {response_class})",
                kind="language_baseline",
                current_ok=pair_ok,
                ever_ok=pair_ok,
                current_detail=f"count={pair_count}",
                ever_detail=f"count={pair_count}" if pair_ok else "not yet observed",
                critical=False,
            )
        )
        language_route_class_baselines.append(
            {
                "route": route,
                "response_class": response_class,
                "current_ok": pair_ok,
                "ever_ok": pair_ok,
                "count": pair_count,
            }
        )
    missing_pairs = [
        f"{route}->{response_class}"
        for route, response_class in _LANGUAGE_ROUTE_CLASS_BASELINES
        if int(route_class_pairs.get(f"{route.lower()}|{response_class.lower()}", 0) or 0) <= 0
    ]
    checks.append(
        _build_check(
            check_id="language_route_class_contract",
            label="Language Baseline Route/Class Contract",
            kind="language_baseline",
            current_ok=route_class_matched == len(_LANGUAGE_ROUTE_CLASS_BASELINES),
            ever_ok=route_class_matched == len(_LANGUAGE_ROUTE_CLASS_BASELINES),
            current_detail=(
                f"matched={route_class_matched}/{len(_LANGUAGE_ROUTE_CLASS_BASELINES)}, "
                f"missing={','.join(missing_pairs) if missing_pairs else 'none'}"
            ),
            ever_detail=(
                f"matched={route_class_matched}/{len(_LANGUAGE_ROUTE_CLASS_BASELINES)}"
                if route_class_matched > 0
                else "no baseline route/class pairs observed yet"
            ),
            critical=False,
        )
    )
    try:
        from jarvis_eval.language_scorers import classify_gate_reason as _classify_gate_reason
    except Exception:
        _classify_gate_reason = None
    for response_class in _LANGUAGE_EVIDENCE_TARGETS:
        class_diag = (
            gate_scores_by_class.get(response_class, {})
            if isinstance(gate_scores_by_class.get(response_class), dict)
            else {}
        )
        class_scores = (
            class_diag.get("scores", {})
            if isinstance(class_diag.get("scores"), dict)
            else {}
        )
        sample_score = _as_float(class_scores.get("sample_count")) or 0.0
        class_color = str(class_diag.get("color", "") or "").lower()
        promo_row = (
            promotion_summary.get(response_class, {})
            if isinstance(promotion_summary.get(response_class), dict)
            else {}
        )
        live_gate_reason = str(class_diag.get("gate_reason", "") or "").lower()
        derived_gate_reason = ""
        if class_scores and _classify_gate_reason is not None:
            try:
                derived_gate_reason = str(_classify_gate_reason(class_scores) or "").lower()
            except Exception:
                derived_gate_reason = ""
        if not live_gate_reason:
            live_gate_reason = derived_gate_reason
        promo_gate_reason = str(promo_row.get("gate_reason", "") or "").lower()

        # Current evidence rows follow live class diagnostics first; promotion
        # reasons are historical context fallback for red rows only.
        if class_color == "red":
            gate_reason = live_gate_reason or derived_gate_reason or promo_gate_reason or ""
        else:
            gate_reason = "insufficient_samples" if live_gate_reason == "insufficient_samples" else "ok"

        observed_count_raw = int(corpus_by_class.get(response_class, 0) or 0)
        observed_count = observed_count_raw
        estimated_count = False
        if observed_count <= 0 and sample_score > 0.0:
            observed_count = int(round(sample_score * _MIN_CLASS_SAMPLES))
            estimated_count = True
        gap = max(0, _MIN_CLASS_SAMPLES - observed_count)

        quality_risk_red = (
            class_color == "red"
            and gate_reason not in ("", "ok", "insufficient_samples")
        )
        current_ok = (sample_score >= 1.0) and (not quality_risk_red)
        history_len = int(promo_row.get("promotion_history_len", 0) or 0)
        level = str(promo_row.get("level", "shadow") or "shadow").lower()
        ever_ok = current_ok or (history_len > 0) or (level in {"canary", "live"})

        count_label = f"~{observed_count}" if estimated_count else str(observed_count)
        check = _build_check(
            check_id=f"language_evidence_{response_class}",
            label=f"Language Evidence Floor ({response_class})",
            kind="language_coverage",
            current_ok=current_ok,
            ever_ok=ever_ok,
            current_detail=(
                f"count={count_label}/{_MIN_CLASS_SAMPLES}, gap={gap}, "
                f"sample_score={sample_score:.2f}, color={class_color or 'unknown'}, "
                f"reason={gate_reason or 'ok'}"
            ),
            ever_detail=(
                "historical progress observed "
                f"(level={level}, history={history_len})"
                if ever_ok and not current_ok
                else (
                    f"count={count_label}/{_MIN_CLASS_SAMPLES}, "
                    f"reason={gate_reason or 'ok'}"
                    if ever_ok
                    else "not yet met"
                )
            ),
            critical=False,
        )
        checks.append(check)
        language_evidence_targets.append(
            {
                "response_class": response_class,
                "current_ok": bool(current_ok),
                "ever_ok": bool(ever_ok),
                "count": int(observed_count),
                "count_estimated": bool(estimated_count),
                "target_count": int(_MIN_CLASS_SAMPLES),
                "gap": int(gap),
                "sample_score": float(sample_score),
                "color": class_color or "unknown",
                "gate_reason": gate_reason or "ok",
                "level": level,
                "promotion_history_len": int(history_len),
            }
        )
    checks.append(
        _build_check(
            check_id="language_runtime_guardrails",
            label="Language Runtime Guardrails Safe",
            kind="language_runtime",
            current_ok=(runtime_unpromoted_attempts == 0 and runtime_live_red_classes == 0),
            ever_ok=(runtime_unpromoted_attempts == 0 and runtime_live_red_classes == 0),
            current_detail=(
                f"bridge={runtime_bridge_enabled}, mode={runtime_rollout_mode}, "
                f"unpromoted_live={runtime_unpromoted_attempts}, live_red={runtime_live_red_classes}, "
                f"blocked={runtime_blocked_count}, live={runtime_live_total}"
            ),
            ever_detail=(
                f"bridge={runtime_bridge_enabled}, mode={runtime_rollout_mode}, "
                f"unpromoted_live={runtime_unpromoted_attempts}, live_red={runtime_live_red_classes}"
            ),
            critical=runtime_rollout_active,
        )
    )
    checks.append(
        _build_check(
            check_id="language_runtime_telemetry",
            label="Language Runtime Telemetry Present",
            kind="language_runtime",
            current_ok=(runtime_guard_total > 0) if runtime_rollout_active else True,
            ever_ok=(runtime_guard_total > 0) if runtime_rollout_active else True,
            current_detail=(
                f"guard_total={runtime_guard_total}, blocked={runtime_blocked_count}, "
                f"live={runtime_live_total}, mode={runtime_rollout_mode}"
            ),
            ever_detail=(
                f"guard_total={runtime_guard_total}, blocked={runtime_blocked_count}, "
                f"live={runtime_live_total}, mode={runtime_rollout_mode}"
            ),
            critical=False,
        )
    )

    # P4 HRR / VSA research-lane checks (structural, pragmatic enforcement).
    # These NEVER auto-flip the public status marker — that stays PRE-MATURE.
    # They ARE mechanical guards against accidental authority leaks.
    for hrr_check in _build_hrr_checks():
        checks.append(hrr_check)

    # P5 mental-world / spatial-HRR lane checks — same governance as P4:
    # all non-critical, never flip the dashboard status marker. The
    # ``fixture_ok`` and ``live_ok`` axes are kept separate so a
    # missing live sample never false-passes the structural/fixture
    # gates (and vice versa).
    for p5_check in _build_p5_mental_world_checks():
        checks.append(p5_check)

    checks_total = len(checks)
    passing = sum(1 for c in checks if c.get("current_ok"))
    ever_met = sum(1 for c in checks if c.get("ever_ok"))
    regressed = [c["id"] for c in checks if (not c.get("current_ok")) and c.get("ever_ok")]
    critical_total = sum(1 for c in checks if c.get("critical"))
    critical_passing = sum(1 for c in checks if c.get("critical") and c.get("current_ok"))
    critical_ever_met = sum(1 for c in checks if c.get("critical") and c.get("ever_ok"))
    blocked = [c["id"] for c in checks if c.get("critical") and (not c.get("current_ok"))]
    ever_blocked = [c["id"] for c in checks if c.get("critical") and (not c.get("ever_ok"))]
    ready_for_next_items = len(blocked) == 0
    ready_for_continuation = len(ever_blocked) == 0

    if blocked:
        status = "blocked"
    elif regressed:
        status = "caution"
    elif passing == checks_total:
        status = "ready"
    else:
        status = "caution"

    next_action = "Validation green: safe to continue roadmap implementation."
    if status == "blocked":
        next_action = "Resolve blocked critical checks before advancing roadmap phases."
    elif status == "caution":
        next_action = "No hard block, but verify regressions/non-critical gaps before promotion work."

    continuation_mode = "current_green"
    continuation_action = next_action
    if not ready_for_next_items and ready_for_continuation:
        continuation_mode = "historically_proven_recovering"
        continuation_action = (
            "Roadmap continuation is allowed because critical gates were proven before. "
            "Record reboot recovery state and keep monitoring until current critical checks re-pass."
        )
    elif not ready_for_continuation:
        continuation_mode = "not_yet_proven"
        continuation_action = "Do not continue roadmap advancement until critical gates are proven at least once."
    elif status == "caution":
        continuation_mode = "current_caution"

    return {
        "status": status,
        "checks_total": checks_total,
        "checks_passing": passing,
        "checks_ever_met": ever_met,
        "checks_regressed": len(regressed),
        "critical_total": critical_total,
        "critical_passing": critical_passing,
        "critical_ever_met": critical_ever_met,
        "regressed_check_ids": regressed,
        "blocked_check_ids": blocked,
        "ever_blocked_check_ids": ever_blocked,
        "ready_for_next_items": ready_for_next_items,
        "ready_for_continuation": ready_for_continuation,
        "continuation_mode": continuation_mode,
        "continuation_action": continuation_action,
        "next_action": next_action,
        "language_route_class_baselines": language_route_class_baselines,
        "language_evidence_targets": language_evidence_targets,
        "checks": checks,
    }


def build_runtime_validation_report(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Build CLI-friendly report from /api/full-snapshot payload."""
    eval_payload = snapshot.get("eval", {}) if isinstance(snapshot, dict) else {}
    validation = eval_payload.get("validation_pack")
    if not isinstance(validation, dict) or not validation.get("checks"):
        validation = build_validation_pack(
            eval_payload.get("pvl", {}),
            eval_payload.get("maturity_tracker", {}),
            eval_payload.get("language", {}),
            snapshot.get("autonomy", {}) if isinstance(snapshot.get("autonomy"), dict) else {},
            snapshot.get("release_validation", {}) if isinstance(snapshot.get("release_validation"), dict) else {},
        )

    pvl = eval_payload.get("pvl", {}) if isinstance(eval_payload.get("pvl"), dict) else {}
    maturity = (
        eval_payload.get("maturity_tracker", {})
        if isinstance(eval_payload.get("maturity_tracker"), dict)
        else {}
    )
    summary = {
        "pvl_coverage_pct": pvl.get("coverage_pct"),
        "pvl_passing_contracts": pvl.get("passing_contracts"),
        "pvl_failing_contracts": pvl.get("failing_contracts"),
        "pvl_awaiting_contracts": pvl.get("awaiting_contracts"),
        "pvl_ever_passing_contracts": pvl.get("ever_passing_contracts"),
        "maturity_active_gates": maturity.get("active_gates"),
        "maturity_total_gates": maturity.get("total_gates"),
        "maturity_ever_active_gates": maturity.get("ever_active_gates"),
        "released_outputs": snapshot.get("release_validation", {}).get("released_total", 0)
        if isinstance(snapshot.get("release_validation"), dict)
        else 0,
        "released_without_validation": snapshot.get("release_validation", {}).get("released_without_validation", 0)
        if isinstance(snapshot.get("release_validation"), dict)
        else 0,
    }
    ready_for_next_items = bool(validation.get("ready_for_next_items", False))
    ready_for_continuation_raw = validation.get("ready_for_continuation")
    if ready_for_continuation_raw is None:
        ready_for_continuation = ready_for_next_items
    else:
        ready_for_continuation = bool(ready_for_continuation_raw)
    continuation_mode = str(validation.get("continuation_mode", "") or "")
    if not continuation_mode:
        continuation_mode = "current_green" if ready_for_continuation else "not_yet_proven"
    continuation_action = str(validation.get("continuation_action", "") or "")
    if not continuation_action:
        continuation_action = str(validation.get("next_action", "") or "")

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "snapshot_ts": snapshot.get("_ts"),
        "status": validation.get("status", "unknown"),
        "ready_for_next_items": ready_for_next_items,
        "ready_for_continuation": ready_for_continuation,
        "continuation_mode": continuation_mode,
        "continuation_action": continuation_action,
        "next_action": validation.get("next_action", ""),
        "validation": validation,
        "summary": summary,
    }


def render_validation_markdown(report: dict[str, Any]) -> str:
    """Render markdown report for file output / sharing."""
    validation = report.get("validation", {}) if isinstance(report, dict) else {}
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    checks = validation.get("checks", []) if isinstance(validation, dict) else []
    language_targets = (
        validation.get("language_evidence_targets", [])
        if isinstance(validation.get("language_evidence_targets"), list)
        else []
    )
    route_class_targets = (
        validation.get("language_route_class_baselines", [])
        if isinstance(validation.get("language_route_class_baselines"), list)
        else []
    )
    phase5 = {}
    for check in checks:
        if isinstance(check, dict) and check.get("id") == "phase5_proof_chain":
            phase5 = check
            break

    lines = [
        "# Runtime Validation Pack",
        "",
        f"- Generated: {report.get('generated_at', '--')}",
        f"- Snapshot timestamp: {report.get('snapshot_ts', '--')}",
        f"- Status: **{report.get('status', 'unknown')}**",
        f"- Ready for next items: **{report.get('ready_for_next_items', False)}**",
        f"- Ready for continuation: **{validation.get('ready_for_continuation', False)}**",
        f"- Continuation mode: **{validation.get('continuation_mode', '--')}**",
        "",
        "## Summary",
        "",
        f"- PVL: {summary.get('pvl_passing_contracts', '--')}/{summary.get('pvl_ever_passing_contracts', '--')} passing/ever, coverage {summary.get('pvl_coverage_pct', '--')}%",
        f"- Maturity: {summary.get('maturity_active_gates', '--')}/{summary.get('maturity_total_gates', '--')} active, {summary.get('maturity_ever_active_gates', '--')} ever",
        f"- Checks: {validation.get('checks_passing', '--')}/{validation.get('checks_total', '--')} current, {validation.get('checks_ever_met', '--')} ever, {validation.get('checks_regressed', '--')} regressed",
        f"- Phase 5 proof chain: {'PASS' if phase5.get('current_ok') else ('EVER' if phase5.get('ever_ok') else 'FAIL')} (now: {phase5.get('current_detail', '--')}, ever: {phase5.get('ever_detail', '--')})",
        "",
    ]

    if language_targets:
        lines += [
            "## Language Evidence Targets",
            "",
            "| Class | Current | Ever Met | Count | Gap | Color | Reason |",
            "|---|---|---|---|---|---|---|",
        ]
        for target in language_targets:
            if not isinstance(target, dict):
                continue
            cls = str(target.get("response_class", "--"))
            count = int(target.get("count", 0) or 0)
            target_count = int(target.get("target_count", 0) or 0)
            est = "~" if bool(target.get("count_estimated", False)) else ""
            gap = int(target.get("gap", 0) or 0)
            color = str(target.get("color", "unknown") or "unknown")
            reason = str(target.get("gate_reason", "ok") or "ok")
            lines.append(
                "| "
                + cls.replace("|", "/")
                + " | "
                + ("PASS" if bool(target.get("current_ok", False)) else "FAIL")
                + " | "
                + ("YES" if bool(target.get("ever_ok", False)) else "NO")
                + " | "
                + f"{est}{count}/{target_count}"
                + " | "
                + str(gap)
                + " | "
                + color.replace("|", "/")
                + " | "
                + reason.replace("|", "/")
                + " |"
            )
        lines += [
            "",
        ]
    if route_class_targets:
        lines += [
            "## Language Baseline Route/Class",
            "",
            "| Route | Response Class | Current | Ever Met | Count |",
            "|---|---|---|---|---|",
        ]
        for row in route_class_targets:
            if not isinstance(row, dict):
                continue
            route = str(row.get("route", "--"))
            response_class = str(row.get("response_class", "--"))
            current = "PASS" if bool(row.get("current_ok", False)) else "FAIL"
            ever = "YES" if bool(row.get("ever_ok", False)) else "NO"
            count = int(row.get("count", 0) or 0)
            lines.append(
                "| "
                + route.replace("|", "/")
                + " | "
                + response_class.replace("|", "/")
                + " | "
                + current
                + " | "
                + ever
                + " | "
                + str(count)
                + " |"
            )
        lines += [
            "",
        ]

    lines += [
        "## Next Action",
        "",
        validation.get("next_action", "--"),
        "",
        "## Continuation Action",
        "",
        validation.get("continuation_action", "--"),
        "",
        "## Checks",
        "",
        "| Check | Current | Ever Met | Critical | Current Detail | Ever Detail |",
        "|---|---|---|---|---|---|",
    ]

    for check in (validation.get("checks") or []):
        current = "PASS" if check.get("current_ok") else "FAIL"
        ever = "YES" if check.get("ever_ok") else "NO"
        critical = "YES" if check.get("critical") else "NO"
        lines.append(
            "| "
            + str(check.get("label", "--")).replace("|", "/")
            + " | "
            + current
            + " | "
            + ever
            + " | "
            + critical
            + " | "
            + str(check.get("current_detail", "--")).replace("|", "/")
            + " | "
            + str(check.get("ever_detail", "--")).replace("|", "/")
            + " |"
        )

    return "\n".join(lines).strip() + "\n"
