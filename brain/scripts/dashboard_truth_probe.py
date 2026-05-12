#!/usr/bin/env python3
"""Dashboard truth probe.

Walks the dashboard ``/api/full-snapshot`` payload and cross-references it
against on-disk evidence under ``~/.jarvis/``. Flags three classes of
dashboard lies:

  1. **Shape violations** - serializer returned the wrong type (e.g.
     ``self_improve.specialists`` collapsed to raw ``{}`` instead of the
     expected ``{"specialists": list, "distillation": dict}``).
  2. **Empty-where-data-exists** - the snapshot claims a panel is empty
     while there is concrete on-disk evidence the panel should be
     populated (e.g. ``distill_*.jsonl`` rows exist but ``specialists``
     list is empty).
  3. **Attestation boundary violations** - ``prior_attested_ok`` claim
     disagrees with the presence/emptiness of the attestation ledger
     file.

The probe is intentionally:
  - read-only (no writes to state)
  - pure (no runtime reconfiguration)
  - idempotent (safe to rerun)

It is designed to be the machine-verifiable truth surface the dashboard
rebuild (P1.7) will depend on. Until that rebuild, its failure modes
are the single most useful signal for "is what the dashboard says what
the system actually is?".

Usage::

    python -m scripts.dashboard_truth_probe
    python -m scripts.dashboard_truth_probe --host 192.168.1.222
    python -m scripts.dashboard_truth_probe --strict  # nonzero on findings
    python -m scripts.dashboard_truth_probe --offline --snapshot-file snap.json

The ``--offline`` mode lets tests exercise the probe without a running
dashboard: feed it a captured snapshot plus an override evidence root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.error import URLError, HTTPError
from urllib.request import urlopen


# --------------------------------------------------------------------------
# Finding type
# --------------------------------------------------------------------------


SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"
SEVERITY_FAIL = "fail"


@dataclass
class Finding:
    code: str
    severity: str
    path: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "path": self.path,
            "message": self.message,
            "evidence": dict(self.evidence),
        }


# --------------------------------------------------------------------------
# Snapshot fetch
# --------------------------------------------------------------------------


def _fetch_snapshot(host: str, port: int, timeout_s: float) -> dict[str, Any]:
    url = f"http://{host}:{port}/api/full-snapshot"
    with urlopen(url, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected /api/full-snapshot response shape")
    return payload


def _load_snapshot_file(path: str) -> dict[str, Any]:
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected dict at top level")
    return payload


# --------------------------------------------------------------------------
# Check helpers
# --------------------------------------------------------------------------


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _list_training_files(evidence_root: Path) -> list[Path]:
    hemi = evidence_root / ".jarvis" / "hemisphere_training"
    if not hemi.exists() or not hemi.is_dir():
        return []
    return sorted(p for p in hemi.glob("distill_*.jsonl") if p.is_file())


def _training_file_has_rows(path: Path, max_scan: int = 2) -> bool:
    """Return True if at least one non-empty line exists in the file.

    Only peeks at the first few lines for speed.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            for _ in range(max_scan):
                line = f.readline()
                if not line:
                    return False
                if line.strip():
                    return True
    except Exception:
        return False
    return False


def _attestation_ledger_path(evidence_root: Path) -> Path:
    return evidence_root / ".jarvis" / "eval" / "ever_proven_attestation.json"


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_specialists_shape(
    snapshot: dict[str, Any], findings: list[Finding]
) -> None:
    """Verify the self_improve.specialists serializer shape contract."""
    si = snapshot.get("self_improve")
    if not _is_mapping(si):
        findings.append(
            Finding(
                code="shape.self_improve.missing",
                severity=SEVERITY_INFO,
                path="self_improve",
                message="self_improve block absent (engine may not be ready)",
            )
        )
        return
    spec = si.get("specialists")
    if spec is None:
        findings.append(
            Finding(
                code="shape.specialists.absent",
                severity=SEVERITY_INFO,
                path="self_improve.specialists",
                message="specialists block not yet emitted",
            )
        )
        return
    if not _is_mapping(spec):
        findings.append(
            Finding(
                code="shape.specialists.not_dict",
                severity=SEVERITY_FAIL,
                path="self_improve.specialists",
                message=(
                    "specialists block must be a dict of shape "
                    f"{{'specialists': list, 'distillation': dict}}; got "
                    f"{type(spec).__name__}"
                ),
                evidence={"actual_type": type(spec).__name__},
            )
        )
        return
    if "_error" in spec:
        findings.append(
            Finding(
                code="shape.specialists.degraded",
                severity=SEVERITY_WARN,
                path="self_improve.specialists",
                message=(
                    "serializer reported degraded fallback "
                    f"(_error={spec.get('_error')})"
                ),
                evidence={"error_class": str(spec.get("_error"))},
            )
        )
    inner = spec.get("specialists")
    if not isinstance(inner, list):
        findings.append(
            Finding(
                code="shape.specialists.list_wrong",
                severity=SEVERITY_FAIL,
                path="self_improve.specialists.specialists",
                message=(
                    "specialists.specialists must be a list; got "
                    f"{type(inner).__name__}"
                ),
                evidence={"actual_type": type(inner).__name__},
            )
        )
    distill = spec.get("distillation")
    if not _is_mapping(distill):
        findings.append(
            Finding(
                code="shape.specialists.distillation_wrong",
                severity=SEVERITY_FAIL,
                path="self_improve.specialists.distillation",
                message=(
                    "specialists.distillation must be a dict; got "
                    f"{type(distill).__name__}"
                ),
                evidence={"actual_type": type(distill).__name__},
            )
        )


def check_specialists_empty_where_data_exists(
    snapshot: dict[str, Any],
    findings: list[Finding],
    evidence_root: Path,
) -> None:
    """Flag the case where distillation JSONL rows exist but the
    specialists list is empty."""
    si = snapshot.get("self_improve")
    if not _is_mapping(si):
        return
    spec = si.get("specialists")
    if not _is_mapping(spec):
        return
    inner = spec.get("specialists")
    if not isinstance(inner, list) or inner:
        return
    populated_files = [
        p.name for p in _list_training_files(evidence_root) if _training_file_has_rows(p)
    ]
    if populated_files:
        findings.append(
            Finding(
                code="empty.specialists.data_exists",
                severity=SEVERITY_WARN,
                path="self_improve.specialists.specialists",
                message=(
                    "specialists list empty but "
                    f"{len(populated_files)} distillation file(s) have rows; "
                    "either the snapshot loop hasn't caught up or the "
                    "serializer is gated incorrectly"
                ),
                evidence={"populated_files": populated_files},
            )
        )


def check_l3_cache_shape(
    snapshot: dict[str, Any], findings: list[Finding]
) -> None:
    """Verify the Phase 6.5 three-axis separation fields are present."""
    self_improve = snapshot.get("self_improve")
    if not _is_mapping(self_improve):
        return
    l3 = self_improve.get("l3_escalation")
    if l3 is None:
        l3 = snapshot.get("l3_escalation")
    if l3 is None:
        return
    if not _is_mapping(l3):
        findings.append(
            Finding(
                code="shape.l3.not_dict",
                severity=SEVERITY_FAIL,
                path="l3_escalation",
                message=f"l3_escalation must be a dict; got {type(l3).__name__}",
                evidence={"actual_type": type(l3).__name__},
            )
        )
        return
    required = ("current_ok", "prior_attested_ok", "request_ok", "activation_ok")
    missing = [k for k in required if k not in l3]
    if missing:
        findings.append(
            Finding(
                code="shape.l3.missing_fields",
                severity=SEVERITY_FAIL,
                path="l3_escalation",
                message=(
                    "l3_escalation missing required three-axis fields: "
                    f"{missing}"
                ),
                evidence={"missing": missing, "present": sorted(l3.keys())},
            )
        )


def check_attestation_consistency(
    snapshot: dict[str, Any],
    findings: list[Finding],
    evidence_root: Path,
) -> None:
    """Cross-reference the prior_attested claim against the ledger file."""
    autonomy = snapshot.get("autonomy")
    if not _is_mapping(autonomy):
        return
    att = autonomy.get("attestation")
    if not _is_mapping(att):
        return
    claimed = bool(att.get("prior_attested_ok", False))
    ledger_path = _attestation_ledger_path(evidence_root)
    has_file = ledger_path.exists()
    if claimed and not has_file:
        findings.append(
            Finding(
                code="attestation.claim_without_ledger",
                severity=SEVERITY_FAIL,
                path="autonomy.attestation.prior_attested_ok",
                message=(
                    "prior_attested_ok=True but attestation ledger file "
                    f"{ledger_path} does not exist"
                ),
                evidence={"ledger_path": str(ledger_path)},
            )
        )
        return
    if not claimed and has_file:
        try:
            with open(ledger_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if raw:
                findings.append(
                    Finding(
                        code="attestation.ledger_without_claim",
                        severity=SEVERITY_WARN,
                        path="autonomy.attestation.prior_attested_ok",
                        message=(
                            f"attestation ledger {ledger_path} is non-empty "
                            "but prior_attested_ok=False"
                        ),
                        evidence={
                            "ledger_path": str(ledger_path),
                            "record_count": (
                                len(raw) if isinstance(raw, (list, dict)) else 0
                            ),
                        },
                    )
                )
        except Exception:
            return


def check_orphaned_fields(
    snapshot: dict[str, Any], findings: list[Finding]
) -> None:
    """Flag well-known snapshot fields that are present-but-empty without
    reason.

    This is intentionally conservative: we only flag fields where an
    empty value is a known smell, not simply a legitimate "no data yet"
    state. Extend this as the dashboard rebuild (P1.7) lands.
    """
    # autonomy block must be a dict (even if contents are pre-gestation).
    autonomy = snapshot.get("autonomy")
    if autonomy is not None and not _is_mapping(autonomy):
        findings.append(
            Finding(
                code="shape.autonomy.not_dict",
                severity=SEVERITY_FAIL,
                path="autonomy",
                message=(
                    "autonomy block must be a dict; got "
                    f"{type(autonomy).__name__}"
                ),
                evidence={"actual_type": type(autonomy).__name__},
            )
        )


def check_hrr_scene_authority(
    snapshot: dict[str, Any], findings: list[Finding]
) -> None:
    """Verify the P5 mental-world lane reports zero-authority + PRE-MATURE.

    Rules (hard-fail on any violation):

    * ``hrr_scene.status`` must be ``"PRE-MATURE"``.
    * ``hrr_scene.lane`` must be ``"spatial_hrr_mental_world"``.
    * Every authority flag must be ``false`` (``writes_memory``,
      ``writes_beliefs``, ``influences_policy``, ``influences_autonomy``,
      ``soul_integrity_influence``, ``llm_raw_vector_exposure``).
    * ``no_raw_vectors_in_api`` must be ``true``.
    * No key anywhere in the ``hrr_scene`` subtree may be named
      ``vector`` or ``raw_vector`` or expose an ``ndarray`` literal.

    If ``hrr_scene`` is missing entirely, that is not a failure: the
    snapshot builder tolerates missing HRR state. The missing-block
    case is reported as info only.
    """
    hrr_scene = snapshot.get("hrr_scene")
    if hrr_scene is None:
        findings.append(
            Finding(
                code="hrr_scene.missing",
                severity=SEVERITY_INFO,
                path="hrr_scene",
                message="hrr_scene block absent; P5 shadow may be disabled",
                evidence={},
            )
        )
        return

    if not _is_mapping(hrr_scene):
        findings.append(
            Finding(
                code="hrr_scene.shape.not_dict",
                severity=SEVERITY_FAIL,
                path="hrr_scene",
                message=f"hrr_scene must be a dict; got {type(hrr_scene).__name__}",
                evidence={"actual_type": type(hrr_scene).__name__},
            )
        )
        return

    status = hrr_scene.get("status")
    if status != "PRE-MATURE":
        findings.append(
            Finding(
                code="hrr_scene.status.not_pre_mature",
                severity=SEVERITY_FAIL,
                path="hrr_scene.status",
                message=f"hrr_scene.status must remain PRE-MATURE; got {status!r}",
                evidence={"actual": status},
            )
        )

    lane = hrr_scene.get("lane")
    if lane != "spatial_hrr_mental_world":
        findings.append(
            Finding(
                code="hrr_scene.lane.wrong",
                severity=SEVERITY_FAIL,
                path="hrr_scene.lane",
                message=f"hrr_scene.lane must be 'spatial_hrr_mental_world'; got {lane!r}",
                evidence={"actual": lane},
            )
        )

    authority_expectations = {
        "writes_memory": False,
        "writes_beliefs": False,
        "influences_policy": False,
        "influences_autonomy": False,
        "soul_integrity_influence": False,
        "llm_raw_vector_exposure": False,
        "no_raw_vectors_in_api": True,
    }
    for flag, expected in authority_expectations.items():
        actual = hrr_scene.get(flag)
        if actual is not expected:
            findings.append(
                Finding(
                    code=f"hrr_scene.authority.{flag}",
                    severity=SEVERITY_FAIL,
                    path=f"hrr_scene.{flag}",
                    message=(
                        f"hrr_scene.{flag} must be {expected}; got {actual!r}"
                    ),
                    evidence={"expected": expected, "actual": actual},
                )
            )

    # Structural: no raw-vector content in the serialized scene.
    def _walk(obj: Any, path: str) -> None:
        if _is_mapping(obj):
            for k, v in obj.items():
                key_low = str(k).lower()
                if key_low in ("vector", "raw_vector", "composite_vector"):
                    findings.append(
                        Finding(
                            code="hrr_scene.raw_vector_leak",
                            severity=SEVERITY_FAIL,
                            path=f"{path}.{k}",
                            message=f"hrr_scene leaked raw-vector key {k!r}",
                            evidence={"path": f"{path}.{k}"},
                        )
                    )
                _walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _walk(item, f"{path}[{i}]")

    _walk(hrr_scene, "hrr_scene")


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


_CHECKS = (
    ("specialists_shape", check_specialists_shape),
    ("l3_cache_shape", check_l3_cache_shape),
    ("orphaned_fields", check_orphaned_fields),
    ("hrr_scene_authority", check_hrr_scene_authority),
)
_CHECKS_WITH_EVIDENCE = (
    ("specialists_empty_where_data_exists", check_specialists_empty_where_data_exists),
    ("attestation_consistency", check_attestation_consistency),
)


def run_probe(
    snapshot: dict[str, Any],
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    """Run every registered check against ``snapshot`` and return a report.

    ``evidence_root`` defaults to ``$HOME``; tests override it to a
    tmp_path with a synthetic ``.jarvis/`` fixture tree.
    """
    if evidence_root is None:
        evidence_root = Path(os.path.expanduser("~"))
    evidence_root = Path(evidence_root)

    findings: list[Finding] = []
    for _, fn in _CHECKS:
        fn(snapshot, findings)
    for _, fn in _CHECKS_WITH_EVIDENCE:
        fn(snapshot, findings, evidence_root)

    severity_counts = {
        SEVERITY_INFO: 0,
        SEVERITY_WARN: 0,
        SEVERITY_FAIL: 0,
    }
    for f in findings:
        severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

    has_fail = severity_counts.get(SEVERITY_FAIL, 0) > 0
    has_warn = severity_counts.get(SEVERITY_WARN, 0) > 0
    overall = "fail" if has_fail else ("warn" if has_warn else "ok")

    return {
        "ok": overall == "ok",
        "status": overall,
        "evidence_root": str(evidence_root),
        "severity_counts": severity_counts,
        "findings": [f.to_dict() for f in findings],
    }


def _print_report(report: dict[str, Any]) -> None:
    status = report.get("status", "unknown")
    sev = report.get("severity_counts", {}) or {}
    badge = {"ok": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]"}.get(
        status, "[INFO]"
    )
    print(f"{badge} Dashboard truth probe: {status.upper()}")
    print(
        f"  Findings: {sev.get(SEVERITY_FAIL, 0)} fail / "
        f"{sev.get(SEVERITY_WARN, 0)} warn / "
        f"{sev.get(SEVERITY_INFO, 0)} info"
    )
    print(f"  Evidence root: {report.get('evidence_root', '--')}")
    print("")
    findings = report.get("findings") or []
    if not findings:
        print("  (no findings)")
        return
    for f in findings:
        sev_badge = {
            SEVERITY_FAIL: "FAIL",
            SEVERITY_WARN: "WARN",
            SEVERITY_INFO: "INFO",
        }.get(f.get("severity", ""), "INFO")
        print(f"  [{sev_badge}] {f.get('code')} @ {f.get('path')}")
        print(f"         {f.get('message')}")
        evidence = f.get("evidence") or {}
        if evidence:
            ev_compact = ", ".join(
                f"{k}={v!r}" for k, v in evidence.items()
            )
            print(f"         evidence: {ev_compact}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Dashboard truth probe: cross-check /api/full-snapshot against "
            "on-disk evidence and serializer shape contracts."
        ),
    )
    p.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    p.add_argument("--port", type=int, default=9200, help="Dashboard port")
    p.add_argument("--timeout", type=float, default=8.0, help="HTTP timeout")
    p.add_argument(
        "--offline",
        action="store_true",
        help="Skip HTTP fetch; require --snapshot-file",
    )
    p.add_argument(
        "--snapshot-file",
        default="",
        help="Path to a JSON snapshot file (used with --offline or as override)",
    )
    p.add_argument(
        "--evidence-root",
        default="",
        help=(
            "Override root directory for on-disk evidence lookups. "
            "Defaults to $HOME. Useful for offline replay / tests."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON report instead of the summary",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Return nonzero exit when findings include warn or fail",
    )
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    if args.offline or args.snapshot_file:
        if not args.snapshot_file:
            print("[ERROR] --offline requires --snapshot-file")
            return 2
        try:
            snapshot = _load_snapshot_file(args.snapshot_file)
        except Exception as exc:
            print(f"[ERROR] Failed to load snapshot file: {exc}")
            return 2
    else:
        try:
            snapshot = _fetch_snapshot(args.host, args.port, args.timeout)
        except (URLError, HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            print(f"[ERROR] Failed to fetch snapshot: {exc}")
            return 2
        except Exception as exc:
            print(f"[ERROR] Unexpected snapshot fetch failure: {exc}")
            return 2

    evidence_root = (
        Path(os.path.expanduser(args.evidence_root))
        if args.evidence_root
        else Path(os.path.expanduser("~"))
    )
    report = run_probe(snapshot, evidence_root=evidence_root)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)

    if args.strict:
        if report["severity_counts"].get(SEVERITY_FAIL, 0) > 0:
            return 1
        if report["severity_counts"].get(SEVERITY_WARN, 0) > 0:
            return 1
    else:
        if report["severity_counts"].get(SEVERITY_FAIL, 0) > 0:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
