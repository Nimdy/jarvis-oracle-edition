"""Code quality feature encoding for the CODE_QUALITY hemisphere specialist.

Encodes an ImprovementRecord (request + plan + patch + sandbox report +
system context) into a fixed 35-dim [0,1] feature vector.  Also provides
label encoding for the 4-class verdict teacher signal.

Learns to predict the outcome of a self-improvement attempt from features
available before apply.  Teacher labels come from the actual
SystemUpgradeReport.verdict recorded after restart-verify.

Dimension blocks:
  dims  0-7:  Request classification
  dims  8-15: Patch features
  dims 16-22: Sandbox results
  dims 23-25: Diagnostics
  dims 26-27: System context
  dims 28-34: Per-module patch history (Track 6)
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

FEATURE_DIM = 35

_REQUEST_TYPES = [
    "performance_optimization",
    "policy_model_upgrade",
    "consciousness_enhancement",
    "bug_fix",
    "architecture_improvement",
]
_REQUEST_TYPE_INDEX = {t: i for i, t in enumerate(_REQUEST_TYPES)}

_VERDICT_CLASSES = [
    "verified_improved",
    "verified_stable",
    "verified_regressed",
    "rolled_back",
]
_VERDICT_INDEX = {v: i for i, v in enumerate(_VERDICT_CLASSES)}

_PROVIDER_ORDINALS = {
    "coder_server": 0.25,
    "ollama": 0.5,
    "claude": 0.75,
    "openai": 1.0,
    "codegen_service": 0.25,
    "local": 0.5,
    "local_retry": 0.5,
}

_WRITE_CATEGORY_ORDINALS = {
    "consciousness": 0.17,
    "policy": 0.33,
    "hemisphere": 0.5,
    "self_improve": 0.67,
    "memory": 0.83,
    "perception": 1.0,
}


def _clamp(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


class CodeQualityEncoder:
    """Encodes a self-improvement attempt into a 35-dim [0,1] feature vector.

    Block layout:
      dims  0-7:  Request classification
      dims  8-15: Patch features
      dims 16-22: Sandbox results
      dims 23-25: Diagnostics
      dims 26-27: System context
      dims 28-34: Per-module patch history (Track 6)
    """

    @staticmethod
    def encode(record: Any, module_history: dict[str, Any] | None = None) -> list[float]:
        """Produce 35-dim [0,1] feature vector from an ImprovementRecord.

        Also accepts the proposal dict format from improvement_proposals.jsonl.
        ``module_history`` is from get_module_patch_history() — per-module
        patch outcome history for the target module.
        """
        vec = [0.0] * FEATURE_DIM

        request = getattr(record, "request", None)
        plan = getattr(record, "plan", None)
        patch = getattr(record, "patch", None)
        report = getattr(record, "report", None)
        iterations = getattr(record, "iterations", 0)

        if request is None and isinstance(record, dict):
            return CodeQualityEncoder._encode_from_dict(record, module_history)

        # Block 1: Request classification (dims 0-7)
        req_type = getattr(request, "type", "") if request else ""
        idx = _REQUEST_TYPE_INDEX.get(req_type)
        if idx is not None:
            vec[idx] = 1.0
        vec[5] = _clamp(getattr(request, "priority", 0.5) if request else 0.5)
        vec[6] = _clamp(getattr(plan, "estimated_risk", 0.5) if plan else 0.5)
        vec[7] = 1.0 if (getattr(request, "requires_approval", False) if request else False) else 0.0

        # Block 2: Patch features (dims 8-15)
        if patch:
            files = getattr(patch, "files", []) or []
            n_files = len(files)
            total_lines = 0
            n_new = 0
            for fd in files:
                new_content = getattr(fd, "new_content", "")
                orig_content = getattr(fd, "original_content", "")
                if new_content:
                    total_lines += abs(len(new_content.splitlines()) - len(orig_content.splitlines()))
                if not orig_content and new_content:
                    n_new += 1

            vec[8] = _clamp(n_files / 3.0)
            vec[9] = _clamp(total_lines / 500.0)
            vec[10] = _clamp(n_new / 1.0)
            vec[11] = _clamp(getattr(patch, "confidence", 0.5))
            escalations = patch.check_capability_escalation() if hasattr(patch, "check_capability_escalation") else []
            vec[12] = 1.0 if escalations else 0.0
        vec[13] = _clamp((iterations - 1) / 4.0) if iterations > 0 else 0.0
        provider = getattr(patch, "provider", "") if patch else ""
        vec[14] = _clamp(_PROVIDER_ORDINALS.get(provider, 0.5))
        write_cat = getattr(plan, "write_category", "self_improve") if plan else "self_improve"
        vec[15] = _clamp(_WRITE_CATEGORY_ORDINALS.get(write_cat, 0.5))

        # Block 3: Sandbox results (dims 16-22)
        if report:
            vec[16] = 1.0 if getattr(report, "lint_passed", False) else 0.0
            vec[17] = 1.0 if getattr(report, "lint_executed", False) else 0.0
            vec[18] = 1.0 if getattr(report, "all_tests_passed", False) else 0.0
            vec[19] = 1.0 if getattr(report, "tests_executed", False) else 0.0
            vec[20] = 1.0 if getattr(report, "sim_passed", False) else 0.0
            vec[21] = 1.0 if getattr(report, "sim_executed", False) else 0.0
            p95_before = getattr(report, "sim_p95_before", 0.0)
            p95_after = getattr(report, "sim_p95_after", 0.0)
            delta = (p95_after - p95_before) / 100.0
            vec[22] = _clamp((delta + 1.0) / 2.0)

        # Block 4: Diagnostics (dims 23-25)
        if report:
            diags = getattr(report, "diagnostics", []) or []
            vec[23] = _clamp(len(diags) / 20.0)
            vec[24] = 1.0 if (hasattr(report, "has_silent_stubs") and report.has_silent_stubs()) else 0.0
            rec = getattr(report, "recommendation", "manual_review")
            if rec == "promote":
                vec[25] = 1.0
            elif rec == "rollback":
                vec[25] = 0.0
            else:
                vec[25] = 0.5

        # Block 5: System context (dims 26-27)
        vec[26] = _clamp(getattr(record, "quarantine_pressure", 0.0) if hasattr(record, "quarantine_pressure") else 0.0)
        vec[27] = _clamp(getattr(record, "soul_integrity", 1.0) if hasattr(record, "soul_integrity") else 1.0)

        # Block 6: Per-module patch history (dims 28-34) — Track 6
        CodeQualityEncoder._encode_module_history(vec, module_history)

        return vec

    @staticmethod
    def _encode_from_dict(d: dict[str, Any], module_history: dict[str, Any] | None = None) -> list[float]:
        """Encode from the JSONL proposal dict format."""
        vec = [0.0] * FEATURE_DIM

        what = d.get("what", {})
        why = d.get("why", {})
        where = d.get("where", {})
        who = d.get("who", {})
        sandbox = d.get("sandbox", {}) or {}

        req_type = what.get("type", "")
        idx = _REQUEST_TYPE_INDEX.get(req_type)
        if idx is not None:
            vec[idx] = 1.0
        vec[5] = _clamp(why.get("priority", 0.5))
        vec[6] = 0.5
        vec[7] = 0.0

        files_mod = where.get("files_modified", []) or []
        vec[8] = _clamp(len(files_mod) / 3.0)
        vec[9] = 0.5
        vec[10] = 0.0
        vec[11] = 0.5
        vec[12] = 0.0
        vec[13] = _clamp((d.get("iterations", 1) - 1) / 4.0)
        vec[14] = _clamp(_PROVIDER_ORDINALS.get(who.get("provider", ""), 0.5))
        vec[15] = 0.5

        vec[16] = 1.0 if sandbox.get("lint_passed") else 0.0
        vec[17] = 1.0 if sandbox.get("lint_executed") else 0.0
        vec[18] = 1.0 if sandbox.get("tests_passed") else 0.0
        vec[19] = 1.0 if sandbox.get("tests_executed") else 0.0
        vec[20] = 1.0 if sandbox.get("sim_passed") else 0.0
        vec[21] = 1.0 if sandbox.get("sim_executed") else 0.0
        p95_after = sandbox.get("sim_p95_after", 0.0)
        vec[22] = _clamp((p95_after / 100.0 + 1.0) / 2.0)

        vec[23] = _clamp(sandbox.get("diagnostics_count", 0) / 20.0)
        vec[24] = 1.0 if sandbox.get("has_silent_stubs") else 0.0
        rec = sandbox.get("recommendation", "manual_review")
        if rec == "promote":
            vec[25] = 1.0
        elif rec == "rollback":
            vec[25] = 0.0
        else:
            vec[25] = 0.5

        vec[26] = 0.0
        vec[27] = 1.0

        CodeQualityEncoder._encode_module_history(vec, module_history)

        return vec

    @staticmethod
    def _encode_module_history(vec: list[float], history: dict[str, Any] | None) -> None:
        """Fill dims 28-34 from module patch history."""
        if not history or not history.get("has_history", False):
            # dims 28-33 stay 0.0, dim 34 = has_patch_history flag
            vec[34] = 0.0
            return

        total = history.get("total_patches", 0)
        vec[28] = _clamp(total / 10.0)

        verdicts = history.get("verdict_counts", {})
        n_improved = verdicts.get("improved", 0)
        n_stable = verdicts.get("stable", 0)
        n_regressed = verdicts.get("regressed", 0)
        n_rolled = verdicts.get("rolled_back", 0)
        if total > 0:
            vec[29] = _clamp((n_improved + n_stable) / total)
            vec[30] = _clamp((n_regressed + n_rolled) / total)

        age = history.get("last_patch_age_s", -1.0)
        if age >= 0:
            vec[31] = _clamp(math.exp(-age / 86400.0))

        vec[32] = 1.0 if history.get("recidivism", False) else 0.0
        vec[33] = _clamp(history.get("avg_iterations", 0.0) / 5.0)
        vec[34] = 1.0

    @staticmethod
    def encode_verdict_label(verdict: str) -> list[float]:
        """Convert a SystemUpgradeReport verdict into a 4-class softmax label."""
        label = [0.0] * len(_VERDICT_CLASSES)
        idx = _VERDICT_INDEX.get(verdict)
        if idx is not None:
            label[idx] = 1.0
        else:
            label[1] = 1.0
        return label
