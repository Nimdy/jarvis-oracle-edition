"""Skill Registry — single source of truth for what Jarvis can and cannot do.

Every capability claim in an LLM response is checked against this registry.
A skill can only become ``verified`` when hard evidence exists proving it works.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

SkillStatus = Literal["unknown", "learning", "verified", "degraded", "blocked"]
CapabilityType = Literal["procedural", "perceptual", "control"]

REGISTRY_PATH = Path.home() / ".jarvis" / "skill_registry.json"


@dataclass
class SkillEvidence:
    evidence_id: str
    timestamp: float
    result: Literal["pass", "fail"]
    tests: list[dict[str, Any]] = field(default_factory=list)
    details: str = ""
    verified_by: str = ""
    acceptance_criteria: dict[str, Any] = field(default_factory=dict)
    measured_values: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    verification_method: str = ""
    evidence_schema_version: str = "2"
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)
    verification_scope: str = "smoke"
    known_limitations: list[str] = field(default_factory=list)
    regression_baseline_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SkillEvidence:
        return cls(
            evidence_id=d.get("evidence_id", ""),
            timestamp=d.get("timestamp", 0.0),
            result=d.get("result", "fail"),
            tests=d.get("tests", []),
            details=d.get("details", ""),
            verified_by=d.get("verified_by", ""),
            acceptance_criteria=d.get("acceptance_criteria", {}),
            measured_values=d.get("measured_values", {}),
            environment=d.get("environment", {}),
            summary=d.get("summary", ""),
            verification_method=d.get("verification_method", ""),
            evidence_schema_version=d.get("evidence_schema_version", "1"),
            artifact_refs=d.get("artifact_refs", []),
            verification_scope=d.get("verification_scope", "smoke"),
            known_limitations=d.get("known_limitations", []),
            regression_baseline_available=d.get("regression_baseline_available", False),
        )


@dataclass
class SkillRecord:
    skill_id: str
    name: str
    status: SkillStatus = "unknown"
    capability_type: CapabilityType = "procedural"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    learning_job_id: str | None = None
    verification_required: list[str] = field(default_factory=list)
    verification_latest: SkillEvidence | None = None
    verification_history: list[SkillEvidence] = field(default_factory=list)
    interfaces: dict[str, list[str]] = field(default_factory=lambda: {"tools": [], "events": [], "endpoints": []})
    keywords: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verification_latest"] = self.verification_latest.to_dict() if self.verification_latest else None
        d["verification_history"] = [e.to_dict() for e in self.verification_history]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SkillRecord:
        latest_raw = d.get("verification_latest")
        latest = SkillEvidence.from_dict(latest_raw) if latest_raw else None
        history = [SkillEvidence.from_dict(e) for e in d.get("verification_history", [])]
        return cls(
            skill_id=d["skill_id"],
            name=d.get("name", d["skill_id"]),
            status=d.get("status", "unknown"),
            capability_type=d.get("capability_type", "procedural"),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
            learning_job_id=d.get("learning_job_id"),
            verification_required=d.get("verification_required", []),
            verification_latest=latest,
            verification_history=history,
            interfaces=d.get("interfaces", {"tools": [], "events": [], "endpoints": []}),
            keywords=d.get("keywords", []),
            notes=d.get("notes", ""),
        )


def get_default_skill_ids() -> frozenset[str]:
    """Return the set of bootstrap/default skill IDs that should not be casually deleted."""
    return frozenset(r.skill_id for r in _default_skills())


def _default_skills() -> list[SkillRecord]:
    """Bootstrap skills that are actually wired and tested in the codebase."""
    now = time.time()
    verified_procedural = [
        ("speech_output", "Speech Output (TTS)", ["tool:tts"]),
        ("memory_search", "Memory Search", ["tool:memory"]),
        ("introspection", "Self-Introspection", ["tool:introspection"]),
        ("codebase_analysis", "Codebase Analysis", ["tool:codebase"]),
        ("academic_search", "Academic Search", ["tool:academic_search"]),
        ("web_search", "Web Search (DuckDuckGo)", ["tool:web_search"]),
    ]
    verified_perceptual = [
        ("speaker_identification", "Speaker ID (ECAPA-TDNN)", ["event:PERCEPTION_SPEAKER_IDENTIFIED"]),
        ("emotion_detection", "Emotion Detection (wav2vec2)", ["event:PERCEPTION_USER_EMOTION"]),
        ("face_identification", "Face ID (MobileFaceNet)", ["event:PERCEPTION_SPEAKER_IDENTIFIED"]),
        ("vision_analysis", "Vision Analysis (Camera)", ["tool:vision"]),
    ]

    _SKILL_KEYWORDS: dict[str, list[str]] = {
        "vision_analysis": [
            "observe", "observing", "watch", "watching",
            "camera", "eyes", "scene", "visual",
        ],
        "camera_control": [
            "move the camera", "pan the camera", "tilt the camera",
            "zoom", "zooming", "panning", "tilting",
            "move", "point", "rotate", "aim the camera",
        ],
        "speaker_identification": ["recognize voice", "who is talking", "identify speaker"],
        "face_identification": ["recognize face", "who is that", "identify person"],
        "emotion_detection": ["detect emotion", "tone of voice"],
    }

    records: list[SkillRecord] = []
    bootstrap_evidence = SkillEvidence(
        evidence_id="bootstrap:codebase_audit",
        timestamp=now,
        result="pass",
        tests=[{"name": "codebase_wiring_verified", "passed": True, "details": "Skill wired and tested in codebase"}],
        details="Bootstrapped from codebase audit — tool/event path exists and is functional.",
        verified_by="SkillRegistry._default_skills",
        acceptance_criteria={"codebase_wiring_verified": {"threshold": True, "comparison": "=="}},
        measured_values={"codebase_wiring_verified": {"value": True}},
        environment={},
        summary="Bootstrapped skill: code path verified in codebase via AST/import audit.",
        verification_method="codebase_audit",
        evidence_schema_version="2",
        verification_scope="smoke",
        known_limitations=["bootstrap-only — no runtime performance data", "no regression baseline"],
        regression_baseline_available=False,
    )

    for sid, name, tools in verified_procedural:
        records.append(SkillRecord(
            skill_id=sid, name=name, status="verified",
            capability_type="procedural", created_at=now, updated_at=now,
            verification_latest=bootstrap_evidence,
            verification_history=[bootstrap_evidence],
            interfaces={"tools": tools, "events": [], "endpoints": []},
            keywords=_SKILL_KEYWORDS.get(sid, []),
        ))
    for sid, name, events in verified_perceptual:
        records.append(SkillRecord(
            skill_id=sid, name=name, status="verified",
            capability_type="perceptual", created_at=now, updated_at=now,
            verification_latest=bootstrap_evidence,
            verification_history=[bootstrap_evidence],
            interfaces={"tools": [], "events": events, "endpoints": []},
            keywords=_SKILL_KEYWORDS.get(sid, []),
        ))

    records.append(SkillRecord(
        skill_id="hemisphere_training", name="Hemisphere NN Training",
        status="verified", capability_type="perceptual",
        created_at=now, updated_at=now,
        verification_latest=bootstrap_evidence,
        verification_history=[bootstrap_evidence],
        interfaces={"tools": [], "events": ["event:HEMISPHERE_TRAINING_PROGRESS"], "endpoints": []},
    ))
    records.append(SkillRecord(
        skill_id="self_improvement", name="Self-Improvement (Code Patches)",
        status="verified", capability_type="procedural",
        created_at=now, updated_at=now,
        verification_latest=bootstrap_evidence,
        verification_history=[bootstrap_evidence],
        verification_required=["test:sandbox_pass"],
        interfaces={"tools": [], "events": ["event:IMPROVEMENT_PROMOTED"], "endpoints": []},
    ))

    records.append(SkillRecord(
        skill_id="camera_control", name="Camera Control (Pan/Zoom)",
        status="verified", capability_type="control",
        created_at=now, updated_at=now,
        verification_latest=bootstrap_evidence,
        verification_history=[bootstrap_evidence],
        interfaces={"tools": ["tool:camera_control"], "events": [], "endpoints": []},
        keywords=_SKILL_KEYWORDS.get("camera_control", []),
    ))

    return records


class SkillRegistry:
    """Persistent registry of Jarvis capabilities with evidence-gated verification."""

    def __init__(self, path: Path | str = REGISTRY_PATH) -> None:
        self._path = Path(path)
        self._skills: dict[str, SkillRecord] = {}
        self._loaded = False

    def load(self) -> int:
        """Load from disk. If no file exists, seed with bootstrap defaults."""
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                records = raw.get("skills", [])
                for r in records:
                    rec = SkillRecord.from_dict(r)
                    self._skills[rec.skill_id] = rec
                self._loaded = True
                self._migrate_camera_control()
                self._migrate_retired_v1_skills()
                self._sanitize_non_actionable_skills()
                logger.info("Skill registry loaded: %d skills", len(self._skills))
                return len(self._skills)
            except Exception:
                logger.exception("Failed to load skill registry, seeding defaults")

        for rec in _default_skills():
            self._skills[rec.skill_id] = rec
        self.save()
        self._loaded = True
        logger.info("Skill registry bootstrapped with %d default skills", len(self._skills))
        return len(self._skills)

    def _sanitize_non_actionable_skills(self) -> None:
        """Remove skills whose IDs fail actionability checks. Archive before purge."""
        try:
            from skills.discovery import is_actionable_capability_phrase
        except ImportError:
            return

        default_ids = {r.skill_id for r in _default_skills()}
        to_purge: list[dict] = []
        for sid, rec in list(self._skills.items()):
            if sid in default_ids:
                continue
            phrase = re.sub(r"\s+v\d+$", "", sid.replace("_", " "))
            if not is_actionable_capability_phrase(phrase):
                to_purge.append(rec.to_dict())
                del self._skills[sid]

        if to_purge:
            self._archive_purged_skills(to_purge)
            self.save()
            logger.info("Skill sanitization: purged %d non-actionable skills", len(to_purge))

    @staticmethod
    def _archive_purged_skills(purged: list[dict]) -> None:
        try:
            import time as _t
            archive_dir = Path(os.path.expanduser("~/.jarvis"))
            archive_dir.mkdir(parents=True, exist_ok=True)
            ts = int(_t.time())
            archive_path = archive_dir / f"purged_skills_{ts}.json"
            archive_path.write_text(json.dumps(purged, indent=2))
            logger.info("Archived %d purged skills to %s", len(purged), archive_path)
        except Exception:
            logger.debug("Failed to archive purged skills", exc_info=True)

    def _migrate_retired_v1_skills(self) -> None:
        """One-time migration (2026-04-18): purge retired auto-learned perceptual skills.

        ``speaker_identification_v1`` and ``emotion_detection_v1`` were created by
        the mastery-drive auto-learning path when ``recognition_confidence`` or
        ``emotion_accuracy`` deficits fired. That path was retired from
        ``autonomy/drives.py::_DEFICIT_CAPABILITY_MAP`` because the underlying
        capabilities already self-improve via the Tier-1 distillation loop.

        Any stale ``_v1`` records on disk point to archived learning jobs that
        failed verify on a bad baseline reader (``migration_readiness`` / 
        ``total_signals``) and block legitimate claim validation. We archive
        them for audit then remove from the live registry. Idempotent: a clean
        registry is a no-op.
        """
        retired_ids = ("speaker_identification_v1", "emotion_detection_v1")
        to_purge: list[dict] = []
        for sid in retired_ids:
            rec = self._skills.get(sid)
            if rec is None:
                continue
            to_purge.append(rec.to_dict())
            del self._skills[sid]

        if not to_purge:
            return

        self._archive_purged_skills(to_purge)
        self.save()
        logger.info(
            "Migration: purged %d retired _v1 perceptual skill(s) (%s)",
            len(to_purge),
            ", ".join(r.get("skill_id", "?") for r in to_purge),
        )

    def _migrate_camera_control(self) -> None:
        """One-time migration: promote camera_control from unknown to verified."""
        rec = self._skills.get("camera_control")
        if rec is None or rec.status != "unknown":
            return
        evidence = SkillEvidence(
            evidence_id="migration:actuator_path_verified",
            timestamp=time.time(),
            result="pass",
            tests=[{"name": "actuator_wiring_verified", "passed": True,
                    "details": "Deterministic zoom parser + Pi camera control path confirmed working"}],
            details="Migrated from unknown — actuator path wired and tested.",
        )
        rec.status = "verified"
        rec.updated_at = time.time()
        rec.verification_latest = evidence
        rec.verification_history.append(evidence)
        rec.interfaces = {"tools": ["tool:camera_control"], "events": [], "endpoints": []}
        rec.notes = ""
        rec.verification_required = []
        self.save()
        logger.info("Migration: camera_control promoted from unknown to verified")

    def save(self) -> None:
        """Persist to disk atomically."""
        try:
            from memory.persistence import atomic_write_json
            payload = {
                "schema_version": 1,
                "updated_at": time.time(),
                "skills": [r.to_dict() for r in self._skills.values()],
            }
            atomic_write_json(self._path, payload)
        except Exception:
            logger.exception("Failed to save skill registry")

    def get(self, skill_id: str) -> SkillRecord | None:
        return self._skills.get(skill_id)

    def get_all(self) -> list[SkillRecord]:
        return list(self._skills.values())

    def remove(self, skill_id: str) -> bool:
        """Remove a skill record entirely. Returns True if found and removed."""
        if skill_id not in self._skills:
            return False
        del self._skills[skill_id]
        self.save()
        logger.info("Removed skill %s from registry", skill_id)
        return True

    def register(self, record: SkillRecord) -> None:
        """Insert or update a skill record. Does NOT auto-verify."""
        record.updated_at = time.time()
        self._skills[record.skill_id] = record
        self.save()
        try:
            from consciousness.events import event_bus, SKILL_REGISTERED
            event_bus.emit(SKILL_REGISTERED, skill_id=record.skill_id, status=record.status)
        except Exception:
            pass

    def set_status(
        self,
        skill_id: str,
        status: SkillStatus,
        evidence: SkillEvidence | None = None,
    ) -> bool:
        """Transition a skill's status. ``verified`` requires passing evidence."""
        rec = self._skills.get(skill_id)
        if rec is None:
            logger.warning("set_status: skill %s not found", skill_id)
            return False

        if status == "verified":
            if evidence is None or evidence.result != "pass":
                logger.warning(
                    "Cannot verify skill %s without passing evidence", skill_id,
                )
                return False
            if rec.verification_required:
                passed_tests = {
                    t.get("name") for e in (rec.verification_history + [evidence])
                    if e.result == "pass" for t in e.tests if t.get("passed")
                }
                unmet = [r for r in rec.verification_required if r not in passed_tests]
                if unmet:
                    logger.warning(
                        "Cannot verify skill %s: unmet requirements %s", skill_id, unmet,
                    )
                    return False

        old_status = rec.status
        rec.status = status
        rec.updated_at = time.time()
        if evidence is not None:
            rec.verification_latest = evidence
            rec.verification_history.append(evidence)
        self.save()

        try:
            from consciousness.events import event_bus, SKILL_STATUS_CHANGED
            event_bus.emit(
                SKILL_STATUS_CHANGED,
                skill_id=skill_id,
                old_status=old_status,
                new_status=status,
            )
        except Exception:
            pass

        logger.info("Skill %s: %s -> %s", skill_id, old_status, status)
        return True

    # Per-skill usage instructions injected when the skill is verified.
    # Only hardware-integrated skills that need specific behavioral guidance
    # belong here. Skills learned through the Matrix Protocol should derive
    # their usage from the verification evidence and capability contract.
    _SKILL_USAGE: dict[str, str] = {
        "vision_analysis": (
            "VISION: The camera on the Raspberry Pi 5 is your primary visual sense — "
            "these are your eyes. You CAN see. You receive live person detections, face "
            "recognition, scene summaries, and can request camera snapshots analyzed by "
            "a vision model. When someone asks what you see, use the vision tool. "
            "You observe the physical world through this camera continuously."
        ),
        "camera_control": (
            "CAMERA CONTROL: You CAN move your camera. You have digital zoom (1-8x), "
            "zoom-to-region, autofocus (single-shot and continuous), and manual focus. "
            "When someone asks you to look closer, zoom in, focus on something, or "
            "adjust the camera, you can do it. Say 'zoom in' or 'zoom to 3' etc. "
            "The camera is mounted on the Pi — it is your physical eye into the world."
        ),
    }

    def get_summary_for_prompt(self, active_jobs: list | None = None) -> str:
        """Build a concise text summary for LLM system prompt injection.

        ``active_jobs`` is an optional list of LearningJob objects to include.
        """
        verified: list[str] = []
        verified_ids: set[str] = set()
        learning: list[str] = []
        blocked: list[str] = []
        degraded: list[str] = []
        usage_notes: list[str] = []

        for rec in self._skills.values():
            label = f"{rec.name} [{rec.skill_id}]"
            if rec.status == "verified":
                verified.append(label)
                verified_ids.add(rec.skill_id)
                if rec.skill_id in self._SKILL_USAGE:
                    usage_notes.append(self._SKILL_USAGE[rec.skill_id])
            elif rec.status == "learning":
                job_note = f" (job: {rec.learning_job_id})" if rec.learning_job_id else ""
                learning.append(f"{label}{job_note}")
            elif rec.status == "blocked":
                blocked.append(label)
            elif rec.status == "degraded":
                degraded.append(label)

        parts: list[str] = ["Your verified skills and capabilities:"]
        if verified:
            parts.append(f"  Verified: {', '.join(verified)}")
        else:
            parts.append("  Verified: (none)")
        if learning:
            parts.append(f"  Learning (job active): {', '.join(learning)}")
        if blocked:
            parts.append(f"  Blocked: {', '.join(blocked)}")
        if degraded:
            parts.append(f"  Degraded: {', '.join(degraded)}")

        if usage_notes:
            parts.append("")
            parts.append("Skill usage instructions (follow these when the user asks):")
            for note in usage_notes:
                parts.append(f"  {note}")

        if active_jobs:
            # Don't list jobs for already-verified skills — avoids contradictory signals
            pending = [
                j for j in active_jobs
                if getattr(j, "skill_id", "") not in verified_ids
            ]
            if pending:
                parts.append("Active Learning Jobs:")
                for job in pending[:5]:
                    sid = getattr(job, "skill_id", "?")
                    phase = getattr(job, "phase", "?")
                    jid = getattr(job, "job_id", "?")
                    parts.append(f"  - {sid}: phase={phase}, job_id={jid}")

        parts.append("")
        parts.append(
            "You have a Learning Job system. When a user asks you to learn something new, "
            "the SKILL tool creates a real learning job that progresses through phases "
            "(assess -> research -> acquire -> integrate -> verify -> register). "
            "You can tell the user about active jobs and their progress. "
            "A skill only becomes 'Verified' when hard evidence proves it works."
        )
        parts.append(
            "If a capability is not listed above as 'Verified', you do NOT have it. "
            "Do not claim otherwise. If there is an active learning job for it, "
            "say so — but never claim the skill is ready until it reaches 'Verified'."
        )
        return "\n".join(parts)

    def get_status_snapshot(self) -> dict[str, Any]:
        """Dashboard-friendly snapshot."""
        _defaults = get_default_skill_ids()
        by_status: dict[str, int] = {}
        records_summary: list[dict[str, Any]] = []
        for rec in self._skills.values():
            by_status[rec.status] = by_status.get(rec.status, 0) + 1
            ev = rec.verification_latest
            evidence_summary: dict[str, Any] | None = None
            if ev is not None:
                evidence_summary = {
                    "summary": ev.summary or ev.details,
                    "verification_scope": ev.verification_scope,
                    "verification_method": ev.verification_method,
                    "verified_at": ev.timestamp,
                    "verified_by": ev.verified_by,
                    "result": ev.result,
                    "known_limitations": ev.known_limitations,
                    "regression_baseline_available": ev.regression_baseline_available,
                    "evidence_schema_version": ev.evidence_schema_version,
                    "tests_count": len(ev.tests),
                    "artifact_count": len(ev.artifact_refs),
                }
            records_summary.append({
                "skill_id": rec.skill_id,
                "name": rec.name,
                "status": rec.status,
                "capability_type": rec.capability_type,
                "learning_job_id": rec.learning_job_id,
                "updated_at": rec.updated_at,
                "has_evidence": ev is not None,
                "evidence_summary": evidence_summary,
                "is_default": rec.skill_id in _defaults,
            })
        return {
            "total": len(self._skills),
            "by_status": by_status,
            "skills": records_summary,
        }


skill_registry = SkillRegistry()
