"""Capability Discovery Engine — detects recurring gaps and proposes learning.

Components:
  - CapabilityFamilyNormalizer: maps raw skill_ids to canonical families
  - BlockFrequencyTracker: counts blocks at the family level
  - GapAnalyzer: fuses 4 evidence streams into CapabilityGap
  - LearningProposer: dispatches research/retry/propose/defer actions

Design invariants:
  1. Never auto-start risky learning without user consent.
  2. User rejection is durable (7-day suppression per family).
  3. Minimum 3 family blocks before any action, 5 before proposing.
  4. Quarantine-aware: no proposals during elevated pressure.
  5. Research before job creation when prior attempt failed.
  6. Max 1 proposal per conversation, max 1 per 24h (unless user asks).
  7. All evidence is family-normalized — never fragments across aliases.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PERSISTENCE_PATH = os.path.join(os.path.expanduser("~"), ".jarvis", "capability_blocks.json")

_GLOBAL_PROPOSAL_COOLDOWN_S = 86400.0  # 24 hours
_REJECTION_SUPPRESSION_S = 604800.0     # 7 days
_MIN_BLOCKS_FOR_ACTION = 3
_MIN_BLOCKS_FOR_PROPOSAL = 5
_MAX_PENDING_PROPOSALS = 5
_MAX_SURFACE_PHRASES = 10


# ---------------------------------------------------------------------------
# Capability Family Normalization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CapabilityFamily:
    family_id: str
    domain: str
    canonical_name: str
    aliases: frozenset[str]
    builtin: bool


BUILTIN_FAMILIES: dict[str, CapabilityFamily] = {
    "camera_control": CapabilityFamily(
        family_id="camera_control",
        domain="actuator",
        canonical_name="Camera Control",
        aliases=frozenset({
            "camera_control", "zoom_reset", "set_camera_zoom",
            "adjust_zoom", "camera_zoom", "pan_camera", "tilt_camera",
        }),
        builtin=True,
    ),
    "singing": CapabilityFamily(
        family_id="singing",
        domain="creative",
        canonical_name="Singing / Vocal Performance",
        aliases=frozenset({
            "singing_v1", "sing", "hum", "vocal_performance",
            "sing_song", "hum_melody",
        }),
        builtin=False,
    ),
    "drawing": CapabilityFamily(
        family_id="drawing",
        domain="creative",
        canonical_name="Drawing / Image Generation",
        aliases=frozenset({
            "drawing", "draw", "paint", "sketch", "image_generation",
            "generate_image",
        }),
        builtin=False,
    ),
    "speaker_diarization": CapabilityFamily(
        family_id="speaker_diarization",
        domain="perception",
        canonical_name="Speaker Diarization",
        aliases=frozenset({
            "speaker_diarization", "diarization", "diarize",
            "who_is_speaking", "multi_speaker",
        }),
        builtin=False,
    ),
    "robot_arm": CapabilityFamily(
        family_id="robot_arm",
        domain="actuator",
        canonical_name="Robot Arm Control",
        aliases=frozenset({
            "robot_arm", "arm_control", "robotic_arm",
            "move_arm", "grab", "pick_up",
        }),
        builtin=False,
    ),
}

_VERB_TO_DOMAIN: dict[str, str] = {
    "sing": "creative", "hum": "creative", "dance": "creative",
    "draw": "creative", "paint": "creative", "sketch": "creative",
    "zoom": "actuator", "pan": "actuator", "tilt": "actuator",
    "grab": "actuator", "move": "actuator", "pick": "actuator",
}

# Phrases that indicate emotional/complaint content, not capability requests
_AFFECTIVE_REJECT_WORDS = frozenset({
    "better", "worse", "useless", "pointless", "waste", "terrible",
    "horrible", "stupid", "dumb", "annoying", "frustrating",
    "disappointed", "angry", "upset", "shadows", "honest", "honestly",
    "transparent", "truthful", "trust", "lying", "lied", "lie",
    "scary", "creepy", "weird", "crazy", "mad",
})

_STANCE_REJECT_PHRASES = (
    "serve you", "serve me", "better serve", "operate in",
    "behind my back", "without telling", "don't do that",
    "stop doing", "never do", "i'll be", "you should be",
    "you need to be", "you shouldn't", "i don't want you to",
    "i don't like", "i hate", "that's not", "what you did",
    "how dare", "who told you", "don't ever",
)

_CAPABILITY_VERBS = frozenset({
    "sing", "hum", "dance", "draw", "paint", "sketch", "play",
    "compose", "write", "generate", "create", "build", "make",
    "control", "zoom", "pan", "tilt", "grab", "move", "pick",
    "detect", "recognize", "identify", "track", "analyze",
    "translate", "transcribe", "summarize", "search",
})

_CAPABILITY_NOUNS = frozenset({
    "image", "song", "music", "picture", "drawing", "video",
    "camera", "arm", "robot", "speaker", "voice", "face",
    "emotion", "language", "code", "document", "file",
    "data", "csv", "json", "xml", "text", "api", "web",
})


def is_actionable_capability_phrase(text: str) -> bool:
    """Returns True if the text describes a concrete capability, not a complaint."""
    lower = text.strip().lower()
    tokens = set(lower.split())

    if len(tokens) < 2:
        return False

    affective_count = len(tokens & _AFFECTIVE_REJECT_WORDS)
    if affective_count >= 2:
        return False
    if affective_count >= 1 and len(tokens) <= 4:
        return False

    for phrase in _STANCE_REJECT_PHRASES:
        if phrase in lower:
            return False

    has_verb = bool(tokens & _CAPABILITY_VERBS)
    has_noun = bool(tokens & _CAPABILITY_NOUNS)
    has_domain_verb = any(tok in _VERB_TO_DOMAIN for tok in tokens)

    return has_verb or has_noun or has_domain_verb

_alias_index: dict[str, str] = {}
for fam_id, fam in BUILTIN_FAMILIES.items():
    for alias in fam.aliases:
        _alias_index[alias] = fam_id

_REJECTED_FAMILY = CapabilityFamily(
    family_id="_rejected",
    domain="rejected",
    canonical_name="Rejected (not a capability)",
    aliases=frozenset(),
    builtin=False,
)


class CapabilityFamilyNormalizer:
    """Maps raw skill_ids and claim text to canonical CapabilityFamily."""

    def __init__(self) -> None:
        self._dynamic_families: dict[str, CapabilityFamily] = {}

    def normalize(self, skill_id: str | None, claimed_text: str = "") -> CapabilityFamily:
        if skill_id:
            sid_lower = re.sub(r"_v\d+$", "", skill_id.lower())
            if sid_lower in _alias_index:
                return BUILTIN_FAMILIES[_alias_index[sid_lower]]
            for alias, fam_id in _alias_index.items():
                if alias in sid_lower or sid_lower in alias:
                    return BUILTIN_FAMILIES[fam_id]

        tokens = set()
        if skill_id:
            tokens.update(skill_id.lower().replace("_", " ").split())
        if claimed_text:
            tokens.update(claimed_text.lower().split())

        best_score = 0.0
        best_fam: CapabilityFamily | None = None
        for fam in BUILTIN_FAMILIES.values():
            overlap = len(tokens & fam.aliases)
            if overlap > 0:
                score = overlap / max(len(tokens), len(fam.aliases))
                if score > best_score and score >= 0.15:
                    best_score = score
                    best_fam = fam
        if best_fam:
            return best_fam

        raw_text = claimed_text or (skill_id or "").replace("_", " ")
        if not is_actionable_capability_phrase(raw_text):
            return _REJECTED_FAMILY

        domain = "unknown"
        for tok in tokens:
            if tok in _VERB_TO_DOMAIN:
                domain = _VERB_TO_DOMAIN[tok]
                break

        key = skill_id or "_".join(sorted(tokens)[:4]) or "unknown"
        key = key.lower().replace(" ", "_")[:60]

        if key in self._dynamic_families:
            return self._dynamic_families[key]

        fam = CapabilityFamily(
            family_id=key,
            domain=domain,
            canonical_name=key.replace("_", " ").title(),
            aliases=frozenset({key}),
            builtin=False,
        )
        self._dynamic_families[key] = fam
        return fam


# ---------------------------------------------------------------------------
# Block Frequency Tracker
# ---------------------------------------------------------------------------

@dataclass
class BlockPattern:
    family: CapabilityFamily
    block_count: int = 0
    session_blocks: int = 0
    last_blocked: float = 0.0
    surface_phrases: list[str] = field(default_factory=list)
    has_active_job: bool = False
    job_status: str | None = None
    job_failure_reason: str | None = None


class BlockFrequencyTracker:
    """Tracks blocked capability claims at the family level."""

    def __init__(self, normalizer: CapabilityFamilyNormalizer) -> None:
        self._normalizer = normalizer
        self._patterns: dict[str, BlockPattern] = {}
        self._load()

    def record_block(
        self,
        skill_id: str | None,
        claimed_text: str,
        registry: Any = None,
        job_store: Any = None,
    ) -> BlockPattern:
        family = self._normalizer.normalize(skill_id, claimed_text)
        if family.family_id == "_rejected":
            return BlockPattern(family=family)
        pattern = self._patterns.get(family.family_id)
        if pattern is None:
            pattern = BlockPattern(family=family)
            self._patterns[family.family_id] = pattern

        pattern.block_count += 1
        pattern.session_blocks += 1
        pattern.last_blocked = time.time()
        if claimed_text and claimed_text not in pattern.surface_phrases:
            pattern.surface_phrases.append(claimed_text)
            if len(pattern.surface_phrases) > _MAX_SURFACE_PHRASES:
                pattern.surface_phrases = pattern.surface_phrases[-_MAX_SURFACE_PHRASES:]

        self._update_job_status(pattern, registry, job_store)
        self._save()
        return pattern

    def get_actionable_patterns(self) -> list[BlockPattern]:
        """Return families with enough evidence to consider action."""
        return [
            p for p in self._patterns.values()
            if p.session_blocks >= _MIN_BLOCKS_FOR_ACTION
            and p.job_status not in ("active", "completed")
        ]

    def get_all_patterns(self) -> dict[str, BlockPattern]:
        return dict(self._patterns)

    def get_snapshot(self) -> list[dict[str, Any]]:
        return sorted(
            [
                {
                    "family_id": p.family.family_id,
                    "domain": p.family.domain,
                    "canonical_name": p.family.canonical_name,
                    "block_count": p.block_count,
                    "session_blocks": p.session_blocks,
                    "last_blocked": p.last_blocked,
                    "surface_phrases": p.surface_phrases[-3:],
                    "job_status": p.job_status,
                }
                for p in self._patterns.values()
            ],
            key=lambda x: x["block_count"],
            reverse=True,
        )

    def _update_job_status(self, pattern: BlockPattern, registry: Any, job_store: Any) -> None:
        if not job_store:
            return
        try:
            all_jobs = job_store.load_all() if hasattr(job_store, "load_all") else {}
            for job in (all_jobs.values() if isinstance(all_jobs, dict) else all_jobs):
                sid = getattr(job, "skill_id", "")
                if sid and sid.lower() in (a.lower() for a in pattern.family.aliases):
                    pattern.has_active_job = getattr(job, "status", "") == "active"
                    pattern.job_status = getattr(job, "status", None)
                    if getattr(job, "failure", None):
                        pattern.job_failure_reason = (job.failure.get("last_error") if isinstance(job.failure, dict) else getattr(job.failure, "last_error", None)) or str(job.failure)
                    return
        except Exception:
            pass

    def _save(self) -> None:
        try:
            path = Path(_PERSISTENCE_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            data: dict[str, Any] = {}
            for fid, p in self._patterns.items():
                data[fid] = {
                    "family_id": p.family.family_id,
                    "domain": p.family.domain,
                    "canonical_name": p.family.canonical_name,
                    "block_count": p.block_count,
                    "last_blocked": p.last_blocked,
                    "surface_phrases": p.surface_phrases,
                    "job_status": p.job_status,
                    "job_failure_reason": p.job_failure_reason,
                }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(path)
        except Exception:
            logger.debug("Failed to save capability blocks", exc_info=True)

    def _load(self) -> None:
        try:
            path = Path(_PERSISTENCE_PATH)
            if not path.exists():
                return
            data = json.loads(path.read_text())
            purged: list[dict[str, Any]] = []
            kept = 0
            for fid, d in data.items():
                # Startup sanitization: reject _rejected sentinel and non-actionable families
                if fid == "_rejected":
                    purged.append({"family_id": fid, "reason": "rejected_sentinel", **d})
                    continue
                canonical = d.get("canonical_name", fid).replace("_", " ")
                if not is_actionable_capability_phrase(canonical):
                    purged.append({"family_id": fid, "reason": "non_actionable", **d})
                    continue

                family = BUILTIN_FAMILIES.get(fid)
                if family is None:
                    family = CapabilityFamily(
                        family_id=fid,
                        domain=d.get("domain", "unknown"),
                        canonical_name=d.get("canonical_name", fid),
                        aliases=frozenset({fid}),
                        builtin=False,
                    )
                self._patterns[fid] = BlockPattern(
                    family=family,
                    block_count=d.get("block_count", 0),
                    session_blocks=0,
                    last_blocked=d.get("last_blocked", 0.0),
                    surface_phrases=d.get("surface_phrases", []),
                    job_status=d.get("job_status"),
                    job_failure_reason=d.get("job_failure_reason"),
                )
                kept += 1

            if purged:
                self._archive_purged_families(purged)
                logger.info(
                    "Capability sanitization: purged %d families, kept %d", len(purged), kept,
                )
        except Exception:
            logger.debug("Failed to load capability blocks", exc_info=True)

    _MAX_PURGE_ARCHIVES = 5

    @staticmethod
    def _archive_purged_families(purged: list[dict[str, Any]]) -> None:
        """Write purged families to a single rolling archive file.

        Keeps only the latest _MAX_PURGE_ARCHIVES files and removes older ones.
        """
        try:
            import time as _t
            archive_dir = Path(os.path.expanduser("~/.jarvis"))
            archive_dir.mkdir(parents=True, exist_ok=True)
            ts = int(_t.time())
            archive_path = archive_dir / f"purged_capability_families_{ts}.json"
            archive_path.write_text(json.dumps(purged, indent=2))

            existing = sorted(archive_dir.glob("purged_capability_families_*.json"))
            max_keep = BlockFrequencyTracker._MAX_PURGE_ARCHIVES
            if len(existing) > max_keep:
                for old in existing[:-max_keep]:
                    try:
                        old.unlink()
                    except OSError:
                        pass
            logger.info("Archived %d purged capability families to %s", len(purged), archive_path)
        except Exception:
            logger.debug("Failed to archive purged families", exc_info=True)


# ---------------------------------------------------------------------------
# Gap Analyzer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CapabilityGap:
    family: CapabilityFamily
    evidence_strength: float
    evidence_sources: tuple[str, ...]
    block_count: int
    surface_phrases: tuple[str, ...]
    has_prior_attempt: bool
    prior_failure_reason: str | None
    suggested_action: str
    priority: float


class GapAnalyzer:
    """Fuses 4 evidence streams into CapabilityGap objects."""

    def analyze(
        self,
        tracker: BlockFrequencyTracker,
        job_store: Any = None,
    ) -> list[CapabilityGap]:
        gaps: list[CapabilityGap] = []

        for pattern in tracker.get_actionable_patterns():
            sources: list[str] = ["block_frequency"]
            strength = min(1.0, pattern.session_blocks / 10.0)

            has_prior = pattern.job_status in ("blocked", "failed")
            if has_prior:
                sources.append("failed_job")
                strength = min(1.0, strength + 0.2)

            action = self._decide_action(pattern, strength)
            priority = self._compute_priority(pattern, strength)

            gaps.append(CapabilityGap(
                family=pattern.family,
                evidence_strength=round(strength, 3),
                evidence_sources=tuple(sources),
                block_count=pattern.block_count,
                surface_phrases=tuple(pattern.surface_phrases[-5:]),
                has_prior_attempt=has_prior,
                prior_failure_reason=pattern.job_failure_reason,
                suggested_action=action,
                priority=round(priority, 3),
            ))

        gaps.sort(key=lambda g: g.priority, reverse=True)
        return gaps

    @staticmethod
    def _decide_action(pattern: BlockPattern, strength: float) -> str:
        if pattern.block_count >= _MIN_BLOCKS_FOR_PROPOSAL:
            if pattern.job_status in ("blocked", "failed") and pattern.job_failure_reason:
                return "retry_with_knowledge"
            if pattern.job_status not in ("active", "completed"):
                return "propose_to_user"
        if pattern.session_blocks >= _MIN_BLOCKS_FOR_ACTION and strength < 0.5:
            return "research"
        return "defer"

    @staticmethod
    def _compute_priority(pattern: BlockPattern, strength: float) -> float:
        base = strength * 0.6
        recency_bonus = 0.0
        if pattern.last_blocked:
            age_h = (time.time() - pattern.last_blocked) / 3600.0
            recency_bonus = max(0.0, 0.2 - age_h * 0.01)
        count_bonus = min(0.2, pattern.block_count * 0.02)
        return min(1.0, base + recency_bonus + count_bonus)


# ---------------------------------------------------------------------------
# Learning Proposer
# ---------------------------------------------------------------------------

@dataclass
class PendingProposal:
    gap: CapabilityGap
    created_at: float
    proposed_at: float | None = None


class LearningProposer:
    """Dispatches research/retry/propose/defer actions from capability gaps."""

    def __init__(self) -> None:
        self._pending: list[PendingProposal] = []
        self._last_proposal_time: float = 0.0
        self._rejected_families: dict[str, float] = {}
        self._proposals_made: int = 0
        self._research_dispatched: int = 0

    def process_gaps(
        self,
        gaps: list[CapabilityGap],
        enqueue_research: Any = None,
    ) -> list[dict[str, Any]]:
        """Process analyzed gaps. Returns list of actions taken."""
        actions: list[dict[str, Any]] = []
        now = time.time()

        for gap in gaps:
            fid = gap.family.family_id
            if self._is_family_rejected(fid, now):
                continue

            if gap.suggested_action == "defer":
                continue

            if gap.suggested_action == "research":
                if enqueue_research and self._can_research():
                    self._dispatch_research(gap, enqueue_research)
                    actions.append({"action": "research", "family": fid})

            elif gap.suggested_action == "retry_with_knowledge":
                if enqueue_research and self._can_research():
                    self._dispatch_retry_research(gap, enqueue_research)
                    actions.append({"action": "retry_research", "family": fid})

            elif gap.suggested_action == "propose_to_user":
                if self._can_queue_proposal():
                    self._queue_proposal(gap, now)
                    actions.append({"action": "queued_proposal", "family": fid})

        return actions

    def get_next_proposal(self) -> PendingProposal | None:
        """Get the next proposal to surface to the user (if cooldown allows)."""
        now = time.time()
        if not self._can_surface_proposal(now):
            return None
        if not self._pending:
            return None
        return self._pending[0]

    def mark_proposed(self, proposal: PendingProposal) -> None:
        proposal.proposed_at = time.time()
        self._last_proposal_time = time.time()
        self._proposals_made += 1

    def record_user_response(self, family_id: str, accepted: bool) -> None:
        if not accepted:
            self._rejected_families[family_id] = time.time()
        self._pending = [p for p in self._pending if p.gap.family.family_id != family_id]

    def bypass_cooldown_for_user_ask(self) -> PendingProposal | None:
        """When user explicitly asks about learning, bypass the 24h cooldown."""
        if self._pending:
            return self._pending[0]
        return None

    def get_snapshot(self) -> dict[str, Any]:
        now = time.time()
        next_eligible = max(0.0, self._last_proposal_time + _GLOBAL_PROPOSAL_COOLDOWN_S - now)
        return {
            "pending_count": len(self._pending),
            "proposals_made": self._proposals_made,
            "research_dispatched": self._research_dispatched,
            "rejected_families": list(self._rejected_families.keys()),
            "next_proposal_eligible_s": round(next_eligible, 0),
            "pending": [
                {
                    "family_id": p.gap.family.family_id,
                    "canonical_name": p.gap.family.canonical_name,
                    "priority": p.gap.priority,
                    "block_count": p.gap.block_count,
                    "created_at": p.created_at,
                }
                for p in self._pending
            ],
        }

    def _can_queue_proposal(self) -> bool:
        """Check if we can add another proposal (capacity + global cooldown)."""
        if len(self._pending) >= _MAX_PENDING_PROPOSALS:
            return False
        return self._can_surface_proposal(time.time())

    def _can_surface_proposal(self, now: float) -> bool:
        """Check if the global cooldown has elapsed for surfacing a proposal."""
        return now - self._last_proposal_time >= _GLOBAL_PROPOSAL_COOLDOWN_S

    def _can_research(self) -> bool:
        return True

    def _is_family_rejected(self, family_id: str, now: float) -> bool:
        rejected_at = self._rejected_families.get(family_id)
        if rejected_at is None:
            return False
        if now - rejected_at > _REJECTION_SUPPRESSION_S:
            del self._rejected_families[family_id]
            return False
        return True

    def _queue_proposal(self, gap: CapabilityGap, now: float) -> None:
        for p in self._pending:
            if p.gap.family.family_id == gap.family.family_id:
                return
        self._pending.append(PendingProposal(gap=gap, created_at=now))
        if len(self._pending) > _MAX_PENDING_PROPOSALS:
            self._pending.sort(key=lambda p: p.gap.priority, reverse=True)
            self._pending = self._pending[:_MAX_PENDING_PROPOSALS]

    def _dispatch_research(self, gap: CapabilityGap, enqueue: Any) -> None:
        try:
            from autonomy.research_intent import ResearchIntent
            intent = ResearchIntent(
                question=f"What techniques exist for {gap.family.canonical_name}?",
                tool_hint="web",
                tags=frozenset({gap.family.domain, "capability_discovery"}),
                scope="exploratory",
                priority=gap.priority * 0.6,
            )
            enqueue(intent)
            self._research_dispatched += 1
        except Exception:
            logger.debug("Failed to dispatch discovery research", exc_info=True)

    def _dispatch_retry_research(self, gap: CapabilityGap, enqueue: Any) -> None:
        try:
            from autonomy.research_intent import ResearchIntent
            reason = gap.prior_failure_reason or "unknown limitation"
            intent = ResearchIntent(
                question=f"Alternative approaches for {gap.family.canonical_name} given: {reason}",
                tool_hint="web",
                tags=frozenset({gap.family.domain, "capability_discovery", "retry"}),
                scope="exploratory",
                priority=gap.priority * 0.7,
            )
            enqueue(intent)
            self._research_dispatched += 1
        except Exception:
            logger.debug("Failed to dispatch retry research", exc_info=True)


# ---------------------------------------------------------------------------
# Module-level access
# ---------------------------------------------------------------------------

_normalizer: CapabilityFamilyNormalizer | None = None
_tracker: BlockFrequencyTracker | None = None
_analyzer: GapAnalyzer | None = None
_proposer: LearningProposer | None = None


def get_normalizer() -> CapabilityFamilyNormalizer:
    global _normalizer
    if _normalizer is None:
        _normalizer = CapabilityFamilyNormalizer()
    return _normalizer


def get_tracker() -> BlockFrequencyTracker:
    global _tracker
    if _tracker is None:
        _tracker = BlockFrequencyTracker(get_normalizer())
    return _tracker


def get_analyzer() -> GapAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = GapAnalyzer()
    return _analyzer


def get_proposer() -> LearningProposer:
    global _proposer
    if _proposer is None:
        _proposer = LearningProposer()
    return _proposer
