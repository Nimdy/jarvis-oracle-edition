"""Skill-transfer feature encoding for the SKILL_TRANSFER Tier-2
specialist.

P3.10 — fifth and final Tier-2 Matrix Protocol specialist. Same
template contract as P3.6 (``positive_memory``), P3.7
(``negative_memory``), P3.8 (``speaker_profile``), and P3.9
(``temporal_pattern``):

  * It writes nothing — no memories, beliefs, identity, autonomy,
    policy authority, HRR/P5 state, Soul Integrity, skill registry,
    capability gate, or learning-job state. It is pure feature
    engineering.
  * It enters CANDIDATE_BIRTH only. Promotion is gated by the
    standard Matrix Protocol lifecycle in
    ``HemisphereOrchestrator._check_specialist_promotions``; this
    module does not bypass any of it.
  * It produces a real-time inferable scalar in ``[0, 1]`` from
    skill-registry / learning-job / capability-gate statistics. It
    does NOT fall back to accuracy-as-proxy.

**Strict capability / truth-boundary contract.** The user
explicitly flagged this lane as the highest-risk Tier-2 specialist
for "similarity-is-capability" drift:

> Similarity is not capability.
>
> The specialist may say:
>   * this skill resembles prior verified skill families
>   * this transfer path may be promising
>
> It must not say:
>   * therefore this new skill is verified
>   * therefore this plugin is safe
>   * therefore this capability can be promoted

This module emits a BOUNDED ADVISORY scalar. It does not gate
capability promotion, mark a skill as verified, or alter the
capability_gate state. It is consumed only by the M6 broadcast-slot
competition, exactly like the other four Tier-2 encoders. The
contract is enforced two ways:

  * The encoder consumes only counts and fractions about the *current*
    registry / job state. It NEVER consumes a "this_skill_is_verified"
    /  "promote_this" / "capability_proven" claim and ignores any
    such key fed in by tests or live state. A regression test feeds
    those keys in and asserts the output is unchanged.
  * A static source-scan test blocks any reference to canonical
    skill-registry / capability-gate mutators (set-status,
    verify, promote, allow, grant, etc.). None present.

Dimension layout (16-dim total, all values clamped to ``[0, 1]``):

  Block A (dims  0-7):  Skill-registry breadth & maturity
  Block B (dims  8-11): Learning-job phase distribution
  Block C (dims 12-15): Cross-skill diversity / overlap

The signal value returned by :func:`compute_signal_value` is a
weighted aggregate over those blocks. Block A leads (registry
breadth is the canonical "skill ecosystem is rich enough for
transfer to be meaningful" signal) and Block B is weighted higher
than C because learning-job phase progress is more directly
observable than diversity heuristics at low registry volume.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)


FEATURE_DIM = 16

# The :class:`HemisphereFocus` value this encoder serves. Kept as a
# string literal (not an enum import) so the schema-emission audit's
# writer-literal scan recognises this module as the live writer for
# ``skill_transfer`` and removes the future-only whitelist entry.
FOCUS_NAME = "skill_transfer"


# Normalisation caps. Conservative values so a brain with a small
# registry (initial set ≈ 5-10 skills) still produces a saturating
# signal — skill_transfer should not require a huge skill ecosystem
# to fire. Tuning is conservative; raising the cap reduces signal
# sensitivity at low volume.
TOTAL_SKILLS_CAP = 20
ACTIVE_JOBS_CAP = 5
ARTIFACT_COUNT_CAP = 10
EVIDENCE_COUNT_CAP = 5
FAILURE_COUNT_CAP = 10
DISTINCT_TYPES_CAP = 3        # procedural / perceptual / control
DISTINCT_STATUSES_CAP = 5     # rough sample of skill status taxonomy

# Phase taxonomy for Block B. We deliberately do NOT distinguish
# ``register`` from ``monitor`` — both are "post-verification"
# states. The phase fractions are advisory.
EARLY_PHASES = frozenset({"assess", "research", "acquire", "integrate"})
MID_PHASES = frozenset({"collect", "train"})
LATE_PHASES = frozenset({"verify", "register", "monitor"})

# Skill-status taxonomy for Block A. Mirrors the labels in the
# registry / matrix lifecycle without importing them — this module
# stays purely string-driven so it can be unit-tested without the
# registry package.
VERIFIED_STATUSES = frozenset({
    "verified", "verified_operational", "verified_limited",
    "promoted",
})
PROBATIONARY_STATUSES = frozenset({"probationary"})
CANDIDATE_STATUSES = frozenset({"candidate", "candidate_birth"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(v: float) -> float:
    if v != v:  # NaN guard
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _safe_attr(obj: Any, name: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return f


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(_as_float(v, default))
    except Exception:
        return default


def _as_seq_of_dicts(v: Any) -> list[Mapping[str, Any]]:
    if v is None:
        return []
    if isinstance(v, (str, bytes)):
        return []
    out: list[Mapping[str, Any]] = []
    try:
        for item in v:
            if isinstance(item, Mapping):
                out.append(item)
    except TypeError:
        return []
    return out


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class SkillTransferEncoder:
    """Encodes skill-registry / learning-job / capability-overlap
    state into a 16-dim ``[0, 1]`` feature vector.

    All public methods are pure (no side effects). They take plain
    Python dicts / sequences, never live singletons, so the encoder
    can be unit-tested without standing up the brain stack.

    The orchestrator-level helper
    :meth:`HemisphereOrchestrator._build_skill_transfer_context` is
    responsible for gathering live state from
    ``skill_registry.get_status_snapshot()`` and (where available)
    ``capability_gate`` and passing it in as a flat dict.

    The encoder NEVER claims a skill is verified, NEVER promotes a
    capability, and NEVER consumes a "this skill is verified" /
    "promote this" claim from input. See the module-level
    ``capability / truth-boundary contract`` note.
    """

    FEATURE_DIM = FEATURE_DIM

    # ------------------------------------------------------------------
    # Block A: Skill-registry breadth & maturity (8 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_registry_block(ctx: Mapping[str, Any]) -> list[float]:
        """Block A: 8 dims of registry-breadth features.

        Expected keys (all optional, missing fields contribute 0.0):

          * ``total_skills`` — int.
          * ``skills`` — sequence of {status, capability_type,
            has_evidence, artifact_count, evidence_count,
            matrix_protocol} dicts. (Embedding-free; no skill
            payloads needed.)
          * ``by_status`` — mapping of status → count (alternate
            input form).
          * ``active_jobs_count`` — int.
        """
        total = max(0, _as_int(_safe_attr(ctx, "total_skills", 0)))
        if total == 0:
            # Try to infer from skills sequence if the caller used it.
            skills_seq = _as_seq_of_dicts(_safe_attr(ctx, "skills", ()))
            total = len(skills_seq)
        else:
            skills_seq = _as_seq_of_dicts(_safe_attr(ctx, "skills", ()))

        total_score = _clamp(total / float(TOTAL_SKILLS_CAP))

        # Per-status fractions. Prefer iterating ``skills`` directly
        # for accuracy; fall back to ``by_status`` if only counts are
        # provided.
        by_status_raw = _safe_attr(ctx, "by_status", {}) or {}
        by_status: dict[str, int] = {}
        try:
            for k, v in by_status_raw.items():
                by_status[str(k).lower()] = max(0, _as_int(v))
        except Exception:
            by_status = {}

        verified_count = 0
        probationary_count = 0
        candidate_count = 0
        evidence_count = 0
        artifact_total = 0
        matrix_count = 0

        if skills_seq:
            for s in skills_seq:
                status = str(_safe_attr(s, "status", "") or "").lower()
                if status in VERIFIED_STATUSES:
                    verified_count += 1
                elif status in PROBATIONARY_STATUSES:
                    probationary_count += 1
                elif status in CANDIDATE_STATUSES:
                    candidate_count += 1
                if bool(_safe_attr(s, "has_evidence", False)):
                    evidence_count += 1
                artifact_total += max(0, _as_int(_safe_attr(s, "artifact_count", 0)))
                if bool(_safe_attr(s, "matrix_protocol", False)):
                    matrix_count += 1
        else:
            for status, n in by_status.items():
                if status in VERIFIED_STATUSES:
                    verified_count += n
                elif status in PROBATIONARY_STATUSES:
                    probationary_count += n
                elif status in CANDIDATE_STATUSES:
                    candidate_count += n

        denom = max(1, total)
        verified_frac = _clamp(verified_count / float(denom)) if total > 0 else 0.0
        probationary_frac = _clamp(probationary_count / float(denom)) if total > 0 else 0.0
        candidate_frac = _clamp(candidate_count / float(denom)) if total > 0 else 0.0
        evidence_frac = _clamp(evidence_count / float(denom)) if total > 0 else 0.0

        # Artifact-density score. Average artifacts per skill is a
        # rough learning-investment proxy; capped per-skill so a
        # single mature skill does not dominate.
        if total > 0:
            avg_artifacts = artifact_total / float(denom)
            artifact_density = _clamp(avg_artifacts / float(ARTIFACT_COUNT_CAP))
        else:
            artifact_density = 0.0

        matrix_frac = _clamp(matrix_count / float(denom)) if total > 0 else 0.0

        active_jobs = max(0, _as_int(_safe_attr(ctx, "active_jobs_count", 0)))
        any_active = 1.0 if active_jobs > 0 else 0.0

        return [
            total_score,
            verified_frac,
            probationary_frac,
            candidate_frac,
            evidence_frac,
            artifact_density,
            matrix_frac,
            any_active,
        ]

    # ------------------------------------------------------------------
    # Block B: Learning-job phase distribution (4 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_jobs_block(ctx: Mapping[str, Any]) -> list[float]:
        """Block B: 4 dims of learning-job-phase features.

        Expected keys (all optional):

          * ``active_jobs`` — sequence of {phase, status,
            phase_age_s, stale, evidence_count} dicts.
          * ``active_jobs_count`` — int (fallback).
          * ``failed_jobs_count`` — int.
        """
        jobs = _as_seq_of_dicts(_safe_attr(ctx, "active_jobs", ()))
        active_jobs_count = max(0, _as_int(_safe_attr(ctx, "active_jobs_count", len(jobs))))
        failed_count = max(0, _as_int(_safe_attr(ctx, "failed_jobs_count", 0)))

        if jobs:
            late = sum(1 for j in jobs if str(_safe_attr(j, "phase", "")).lower() in LATE_PHASES)
            mid = sum(1 for j in jobs if str(_safe_attr(j, "phase", "")).lower() in MID_PHASES)
            stale = sum(1 for j in jobs if bool(_safe_attr(j, "stale", False)))
            denom = max(1, len(jobs))
            late_frac = _clamp(late / float(denom))
            mid_frac = _clamp(mid / float(denom))
            stale_frac = _clamp(stale / float(denom))
        else:
            late_frac = 0.0
            mid_frac = 0.0
            # Without job data, "no jobs are stale" is vacuously true
            # but should NOT credit the system. Gate on having any
            # active or failed jobs to make a stale claim.
            stale_frac = 0.0

        # Low-failure score — gated on having any registry / job
        # signal at all (otherwise empty context yields a vacuous
        # 1.0). The gate uses ``active_jobs_count`` OR
        # ``failed_jobs_count`` OR a non-zero registry.
        total_skills = max(0, _as_int(_safe_attr(ctx, "total_skills", 0)))
        has_skill_signal = (
            active_jobs_count > 0 or failed_count > 0 or total_skills > 0
        )
        if has_skill_signal:
            low_failure_score = _clamp(
                1.0 - (failed_count / float(FAILURE_COUNT_CAP)),
            )
        else:
            low_failure_score = 0.0

        return [
            late_frac,
            mid_frac,
            stale_frac,
            low_failure_score,
        ]

    # ------------------------------------------------------------------
    # Block C: Cross-skill diversity / overlap (4 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def encode_overlap_block(ctx: Mapping[str, Any]) -> list[float]:
        """Block C: 4 dims of skill-diversity / overlap features.

        Expected keys (all optional):

          * ``skills`` — sequence of dicts.
          * ``capability_type_overlap`` — pre-normalised in
            ``[0, 1]``; advisory, defaults 0.0.
          * ``transfer_advisory`` — pre-normalised in ``[0, 1]``;
            advisory, defaults 0.0.

        IMPORTANT: this block computes diversity / overlap *as
        observed* in the registry. It does NOT consume any
        "promote-this-skill" hint and the orchestrator-level builder
        does not surface one.
        """
        skills_seq = _as_seq_of_dicts(_safe_attr(ctx, "skills", ()))

        # Capability-type diversity: 1.0 when the registry covers
        # every supported capability type (procedural / perceptual /
        # control). 0.0 when empty.
        types_seen: set[str] = set()
        statuses_seen: set[str] = set()
        for s in skills_seq:
            t = str(_safe_attr(s, "capability_type", "") or "").lower()
            if t:
                types_seen.add(t)
            st = str(_safe_attr(s, "status", "") or "").lower()
            if st:
                statuses_seen.add(st)
        type_diversity = _clamp(len(types_seen) / float(DISTINCT_TYPES_CAP))
        status_diversity = _clamp(len(statuses_seen) / float(DISTINCT_STATUSES_CAP))

        capability_type_overlap = _clamp(
            _as_float(_safe_attr(ctx, "capability_type_overlap", 0.0)),
        )
        transfer_advisory = _clamp(
            _as_float(_safe_attr(ctx, "transfer_advisory", 0.0)),
        )

        return [
            type_diversity,
            status_diversity,
            capability_type_overlap,
            transfer_advisory,
        ]

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    @staticmethod
    def encode(ctx: Mapping[str, Any]) -> list[float]:
        """Produce the 16-dim ``[0, 1]`` feature vector from context."""
        block_a = SkillTransferEncoder.encode_registry_block(ctx)
        block_b = SkillTransferEncoder.encode_jobs_block(ctx)
        block_c = SkillTransferEncoder.encode_overlap_block(ctx)
        vec = block_a + block_b + block_c
        assert len(vec) == FEATURE_DIM, (
            f"SkillTransferEncoder.encode produced {len(vec)} dims, "
            f"expected {FEATURE_DIM}"
        )
        for i, v in enumerate(vec):
            assert 0.0 <= v <= 1.0, f"feature[{i}] = {v} out of [0,1]"
        return vec

    @staticmethod
    def compute_signal_value(ctx: Mapping[str, Any]) -> float:
        """Real-time inferable scalar in ``[0, 1]``.

        Weighted aggregate over the three feature blocks:

          * Block A (registry breadth & maturity)  at weight 0.50
          * Block B (learning-job phase)           at weight 0.30
          * Block C (diversity / overlap)          at weight 0.20

        Block A leads because skill-registry breadth is the canonical
        "is the skill ecosystem rich enough for transfer to be
        meaningful?" signal. Block B is weighted higher than C
        because learning-job phase progress is more directly
        observable than diversity heuristics at low registry volume.

        This function is the explicit replacement for the
        accuracy-as-proxy fallback in
        ``HemisphereOrchestrator._compute_signal_value``. For the
        ``skill_transfer`` focus the orchestrator dispatches to this
        function before attempting network inference; the result is
        always defined and always in ``[0, 1]``, even at
        CANDIDATE_BIRTH when the underlying NN is untrained.

        **Capability boundary.** A high signal here is NOT a
        promotion vote. The encoder advises that the transfer
        context is rich enough to be observed; the actual
        capability_gate / skill_registry promotion path is
        completely separate and unaffected by this scalar.
        """
        block_a = SkillTransferEncoder.encode_registry_block(ctx)
        block_b = SkillTransferEncoder.encode_jobs_block(ctx)
        block_c = SkillTransferEncoder.encode_overlap_block(ctx)

        signal = (
            0.50 * _mean(block_a)
            + 0.30 * _mean(block_b)
            + 0.20 * _mean(block_c)
        )
        return _clamp(signal)
