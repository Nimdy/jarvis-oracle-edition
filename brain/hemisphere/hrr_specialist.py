"""Tier-1 ``hrr_encoder`` specialist stub (P4, PRE-MATURE).

Scope this sprint:

* Declares :class:`HRREncoder` and :data:`FEATURE_DIM`.
* Reads purely from :func:`library.vsa.status.get_hrr_status` when asked for
  features — never reaches into canonical memory / belief / policy / etc.
* Returns an all-zero vector when the runtime flag ``ENABLE_HRR_SHADOW=0``
  (the default). Returns derived-but-bounded features only when samples
  exist and gates pass.

Explicitly NOT in scope:

* No network is registered under :class:`HemisphereFocus.HRR_ENCODER` this
  sprint. The orchestrator's broadcast-slot loop calls
  ``_get_best_network(HRR_ENCODER)`` which returns ``None`` and skips.
* No Tier-2 ``holographic_cognition`` focus is declared — there is a
  negative test (``test_no_tier2_holographic_cognition_registered``)
  pinning that.
* No distillation teacher signal wiring — ``hrr_encoder`` is intentionally
  absent from every distillation teacher registry in this process.

The lifecycle stage the scaffolding pins this focus to when spawned is
:attr:`SpecialistLifecycleStage.CANDIDATE_BIRTH`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from hemisphere.types import HemisphereFocus, SpecialistLifecycleStage


FEATURE_DIM = 8

# Order of feature slots — stable so the frozen vector has a known layout.
FEATURE_NAMES = (
    "enabled",
    "samples_total_ratio",       # log-scaled, clamped to [0,1]
    "samples_retained_ratio",    # samples_retained / ring_capacity
    "last_binding_cleanliness",
    "last_cleanup_accuracy",
    "last_similarity_to_previous",
    "sim_shadow_help_rate",
    "recall_advisory_help_rate",
)

TARGET_FOCUS: HemisphereFocus = HemisphereFocus.HRR_ENCODER
TARGET_LIFECYCLE: SpecialistLifecycleStage = SpecialistLifecycleStage.CANDIDATE_BIRTH


def _clamp(v: float) -> float:
    if v is None:
        return 0.0
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return 0.0
    if fv < 0.0:
        return 0.0
    if fv > 1.0:
        return 1.0
    return fv


def _log_ratio(n: int, ceil: int = 500) -> float:
    """Smooth log-scaled clamp of a counter to [0,1]."""
    if n is None or ceil <= 0:
        return 0.0
    if n <= 0:
        return 0.0
    import math as _math

    return _clamp(_math.log1p(float(n)) / _math.log1p(float(ceil)))


class HRREncoder:
    """Feature encoder for the dormant Tier-1 ``hrr_encoder`` specialist.

    Reads :func:`library.vsa.status.get_hrr_status` (no writes) and emits a
    :data:`FEATURE_DIM`-long ``[0,1]`` vector summarizing HRR shadow health.

    The encoder is deterministic and side-effect-free. When the HRR runtime
    is disabled (default), it returns ``[0.0] * FEATURE_DIM``.
    """

    focus: HemisphereFocus = TARGET_FOCUS
    lifecycle: SpecialistLifecycleStage = TARGET_LIFECYCLE

    @staticmethod
    def encode(status: Optional[Dict[str, Any]] = None) -> list:
        """Produce a ``FEATURE_DIM``-long ``[0,1]`` vector from HRR status.

        If ``status`` is omitted, this method reads the live status via
        :func:`library.vsa.status.get_hrr_status`.
        """
        if status is None:
            try:
                from library.vsa.status import get_hrr_status

                status = get_hrr_status()
            except Exception:
                return [0.0] * FEATURE_DIM

        if not isinstance(status, dict) or not status.get("enabled"):
            return [0.0] * FEATURE_DIM

        world = status.get("world_shadow") or {}
        sim = status.get("simulation_shadow") or {}
        recall = status.get("recall_advisory") or {}

        ring_world = max(1, int(world.get("ring_capacity") or 1))
        vec = [
            1.0,  # enabled bit
            _log_ratio(int(world.get("samples_total") or 0), ceil=ring_world * 100),
            _clamp((world.get("samples_retained") or 0) / ring_world),
            _clamp(world.get("binding_cleanliness")),
            _clamp(world.get("cleanup_accuracy")),
            _clamp(world.get("similarity_to_previous")),
            _clamp(sim.get("last_cleanliness_after") if isinstance(sim, dict) else None),
            _clamp(recall.get("help_rate") if isinstance(recall, dict) else None),
        ]
        if len(vec) != FEATURE_DIM:
            raise RuntimeError(
                f"HRREncoder.encode produced {len(vec)} features, expected {FEATURE_DIM}"
            )
        return vec

    @staticmethod
    def describe() -> Dict[str, Any]:
        """Return a static specialist-stub descriptor for dashboards."""
        return {
            "focus": TARGET_FOCUS.value,
            "lifecycle": TARGET_LIFECYCLE.value,
            "tier": 1,
            "feature_dim": FEATURE_DIM,
            "feature_names": list(FEATURE_NAMES),
            "authority": "none",
            "notes": (
                "PRE-MATURE P4 HRR shadow substrate. No network is registered "
                "under this focus. No canonical memory / belief / policy / "
                "autonomy / Soul Integrity / LLM-articulation path reads this "
                "encoder. Tier-2 holographic_cognition is intentionally not "
                "declared this sprint."
            ),
        }
