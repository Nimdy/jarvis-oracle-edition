"""Grounding-Ring passive metrics + shadow promotion gate (SPARK_DESIGN §8 P0).

Phase 0 of the Grounding Ring: compute the five external-grounding baselines
**read-only** from the live belief graph / memory provenance / fractal recall,
and surface them so the baselines (3.3x grounded:inferred, 0.857 orphan_rate,
~1.0 avg_chain_length, ~0 external_validation_rate) become observable BEFORE
any mechanism can move them.

ZERO AUTHORITY. Nothing here drives a lever. The metrics are computed, cached,
and logged; no cadence, reward, curiosity weight, autonomy action, or
user-facing token reads them to act. The :class:`SparkPromotion` gate exists
from day one (cloned from ``cognition/promotion.py``: shadow=0/advisory=1/
active=2, accuracy-gated, auto-demoting) and **defaults to shadow**, so the
DEFAULT runtime behavior is unchanged. P0 ships the instrument and the gate;
later phases earn promotion. This module never flips the gate to active.

Honesty guardrails enforced here (SPARK_DESIGN §7):
  * View-only epistemics: we read ``belief_store.get_active_beliefs()`` and
    ``compute_integrity(...)`` but NEVER mutate the frozen ``BeliefRecord`` or
    ``beliefs.jsonl``.
  * ``external_validation_rate`` is movable only by a real external validator;
    in P0 there is no emitter, so it is honestly ~0.0 (default-safe).
  * Enumeration is sampled/cached (like the existing 100-item
    ``belief_graph_coverage`` block in the orchestrator) to avoid tick latency.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Provenance classes, split so the KEYSTONE ratio stays comparable to its
# baseline and to the audit's source-trust concern. Mirrors the ProvenanceType
# vocabulary in consciousness/events.
#
# HONESTY / OPEN-QUESTION #1 (SPARK_DESIGN §11): does source-cited *external
# knowledge* (web/academic) count as "grounded"? It is NOT grounding of JARVIS's
# own world (its senses + the operator), and it is already the dominant class —
# so folding it into the denominator DILUTES the very inference-vs-grounding
# signal the keystone metric exists to surface. We therefore keep the headline
# ``grounded_inferred_ratio`` NARROW (directly-grounded observation only) so it
# matches the 3.3x baseline, and expose the broad view as a SEPARATE, clearly
# labelled ``grounded_inferred_ratio_incl_external``. Neither hides the other;
# whether external_source should later be promoted into "grounded" is David's
# open-question-#1 decision, not a silent default.
_OBSERVATION_PROVENANCE = frozenset({"observed", "user_claim"})  # senses + operator
_EXTERNAL_KNOWLEDGE_PROVENANCE = frozenset({"external_source"})   # cited web/academic
# Broad set (observation + external knowledge), retained for the _incl_external view.
_GROUNDED_PROVENANCE = _OBSERVATION_PROVENANCE | _EXTERNAL_KNOWLEDGE_PROVENANCE
_INFERRED_PROVENANCE = frozenset({"model_inference"})

# Cache TTL for the (potentially full-store) belief enumeration. The orchestrator
# tick cadence is ~5s; a 30s cache keeps enumeration off the hot path, matching
# the spirit of the existing belief_graph_coverage sampling.
_CACHE_TTL_S = 30.0
# Sample cap on belief enumeration so a large store can't add tick latency.
_BELIEF_SAMPLE_CAP = 1000


@dataclass
class SparkMetrics:
    """The five P0 grounding-ring baselines, all read-only / observability."""

    orphan_rate: float = 0.0
    inference_validation_gap: float = 0.0
    external_validation_rate: float = 0.0
    # KEYSTONE: model_inference / directly-grounded observation (senses+operator),
    # comparable to the 3.3x baseline and to the audit's source-trust concern.
    grounded_inferred_ratio: float = 0.0
    # BROAD VIEW: model_inference / (observation + cited external knowledge). Lower
    # because external_source dominates; kept separate so neither masks the other.
    grounded_inferred_ratio_incl_external: float = 0.0
    # CONTINUITY: the SAME narrow ratio computed on the MEMORY STORE (all recorded
    # memories — a different, larger population than the active belief graph). This
    # is the historic ~3.5x figure tracked in SPARK_DESIGN; kept alongside the
    # belief-graph keystone per operator decision (2026-05-31) so re-baselining the
    # keystone to belief-graph reality doesn't lose the previously-tracked number.
    grounded_inferred_ratio_memory_store: float = 0.0
    avg_chain_length: float = 0.0
    # Provenance / sample bookkeeping (honesty: where did the numbers come from)
    grounded_count: int = 0          # directly-grounded observation (narrow / keystone)
    external_knowledge_count: int = 0  # cited external_source (web/academic)
    grounded_count_incl_external: int = 0  # observation + external knowledge (broad)
    memory_store_grounded_count: int = 0   # memory store: observed + user_claim
    memory_store_inferred_count: int = 0   # memory store: model_inference
    inferred_count: int = 0
    sampled_beliefs: int = 0
    sources_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "orphan_rate": round(self.orphan_rate, 4),
            "inference_validation_gap": round(self.inference_validation_gap, 4),
            "external_validation_rate": round(self.external_validation_rate, 4),
            "grounded_inferred_ratio": round(self.grounded_inferred_ratio, 4),
            "grounded_inferred_ratio_incl_external": round(
                self.grounded_inferred_ratio_incl_external, 4),
            "grounded_inferred_ratio_memory_store": round(
                self.grounded_inferred_ratio_memory_store, 4),
            "avg_chain_length": round(self.avg_chain_length, 4),
            "grounded_count": self.grounded_count,
            "external_knowledge_count": self.external_knowledge_count,
            "grounded_count_incl_external": self.grounded_count_incl_external,
            "memory_store_grounded_count": self.memory_store_grounded_count,
            "memory_store_inferred_count": self.memory_store_inferred_count,
            "inferred_count": self.inferred_count,
            "sampled_beliefs": self.sampled_beliefs,
            "sources_available": self.sources_available,
        }


# Module-level cache: (computed_at, SparkMetrics)
_cache: tuple[float, SparkMetrics] = (0.0, SparkMetrics())


def compute_spark_metrics(engine: Any | None) -> SparkMetrics:
    """Compute the five P0 baselines read-only from live subsystems.

    Cached for ``_CACHE_TTL_S`` so repeated tick calls don't re-enumerate the
    belief store. Default-safe: any missing source yields 0.0 (and
    ``sources_available=False``), never raising.

    Sources:
      * orphan_rate            — belief_graph integrity.compute_integrity
      * grounded_inferred_ratio — KEYSTONE: model_inference / directly-grounded
                                  observation (observed + user_claim); baseline
                                  ~3.3x. A separate ..._incl_external also divides
                                  by cited external_source (open-question #1).
      * inference_validation_gap — model_inference − directly-grounded observation
                                  (raw signed; positive ⇒ inference exceeds grounding)
      * external_validation_rate — gate-recorded external-validator outcomes
                                  only (never a provenance proxy); ~0 in P0
      * avg_chain_length        — fractal_recall.get_state()["avg_chain_length"]
    """
    global _cache
    now = time.time()
    if now - _cache[0] < _CACHE_TTL_S:
        return _cache[1]

    metrics = SparkMetrics()

    cs = getattr(engine, "consciousness", None) if engine is not None else None

    # --- orphan_rate (live ~0.857) via belief graph integrity --------------
    try:
        bg = getattr(cs, "_belief_graph", None) if cs else None
        if bg is None:
            from epistemic.belief_graph import BeliefGraph
            bg = BeliefGraph.get_instance()
        edge_store = getattr(bg, "_edge_store", None) if bg else None
        belief_store = getattr(bg, "_belief_store", None) if bg else None
        if edge_store is not None and belief_store is not None:
            from epistemic.belief_graph.integrity import compute_integrity
            integrity = compute_integrity(edge_store, belief_store)
            metrics.orphan_rate = float(integrity.get("orphan_rate", 0.0) or 0.0)
            metrics.sources_available = True
    except Exception:
        logger.debug("spark_metrics: orphan_rate unavailable", exc_info=True)

    # --- provenance-derived ratios (view-only, never mutates beliefs) ------
    try:
        belief_store = None
        bg = getattr(cs, "_belief_graph", None) if cs else None
        if bg is not None:
            belief_store = getattr(bg, "_belief_store", None)
        if belief_store is None:
            from epistemic.contradiction_engine import ContradictionEngine
            ce = ContradictionEngine.get_instance()
            belief_store = getattr(ce, "_belief_store", None) if ce else None
        if belief_store is not None and hasattr(belief_store, "get_active_beliefs"):
            beliefs = belief_store.get_active_beliefs()
            if len(beliefs) > _BELIEF_SAMPLE_CAP:
                beliefs = beliefs[-_BELIEF_SAMPLE_CAP:]
            observation = 0   # directly-grounded: senses + operator (narrow)
            external = 0      # cited external knowledge (web/academic)
            inferred = 0      # model_inference
            for b in beliefs:
                prov = getattr(b, "provenance", "unknown")
                if prov in _OBSERVATION_PROVENANCE:
                    observation += 1
                elif prov in _EXTERNAL_KNOWLEDGE_PROVENANCE:
                    external += 1
                if prov in _INFERRED_PROVENANCE:
                    inferred += 1
            broad = observation + external
            metrics.grounded_count = observation
            metrics.external_knowledge_count = external
            metrics.grounded_count_incl_external = broad
            metrics.inferred_count = inferred
            metrics.sampled_beliefs = len(beliefs)
            if beliefs:
                metrics.sources_available = True
            # KEYSTONE grounded:inferred ratio = model_inference / directly-grounded
            # OBSERVATION (observed+user_claim). Comparable to the 3.3x baseline and
            # to the audit's source-trust concern (see _OBSERVATION_PROVENANCE note).
            if observation > 0:
                metrics.grounded_inferred_ratio = inferred / observation
            elif inferred > 0:
                metrics.grounded_inferred_ratio = float(inferred)
            # BROAD view = model_inference / (observation + cited external knowledge).
            if broad > 0:
                metrics.grounded_inferred_ratio_incl_external = inferred / broad
            elif inferred > 0:
                metrics.grounded_inferred_ratio_incl_external = float(inferred)
            # inference_validation_gap = model_inference − directly-grounded
            # observation (raw signed): positive ⇒ more inference than grounding.
            metrics.inference_validation_gap = float(inferred - observation)
            # external_validation_rate — HONESTY (SPARK §7/§9): movable ONLY by a
            # real external-validator event (BELIEF_EXTERNALLY_CONFIRMED / user
            # answer / world-model validation), never a static provenance proxy.
            # In P0 no emitter exists, so this is read from the gate's recorded
            # validation outcomes and is honestly ~0 until P3+ emits them.
            try:
                gate = SparkPromotion.get_instance()
                gate_status = gate.get_status()
                metrics.external_validation_rate = float(
                    gate_status.get("external_validation_rate", 0.0) or 0.0
                )
            except Exception:
                metrics.external_validation_rate = 0.0
    except Exception:
        logger.debug("spark_metrics: provenance ratios unavailable", exc_info=True)

    # --- CONTINUITY: memory-store grounded:inferred (the historic ~3.5x) ----
    # Different population from the belief-graph keystone above: ALL recorded
    # memories, via memory_storage.get_stats()["by_provenance"]. View-only; kept
    # alongside per operator decision so the belief-graph re-baseline doesn't lose
    # the previously-tracked figure.
    try:
        ms = getattr(engine, "memory_storage", None) if engine is not None else None
        if ms is not None and hasattr(ms, "get_stats"):
            bp = (ms.get_stats() or {}).get("by_provenance", {}) or {}
            ms_obs = int(bp.get("observed", 0) or 0) + int(bp.get("user_claim", 0) or 0)
            ms_inf = int(bp.get("model_inference", 0) or 0)
            metrics.memory_store_grounded_count = ms_obs
            metrics.memory_store_inferred_count = ms_inf
            if ms_obs > 0:
                metrics.grounded_inferred_ratio_memory_store = ms_inf / ms_obs
            elif ms_inf > 0:
                metrics.grounded_inferred_ratio_memory_store = float(ms_inf)
    except Exception:
        logger.debug("spark_metrics: memory-store ratio unavailable", exc_info=True)

    # --- avg_chain_length (live ~1.0) via fractal recall -------------------
    try:
        fr = getattr(cs, "_fractal_recall_engine", None) if cs else None
        if fr is not None and hasattr(fr, "get_state"):
            state = fr.get_state()
            acl = state.get("avg_chain_length")
            if acl is not None:
                metrics.avg_chain_length = float(acl)
                metrics.sources_available = True
    except Exception:
        logger.debug("spark_metrics: avg_chain_length unavailable", exc_info=True)

    _cache = (now, metrics)
    return metrics


# ---------------------------------------------------------------------------
# Shadow promotion gate (cloned from cognition/promotion.py).
#
# Defaults to shadow (level 0): the spark metrics are COMPUTED and LOGGED but
# drive NO lever. This gate is the single switch that later phases (P4/P5) will
# consult before allowing any spark metric to influence cadence / reward /
# curiosity. In P0 nothing calls ``is_active()`` to act; the gate exists so the
# zero-authority guarantee is structural and auditable from day one.
# ---------------------------------------------------------------------------

SPARK_PROMOTION_PATH = os.path.join(
    os.path.expanduser("~"), ".jarvis", "spark_promotion.json",
)

# P0 baselines are aspirational targets, recorded so the success direction is
# unambiguous in telemetry. They are NOT optimization targets (SPARK §7).
SPARK_BASELINES = {
    # KEYSTONE reads the active BELIEF GRAPH (consistent with orphan_rate /
    # avg_chain_length, which are also belief-graph metrics). Belief-graph reality
    # measured at the P0 cold boot 2026-05-31: 82 model_inference vs only 4
    # directly-grounded (observed+user_claim) of 583 active beliefs → ~20.5x.
    # This SUPERSEDES the old 3.3x, which was mistakenly computed on the MEMORY
    # STORE — a different (larger) population. The small grounded denominator makes
    # the ratio volatile until grounding accrues, but the target direction (DOWN)
    # is unambiguous. The memory-store view (~3.5x) is retained separately below
    # for continuity (operator decision 2026-05-31).
    "grounded_inferred_ratio": 20.5,              # belief graph; target ↓
    "grounded_inferred_ratio_memory_store": 3.5,  # memory store; continuity view
    "orphan_rate": 0.857,             # target: trending DOWN
    "avg_chain_length": 1.0,          # target: rising ABOVE 1.0
    "external_validation_rate": 0.0,  # target: rising toward >0.40
}

SPARK_MIN_OUTCOMES = 20            # ≥20 external-validation outcomes (SPARK §3 gate 1)
SPARK_MIN_SHADOW_HOURS = 4.0       # live-soak floor before any promotion
SPARK_PROMOTE_VALIDATION_RATE = 0.40  # external_validation_rate ≥0.40
SPARK_DEMOTE_VALIDATION_RATE = 0.20
SPARK_DEMOTE_WINDOW = 20
SPARK_TRANSITION_COOLDOWN_S = 300.0


@dataclass
class _SparkPromotionState:
    level: int = 0  # 0=shadow, 1=advisory, 2=active — DEFAULTS TO SHADOW
    shadow_start_ts: float = field(default_factory=time.time)
    total_outcomes: int = 0
    validation_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    last_promoted_at: float = 0.0
    last_demoted_at: float = 0.0


class SparkPromotion:
    """Zero-authority promotion gate for the Grounding Ring.

    Identical shape to ``WorldModelPromotion`` but gated on
    ``external_validation_rate`` (never a self-score). P0 never promotes — it
    has no outcomes to record yet (the validator events are reserved, not
    emitted). The gate is present so promotion is the only path to authority.
    """

    _instance: SparkPromotion | None = None

    def __init__(self) -> None:
        self._state = _SparkPromotionState()
        self._load()

    @classmethod
    def get_instance(cls) -> SparkPromotion:
        if cls._instance is None:
            cls._instance = SparkPromotion()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    @property
    def level(self) -> int:
        return self._state.level

    def is_shadow(self) -> bool:
        return self._state.level == 0

    def is_active(self) -> bool:
        """True only when promoted to active. ALWAYS False in P0."""
        return self._state.level >= 2

    def record_external_validation(self, validated: bool) -> None:
        """Record an external-validator outcome (P3+ only).

        ``validated`` is set by an external validator (source-cited finding,
        user yes/no, or world-model prediction validated) — never self-scored.
        Being corrected (refuted) still counts as a grounding success upstream;
        here we only track the rate at which beliefs got an external touch.
        """
        self._state.total_outcomes += 1
        self._state.validation_history.append(1.0 if validated else 0.0)
        self._check_transitions()

    def get_status(self) -> dict[str, Any]:
        hist = list(self._state.validation_history)
        rate = sum(hist) / len(hist) if hist else 0.0
        hours = (time.time() - self._state.shadow_start_ts) / 3600.0
        return {
            "level": self._state.level,
            "level_name": {0: "shadow", 1: "advisory", 2: "active"}.get(
                self._state.level, "unknown"),
            "authority": "zero_authority_shadow" if self._state.level == 0 else (
                "advisory" if self._state.level == 1 else "active"),
            "total_outcomes": self._state.total_outcomes,
            "external_validation_rate": round(rate, 4),
            "window_size": len(hist),
            "hours_in_shadow": round(hours, 1),
            "promotion_ready": self._promotion_eligible(),
            "baselines": dict(SPARK_BASELINES),
            "drives_levers": self.is_active(),
        }

    def save(self) -> None:
        data = {
            "level": self._state.level,
            "shadow_start_ts": self._state.shadow_start_ts,
            "total_outcomes": self._state.total_outcomes,
            "validation_history": list(self._state.validation_history),
            "last_promoted_at": self._state.last_promoted_at,
            "last_demoted_at": self._state.last_demoted_at,
        }
        try:
            path = Path(SPARK_PROMOTION_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(path)
        except Exception:
            logger.debug("Failed to save spark promotion state", exc_info=True)

    def _load(self) -> None:
        try:
            path = Path(SPARK_PROMOTION_PATH)
            if not path.exists():
                return
            data = json.loads(path.read_text())
            self._state.level = int(data.get("level", 0) or 0)
            self._state.shadow_start_ts = data.get("shadow_start_ts", time.time())
            self._state.total_outcomes = int(data.get("total_outcomes", 0) or 0)
            for v in data.get("validation_history", []):
                self._state.validation_history.append(float(v))
            self._state.last_promoted_at = data.get("last_promoted_at", 0.0)
            self._state.last_demoted_at = data.get("last_demoted_at", 0.0)
        except Exception:
            logger.debug("Failed to load spark promotion state", exc_info=True)

    def _promotion_eligible(self) -> bool:
        if self._state.level >= 2:
            return False
        if self._state.total_outcomes < SPARK_MIN_OUTCOMES:
            return False
        hours = (time.time() - self._state.shadow_start_ts) / 3600.0
        if hours < SPARK_MIN_SHADOW_HOURS:
            return False
        hist = list(self._state.validation_history)
        if len(hist) < SPARK_MIN_OUTCOMES:
            return False
        rate = sum(hist) / len(hist)
        return rate >= SPARK_PROMOTE_VALIDATION_RATE

    def _check_transitions(self) -> None:
        now = time.time()
        last_transition = max(self._state.last_promoted_at, self._state.last_demoted_at)
        if last_transition > 0 and (now - last_transition) < SPARK_TRANSITION_COOLDOWN_S:
            return

        if self._promotion_eligible():
            old = self._state.level
            self._state.level = min(self._state.level + 1, 2)
            if self._state.level != old:
                self._state.last_promoted_at = now
                logger.info("Spark gate promoted: level %d → %d", old, self._state.level)
                self.save()
            return

        hist = list(self._state.validation_history)
        if len(hist) >= SPARK_DEMOTE_WINDOW and self._state.level > 0:
            recent = hist[-SPARK_DEMOTE_WINDOW:]
            rate = sum(recent) / len(recent)
            if rate < SPARK_DEMOTE_VALIDATION_RATE:
                old = self._state.level
                self._state.level = max(self._state.level - 1, 0)
                self._state.last_demoted_at = now
                self._state.shadow_start_ts = now
                logger.warning(
                    "Spark gate demoted: level %d → %d (rate %.2f < %.2f)",
                    old, self._state.level, rate, SPARK_DEMOTE_VALIDATION_RATE,
                )
                self.save()
