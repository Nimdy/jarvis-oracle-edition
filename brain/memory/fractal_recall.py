"""Fractal Recall — background associative recall engine.

Surfaces grounded memory chains during waking operation, feeds them into
the curiosity/proactive/epistemic stack.  Never speaks directly, never
creates canonical memories (Rollout 1).

Spec reference: docs/BIG_DWAG_LOSTLEVEL_MEMORY_UPGRADES.md
"""

from __future__ import annotations

import logging
import math
import time as _time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

CueClass = Literal[
    "human_present",
    "ambient_environmental",
    "reflective_internal",
    "technical_self_model",
]

GovernanceAction = Literal[
    "ignore",
    "hold_for_curiosity",
    "eligible_for_proactive",
    "reflective_only",
]

SeedClass = Literal[
    "identity_seed",
    "bootstrap_seed",
    "system_seed",
    "generic_seed",
]

# ---------------------------------------------------------------------------
# Constants  (all configurable module-level values)
# ---------------------------------------------------------------------------

FRACTAL_RECALL_INTERVAL_S = 30.0
MIN_CUE_STRENGTH = 0.15
RESONANCE_THRESHOLD = 0.40
CHAIN_CONTINUATION_THRESHOLD = 0.35
RECALL_COOLDOWN_S = 120.0
MAX_RECALLS_PER_HOUR = 5
MAX_CHAIN_LENGTH = 5
MAX_CHAIN_DEPTH = 3
RECENT_RECALL_HISTORY_MAXLEN = 50
LOW_SIGNAL_MARGIN = 0.05

W_SEM = 0.25
W_TAG = 0.18
W_TEMPORAL = 0.12
W_EMOTION = 0.12
W_ASSOC = 0.08
W_PROVENANCE = 0.10
W_MODE_FIT = 0.05
W_RECENCY = 0.10

# Later-rollout constants (defined, never called in Rollout 1)
RECALL_BOOST = 0.0025
FRACTAL_REINFORCE_COOLDOWN_S = 4 * 3600
FRACTAL_REINFORCE_CAP_24H = 0.02

_STOP_WORDS = frozenset({
    "the", "and", "for", "are", "was", "were", "been", "being", "have", "has",
    "had", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "this", "that", "these", "those", "with", "from", "into",
    "about", "your", "what", "how", "why", "when", "where", "who", "which",
    "you", "not", "but", "its", "all", "any", "more", "some", "than", "too",
    "very", "just", "also", "now", "here", "there", "then", "each", "every",
})

# System-self lexicon for technical_self_model detection
_SELF_LEXICON = frozenset({
    "jarvis", "system", "memory", "brain", "consciousness", "architecture",
    "subsystem", "belief", "status", "self", "identity", "soul", "epistemic",
    "calibration",
})

# Identity-sensitivity tag keywords
_IDENTITY_KEYWORDS = frozenset({
    "identity", "creator", "soul", "self", "who_am_i",
    "core_value", "relationship",
})

# Dream/synthetic provenances — hard block (fitness < 0)
_DREAM_PROVENANCES = frozenset({"dream_observer", "dream_replay"})

# Tags indicating dream origin
_DREAM_ORIGIN_TAGS = frozenset({
    "dream_insight", "dream_hypothesis", "sleep_candidate",
    "dream_artifact", "dream_consolidation",
})

# Meta-tags that are bookkeeping, not content — excluded from tag overlap scoring
_META_TAG_PREFIXES = (
    "consolidated", "dream_consolidation", "dream_artifact",
    "dream_consolidation_proposal", "consolidated_into:",
    "claim_type:", "automatic",
)


def _filter_content_tags(tags: set[str]) -> set[str]:
    """Strip bookkeeping meta-tags, keeping only content-bearing tags."""
    return {t for t in tags if not any(t.startswith(p) for p in _META_TAG_PREFIXES)}

# ---------------------------------------------------------------------------
# Provenance fitness matrix  (spec section 8.6)
# ---------------------------------------------------------------------------

_PROVENANCE_FITNESS: dict[str, dict[CueClass, float]] = {
    "observed":          {"human_present": 1.0, "ambient_environmental": 0.8, "reflective_internal": 0.9, "technical_self_model": 0.5},
    "user_claim":        {"human_present": 1.0, "ambient_environmental": 0.6, "reflective_internal": 0.8, "technical_self_model": 0.4},
    "conversation":      {"human_present": 0.9, "ambient_environmental": 0.5, "reflective_internal": 0.7, "technical_self_model": 0.3},
    "model_inference":   {"human_present": 0.3, "ambient_environmental": 0.3, "reflective_internal": 0.5, "technical_self_model": 0.4},
    "external_source":   {"human_present": 0.1, "ambient_environmental": 0.2, "reflective_internal": 0.4, "technical_self_model": 0.7},
    "experiment_result": {"human_present": 0.2, "ambient_environmental": 0.2, "reflective_internal": 0.5, "technical_self_model": 0.8},
    "derived_pattern":   {"human_present": 0.2, "ambient_environmental": 0.3, "reflective_internal": 0.6, "technical_self_model": 0.7},
    "unknown":           {"human_present": 0.0, "ambient_environmental": 0.0, "reflective_internal": 0.0, "technical_self_model": 0.0},
}

_SEED_FITNESS: dict[SeedClass, dict[CueClass, float]] = {
    "identity_seed":  {"human_present": 0.5, "ambient_environmental": 0.3, "reflective_internal": 0.5, "technical_self_model": 0.4},
    "bootstrap_seed": {"human_present": 0.3, "ambient_environmental": 0.3, "reflective_internal": 0.3, "technical_self_model": 0.3},
    "system_seed":    {"human_present": 0.2, "ambient_environmental": 0.2, "reflective_internal": 0.3, "technical_self_model": 0.5},
    "generic_seed":   {"human_present": 0.3, "ambient_environmental": 0.2, "reflective_internal": 0.3, "technical_self_model": 0.3},
}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AmbientCue:
    text: str
    tags: tuple[str, ...]
    hour_bucket: int
    emotion: str
    speaker_id: str | None
    mode: str
    topic: str | None
    engagement: float
    scene_entities: tuple[str, ...]
    cue_strength: float
    cue_class: CueClass


@dataclass(slots=True)
class RecallCandidate:
    memory_id: str
    memory: Any
    resonance: float
    semantic_score: float
    tag_score: float
    temporal_score: float
    emotion_score: float
    association_score: float
    provenance_weight: float
    mode_fit: float
    recency_penalty: float
    dominant_source: str
    source_paths: tuple[str, ...]
    dominant_tag: str | None
    identity_sensitive: bool


@dataclass(slots=True)
class FractalRecallResult:
    cue: AmbientCue
    seed: RecallCandidate
    chain: list[RecallCandidate]
    governance_recommended_action: GovernanceAction
    governance_confidence: float
    governance_reason_codes: tuple[str, ...]
    provenance_mix: dict[str, int]
    timestamp: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_seed(memory: Any) -> SeedClass:
    """Classify a seed memory by inspecting tags/payload for identity vs
    bootstrap vs system markers."""
    tags = set(getattr(memory, "tags", ()) or ())
    is_core = getattr(memory, "is_core", False)

    if is_core or tags & {"identity", "soul", "core_value", "who_am_i"}:
        return "identity_seed"
    if "gestation" in tags or any(t.startswith("gestation") for t in tags):
        return "bootstrap_seed"
    if any(t.startswith("system_") for t in tags):
        return "system_seed"
    return "generic_seed"


def is_identity_sensitive(memory: Any) -> bool:
    """Return True if memory touches identity / creator / self-model claims."""
    tags = set(getattr(memory, "tags", ()) or ())
    if tags & _IDENTITY_KEYWORDS:
        return True
    mem_type = getattr(memory, "type", "")
    if mem_type == "core":
        return True
    payload = getattr(memory, "payload", "")
    if isinstance(payload, str):
        payload_lower = payload.lower()
        if any(kw in payload_lower for kw in ("creator", "who am i", "my identity", "my soul")):
            return True
    prov = getattr(memory, "provenance", "")
    if prov == "seed":
        sc = _classify_seed(memory)
        if sc == "identity_seed":
            return True
    return False


def provenance_fitness(provenance: str, cue_class: CueClass, memory: Any = None) -> float:
    """Lookup provenance fitness for a given cue class.

    Returns < 0 for hard-blocked provenances (dream/synthetic).
    """
    if provenance in _DREAM_PROVENANCES:
        return -1.0

    tags = set(getattr(memory, "tags", ()) or ()) if memory else set()
    if tags & _DREAM_ORIGIN_TAGS:
        return -1.0

    if provenance == "seed" and memory is not None:
        sc = _classify_seed(memory)
        return _SEED_FITNESS.get(sc, _SEED_FITNESS["generic_seed"]).get(cue_class, 0.0)

    row = _PROVENANCE_FITNESS.get(provenance)
    if row is None:
        return 0.0
    return row.get(cue_class, 0.0)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class FractalRecallEngine:
    """Background associative recall engine — event-only mode (Rollout 1)."""

    def __init__(
        self,
        memory_storage: Any,
        vector_store: Any,
        scene_tracker: Any = None,
        emotion_classifier: Any = None,
        attention_core: Any = None,
        mode_manager: Any = None,
        world_state: Any = None,
        event_bus: Any = None,
        kernel_thought_emitter: Any = None,
        clock: Any = None,
        logger_override: Any = None,
    ) -> None:
        self._storage = memory_storage
        self._vector_store = vector_store
        self._scene_tracker = scene_tracker
        self._emotion_classifier = emotion_classifier
        self._attention_core = attention_core
        self._mode_manager = mode_manager
        self._world_state = world_state
        self._event_bus = event_bus
        self._kernel_thought_emitter = kernel_thought_emitter
        self._clock = clock or _time.time
        self._log = logger_override or logger

        # Telemetry
        self._last_recall_ts: float = 0.0
        self._recent_recall_timestamps: deque[float] = deque(maxlen=MAX_RECALLS_PER_HOUR * 2)
        self._recall_history: deque[dict[str, Any]] = deque(maxlen=RECENT_RECALL_HISTORY_MAXLEN)
        self._governance_outcomes: Counter[str] = Counter()
        self._blocked_mode_skips: int = 0
        self._cooldown_skips: int = 0
        self._low_signal_skips: int = 0
        self._no_candidates_skips: int = 0
        self._no_seed_skips: int = 0
        self._total_ticks: int = 0
        self._total_recalls: int = 0
        self._avg_resonance_running: float = 0.0
        self._avg_chain_length_running: float = 0.0
        self._cumulative_provenance_mix: Counter[str] = Counter()

        self._hrr_advisor = None
        try:
            from library.vsa.runtime_config import HRRRuntimeConfig
            from library.vsa.status import (
                register_recall_advisory_reader,
                register_recall_advisory_recent,
            )

            _hrr_cfg = HRRRuntimeConfig.from_env()
            if _hrr_cfg.enabled:
                from memory.hrr_recall_advisor import HRRRecallAdvisor

                self._hrr_advisor = HRRRecallAdvisor(_hrr_cfg)
                register_recall_advisory_reader(self._hrr_advisor.status)
                if hasattr(self._hrr_advisor, "recent"):
                    register_recall_advisory_recent(self._hrr_advisor.recent)
        except Exception:
            self._hrr_advisor = None

    # ===== PUBLIC API ======================================================

    def tick(self, now: float) -> FractalRecallResult | None:
        """Main entry — called every FRACTAL_RECALL_INTERVAL_S.

        Stage 1 note: IntentionResolver's consciousness_system tick
        may consult this engine read-only for associative topic
        reinforcement when building ResolverSignal features. The
        resolver never writes recall events — it is read-only here.
        """
        self._total_ticks += 1

        # Cooldown
        if now - self._last_recall_ts < RECALL_COOLDOWN_S:
            self._cooldown_skips += 1
            return None

        # Hourly cap
        cutoff = now - 3600.0
        hourly = sum(1 for t in self._recent_recall_timestamps if t > cutoff)
        if hourly >= MAX_RECALLS_PER_HOUR:
            self._cooldown_skips += 1
            return None

        # Build cue
        cue = self.build_cue(now)
        if cue is None or cue.cue_strength < MIN_CUE_STRENGTH:
            self._low_signal_skips += 1
            return None

        # Probe
        candidates = self.probe(cue, now)
        if not candidates:
            self._no_candidates_skips += 1
            return None

        # Select seed
        seed = self.select_seed(candidates, cue)
        if seed is None:
            self._no_seed_skips += 1
            if candidates:
                top = sorted(candidates, key=lambda c: c.resonance, reverse=True)[:3]
                self._log.info(
                    "Fractal recall: no seed above %.2f — top %d candidates: %s (cue=%s, vs=%s)",
                    RESONANCE_THRESHOLD,
                    len(candidates),
                    [(c.memory_id[:8], round(c.resonance, 3),
                      f"sem={c.semantic_score:.2f} tag={c.tag_score:.2f} prov={c.provenance_weight:.2f} mode={c.mode_fit:.2f}")
                     for c in top],
                    cue.cue_class,
                    "yes" if self._vector_store and getattr(self._vector_store, "available", False) else "no",
                )
            return None

        # Walk chain
        chain = self.walk_chain(seed, cue, now)
        if not chain:
            return None

        # Governance
        action, confidence, reason_codes = self.recommend_governance_action(cue, chain)

        # Provenance mix
        prov_mix: Counter[str] = Counter()
        for c in chain:
            prov = getattr(c.memory, "provenance", "unknown")
            prov_mix[prov] += 1

        result = FractalRecallResult(
            cue=cue,
            seed=seed,
            chain=chain,
            governance_recommended_action=action,
            governance_confidence=confidence,
            governance_reason_codes=reason_codes,
            provenance_mix=dict(prov_mix),
            timestamp=now,
        )

        # Persist telemetry
        self._last_recall_ts = now
        self._recent_recall_timestamps.append(now)
        self._total_recalls += 1
        self._governance_outcomes[action] += 1
        self._cumulative_provenance_mix.update(prov_mix)

        # Running averages
        avg_res = sum(c.resonance for c in chain) / len(chain) if chain else 0.0
        n = self._total_recalls
        self._avg_resonance_running = (
            self._avg_resonance_running * (n - 1) / n + avg_res / n
        )
        self._avg_chain_length_running = (
            self._avg_chain_length_running * (n - 1) / n + len(chain) / n
        )

        self._recall_history.append({
            "ts": now,
            "cue_class": cue.cue_class,
            "cue_summary": cue.text[:120],
            "seed_id": seed.memory_id,
            "seed_score": round(seed.resonance, 4),
            "chain_ids": [c.memory_id for c in chain],
            "provenance_mix": dict(prov_mix),
            "governance_action": action,
            "governance_confidence": round(confidence, 4),
            "reason_codes": list(reason_codes),
        })

        if self._hrr_advisor is not None:
            try:
                self._hrr_advisor.observe(cue, seed, chain, governance_action=action)
            except Exception:
                pass  # observer never affects recall output

        return result

    # ===== CUE CONSTRUCTION ================================================

    def build_cue(self, now: float | None = None) -> AmbientCue | None:
        """Gather live perception state and build a deterministic ambient cue."""
        now = now or self._clock()

        scene_entities = self._gather_scene_entities()
        emotion = self._gather_emotion()
        speaker_id = self._gather_speaker()
        engagement = self._gather_engagement()
        mode = self._gather_mode()
        topic = self._gather_topic()
        hour_bucket = _time.localtime(now).tm_hour

        # Cue class — hard rules in priority order (spec 6.4)
        cue_class = self._classify_cue_class(
            topic=topic, mode=mode, speaker_id=speaker_id,
            engagement=engagement, scene_entities=scene_entities,
        )

        # Cue strength (spec 6.5)
        cue_strength = (
            0.30 * float(bool(scene_entities))
            + 0.20 * float(bool(emotion and emotion != "neutral"))
            + 0.20 * float(bool(speaker_id))
            + 0.20 * float(bool(topic))
            + 0.10 * min(max(engagement, 0.0), 1.0)
        )

        # Text construction (spec 6.2)
        parts: list[str] = []
        if scene_entities:
            parts.append("scene: " + ", ".join(scene_entities[:8]))
        if emotion:
            parts.append(f"emotion: {emotion}")
        if speaker_id:
            parts.append(f"speaker: {speaker_id}")
        if topic:
            parts.append(f"topic: {topic}")
        parts.append(f"mode: {mode}")
        text = " | ".join(parts)

        # Tags (spec 6.3 + vocabulary alignment)
        tags: list[str] = []
        for e in scene_entities[:8]:
            tags.append(e.lower().replace(" ", "_"))
        if emotion and emotion != "neutral":
            tags.append(emotion.lower())
            tags.append("emotion")
        if speaker_id:
            tags.append(f"speaker:{speaker_id}")
        if topic:
            for tok in topic.lower().split()[:6]:
                cleaned = tok.strip(".,!?;:'\"")
                if len(cleaned) > 2 and cleaned not in _STOP_WORDS:
                    tags.append(cleaned)

        # Conversation-context tags that match stored memory vocabulary
        conv_state = None
        if self._world_state:
            try:
                conv_state = getattr(self._world_state, "conversation", None)
            except Exception:
                pass

        if conv_state and getattr(conv_state, "active", False):
            tags.append("conversation")
            route = getattr(conv_state, "current_route", "")
            if route:
                route_tag_map = {
                    "INTROSPECTION": "self_reflection",
                    "STATUS": "self_reflection",
                    "MEMORY": "self_reflection",
                    "IDENTITY": "identity",
                    "SKILL": "preference",
                }
                mapped = route_tag_map.get(route.upper())
                if mapped:
                    tags.append(mapped)

        if scene_entities:
            tags.append("scene")
        tags.append(mode)
        if mode == "conversational":
            tags.append("conversation")
        tags.append(f"hour_{hour_bucket}")

        return AmbientCue(
            text=text,
            tags=tuple(tags),
            hour_bucket=hour_bucket,
            emotion=emotion,
            speaker_id=speaker_id,
            mode=mode,
            topic=topic,
            engagement=engagement,
            scene_entities=tuple(scene_entities),
            cue_strength=cue_strength,
            cue_class=cue_class,
        )

    def _classify_cue_class(
        self,
        *,
        topic: str | None,
        mode: str,
        speaker_id: str | None,
        engagement: float,
        scene_entities: list[str],
    ) -> CueClass:
        """Hard rules in priority order (spec section 6.4)."""
        # 1. technical_self_model
        if topic:
            topic_lower = topic.lower()
            if any(w in topic_lower for w in _SELF_LEXICON):
                return "technical_self_model"

        # Check world_state for route/intent if available
        if self._world_state:
            try:
                ws = self._world_state
                conv_state = getattr(ws, "conversation", None)
                if conv_state:
                    route = getattr(conv_state, "current_route", None)
                    if route and route.upper() in {"STATUS", "MEMORY", "INTROSPECTION", "SELF_MODEL"}:
                        return "technical_self_model"
            except Exception:
                pass

        # 2. human_present
        person_detected = bool(speaker_id)
        if not person_detected and self._attention_core:
            try:
                attn = self._attention_core.get_state()
                person_detected = attn.get("person_present", False)
            except Exception:
                pass
        if (speaker_id or person_detected) and engagement > 0.30:
            return "human_present"

        # 3. reflective_internal
        if mode in ("reflective", "passive"):
            has_conversation = False
            if self._world_state:
                try:
                    conv = getattr(self._world_state, "conversation", None)
                    if conv and getattr(conv, "turn_count", 0) > 0:
                        has_conversation = True
                except Exception:
                    pass
            if not has_conversation and not speaker_id:
                return "reflective_internal"

        # 4. ambient_environmental (fallback)
        return "ambient_environmental"

    # ===== PROBE / MERGE ===================================================

    def probe(self, cue: AmbientCue, now: float) -> list[RecallCandidate]:
        """Multi-path probe: semantic + tag + temporal, merged by memory_id."""
        merged: dict[str, dict[str, Any]] = {}

        # Semantic search
        if self._vector_store and getattr(self._vector_store, "available", False):
            try:
                sem_results = self._vector_store.search(cue.text, top_k=10)
                for r in sem_results:
                    mid = r.get("memory_id", "")
                    if not mid:
                        continue
                    sim = r.get("similarity", 0.0)
                    entry = merged.setdefault(mid, {
                        "semantic_score": 0.0, "tag_score": 0.0,
                        "temporal_score": 0.0, "source_paths": [],
                    })
                    entry["semantic_score"] = max(entry["semantic_score"], sim)
                    entry["source_paths"].append("semantic")
            except Exception:
                self._log.debug("Fractal probe semantic search failed", exc_info=True)

        # Tag lookup
        for tag in cue.tags:
            try:
                tag_mems = self._storage.get_by_tag(tag)
                for mem in tag_mems:
                    mid = mem.id
                    entry = merged.setdefault(mid, {
                        "semantic_score": 0.0, "tag_score": 0.0,
                        "temporal_score": 0.0, "source_paths": [],
                    })
                    if "tag" not in entry["source_paths"]:
                        entry["source_paths"].append("tag")
            except Exception:
                pass

        # Temporal probe — hour bucket match
        try:
            all_recent = self._storage.get_recent(200)
            for mem in all_recent:
                ts = getattr(mem, "timestamp", 0.0)
                if ts <= 0:
                    continue
                mem_hour = _time.localtime(ts).tm_hour
                if mem_hour == cue.hour_bucket:
                    t_score = 1.0
                elif abs(mem_hour - cue.hour_bucket) <= 1 or abs(mem_hour - cue.hour_bucket) >= 23:
                    t_score = 0.5
                else:
                    continue
                mid = mem.id
                entry = merged.setdefault(mid, {
                    "semantic_score": 0.0, "tag_score": 0.0,
                    "temporal_score": 0.0, "source_paths": [],
                })
                entry["temporal_score"] = max(entry["temporal_score"], t_score)
                if "temporal" not in entry["source_paths"]:
                    entry["source_paths"].append("temporal")
        except Exception:
            self._log.debug("Fractal probe temporal scan failed", exc_info=True)

        # Compute tag scores for all merged candidates
        cue_tags = set(cue.tags)

        candidates: list[RecallCandidate] = []
        for mid, entry in merged.items():
            mem = self._storage.get(mid)
            if mem is None:
                continue

            prov = getattr(mem, "provenance", "unknown")
            pf = provenance_fitness(prov, cue.cue_class, mem)
            if pf < 0:
                continue

            mem_tags = set(getattr(mem, "tags", ()) or ())
            content_tags = _filter_content_tags(mem_tags)
            tag_overlap = len(content_tags & cue_tags)
            tag_score = tag_overlap / max(len(content_tags), 1)

            # Resolve remaining scores
            semantic_score = entry["semantic_score"]
            temporal_score = entry["temporal_score"]
            emotion_score = self._compute_emotion_score(mem, cue)
            association_score = min(getattr(mem, "association_count", 0), 10) / 10.0
            recency_penalty = self._compute_recency_penalty(mem, now)
            mode_fit = self._compute_mode_fit(mem, cue)
            id_sensitive = is_identity_sensitive(mem)

            dominant_tag: str | None = None
            if mem_tags:
                dominant_tag = max(mem_tags, key=lambda t: 1 if t in cue_tags else 0)

            resonance = max(0.0, min(1.0,
                W_SEM * semantic_score
                + W_TAG * tag_score
                + W_TEMPORAL * temporal_score
                + W_EMOTION * emotion_score
                + W_ASSOC * association_score
                + W_PROVENANCE * pf
                + W_MODE_FIT * mode_fit
                - W_RECENCY * recency_penalty
            ))

            candidates.append(RecallCandidate(
                memory_id=mid,
                memory=mem,
                resonance=resonance,
                semantic_score=semantic_score,
                tag_score=tag_score,
                temporal_score=temporal_score,
                emotion_score=emotion_score,
                association_score=association_score,
                provenance_weight=pf,
                mode_fit=mode_fit,
                recency_penalty=recency_penalty,
                dominant_source=entry["source_paths"][0] if entry["source_paths"] else "unknown",
                source_paths=tuple(entry["source_paths"]),
                dominant_tag=dominant_tag,
                identity_sensitive=id_sensitive,
            ))

        return candidates

    # ===== SEED SELECTION ==================================================

    def select_seed(
        self, candidates: list[RecallCandidate], cue: AmbientCue,
    ) -> RecallCandidate | None:
        """Return the highest-resonance candidate above threshold. No fallback."""
        candidates.sort(key=lambda c: c.resonance, reverse=True)
        for c in candidates:
            if c.resonance >= RESONANCE_THRESHOLD:
                return c
        return None

    # ===== CHAIN WALKING ===================================================

    def walk_chain(
        self, seed: RecallCandidate, cue: AmbientCue, now: float,
    ) -> list[RecallCandidate]:
        """BFS/DFS via association graph, anchored to the original cue."""
        chain: list[RecallCandidate] = [seed]
        visited: set[str] = {seed.memory_id}
        tag_counts: Counter[str] = Counter()
        prov_counts: Counter[str] = Counter()

        if seed.dominant_tag:
            tag_counts[seed.dominant_tag] += 1
        prov_counts[getattr(seed.memory, "provenance", "unknown")] += 1

        consecutive_weak = 0

        neighbors = self._storage.get_related(seed.memory_id, depth=MAX_CHAIN_DEPTH)

        for mem in neighbors:
            if len(chain) >= MAX_CHAIN_LENGTH:
                break

            mid = mem.id
            if mid in visited:
                continue
            visited.add(mid)

            prov = getattr(mem, "provenance", "unknown")
            pf = provenance_fitness(prov, cue.cue_class, mem)
            if pf < 0:
                break

            # Score against original cue
            cue_tags = set(cue.tags)
            mem_tags = set(getattr(mem, "tags", ()) or ())
            content_tags = _filter_content_tags(mem_tags)
            tag_overlap = len(content_tags & cue_tags)
            tag_score = tag_overlap / max(len(content_tags), 1)
            emotion_score = self._compute_emotion_score(mem, cue)
            association_score = min(getattr(mem, "association_count", 0), 10) / 10.0
            recency_penalty = self._compute_recency_penalty(mem, now)
            mode_fit = self._compute_mode_fit(mem, cue)

            resonance = max(0.0, min(1.0,
                W_SEM * 0.0  # no semantic score for neighbor hops
                + W_TAG * tag_score
                + W_TEMPORAL * 0.0
                + W_EMOTION * emotion_score
                + W_ASSOC * association_score
                + W_PROVENANCE * pf
                + W_MODE_FIT * mode_fit
                - W_RECENCY * recency_penalty
            ))

            # Stop: below continuation threshold
            if resonance < CHAIN_CONTINUATION_THRESHOLD:
                consecutive_weak += 1
                if consecutive_weak >= 2:
                    break
                continue

            # Stop: two consecutive near-threshold hops
            if resonance < CHAIN_CONTINUATION_THRESHOLD + LOW_SIGNAL_MARGIN:
                consecutive_weak += 1
                if consecutive_weak >= 2:
                    break
            else:
                consecutive_weak = 0

            dominant_tag: str | None = None
            if mem_tags:
                dominant_tag = max(mem_tags, key=lambda t: 1 if t in cue_tags else 0)

            # Stop: same dominant tag repeated > 2 times
            if dominant_tag and tag_counts[dominant_tag] >= 2:
                break

            # Stop: same provenance type repeated > 3 times
            if prov_counts[prov] >= 3:
                break

            tag_counts[dominant_tag or ""] += 1
            prov_counts[prov] += 1

            candidate = RecallCandidate(
                memory_id=mid,
                memory=mem,
                resonance=resonance,
                semantic_score=0.0,
                tag_score=tag_score,
                temporal_score=0.0,
                emotion_score=emotion_score,
                association_score=association_score,
                provenance_weight=pf,
                mode_fit=mode_fit,
                recency_penalty=recency_penalty,
                dominant_source="association",
                source_paths=("association",),
                dominant_tag=dominant_tag,
                identity_sensitive=is_identity_sensitive(mem),
            )
            chain.append(candidate)

        return chain

    # ===== GOVERNANCE ======================================================

    def recommend_governance_action(
        self, cue: AmbientCue, chain: list[RecallCandidate],
    ) -> tuple[GovernanceAction, float, tuple[str, ...]]:
        """Determine governance recommendation with confidence and reason codes."""
        reason_codes: list[str] = []
        has_identity = any(c.identity_sensitive for c in chain)
        avg_resonance = sum(c.resonance for c in chain) / len(chain) if chain else 0.0

        # Grounding ratio: observed / user_claim / conversation provenances
        grounded_provs = {"observed", "user_claim", "conversation"}
        grounded_count = sum(
            1 for c in chain
            if getattr(c.memory, "provenance", "unknown") in grounded_provs
        )
        grounding_ratio = grounded_count / len(chain) if chain else 0.0

        # Provenance contamination
        blocked_count = sum(
            1 for c in chain
            if getattr(c.memory, "provenance", "unknown") in _DREAM_PROVENANCES
        )
        contamination = blocked_count / len(chain) if chain else 0.0

        # Repetition check
        prov_types = [getattr(c.memory, "provenance", "unknown") for c in chain]
        prov_counter = Counter(prov_types)
        most_common_prov_count = prov_counter.most_common(1)[0][1] if prov_counter else 0
        repetitive = most_common_prov_count > 3 if len(chain) > 3 else False

        # --- ignore ---
        if cue.cue_strength < MIN_CUE_STRENGTH + LOW_SIGNAL_MARGIN:
            reason_codes.append("low_cue_strength")
        if grounding_ratio < 0.3:
            reason_codes.append("blocked_provenance_mix")
        if repetitive:
            reason_codes.append("repetitive_chain")
        if avg_resonance < RESONANCE_THRESHOLD:
            reason_codes.append("low_confidence")
        if contamination > 0.5:
            reason_codes.append("blocked_provenance_mix")

        if "low_cue_strength" in reason_codes or "low_confidence" in reason_codes or contamination > 0.5:
            confidence = self._compute_governance_confidence(avg_resonance, grounding_ratio, cue.cue_strength, repetitive)
            return "ignore", confidence, tuple(reason_codes)
        if repetitive and grounding_ratio < 0.3:
            confidence = self._compute_governance_confidence(avg_resonance, grounding_ratio, cue.cue_strength, repetitive)
            return "ignore", confidence, tuple(reason_codes)

        # --- reflective_only ---
        if has_identity:
            reason_codes.append("identity_sensitive")
            confidence = self._compute_governance_confidence(avg_resonance, grounding_ratio, cue.cue_strength, repetitive)
            return "reflective_only", confidence, tuple(reason_codes)
        if cue.cue_class == "reflective_internal":
            reason_codes.append("technical_context")
            confidence = self._compute_governance_confidence(avg_resonance, grounding_ratio, cue.cue_strength, repetitive)
            return "reflective_only", confidence, tuple(reason_codes)
        if cue.cue_class == "technical_self_model":
            reason_codes.append("technical_context")
            confidence = self._compute_governance_confidence(avg_resonance, grounding_ratio, cue.cue_strength, repetitive)
            return "reflective_only", confidence, tuple(reason_codes)

        # --- eligible_for_proactive ---
        if (
            grounding_ratio >= 0.5
            and cue.cue_class == "human_present"
            and cue.engagement > 0.30
            and not has_identity
            and contamination == 0.0
            and not repetitive
            and avg_resonance >= RESONANCE_THRESHOLD
        ):
            reason_codes.append("high_grounding")
            reason_codes.append("high_relevance")
            if cue.engagement > 0.6:
                reason_codes.append("high_scene_overlap")
            confidence = self._compute_governance_confidence(avg_resonance, grounding_ratio, cue.cue_strength, repetitive)
            return "eligible_for_proactive", confidence, tuple(reason_codes)

        # --- hold_for_curiosity (default for anything interesting) ---
        if avg_resonance >= CHAIN_CONTINUATION_THRESHOLD:
            if cue.emotion and cue.emotion != "neutral":
                reason_codes.append("emotional_salience")
            if cue.topic:
                reason_codes.append("novel_topic")
            if not reason_codes:
                reason_codes.append("high_relevance")
            confidence = self._compute_governance_confidence(avg_resonance, grounding_ratio, cue.cue_strength, repetitive)
            return "hold_for_curiosity", confidence, tuple(reason_codes)

        # Fallback
        if not reason_codes:
            reason_codes.append("low_confidence")
        confidence = self._compute_governance_confidence(avg_resonance, grounding_ratio, cue.cue_strength, repetitive)
        return "ignore", confidence, tuple(reason_codes)

    def _compute_governance_confidence(
        self, avg_resonance: float, grounding_ratio: float,
        cue_strength: float, repetitive: bool,
    ) -> float:
        """Weighted confidence score [0, 1] (spec section 11.3)."""
        rep_inv = 0.0 if repetitive else 1.0
        raw = (
            0.40 * avg_resonance
            + 0.25 * grounding_ratio
            + 0.20 * cue_strength
            + 0.15 * rep_inv
        )
        return max(0.0, min(1.0, raw))

    # ===== EVENT EMISSION ==================================================

    def emit_surface(self, result: FractalRecallResult) -> None:
        """Emit FRACTAL_RECALL_SURFACED and KERNEL_THOUGHT. No TTS. No memory writes."""
        if not self._event_bus:
            return

        chain_items = []
        for c in result.chain:
            mem = c.memory
            chain_items.append({
                "memory_id": c.memory_id,
                "provenance": getattr(mem, "provenance", "unknown"),
                "memory_type": getattr(mem, "type", ""),
                "resonance": round(c.resonance, 4),
                "dominant_source": c.dominant_source,
                "identity_sensitive": c.identity_sensitive,
            })

        chain_payload_preview = []
        for c in result.chain:
            payload = getattr(c.memory, "payload", "")
            if isinstance(payload, str):
                chain_payload_preview.append(payload[:80])
            else:
                chain_payload_preview.append(str(payload)[:80])

        try:
            from consciousness.events import FRACTAL_RECALL_SURFACED
            self._event_bus.emit(
                FRACTAL_RECALL_SURFACED,
                cue_summary=result.cue.text[:200],
                cue_class=result.cue.cue_class,
                cue_strength=round(result.cue.cue_strength, 4),
                seed_id=result.seed.memory_id,
                seed_score=round(result.seed.resonance, 4),
                chain_ids=[c.memory_id for c in result.chain],
                chain_payload_preview=chain_payload_preview,
                chain_items=chain_items,
                resonance_scores=[round(c.resonance, 4) for c in result.chain],
                provenance_mix=result.provenance_mix,
                governance_recommended_action=result.governance_recommended_action,
                governance_confidence=round(result.governance_confidence, 4),
                governance_reason_codes=list(result.governance_reason_codes),
                timestamp=result.timestamp,
            )
        except Exception:
            self._log.debug("Fractal recall event emission failed", exc_info=True)

        # KERNEL_THOUGHT summary
        try:
            from consciousness.events import KERNEL_THOUGHT
            summary = (
                f"Fractal recall surfaced {len(result.chain)} memories "
                f"(cue={result.cue.cue_class}, "
                f"governance={result.governance_recommended_action}, "
                f"confidence={result.governance_confidence:.2f})"
            )
            self._event_bus.emit(KERNEL_THOUGHT, content=summary, tone="observational")
        except Exception:
            pass

    # ===== DASHBOARD STATE =================================================

    def get_state(self) -> dict[str, Any]:
        """Dashboard-ready snapshot dict (spec section 17)."""
        now = self._clock()
        cutoff = now - 3600.0
        count_1h = sum(1 for t in self._recent_recall_timestamps if t > cutoff)

        recent_recalls = list(self._recall_history)[-10:]

        return {
            "enabled": True,
            "total_ticks": self._total_ticks,
            "last_recall_ts": self._last_recall_ts,
            "total_count": self._total_recalls,
            "count_1h": count_1h,
            "blocked_mode_skips": self._blocked_mode_skips,
            "cooldown_skips": self._cooldown_skips,
            "low_signal_skips": self._low_signal_skips,
            "no_candidates_skips": self._no_candidates_skips,
            "no_seed_skips": self._no_seed_skips,
            "avg_resonance": round(self._avg_resonance_running, 4),
            "avg_chain_length": round(self._avg_chain_length_running, 2),
            "governance_outcomes": dict(self._governance_outcomes),
            "provenance_mix": dict(self._cumulative_provenance_mix),
            "recent_recalls": recent_recalls,
        }

    # ===== INTERNAL SCORING HELPERS ========================================

    def _compute_emotion_score(self, mem: Any, cue: AmbientCue) -> float:
        """1.0 if memory carries matching emotion tag and cue emotion is non-neutral."""
        if not cue.emotion or cue.emotion == "neutral":
            return 0.0
        mem_tags = set(getattr(mem, "tags", ()) or ())
        if cue.emotion.lower() in mem_tags:
            return 1.0
        return 0.0

    @staticmethod
    def _compute_recency_penalty(mem: Any, now: float) -> float:
        """Exponential decay penalty for recently accessed memories."""
        last_accessed = getattr(mem, "last_accessed", 0.0)
        if last_accessed <= 0:
            return 0.0
        return math.exp(-max(now - last_accessed, 0.0) / 3600.0)

    @staticmethod
    def _compute_mode_fit(mem: Any, cue: AmbientCue) -> float:
        """Simple v1: autobiographical/personal high in conversational/reflective,
        codebase/system high only in technical_self_model."""
        mem_type = getattr(mem, "type", "")
        mem_tags = set(getattr(mem, "tags", ()) or ())
        prov = getattr(mem, "provenance", "unknown")

        is_autobiographical = (
            mem_type in ("conversation", "observation", "user_preference")
            or prov in ("observed", "user_claim", "conversation")
        )
        is_technical = (
            prov in ("external_source", "experiment_result", "derived_pattern")
            or mem_type in ("factual_knowledge", "self_improvement")
            or bool(mem_tags & {"code_sourced", "codebase", "architecture"})
        )

        if cue.cue_class == "technical_self_model":
            return 0.8 if is_technical else 0.2
        if cue.cue_class in ("human_present", "reflective_internal"):
            return 0.8 if is_autobiographical else 0.3
        # ambient_environmental
        if is_technical:
            return 0.3
        return 0.5

    # ===== PERCEPTION GATHERING ============================================

    def _gather_scene_entities(self) -> list[str]:
        if not self._scene_tracker:
            return []
        try:
            state = self._scene_tracker.get_state()
            entities = state.get("entities", [])
            return [
                e.get("label", "") for e in entities
                if isinstance(e, dict) and e.get("state") == "visible"
            ][:8]
        except Exception:
            return []

    def _gather_emotion(self) -> str:
        if self._emotion_classifier:
            try:
                return getattr(self._emotion_classifier, "current_emotion", "neutral") or "neutral"
            except Exception:
                pass
        return "neutral"

    def _gather_speaker(self) -> str | None:
        if self._attention_core:
            try:
                state = self._attention_core.get_state()
                speaker = state.get("speaker_identity")
                if speaker and speaker != "unknown":
                    return speaker
            except Exception:
                pass
        # Fall back to world-state face/fusion identity when voice is inactive
        if self._world_state:
            try:
                ws = getattr(self._world_state, "current_state", None)
                user = getattr(ws, "user", None) if ws else None
                if user and getattr(user, "present", False):
                    name = getattr(user, "speaker_name", "")
                    if name and name != "unknown":
                        return name
            except Exception:
                pass
        return None

    def _gather_engagement(self) -> float:
        if self._attention_core:
            try:
                state = self._attention_core.get_state()
                eng = state.get("engagement_level", 0.0)
                if isinstance(eng, (int, float)) and eng > 0.0:
                    return float(eng)
            except Exception:
                pass
        if self._world_state:
            try:
                ws = getattr(self._world_state, "current_state", None)
                user = getattr(ws, "user", None) if ws else None
                if user:
                    val = getattr(user, "engagement", 0.0)
                    if isinstance(val, (int, float)):
                        return float(val)
            except Exception:
                pass
        return 0.0

    def _gather_mode(self) -> str:
        if self._mode_manager:
            try:
                return self._mode_manager.mode
            except Exception:
                pass
        return "passive"

    def _gather_topic(self) -> str | None:
        if self._world_state:
            try:
                conv = getattr(self._world_state, "conversation", None)
                if conv:
                    topic = getattr(conv, "topic", None)
                    if topic:
                        return topic
                    last_text = getattr(conv, "last_user_text", None)
                    if last_text and len(last_text) > 5:
                        return last_text[:80]
            except Exception:
                pass
        return None
