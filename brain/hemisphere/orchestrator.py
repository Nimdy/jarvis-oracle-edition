"""HemisphereOrchestrator: top-level coordinator for the hemisphere NN system.

Replaces the React component NeuralEvolutionEngine.tsx.
Called from ConsciousnessSystem.on_tick() on a 120-second interval.
Manages auto-construction, evolution scheduling, and migration evaluation.
"""

from __future__ import annotations

import copy
import logging
import threading
import time as _time
import uuid as _uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from hemisphere.types import (
    DesignStrategy,
    DistillationConfig,
    DISTILLATION_CONFIGS,
    DynamicFocus,
    HemisphereFocus,
    HemisphereState,
    NetworkArchitecture,
    NetworkStatus,
)
from hemisphere.architect import NeuralArchitect
from hemisphere.engine import HemisphereEngine
from hemisphere.evolution import EvolutionEngine
from hemisphere.gap_detector import CognitiveGapDetector, CognitiveGap
from hemisphere.migration import MigrationAnalyzer
from hemisphere.registry import HemisphereRegistry
from hemisphere.data_feed import (
    HemisphereDataFeed,
    get_hemisphere_data_feed,
    get_safe_data_feed,
    prepare_distillation_tensors,
    should_initiate_evolution,
)
from hemisphere import event_bridge

logger = logging.getLogger(__name__)

OUTCOME_BUFFER_MAXLEN = 200
RETRAIN_THRESHOLD = 20

MAX_NETWORKS_PER_FOCUS = 5
MAX_TOTAL_NETWORKS = 12
EVOLUTION_DENSITY_THRESHOLD = 0.7
MIN_NETWORKS_FOR_EVOLUTION = 1
MIGRATION_CHECK_INTERVAL_S = 600.0
SUNSET_CYCLES = 10              # prune NN if no impact after this many evolution cycles


RESEARCH_CACHE_TTL_S = 600.0  # re-query memory every 10 minutes

TIER1_MIN_ACCURACY = 0.05
TIER1_MAX_CONSECUTIVE_FAILURES = 3
TIER1_MIN_SAMPLES_FOR_ACCURACY_FLOOR = 50  # don't apply accuracy floor with fewer samples (cold-start guard)

DISTILLATION_CADENCE_S = 300.0
DISTILLATION_CADENCE_DEEP_S = 120.0
DISTILLATION_REGRESSION_DELTA = 0.05
DISTILLATION_REGRESSION_MIN_SAMPLES = 50  # skip regression gate with fewer paired samples (cold-start overfit guard)
DISTILLATION_MAX_CONSECUTIVE_REGRESSIONS = 3
DISTILLATION_REGRESSION_COOLDOWN_S = 900.0
DISTILLATION_REGRESSION_COOLDOWN_BACKOFF = 2.0
DISTILLATION_REGRESSION_COOLDOWN_MAX_S = 7200.0
DISTILLATION_COOLDOWN_LOG_INTERVAL_S = 60.0
# Window after a dependency retrain during which dependents should skip, to
# avoid consuming freshly-perturbed student weights. The check uses a cycle-
# start snapshot of retrain timestamps so specialists processed later in the
# same loop iteration (e.g. emotion_depth after speaker_repr) are not blocked
# by dependencies that retrained moments earlier in the same cycle.
DISTILLATION_DEP_RETRAIN_WINDOW_S = 60.0

SLOT_SWAP_THRESHOLD = 1.15
SLOT_MIN_DWELL_CYCLES = 3

EXPANSION_MIN_PROMOTED = 2
EXPANSION_MIN_IMPACT = 0.05
EXPANSION_STABILITY_DAYS = 7
EXPANDED_SLOT_COUNT = 6

_TIER1_FOCUSES = frozenset({
    HemisphereFocus.EMOTION_DEPTH,
    HemisphereFocus.SPEAKER_REPR,
    HemisphereFocus.FACE_REPR,
    HemisphereFocus.PERCEPTION_FUSION,
    HemisphereFocus.VOICE_INTENT,
    HemisphereFocus.SPEAKER_DIARIZE,
    HemisphereFocus.PLAN_EVALUATOR,
    HemisphereFocus.DIAGNOSTIC,
    HemisphereFocus.CODE_QUALITY,
    HemisphereFocus.CLAIM_CLASSIFIER,
    HemisphereFocus.DREAM_SYNTHESIS,
    HemisphereFocus.SKILL_ACQUISITION,
    # P4 HRR shadow substrate: Tier-1 stub. No network is registered under
    # this focus this sprint, so _collect_tier1_features /
    # _update_broadcast_slots skip it (get_best_network returns None). The
    # focus exists solely to reserve a lifecycle seat at CANDIDATE_BIRTH.
    HemisphereFocus.HRR_ENCODER,
})

_SHADOW_ONLY_TIER1_FOCUSES = frozenset({
    HemisphereFocus.PLAN_EVALUATOR,
    HemisphereFocus.DIAGNOSTIC,
    HemisphereFocus.CODE_QUALITY,
    HemisphereFocus.CLAIM_CLASSIFIER,
    HemisphereFocus.DREAM_SYNTHESIS,
    HemisphereFocus.SKILL_ACQUISITION,
    HemisphereFocus.HRR_ENCODER,
})

_STANDARD_FOCUSES = frozenset({
    HemisphereFocus.MEMORY,
    HemisphereFocus.MOOD,
    HemisphereFocus.TRAITS,
    HemisphereFocus.GENERAL,
})


class HemisphereOrchestrator:
    """Coordinates hemisphere network design, training, evolution, and migration."""

    def __init__(self, device: str = "cpu") -> None:
        self._architect = NeuralArchitect()
        self._engine = HemisphereEngine(device=device)
        self._evolution_engines: dict[str, EvolutionEngine] = {
            f.value: EvolutionEngine() for f in HemisphereFocus
            if f in _STANDARD_FOCUSES
        }
        self._migration = MigrationAnalyzer()
        self._registry = HemisphereRegistry()
        self._gap_detector = CognitiveGapDetector()
        try:
            self._gap_detector.load_state()
        except Exception:
            logger.debug("GapDetector state load failed (non-critical)")

        self._networks: dict[str, NetworkArchitecture] = {}
        self._dynamic_focuses: dict[str, DynamicFocus] = {}
        self._build_lock = threading.Lock()
        self._networks_lock = threading.Lock()
        self._last_migration_check: float = 0.0
        self._cycle_count = 0
        self._enabled = True

        self._outcome_buffer: dict[str, deque] = {
            f.value: deque(maxlen=OUTCOME_BUFFER_MAXLEN) for f in HemisphereFocus
        }
        self._outcomes_since_train: dict[str, int] = {f.value: 0 for f in HemisphereFocus}

        self._research_cache: list[str] = []
        self._research_cache_time: float = 0.0

        # Distillation state
        self._distillation_collector: Any = None
        self._last_distillation_time: float = 0.0
        self._distillation_encoder_ids: dict[str, str] = {}
        self._tier1_regression_counts: dict[str, int] = {}
        self._tier1_regression_cooldown_until: dict[str, float] = {}
        self._tier1_cooldown_last_log: dict[str, float] = {}
        self._tier1_regression_cooldown_strikes: dict[str, int] = {}
        self._tier1_last_retrain_time: dict[str, float] = {}

        # Tier-1 accuracy gating: tracks consecutive build failures per focus
        self._tier1_failure_counts: dict[str, int] = {}
        self._tier1_disabled: set[str] = set()

        # Global Broadcast Slots (top-N signals fed to policy StateEncoder)
        self._num_broadcast_slots: int = 4
        self._broadcast_slots: list[dict[str, Any]] = [
            {"name": f.value, "value": 0.0, "score": 0.0, "dwell": 0}
            for f in (HemisphereFocus.MEMORY, HemisphereFocus.MOOD,
                      HemisphereFocus.TRAITS, HemisphereFocus.GENERAL)
        ]

        # M6 expansion state
        self._expansion_triggered: bool = False
        self._expansion_triggered_at: float = 0.0
        self._last_boot_stabilization_log: float = 0.0

        event_bridge.subscribe_plan_review()

    # ------------------------------------------------------------------
    # Research consultation
    # ------------------------------------------------------------------

    def _refresh_research_priors(self) -> None:
        """Query memory for NN-related research and feed it to the architect.

        Cached for RESEARCH_CACHE_TTL_S to avoid repeated queries every 120s tick.
        """
        now = _time.time()
        if now - self._research_cache_time < RESEARCH_CACHE_TTL_S and self._research_cache:
            return

        findings: list[str] = []
        try:
            from memory.search import semantic_search, search_by_tag

            nn_keywords = {
                "neural", "network", "architecture", "training", "activation",
                "optimizer", "learning rate", "regularization", "dropout",
                "loss function", "crossentropy", "gelu", "residual",
            }

            candidates = semantic_search(
                "neural network architecture design training techniques", top_k=10,
            )
            for m in candidates:
                if "autonomous_research" in (m.tags or []):
                    text = m.payload if isinstance(m.payload, str) else str(m.payload)
                    findings.append(text[:300])

            if len(findings) < 3:
                all_research = search_by_tag("autonomous_research")
                for m in all_research:
                    text = m.payload if isinstance(m.payload, str) else str(m.payload)
                    payload_lower = text.lower()
                    hits = sum(1 for kw in nn_keywords if kw in payload_lower)
                    if hits >= 2 and text[:300] not in findings:
                        findings.append(text[:300])
                        if len(findings) >= 10:
                            break

        except Exception:
            logger.debug("Hemisphere research query failed", exc_info=True)

        self._research_cache = findings
        self._research_cache_time = now
        self._architect.set_research_priors(findings)

        if findings:
            logger.info(
                "Hemisphere research priors updated: %d finding(s)", len(findings),
            )

    # ------------------------------------------------------------------
    # Main entry point (called from consciousness tick)
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        engine_state: dict[str, Any],
        memories: list[Any],
        traits: list[str],
    ) -> None:
        """Execute one hemisphere cycle. Safe to call frequently; internally gated."""
        if not self._enabled:
            return

        now = _time.time()
        feed = get_safe_data_feed(engine_state, memories, traits)
        boot_stabilization_active = bool(engine_state.get("boot_stabilization_active", False))
        if boot_stabilization_active:
            self._cycle_count += 1
            self._refresh_research_priors()
            self._feed_gap_detector(engine_state)
            if now - self._last_boot_stabilization_log >= 60.0:
                self._last_boot_stabilization_log = now
                remaining = float(engine_state.get("boot_stabilization_remaining_s", 0.0) or 0.0)
                logger.info(
                    "Hemisphere startup stabilization active (%.0fs remaining): "
                    "skipping construction/evolution/distillation/retraining",
                    remaining,
                )
            return

        if not should_initiate_evolution(feed):
            return

        self._cycle_count += 1
        constructed_this_cycle: set[str] = set()

        # Phase -1: refresh research priors (cached, lightweight)
        self._refresh_research_priors()

        # Phase 0: prune networks past sunset deadline or over cap
        self._prune_networks()

        # Phase 0.25: check specialist promotion ladder
        self._check_specialist_promotions()

        # Phase 0.3: check if broadcast expansion should trigger
        self._check_expansion_trigger()

        # Phase 0.5: feed cognitive gap detector with current metrics
        self._feed_gap_detector(engine_state)

        # Phase 1: gap-driven construction (purpose-built NNs from cognitive gaps)
        gaps = self._gap_detector.detect_gaps()
        for gap in gaps:
            if self._total_network_count() >= MAX_TOTAL_NETWORKS:
                logger.info("At network cap (%d), skipping gap-driven construction", MAX_TOTAL_NETWORKS)
                break
            if not self._has_network_for_gap(gap):
                self._construct_from_gap(feed, gap)
                constructed_this_cycle.add(gap.dimension)

        # Phase 2: construct for standard focuses only if they have zero networks
        for focus in _STANDARD_FOCUSES:
            if self._total_network_count() >= MAX_TOTAL_NETWORKS:
                break
            focus_nets = self._get_networks_for_focus(focus)
            if not focus_nets:
                self._construct_network(feed, focus)
                constructed_this_cycle.add(focus.value)

        # Phase 3: evolve existing networks (standard focuses only)
        for focus in _STANDARD_FOCUSES:
            if focus.value in constructed_this_cycle:
                continue
            focus_nets = self._get_networks_for_focus(focus)
            if len(focus_nets) >= MIN_NETWORKS_FOR_EVOLUTION:
                self._evolve_focus(feed, focus, focus_nets)

        # Phase 3.5: distillation cycle (Tier-1 student training)
        self._maybe_run_distillation(engine_state)

        # Phase 4: outcome-triggered retraining
        for focus in HemisphereFocus:
            if self._outcomes_since_train.get(focus.value, 0) >= RETRAIN_THRESHOLD:
                best = self._get_best_network(focus)
                if best is not None:
                    self._retrain_from_outcomes(feed, focus)
                    self._outcomes_since_train[focus.value] = 0

        # Phase 5: increment dynamic focus lifecycle
        for df in self._dynamic_focuses.values():
            df.cycles_alive += 1

        # Phase 6: periodic migration readiness check
        if now - self._last_migration_check > MIGRATION_CHECK_INTERVAL_S:
            self._last_migration_check = now
            self._check_migration(feed)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _construct_network(
        self, feed: HemisphereDataFeed, focus: HemisphereFocus,
    ) -> None:
        """Design and build a network in a background thread."""
        if not self._build_lock.acquire(blocking=False):
            return

        def _build() -> None:
            try:
                decision = self._architect.analyze_consciousness_data(feed)
                complexity = {
                    DesignStrategy.CONSERVATIVE: "simple",
                    DesignStrategy.EXPERIMENTAL: "complex",
                    DesignStrategy.ADAPTIVE: "medium",
                }.get(decision.design_strategy, "medium")
                topology = self._architect.design_architecture(feed, focus, complexity)

                event_bridge.emit_architecture_designed(
                    network_id="pending",
                    strategy=decision.design_strategy.value,
                    total_parameters=topology.total_parameters,
                    focus=focus.value,
                    reasoning=decision.reasoning,
                )

                network = self._engine.build_network(
                    topology, feed, decision, focus,
                    emit_fn=event_bridge.emit_construction_event,
                )
                with self._networks_lock:
                    self._networks[network.id] = network

                self._registry.register(
                    network,
                    save_fn=self._engine.save_model,
                )
                self._registry.promote(focus.value, self._registry.get_versions(focus.value)[-1].version)

                event_bridge.emit_network_ready(
                    network.id, network.name, focus.value,
                    network.performance.accuracy,
                )

                if focus == HemisphereFocus.CUSTOM:
                    acc = network.performance.accuracy
                    for df in self._dynamic_focuses.values():
                        if not df.deprecated:
                            df.impact_score = max(df.impact_score, acc)

                logger.info(
                    "Hemisphere %s network built: %s (%.1f%% accuracy, %d params)",
                    focus.value, network.name,
                    network.performance.accuracy * 100,
                    topology.total_parameters,
                )
            except Exception:
                logger.exception("Failed to construct %s hemisphere network", focus.value)
                event_bridge.emit_performance_warning(
                    "construction", f"Failed to build {focus.value} hemisphere",
                    "high",
                )
            finally:
                self._build_lock.release()

        t = threading.Thread(target=_build, daemon=True, name=f"hemi-build-{focus.value}")
        t.start()

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def _evolve_focus(
        self,
        feed: HemisphereDataFeed,
        focus: HemisphereFocus,
        focus_nets: list[NetworkArchitecture],
    ) -> None:
        evo = self._evolution_engines[focus.value]
        topology = evo.evolve(focus_nets, feed, focus)
        if topology is None:
            return

        attempt_id = evo.attempts[-1].id if evo.attempts else None

        if not self._build_lock.acquire(blocking=False):
            logger.debug("Evolution skipped for %s — build lock held", focus.value)
            return

        at_capacity = len(focus_nets) >= MAX_NETWORKS_PER_FOCUS
        weakest = min(focus_nets, key=lambda n: n.performance.accuracy) if at_capacity else None

        def _build_evolved() -> None:
            try:
                decision = self._architect.analyze_consciousness_data(feed)
                network = self._engine.build_network(
                    topology, feed, decision, focus,
                    emit_fn=event_bridge.emit_construction_event,
                )

                if attempt_id:
                    kept = network.performance.accuracy > weakest.performance.accuracy if weakest else True
                    lessons = (
                        f"accuracy={network.performance.accuracy:.1%}",
                        f"params={topology.total_parameters}",
                        "kept" if kept else "discarded",
                    )
                    evo.record_outcome(attempt_id, network.performance, lessons)

                if weakest and network.performance.accuracy > weakest.performance.accuracy:
                    with self._networks_lock:
                        self._networks.pop(weakest.id, None)
                    logger.info(
                        "Replaced weakest %s net %s (%.1f%%) with evolved %s (%.1f%%)",
                        focus.value, weakest.name,
                        weakest.performance.accuracy * 100,
                        network.name, network.performance.accuracy * 100,
                    )
                elif weakest:
                    logger.info(
                        "Evolved %s net %s (%.1f%%) not better than weakest (%.1f%%), discarding",
                        focus.value, network.name,
                        network.performance.accuracy * 100,
                        weakest.performance.accuracy * 100,
                    )
                    return

                with self._networks_lock:
                    self._networks[network.id] = network
                self._registry.register(network, save_fn=self._engine.save_model)

                parent_names = [n.name for n in focus_nets[:2]]
                event_bridge.emit_evolution_complete(
                    evo.generation, parent_names, focus.value,
                    topology.total_parameters,
                )

                active = self._registry.get_active(focus.value)
                if active is None or network.performance.accuracy > active.accuracy:
                    versions = self._registry.get_versions(focus.value)
                    if versions:
                        self._registry.promote(focus.value, versions[-1].version)

                logger.info(
                    "Evolved %s hemisphere gen %d: %s (%.1f%%, %d params)",
                    focus.value, evo.generation, network.name,
                    network.performance.accuracy * 100,
                    topology.total_parameters,
                )
            except Exception:
                logger.exception("Evolution build failed for %s", focus.value)
            finally:
                self._build_lock.release()

        t = threading.Thread(target=_build_evolved, daemon=True, name=f"hemi-evo-{focus.value}")
        t.start()

    # ------------------------------------------------------------------
    # Gap-driven construction
    # ------------------------------------------------------------------

    def _construct_from_gap(self, feed: HemisphereDataFeed, gap: CognitiveGap) -> None:
        """Build a purpose-driven NN from a cognitive gap detection."""
        focus_name = gap.suggested_focus
        df = DynamicFocus(
            name=focus_name,
            input_features=gap.proposed_input_features,
            output_target=gap.proposed_output,
            source_dimension=gap.dimension,
            gap_severity=gap.severity,
            sunset_deadline=_time.time() + SUNSET_CYCLES * 120,  # ~20 min at 120s cycles
        )
        self._dynamic_focuses[focus_name] = df

        logger.info("Gap-driven NN construction: %s (severity=%.2f, dimension=%s)",
                     focus_name, gap.severity, gap.dimension)
        self._construct_network(feed, HemisphereFocus.CUSTOM)

    def _has_network_for_gap(self, gap: CognitiveGap) -> bool:
        """Check if a NN already addresses this cognitive gap."""
        for df in self._dynamic_focuses.values():
            if df.source_dimension == gap.dimension and not df.deprecated:
                return True
        return False

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _feed_gap_detector(self, engine_state: dict) -> None:
        """Derive cognitive dimension scores from engine state and feed to gap detector."""
        try:
            cs = engine_state.get("consciousness", {})
            analytics = cs.get("analytics", {})
            health = analytics.get("health", {})
            components = health.get("components", {})

            confidence = cs.get("confidence_avg", 0.5)
            reasoning = cs.get("reasoning_quality", 0.5)
            awareness = cs.get("awareness_level", 0.5)

            self._gap_detector.update_dimension_score("response_quality", confidence)
            self._gap_detector.update_dimension_score("context_awareness", min(1.0, awareness))
            self._gap_detector.update_dimension_score("mood_prediction", components.get("personality_health", 0.5))
            self._gap_detector.update_dimension_score("trait_consistency", components.get("personality_health", 0.5))

            mem_health = components.get("memory_health", 0.5)
            self._gap_detector.update_dimension_score("memory_recall", mem_health)

            si = engine_state.get("self_improve", {})
            total = si.get("total_improvements", 0) + si.get("total_failures", 0)
            si_score = si.get("total_improvements", 0) / max(1, total) if total > 0 else 0.5
            wr = si.get("win_rate") or {}
            spr = float(wr.get("sandbox_pass_rate", 0.5))
            self._gap_detector.update_metric("self_improvement", "patch_success_rate", spr)
            self._gap_detector.update_metric("self_improvement", "lint_pass_rate", spr)
            blended = 0.55 * spr + 0.45 * si_score if total > 0 else spr
            self._gap_detector.update_dimension_score("self_improvement", blended)

            # Perceptual dimensions — use public APIs
            try:
                from perception.identity_fusion import _active_instance as _id_fusion
                if _id_fusion:
                    status = _id_fusion.get_status()
                    voice = status.get("voice_signal", {})
                    face = status.get("face_signal", {})
                    self._gap_detector.update_dimension_score(
                        "recognition_confidence",
                        max(voice.get("confidence", 0.0), face.get("confidence", 0.0)),
                    )
                else:
                    self._gap_detector.update_dimension_score("recognition_confidence", 0.5)
            except Exception:
                self._gap_detector.update_dimension_score("recognition_confidence", 0.5)

            try:
                from perception.emotion import emotion_classifier
                if emotion_classifier:
                    emo_stats = emotion_classifier.get_stats() if hasattr(emotion_classifier, "get_stats") else {}
                    emo_confidence = emo_stats.get("avg_confidence", 0.0)
                    emo_healthy = emo_stats.get("model_healthy", False)
                    if emo_healthy and emo_confidence > 0:
                        self._gap_detector.update_dimension_score("emotion_accuracy", emo_confidence)
                        latency = emo_stats.get("avg_inference_ms", 0.0)
                        latency_score = max(0.0, 1.0 - latency / 500.0) if latency > 0 else 0.5
                        self._gap_detector.update_dimension_score("perception_latency", latency_score)
                    else:
                        self._gap_detector.update_dimension_score("emotion_accuracy", 0.5)
                        self._gap_detector.update_dimension_score("perception_latency", 0.5)
                else:
                    self._gap_detector.update_dimension_score("emotion_accuracy", 0.5)
                    self._gap_detector.update_dimension_score("perception_latency", 0.5)
            except Exception:
                self._gap_detector.update_dimension_score("emotion_accuracy", 0.5)
                self._gap_detector.update_dimension_score("perception_latency", 0.5)
        except Exception:
            pass

    def _prune_networks(self) -> None:
        """Remove networks past sunset deadline or when over cap."""
        now = _time.time()
        pruned: list[str] = []

        # Prune dynamic focus NNs past their sunset deadline with no impact
        for name, df in list(self._dynamic_focuses.items()):
            if df.deprecated:
                continue
            if now > df.sunset_deadline and df.impact_score < 0.1:
                df.deprecated = True
                pruned.append(name)
                logger.info("Pruned NN focus '%s' (sunset, impact=%.2f)", name, df.impact_score)

        # Retire probationary specialists with low utility (> 24h old, impact < 0.1)
        self._retire_low_utility_probationary(now, pruned)

        # Enforce MAX_TOTAL_NETWORKS by removing weakest
        while self._total_network_count() > MAX_TOTAL_NETWORKS:
            weakest = self._find_weakest_network()
            if weakest:
                with self._networks_lock:
                    self._networks.pop(weakest.id, None)
                self._engine.remove_model(weakest.id)
                pruned.append(weakest.name)
                logger.info("Pruned weakest network '%s' (accuracy=%.2f%%) to stay under cap",
                            weakest.name, weakest.performance.accuracy * 100)
            else:
                break

        if pruned:
            event_bridge.emit_performance_warning(
                "pruning", f"Pruned {len(pruned)} network(s): {', '.join(pruned[:3])}", "info",
            )

    def _retire_low_utility_probationary(
        self, now: float, pruned: list[str],
    ) -> None:
        """Auto-retire probationary specialists that have not proven their value."""
        from hemisphere.types import (
            SpecialistLifecycleStage, MATRIX_ELIGIBLE_FOCUSES,
        )
        _MIN_AGE_S = 86400.0  # 24 hours before eligible for retirement
        _MIN_IMPACT = 0.1

        with self._networks_lock:
            candidates = [
                n for n in self._networks.values()
                if n.focus in MATRIX_ELIGIBLE_FOCUSES
                and n.specialist_lifecycle in (
                    SpecialistLifecycleStage.CANDIDATE_BIRTH,
                    SpecialistLifecycleStage.PROBATIONARY_TRAINING,
                )
                and (now - n.created_at) > _MIN_AGE_S
                and n.specialist_impact_score < _MIN_IMPACT
            ]

        for net in candidates:
            net.specialist_lifecycle = SpecialistLifecycleStage.RETIRED
            net.status = NetworkStatus.DEPRECATED
            with self._networks_lock:
                self._networks.pop(net.id, None)
            self._engine.remove_model(net.id)
            pruned.append(net.name)
            logger.info(
                "Retired probationary specialist '%s' (focus=%s, impact=%.3f, age=%.0fh)",
                net.name, net.focus.value, net.specialist_impact_score,
                (now - net.created_at) / 3600,
            )

    def count_probationary_specialists(self) -> int:
        """Count active probationary specialists (not retired/promoted)."""
        from hemisphere.types import (
            SpecialistLifecycleStage, MATRIX_ELIGIBLE_FOCUSES,
        )
        with self._networks_lock:
            return sum(
                1 for n in self._networks.values()
                if n.focus in MATRIX_ELIGIBLE_FOCUSES
                and n.specialist_lifecycle is not None
                and n.specialist_lifecycle not in (
                    SpecialistLifecycleStage.RETIRED,
                    SpecialistLifecycleStage.PROMOTED,
                )
            )

    def create_probationary_specialist(
        self,
        focus: HemisphereFocus,
        job_id: str,
        name: str = "",
    ) -> NetworkArchitecture | None:
        """Create a probationary specialist NN for a Matrix Protocol job.

        Only perceptual/transfer-worthy focuses may spawn specialists.
        Enforces the hard cap on probationary specialists.
        Returns the created architecture, or None if capped/ineligible.
        """
        from hemisphere.types import (
            SpecialistLifecycleStage, MATRIX_ELIGIBLE_FOCUSES,
            MAX_PROBATIONARY_SPECIALISTS,
        )

        if focus not in MATRIX_ELIGIBLE_FOCUSES:
            logger.warning(
                "Focus %s is not eligible for Matrix specialist birth", focus.value,
            )
            return None

        current_count = self.count_probationary_specialists()
        if current_count >= MAX_PROBATIONARY_SPECIALISTS:
            logger.info(
                "Probationary specialist cap reached (%d/%d) — cannot create for %s",
                current_count, MAX_PROBATIONARY_SPECIALISTS, focus.value,
            )
            return None

        net_name = name or f"matrix_{focus.value}"
        arch = self._construct_for_focus(focus, net_name)
        if arch is not None:
            arch.specialist_lifecycle = SpecialistLifecycleStage.CANDIDATE_BIRTH
            arch.specialist_job_id = job_id
            logger.info(
                "Created probationary specialist '%s' (focus=%s, job=%s)",
                net_name, focus.value, job_id,
            )
        return arch

    def _check_specialist_promotions(self) -> None:
        """Advance specialist NNs through their lifecycle based on evidence.

        Promotion ladder:
          candidate_birth -> probationary_training (has started training)
          probationary_training -> verified_probationary (protocol passed)
          verified_probationary -> broadcast_eligible (impact > 0.3)
          broadcast_eligible -> promoted (consistently in broadcast slot for 10+ cycles)
        """
        from hemisphere.types import (
            SpecialistLifecycleStage, MATRIX_ELIGIBLE_FOCUSES,
        )

        broadcast_names = {s["name"] for s in self._broadcast_slots}

        with self._networks_lock:
            specialists = [
                n for n in self._networks.values()
                if n.focus in MATRIX_ELIGIBLE_FOCUSES
                and n.specialist_lifecycle is not None
                and n.specialist_lifecycle not in (
                    SpecialistLifecycleStage.RETIRED,
                    SpecialistLifecycleStage.PROMOTED,
                )
            ]

        for net in specialists:
            stage = net.specialist_lifecycle
            if stage == SpecialistLifecycleStage.CANDIDATE_BIRTH:
                if net.training_progress.current_epoch > 0:
                    net.specialist_lifecycle = SpecialistLifecycleStage.PROBATIONARY_TRAINING
                    logger.info("Specialist '%s' advanced to probationary_training", net.name)

            elif stage == SpecialistLifecycleStage.PROBATIONARY_TRAINING:
                if net.performance.accuracy > 0.5:
                    net.specialist_lifecycle = SpecialistLifecycleStage.VERIFIED_PROBATIONARY
                    # Stamp the verification timestamp at the transition.
                    # This is the only site that writes `specialist_verification_ts`;
                    # downstream readers (_compute_impact_score recency decay,
                    # _check_expansion_trigger stability window) depend on it.
                    # Re-entering VERIFIED_PROBATIONARY (e.g. after retirement
                    # and re-training) is a legitimate re-stamp: the stability
                    # window resets with the new verification event.
                    net.specialist_verification_ts = _time.time()
                    logger.info(
                        "Specialist '%s' advanced to verified_probationary (ts=%.0f)",
                        net.name, net.specialist_verification_ts,
                    )

            elif stage == SpecialistLifecycleStage.VERIFIED_PROBATIONARY:
                if net.specialist_impact_score > 0.3:
                    net.specialist_lifecycle = SpecialistLifecycleStage.BROADCAST_ELIGIBLE
                    logger.info("Specialist '%s' advanced to broadcast_eligible (impact=%.2f)",
                                net.name, net.specialist_impact_score)

            elif stage == SpecialistLifecycleStage.BROADCAST_ELIGIBLE:
                if net.focus.value in broadcast_names:
                    slot_dwell = 0
                    for s in self._broadcast_slots:
                        if s["name"] == net.focus.value:
                            slot_dwell = s.get("dwell", 0)
                            break
                    if slot_dwell >= 10:
                        net.specialist_lifecycle = SpecialistLifecycleStage.PROMOTED
                        logger.info(
                            "Specialist '%s' PROMOTED (dwell=%d, impact=%.2f)",
                            net.name, slot_dwell, net.specialist_impact_score,
                        )

    def _construct_for_focus(
        self, focus: HemisphereFocus, name: str,
    ) -> NetworkArchitecture | None:
        """Construct a minimal network for a given focus. Returns arch or None."""
        try:
            from hemisphere.types import (
                NetworkTopology, LayerDefinition, PerformanceMetrics,
                TrainingProgress, DesignStrategy,
            )
            topology = NetworkTopology(
                input_size=16,
                layers=(
                    LayerDefinition(id="h1", layer_type="hidden", node_count=32,
                                    activation="relu", dropout=0.1),
                    LayerDefinition(id="out", layer_type="output", node_count=8,
                                    activation="tanh"),
                ),
                output_size=8,
                total_parameters=32 * 16 + 32 + 8 * 32 + 8,
                activation_functions=("relu", "tanh"),
            )
            arch = NetworkArchitecture(
                id=f"{name}_{int(_time.time())}",
                name=name,
                focus=focus,
                topology=topology,
                performance=PerformanceMetrics(),
                training_progress=TrainingProgress(),
                status=NetworkStatus.DESIGNING,
            )
            with self._networks_lock:
                self._networks[arch.id] = arch
            return arch
        except Exception:
            logger.exception("Failed to construct specialist for %s", focus.value)
            return None

    def _total_network_count(self) -> int:
        with self._networks_lock:
            return len([n for n in self._networks.values()
                         if n.status in (NetworkStatus.READY, NetworkStatus.ACTIVE, NetworkStatus.TRAINING)])

    def _find_weakest_network(self) -> NetworkArchitecture | None:
        with self._networks_lock:
            active = [n for n in self._networks.values()
                      if n.status in (NetworkStatus.READY, NetworkStatus.ACTIVE)]
        if not active:
            return None
        return min(active, key=lambda n: n.performance.accuracy)

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def _check_migration(self, feed: HemisphereDataFeed) -> None:
        for focus in HemisphereFocus:
            best = self._get_best_network(focus)
            if best is None:
                continue
            readiness = self._migration.analyze_readiness(best, feed)
            event_bridge.emit_migration_decision(
                best.id, readiness.should_migrate,
                readiness.reasoning, readiness.confidence,
            )

    # ------------------------------------------------------------------
    # Distillation (Tier-1 student training)
    # ------------------------------------------------------------------

    def _get_distillation_collector(self) -> Any:
        if self._distillation_collector is None:
            try:
                from hemisphere.distillation import distillation_collector
                self._distillation_collector = distillation_collector
            except Exception:
                return None
        return self._distillation_collector

    @staticmethod
    def _clone_state_dict(state_dict: Any) -> Any:
        """Deep-copy state dict so a failed retrain can restore exact weights."""
        if not isinstance(state_dict, dict):
            return copy.deepcopy(state_dict)
        cloned: dict[Any, Any] = {}
        for key, value in state_dict.items():
            try:
                if hasattr(value, "detach") and hasattr(value, "clone"):
                    cloned[key] = value.detach().clone()
                else:
                    cloned[key] = copy.deepcopy(value)
            except Exception:
                cloned[key] = copy.deepcopy(value)
        return cloned

    def _snapshot_model_state(self, network_id: str) -> Any | None:
        model = getattr(self._engine, "_active_models", {}).get(network_id)
        if model is None or not hasattr(model, "state_dict"):
            return None
        try:
            return self._clone_state_dict(model.state_dict())
        except Exception:
            logger.debug("Failed to snapshot model state for %s", network_id, exc_info=True)
            return None

    def _restore_model_state(self, network_id: str, snapshot: Any | None) -> bool:
        if snapshot is None:
            return False
        model = getattr(self._engine, "_active_models", {}).get(network_id)
        if model is None or not hasattr(model, "load_state_dict"):
            return False
        try:
            model.load_state_dict(snapshot)
            return True
        except Exception:
            logger.debug("Failed to restore model state for %s", network_id, exc_info=True)
            return False

    def _restore_distillation_baseline(
        self,
        network: NetworkArchitecture,
        old_accuracy: float,
        old_loss: float,
        model_snapshot: Any | None,
    ) -> bool:
        from hemisphere.types import PerformanceMetrics as _PM
        from dataclasses import asdict

        rollback = asdict(network.performance)
        rollback["accuracy"] = old_accuracy
        rollback["loss"] = old_loss
        network.performance = _PM(**rollback)
        return self._restore_model_state(network.id, model_snapshot)

    @staticmethod
    def _compute_regression_cooldown_s(strikes: int) -> float:
        """Escalate cooldown when a Tier-1 retrain repeatedly regresses."""
        level = max(1, int(strikes))
        cooldown_s = DISTILLATION_REGRESSION_COOLDOWN_S * (
            DISTILLATION_REGRESSION_COOLDOWN_BACKOFF ** (level - 1)
        )
        return min(DISTILLATION_REGRESSION_COOLDOWN_MAX_S, cooldown_s)

    def _maybe_run_distillation(self, engine_state: dict[str, Any]) -> None:
        """Run distillation cycle if enough time has elapsed."""
        now = _time.time()
        mode = engine_state.get("mode", "passive")
        cadence = DISTILLATION_CADENCE_DEEP_S if mode in ("deep_learning", "dreaming") else DISTILLATION_CADENCE_S

        gestation = engine_state.get("gestation", {})
        gestation_phase = gestation.get("phase", -1)
        if gestation_phase in (0, 1):
            cadence = 60.0
        elif gestation_phase == 2:
            cadence = 45.0

        if now - self._last_distillation_time < cadence:
            return
        self._last_distillation_time = now

        if mode in ("dreaming", "sleep", "deep_learning"):
            collector = self._get_distillation_collector()
            if collector is not None:
                collector.review_quarantine()

        self._run_distillation_cycle()

    def _run_distillation_cycle(self) -> None:
        """Train Tier-1 distilled specialists from teacher signals."""
        collector = self._get_distillation_collector()
        if collector is None:
            return

        now = _time.time()
        trained = 0
        cycle_start_retrain_time: dict[str, float] = dict(self._tier1_last_retrain_time)
        for focus_name, config in DISTILLATION_CONFIGS.items():
            try:
                if focus_name in self._tier1_disabled:
                    continue

                cooldown_until = self._tier1_regression_cooldown_until.get(focus_name, 0.0)
                if cooldown_until > now:
                    last_log = self._tier1_cooldown_last_log.get(focus_name, 0.0)
                    if now - last_log >= DISTILLATION_COOLDOWN_LOG_INTERVAL_S:
                        self._tier1_cooldown_last_log[focus_name] = now
                        remaining = int(max(0.0, cooldown_until - now))
                        regressions = self._tier1_regression_counts.get(focus_name, 0)
                        logger.info(
                            "Distillation %s cooldown active (%ds remaining, regressions=%d)",
                            focus_name, remaining, regressions,
                        )
                    continue
                if cooldown_until > 0.0:
                    self._tier1_regression_cooldown_until.pop(focus_name, None)
                    self._tier1_cooldown_last_log.pop(focus_name, None)
                    self._tier1_regression_counts[focus_name] = 0

                if config.depends_on:
                    all_deps_ready = all(
                        self._get_best_network(HemisphereFocus(d)) is not None
                        for d in config.depends_on
                    )
                    if not all_deps_ready:
                        continue
                    # Use a cycle-start snapshot so a dependency retrained
                    # earlier in this same loop iteration does not block its
                    # dependents (ordering artefact, not a real freshness
                    # concern). Real cross-cycle protection is preserved:
                    # stamps written during this cycle take effect on the
                    # NEXT cycle.
                    dep_recently_retrained = any(
                        now - cycle_start_retrain_time.get(d, 0.0)
                        < DISTILLATION_DEP_RETRAIN_WINDOW_S
                        for d in config.depends_on
                    )
                    if dep_recently_retrained:
                        logger.debug(
                            "Distillation %s skipped: dependency recently retrained",
                            focus_name,
                        )
                        continue

                if config.student_type == "approximator":
                    source = getattr(config, "feature_source", "audio_features")
                    if collector.count(source) < config.min_samples:
                        continue
                    if collector.count(config.teacher) < config.min_samples:
                        continue
                elif collector.count(config.teacher) < config.min_samples:
                    continue

                tensors = prepare_distillation_tensors(focus_name, collector, config)
                if tensors is None:
                    continue

                features, labels, weights = tensors
                focus = HemisphereFocus(focus_name)
                existing = self._get_best_network(focus)

                if existing is not None:
                    model_input_dim = getattr(existing.topology, "input_size", None)
                    if model_input_dim is not None and model_input_dim != config.input_dim:
                        logger.info(
                            "Distillation %s input_dim changed (%d→%d), rebuilding",
                            focus.value, model_input_dim, config.input_dim,
                        )
                        with self._networks_lock:
                            self._networks.pop(existing.id, None)
                        self._engine.remove_model(existing.id)
                        self._build_distillation_network(focus, config, features, labels, weights)
                        self._tier1_regression_counts[focus.value] = 0
                        self._tier1_regression_cooldown_strikes.pop(focus.value, None)
                        self._tier1_last_retrain_time[focus.value] = now
                        trained += 1
                        continue

                    old_accuracy = existing.performance.accuracy
                    old_loss = existing.performance.loss
                    model_snapshot = self._snapshot_model_state(existing.id)

                    loss = self._engine.train_distillation(
                        existing, features, labels, weights,
                        loss_name=config.loss, epochs=20,
                    )
                    new_accuracy = existing.performance.accuracy
                    n_samples = features.shape[0] if hasattr(features, 'shape') else len(features)

                    if new_accuracy < TIER1_MIN_ACCURACY and n_samples >= TIER1_MIN_SAMPLES_FOR_ACCURACY_FLOOR:
                        restored_weights = self._restore_distillation_baseline(
                            existing,
                            old_accuracy,
                            old_loss,
                            model_snapshot,
                        )
                        logger.warning(
                            "Tier-1 specialist '%s' retrain below accuracy floor: "
                            "%.3f%% → rolled back to %.3f%% (weights_restored=%s)",
                            focus.value, new_accuracy * 100, old_accuracy * 100, restored_weights,
                        )
                        continue
                    elif new_accuracy < TIER1_MIN_ACCURACY:
                        restored_weights = self._restore_distillation_baseline(
                            existing,
                            old_accuracy,
                            old_loss,
                            model_snapshot,
                        )
                        logger.info(
                            "Tier-1 specialist '%s' below accuracy floor (%.3f%%) but only "
                            "%d samples — baseline restored (need %d, weights_restored=%s)",
                            focus.value, new_accuracy * 100, n_samples,
                            TIER1_MIN_SAMPLES_FOR_ACCURACY_FLOOR, restored_weights,
                        )
                        continue

                    self._tier1_failure_counts[focus.value] = 0

                    if (
                        old_accuracy > 0
                        and n_samples >= DISTILLATION_REGRESSION_MIN_SAMPLES
                        and new_accuracy < old_accuracy - DISTILLATION_REGRESSION_DELTA
                    ):
                        restored_weights = self._restore_distillation_baseline(
                            existing,
                            old_accuracy,
                            old_loss,
                            model_snapshot,
                        )
                        regressions = self._tier1_regression_counts.get(focus.value, 0) + 1
                        self._tier1_regression_counts[focus.value] = regressions
                        if regressions >= DISTILLATION_MAX_CONSECUTIVE_REGRESSIONS:
                            strikes = self._tier1_regression_cooldown_strikes.get(focus.value, 0) + 1
                            self._tier1_regression_cooldown_strikes[focus.value] = strikes
                            cooldown_s = self._compute_regression_cooldown_s(strikes)
                            cooldown_until = now + cooldown_s
                            self._tier1_regression_cooldown_until[focus.value] = cooldown_until
                            self._tier1_cooldown_last_log[focus.value] = now
                            logger.warning(
                                "Distillation retrain %s regressed (%.3f→%.3f), rolled back "
                                "(weights_restored=%s, regressions=%d/%d, cooldown=%ds, strikes=%d)",
                                focus.value, old_accuracy, new_accuracy, restored_weights,
                                regressions, DISTILLATION_MAX_CONSECUTIVE_REGRESSIONS,
                                int(cooldown_s), strikes,
                            )
                        else:
                            logger.warning(
                                "Distillation retrain %s regressed (%.3f→%.3f), rolled back "
                                "(weights_restored=%s, regressions=%d/%d)",
                                focus.value, old_accuracy, new_accuracy, restored_weights,
                                regressions, DISTILLATION_MAX_CONSECUTIVE_REGRESSIONS,
                            )
                        continue
                    else:
                        self._tier1_regression_counts[focus.value] = 0
                        self._tier1_regression_cooldown_strikes.pop(focus.value, None)
                        self._tier1_regression_cooldown_until.pop(focus.value, None)
                        self._tier1_cooldown_last_log.pop(focus.value, None)
                        self._tier1_last_retrain_time[focus.value] = now
                        if config.student_type == "compressor" and config.bottleneck_dim > 0:
                            n_enc_layers = 3
                            enc_id = self._engine.extract_encoder(existing.id, n_enc_layers)
                            if enc_id:
                                self._distillation_encoder_ids[focus.value] = enc_id
                        trained += 1
                        logger.info("Distillation retrain %s: loss=%.4f, acc=%.3f→%.3f, samples=%d",
                                    focus.value, loss, old_accuracy, new_accuracy, features.shape[0])
                else:
                    self._build_distillation_network(focus, config, features, labels, weights)
                    self._tier1_regression_counts[focus.value] = 0
                    self._tier1_regression_cooldown_strikes.pop(focus.value, None)
                    self._tier1_regression_cooldown_until.pop(focus.value, None)
                    self._tier1_cooldown_last_log.pop(focus.value, None)
                    self._tier1_last_retrain_time[focus.value] = now
                    trained += 1
            except Exception:
                logger.warning("Distillation cycle failed for %s", focus_name, exc_info=True)

        if trained > 0:
            try:
                from consciousness.events import event_bus, HEMISPHERE_DISTILLATION_STATS
                event_bus.emit(HEMISPHERE_DISTILLATION_STATS, trained=trained, stats=collector.get_stats())
            except Exception:
                pass

    def _build_distillation_network(
        self,
        focus: HemisphereFocus,
        config: DistillationConfig,
        features: Any,
        labels: Any,
        weights: Any,
    ) -> None:
        """Build and train a new Tier-1 distillation network (synchronous — tiny models)."""
        try:
            import uuid as _u
            from hemisphere.types import (
                DesignStrategy, NetworkStatus, PerformanceMetrics, TrainingProgress,
            )

            topology = self._architect.design_distillation_topology(config)
            nid = f"distill_{_u.uuid4().hex[:10]}"

            event_bridge.emit_architecture_designed(
                network_id=nid, strategy="adaptive",
                total_parameters=topology.total_parameters,
                focus=focus.value,
                reasoning=f"Tier-1 {config.student_type} for {config.teacher}",
            )

            model = self._engine.build_model(topology)
            self._engine._active_models[nid] = model

            name = f"Distill-{config.student_type.title()}-{focus.value}"
            network = NetworkArchitecture(
                id=nid,
                name=name,
                focus=focus,
                topology=topology,
                performance=PerformanceMetrics(),
                training_progress=TrainingProgress(),
                is_active=False,
                status=NetworkStatus.READY,
                design_reasoning=f"Tier-1 distillation: {config.student_type} for {config.teacher}",
            )
            network._model_ref = model

            loss = self._engine.train_distillation(
                network, features, labels, weights,
                loss_name=config.loss, epochs=30,
            )

            accuracy = network.performance.accuracy
            n_samples = features.shape[0] if hasattr(features, 'shape') else len(features)
            if accuracy < TIER1_MIN_ACCURACY and n_samples >= TIER1_MIN_SAMPLES_FOR_ACCURACY_FLOOR:
                self._tier1_failure_counts[focus.value] = \
                    self._tier1_failure_counts.get(focus.value, 0) + 1
                failures = self._tier1_failure_counts[focus.value]
                if failures >= TIER1_MAX_CONSECUTIVE_FAILURES:
                    self._tier1_disabled.add(focus.value)
                    logger.warning(
                        "Tier-1 specialist '%s' disabled for session: %d consecutive "
                        "builds below %.0f%% accuracy floor (last=%.3f%%)",
                        focus.value, failures, TIER1_MIN_ACCURACY * 100,
                        accuracy * 100,
                    )
                else:
                    logger.warning(
                        "Tier-1 specialist '%s' below accuracy floor: %.3f%% "
                        "(failure %d/%d, not registered)",
                        focus.value, accuracy * 100, failures,
                        TIER1_MAX_CONSECUTIVE_FAILURES,
                    )
                self._engine.remove_model(network.id)
                return
            elif accuracy < TIER1_MIN_ACCURACY:
                logger.info(
                    "Tier-1 specialist '%s' below accuracy floor (%.3f%%) but only "
                    "%d samples — registering anyway (need %d for penalty)",
                    focus.value, accuracy * 100, n_samples,
                    TIER1_MIN_SAMPLES_FOR_ACCURACY_FLOOR,
                )
                # Don't count as failure — let it register and improve with more data

            self._tier1_failure_counts[focus.value] = 0

            with self._networks_lock:
                self._networks[network.id] = network
            self._registry.register(network, save_fn=self._engine.save_model)
            versions = self._registry.get_versions(focus.value)
            if versions:
                self._registry.promote(focus.value, versions[-1].version)

            if config.student_type == "compressor" and config.bottleneck_dim > 0:
                n_enc_layers = 3
                enc_id = self._engine.extract_encoder(network.id, n_enc_layers)
                if enc_id:
                    self._distillation_encoder_ids[focus.value] = enc_id

            event_bridge.emit_network_ready(
                network.id, network.name, focus.value,
                network.performance.accuracy,
            )
            logger.info("Distillation %s built: %s (loss=%.4f, %d params)",
                        focus.value, network.name, loss, topology.total_parameters)
        except Exception:
            logger.exception("Distillation build failed for %s", focus.value)

    def _remove_distillation_model(
        self, focus: HemisphereFocus, network: NetworkArchitecture,
    ) -> None:
        """Remove a broken distillation model from all registries and bindings."""
        with self._networks_lock:
            self._networks.pop(network.id, None)
        self._engine.remove_model(network.id)
        self._registry.deactivate(focus.value, network.id)
        self._distillation_encoder_ids.pop(focus.value, None)

    def _has_ready_network(self, focus: HemisphereFocus) -> bool:
        return self._get_best_network(focus) is not None

    # ------------------------------------------------------------------
    # Hemisphere signals — Global Broadcast Slots (for StateEncoder)
    # ------------------------------------------------------------------

    def expand_broadcast_slots(self) -> None:
        """Expand from 4 to EXPANDED_SLOT_COUNT broadcast slots.

        Called when the M6 expansion trigger fires and shadow evaluation
        starts. New slots are initialized empty so they don't interfere
        with live encoding (which still reads only slot_0..slot_3).
        """
        if self._num_broadcast_slots >= EXPANDED_SLOT_COUNT:
            return
        while len(self._broadcast_slots) < EXPANDED_SLOT_COUNT:
            self._broadcast_slots.append(
                {"name": "", "value": 0.0, "score": 0.0, "dwell": 0}
            )
        self._num_broadcast_slots = EXPANDED_SLOT_COUNT
        logger.info(
            "Broadcast slots expanded: %d -> %d",
            4, self._num_broadcast_slots,
        )

    def contract_broadcast_slots(self) -> None:
        """Revert to 4 broadcast slots (M6 rollback)."""
        if self._num_broadcast_slots <= 4:
            return
        self._broadcast_slots = self._broadcast_slots[:4]
        self._num_broadcast_slots = 4
        logger.info("Broadcast slots contracted back to 4")

    def get_hemisphere_signals(self) -> dict[str, float]:
        """Global Broadcast: compute all signals, broadcast top-N with hysteresis.

        All hemisphere signals (Tier-1 distilled + Tier-2 standard + Matrix
        specialists) compete for _num_broadcast_slots broadcast slots.
        The live StateEncoder reads slot_0..slot_3; the shadow encoder
        (during M6 evaluation) additionally reads slot_4..slot_5.
        """
        all_candidates: list[tuple[str, float, float]] = []
        tier1_extras = self._collect_tier1_features()

        for focus in HemisphereFocus:
            if focus in (HemisphereFocus.CUSTOM, HemisphereFocus.SYSTEM_UPGRADES):
                continue
            if focus in _SHADOW_ONLY_TIER1_FOCUSES:
                continue
            best = self._get_best_network(focus)
            if best is None:
                continue
            try:
                if focus in _TIER1_FOCUSES:
                    value = self._compute_tier1_signal_value(best, focus)
                else:
                    value = self._compute_signal_value(best, extra_features=tier1_extras)
                score = self._compute_signal_score(best, focus)
                all_candidates.append((focus.value, value, score))
            except Exception:
                pass

        all_candidates.sort(key=lambda x: x[2], reverse=True)

        num_slots = self._num_broadcast_slots
        for i in range(num_slots):
            if i >= len(all_candidates):
                break
            incumbent = self._broadcast_slots[i]
            newcomer_name, newcomer_val, newcomer_score = all_candidates[i]

            if incumbent["score"] == 0.0 and incumbent["dwell"] == 0:
                self._broadcast_slots[i] = {
                    "name": newcomer_name, "value": newcomer_val,
                    "score": newcomer_score, "dwell": 0,
                }
            elif incumbent["name"] == newcomer_name:
                incumbent["value"] = newcomer_val
                incumbent["score"] = newcomer_score
                incumbent["dwell"] += 1
            elif (newcomer_score > incumbent["score"] * SLOT_SWAP_THRESHOLD
                  and incumbent["dwell"] >= SLOT_MIN_DWELL_CYCLES):
                self._broadcast_slots[i] = {
                    "name": newcomer_name, "value": newcomer_val,
                    "score": newcomer_score, "dwell": 0,
                }

        result: dict[str, float] = {}
        assignments: dict[str, str] = {}
        for i in range(num_slots):
            result[f"slot_{i}"] = self._broadcast_slots[i]["value"]
            assignments[f"slot_{i}"] = self._broadcast_slots[i]["name"]
        result["_assignments"] = assignments  # type: ignore[assignment]
        return result

    def _compute_signal_value(self, network: NetworkArchitecture, extra_features: dict[str, float] | None = None) -> float:
        """Run inference on a network and return its mean output as a signal.

        Tier-2 Matrix Protocol specialists with a registered focus-specific
        encoder bypass the network/accuracy fallback chain entirely and
        return a deterministic real-time scalar in ``[0, 1]`` (P3.6+).
        That contract is required so a Tier-2 specialist at CANDIDATE_BIRTH
        — where ``performance.accuracy`` is 0.0 — does not silently rank
        last in the broadcast-slot competition just because its network
        is untrained.
        """
        focus_specific = self._matrix_focus_signal(network)
        if focus_specific is not None:
            return focus_specific
        try:
            input_vec = self._build_inference_input(network.topology.input_size, extra_features=extra_features)
            output = self._engine.infer(network.id, input_vec)
            if output:
                return float(sum(output) / len(output))
        except Exception:
            pass
        active = self._registry.get_active(network.focus.value)
        return active.accuracy if active else 0.0

    def _matrix_focus_signal(self, network: NetworkArchitecture) -> float | None:
        """Dispatch to a Tier-2 focus-specific signal computer if registered.

        Returns ``None`` for focuses without a registered encoder so the
        caller falls through to the generic inference path. For focuses
        with an encoder, the return value is a clamped ``[0, 1]`` scalar
        derived from real-time perception/memory/mood state.

        This method is the architectural replacement for the
        accuracy-as-proxy fallback for Tier-2 specialists. Adding a new
        Tier-2 focus only requires adding one branch here plus an
        encoder module under ``brain/hemisphere/``.
        """
        if network.focus is HemisphereFocus.POSITIVE_MEMORY:
            try:
                from hemisphere.positive_memory_encoder import (
                    PositiveMemoryEncoder,
                )
                ctx = self._build_positive_memory_context()
                return float(PositiveMemoryEncoder.compute_signal_value(ctx))
            except Exception:
                logger.debug(
                    "positive_memory signal computation failed", exc_info=True,
                )
                return 0.0
        if network.focus is HemisphereFocus.NEGATIVE_MEMORY:
            try:
                from hemisphere.negative_memory_encoder import (
                    NegativeMemoryEncoder,
                )
                ctx = self._build_negative_memory_context()
                return float(NegativeMemoryEncoder.compute_signal_value(ctx))
            except Exception:
                logger.debug(
                    "negative_memory signal computation failed", exc_info=True,
                )
                return 0.0
        if network.focus is HemisphereFocus.SPEAKER_PROFILE:
            try:
                from hemisphere.speaker_profile_encoder import (
                    SpeakerProfileEncoder,
                )
                ctx = self._build_speaker_profile_context()
                return float(SpeakerProfileEncoder.compute_signal_value(ctx))
            except Exception:
                logger.debug(
                    "speaker_profile signal computation failed", exc_info=True,
                )
                return 0.0
        if network.focus is HemisphereFocus.TEMPORAL_PATTERN:
            try:
                from hemisphere.temporal_pattern_encoder import (
                    TemporalPatternEncoder,
                )
                ctx = self._build_temporal_pattern_context()
                return float(TemporalPatternEncoder.compute_signal_value(ctx))
            except Exception:
                logger.debug(
                    "temporal_pattern signal computation failed", exc_info=True,
                )
                return 0.0
        if network.focus is HemisphereFocus.SKILL_TRANSFER:
            try:
                from hemisphere.skill_transfer_encoder import (
                    SkillTransferEncoder,
                )
                ctx = self._build_skill_transfer_context()
                return float(SkillTransferEncoder.compute_signal_value(ctx))
            except Exception:
                logger.debug(
                    "skill_transfer signal computation failed", exc_info=True,
                )
                return 0.0
        return None

    def _build_positive_memory_context(self) -> dict[str, Any]:
        """Gather real-time context for the positive_memory encoder.

        This is the only place that touches live singletons on behalf of
        the encoder. The encoder itself is a pure function of this
        context dict so it stays unit-testable. Every field is best-
        effort and degrades to a default on import / attribute failure;
        the encoder treats missing fields as 0.0 contributions.
        """
        ctx: dict[str, Any] = {
            "recent_memories": [],
            "recent_episodes": [],
            "recent_turn_count": 0,
            "max_turns": 50,
            "memory_density": 0.0,
            "max_memories": 1000,
            "traits": (),
            "mood_positivity": 0.0,
            "emotion_positive_bias": 0.0,
            "contradiction_debt": 0.0,
        }

        # Recent memories + density (same source as _build_inference_input).
        try:
            from memory.storage import memory_storage
            from memory.maintenance import MAX_MEMORIES
            mems = memory_storage.get_all() or []
            ctx["recent_memories"] = mems[-32:]
            ctx["max_memories"] = MAX_MEMORIES
            if MAX_MEMORIES > 0:
                ctx["memory_density"] = min(len(mems) / float(MAX_MEMORIES), 1.0)
        except Exception:
            pass

        # Recent episodes (positive-affect / revisit reinforcement).
        # ``EpisodicMemory`` is not a module-level singleton in this
        # codebase — the live instance is owned by ``main.py`` and
        # threaded into the engine. The orchestrator does not currently
        # carry that handle, so episodes contribute via the recent-turn
        # count surrogate below; Block B degrades cleanly to Block A's
        # signal when episode access is unavailable.

        # Note: ``emotion_positive_bias`` is intentionally left at 0.0
        # for now. The Tier-1 ``emotion_depth`` specialist exposes a
        # reconstruction-fidelity scalar via ``_compute_tier1_signal_value``,
        # not a positive-class probability, so wiring it here would
        # silently inflate the positive_memory signal with accuracy
        # data — exactly the failure mode P3.6 forbids. A proper
        # positive-class hook lands when the emotion classifier exposes
        # a calibrated valence head; until then Block C relies on
        # traits, mood_positivity, and coherence_positive (each of
        # which is real-time inferable on its own).

        return ctx

    def _build_negative_memory_context(self) -> dict[str, Any]:
        """Gather real-time context for the negative_memory encoder.

        Mirror of :meth:`_build_positive_memory_context` for P3.7. The
        encoder itself is a pure function of this context dict so it
        stays unit-testable; the orchestrator is the only place that
        touches live singletons on its behalf.

        Every field is best-effort and degrades to a default on import
        / attribute failure; the encoder treats missing fields as 0.0
        contributions. This keeps the dispatch path safe even if the
        quarantine subsystem is offline (e.g. during boot stabilisation).
        """
        ctx: dict[str, Any] = {
            "recent_memories": [],
            "recent_episodes": [],
            "memory_density": 0.0,
            "max_memories": 1000,
            "tier1_failure_rate": 0.0,
            "quarantine_pressure": 0.0,
            "contradiction_debt": 0.0,
            "low_confidence_retrieval_rate": 0.0,
            "regression_pressure": 0.0,
        }

        # Recent memories + density (same source as the positive_memory
        # context builder; we deliberately read the same canonical
        # surface so both Tier-2 specialists see identical history).
        try:
            from memory.storage import memory_storage
            from memory.maintenance import MAX_MEMORIES
            mems = memory_storage.get_all() or []
            ctx["recent_memories"] = mems[-32:]
            ctx["max_memories"] = MAX_MEMORIES
            if MAX_MEMORIES > 0:
                ctx["memory_density"] = min(len(mems) / float(MAX_MEMORIES), 1.0)
        except Exception:
            pass

        # Tier-1 distillation failure rate. The orchestrator already
        # tracks per-focus failure counts for distillation rollback
        # safety. We expose the normalised rate as a system-friction
        # signal that does not require tagged memory volume to fire,
        # which keeps the Block B signal alive during low-conversation
        # periods (the watch item flagged for emotion_depth).
        try:
            failures = sum(self._tier1_failure_counts.values())
            tier1_count = max(len(_TIER1_FOCUSES), 1)
            # Normalise against (focus_count * regression_threshold) so a
            # single rolling failure does not saturate the dim.
            denom = float(tier1_count * 3)
            ctx["tier1_failure_rate"] = min(failures / denom, 1.0) if denom > 0 else 0.0
        except Exception:
            pass

        # Quarantine pressure (composite, already in [0, 1]). The
        # PressureState exposes a get_snapshot() dict with ``composite``
        # as the canonical scalar; we treat any failure (module not
        # importable, snapshot empty) as 0.0 pressure rather than
        # raising into the tick loop.
        try:
            from epistemic.quarantine.pressure import get_quarantine_pressure
            snap = get_quarantine_pressure().get_snapshot() or {}
            ctx["quarantine_pressure"] = float(snap.get("composite", 0.0) or 0.0)
        except Exception:
            pass

        # Note: ``contradiction_debt`` is currently left at 0.0. The
        # belief-graph subsystem tracks contradictions but does not
        # expose a single normalised debt scalar; wiring an unnormalised
        # count here would inflate Block C's contribution. The proper
        # hook lands when the BeliefGraph exposes a calibrated debt
        # accessor, mirroring the P3.6 ``emotion_positive_bias`` plan.
        # ``low_confidence_retrieval_rate`` and ``regression_pressure``
        # are similarly advisory until their telemetry hooks land.

        return ctx

    def _build_speaker_profile_context(self) -> dict[str, Any]:
        """Gather real-time context for the speaker_profile encoder.

        Mirror of :meth:`_build_positive_memory_context` /
        :meth:`_build_negative_memory_context` for P3.8. The encoder
        itself is a pure function of this context dict so it stays
        unit-testable; the orchestrator is the only place that touches
        live singletons on its behalf.

        **No raw embedding leak.** This builder consumes only
        :meth:`IdentityFusion.get_status` (already embedding-free by
        design — see ``brain/perception/identity_fusion.py``) and the
        soul-service relationship registry size. ``speaker_repr``
        ECAPA vectors and ``face_repr`` vectors NEVER cross this
        boundary. The encoder enforces this contract a second time
        with a regression test that feeds embedding-shaped keys into
        the context and asserts the output is unchanged.

        Every field is best-effort and degrades to a default on import
        / attribute failure; the encoder treats missing fields as 0.0
        contributions. This keeps the dispatch path safe even if the
        identity-fusion subsystem is offline (e.g. during boot
        stabilisation).
        """
        ctx: dict[str, Any] = {
            "identity_confidence": 0.0,
            "is_known": False,
            "voice_confidence": 0.0,
            "face_confidence": 0.0,
            "voice_name": "",
            "face_name": "",
            "conflict": False,
            "voice_age_s": 0.0,
            "face_age_s": 0.0,
            "flip_count": 0,
            "visible_person_count": 0,
            "multi_person_suppression_active": False,
            "cold_start_active": False,
            "voice_trust_state": "",
            "known_speakers_count": 0,
            "current_speaker_interaction_count": 0,
            "relationships_count": 0,
            "rapport_stability": 0.0,
        }

        # IdentityFusion status is held by IdentityResolver during boot
        # (set via ``identity_resolver.set_fusion(self.identity_fusion)``
        # in perception_orchestrator). The ``_fusion`` attribute is
        # private; we read it defensively and fall through silently if
        # the wire-up has not happened yet.
        try:
            from identity.resolver import IdentityResolver
            resolver = IdentityResolver.get_instance()
            fusion = getattr(resolver, "_fusion", None)
            if fusion is not None:
                status = fusion.get_status() or {}
                ctx["identity_confidence"] = float(status.get("confidence", 0.0) or 0.0)
                ctx["is_known"] = bool(status.get("is_known", False))
                ctx["conflict"] = bool(status.get("conflict", False))
                ctx["flip_count"] = int(status.get("flip_count", 0) or 0)
                ctx["visible_person_count"] = int(
                    status.get("visible_person_count", 0) or 0,
                )
                ctx["multi_person_suppression_active"] = bool(
                    status.get("multi_person_suppression_active", False),
                )
                ctx["cold_start_active"] = bool(status.get("cold_start_active", False))
                ctx["voice_trust_state"] = str(status.get("voice_trust_state", "") or "")
                voice = status.get("voice_signal") or {}
                face = status.get("face_signal") or {}
                ctx["voice_name"] = str(voice.get("name", "") or "")
                ctx["voice_confidence"] = float(voice.get("confidence", 0.0) or 0.0)
                ctx["voice_age_s"] = float(voice.get("age_s", 0.0) or 0.0)
                ctx["face_name"] = str(face.get("name", "") or "")
                ctx["face_confidence"] = float(face.get("confidence", 0.0) or 0.0)
                ctx["face_age_s"] = float(face.get("age_s", 0.0) or 0.0)
            ctx["known_speakers_count"] = len(resolver.get_known_names())
        except Exception:
            pass

        # Relationship registry size from the soul service (read-only).
        # We deliberately do not read individual relationship payloads —
        # only the count enters the encoder.
        try:
            from consciousness.soul import soul_service
            rels = getattr(soul_service.identity, "relationships", {}) or {}
            ctx["relationships_count"] = len(rels)
        except Exception:
            pass

        # Note: ``current_speaker_interaction_count`` and
        # ``rapport_stability`` are intentionally left at 0 / 0.0 for
        # this ship. The first lives on the perception_orchestrator's
        # ``speaker_id._profiles`` (no module-level singleton — same
        # pattern as P3.6 episode access), and the second requires a
        # calibrated rapport-stability accessor that does not exist
        # yet. Both degrade cleanly; tests cover the populated paths
        # via synthetic context dicts.

        return ctx

    def _build_temporal_pattern_context(self) -> dict[str, Any]:
        """Gather real-time context for the temporal_pattern encoder.

        Mirror of :meth:`_build_positive_memory_context` /
        :meth:`_build_negative_memory_context` /
        :meth:`_build_speaker_profile_context` for P3.9. The encoder
        itself is a pure function of this context dict so it stays
        unit-testable; the orchestrator is the only place that touches
        live singletons on its behalf.

        **Strict privacy / truth-boundary contract.** This builder
        emits ONLY:

          * Counts of activity events in fixed time windows.
          * Time deltas (``seconds_since_last_activity``,
            ``idle_seconds``).
          * Mode-manager scalars (``mode``, ``mode_duration_s``,
            ``mode_history_len``, ``mode_transitions_last_hour``).
          * A short list of recent activity timestamps for cadence-CV
            computation.

        It does NOT emit any timestamp-of-day, weekday, or
        per-speaker-schedule field. The encoder source is statically
        scanned by a regression test for any reference to
        ``hour_of_day`` / ``weekday`` / ``schedule`` / ``calendar``
        keys; adding such a key in a future PR forces a deliberate
        sign-off on the privacy contract.

        Every field is best-effort and degrades to a default on
        import / attribute failure; the encoder treats missing
        fields as 0.0 contributions (or saturates the recency
        sentinel to "infinitely stale"). This keeps the dispatch
        path safe even if memory storage or the mode manager is
        offline (e.g. during boot stabilisation).
        """
        import time

        now = time.time()
        ctx: dict[str, Any] = {
            "seconds_since_last_activity": float("inf"),
            "count_last_10min": 0,
            "count_last_1hour": 0,
            "count_last_24hour": 0,
            "idle_seconds": float("inf"),
            "mode": "",
            "mode_duration_s": 0.0,
            "mode_history_len": 0,
            "mode_transitions_last_hour": 0,
            "recent_activity_timestamps": (),
            "medium_activity_timestamps": (),
            "activity_count_last_30min": 0,
            "activity_count_prior_30min": 0,
            "rhythm_familiarity": 0.0,
        }

        # Memory-storage timestamps as the activity proxy. Any
        # canonical memory write counts as a derived "activity" event;
        # the encoder is a pure function of *counts and timestamps*,
        # not of memory content, so this read does not leak any
        # user-content data into the signal.
        try:
            from memory.storage import memory_storage
            all_mems = memory_storage.get_all() or []
            timestamps_60m: list[float] = []
            timestamps_24h: list[float] = []
            count_10m = 0
            count_60m = 0
            count_24h = 0
            count_last_30 = 0
            count_prior_30 = 0
            most_recent = 0.0
            for m in all_mems:
                ts = float(getattr(m, "timestamp", 0.0) or 0.0)
                if ts <= 0.0:
                    continue
                age = now - ts
                if age < 0.0:
                    continue
                if age <= 600.0:
                    count_10m += 1
                if age <= 3600.0:
                    count_60m += 1
                    timestamps_60m.append(ts)
                if age <= 86400.0:
                    count_24h += 1
                    timestamps_24h.append(ts)
                if age <= 1800.0:
                    count_last_30 += 1
                elif age <= 3600.0:
                    count_prior_30 += 1
                if ts > most_recent:
                    most_recent = ts
            ctx["count_last_10min"] = count_10m
            ctx["count_last_1hour"] = count_60m
            ctx["count_last_24hour"] = count_24h
            ctx["activity_count_last_30min"] = count_last_30
            ctx["activity_count_prior_30min"] = count_prior_30
            # Cap timestamp lists to keep CV computation cheap.
            if timestamps_60m:
                ctx["recent_activity_timestamps"] = tuple(
                    sorted(timestamps_60m)[-50:]
                )
            if timestamps_24h:
                ctx["medium_activity_timestamps"] = tuple(
                    sorted(timestamps_24h)[-100:]
                )
            if most_recent > 0.0:
                ctx["seconds_since_last_activity"] = max(0.0, now - most_recent)
                ctx["idle_seconds"] = max(0.0, now - most_recent)
        except Exception:
            pass

        # Mode-manager state. ``mode_manager.get_state()`` exposes
        # mode, since (timestamp), duration_s, and history_len. We
        # additionally read ``_history`` defensively to count how
        # many transitions have occurred in the last hour. No
        # transition reasons or content cross the boundary — only
        # the count.
        try:
            from consciousness.modes import mode_manager
            state = mode_manager.get_state() or {}
            mode_label = str(state.get("mode", "") or "")
            ctx["mode"] = mode_label
            ctx["mode_duration_s"] = float(state.get("duration_s", 0.0) or 0.0)
            ctx["mode_history_len"] = int(state.get("history_len", 0) or 0)
            history = getattr(mode_manager, "_history", None) or []
            transitions_last_hour = 0
            for entry in history:
                t = float(entry.get("time", 0.0) or 0.0)
                if t > 0.0 and (now - t) <= 3600.0:
                    transitions_last_hour += 1
            ctx["mode_transitions_last_hour"] = transitions_last_hour
        except Exception:
            pass

        # ``rhythm_familiarity`` is intentionally left at 0.0 for this
        # ship. A calibrated "this hour resembles recent windows"
        # accessor does not exist yet — wiring an unnormalised proxy
        # here would risk drifting into the schedule-claim space the
        # privacy contract forbids. Encoder degrades cleanly.

        return ctx

    def _build_skill_transfer_context(self) -> dict[str, Any]:
        """Gather real-time context for the skill_transfer encoder.

        Mirror of :meth:`_build_positive_memory_context` /
        :meth:`_build_negative_memory_context` /
        :meth:`_build_speaker_profile_context` /
        :meth:`_build_temporal_pattern_context` for P3.10. The
        encoder itself is a pure function of this context dict so it
        stays unit-testable; the orchestrator is the only place that
        touches live singletons on its behalf.

        **Capability / truth-boundary contract.** This builder emits
        ONLY:

          * ``total_skills`` and a ``by_status`` count map.
          * A list of skill descriptors with status, capability_type,
            has_evidence (bool), artifact_count, evidence_count, and
            matrix_protocol (bool).
          * ``active_jobs_count`` / ``failed_jobs_count`` integer
            counts (when accessible).

        It does NOT emit:

          * Any "this skill is verified — promote it" hint.
          * Any raw skill payload, plugin source, or capability
            secret.
          * Any reference to the capability_gate authoritative
            decision; capability promotion remains gated by the
            existing capability_gate path, which this encoder does
            not touch.

        The downstream encoder is statically scanned by a
        regression test for any reference to canonical mutators
        (``skill_registry.set_status`` / ``capability_gate.allow`` /
        ``promote_capability`` / etc.). None present.
        """
        ctx: dict[str, Any] = {
            "total_skills": 0,
            "by_status": {},
            "skills": (),
            "active_jobs_count": 0,
            "active_jobs": (),
            "failed_jobs_count": 0,
            "capability_type_overlap": 0.0,
            "transfer_advisory": 0.0,
        }

        # Skill registry summary — the registry IS a module-level
        # singleton, so it is safe to import here. The summary is
        # the caller-level read; we do not call mutators or expose
        # internal payloads.
        try:
            from skills.registry import skill_registry
            snap = skill_registry.get_status_snapshot() or {}
            ctx["total_skills"] = int(snap.get("total", 0) or 0)
            by_status_raw = snap.get("by_status", {}) or {}
            try:
                ctx["by_status"] = {
                    str(k): int(v or 0) for k, v in by_status_raw.items()
                }
            except Exception:
                ctx["by_status"] = {}

            skills_in: list[dict[str, Any]] = []
            for s in (snap.get("skills") or []):
                if not isinstance(s, dict):
                    continue
                evidence_summary = s.get("evidence_summary") or {}
                artifact_count = 0
                evidence_count = 0
                has_evidence = False
                if isinstance(evidence_summary, dict):
                    artifact_count = int(
                        evidence_summary.get("artifact_count", 0) or 0,
                    )
                    evidence_count = int(
                        evidence_summary.get("evidence_count", 0) or 0,
                    )
                    has_evidence = bool(
                        evidence_summary.get("has_evidence",
                                             evidence_count > 0
                                             or artifact_count > 0),
                    )
                skills_in.append({
                    "status": str(s.get("status", "") or ""),
                    "capability_type": str(s.get("capability_type", "") or ""),
                    "has_evidence": has_evidence,
                    "artifact_count": artifact_count,
                    "evidence_count": evidence_count,
                    "matrix_protocol": bool(s.get("matrix_protocol", False)),
                })
            ctx["skills"] = tuple(skills_in)
        except Exception:
            pass

        # Learning-job counts. ``LearningJobOrchestrator`` is owned by
        # ``main.py`` and not module-level in this codebase, so we
        # cannot read its full state from here. We DO read the
        # registry's active-job count if present (best-effort) and
        # otherwise leave the count at 0; the encoder degrades
        # cleanly.
        try:
            from skills.registry import skill_registry
            snap = skill_registry.get_status_snapshot() or {}
            jobs_block = snap.get("learning_jobs") or {}
            if isinstance(jobs_block, dict):
                ctx["active_jobs_count"] = int(
                    jobs_block.get("active_count", 0) or 0,
                )
                ctx["failed_jobs_count"] = int(
                    jobs_block.get("failed_count", 0) or 0,
                )
        except Exception:
            pass

        # ``capability_type_overlap`` and ``transfer_advisory`` are
        # intentionally left at 0.0 for this ship. They are
        # advisory dims that would require a calibrated cross-skill
        # similarity surface; wiring an unnormalised proxy here
        # would risk drifting into the "similarity equals
        # capability" failure mode the contract forbids. Encoder
        # degrades cleanly without them.

        return ctx

    def _compute_tier1_signal_value(self, network: NetworkArchitecture, focus: HemisphereFocus) -> float:
        """Tier-1 compressor/approximator signal: use accuracy as the signal.

        Compressor networks are autoencoders trained on teacher embeddings
        (192-dim speaker, 512-dim face).  At broadcast-slot inference time we
        don't have a live teacher embedding to feed them, so running the
        network on memory-state features would produce garbage.  Instead, use
        the training accuracy (which reflects reconstruction fidelity) as a
        stable, meaningful proxy signal for the policy StateEncoder.
        """
        active = self._registry.get_active(network.focus.value)
        if active:
            return float(active.accuracy)
        return float(network.performance.accuracy)

    def _compute_signal_score(self, network: NetworkArchitecture, focus: HemisphereFocus) -> float:
        """Compute an impact score for ranking in the Global Broadcast.

        For Matrix Protocol specialists, the score also incorporates the
        specialist impact formula so they must earn broadcast influence.
        """
        acc = network.performance.accuracy
        reliability = network.performance.reliability
        is_tier1 = focus in _TIER1_FOCUSES
        tier_bonus = 0.05 if is_tier1 else 0.0
        base_score = acc * 0.6 + reliability * 0.3 + tier_bonus + 0.1

        if network.specialist_lifecycle is not None:
            specialist_score = self._compute_specialist_impact(network)
            network.specialist_impact_score = specialist_score
            return base_score * 0.5 + specialist_score * 0.5

        return base_score

    @staticmethod
    def _compute_specialist_impact(network: NetworkArchitecture) -> float:
        """Impact score for Matrix Protocol specialists.

        impact = 0.35*verification + 0.25*downstream_reward_delta
               + 0.20*recent_utility + 0.10*stability + 0.10*recency
        """
        from hemisphere.types import SpecialistLifecycleStage

        verification = 0.0
        if network.specialist_lifecycle in (
            SpecialistLifecycleStage.VERIFIED_PROBATIONARY,
            SpecialistLifecycleStage.BROADCAST_ELIGIBLE,
            SpecialistLifecycleStage.PROMOTED,
        ):
            verification = 1.0
        elif network.specialist_lifecycle == SpecialistLifecycleStage.PROBATIONARY_TRAINING:
            verification = 0.3

        downstream_reward = min(1.0, network.performance.accuracy * 1.5)

        age_hours = (_time.time() - network.created_at) / 3600.0
        recent_utility = max(0, 1.0 - age_hours / 168.0)  # decays over 7 days

        stability = network.performance.reliability

        recency = 1.0
        if network.specialist_verification_ts > 0:
            since_verify = (_time.time() - network.specialist_verification_ts) / 3600.0
            recency = max(0, 1.0 - since_verify / 336.0)  # decays over 14 days

        return (
            0.35 * verification
            + 0.25 * downstream_reward
            + 0.20 * recent_utility
            + 0.10 * stability
            + 0.10 * recency
        )

    def _build_inference_input(self, input_size: int, extra_features: dict[str, float] | None = None) -> list[float]:
        """Build a feature vector for inference from current system state.

        extra_features are appended after the base 8 features, enabling
        Tier-2 hemispheres to receive Tier-1 distilled signals as input.
        """
        from memory.storage import memory_storage
        memories = memory_storage.get_all()
        if not memories:
            return [0.5] * input_size
        recent = memories[-1]
        weight = min(getattr(recent, "weight", 0.5), 1.0)
        tags = getattr(recent, "tags", ())
        mem_type = getattr(recent, "type", "observation")
        decay = min(getattr(recent, "decay_rate", 0.01), 0.1) * 10.0
        is_core = 1.0 if mem_type == "core" else 0.0
        from memory.maintenance import MAX_MEMORIES
        density = min(len(memories) / float(MAX_MEMORIES), 1.0)
        row = [weight, min(len(tags), 10) / 10.0, is_core, decay, density, 0.0, 0.0, 0.0]
        if extra_features:
            for key in sorted(extra_features.keys()):
                row.append(extra_features[key])
        while len(row) < input_size:
            row.append(0.0)
        return row[:input_size]

    def _collect_tier1_features(self) -> dict[str, float]:
        """Collect Tier-1 distilled NN signals for Tier-2 enrichment."""
        features: dict[str, float] = {}
        for focus in _TIER1_FOCUSES:
            if focus in _SHADOW_ONLY_TIER1_FOCUSES:
                continue
            best = self._get_best_network(focus)
            if best is None:
                continue
            try:
                sig = self._compute_tier1_signal_value(best, focus)
                features[f"{focus.value}_signal"] = sig
            except Exception:
                pass
        return features

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_distillation_stats(self) -> dict[str, Any]:
        """Return distillation collector stats for learning-job context."""
        collector = self._get_distillation_collector()
        if collector is None:
            return {}
        try:
            return collector.get_stats()
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Dashboard state
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        evo_gens = {
            fk: evo.generation for fk, evo in self._evolution_engines.items()
        }
        hemi_state = self._registry.get_state(evo_gens)
        with self._networks_lock:
            loaded_by_focus = {
                net.focus: net
                for net in self._networks.values()
                if net.status in (NetworkStatus.READY, NetworkStatus.ACTIVE)
            }

        distill_stats = {}
        collector = self._get_distillation_collector()
        if collector:
            distill_stats = collector.get_stats()
        now = _time.time()

        slot_info = []
        for s in self._broadcast_slots:
            slot_info.append({"name": s["name"], "value": round(s["value"], 4),
                              "score": round(s["score"], 4), "dwell": s["dwell"]})

        hemisphere_rows = []
        total_loaded_networks = 0
        total_loaded_parameters = 0
        for s in hemi_state.hemispheres:
            loaded = loaded_by_focus.get(s.focus)
            actual_network_count = 1 if loaded else 0
            actual_active_id = loaded.id if loaded else None
            focus_val = s.focus.value
            if focus_val in self._tier1_disabled:
                actual_status = "disabled"
            elif focus_val in self._tier1_failure_counts and self._tier1_failure_counts[focus_val] > 0:
                actual_status = "quarantined"
            else:
                actual_status = "active" if loaded else ("inactive" if s.total_attempts > 0 else "idle")
            actual_migration_readiness = (
                round(loaded.performance.migration_readiness, 4)
                if loaded else 0.0
            )
            if loaded:
                total_loaded_networks += 1
                total_loaded_parameters += loaded.topology.total_parameters

            hemisphere_rows.append({
                "focus": s.focus.value,
                "network_count": actual_network_count,
                "active_network_id": actual_active_id,
                "best_accuracy": round(s.best_accuracy, 4),
                "best_training_accuracy": round(s.best_training_accuracy, 4),
                "best_validation_accuracy": round(s.best_validation_accuracy, 4),
                "best_loss": round(s.best_loss, 4) if s.best_loss < 1e6 else None,
                "total_attempts": s.total_attempts,
                "evolution_generations": s.evolution_generations,
                "migration_readiness": actual_migration_readiness,
                "status": actual_status,
            })

        return {
            "hemisphere_state": {
                "hemispheres": hemisphere_rows,
                "total_networks": total_loaded_networks,
                "total_parameters": total_loaded_parameters,
                "active_substrate": hemi_state.active_substrate.value,
                "overall_migration_readiness": round(hemi_state.overall_migration_readiness, 4),
                "evolution_active": total_loaded_networks > 0,
                "max_total_networks": MAX_TOTAL_NETWORKS,
            },
            "cycle_count": self._cycle_count,
            "migration_history_count": len(self._migration.migration_history),
            "enabled": self._enabled,
            "gap_detector": self._gap_detector.get_state(),
            "dynamic_focuses": {
                name: {
                    "source_dimension": df.source_dimension,
                    "gap_severity": round(df.gap_severity, 3),
                    "impact_score": round(df.impact_score, 3),
                    "cycles_alive": df.cycles_alive,
                    "deprecated": df.deprecated,
                }
                for name, df in self._dynamic_focuses.items()
            },
            "distillation": distill_stats,
            "tier1_gating": {
                "failure_counts": dict(self._tier1_failure_counts),
                "disabled_for_session": sorted(self._tier1_disabled),
                "accuracy_floor": TIER1_MIN_ACCURACY,
                "max_failures": TIER1_MAX_CONSECUTIVE_FAILURES,
                "regression_counts": dict(self._tier1_regression_counts),
                "regression_delta": DISTILLATION_REGRESSION_DELTA,
                "regression_min_samples": DISTILLATION_REGRESSION_MIN_SAMPLES,
                "regression_threshold": DISTILLATION_MAX_CONSECUTIVE_REGRESSIONS,
                "regression_cooldown_window_s": DISTILLATION_REGRESSION_COOLDOWN_S,
                "regression_cooldown_backoff": DISTILLATION_REGRESSION_COOLDOWN_BACKOFF,
                "regression_cooldown_max_s": DISTILLATION_REGRESSION_COOLDOWN_MAX_S,
                "regression_cooldown_strikes": dict(self._tier1_regression_cooldown_strikes),
                "regression_cooldown_remaining_s": {
                    k: round(v - now, 1)
                    for k, v in self._tier1_regression_cooldown_until.items()
                    if v > now
                },
            },
            "broadcast_slots": slot_info,
            "num_broadcast_slots": self._num_broadcast_slots,
            "matrix_specialists": self._get_specialist_summary(),
            "expansion": {
                "triggered": self._expansion_triggered,
                "triggered_at": self._expansion_triggered_at,
                "slot_count": self._num_broadcast_slots,
            },
        }

    def _check_expansion_trigger(self) -> None:
        """Check if broadcast slot expansion (4->6) should be triggered.

        Three conditions must all be met:
        1. At least EXPANSION_MIN_PROMOTED specialists at PROMOTED stage
        2. Mean impact score across promoted specialists > EXPANSION_MIN_IMPACT
        3. All promoted specialists stable for >= EXPANSION_STABILITY_DAYS
        """
        if self._expansion_triggered:
            return

        from hemisphere.types import (
            SpecialistLifecycleStage, MATRIX_ELIGIBLE_FOCUSES,
        )

        now = _time.time()
        stability_threshold = now - (EXPANSION_STABILITY_DAYS * 86400.0)

        with self._networks_lock:
            promoted = [
                n for n in self._networks.values()
                if n.focus in MATRIX_ELIGIBLE_FOCUSES
                and n.specialist_lifecycle == SpecialistLifecycleStage.PROMOTED
            ]

        if len(promoted) < EXPANSION_MIN_PROMOTED:
            return

        mean_impact = sum(n.specialist_impact_score for n in promoted) / len(promoted)
        if mean_impact <= EXPANSION_MIN_IMPACT:
            return

        earliest_verification = min(n.specialist_verification_ts for n in promoted)
        if earliest_verification <= 0 or earliest_verification > stability_threshold:
            return

        self._expansion_triggered = True
        self._expansion_triggered_at = now
        logger.info(
            "M6 expansion triggered: %d promoted specialists, "
            "mean_impact=%.3f, earliest_verification=%.0fs ago",
            len(promoted), mean_impact, now - earliest_verification,
        )

        from consciousness.events import MATRIX_EXPANSION_TRIGGERED, event_bus
        event_bus.emit(
            MATRIX_EXPANSION_TRIGGERED,
            promoted_count=len(promoted),
            mean_impact=mean_impact,
            earliest_verification_age_s=now - earliest_verification,
        )

    @property
    def expansion_triggered(self) -> bool:
        return self._expansion_triggered

    @property
    def num_broadcast_slots(self) -> int:
        return self._num_broadcast_slots

    def _get_specialist_summary(self) -> list[dict[str, Any]]:
        from hemisphere.types import MATRIX_ELIGIBLE_FOCUSES
        with self._networks_lock:
            specialists = [
                n for n in self._networks.values()
                if n.focus in MATRIX_ELIGIBLE_FOCUSES
                and n.specialist_lifecycle is not None
            ]
        return [
            {
                "name": n.name,
                "focus": n.focus.value,
                "lifecycle": n.specialist_lifecycle.value if n.specialist_lifecycle else "unknown",
                "impact_score": round(n.specialist_impact_score, 3),
                "job_id": n.specialist_job_id,
                "accuracy": round(n.performance.accuracy, 3),
            }
            for n in specialists
        ]

    # ------------------------------------------------------------------
    # Model restoration on startup
    # ------------------------------------------------------------------

    def restore_models(self) -> int:
        """Reload persisted models from disk and populate _networks so
        restored models compete for broadcast slots immediately."""
        count = 0
        for focus in HemisphereFocus:
            active = self._registry.get_active(focus.value)
            if active is None:
                continue
            if active.network_id in self._networks:
                continue
            topo_json = active.topology_json
            if not topo_json:
                logger.warning(
                    "Skipping %s hemisphere v%d restore: empty topology_json "
                    "(model was registered by older code before topology serialization)",
                    focus.value, active.version,
                )
                continue
            try:
                from hemisphere.types import (
                    LayerDefinition, NetworkTopology,
                    NetworkArchitecture, NetworkStatus,
                    PerformanceMetrics, TrainingProgress,
                )
                layers_data = topo_json.get("layers", [])
                layers = tuple(
                    LayerDefinition(
                        id=ld["id"],
                        layer_type=ld["layer_type"],
                        node_count=ld["node_count"],
                        activation=ld["activation"],
                        dropout=float(ld.get("dropout", 0.0) or 0.0),
                    )
                    for ld in layers_data
                )
                topo = NetworkTopology(
                    input_size=topo_json["input_size"],
                    layers=layers,
                    output_size=topo_json["output_size"],
                    total_parameters=topo_json["total_parameters"],
                    activation_functions=tuple(ld["activation"] for ld in layers_data),
                )
                import os
                if os.path.exists(active.path):
                    model = self._engine.load_model(active.network_id, topo, active.path)

                    net = NetworkArchitecture(
                        id=active.network_id,
                        name=active.name or f"Restored-{focus.value}-v{active.version}",
                        focus=focus,
                        topology=topo,
                        performance=PerformanceMetrics(
                            accuracy=float(active.accuracy),
                            loss=float(active.loss),
                            validation_accuracy=float(active.accuracy),
                            validation_loss=float(active.loss),
                            reliability=max(0.5, float(active.accuracy) * 0.9),
                            migration_readiness=float(active.accuracy),
                            last_evaluated=_time.time(),
                        ),
                        training_progress=TrainingProgress(
                            total_epochs=0,
                            learning_rate=0.0,
                            batch_size=0,
                            is_training=False,
                        ),
                        status=NetworkStatus.ACTIVE,
                        is_active=True,
                        created_at=float(active.created_at),
                        design_reasoning="Restored from HemisphereRegistry active version",
                    )
                    net._model_ref = model
                    with self._networks_lock:
                        self._networks[net.id] = net

                    count += 1
                    logger.info("Restored %s hemisphere model v%d (discoverable)", focus.value, active.version)

                    logger.info(
                        "Restored Tier-1 '%s' at %.3f%% accuracy",
                        focus.value, float(active.accuracy) * 100,
                    )
                else:
                    self._registry.discard_version(
                        focus.value,
                        active.version,
                        delete_weights=False,
                        reason="missing_weights",
                    )
                    logger.warning(
                        "Active %s hemisphere model missing at %s; discarded stale registry entry",
                        focus.value,
                        active.path,
                    )
            except Exception:
                self._registry.discard_version(
                    focus.value,
                    active.version,
                    delete_weights=True,
                    reason="restore_incompatible",
                )
                logger.exception(
                    "Failed to restore %s hemisphere model; discarded incompatible checkpoint v%d",
                    focus.value,
                    active.version,
                )
        return count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_networks_for_focus(self, focus: HemisphereFocus) -> list[NetworkArchitecture]:
        with self._networks_lock:
            return [
                n for n in self._networks.values()
                if n.focus == focus and n.status in (NetworkStatus.READY, NetworkStatus.ACTIVE)
            ]

    def _get_best_network(self, focus: HemisphereFocus) -> NetworkArchitecture | None:
        nets = self._get_networks_for_focus(focus)
        if not nets:
            return None
        return max(nets, key=lambda n: n.performance.accuracy)

    def record_outcome(self, outcome_type: str, data: dict[str, Any]) -> None:
        """Feed real-world outcome signals to the matching hemisphere focus.

        Stores signals in a per-focus buffer so hemisphere NNs can
        incorporate real conversation quality into their training.
        """
        _OUTCOME_FOCUS_MAP = {
            "conversation": HemisphereFocus.MEMORY,
            "emotion": HemisphereFocus.MOOD,
            "memory_recall": HemisphereFocus.MEMORY,
            "trait_shift": HemisphereFocus.TRAITS,
            "error": HemisphereFocus.GENERAL,
        }
        focus = _OUTCOME_FOCUS_MAP.get(outcome_type)
        if focus is None:
            return

        success = data.get("success", data.get("completed", False))
        quality = data.get("quality", 0.5)
        signal = quality if success else quality * 0.3

        self._outcome_buffer[focus.value].append({
            "timestamp": _time.time(),
            "signal": signal,
            "type": outcome_type,
            "data": data,
        })
        self._outcomes_since_train[focus.value] = (
            self._outcomes_since_train.get(focus.value, 0) + 1
        )

        logger.debug("Hemisphere outcome stored: %s → %s (signal=%.2f, buffer=%d)",
                      outcome_type, focus.value, signal, len(self._outcome_buffer[focus.value]))

    def get_outcome_history(self, focus: str, window_s: float = 3600.0) -> list[dict]:
        """Return recent outcome signals for a hemisphere focus."""
        cutoff = _time.time() - window_s
        return [o for o in self._outcome_buffer.get(focus, []) if o["timestamp"] > cutoff]

    def _retrain_from_outcomes(self, feed: HemisphereDataFeed, focus: HemisphereFocus) -> None:
        """Retrain the active network for a focus using outcome-enriched data."""
        best = self._get_best_network(focus)
        if best is None:
            return
        if not self._build_lock.acquire(blocking=False):
            return

        outcome_history = list(self._outcome_buffer.get(focus.value, []))

        def _retrain() -> None:
            try:
                from hemisphere.data_feed import prepare_training_tensors
                from hemisphere.engine import HemisphereEngine

                features, labels = prepare_training_tensors(
                    feed, focus, best.topology.input_size, best.topology.output_size,
                    outcome_history=outcome_history,
                )
                if features is not None and labels is not None:
                    self._engine.retrain_network(best, features, labels)
                    logger.info("Hemisphere %s retrained with %d outcome signals",
                                focus.value, len(outcome_history))
            except Exception:
                logger.debug("Outcome-triggered retrain failed for %s", focus.value, exc_info=True)
            finally:
                self._build_lock.release()

        t = threading.Thread(target=_retrain, daemon=True, name=f"hemi-retrain-{focus.value}")
        t.start()

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def dispose(self) -> None:
        self._enabled = False
        self._engine.dispose()
