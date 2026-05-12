"""Unified World Model — fuses subsystem snapshots into a single belief state.

Reads from 9 signal sources on every tick (default 5 s), builds a
:class:`WorldState`, detects deltas, runs the :class:`CausalEngine`, and
validates expired predictions.  Shadow-only in Phase 1; promotion to
advisory / active is handled by :class:`WorldModelPromotion`.

Phase 3 adds a :class:`MentalSimulator` that runs hypothetical projections
in shadow mode during each tick cycle.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

from consciousness.events import (
    WORLD_MODEL_DELTA,
    WORLD_MODEL_UPDATE,
    WORLD_MODEL_PREDICTION_VALIDATED,
    event_bus,
)
from cognition.causal_engine import CausalEngine
from cognition.planner import WorldPlanner
from cognition.promotion import WorldModelPromotion, SimulatorPromotion
from cognition.simulator import MentalSimulator
from cognition.world_state import (
    ConversationState,
    PhysicalState,
    SystemState,
    UserState,
    WorldDelta,
    WorldState,
    _compute_uncertainty,
)
from cognition.world_schema import CanonicalWorldState
from cognition.world_adapters import CanonicalWorldProjector

logger = logging.getLogger(__name__)

MAX_DELTA_HISTORY = 200


class WorldModel:
    """The agent's unified belief state about reality."""

    def __init__(
        self,
        scene_tracker: Any = None,
        attention: Any = None,
        presence: Any = None,
        mode_manager: Any = None,
        goal_manager: Any = None,
        episodes: Any = None,
        health_monitor: Any = None,
        analytics: Any = None,
        perc_orch: Any = None,
    ) -> None:
        self._scene_tracker = scene_tracker
        self._attention = attention
        self._presence = presence
        self._mode_manager = mode_manager
        self._goal_manager = goal_manager
        self._episodes = episodes
        self._health_monitor = health_monitor
        self._analytics = analytics
        self._perc_orch = perc_orch

        self._current: WorldState = WorldState()
        self._previous: WorldState | None = None
        self._version: int = 0
        self._tick_number: int = 0
        self._boot_ts: float = time.time()

        self._deltas: deque[WorldDelta] = deque(maxlen=MAX_DELTA_HISTORY)
        self._causal = CausalEngine()
        self._promotion = WorldModelPromotion()

        self._update_count: int = 0

        # Phase 3: Mental Simulator (shadow mode)
        # Simulator gets its own CausalEngine so simulation predictions
        # don't pollute the real prediction pipeline (accuracy, cooldown,
        # deque eviction).
        self._sim_causal = CausalEngine()
        self._simulator = MentalSimulator(self._sim_causal)
        self._sim_promotion = SimulatorPromotion()
        self._planner = WorldPlanner()

        # Canonical substrate (runs in parallel with legacy WorldState)
        self._canonical_projector = CanonicalWorldProjector()
        self._canonical: CanonicalWorldState | None = None

    # -- Public API ---------------------------------------------------------

    def update(self) -> WorldState:
        """Rebuild the world state from all sources, detect deltas, run causal engine."""
        now = time.time()
        self._tick_number += 1
        self._version += 1

        physical = self._read_physical(now)
        user = self._read_user(now)
        conversation = self._read_conversation(now)
        system = self._read_system(now)

        staleness = {
            "physical": now - physical.last_update_ts if physical.last_update_ts else 999.0,
            "user": now - user.last_update_ts if user.last_update_ts else 999.0,
            "conversation": now - conversation.last_update_ts if conversation.last_update_ts else 999.0,
            "system": now - system.last_update_ts if system.last_update_ts else 999.0,
        }

        uncertainty = {
            "physical": _compute_uncertainty(
                "physical", staleness["physical"],
                [physical.visible_count / max(physical.entity_count, 1)]
                if physical.entity_count > 0 else [],
            ),
            "user": _compute_uncertainty(
                "user", staleness["user"],
                [user.presence_confidence, user.emotion_confidence,
                 user.speaker_confidence],
            ),
            "conversation": _compute_uncertainty(
                "conversation", staleness["conversation"],
                [1.0] if conversation.active else [0.0],
            ),
            "system": _compute_uncertainty(
                "system", staleness["system"],
                [system.health_score, system.confidence],
            ),
        }

        new_state = WorldState(
            physical=physical,
            user=user,
            conversation=conversation,
            system=system,
            version=self._version,
            timestamp=now,
            tick_number=self._tick_number,
            staleness=staleness,
            uncertainty=uncertainty,
        )

        self._previous = self._current
        self._current = new_state

        # --- Canonical substrate projection (parallel, non-destructive) ---
        try:
            spatial_fused = None
            if self._perc_orch and hasattr(self._perc_orch, "_spatial_fused"):
                spatial_fused = self._perc_orch._spatial_fused or None
            self._canonical = self._canonical_projector.build(
                now=now,
                scene=self._scene_tracker.get_state() if self._scene_tracker else {},
                attention=self._read_attention(),
                presence=self._read_presence(),
                identity=self._read_speaker(),
                spatial=spatial_fused,
            )
        except Exception:
            logger.debug("WorldModel: canonical projection failed", exc_info=True)

        new_deltas = self._detect_deltas(self._previous, self._current)
        for d in new_deltas:
            self._deltas.append(d)
            try:
                event_bus.emit(
                    WORLD_MODEL_DELTA,
                    facet=d.facet,
                    event=d.event,
                    details=d.details,
                    timestamp=d.timestamp,
                    version=self._version,
                )
            except Exception:
                pass

        predictions = self._causal.infer(self._current, new_deltas)
        validated = self._causal.validate_predictions(self._current)
        for v in validated:
            self._promotion.record_outcome(v.outcome == "hit")
            try:
                event_bus.emit(
                    WORLD_MODEL_PREDICTION_VALIDATED,
                    prediction_label=v.label,
                    outcome=v.outcome,
                    prediction_confidence=v.confidence,
                    version=self._version,
                )
            except Exception:
                pass

        # Phase 3: validate simulator predictions against real state
        sim_validated = self._sim_causal.validate_predictions(self._current)
        for sv in sim_validated:
            self._sim_promotion.record_outcome(sv.outcome == "hit")

        self._update_count += 1

        try:
            event_bus.emit(
                WORLD_MODEL_UPDATE,
                version=self._version,
                delta_count=len(new_deltas),
                prediction_count=len(predictions),
                validated_count=len(validated),
            )
        except Exception:
            pass

        if self._update_count <= 3 or self._update_count % 60 == 0:
            logger.info(
                "WorldModel v%d: phys=%d/%d user=%s conv=%s mode=%s unc=[%.2f,%.2f,%.2f,%.2f] "
                "deltas=%d pred=%d validated=%d promo=%d "
                "canonical=[obs=%d ent=%d rel=%d zone=%d arch=%s]",
                self._version,
                physical.visible_count, physical.entity_count,
                "present" if user.present else "absent",
                "active" if conversation.active else "idle",
                system.mode,
                uncertainty["physical"], uncertainty["user"],
                uncertainty["conversation"], uncertainty["system"],
                len(new_deltas), len(predictions), len(validated),
                self._promotion.level,
                len(self._canonical.observations) if self._canonical else 0,
                len(self._canonical.entities) if self._canonical else 0,
                len(self._canonical.relations) if self._canonical else 0,
                len(self._canonical.zones) if self._canonical else 0,
                ",".join(self._canonical.archetypes_active) if self._canonical else "none",
            )

        if self._update_count % 120 == 0 and self._update_count > 0:
            diag = self.get_diagnostics()
            leg = diag["legacy"]
            can = diag["canonical"]
            logger.info(
                "WorldModel diagnostics v%d: "
                "legacy=[rules=%d cooldown=%d validated=%d acc=%.2f "
                "hits=%d miss=%d skip_pend=%d skip_cond=%d skip_conf=%d] "
                "canonical=[obs=%d ent=%d rel=%d zone=%d arch=%s "
                "ent_kinds=%s zone_kinds=%s] promo=%s",
                diag["version"],
                leg["rules_total"], leg["rules_cooldown"],
                leg["total_validated"], leg["overall_accuracy"],
                leg["total_hits"], leg["total_misses"],
                leg["last_skipped_pending"], leg["last_skipped_condition"],
                leg["last_skipped_conflict"],
                can["observation_count"], can["entity_count"],
                can["relation_count"], can["zone_count"],
                ",".join(can["archetypes_active"]) or "none",
                can["entity_kinds"], can["zone_kinds"],
                diag["promotion"].get("level", "?"),
            )

        if validated and self._update_count % 10 == 0:
            self._promotion.save()

        if self._update_count % 50 == 0:
            self._sim_promotion.save()

        # Phase 3: run shadow simulations on detected deltas
        if new_deltas:
            self._run_shadow_simulations(new_deltas)
        self._run_shadow_planner(new_deltas)

        return self._current

    def get_state(self) -> dict[str, Any]:
        """Dashboard-friendly snapshot of the full belief state."""
        return {
            "version": self._current.version,
            "timestamp": self._current.timestamp,
            "tick_number": self._current.tick_number,
            "update_count": self._update_count,
            "staleness": dict(self._current.staleness),
            "uncertainty": dict(self._current.uncertainty),
            "physical": {
                "entity_count": self._current.physical.entity_count,
                "visible_count": self._current.physical.visible_count,
                "stable_count": self._current.physical.stable_count,
                "display_surfaces": len(self._current.physical.display_surfaces),
                "person_count": self._current.physical.person_count,
            },
            "user": {
                "present": self._current.user.present,
                "engagement": round(self._current.user.engagement, 2),
                "emotion": self._current.user.emotion,
                "speaker": self._current.user.speaker_name or "unknown",
                "gesture": self._current.user.gesture,
            },
            "conversation": {
                "active": self._current.conversation.active,
                "topic": self._current.conversation.topic,
                "turn_count": self._current.conversation.turn_count,
                "follow_up": self._current.conversation.follow_up_active,
            },
            "system": {
                "mode": self._current.system.mode,
                "health": round(self._current.system.health_score, 2),
                "confidence": round(self._current.system.confidence, 2),
                "autonomy_level": self._current.system.autonomy_level,
                "active_goal": self._current.system.active_goal_title or None,
                "memory_count": self._current.system.memory_count,
                "uptime_s": round(self._current.system.uptime_s, 0),
            },
            "causal": self._causal.get_accuracy(),
            "predictions": self._causal.get_pending_predictions(),
            "recent_validated": self._causal.get_recent_validated(10),
            "promotion": self._promotion.get_status(),
            "simulator": self._simulator.get_stats(),
            "simulator_promotion": self._sim_promotion.get_status(),
            "planner": self._planner.get_state(),
            "recent_simulations": self._simulator.get_recent_traces(5),
            "recent_deltas": [
                {
                    "facet": d.facet,
                    "event": d.event,
                    "details": d.details,
                    "confidence": d.confidence,
                    "timestamp": d.timestamp,
                }
                for d in list(self._deltas)[-20:]
            ],
            "spatial_active": bool(
                self._perc_orch
                and hasattr(self._perc_orch, "_spatial_fused")
                and self._perc_orch._spatial_fused
            ),
        }

    def get_canonical_state(self) -> dict[str, Any]:
        """Canonical substrate snapshot — entities, relations, zones, archetypes."""
        cs = self._canonical
        if not cs:
            return {
                "observation_count": 0,
                "entities": [],
                "relations": [],
                "zones": [],
                "archetypes_active": [],
                "entity_count": 0,
                "relation_count": 0,
                "zone_count": 0,
            }
        return {
            "observation_count": len(cs.observations),
            "entities": [
                {
                    "entity_id": e.entity_id,
                    "kind": e.kind,
                    "label": e.label,
                    "confidence": e.confidence,
                    "region_id": e.region_id,
                    "stable": e.stable,
                    "tags": list(e.tags),
                }
                for e in cs.entities
            ],
            "relations": [
                {
                    "relation_id": r.relation_id,
                    "kind": r.kind,
                    "subject_id": r.subject_id,
                    "object_id": r.object_id,
                    "confidence": r.confidence,
                    "tags": list(r.tags),
                }
                for r in cs.relations
            ],
            "zones": [
                {
                    "zone_id": z.zone_id,
                    "label": z.label,
                    "kind": z.kind,
                    "confidence": z.confidence,
                    "norms": list(z.norms),
                    "tags": list(z.tags),
                }
                for z in cs.zones
            ],
            "archetypes_active": list(cs.archetypes_active),
            "entity_count": len(cs.entities),
            "relation_count": len(cs.relations),
            "zone_count": len(cs.zones),
        }

    def get_diagnostics(self) -> dict[str, Any]:
        """Combined legacy + canonical diagnostics for observability."""
        cs = self._canonical
        causal = self._causal.get_accuracy()

        canonical_entity_kinds: dict[str, int] = {}
        canonical_zone_kinds: dict[str, int] = {}
        if cs:
            for e in cs.entities:
                canonical_entity_kinds[e.kind] = canonical_entity_kinds.get(e.kind, 0) + 1
            for z in cs.zones:
                canonical_zone_kinds[z.kind] = canonical_zone_kinds.get(z.kind, 0) + 1

        return {
            "tick": self._tick_number,
            "version": self._version,
            "update_count": self._update_count,
            "legacy": {
                "rules_total": causal.get("rules_total", 0),
                "rules_cooldown": causal.get("rules_cooldown", 0),
                "total_validated": causal.get("total_validated", 0),
                "overall_accuracy": causal.get("overall_accuracy", 0.0),
                "total_hits": causal.get("total_hits", 0),
                "total_misses": causal.get("total_misses", 0),
                "pending": causal.get("pending", 0),
                "last_skipped_pending": causal.get("last_skipped_pending", 0),
                "last_skipped_condition": causal.get("last_skipped_condition", 0),
                "last_skipped_conflict": causal.get("last_skipped_conflict", 0),
                "per_rule": causal.get("per_rule", {}),
            },
            "canonical": {
                "observation_count": len(cs.observations) if cs else 0,
                "entity_count": len(cs.entities) if cs else 0,
                "relation_count": len(cs.relations) if cs else 0,
                "zone_count": len(cs.zones) if cs else 0,
                "archetypes_active": list(cs.archetypes_active) if cs else [],
                "entity_kinds": canonical_entity_kinds,
                "zone_kinds": canonical_zone_kinds,
            },
            "promotion": self._promotion.get_status(),
            "simulator": self._simulator.get_stats(),
            "simulator_promotion": self._sim_promotion.get_status(),
            "planner": self._planner.get_state(),
        }

    def get_deltas(self, recent_n: int = 20) -> list[WorldDelta]:
        return list(self._deltas)[-recent_n:]

    def get_predictions(self) -> list[dict[str, Any]]:
        return self._causal.get_pending_predictions()

    @property
    def promotion(self) -> WorldModelPromotion:
        return self._promotion

    @property
    def simulator(self) -> MentalSimulator:
        return self._simulator

    @property
    def current_state(self) -> WorldState:
        return self._current

    # -- Shadow simulation --------------------------------------------------

    def _run_shadow_simulations(self, deltas: list[WorldDelta]) -> None:
        """Run shadow simulations on recent deltas (Phase 3).

        Picks the most interesting delta (highest confidence, transition events
        preferred over steady-state), simulates forward, and logs the trace.
        Only runs every 3rd tick to stay within budget.
        """
        if self._update_count % 3 != 0:
            return

        interesting = [
            d for d in deltas
            if d.event not in ("entity_moved", "display_content_changed")
        ] or deltas[:1]

        delta = max(interesting, key=lambda d: d.confidence)

        try:
            trace = self._simulator.simulate(
                self._current, delta, source="shadow_tick",
            )
            if trace.depth > 0 and self._update_count % 30 == 0:
                rules_summary = []
                for s in trace.steps:
                    rules_summary.extend(s.applied_rules)
                logger.info(
                    "Shadow simulation: %s → %d steps, conf=%.3f, %.1fms, rules=%s",
                    delta.event, trace.depth, trace.total_confidence,
                    trace.elapsed_ms, rules_summary[:6],
                )
        except Exception:
            logger.debug("Shadow simulation failed", exc_info=True)

    def _run_shadow_planner(self, deltas: list[WorldDelta]) -> None:
        """Run planner in advisory-shadow mode (Phase 7 scaffold)."""
        try:
            self._planner.evaluate(
                world_state=self._current,
                deltas=deltas,
                simulator=self._simulator,
                simulator_promotion_level=self._sim_promotion.level,
                goal_title=self._current.system.active_goal_title,
            )
        except Exception:
            logger.debug("Shadow planner tick failed", exc_info=True)

    # -- Facet readers ------------------------------------------------------

    def _read_physical(self, now: float) -> PhysicalState:
        if not self._scene_tracker:
            return PhysicalState(last_update_ts=now)
        try:
            s = self._scene_tracker.get_state()
            entities = tuple(s.get("entities", []))
            pres = self._read_presence()
            person_count = 1 if pres.get("is_present", False) else 0
            return PhysicalState(
                entities=entities,
                entity_count=s.get("entity_count", 0),
                visible_count=s.get("visible_count", 0),
                stable_count=s.get("stable_count", 0),
                display_surfaces=tuple(s.get("display_surfaces", [])),
                display_content=tuple(s.get("display_content", [])),
                region_visibility=dict(s.get("region_visibility", {})),
                person_count=person_count,
                last_update_ts=now,
            )
        except Exception:
            logger.debug("WorldModel: physical read failed", exc_info=True)
            return PhysicalState(last_update_ts=now)

    _EMOTION_TRUST_THRESHOLD = 0.25

    def _read_user(self, now: float) -> UserState:
        att = self._read_attention()
        pres = self._read_presence()
        speaker = self._read_speaker()

        last_interaction = att.get("last_interaction_time", 0.0)
        since_interaction = now - last_interaction if last_interaction else 999.0

        emo = att.get("user_emotion", "neutral")
        emo_conf = att.get("emotion_confidence", 0.0)
        emo_trusted = emo != "neutral" and emo_conf >= self._EMOTION_TRUST_THRESHOLD

        return UserState(
            present=pres.get("is_present", False),
            presence_confidence=pres.get("confidence", 0.0),
            engagement=att.get("engagement_level", 0.0),
            emotion=emo if emo_trusted else "neutral",
            emotion_confidence=emo_conf if emo_trusted else 0.0,
            emotion_trusted=emo_trusted,
            speaker_name=speaker.get("name", ""),
            speaker_confidence=speaker.get("confidence", 0.0),
            identity_method=speaker.get("identity_method", ""),
            gesture=att.get("gesture", ""),
            gesture_confidence=att.get("gesture_confidence", 0.0),
            seconds_since_last_interaction=since_interaction,
            last_update_ts=now,
        )

    def _read_conversation(self, now: float) -> ConversationState:
        if not self._perc_orch:
            return ConversationState(last_update_ts=now)
        try:
            active_conv = getattr(self._perc_orch, "_active_conversation", None) or {}
            conv_id = active_conv.get("id", "")
            cancelled = active_conv.get("cancelled", False)
            is_active = bool(conv_id) and not cancelled

            last_user = ""
            last_response = getattr(self._perc_orch, "_last_response_text", "") or ""

            ep = self._episodes
            turn_count = 0
            topic = ""
            if ep:
                active_ep = ep.get_active_episode()
                if active_ep:
                    turn_count = active_ep.turn_count() if callable(getattr(active_ep, "turn_count", None)) else len(getattr(active_ep, "turns", []))
                    topic = getattr(active_ep, "topic", "")
                    turns = getattr(active_ep, "turns", [])
                    for t in reversed(turns):
                        if getattr(t, "role", "") == "user":
                            last_user = getattr(t, "text", "")
                            break

            follow_up = getattr(self._perc_orch, "_follow_up_active", False)

            return ConversationState(
                active=is_active,
                topic=topic,
                last_user_text=last_user[:200],
                last_response_text=last_response[:200],
                conversation_id=conv_id,
                turn_count=turn_count,
                follow_up_active=follow_up,
                current_route="",
                last_update_ts=now,
            )
        except Exception:
            logger.debug("WorldModel: conversation read failed", exc_info=True)
            return ConversationState(last_update_ts=now)

    def _read_system(self, now: float) -> SystemState:
        mode = "passive"
        if self._mode_manager:
            try:
                ms = self._mode_manager.get_state() if hasattr(self._mode_manager, "get_state") else {}
                mode = ms.get("mode", getattr(self._mode_manager, "mode", "passive"))
            except Exception:
                pass

        health_score = 1.0
        if self._health_monitor:
            try:
                summary = self._health_monitor.get_summary()
                health_score = summary.get("overall", 1.0)
            except Exception:
                pass

        confidence = 0.0
        if self._analytics:
            try:
                full = self._analytics.get_full_state()
                confidence = full.get("confidence_avg", 0.0)
            except Exception:
                pass

        autonomy_level = 0
        try:
            from consciousness.engine import ConsciousnessEngine
            eng = getattr(self, "_engine_ref", None)
            if eng and hasattr(eng, "_autonomy_orchestrator") and eng._autonomy_orchestrator:
                astatus = eng._autonomy_orchestrator.get_status()
                autonomy_level = astatus.get("level", 0)
        except Exception:
            pass

        goal_title = ""
        goal_kind = ""
        if self._goal_manager:
            try:
                gs = self._goal_manager.get_status()
                focus = gs.get("current_focus")
                if focus:
                    goal_title = focus.get("title", "")
                    goal_kind = focus.get("kind", "")
            except Exception:
                pass

        memory_count = 0
        try:
            from memory.storage import memory_storage
            memory_count = memory_storage.count()
        except Exception:
            pass

        uptime = now - self._boot_ts

        return SystemState(
            mode=mode,
            health_score=health_score,
            confidence=confidence,
            autonomy_level=autonomy_level,
            active_goal_title=goal_title,
            active_goal_kind=goal_kind,
            memory_count=memory_count,
            uptime_s=uptime,
            last_update_ts=now,
        )

    # -- Helper readers -----------------------------------------------------

    def _read_attention(self) -> dict[str, Any]:
        if not self._attention:
            return {}
        try:
            return self._attention.get_state()
        except Exception:
            return {}

    def _read_presence(self) -> dict[str, Any]:
        if not self._presence:
            return {}
        try:
            return self._presence.get_state()
        except Exception:
            return {}

    def _read_speaker(self) -> dict[str, Any]:
        if not self._perc_orch:
            return {}
        try:
            return dict(getattr(self._perc_orch, "_current_speaker", {}) or {})
        except Exception:
            return {}

    # -- Delta detection ----------------------------------------------------

    def _detect_deltas(self, prev: WorldState, curr: WorldState) -> list[WorldDelta]:
        """Compare two consecutive snapshots and produce typed deltas."""
        if prev.version == 0:
            return []
        now = curr.timestamp
        deltas: list[WorldDelta] = []

        # --- Physical ---
        prev_labels = {e.get("label") for e in prev.physical.entities}
        curr_labels = {e.get("label") for e in curr.physical.entities}
        for label in curr_labels - prev_labels:
            deltas.append(WorldDelta(
                facet="physical", event="entity_appeared",
                details={"label": label}, timestamp=now,
            ))
        for label in prev_labels - curr_labels:
            deltas.append(WorldDelta(
                facet="physical", event="entity_disappeared",
                details={"label": label}, timestamp=now,
            ))

        prev_dc = {dc.get("surface_id"): dc.get("content_type")
                    for dc in prev.physical.display_content}
        curr_dc = {dc.get("surface_id"): dc.get("content_type")
                    for dc in curr.physical.display_content}
        for sid in set(prev_dc) | set(curr_dc):
            if prev_dc.get(sid) != curr_dc.get(sid):
                deltas.append(WorldDelta(
                    facet="physical", event="display_content_changed",
                    details={"surface_id": sid,
                             "from": prev_dc.get(sid), "to": curr_dc.get(sid)},
                    timestamp=now,
                ))

        # --- User ---
        if not prev.user.present and curr.user.present:
            deltas.append(WorldDelta(
                facet="user", event="user_arrived",
                details={"confidence": curr.user.presence_confidence},
                timestamp=now,
            ))
        elif prev.user.present and not curr.user.present:
            deltas.append(WorldDelta(
                facet="user", event="user_departed", timestamp=now,
            ))

        _EMOTION_DELTA_MIN_CONFIDENCE = 0.25
        if (prev.user.emotion != curr.user.emotion
                and curr.user.emotion != "neutral"
                and curr.user.emotion_confidence >= _EMOTION_DELTA_MIN_CONFIDENCE):
            deltas.append(WorldDelta(
                facet="user", event="emotion_changed",
                details={"from": prev.user.emotion, "to": curr.user.emotion,
                         "confidence": curr.user.emotion_confidence},
                timestamp=now,
            ))

        eng_prev = prev.user.engagement
        eng_curr = curr.user.engagement
        for threshold in (0.3, 0.5, 0.7):
            crossed_up = eng_prev < threshold <= eng_curr
            crossed_down = eng_prev >= threshold > eng_curr
            if crossed_up or crossed_down:
                deltas.append(WorldDelta(
                    facet="user", event="engagement_crossed_threshold",
                    details={"threshold": threshold,
                             "direction": "up" if crossed_up else "down",
                             "value": round(eng_curr, 2)},
                    timestamp=now,
                ))
                break  # one threshold crossing per tick

        if prev.user.speaker_name != curr.user.speaker_name and curr.user.speaker_name:
            deltas.append(WorldDelta(
                facet="user", event="speaker_changed",
                details={"from": prev.user.speaker_name,
                         "to": curr.user.speaker_name},
                timestamp=now,
            ))

        # --- Conversation ---
        if not prev.conversation.active and curr.conversation.active:
            deltas.append(WorldDelta(
                facet="conversation", event="conversation_started",
                details={"conversation_id": curr.conversation.conversation_id},
                timestamp=now,
            ))
        elif prev.conversation.active and not curr.conversation.active:
            deltas.append(WorldDelta(
                facet="conversation", event="conversation_ended",
                details={"conversation_id": prev.conversation.conversation_id},
                timestamp=now,
            ))

        if (curr.conversation.topic and prev.conversation.topic
                and prev.conversation.topic != curr.conversation.topic):
            deltas.append(WorldDelta(
                facet="conversation", event="topic_changed",
                details={"from": prev.conversation.topic,
                         "to": curr.conversation.topic},
                timestamp=now,
            ))

        if not prev.conversation.follow_up_active and curr.conversation.follow_up_active:
            deltas.append(WorldDelta(
                facet="conversation", event="follow_up_started",
                timestamp=now,
            ))

        # --- System ---
        if prev.system.mode != curr.system.mode:
            deltas.append(WorldDelta(
                facet="system", event="mode_changed",
                details={"from": prev.system.mode, "to": curr.system.mode},
                timestamp=now,
            ))

        if prev.system.health_score >= 0.7 and curr.system.health_score < 0.7:
            deltas.append(WorldDelta(
                facet="system", event="health_degraded",
                details={"from": round(prev.system.health_score, 2),
                         "to": round(curr.system.health_score, 2)},
                timestamp=now,
            ))
        elif prev.system.health_score < 0.7 and curr.system.health_score >= 0.7:
            deltas.append(WorldDelta(
                facet="system", event="health_recovered",
                details={"from": round(prev.system.health_score, 2),
                         "to": round(curr.system.health_score, 2)},
                timestamp=now,
            ))

        return deltas

    # -- Context summary for LLM injection (promotion >= 1) -----------------

    def build_context_summary(self) -> str:
        """Produce a concise situational awareness block for the LLM prompt."""
        ws = self._current
        lines: list[str] = ["Situational Awareness (World Model):"]

        # Physical
        if ws.physical.entity_count > 0:
            vis = ws.physical.visible_count
            ent = ws.physical.entity_count
            displays = len(ws.physical.display_surfaces)
            lines.append(f"  Room: {vis}/{ent} objects visible, {displays} displays")

            content_labels = [
                dc.get("activity_label") or dc.get("content_type", "unknown")
                for dc in ws.physical.display_content
            ]
            if content_labels:
                lines.append(f"  Display activity: {', '.join(content_labels)}")
        else:
            lines.append("  Room: no scene data")

        # User
        if ws.user.present:
            parts = [f"present (engagement {ws.user.engagement:.0%})"]
            if ws.user.emotion != "neutral" and ws.user.emotion_confidence >= 0.25:
                parts.append(f"emotion: {ws.user.emotion} ({ws.user.emotion_confidence:.0%} confidence)")
            elif ws.user.emotion != "neutral":
                parts.append("emotion: untrusted/unavailable")
            if ws.user.speaker_name:
                parts.append(f"speaker: {ws.user.speaker_name}")
            basis = []
            if ws.user.presence_confidence > 0:
                basis.append("voice/presence" if ws.user.speaker_confidence > 0 else "presence")
            if "face" in ws.user.identity_method:
                basis.append("visual")
            elif ws.physical.person_count > 0:
                basis.append("presence (no visual)")
            else:
                basis.append("no visual confirmation")
            parts.append(f"basis: {', '.join(basis)}")
            lines.append(f"  User: {', '.join(parts)}")
        else:
            lines.append("  User: absent")

        # Conversation
        if ws.conversation.active:
            parts = ["active"]
            if ws.conversation.topic:
                parts.append(f"topic: {ws.conversation.topic}")
            if ws.conversation.turn_count:
                parts.append(f"{ws.conversation.turn_count} turns")
            lines.append(f"  Conversation: {', '.join(parts)}")
        else:
            lines.append("  Conversation: idle")

        # Predictions
        pending = self._causal.get_pending_predictions()
        if pending:
            labels = [p["label"] for p in pending[:3]]
            lines.append(f"  Predictions: {', '.join(labels)}")

        # Uncertainty
        high_unc = [f for f, v in ws.uncertainty.items() if v > 0.6]
        if high_unc:
            lines.append(f"  High uncertainty: {', '.join(high_unc)}")

        return "\n".join(lines)
