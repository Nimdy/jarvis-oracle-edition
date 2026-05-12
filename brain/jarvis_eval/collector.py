"""Periodic snapshot collector for the eval sidecar.

The eval sidecar is a read-only observer. This collector reads public
stats APIs from subsystems every COLLECTOR_INTERVAL_S and records
them as EvalSnapshot records. All reads are wrapped in try/except —
a failed subsystem read is logged and skipped.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from jarvis_eval.config import COLLECTOR_INTERVAL_S
from jarvis_eval.contracts import EvalSnapshot

logger = logging.getLogger(__name__)


class EvalCollector:
    """Reads subsystem stats periodically and produces snapshots."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._run_id: str = ""
        self._task: asyncio.Task | None = None
        self._snapshots_collected: int = 0
        self._last_collect_ts: float = 0.0
        self._collect_errors: int = 0

    def start(self, run_id: str, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._run_id = run_id
        target_loop = loop or asyncio.get_event_loop()
        self._task = target_loop.create_task(self._collect_loop())

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def _collect_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(COLLECTOR_INTERVAL_S)
                self.collect_once()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.warning("Eval collector loop error", exc_info=True)

    def collect_once(self) -> list[EvalSnapshot]:
        """Run one collection pass across all sources. Returns snapshots."""
        snapshots: list[EvalSnapshot] = []
        for source, reader in self._source_readers():
            try:
                metrics = reader()
                if metrics:
                    snap = EvalSnapshot(
                        source=source,
                        metrics=metrics,
                        run_id=self._run_id,
                    )
                    snapshots.append(snap)
            except Exception:
                self._collect_errors += 1
                logger.debug("Eval collector: failed to read %s", source, exc_info=True)
        self._snapshots_collected += len(snapshots)
        self._last_collect_ts = time.time()
        return snapshots

    def _source_readers(self) -> list[tuple[str, Any]]:
        """Build list of (source_name, callable) for each subsystem."""
        readers: list[tuple[str, Any]] = []
        engine = self._engine

        readers.append(("engine_state", lambda: engine.get_state()))

        if hasattr(engine, "consciousness") and engine.consciousness:
            cs = engine.consciousness
            readers.append(("consciousness", lambda: cs.get_state().to_dict()))
            readers.append(("dream_artifacts", lambda: self._read_dream_artifacts(cs)))
            readers.append(("cortex", lambda: cs.get_cortex_stats()))
            readers.append(("health", lambda: cs.check_health()))

        readers.append(("contradiction", self._read_contradiction))
        readers.append(("observer", self._read_observer))
        readers.append(("memory", self._read_memory))
        readers.append(("truth_calibration", self._read_truth_calibration))
        readers.append(("belief_graph", self._read_belief_graph))
        readers.append(("mutation_governor", self._read_mutation_governor))
        readers.append(("soul_integrity", self._read_soul_integrity))
        readers.append(("reflective_audit", self._read_reflective_audit))
        readers.append(("quarantine", self._read_quarantine))
        readers.append(("library", self._read_library))

        # -- PVL: expanded snapshot sources --
        readers.append(("policy_telemetry", self._read_policy_telemetry))
        readers.append(("hemisphere", self._read_hemisphere))
        readers.append(("capability_gate", self._read_capability_gate))
        readers.append(("addressee_gate", self._read_addressee_gate))
        readers.append(("autonomy", self._read_autonomy))
        readers.append(("gestation", self._read_gestation))
        readers.append(("identity", self._read_identity))
        readers.append(("study_telemetry", self._read_study_telemetry))
        readers.append(("experience_buffer", self._read_experience_buffer))
        readers.append(("world_model_promotion", self._read_world_model_promotion))
        readers.append(("simulator_promotion", self._read_simulator_promotion))
        readers.append(("scene", self._read_scene))
        readers.append(("speakers", self._read_speakers))
        readers.append(("language", self._read_language))
        readers.append(("matrix", self._read_matrix))
        readers.append(("skills", self._read_skills))
        readers.append(("learning_jobs", self._read_learning_jobs))
        readers.append(("faces", self._read_faces))
        readers.append(("proactive", self._read_proactive))
        readers.append(("system_upgrades", self._read_system_upgrades))
        readers.append(("intentions", self._read_intentions))

        return readers

    # ── Derived snapshot helpers ──────────────────────────────────────

    @staticmethod
    def _read_dream_artifacts(cs: Any) -> dict[str, Any]:
        stats = cs.get_dream_artifact_stats()
        buf = stats.get("buffer", {})
        by_state = buf.get("by_state", {})
        created = max(buf.get("total_created", 0), sum(by_state.values()))
        promoted = max(buf.get("total_promoted", 0), by_state.get("promoted", 0))
        stats["promotion_rate"] = promoted / created if created > 0 else 0.0
        return stats

    # ── Original sources ────────────────────────────────────────────

    def _read_contradiction(self) -> dict[str, Any]:
        from epistemic.contradiction_engine import ContradictionEngine
        return ContradictionEngine.get_instance().get_state()

    def _read_observer(self) -> dict[str, Any]:
        cs = self._engine.consciousness
        if not cs:
            return {}
        observer = getattr(cs, "observer", None)
        if not observer:
            return {}
        result = observer.get_epistemic_stats()
        result["state"] = observer.state.to_dict()
        result["observation_summary"] = observer.get_observation_summary()
        return result

    def _read_memory(self) -> dict[str, Any]:
        from memory.storage import memory_storage
        stats = memory_storage.get_stats()
        stats["associations"] = memory_storage.get_association_stats()
        return stats

    def _read_truth_calibration(self) -> dict[str, Any]:
        from epistemic.calibration import TruthCalibrationEngine
        return TruthCalibrationEngine.get_instance().get_state()

    def _read_belief_graph(self) -> dict[str, Any]:
        from epistemic.belief_graph import BeliefGraph
        graph = BeliefGraph.get_instance()
        return graph.get_state() if graph else {}

    def _read_mutation_governor(self) -> dict[str, Any]:
        cs = getattr(self._engine, "consciousness", None)
        governor = getattr(cs, "governor", None) if cs is not None else None
        if governor is None:
            return {}
        return {
            "mutation_count": governor.mutation_count,
            "rollback_count": governor.rollback_count,
            "mutations_this_hour": governor.mutations_this_hour,
            "total_rejections": governor.total_rejections,
            "active_monitor": governor.get_active_monitor() is not None,
        }

    def _read_soul_integrity(self) -> dict[str, Any]:
        from epistemic.soul_integrity.index import SoulIntegrityIndex
        return SoulIntegrityIndex.get_instance().get_state()

    def _read_reflective_audit(self) -> dict[str, Any]:
        from epistemic.reflective_audit.engine import ReflectiveAuditEngine
        state = ReflectiveAuditEngine.get_instance().get_state()
        reports = state.get("recent_reports", [])
        if reports:
            state["latest_score"] = reports[-1].get("score")
        return state

    def _read_quarantine(self) -> dict[str, Any]:
        from epistemic.quarantine.pressure import get_quarantine_pressure
        return get_quarantine_pressure().get_snapshot()

    def _read_library(self) -> dict[str, Any]:
        from library.source import source_store
        if not source_store:
            return {}
        stats = source_store.get_stats()
        total = stats.get("total", 0)
        by_depth = stats.get("by_content_depth", {})
        substantive = (
            by_depth.get("abstract", 0)
            + by_depth.get("tldr", 0)
            + by_depth.get("full_text", 0)
        )
        stats["substantive_ratio"] = substantive / total if total > 0 else 0.0
        return stats

    # ── PVL: new snapshot sources ───────────────────────────────────

    def _read_policy_telemetry(self) -> dict[str, Any]:
        from policy.telemetry import policy_telemetry
        result = policy_telemetry.snapshot()
        return result if isinstance(result, dict) else {}

    def _read_hemisphere(self) -> dict[str, Any]:
        orch = getattr(self._engine, "_hemisphere_orchestrator", None)
        if orch is None:
            return {}
        state = orch.get_state()
        if not isinstance(state, dict):
            return {}
        result: dict[str, Any] = {}
        hs = state.get("hemisphere_state", {})
        if isinstance(hs, dict):
            result["total_networks"] = hs.get("total_networks", 0)
            result["total_parameters"] = hs.get("total_parameters", 0)
        result["broadcast_slots"] = state.get("broadcast_slots", [])
        bc = state.get("broadcast_slots", [])
        result["broadcast_slots_count"] = len(
            [s for s in bc if isinstance(s, dict) and s.get("score", 0) > 0]
        ) if isinstance(bc, list) else 0
        dist = state.get("distillation", {})
        result["distillation"] = dist if isinstance(dist, dict) else {}
        if isinstance(dist, dict):
            teachers = dist.get("teachers", {})
            result["distillation.total_signals"] = sum(
                t.get("total", 0) for t in teachers.values()
            ) if isinstance(teachers, dict) else dist.get("total_signals", 0)

        import math
        networks = hs.get("networks", []) if isinstance(hs, dict) else []
        has_nan_loss = False
        for net in (networks if isinstance(networks, list) else []):
            if isinstance(net, dict):
                loss = net.get("last_loss")
                if loss is not None and (math.isnan(loss) or math.isinf(loss)):
                    has_nan_loss = True
                    break
        result["has_nan_loss"] = has_nan_loss
        result["all_losses_valid"] = 1.0 if not has_nan_loss else 0.0

        return result

    def _read_capability_gate(self) -> dict[str, Any]:
        try:
            from skills.capability_gate import capability_gate
            result = capability_gate.get_stats()
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}

    def _read_addressee_gate(self) -> dict[str, Any]:
        po = getattr(self._engine, "_perception_orchestrator", None)
        if po is None:
            return {}
        gate = getattr(po, "_addressee_gate", None)
        if gate is None:
            return {}
        result = gate.get_stats()
        return result if isinstance(result, dict) else {}

    def _read_autonomy(self) -> dict[str, Any]:
        orch = getattr(self._engine, "_autonomy_orchestrator", None)
        if orch is None:
            return {}
        status = orch.get_status()
        if not isinstance(status, dict):
            return {}
        pm = status.get("policy_memory", {})
        promo = status.get("promotion", {})
        return {
            "enabled": status.get("enabled"),
            "autonomy_level": status.get("autonomy_level"),
            "queue_size": len(status.get("queue", [])),
            "completed_total": status.get("completed_total", 0),
            "total_wins": pm.get("total_wins", 0) if isinstance(pm, dict) else 0,
            "overall_win_rate": pm.get("overall_win_rate", 0.0) if isinstance(pm, dict) else 0.0,
            "total_outcomes": pm.get("total_outcomes", 0) if isinstance(pm, dict) else 0,
            "eligible_for_l2": promo.get("eligible_for_l2", False) if isinstance(promo, dict) else False,
            "eligible_for_l3": promo.get("eligible_for_l3", False) if isinstance(promo, dict) else False,
        }

    def _read_gestation(self) -> dict[str, Any]:
        gm = getattr(self._engine, "_gestation_manager", None)
        if gm is None:
            return {}
        status = gm.get_status()
        if not isinstance(status, dict):
            return {}
        return {
            "active": status.get("active", False),
            "phase": status.get("phase"),
            "phase_name": status.get("phase_name"),
            "readiness": status.get("readiness", {}),
        }

    def _read_identity(self) -> dict[str, Any]:
        po = getattr(self._engine, "_perception_orchestrator", None)
        if po is None:
            return {}
        fusion = getattr(po, "_identity_fusion", None)
        if fusion is None:
            return {}
        result = fusion.get_status()
        return result if isinstance(result, dict) else {}

    def _read_study_telemetry(self) -> dict[str, Any]:
        try:
            from library.study import get_study_telemetry
            return get_study_telemetry()
        except ImportError:
            return {}

    def _read_experience_buffer(self) -> dict[str, Any]:
        buf = getattr(self._engine, "_experience_buffer", None)
        if buf is None:
            return {}
        if hasattr(buf, "get_stats"):
            result = buf.get_stats()
            if not isinstance(result, dict):
                result = {}
        else:
            result = {"size": len(buf)}
        if hasattr(buf, "_buffer") and buf._buffer:
            rewards = [
                e.reward for e in buf._buffer
                if hasattr(e, "reward") and e.reward is not None
            ]
            if rewards:
                result["avg_reward"] = sum(rewards) / len(rewards)
        return result

    def _read_world_model_promotion(self) -> dict[str, Any]:
        cs = getattr(self._engine, "consciousness", None)
        if cs is None:
            return {}
        wm = getattr(cs, "_world_model", None)
        if wm is None:
            return {}
        promo = getattr(wm, "_promotion", None)
        if promo is None:
            return {}
        result = promo.get_status()
        return result if isinstance(result, dict) else {}

    def _read_simulator_promotion(self) -> dict[str, Any]:
        cs = getattr(self._engine, "consciousness", None)
        if cs is None:
            return {}
        wm = getattr(cs, "_world_model", None)
        if wm is None:
            return {}
        sim_promo = getattr(wm, "_sim_promotion", None)
        if sim_promo is None:
            return {}
        result = sim_promo.get_status()
        return result if isinstance(result, dict) else {}

    def _read_scene(self) -> dict[str, Any]:
        cs = getattr(self._engine, "consciousness", None)
        if cs is None:
            return {}
        st = getattr(cs, "_scene_continuity_module", None)
        if st is None:
            return {}
        state = st.get_state()
        if not isinstance(state, dict):
            return {}
        return {
            "entity_count": state.get("entity_count", 0),
            "visible_count": state.get("visible_count", 0),
            "stable_count": state.get("stable_count", 0),
            "update_count": state.get("update_count", 0),
        }

    def _read_speakers(self) -> dict[str, Any]:
        po = getattr(self._engine, "_perception_orchestrator", None)
        if po is None:
            return {}
        sid = getattr(po, "speaker_id", None)
        if sid is None:
            return {}
        try:
            profiles = sid.get_profiles_summary()
            return {"enrolled_count": len(profiles) if isinstance(profiles, list) else 0}
        except Exception:
            return {}

    def _read_language(self) -> dict[str, Any]:
        try:
            from reasoning.language_corpus import language_corpus
            from reasoning.language_telemetry import language_quality_telemetry

            corpus = language_corpus.get_stats()
            quality = language_quality_telemetry.get_stats()
            runtime_guard = quality.get("runtime_guard", {})
            if not isinstance(runtime_guard, dict):
                runtime_guard = {}
            try:
                from reasoning.language_runtime_bridge import load_runtime_language_policy

                runtime_policy = load_runtime_language_policy()
            except Exception:
                runtime_policy = None
            runtime_rollout_mode = (
                str(getattr(runtime_policy, "rollout_mode", "off") or "off")
                if runtime_policy is not None
                else "off"
            )
            runtime_bridge_enabled = (
                bool(getattr(runtime_policy, "enabled", False))
                if runtime_policy is not None
                else False
            )
            runtime_canary_classes = (
                sorted(list(getattr(runtime_policy, "canary_classes", set()) or []))
                if runtime_policy is not None
                else []
            )
            runtime_mode_code = {"off": 0, "canary": 1, "full": 2}.get(runtime_rollout_mode, 0)

            # Compute provenance coverage for PVL contract
            # Exclude negative examples (intentional bad training data)
            total = int(corpus.get("total_examples", 0))
            by_prov = corpus.get("counts_by_provenance", {})
            by_class = corpus.get("counts_by_response_class", {})
            neg_count = int(by_class.get("negative_example", 0))
            non_neg_total = total - neg_count

            from jarvis_eval.language_scorers import _is_grounded_verdict
            grounded_count = sum(
                v for k, v in by_prov.items() if _is_grounded_verdict(k)
            )
            provenance_coverage = (grounded_count / non_neg_total) if non_neg_total > 0 else 0.0

            result = {
                "total_examples": total,
                "corpus_total_examples": total,
                "corpus_response_classes": corpus.get("counts_by_response_class", {}),
                "corpus_route_class_pairs": corpus.get("counts_by_route_class", {}),
                "corpus_routes": corpus.get("counts_by_route", {}),
                "corpus_recent_examples": corpus.get("recent_examples", []),
                "corpus_last_capture_ts": float(corpus.get("last_capture_ts", 0.0) or 0.0),
                "quality_total_events": int(quality.get("total_events", 0)),
                "native_usage_rate": float(quality.get("native_usage_rate", 0.0) or 0.0),
                "quality_native_usage_rate": float(quality.get("native_usage_rate", 0.0) or 0.0),
                "fail_closed_rate": float(quality.get("fail_closed_rate", 0.0) or 0.0),
                "quality_fail_closed_rate": float(quality.get("fail_closed_rate", 0.0) or 0.0),
                "provenance_coverage": provenance_coverage,
                "quality_counts_by_class": quality.get("counts_by_response_class", {}),
                "quality_counts_by_outcome": quality.get("counts_by_outcome", {}),
                "quality_native_used_by_class": quality.get("native_used_by_class", {}),
                "quality_fail_closed_by_class": quality.get("fail_closed_by_class", {}),
                "quality_last_event_ts": float(quality.get("last_event_ts", 0.0) or 0.0),
                "quality_runtime_guard": runtime_guard,
                "runtime_bridge_enabled": runtime_bridge_enabled,
                "runtime_rollout_mode": runtime_rollout_mode,
                "runtime_rollout_mode_code": runtime_mode_code,
                "runtime_canary_classes": runtime_canary_classes,
                "runtime_guard_total": int(runtime_guard.get("total", 0) or 0),
                "runtime_live_total": int(runtime_guard.get("live_total", 0) or 0),
                "runtime_blocked_by_guard_count": int(runtime_guard.get("blocked_by_guard_count", 0) or 0),
                "runtime_unpromoted_live_attempts": int(runtime_guard.get("unpromoted_live_attempts", 0) or 0),
                "runtime_live_red_classes": int(runtime_guard.get("live_red_classes", 0) or 0),
                "runtime_live_by_class": runtime_guard.get("live_by_class", {}),
                "runtime_blocked_by_class": runtime_guard.get("blocked_by_class", {}),
                "runtime_by_rollout_mode": runtime_guard.get("by_rollout_mode", {}),
                "runtime_by_reason": runtime_guard.get("by_reason", {}),
                "runtime_by_promotion_level": runtime_guard.get("by_promotion_level", {}),
                "runtime_live_rate": float(runtime_guard.get("live_rate", 0.0) or 0.0),
                "runtime_blocked_rate": float(runtime_guard.get("blocked_rate", 0.0) or 0.0),
                "runtime_last_ts": float(runtime_guard.get("last_ts", 0.0) or 0.0),
            }

            # Phase D gate scores
            try:
                from jarvis_eval.language_scorers import (
                    BOUNDED_RESPONSE_CLASSES,
                    compute_gate_scores,
                    classify_gate,
                    classify_gate_reason,
                )
                gate_scores = compute_gate_scores(corpus, quality)
                result["gate_scores"] = gate_scores
                gate_color = classify_gate(gate_scores)
                result["gate_color"] = gate_color
                result["gate_color_code"] = {"red": 0, "yellow": 1, "green": 2}.get(gate_color, 0)
                by_class: dict[str, Any] = {}
                for rc in BOUNDED_RESPONSE_CLASSES:
                    rc_scores = compute_gate_scores(corpus, quality, rc)
                    by_class[rc] = {
                        "scores": rc_scores,
                        "color": classify_gate(rc_scores),
                        "gate_reason": classify_gate_reason(rc_scores),
                    }
                result["gate_scores_by_class"] = by_class
            except Exception:
                pass

            # Phase D promotion governor — evaluate gates and update state
            try:
                from jarvis_eval.language_promotion import LanguagePromotionGovernor
                from jarvis_eval.language_scorers import classify_gate_reason
                gov = LanguagePromotionGovernor.get_instance()
                eval_results = gov.evaluate(corpus, quality)
                result["promotion_summary"] = eval_results

                level_counts = {"shadow": 0, "canary": 0, "live": 0}
                color_counts = {"green": 0, "yellow": 0, "red": 0}
                total_evals = 0
                max_red_streak = 0
                max_green_streak = 0
                red_data_limited_classes = 0
                for row in eval_results.values():
                    if not isinstance(row, dict):
                        continue
                    level = str(row.get("level", "shadow") or "shadow")
                    color = str(row.get("color", "red") or "red")
                    gate_reason = str(row.get("gate_reason", "") or "")
                    if level in level_counts:
                        level_counts[level] += 1
                    if color in color_counts:
                        color_counts[color] += 1
                    if color == "red":
                        if not gate_reason:
                            gate_reason = classify_gate_reason(
                                row.get("scores", {}) if isinstance(row.get("scores"), dict) else {},
                            )
                        if gate_reason == "insufficient_samples":
                            red_data_limited_classes += 1
                    total_evals += int(row.get("total_evaluations", 0) or 0)
                    max_red_streak = max(max_red_streak, int(row.get("consecutive_red", 0) or 0))
                    max_green_streak = max(max_green_streak, int(row.get("consecutive_green", 0) or 0))

                red_total = int(color_counts["red"])
                red_quality_classes = max(0, red_total - int(red_data_limited_classes))

                result["promotion_aggregate"] = {
                    "levels": dict(level_counts),
                    "colors": dict(color_counts),
                    "total_evaluations": int(total_evals),
                    "max_consecutive_red": int(max_red_streak),
                    "max_consecutive_green": int(max_green_streak),
                    "red_quality_classes": int(red_quality_classes),
                    "red_data_limited_classes": int(red_data_limited_classes),
                }
                # Flatten key metrics for PVL snapshot contracts.
                result["promotion_shadow_count"] = int(level_counts["shadow"])
                result["promotion_canary_count"] = int(level_counts["canary"])
                result["promotion_live_count"] = int(level_counts["live"])
                result["promotion_green_classes"] = int(color_counts["green"])
                result["promotion_yellow_classes"] = int(color_counts["yellow"])
                result["promotion_red_classes"] = red_total
                result["promotion_red_quality_classes"] = int(red_quality_classes)
                result["promotion_red_data_limited_classes"] = int(red_data_limited_classes)
                result["promotion_data_limited_classes"] = int(red_data_limited_classes)
                result["promotion_total_evaluations"] = int(total_evals)
                result["promotion_max_consecutive_red"] = int(max_red_streak)
            except Exception:
                pass

            # Phase C harness state (tokenizer/objective/splits/checkpoint/shadow student).
            try:
                from reasoning.language_phasec import get_phasec_status
                result["phase_c"] = get_phasec_status()
            except Exception:
                pass

            return result
        except Exception:
            return {}

    def _read_skills(self) -> dict[str, Any]:
        try:
            from skills.registry import skill_registry

            snapshot = skill_registry.get_status_snapshot()
            if not isinstance(snapshot, dict):
                return {}
            by_status = snapshot.get("by_status", {})
            if not isinstance(by_status, dict):
                by_status = {}
            return {
                "total": int(snapshot.get("total", 0) or 0),
                "verified_count": int(by_status.get("verified", 0) or 0),
                "learning_count": int(by_status.get("learning", 0) or 0),
                "blocked_count": int(by_status.get("blocked", 0) or 0),
                "degraded_count": int(by_status.get("degraded", 0) or 0),
            }
        except Exception:
            return {}

    def _read_learning_jobs(self) -> dict[str, Any]:
        try:
            orch = getattr(self._engine, "_learning_job_orchestrator", None)
            if not orch:
                return {}
            status = orch.get_status()
            if not isinstance(status, dict):
                return {}

            phase_transition_count = 0
            try:
                # Count persisted phase changes across all jobs (active + terminal).
                for job in orch.store.load_all():
                    events = getattr(job, "events", [])
                    if not isinstance(events, list):
                        continue
                    phase_transition_count += sum(
                        1 for ev in events if isinstance(ev, dict) and ev.get("type") == "phase_changed"
                    )
            except Exception:
                phase_transition_count = 0

            return {
                "active_count": int(status.get("active_count", 0) or 0),
                "total_count": int(status.get("total_count", 0) or 0),
                "completed_count": int(status.get("completed_count", 0) or 0),
                "failed_count": int(status.get("failed_count", 0) or 0),
                "phase_transition_count": int(phase_transition_count),
            }
        except Exception:
            return {}

    def _read_faces(self) -> dict[str, Any]:
        po = getattr(self._engine, "_perception_orchestrator", None)
        if po is None:
            return {}
        fid = getattr(po, "face_id", None)
        if fid is None:
            return {}
        try:
            profiles = fid.get_profiles_summary()
            return {"enrolled_count": len(profiles) if isinstance(profiles, list) else 0}
        except Exception:
            return {}

    def _read_matrix(self) -> dict[str, Any]:
        try:
            orch = getattr(self._engine, "_learning_job_orchestrator", None)
            if not orch:
                return {}
            active = 0
            completed = 0
            specialist_count = 0
            for job in orch.get_active_jobs():
                if getattr(job, "matrix_protocol", False):
                    active += 1
            try:
                for job in orch.store.load_all():
                    if getattr(job, "matrix_protocol", False) and getattr(job, "status", "") == "completed":
                        completed += 1
            except Exception:
                completed = 0
            hemi = getattr(self._engine, "_hemisphere_orchestrator", None)
            if hemi:
                state = hemi.get_state()
                hs = state.get("hemisphere_state", {})
                for hp in hs.get("hemispheres", []):
                    if hp.get("specialist_lifecycle") and hp["specialist_lifecycle"] != "none":
                        specialist_count += 1
            return {
                "active_matrix_jobs": active,
                "completed_matrix_jobs": completed,
                "matrix_jobs_observed": active + completed,
                "specialist_count": specialist_count,
            }
        except Exception:
            return {}

    def _read_proactive(self) -> dict[str, Any]:
        try:
            from personality.proactive import proactive_behavior, _governor
            return {
                "pending_question": proactive_behavior.get_pending_question(),
                "dialogue_count": len(proactive_behavior._dialogue_history),
                "asked_count": len(proactive_behavior._asked_questions),
                "recent_interjections": len(_governor._recent_times),
            }
        except Exception:
            return {}

    def _read_system_upgrades(self) -> dict[str, Any]:
        try:
            from self_improve.system_upgrade_report import get_pvl_snapshot
            return get_pvl_snapshot()
        except Exception:
            return {}

    def _read_intentions(self) -> dict[str, Any]:
        """Intention registry truth-layer snapshot (Stage 0)."""
        try:
            from cognition.intention_registry import intention_registry
            status = intention_registry.get_status()
            hist = status.get("outcome_histogram_7d", {}) or {}
            # Graduation readiness — Stage-0 → Stage-1 observability signal.
            # This is an advisory verdict surfaced as PVL for transparency;
            # it does NOT gate any runtime behavior.
            try:
                grad = intention_registry.get_graduation_status()
            except Exception:
                grad = {"gates": [], "registry_gates_passed": False}
            grad_gates = grad.get("gates") or []
            grad_reg_ok = bool(grad.get("registry_gates_passed", False))
            grad_pass_count = sum(
                1 for g in grad_gates if g.get("status") == "pass"
            )
            grad_pending_count = sum(
                1 for g in grad_gates if g.get("status") == "pending"
            )
            grad_unknown_count = sum(
                1 for g in grad_gates if g.get("status") == "unknown"
            )
            return {
                "open_count": int(status.get("open_count", 0) or 0),
                "resolved_buffer_count": int(status.get("resolved_buffer_count", 0) or 0),
                "most_recent_open_intention_age_s": float(
                    status.get("most_recent_open_intention_age_s", 0.0) or 0.0
                ),
                "oldest_open_intention_age_s": float(
                    status.get("oldest_open_intention_age_s", 0.0) or 0.0
                ),
                "total_registered": int(status.get("total_registered", 0) or 0),
                "total_resolved": int(status.get("total_resolved", 0) or 0),
                "total_failed": int(status.get("total_failed", 0) or 0),
                "total_stale": int(status.get("total_stale", 0) or 0),
                "total_abandoned": int(status.get("total_abandoned", 0) or 0),
                "resolved_7d": int(hist.get("resolved", 0) or 0),
                "failed_7d": int(hist.get("failed", 0) or 0),
                "stale_7d": int(hist.get("stale", 0) or 0),
                "abandoned_7d": int(hist.get("abandoned", 0) or 0),
                "errors": int(status.get("errors", 0) or 0),
                "loaded": 1.0 if status.get("loaded") else 0.0,
                # Stage-1 graduation observability (see
                # docs/INTENTION_STAGE_1_DESIGN.md and the PVL contract
                # ``intention_graduation_readiness_reported``).
                "graduation_registry_gates_passed": 1.0 if grad_reg_ok else 0.0,
                "graduation_pass_gates": int(grad_pass_count),
                "graduation_pending_gates": int(grad_pending_count),
                "graduation_unknown_gates": int(grad_unknown_count),
                "graduation_gates_reported": 1.0 if grad_gates else 0.0,
            }
        except Exception:
            return {}

    def get_stats(self) -> dict[str, Any]:
        return {
            "snapshots_collected": self._snapshots_collected,
            "last_collect_ts": self._last_collect_ts,
            "collect_errors": self._collect_errors,
            "interval_s": COLLECTOR_INTERVAL_S,
        }
