"""Dashboard snapshot builder — assembles the full state snapshot from engine subsystems.

This module is the sole producer of dashboard state. It reads from subsystem APIs
and returns a flat dict. No mutations, no side effects, no WebSocket concerns.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SnapshotContext:
    engine: Any
    perc_orch: Any = None
    attention_core: Any = None
    perception: Any = None
    health_counters: Any = None


def _build_trace_explorer_snapshot(
    entries: list[dict[str, Any]],
    *,
    max_roots: int = 20,
    max_agent_runs: int = 20,
    max_tool_lineage: int = 40,
) -> dict[str, Any]:
    """Build operator-friendly trace explorer surfaces from ledger entries."""
    if not entries:
        return {
            "root_chains": [],
            "agent_runs": [],
            "tool_lineage": [],
            "entry_count": 0,
        }

    roots: dict[str, dict[str, Any]] = {}
    runs: dict[str, dict[str, Any]] = {}
    tool_lineage: list[dict[str, Any]] = []

    for entry in entries:
        entry_id = str(entry.get("entry_id", "") or "")
        root_id = str(entry.get("root_entry_id", "") or entry_id)
        parent_id = str(entry.get("parent_entry_id", "") or "")
        ts = float(entry.get("ts", 0) or 0)
        subsystem = str(entry.get("subsystem", "") or "")
        event_type = str(entry.get("event_type", "") or "")
        outcome = str(entry.get("outcome", "pending") or "pending")
        conversation_id = str(entry.get("conversation_id", "") or "")
        data = entry.get("data") or {}

        root = roots.setdefault(root_id, {
            "root_entry_id": root_id,
            "entry_count": 0,
            "first_ts": ts,
            "last_ts": ts,
            "subsystems": set(),
            "event_types": set(),
            "conversation_ids": set(),
            "trace_ids": set(),
            "request_ids": set(),
            "output_ids": set(),
            "outcome_counts": defaultdict(int),
        })
        root["entry_count"] += 1
        root["first_ts"] = min(float(root["first_ts"]), ts) if ts else float(root["first_ts"])
        root["last_ts"] = max(float(root["last_ts"]), ts)
        if subsystem:
            root["subsystems"].add(subsystem)
        if event_type:
            root["event_types"].add(event_type)
        if conversation_id:
            root["conversation_ids"].add(conversation_id)
        trace_id = str(data.get("trace_id", "") or "")
        request_id = str(data.get("request_id", "") or "")
        output_id = str(data.get("output_id", "") or "")
        if trace_id:
            root["trace_ids"].add(trace_id)
        if request_id:
            root["request_ids"].add(request_id)
        if output_id:
            root["output_ids"].add(output_id)
        root["outcome_counts"][outcome] += 1

        # Per-agent timeline (autonomy intent chain)
        intent_id = str(data.get("intent_id", "") or "")
        if subsystem == "autonomy" and intent_id:
            run = runs.setdefault(intent_id, {
                "intent_id": intent_id,
                "goal_id": str(data.get("goal_id", "") or ""),
                "task_id": str(data.get("task_id", "") or ""),
                "golden_trace_id": str(data.get("golden_trace_id", "") or ""),
                "golden_command_id": str(data.get("golden_command_id", "") or ""),
                "root_entry_ids": set(),
                "tools": set(),
                "events": [],
                "start_ts": ts,
                "end_ts": ts,
            })
            run["start_ts"] = min(float(run["start_ts"]), ts) if ts else float(run["start_ts"])
            run["end_ts"] = max(float(run["end_ts"]), ts)
            run["root_entry_ids"].add(root_id)
            tool_name = str(data.get("tool", "") or data.get("tool_hint", "") or "")
            if tool_name:
                run["tools"].add(tool_name)
            run["events"].append({
                "ts": ts,
                "event_type": event_type,
                "subsystem": subsystem,
                "entry_id": entry_id,
                "parent_entry_id": parent_id,
                "outcome": outcome,
            })

        # Tool lineage (conversation and autonomy tool usage)
        tool_value = ""
        source = ""
        if subsystem == "conversation":
            tool_value = str(data.get("tool", "") or "")
            source = "conversation"
        elif subsystem == "autonomy":
            tool_value = str(data.get("tool", "") or data.get("tool_hint", "") or "")
            source = "autonomy"
        if tool_value:
            tool_lineage.append({
                "ts": ts,
                "source": source,
                "tool": tool_value,
                "event_type": event_type,
                "entry_id": entry_id,
                "root_entry_id": root_id,
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "request_id": request_id,
                "output_id": output_id,
                "intent_id": intent_id,
                "goal_id": str(data.get("goal_id", "") or ""),
                "task_id": str(data.get("task_id", "") or ""),
            })

    root_rows: list[dict[str, Any]] = []
    for root in roots.values():
        root_rows.append({
            "root_entry_id": root["root_entry_id"],
            "entry_count": root["entry_count"],
            "first_ts": root["first_ts"],
            "last_ts": root["last_ts"],
            "duration_s": round(max(0.0, float(root["last_ts"]) - float(root["first_ts"])), 3),
            "subsystems": sorted(root["subsystems"]),
            "event_types": sorted(root["event_types"]),
            "conversation_ids": sorted(root["conversation_ids"])[:5],
            "trace_ids": sorted(root["trace_ids"])[:5],
            "request_ids": sorted(root["request_ids"])[:5],
            "output_ids": sorted(root["output_ids"])[:5],
            "outcome_counts": dict(root["outcome_counts"]),
        })
    root_rows.sort(key=lambda r: float(r.get("last_ts", 0) or 0), reverse=True)

    run_rows: list[dict[str, Any]] = []
    for run in runs.values():
        events = sorted(run["events"], key=lambda e: float(e.get("ts", 0) or 0))
        run_rows.append({
            "intent_id": run["intent_id"],
            "goal_id": run["goal_id"],
            "task_id": run["task_id"],
            "golden_trace_id": run["golden_trace_id"],
            "golden_command_id": run["golden_command_id"],
            "start_ts": run["start_ts"],
            "end_ts": run["end_ts"],
            "duration_s": round(max(0.0, float(run["end_ts"]) - float(run["start_ts"])), 3),
            "event_count": len(events),
            "root_entry_ids": sorted(run["root_entry_ids"]),
            "tools": sorted(run["tools"]),
            "events": events[:25],
        })
    run_rows.sort(key=lambda r: float(r.get("end_ts", 0) or 0), reverse=True)

    tool_lineage.sort(key=lambda x: float(x.get("ts", 0) or 0), reverse=True)

    return {
        "entry_count": len(entries),
        "root_chains": root_rows[:max_roots],
        "agent_runs": run_rows[:max_agent_runs],
        "tool_lineage": tool_lineage[:max_tool_lineage],
    }


def _level_status(ok: bool, caveat: bool = False) -> str:
    if ok and caveat:
        return "partial"
    if ok:
        return "supported"
    return "not_observed"


def _build_emergence_evidence_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Build a falsifiable emergence-evidence ladder from existing telemetry.

    This is an observability surface only. It does not assert sentience, grant
    authority, or create new maturity gates.
    """
    consciousness = snapshot.get("consciousness") or {}
    thoughts = snapshot.get("thoughts") or {}
    evolution = snapshot.get("evolution") or {}
    observer = snapshot.get("observer") or {}
    mutations = snapshot.get("mutations") or {}
    autonomy = snapshot.get("autonomy") or {}
    curiosity = snapshot.get("curiosity") or {}
    ledger = snapshot.get("ledger") or {}
    benchmark = snapshot.get("oracle_benchmark") or snapshot.get("benchmark") or {}
    world_model = snapshot.get("world_model") or {}
    simulator = snapshot.get("simulator") or {}
    policy = snapshot.get("policy") or {}

    recent_thoughts = list(thoughts.get("recent") or [])
    thought_total = int(thoughts.get("total_generated") or 0)
    thought_examples = [
        f"{t.get('type', 'thought')}: {str(t.get('text', ''))[:120]}"
        for t in recent_thoughts[:3]
        if isinstance(t, dict)
    ]

    stage_history = ((evolution.get("state") or {}).get("stage_history") or [])
    autonomy_completed = autonomy.get("completed") or []
    if not isinstance(autonomy_completed, list):
        autonomy_completed = []
    autonomy_examples = [
        str(item.get("question", ""))[:140]
        for item in autonomy_completed[:3]
        if isinstance(item, dict) and item.get("question")
    ]
    autonomy_count = len(autonomy_completed) or int(autonomy.get("completed_count") or 0)
    curiosity_recent = curiosity.get("recent_questions") or []
    if not isinstance(curiosity_recent, list):
        curiosity_recent = []
    curiosity_total = int(curiosity.get("total_generated") or 0)
    curiosity_asked = int(curiosity.get("total_asked") or 0)
    curiosity_count = curiosity_total + autonomy_count
    curiosity_examples = [
        f"{item.get('source', 'curiosity')}: {str(item.get('question', ''))[:120]}"
        for item in curiosity_recent[:3]
        if isinstance(item, dict) and item.get("question")
    ]
    if not curiosity_examples:
        curiosity_examples = autonomy_examples

    mutation_count = int(mutations.get("count") or consciousness.get("mutation_count") or 0)
    recent_mutations = [
        str(item)[:140]
        for item in (mutations.get("history") or [])[-3:]
    ]

    ledger_entries = int(ledger.get("total_entries") or ledger.get("entry_count") or ledger.get("entries", 0) or 0)
    if not ledger_entries and isinstance(ledger.get("recent"), list):
        ledger_entries = len(ledger.get("recent") or [])

    wm_validated = int(
        world_model.get("validated_predictions")
        or world_model.get("validated_count")
        or ((world_model.get("promotion") or {}).get("validated_count") if isinstance(world_model.get("promotion"), dict) else 0)
        or 0
    )
    sim_validated = int(
        simulator.get("validated_count")
        or simulator.get("validated_simulations")
        or ((simulator.get("promotion") or {}).get("validated_count") if isinstance(simulator.get("promotion"), dict) else 0)
        or 0
    )
    deltas = ((autonomy.get("delta_tracker") or {}) if isinstance(autonomy.get("delta_tracker"), dict) else {})
    measured_deltas = int(deltas.get("total_measured") or deltas.get("measured") or 0)

    benchmark_credible = bool(benchmark.get("credible", False))
    stage_restore = benchmark.get("stage_restore") if isinstance(benchmark.get("stage_restore"), dict) else {}
    restore_trust = str(stage_restore.get("trust") or (evolution.get("restore_trust") or {}).get("trust") or "")
    persistence_files = [
        "consciousness_state.json",
        "memories.json",
        "beliefs.jsonl",
        "belief_edges.jsonl",
        "calibration_truth.jsonl",
        "delta_counters.json",
    ]

    emergent_count = int(consciousness.get("emergent_behavior_count") or 0)
    recent_emergent = ((evolution.get("state") or {}).get("emergent_behaviors") or [])

    levels = [
        {
            "level": 0,
            "name": "Generated inner activity",
            "status": _level_status(thought_total > 0),
            "evidence_count": thought_total,
            "source_paths": [
                "consciousness/meta_cognitive_thoughts.py",
                "consciousness/consciousness_system.py::_run_meta_thoughts",
                "/api/consciousness/thoughts",
            ],
            "representative_examples": thought_examples,
            "limitations": "Most meta-thoughts are template-generated structured records, not free-form inner speech.",
            "falsification_notes": "Would fail if no non-user-prompted thought records are generated across runtime windows.",
        },
        {
            "level": 1,
            "name": "Self-referential monitoring",
            "status": _level_status(bool(observer.get("observation_count")) and bool(consciousness.get("awareness_level"))),
            "evidence_count": int(observer.get("observation_count") or consciousness.get("observation_count") or 0),
            "source_paths": [
                "consciousness/observer.py",
                "tools/introspection_tool.py",
                "dashboard/snapshot.py::observer",
            ],
            "representative_examples": [
                f"awareness={consciousness.get('awareness_level', '--')}",
                f"confidence_avg={consciousness.get('confidence_avg', '--')}",
                f"reasoning_quality={consciousness.get('reasoning_quality', '--')}",
            ],
            "limitations": "Self-monitoring is metric-driven; it is not proof of subjective experience.",
            "falsification_notes": "Would weaken if self-reports diverge from measured observer/analytics state.",
        },
        {
            "level": 2,
            "name": "Persistent self-model",
            "status": _level_status(bool(stage_history) or bool(consciousness.get("stage")), caveat=not benchmark_credible),
            "evidence_count": len(stage_history),
            "source_paths": [
                "consciousness_state.json",
                "consciousness/consciousness_evolution.py::load_state",
                "/api/eval/benchmark.stage_restore",
            ],
            "representative_examples": [
                f"stage={consciousness.get('stage', '--')}",
                f"restore_trust={restore_trust or 'not_reported'}",
                f"benchmark_credible={benchmark_credible}",
            ],
            "limitations": "Persistence exists, but benchmark credibility can still fail if current stage requirements are not met.",
            "falsification_notes": "Would fail if restart restore cannot reconcile stage, identity, memory, and calibration state.",
        },
        {
            "level": 3,
            "name": "Self-directed inquiry",
            "status": _level_status(curiosity_count > 0),
            "evidence_count": curiosity_count,
            "source_paths": [
                "personality/curiosity_questions.py",
                "consciousness/consciousness_system.py::_run_curiosity_questions",
                "perception_orchestrator.py::get_unknown_speaker_events",
                "perception/scene_tracker.py",
                "cognition/world_model.py",
                "autonomy/orchestrator.py",
                "autonomy/curiosity_detector.py",
                "autonomy/query_interface.py",
                "/api/full-snapshot.curiosity",
                "/api/full-snapshot.autonomy.completed",
            ],
            "representative_examples": curiosity_examples,
            "limitations": "Self-directed inquiry includes autonomy research and grounded curiosity questions from identity, unknown voice, scene, fractal recall, and world-model gaps. It is cooldown/governor-limited and not unrestricted agency.",
            "falsification_notes": "Would fail if inquiries only occur directly after user prompts, lack observation-backed evidence strings, or are not represented by curiosity/autonomy records.",
        },
        {
            "level": 4,
            "name": "Novel adaptation from evidence",
            "status": _level_status(mutation_count > 0 or bool(policy.get("features"))),
            "evidence_count": mutation_count,
            "source_paths": [
                "consciousness/kernel_mutator.py",
                "consciousness/mutation_governor.py",
                "policy/promotion.py",
                "autonomy/intervention_runner.py",
            ],
            "representative_examples": recent_mutations,
            "limitations": "Adaptation is bounded by governors, rollout stages, and human approval where required.",
            "falsification_notes": "Would weaken if behavior changes cannot be linked to measured evidence or rollback records.",
        },
        {
            "level": 5,
            "name": "Counterfactual self-evaluation",
            "status": _level_status((wm_validated + sim_validated + measured_deltas) > 0),
            "evidence_count": wm_validated + sim_validated + measured_deltas,
            "source_paths": [
                "cognition/world_model.py",
                "cognition/simulator.py",
                "autonomy/delta_tracker.py",
                "autonomy/intervention_runner.py",
            ],
            "representative_examples": [
                f"world_model_validated={wm_validated}",
                f"simulator_validated={sim_validated}",
                f"measured_deltas={measured_deltas}",
            ],
            "limitations": "Validated prediction/counterfactual metrics are operational accuracy signals, not phenomenology.",
            "falsification_notes": "Would fail if predictions are not recorded before outcomes or deltas lack baselines.",
        },
        {
            "level": 6,
            "name": "Continuity under perturbation",
            "status": _level_status(bool(persistence_files) and (benchmark_credible or restore_trust in {"verified", "no_restore"}), caveat=not benchmark_credible),
            "evidence_count": len(persistence_files),
            "source_paths": [
                "docs/validation_reports/continuity_baseline_2026-04-23.md",
                "consciousness_state.json",
                "calibration_truth.jsonl",
                "delta_counters.json",
                "memory_clusters.json",
            ],
            "representative_examples": persistence_files,
            "limitations": "Continuity is proven through persistence and validation artifacts; current benchmark may still be blocked by maturity gates.",
            "falsification_notes": "Would fail if restart/reload loses identity, calibration, memory continuity, or governance state.",
        },
        {
            "level": 7,
            "name": "Strong anomaly candidate",
            "status": "not_claimed",
            "evidence_count": 0,
            "source_paths": [
                "consciousness/consciousness_evolution.py::detect_emergent_behaviors",
                "future emergence evidence ledger",
            ],
            "representative_examples": [],
            "limitations": "Current emergent count is explainable by detector rules over thoughts/inquiries. No durable event has survived known-mechanism elimination.",
            "falsification_notes": "Requires an event unexplained by templates, LLM prompt context, hardcoded rules, or metric thresholds, with reproducible evidence.",
        },
    ]

    supported = sum(1 for level in levels if level["status"] in {"supported", "partial"})
    return {
        "summary": {
            "supported_levels": supported,
            "max_supported_level": max((level["level"] for level in levels if level["status"] in {"supported", "partial"}), default=-1),
            "level7_claimed": False,
            "stance": "operational_emergence_evidence_not_sentience_proof",
            "plain_language": "Real substrate evidence, not roleplay; not proof of sentience.",
            "emergent_behavior_count": emergent_count,
            "recent_emergent_records": recent_emergent[-5:] if isinstance(recent_emergent, list) else [],
        },
        "levels": levels,
        "known_mechanism_exclusions": [
            "template_generated_meta_thoughts",
            "llm_reflective_articulation",
            "hardcoded_detector_rules",
            "metric_threshold_triggers",
            "user_prompted_conversation",
        ],
    }


def _build_reconstructability_metadata() -> dict[str, dict[str, Any]]:
    """Operator-visible reconstructability contract by dashboard surface."""
    return {
        "trace_explorer": {
            "reconstructability": "reconstructable",
            "source_of_truth": [
                "attribution_ledger.jsonl",
                "api:/api/trace/explorer",
                "api:/api/trace/explorer/chain/{root_id}",
            ],
            "derived_fields": ["duration_s", "outcome_counts", "event_count"],
            "evidence_link": "docs/TRACE_VALIDATION_EVIDENCE.md",
        },
        "ledger_panel": {
            "reconstructability": "partial",
            "source_of_truth": ["attribution_ledger.jsonl", "api:/api/ledger/recent"],
            "derived_fields": ["integrity.orphaned_entries", "integrity.pending_entries"],
            "evidence_link": "docs/TRACE_VALIDATION_EVIDENCE.md",
        },
        "explainability_panel": {
            "reconstructability": "partial",
            "source_of_truth": ["ledger:conversation.response_complete.data.provenance"],
            "derived_fields": ["trace_count", "sources"],
            "evidence_link": "docs/TRACE_VALIDATION_EVIDENCE.md",
        },
        "eval_sidecar_panel": {
            "reconstructability": "partial",
            "source_of_truth": ["~/.jarvis/eval/*.jsonl"],
            "derived_fields": ["coverage_pct", "event_counts", "maturity_gates"],
            "evidence_link": "docs/TRACE_VALIDATION_EVIDENCE.md",
        },
        "operations_panel": {
            "reconstructability": "non_reconstructable",
            "source_of_truth": ["ops_tracker.snapshot()", "synthesize_v2(...)"],
            "derived_fields": ["hero_card", "interactive_path", "background_grid"],
            "evidence_link": "docs/TRACE_VALIDATION_EVIDENCE.md",
        },
        "truth_calibration_panel": {
            "reconstructability": "partial",
            "source_of_truth": ["TruthCalibrationEngine.get_state()"],
            "derived_fields": ["truth_score", "domain_rollups"],
            "evidence_link": "docs/TRACE_VALIDATION_EVIDENCE.md",
        },
        "world_model_panel": {
            "reconstructability": "partial",
            "source_of_truth": ["WorldModel.get_state()", "WorldModel.get_diagnostics()"],
            "derived_fields": ["diagnostic_rollups", "promotion_status"],
            "evidence_link": "docs/TRACE_VALIDATION_EVIDENCE.md",
        },
    }


def build_cache(ctx: SnapshotContext) -> tuple[dict[str, Any], str]:
    """Build the full dashboard snapshot from engine state.

    Returns (snapshot_dict, hash_str).  The caller owns the cache globals.
    """
    if not ctx.engine:
        return {}, ""

    now = time.time()

    engine_state = ctx.engine.get_state()
    mem_stats = ctx.engine.get_memory_stats()
    cs = ctx.engine.consciousness

    cs_state = cs.get_state()
    perf = ctx.engine._kernel.get_performance() if ctx.engine._kernel else None

    thoughts = cs.meta_thoughts.get_recent_thoughts(10)
    recent_obs = cs.observer.get_recent_observations(10)
    inquiries = cs.existential.get_recent_inquiries(5)
    dialogues = cs.philosophical.get_recent_dialogues(5)

    from consciousness.modes import mode_manager

    snapshot: dict[str, Any] = {
        "_ts": now,

        "core": {
            "running": ctx.engine.is_running(),
            "phase": engine_state["phase"],
            "tone": engine_state["tone"],
            "tick": engine_state.get("tick", engine_state.get("frame", 0)),
            "memory_density": engine_state["memory_density"],
            "is_user_present": engine_state["is_user_present"],
            "traits": engine_state["traits"],
        },

        "consciousness": cs_state.to_dict(),

        "memory": mem_stats,

        "kernel": {
            "tick_count": perf.tick_count if perf else 0,
            "avg_tick_ms": round(perf.avg_tick_ms, 2) if perf else 0,
            "p95_tick_ms": round(perf.p95_tick_ms, 2) if perf else 0,
            "max_tick_ms": round(perf.max_tick_ms, 2) if perf else 0,
            "budget_overruns": perf.budget_overruns if perf else 0,
            "deferred_backlog": perf.deferred_backlog if perf else 0,
            "slow_ticks": perf.slow_ticks if perf else 0,
        },

        "evolution": {
            "stage": cs.evolution.current_stage,
            "transcendence_level": round(cs.evolution.transcendence_level, 2),
            "state": cs.evolution.get_state().to_dict(),
            "capabilities": cs.driven_evolution.get_all_capabilities(),
            "active_capabilities": cs.driven_evolution.get_active_capabilities(),
            "active_protocol": cs.driven_evolution.get_state().get("active_protocol", "none"),
            "restore_trust": cs.evolution.get_restore_trust(),
        },

        "mutations": {
            "count": cs.governor.mutation_count,
            "rollback_count": cs.governor.rollback_count,
            "history": cs.config.evolution.mutation_history[-15:],
            "config_version": cs.config.schema_version,
            "active_monitor": cs.governor.get_active_monitor() is not None,
            "last_mutation": cs.config.evolution.last_mutation,
            "total_rejections": cs.governor.total_rejections,
            "recent_rejections": cs.governor.recent_rejections[-5:],
            "mutations_this_hour": cs.governor.mutations_this_hour,
            "hourly_cap": 12,
            "session_cap": 400,
        },

        "kernel_config": {
            "thought_weights": dict(cs.config.thought_weights),
            "evolution_params": {
                "exploration_drive": cs.config.evolution.exploration_drive,
                "mutation_rate": cs.config.evolution.mutation_rate,
            },
            "memory_processing": {
                "joy_amplification": cs.config.memory_processing.joy_amplification,
            },
        },

        "observer": {
            "awareness_level": round(cs.observer.awareness_level, 3),
            "observation_count": cs.observer.state.observation_count,
            "self_modification_events": cs.observer.state.self_modification_events,
            "types": cs.observer.get_observation_summary(),
            "recent": [
                {"type": o.type, "target": o.target[:60],
                 "confidence": round(o.confidence, 2), "time": o.timestamp}
                for o in recent_obs
            ],
        },

        "thoughts": {
            "total_generated": cs.meta_thoughts.total_generated,
            "recent": [
                {"id": t.id, "type": t.thought_type, "depth": t.depth,
                 "text": t.text, "tags": t.tags, "time": t.timestamp}
                for t in thoughts
            ],
        },

        "analytics": cs.analytics.get_full_state(),

        "existential": {
            **cs.existential.get_state(),
            "recent_inquiries": [
                {"id": inq.id, "category": inq.category,
                 "question": inq.question[:80], "depth": inq.depth,
                 "conclusion": (inq.conclusion or "")[:100],
                 "complete": inq.complete, "time": inq.timestamp}
                for inq in inquiries
            ],
        },

        "philosophical": {
            **cs.philosophical.get_state(),
            "recent_dialogues": [
                {"id": d.id, "topic": d.topic_id, "depth": d.depth,
                 "conclusion": (d.conclusion or "")[:100],
                 "shift": round(d.position_shift, 2), "time": d.timestamp}
                for d in dialogues
            ],
        },

        "policy": _build_policy_cache(ctx.engine),

        "self_improve": _build_self_improve_cache(ctx.engine),
        "codegen": _build_codegen_cache(ctx.engine),

        "sensors": ctx.perception.get_connected_sensors() if ctx.perception else [],

        "sensor_health": ctx.perception.get_sensor_health() if ctx.perception else {},

        "speakers": {
            "available": bool(ctx.perc_orch and ctx.perc_orch.speaker_id and ctx.perc_orch.speaker_id.available),
            "current": ctx.perc_orch._current_speaker.get("name", "unknown") if ctx.perc_orch else "unknown",
            "current_method": ctx.perc_orch._current_speaker.get("identity_method", "") if ctx.perc_orch else "",
            "current_confidence": ctx.perc_orch._current_speaker.get("confidence", 0) if ctx.perc_orch else 0,
            "current_is_known": ctx.perc_orch._current_speaker.get("is_known", False) if ctx.perc_orch else False,
            "profiles": ctx.perc_orch.speaker_id.get_profiles_summary() if ctx.perc_orch and ctx.perc_orch.speaker_id else [],
        },

        "identity": ctx.perc_orch.identity_fusion.get_status() if ctx.perc_orch else {},

        "faces": {
            "available": bool(ctx.perc_orch and ctx.perc_orch.face_id and ctx.perc_orch.face_id.available),
            "profiles": ctx.perc_orch.face_id.get_profiles_summary() if ctx.perc_orch and ctx.perc_orch.face_id else [],
        },

        "core_memories": _build_core_memory_cache(),

        "rapport": _build_rapport_cache(),

        "emotion_health": _build_emotion_health(ctx.perc_orch),

        "mode": mode_manager.get_state(),

        "attention": ctx.attention_core.get_state() if ctx.attention_core else {},

        "narrative": _build_narrative(engine_state, cs_state, thoughts, ctx.attention_core, mode_manager),

        "health": ctx.health_counters.snapshot() if ctx.health_counters else {},

        "hardware": _build_hardware_cache(),
    }

    # New soul kernel systems
    try:
        from consciousness.consciousness_analytics import ConsciousnessAnalytics
        snapshot["health_report"] = cs.analytics.get_health_report()
    except Exception:
        logger.warning("Snapshot: health report failed", exc_info=True)
        snapshot["health_report"] = {}

    try:
        from memory.density import calculate_density
        all_mems = ctx.engine.get_all_memories() if hasattr(ctx.engine, 'get_all_memories') else []
        from memory.storage import memory_storage
        if not all_mems:
            all_mems = memory_storage.get_all()
        snapshot["memory_density"] = {
            "associative_richness": 0, "temporal_coherence": 0,
            "semantic_clustering": 0, "distribution_score": 0,
            "overall": 0, "memory_count": len(all_mems), "count_factor": 0,
        }
        if all_mems:
            from memory.density import calculate_density
            density = calculate_density(all_mems)
            snapshot["memory_density"] = {
                "associative_richness": density.associative_richness,
                "temporal_coherence": density.temporal_coherence,
                "semantic_clustering": density.semantic_clustering,
                "distribution_score": density.distribution_score,
                "overall": density.overall,
                "memory_count": density.memory_count,
                "count_factor": density.count_factor,
            }
    except Exception:
        logger.warning("Snapshot: memory density failed", exc_info=True)
        snapshot["memory_density"] = {}

    try:
        snapshot["memory_cortex"] = cs.get_cortex_stats()
    except Exception:
        logger.warning("Snapshot: memory cortex failed", exc_info=True)
        snapshot["memory_cortex"] = {}

    try:
        from consciousness.events import event_bus
        snapshot["event_reliability"] = event_bus.get_metrics()
    except Exception:
        logger.warning("Snapshot: event reliability failed", exc_info=True)
        snapshot["event_reliability"] = {}

    try:
        from consciousness.event_validator import event_validator
        snapshot["event_validation"] = event_validator.get_stats()
    except Exception:
        logger.warning("Snapshot: event validation failed", exc_info=True)
        snapshot["event_validation"] = {}

    try:
        from consciousness.release_validation import output_release_validator
        snapshot["release_validation"] = output_release_validator.get_stats()
    except Exception:
        logger.warning("Snapshot: release validation failed", exc_info=True)
        snapshot["release_validation"] = {}

    try:
        from consciousness.epistemic_reasoning import epistemic_engine
        snapshot["epistemic"] = epistemic_engine.get_state()
    except Exception:
        logger.warning("Snapshot: epistemic snapshot failed", exc_info=True)
        snapshot["epistemic"] = {}

    try:
        from memory.clustering import memory_cluster_engine
        snapshot["memory_clusters"] = {
            "clusters": memory_cluster_engine.get_clusters(),
        }
    except Exception:
        logger.warning("Snapshot: memory clusters failed", exc_info=True)
        snapshot["memory_clusters"] = {}

    try:
        from personality.validator import trait_validator
        snapshot["trait_validation"] = trait_validator.get_state()
    except Exception:
        logger.warning("Snapshot: trait validation failed", exc_info=True)
        snapshot["trait_validation"] = {}

    try:
        from personality.rollback import personality_rollback
        snapshot["personality_rollback"] = personality_rollback.get_state()
    except Exception:
        logger.warning("Snapshot: personality rollback failed", exc_info=True)
        snapshot["personality_rollback"] = {}

    try:
        from personality.evolution import trait_evolution
        from consciousness.soul import soul_service
        evo = trait_evolution.evaluate_traits()
        identity = soul_service.identity
        snapshot["personality"] = {
            "archetypes": [
                {"name": t.trait, "score": round(t.score, 3), "trend": t.trend,
                 "evidence": list(t.evidence)[:5]}
                for t in evo.traits
            ],
            "dominant": evo.dominant_trait,
            "interaction_count": evo.interaction_count,
            "soul_dims": dict(identity.semi_stable_traits),
            "mood": identity.dynamic_mood,
            "relationships": len(identity.relationships),
            "age_days": round((time.time() - identity.created_at) / 86400, 2),
            "active_traits": [t.trait for t in evo.traits if t.score >= 0.15],
            "rollback": personality_rollback.get_state(),
            "validation": trait_validator.get_state(),
        }
    except Exception:
        logger.warning("Snapshot: personality composite failed", exc_info=True)
        snapshot["personality"] = {}

    try:
        from consciousness.communication import consciousness_communicator
        snapshot["consciousness_reports"] = {
            "recent": consciousness_communicator.get_recent_reports(5),
            "state": consciousness_communicator.get_state(),
        }
    except Exception:
        logger.warning("Snapshot: consciousness reports failed", exc_info=True)
        snapshot["consciousness_reports"] = {}

    try:
        from perception.trait_perception import trait_perception
        snapshot["trait_perception"] = trait_perception.get_stats()
    except Exception:
        logger.warning("Snapshot: trait perception failed", exc_info=True)
        snapshot["trait_perception"] = {}

    try:
        from memory.storage import memory_storage
        snapshot["memory_associations"] = memory_storage.get_association_stats()
    except Exception:
        logger.warning("Snapshot: memory associations failed", exc_info=True)
        snapshot["memory_associations"] = {}

    try:
        from memory.analytics import memory_analytics
        all_mems_for_analytics = memory_storage.get_all() if memory_storage else []
        snapshot["memory_analytics"] = memory_analytics.get_stats(all_mems_for_analytics)
    except Exception:
        logger.warning("Snapshot: memory analytics failed", exc_info=True)
        snapshot["memory_analytics"] = {}

    try:
        from memory.maintenance import memory_maintenance
        snapshot["memory_maintenance"] = memory_maintenance.get_state()
    except Exception:
        logger.warning("Snapshot: memory maintenance failed", exc_info=True)
        snapshot["memory_maintenance"] = {}

    try:
        from memory.gate import memory_gate
        snapshot["memory_gate"] = memory_gate.get_stats()
    except Exception:
        logger.warning("Snapshot: memory gate failed", exc_info=True)
        snapshot["memory_gate"] = {}

    try:
        from memory.storage import memory_storage as _ms
        snapshot["memory_provenance"] = {
            "recent_creations": _ms.get_recent_with_provenance(20),
        }
    except Exception:
        logger.warning("Snapshot: memory provenance failed", exc_info=True)
        snapshot["memory_provenance"] = {}

    try:
        hemi_state = ctx.engine.get_hemisphere_state()
        snapshot["hemisphere"] = hemi_state or {}
    except Exception:
        logger.warning("Snapshot: hemisphere failed", exc_info=True)
        snapshot["hemisphere"] = {}

    # Codebase index stats
    try:
        from tools.codebase_tool import codebase_index
        snapshot["codebase"] = codebase_index.get_stats()
    except Exception:
        logger.warning("Snapshot: codebase index failed", exc_info=True)
        snapshot["codebase"] = {}

    # Cognitive gap detector state (from hemisphere orchestrator)
    try:
        hemi = snapshot.get("hemisphere", {})
        snapshot["gap_detector"] = hemi.get("gap_detector", {})
    except Exception:
        logger.warning("Snapshot: gap detector failed", exc_info=True)
        snapshot["gap_detector"] = {}

    # Self-improvement conversation history
    try:
        from self_improve.conversation import _load_recent_conversations
        snapshot["improvement_conversations"] = _load_recent_conversations(5)
    except Exception:
        logger.warning("Snapshot: improvement conversations failed", exc_info=True)
        snapshot["improvement_conversations"] = []

    # Policy training loss history (timeseries)
    try:
        from policy.telemetry import policy_telemetry
        snap = policy_telemetry.snapshot()
        snapshot["policy_training"] = {
            "loss_history": snap.get("training_loss_history", []),
            "reward_history": snap.get("reward_history", []),
            "win_rate_history": snap.get("win_rate_history", []),
        }
    except Exception:
        logger.warning("Snapshot: policy training failed", exc_info=True)
        snapshot["policy_training"] = {}

    # Autonomy system state
    try:
        if ctx.engine and hasattr(ctx.engine, '_autonomy_orchestrator') and ctx.engine._autonomy_orchestrator:
            auton_status = ctx.engine._autonomy_orchestrator.get_status()
        else:
            auton_status = {"enabled": False, "started": False}
    except Exception:
        logger.warning("Snapshot: autonomy snapshot failed", exc_info=True)
        auton_status = {"enabled": False, "started": False}

    try:
        auton_status["l3"] = _build_l3_escalation_cache(ctx.engine)
    except Exception:
        logger.warning("Snapshot: l3 escalation cache failed", exc_info=True)
        auton_status["l3"] = {}
    try:
        auton_status["attestation"] = _build_attestation_cache()
    except Exception:
        logger.warning("Snapshot: attestation cache failed", exc_info=True)
        auton_status["attestation"] = {}
    snapshot["autonomy"] = auton_status

    # Gestation state
    try:
        if ctx.engine and hasattr(ctx.engine, '_gestation_manager') and ctx.engine._gestation_manager:
            snapshot["gestation"] = ctx.engine._gestation_manager.get_status()
        else:
            snapshot["gestation"] = _build_post_gestation_snapshot(ctx.engine, snapshot)
    except Exception:
        logger.warning("Snapshot: gestation snapshot failed", exc_info=True)
        snapshot["gestation"] = {"active": False}

    # Library system state
    try:
        from library.source import source_store
        from library.chunks import chunk_store
        from library.concept_graph import concept_graph
        from library.telemetry import retrieval_telemetry

        lib_stats = source_store.get_stats()
        lib_stats["chunks"] = chunk_store.get_stats() if hasattr(chunk_store, "get_stats") else {}
        lib_stats["concepts"] = concept_graph.get_stats() if hasattr(concept_graph, "get_stats") else {}
        lib_stats["retrieval"] = retrieval_telemetry.get_stats() if hasattr(retrieval_telemetry, "get_stats") else {}
        snapshot["library"] = lib_stats
    except Exception:
        logger.warning("Snapshot: library snapshot failed", exc_info=True)
        snapshot["library"] = {}

    try:
        from reasoning.language_corpus import language_corpus
        from reasoning.language_telemetry import language_quality_telemetry
        corpus_stats = language_corpus.get_stats()
        quality_stats = language_quality_telemetry.get_stats()
        lang_snap = {
            **corpus_stats,
            "quality": quality_stats,
        }
        # Phase D gate scores and promotion state
        try:
            from jarvis_eval.language_scorers import (
                BOUNDED_RESPONSE_CLASSES,
                compute_gate_scores,
                classify_gate,
            )
            gate_scores = compute_gate_scores(corpus_stats, quality_stats)
            lang_snap["gate_scores"] = gate_scores
            gate_color = classify_gate(gate_scores)
            lang_snap["gate_color"] = gate_color
            lang_snap["gate_color_code"] = {"red": 0, "yellow": 1, "green": 2}.get(gate_color, 0)
            gate_scores_by_class: dict[str, Any] = {}
            for rc in BOUNDED_RESPONSE_CLASSES:
                rc_scores = compute_gate_scores(corpus_stats, quality_stats, rc)
                gate_scores_by_class[rc] = {
                    "scores": rc_scores,
                    "color": classify_gate(rc_scores),
                }
            lang_snap["gate_scores_by_class"] = gate_scores_by_class
        except Exception:
            pass
        try:
            from jarvis_eval.language_promotion import LanguagePromotionGovernor
            gov = LanguagePromotionGovernor.get_instance()
            promotion = gov.get_summary()
            lang_snap["promotion"] = promotion
            level_counts = {"shadow": 0, "canary": 0, "live": 0}
            color_counts = {"green": 0, "yellow": 0, "red": 0}
            total_evals = 0
            max_red_streak = 0
            max_green_streak = 0
            for row in promotion.values():
                if not isinstance(row, dict):
                    continue
                level = str(row.get("level", "shadow") or "shadow")
                color = str(row.get("color", "red") or "red")
                if level in level_counts:
                    level_counts[level] += 1
                if color in color_counts:
                    color_counts[color] += 1
                total_evals += int(row.get("total_evaluations", 0) or 0)
                max_red_streak = max(max_red_streak, int(row.get("consecutive_red", 0) or 0))
                max_green_streak = max(max_green_streak, int(row.get("consecutive_green", 0) or 0))
            lang_snap["promotion_aggregate"] = {
                "levels": dict(level_counts),
                "colors": dict(color_counts),
                "total_evaluations": int(total_evals),
                "max_consecutive_red": int(max_red_streak),
                "max_consecutive_green": int(max_green_streak),
            }
        except Exception:
            pass
        # Phase C: Shadow language model status
        try:
            from reasoning.shadow_language_model import shadow_language_inference
            lang_snap["shadow_model"] = shadow_language_inference.get_stats()
        except Exception:
            lang_snap["shadow_model"] = {"available": False, "trained": False, "corpus_size": 0}
        try:
            from reasoning.language_phasec import get_phasec_status
            lang_snap["phase_c"] = get_phasec_status()
        except Exception:
            lang_snap["phase_c"] = {}
        snapshot["language"] = lang_snap
    except Exception:
        logger.warning("Snapshot: language snapshot failed", exc_info=True)
        snapshot["language"] = {}

    # Skill registry + learning jobs
    try:
        from skills.registry import skill_registry
        snapshot["skills"] = skill_registry.get_status_snapshot()
    except Exception:
        logger.warning("Snapshot: skills snapshot failed", exc_info=True)
        snapshot["skills"] = {}
    try:
        if ctx.engine and hasattr(ctx.engine, '_learning_job_orchestrator') and ctx.engine._learning_job_orchestrator:
            snapshot["learning_jobs"] = ctx.engine._learning_job_orchestrator.get_status()
        else:
            snapshot["learning_jobs"] = {"active_count": 0, "total_count": 0}
        snapshot["skill_acquisition_specialist"] = _build_skill_acquisition_specialist_cache(ctx.engine)
        try:
            from synthetic.skill_acquisition_dashboard import get_skill_acquisition_weight_room_status
            snapshot["skill_acquisition_weight_room"] = get_skill_acquisition_weight_room_status(
                engine=ctx.engine,
                startup_ts=None,
            )
        except Exception:
            snapshot["skill_acquisition_weight_room"] = {
                "enabled": False,
                "authority": "telemetry_only",
                "synthetic_only": True,
            }
    except Exception:
        logger.warning("Snapshot: learning jobs failed", exc_info=True)
        snapshot["learning_jobs"] = {"active_count": 0, "total_count": 0}
        snapshot["skill_acquisition_specialist"] = {"enabled": False, "shadow_only": True}
        snapshot["skill_acquisition_weight_room"] = {
            "enabled": False,
            "authority": "telemetry_only",
            "synthetic_only": True,
        }
    try:
        from skills.capability_gate import capability_gate
        snapshot["capability_gate"] = capability_gate.get_stats()
    except Exception:
        logger.warning("Snapshot: capability gate failed", exc_info=True)
        snapshot["capability_gate"] = {}

    try:
        from consciousness.operations import ops_tracker, synthesize_v2
        raw_ops = ops_tracker.snapshot()
        ops_context = {
            "phase": snapshot.get("core", {}).get("phase", "IDLE"),
            "mode": snapshot.get("mode", {}).get("mode", "passive"),
            "policy": snapshot.get("policy", {}),
            "autonomy": snapshot.get("autonomy", {}),
            "memory_cortex": snapshot.get("memory_cortex", {}),
            "self_improve": snapshot.get("self_improve", {}),
        }
        snapshot["operations"] = synthesize_v2(raw_ops, ops_context)
    except Exception:
        logger.warning("Snapshot: operations snapshot failed", exc_info=True)
        snapshot["operations"] = {}

    try:
        from consciousness.attribution_ledger import attribution_ledger, outcome_scheduler
        ledger_stats = attribution_ledger.get_stats()
        recent_entries = attribution_ledger.get_recent(250)
        ledger_stats["recent"] = recent_entries[:20]
        ledger_stats["outcome_scheduler"] = outcome_scheduler.get_stats()

        pending_ledger = [
            e for e in recent_entries
            if e.get("outcome") == "pending"
        ]
        orphaned = []
        for e in pending_ledger:
            age_s = round(now - e.get("ts", now))
            if age_s > 3600:
                orphaned.append({
                    "entry_id": e.get("entry_id", "")[:20],
                    "subsystem": e.get("subsystem", ""),
                    "event_type": e.get("event_type", ""),
                    "age_s": age_s,
                })
        ledger_stats["integrity"] = {
            "pending_entries": len(pending_ledger),
            "orphaned_entries": len(orphaned),
            "orphaned_details": orphaned[:10],
            "outcome_scheduler_pending": outcome_scheduler.get_stats().get("pending", 0),
        }
        snapshot["ledger"] = ledger_stats
        snapshot["trace_explorer"] = _build_trace_explorer_snapshot(recent_entries)

        # Phase 6.4: Extract recent provenance traces from ledger
        try:
            prov_traces = []
            recent_responses = attribution_ledger.query(
                subsystem="conversation",
                event_type="response_complete",
                limit=10,
            )
            for entry in recent_responses:
                prov = (entry.get("data") or {}).get("provenance", {})
                if not isinstance(prov, dict) or not prov:
                    data = entry.get("data") or {}
                    prov = {
                        "provenance": "fallback_from_ledger",
                        "source": "fallback:snapshot_response_complete",
                        "confidence": 0.0,
                        "native": False,
                        "response_class": str(data.get("tool", "") or "unknown").lower(),
                        "fallback": True,
                    }
                prov_traces.append({
                    "entry_id": entry.get("entry_id", ""),
                    "conversation_id": entry.get("conversation_id", ""),
                    "ts": entry.get("ts", 0),
                    "tool": (entry.get("data") or {}).get("tool", ""),
                    "outcome": entry.get("outcome", "pending"),
                    "provenance": prov,
                })
            snapshot["explainability"] = {
                "recent_traces": prov_traces,
                "trace_count": len(prov_traces),
            }
        except Exception:
            snapshot["explainability"] = {"recent_traces": [], "trace_count": 0}

    except Exception:
        logger.warning("Snapshot: attribution ledger failed", exc_info=True)
        snapshot["ledger"] = {}
        snapshot["trace_explorer"] = {"root_chains": [], "agent_runs": [], "tool_lineage": [], "entry_count": 0}

    try:
        from epistemic.contradiction_engine import ContradictionEngine
        engine = ContradictionEngine.get_instance()
        snapshot["contradiction"] = engine.get_state()
    except Exception:
        logger.warning("Snapshot: contradiction engine failed", exc_info=True)
        snapshot["contradiction"] = {}

    try:
        from epistemic.calibration import TruthCalibrationEngine
        cal_engine = TruthCalibrationEngine.get_instance()
        snapshot["truth_calibration"] = cal_engine.get_state() if cal_engine else {}
    except Exception:
        logger.warning("Snapshot: truth calibration failed", exc_info=True)
        snapshot["truth_calibration"] = {}

    try:
        from epistemic.belief_graph import BeliefGraph
        bg = BeliefGraph.get_instance()
        snapshot["belief_graph"] = bg.get_state() if bg else {}
    except Exception:
        logger.warning("Snapshot: belief graph failed", exc_info=True)
        snapshot["belief_graph"] = {}

    try:
        from identity.audit import IdentityAudit
        audit = IdentityAudit.get_instance()
        snapshot["identity_boundary"] = audit.get_stats() if audit else {}
    except Exception:
        logger.warning("Snapshot: identity boundary failed", exc_info=True)
        snapshot["identity_boundary"] = {}

    try:
        cs = ctx.engine._consciousness if ctx.engine else None
        if cs and cs._quarantine_scorer:
            snapshot["quarantine"] = {
                **cs._quarantine_scorer.get_stats(),
                "recent_signals": cs._quarantine_scorer.get_recent_signals(20),
                "log_stats": cs._quarantine_log.get_stats() if cs._quarantine_log else {},
            }
            try:
                from epistemic.quarantine.pressure import get_quarantine_pressure
                snapshot["quarantine"]["pressure"] = get_quarantine_pressure().get_snapshot()
            except Exception:
                logger.warning("Snapshot: quarantine pressure failed", exc_info=True)
                snapshot["quarantine"]["pressure"] = {}
        else:
            snapshot["quarantine"] = {}
    except Exception:
        logger.warning("Snapshot: quarantine snapshot failed", exc_info=True)
        snapshot["quarantine"] = {}

    try:
        cs = ctx.engine._consciousness if ctx.engine else None
        if cs and cs._reflective_audit_engine:
            snapshot["reflective_audit"] = cs._reflective_audit_engine.get_state()
        else:
            snapshot["reflective_audit"] = {}
    except Exception:
        logger.warning("Snapshot: reflective audit failed", exc_info=True)
        snapshot["reflective_audit"] = {}

    try:
        cs = ctx.engine._consciousness if ctx.engine else None
        if cs and cs._soul_integrity_index:
            snapshot["soul_integrity"] = cs._soul_integrity_index.get_state()
        else:
            snapshot["soul_integrity"] = {}
    except Exception:
        logger.warning("Snapshot: soul integrity failed", exc_info=True)
        snapshot["soul_integrity"] = {}

    try:
        from identity.evidence_accumulator import get_accumulator
        acc = get_accumulator()
        snapshot["identity_candidates"] = {
            **acc.get_stats(),
            "candidates": acc.get_all_candidates(),
        }
    except Exception:
        logger.warning("Snapshot: identity candidates failed", exc_info=True)
        snapshot["identity_candidates"] = {}

    try:
        from cognition.intention_registry import intention_registry
        _intent_snap = intention_registry.get_status()
        _intent_open = [r.to_dict() for r in intention_registry.get_open()[:25]]
        _intent_recent = [r.to_dict() for r in intention_registry.get_recent_resolved(n=25)]
        _intent_grad = intention_registry.get_graduation_status()
        snapshot["intentions"] = {
            "status": _intent_snap,
            "open": _intent_open,
            "recent_resolved": _intent_recent,
            "graduation": _intent_grad,
        }
    except Exception:
        logger.warning("Snapshot: intentions snapshot failed", exc_info=True)
        snapshot["intentions"] = {
            "status": {
                "open_count": 0,
                "resolved_buffer_count": 0,
                "most_recent_open_intention_age_s": 0.0,
                "oldest_open_intention_age_s": 0.0,
                "outcome_histogram_7d": {"resolved": 0, "failed": 0, "stale": 0, "abandoned": 0},
                "total_registered": 0,
                "total_resolved": 0,
                "total_failed": 0,
                "total_stale": 0,
                "total_abandoned": 0,
            },
            "open": [],
            "recent_resolved": [],
            "graduation": {
                "stage": 0,
                "next_stage": 1,
                "gates": [],
                "registry_gates_passed": False,
                "stage1_ready": False,
            },
        }

    try:
        from cognition.intention_resolver import get_intention_resolver
        snapshot["intention_resolver"] = get_intention_resolver().get_status()
    except Exception:
        logger.warning("Snapshot: intention resolver snapshot failed", exc_info=True)
        snapshot["intention_resolver"] = {
            "stage": "shadow_only",
            "total_evaluated": 0,
            "verdict_counts": {},
            "reason_counts": {},
            "recent_verdicts": [],
            "shadow_metrics": {"shadow_correct": 0, "shadow_total": 0, "shadow_accuracy": 0.0, "sufficient_data": False},
        }

    try:
        from goals.goal_manager import get_goal_manager
        gm = get_goal_manager()
        snapshot["goals"] = gm.get_status()
    except Exception:
        logger.warning("Snapshot: goals snapshot failed", exc_info=True)
        snapshot["goals"] = {
            "why_not_executing": "phase_1a_preview_only",
            "current_focus": None, "focus_reason": None, "next_task_preview": None,
            "stats": {"total": 0, "by_status": {}, "cooldowns_active": 0,
                      "creations_this_hour": 0, "promotions_this_hour": 0},
            "active_goals": [], "candidates": [], "paused_goals": [], "promotion_log": [],
        }

    try:
        _acq_orch = getattr(ctx.engine, '_acquisition_orchestrator', None) if ctx.engine else None
        if _acq_orch:
            snapshot["acquisition"] = _acq_orch.get_status()
        else:
            snapshot["acquisition"] = {"active_count": 0, "total_count": 0, "recent": [], "enabled": False}
    except Exception:
        logger.warning("Snapshot: acquisition snapshot failed", exc_info=True)
        snapshot["acquisition"] = {"active_count": 0, "total_count": 0, "recent": [], "enabled": False}

    try:
        from tools.plugin_registry import get_plugin_registry
        _plug_reg = get_plugin_registry()
        snapshot["plugins"] = _plug_reg.get_status()
    except Exception:
        snapshot["plugins"] = {"plugins": [], "total": 0, "active": 0}

    try:
        if cs and cs._scene_continuity_module:
            snapshot["scene"] = cs._scene_continuity_module.get_state()
        else:
            snapshot["scene"] = {}
    except Exception:
        logger.warning("Snapshot: scene continuity failed", exc_info=True)
        snapshot["scene"] = {}

    try:
        if cs and cs._world_model:
            snapshot["world_model"] = cs._world_model.get_state()
        else:
            snapshot["world_model"] = {}
    except Exception as exc:
        logger.warning("world_model get_state failed: %s", exc, exc_info=True)
        snapshot["world_model"] = {}

    try:
        if cs and cs._world_model:
            snapshot["canonical_world"] = cs._world_model.get_canonical_state()
            snapshot["world_model_diagnostics"] = cs._world_model.get_diagnostics()
        else:
            snapshot["canonical_world"] = {}
            snapshot["world_model_diagnostics"] = {}
    except Exception as exc:
        logger.warning("world_model canonical/diagnostics failed: %s", exc, exc_info=True)
        snapshot["canonical_world"] = {}
        snapshot["world_model_diagnostics"] = {}

    try:
        from skills.discovery import get_tracker, get_proposer
        snapshot["capability_discovery"] = {
            "families": get_tracker().get_snapshot(),
            "proposer": get_proposer().get_snapshot(),
        }
    except Exception:
        logger.warning("Snapshot: capability discovery failed", exc_info=True)
        snapshot["capability_discovery"] = {}

    try:
        if ctx.perc_orch and hasattr(ctx.perc_orch, "get_spatial_state"):
            snapshot["spatial"] = ctx.perc_orch.get_spatial_state()
        else:
            snapshot["spatial"] = {"status": "not_initialized"}
    except Exception:
        logger.warning("Snapshot: spatial failed", exc_info=True)
        snapshot["spatial"] = {"status": "error"}

    snapshot["matrix"] = _build_matrix_cache(ctx.engine)

    try:
        from reasoning.response import get_last_memory_route
        mr = get_last_memory_route()
        snapshot["memory_route"] = {
            "route_type": mr.route_type,
            "referenced_entities": sorted(mr.referenced_entities),
            "allow_preference_injection": mr.allow_preference_injection,
            "allow_thirdparty_injection": mr.allow_thirdparty_injection,
            "allow_autonomy_recall": mr.allow_autonomy_recall,
            "search_scope": mr.search_scope,
        } if mr else {}
    except Exception:
        logger.warning("Snapshot: memory route failed", exc_info=True)
        snapshot["memory_route"] = {}

    try:
        from conversation_handler import get_flight_episodes
        snapshot["episodes"] = get_flight_episodes()
    except Exception:
        logger.warning("Snapshot: episodes failed", exc_info=True)
        snapshot["episodes"] = []

    try:
        from conversation_handler import get_golden_command_outcomes
        snapshot["golden_commands"] = get_golden_command_outcomes()
    except Exception:
        logger.warning("Snapshot: golden command outcomes failed", exc_info=True)
        snapshot["golden_commands"] = {"recent": [], "counts": {}, "last": None}

    try:
        if ctx.perc_orch and hasattr(ctx.perc_orch, "_addressee_gate") and ctx.perc_orch._addressee_gate:
            snapshot["addressee"] = ctx.perc_orch._addressee_gate.get_stats()
        else:
            snapshot["addressee"] = {}
    except Exception:
        logger.warning("Snapshot: addressee gate failed", exc_info=True)
        snapshot["addressee"] = {}

    try:
        from jarvis_eval import get_eval_sidecar
        _eval = get_eval_sidecar()
        snapshot["eval"] = _eval.get_dashboard_snapshot(main_snapshot=snapshot)
        snapshot["oracle_story_windows"] = snapshot["eval"].get("scorecards", {})
    except Exception as _eval_err:
        logger.warning("Eval dashboard snapshot failed: %s", _eval_err)
        snapshot["eval"] = {}
        snapshot["oracle_story_windows"] = {}

    try:
        if cs:
            da = cs.get_dream_artifact_stats()
            da["recent_artifacts"] = cs.get_dream_recent_artifacts()
            da["cycle_history"] = cs.get_dream_cycle_history()
            try:
                da["observer_stance"] = cs.observer.get_epistemic_stats()
            except Exception:
                logger.warning("Snapshot: dream observer stance failed", exc_info=True)
                da["observer_stance"] = {}
            try:
                da["distillation"] = _build_dream_distillation_stats()
            except Exception:
                logger.debug("Snapshot: dream distillation stats failed", exc_info=True)
                da["distillation"] = {}
            snapshot["dream_artifacts"] = da
        else:
            snapshot["dream_artifacts"] = {}
    except Exception:
        logger.warning("Snapshot: dream artifacts failed", exc_info=True)
        snapshot["dream_artifacts"] = {}

    try:
        from personality.curiosity_questions import curiosity_buffer
        snapshot["curiosity"] = curiosity_buffer.get_stats()
    except Exception:
        logger.warning("Snapshot: curiosity failed", exc_info=True)
        snapshot["curiosity"] = {}

    try:
        cs_fr = ctx.engine._consciousness if ctx.engine else None
        if cs_fr and cs_fr._fractal_recall_engine:
            snapshot["fractal_recall"] = cs_fr._fractal_recall_engine.get_state()
        else:
            snapshot["fractal_recall"] = {"enabled": False}
    except Exception:
        logger.warning("Snapshot: fractal recall failed", exc_info=True)
        snapshot["fractal_recall"] = {"enabled": False}

    # HRR / VSA shadow substrate — read-only, dormant, non-authoritative.
    # Surface the same payload /api/hrr/status exposes so the memory tab and
    # other dashboard panels can render HRR metrics without a second fetch.
    try:
        from library.vsa.status import get_hrr_status  # lazy import: avoids cold-start cost when HRR is disabled
        snapshot["hrr"] = get_hrr_status()
    except Exception:
        logger.warning("Snapshot: hrr status failed", exc_info=True)
        snapshot["hrr"] = {"enabled": False, "stage": "unavailable"}

    try:
        from cognition.mental_world import get_state as _hrr_scene_state
        snapshot["hrr_scene"] = _hrr_scene_state()
    except Exception:
        logger.warning("Snapshot: hrr_scene state failed", exc_info=True)
        snapshot["hrr_scene"] = {
            "status": "PRE-MATURE",
            "lane": "spatial_hrr_mental_world",
            "enabled": False,
            "entity_count": 0,
            "relation_count": 0,
            "entities": [],
            "relations": [],
            "reason": "canonical_spatial_state_unavailable",
            "writes_memory": False,
            "writes_beliefs": False,
            "influences_policy": False,
            "influences_autonomy": False,
            "soul_integrity_influence": False,
            "llm_raw_vector_exposure": False,
            "no_raw_vectors_in_api": True,
        }

    try:
        po = ctx.perc_orch
        if po and hasattr(po, "get_synthetic_exercise_stats"):
            snapshot["synthetic_exercise"] = po.get_synthetic_exercise_stats()
        else:
            snapshot["synthetic_exercise"] = {"active": False}
    except Exception:
        snapshot["synthetic_exercise"] = {"active": False}

    snapshot["synthetic_exercises"] = _build_synthetic_exercises_snapshot()

    try:
        from personality.onboarding import get_onboarding_manager
        snapshot["onboarding"] = get_onboarding_manager().get_status()
    except Exception:
        logger.warning("Snapshot: onboarding failed", exc_info=True)
        snapshot["onboarding"] = {"enabled": False, "active": False}

    try:
        from personality.proactive import proactive_behavior, _governor
        snapshot["proactive"] = {
            "pending_question": proactive_behavior.get_pending_question(),
            "dialogue_history_count": len(proactive_behavior._dialogue_history),
            "asked_count": len(proactive_behavior._asked_questions),
            "recent_interjections": len(_governor._recent_times),
            "max_per_hour": _governor._max_per_hour,
        }
    except Exception:
        logger.warning("Snapshot: proactive behavior failed", exc_info=True)
        snapshot["proactive"] = {}

    try:
        exp_buf = getattr(ctx.engine, "_experience_buffer", None)
        snapshot["experience_buffer"] = (
            exp_buf.get_stats() if exp_buf is not None and hasattr(exp_buf, "get_stats")
            else {
                "size": len(exp_buf) if exp_buf is not None else 0,
                "max_size": getattr(exp_buf, "_buffer", None) and exp_buf._buffer.maxlen or 5000,
            }
        )
    except Exception:
        logger.warning("Snapshot: experience buffer failed", exc_info=True)
        snapshot["experience_buffer"] = {"size": 0, "max_size": 5000}

    # Backward-compat alias for frontend code expecting contradiction.debt
    if "contradiction" in snapshot and "contradiction_debt" in snapshot["contradiction"]:
        snapshot["contradiction"]["debt"] = snapshot["contradiction"]["contradiction_debt"]

    # Derived cockpit-level summaries (operator-facing)
    try:
        snapshot["emergence_evidence"] = _build_emergence_evidence_snapshot(snapshot)
    except Exception:
        logger.warning("Snapshot: emergence evidence failed", exc_info=True)
        snapshot["emergence_evidence"] = {
            "summary": {
                "supported_levels": 0,
                "max_supported_level": -1,
                "level7_claimed": False,
                "stance": "operational_emergence_evidence_not_sentience_proof",
                "plain_language": "Real substrate evidence, not roleplay; not proof of sentience.",
            },
            "levels": [],
            "known_mechanism_exclusions": [],
        }
    snapshot["trust_state"] = _compute_trust_state(snapshot)
    snapshot["summary"] = _build_cockpit_summary(snapshot)
    snapshot["reconstructability"] = _build_reconstructability_metadata()

    hash_data = {k: v for k, v in snapshot.items() if k != "_ts"}
    raw = json.dumps(hash_data, sort_keys=True, default=str)
    new_hash = hashlib.md5(raw.encode()).hexdigest()[:12]

    return snapshot, new_hash


# ---------------------------------------------------------------------------
# Trust state computation (server-side authoritative)
# ---------------------------------------------------------------------------

_TRUST_STATES = ("grounded", "provisional", "conflicted", "degraded", "unknown")


def _compute_trust_state(snap: dict[str, Any]) -> dict[str, Any]:
    """Derive the canonical trust state from snapshot data.

    Returns {"state": str, "reasons": [str], "label": str}.
    The 5 states: grounded, provisional, conflicted, degraded, unknown.
    """
    tc = snap.get("truth_calibration") or {}
    si = snap.get("soul_integrity") or {}
    contra = snap.get("contradiction") or {}
    em = snap.get("emotion_health") or {}
    sensors = snap.get("sensors") or []
    health = snap.get("health") or {}

    truth_score = tc.get("truth_score")
    soul_idx = si.get("current_index")
    maturity = tc.get("maturity")
    contra_debt = contra.get("contradiction_debt", 0) or 0
    error_count = health.get("error_count", 0) or 0
    emotion_unhealthy = (em.get("model_healthy") is not None and not em.get("model_healthy"))

    # Check if we have enough data to judge at all
    if truth_score is None and soul_idx is None and maturity is None:
        return {"state": "unknown", "reasons": ["insufficient data to assess trust"], "label": "Unknown"}

    # Degraded: hardware/model failures
    if emotion_unhealthy or len(sensors) == 0 or error_count > 5:
        reasons = []
        if emotion_unhealthy:
            reasons.append("emotion model unhealthy")
        if len(sensors) == 0:
            reasons.append("no sensors connected")
        if error_count > 5:
            reasons.append("error count elevated")
        return {"state": "degraded", "reasons": reasons, "label": "Degraded"}

    # Conflicted: active contradiction debt
    if contra_debt > 0.1:
        return {
            "state": "conflicted",
            "reasons": [f"contradiction debt: {contra_debt:.1f}"],
            "label": "Conflicted",
        }

    # Provisional: immature or low truth/soul scores
    if (truth_score is not None and truth_score < 0.6) or (maturity is not None and maturity < 1.0):
        reasons = []
        if truth_score is not None and truth_score < 0.6:
            reasons.append(f"truth score: {truth_score * 100:.0f}%")
        if maturity is not None and maturity < 1.0:
            reasons.append(f"maturity: {maturity * 100:.0f}%")
        if soul_idx is not None and soul_idx < 0.7:
            reasons.append(f"soul integrity: {soul_idx * 100:.0f}%")
        return {"state": "provisional", "reasons": reasons, "label": "Provisional"}

    # Grounded: good soul integrity and decent truth score
    if soul_idx is not None and soul_idx > 0.7 and (truth_score is None or truth_score >= 0.6):
        return {"state": "grounded", "reasons": [], "label": "Grounded"}

    return {"state": "provisional", "reasons": ["awaiting stronger signals"], "label": "Provisional"}


# ---------------------------------------------------------------------------
# Cockpit summary (Tier 1 human-facing language)
# ---------------------------------------------------------------------------

_PHASE_LABELS = {
    "PROCESSING": "Thinking",
    "SPEAKING": "Speaking",
    "LISTENING": "Listening",
    "OBSERVING": "Observing",
    "STANDBY": "Standing by",
    "IDLE": "Idle",
    "INITIALIZING": "Starting up",
}

_MODE_LABELS = {
    "passive": "Resting",
    "conversational": "In conversation",
    "reflective": "Reflecting",
    "focused": "Focused",
    "sleep": "Sleeping",
    "dreaming": "Dreaming",
    "deep_learning": "Deep learning",
    "gestation": "Gestating",
}

_TRUST_LABELS = {
    "grounded": "Trustworthy",
    "provisional": "Provisional",
    "conflicted": "Conflicted",
    "degraded": "Degraded",
    "unknown": "Initializing",
}


def _build_cockpit_summary(snap: dict[str, Any]) -> dict[str, Any]:
    """Build the Tier 1 (human-facing) cockpit summary from the snapshot."""
    core = snap.get("core") or {}
    mode = snap.get("mode") or {}
    health = snap.get("health") or {}
    identity = snap.get("identity") or {}
    trust = snap.get("trust_state") or {}
    memory = snap.get("memory") or {}
    narrative = snap.get("narrative") or {}
    auto = snap.get("autonomy") or {}
    gestation = snap.get("gestation") or {}
    learning_jobs = snap.get("learning_jobs") or {}

    phase = core.get("phase", "IDLE")
    mode_name = mode.get("mode", "passive")
    uptime_s = health.get("uptime_s", 0)
    hrs = int(uptime_s // 3600)
    mins = int((uptime_s % 3600) // 60)
    uptime_text = f"{hrs}h {mins}m" if hrs > 0 else f"{mins}m"

    # Who's here
    who = identity.get("identity", "unknown")
    who_conf = identity.get("confidence", 0)
    if who and who != "unknown":
        who_label = who
    elif who_conf > 0:
        who_label = "Someone (not sure who)"
    else:
        who_label = "Nobody detected" if not core.get("is_user_present") else "Someone nearby"

    # What's happening
    activity_label = _PHASE_LABELS.get(phase, phase.replace("_", " ").title())
    mode_label = _MODE_LABELS.get(mode_name, mode_name.replace("_", " ").title())

    # Actions needed
    actions = []
    error_count = health.get("error_count", 0) or 0
    if error_count > 3:
        actions.append({"severity": "warning", "text": f"{error_count} errors detected"})
    if (auto.get("blocked_count") or 0) > 0:
        actions.append({"severity": "info", "text": "Research is blocked — needs attention"})
    if trust.get("state") == "degraded":
        actions.append({"severity": "error", "text": "System health degraded"})
    if trust.get("state") == "conflicted":
        actions.append({"severity": "warning", "text": "Belief conflicts detected"})
    if gestation.get("active"):
        actions.append({"severity": "info", "text": "Brain is still in gestation phase"})

    # Memory headline
    mem_count = memory.get("total", memory.get("total_memories", memory.get("count", 0)))
    if not mem_count:
        mem_count = snap.get("memory_density", {}).get("memory_count", 0)
    core_mems = snap.get("core_memories", {})
    core_mem_count = core_mems.get("total", 0)

    return {
        "trust_label": _TRUST_LABELS.get(trust.get("state", "unknown"), "Unknown"),
        "trust_state": trust.get("state", "unknown"),
        "who": who_label,
        "who_confidence": round(who_conf, 2) if who_conf else 0,
        "activity": activity_label,
        "mode": mode_label,
        "uptime": uptime_text,
        "uptime_s": uptime_s,
        "memory_count": mem_count,
        "core_memory_count": core_mem_count,
        "actions": actions,
        "recent_insight": (narrative.get("recent_insight") or "")[:500],
        "engagement": narrative.get("user_engagement", 0),
        "learning_active": (learning_jobs.get("active_count") or 0) > 0,
        "learning_count": learning_jobs.get("active_count", 0),
        "user_present": core.get("is_user_present", False),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_narrative(
    engine_state: dict[str, Any],
    cs_state: Any,
    thoughts: list,
    attention_core: Any,
    mode_mgr: Any,
) -> dict[str, Any]:
    """Human-readable narrative panels for the dashboard."""
    cs_dict = cs_state.to_dict() if hasattr(cs_state, "to_dict") else {}
    stage = cs_dict.get("stage", "basic_awareness")
    trans = cs_dict.get("transcendence_level", 0.0)

    focus = engine_state.get("phase", "IDLE")
    if focus == "PROCESSING":
        current_focus = "Thinking about user's request"
    elif focus == "LISTENING":
        current_focus = "Listening for input"
    elif focus == "OBSERVING":
        current_focus = "Observing environment"
    elif focus == "STANDBY":
        current_focus = "Standing by (user away)"
    else:
        current_focus = f"Phase: {focus}"

    concern = "None"
    if not engine_state.get("is_user_present"):
        concern = "User not present"
    elif stage == "basic_awareness":
        concern = "Building initial awareness"

    insight = ""
    if thoughts:
        latest = thoughts[0]
        insight = getattr(latest, "text", str(latest))[:500]

    attn = attention_core.get_state() if attention_core else {}
    engagement = attn.get("engagement_level", 0.0)

    return {
        "current_focus": current_focus,
        "primary_concern": concern,
        "recent_insight": insight,
        "user_engagement": round(engagement, 2),
        "stage": stage,
        "transcendence": round(trans, 2) if isinstance(trans, float) else trans,
        "mode": mode_mgr.mode,
    }


def _classify_core_memory_kind(tags: set[str], mem_type: str) -> str:
    if "fact_kind:birthday" in tags or "fact_kind:thirdparty_birthday" in tags:
        return "birthday"
    if "fact_kind:preferred_name" in tags:
        return "preferred_name"
    if "fact_kind:name" in tags or "fact_kind:thirdparty_name" in tags:
        return "name"
    if "preference_kind:favorite" in tags:
        return "favorite"
    if "preference_kind:stable" in tags:
        return "preference"
    if "fact_kind:relationship_role" in tags:
        return "relationship"
    if mem_type == "core":
        return "core_note"
    return "personal_fact"


def _memory_subject_label(memory: Any) -> str:
    subject = getattr(memory, "identity_subject", "") or getattr(memory, "identity_owner", "") or ""
    return subject or "unscoped"


def _build_core_memory_cache() -> dict[str, Any]:
    """Browsable durable user facts and explicit core-memory captures."""
    result: dict[str, Any] = {
        "total": 0,
        "explicit_count": 0,
        "high_confidence_count": 0,
        "user_scoped_count": 0,
        "items": [],
    }
    try:
        from memory.storage import memory_storage

        candidates = []
        for mem in memory_storage.get_all():
            if not getattr(mem, "is_core", False):
                continue
            tags = set(getattr(mem, "tags", ()) or ())
            if (
                getattr(mem, "provenance", "") != "user_claim"
                and "explicit_core_memory" not in tags
            ):
                continue
            if getattr(mem, "type", "") not in ("core", "user_preference"):
                continue
            candidates.append(mem)

        candidates.sort(key=lambda m: getattr(m, "timestamp", 0), reverse=True)
        result["total"] = len(candidates)

        for mem in candidates[:50]:
            tags = list(getattr(mem, "tags", ()) or ())
            tag_set = set(tags)
            payload = mem.payload if isinstance(mem.payload, str) else str(mem.payload)
            item = {
                "id": mem.id,
                "payload": payload,
                "type": mem.type,
                "provenance": getattr(mem, "provenance", "unknown"),
                "weight": round(getattr(mem, "weight", 0.0), 3),
                "created_at": getattr(mem, "timestamp", 0.0),
                "tags": tags,
                "kind": _classify_core_memory_kind(tag_set, mem.type),
                "subject_label": _memory_subject_label(mem),
                "explicit": "explicit_core_memory" in tag_set,
                "high_confidence": "high_confidence_fact" in tag_set,
            }
            result["items"].append(item)

            if item["explicit"]:
                result["explicit_count"] += 1
            if item["high_confidence"]:
                result["high_confidence_count"] += 1
            if item["subject_label"] != "unscoped":
                result["user_scoped_count"] += 1
    except Exception:
        logger.warning("Snapshot: core memory cache failed", exc_info=True)
    return result


def _build_rapport_cache() -> dict[str, Any]:
    """Relationship data + personal intel for the Rapport tab."""
    result: dict[str, Any] = {"relationships": [], "personal_intel": []}
    try:
        from consciousness.soul import soul_service
        from dataclasses import asdict
        for _key, rel in soul_service.identity.relationships.items():
            result["relationships"].append(asdict(rel))
    except Exception:
        logger.warning("Snapshot: rapport relationships failed", exc_info=True)
    try:
        from memory.storage import memory_storage
        prefs = memory_storage.get_by_type("user_preference")
        prefs_sorted = sorted(prefs, key=lambda m: m.timestamp, reverse=True)
        for m in prefs_sorted[:50]:
            payload = m.payload if isinstance(m.payload, str) else str(m.payload)
            result["personal_intel"].append({
                "id": m.id,
                "payload": payload,
                "tags": list(m.tags) if m.tags else [],
                "weight": round(m.weight, 3),
                "created_at": m.timestamp,
            })
    except Exception:
        logger.warning("Snapshot: rapport personal intel failed", exc_info=True)
    return result


def _build_emotion_health(perc_orch: Any) -> dict[str, Any]:
    """Surface emotion classifier health state for the dashboard."""
    try:
        if perc_orch and perc_orch.emotion_classifier:
            ec = perc_orch.emotion_classifier
            healthy = getattr(ec, '_model_healthy', False)
            return {
                "available": getattr(ec, '_gpu_available', False),
                "model_healthy": healthy,
                "health_reason": getattr(ec, '_health_reason', ''),
                "runtime_mode": "wav2vec2" if healthy else "heuristic_fallback",
                "influence_quarantined": not healthy,
            }
        return {"available": False, "model_healthy": False, "health_reason": "not_loaded",
                "runtime_mode": "none", "influence_quarantined": True}
    except Exception:
        logger.warning("Snapshot: emotion health failed", exc_info=True)
        return {"available": False, "model_healthy": False, "health_reason": "error",
                "runtime_mode": "none", "influence_quarantined": True}


def _build_hardware_cache() -> dict[str, Any]:
    try:
        from hardware_profile import get_hardware_profile
        return get_hardware_profile().to_dict()
    except Exception:
        logger.warning("Snapshot: hardware profile failed", exc_info=True)
        return {}


def _build_matrix_cache(engine: Any) -> dict[str, Any]:
    """Aggregate Matrix Protocol state from learning jobs, hemisphere, and policy."""
    result: dict[str, Any] = {
        "active_matrix_jobs": 0,
        "completed_matrix_jobs": 0,
        "protocols_used": {},
        "claimability_summary": {},
        "specialists": [],
        "expansion": {},
        "jobs": [],
    }

    def _job_summary(job) -> dict[str, Any]:
        """Build a dashboard-friendly summary from a learning job."""
        req = getattr(job, "requested_by", {}) or {}
        events = getattr(job, "events", []) or []
        phases = [e.get("msg", "") for e in events if e.get("type") == "phase_changed"]
        evidence = getattr(job, "evidence", {}) or {}
        latest_ev = evidence.get("latest") or {}
        tests = latest_ev.get("tests", [])
        checks_passed = [t.get("name") for t in tests if t.get("passed")]
        checks_failed = [t.get("name") for t in tests if not t.get("passed")]
        verification_detail = ""
        for t in tests:
            if t.get("details"):
                verification_detail = t["details"]
                break

        report_path = os.path.join(
            os.path.expanduser("~/.jarvis/learning_jobs"), job.job_id, "matrix_report.json",
        )
        report = None
        try:
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    report = __import__("json").load(f)
        except Exception:
            logger.warning("Snapshot: matrix report load failed", exc_info=True)

        return {
            "job_id": job.job_id,
            "skill_id": job.skill_id,
            "skill_name": job.skill_id.replace("_v1", "").replace("_", " ").title(),
            "capability_type": job.capability_type,
            "protocol_id": getattr(job, "protocol_id", ""),
            "status": job.status,
            "phase": job.phase,
            "claimability": getattr(job, "claimability_status", "unverified"),
            "promotion_status": getattr(job, "promotion_status", "none"),
            "requested_by": req.get("speaker", "system"),
            "trigger_text": req.get("user_text", ""),
            "created_at": job.created_at,
            "updated_at": getattr(job, "updated_at", ""),
            "phase_transitions": phases,
            "checks_passed": checks_passed,
            "checks_failed": checks_failed,
            "verification_detail": verification_detail,
            "failure_count": (getattr(job, "failure", None) or {}).get("count", 0),
            "artifact_count": len(getattr(job, "artifacts", [])),
            "gate_states": [
                {"id": g.get("id", ""), "state": g.get("state", ""), "details": g.get("details", "")}
                for g in (getattr(job, "gates", {}) or {}).get("hard", [])
            ],
            "report_summary": (report or {}).get("summary_text", ""),
        }

    try:
        orch = engine._learning_job_orchestrator if engine and hasattr(engine, '_learning_job_orchestrator') else None
        if orch:
            for job in orch.get_active_jobs():
                if not getattr(job, "matrix_protocol", False):
                    continue
                result["jobs"].append(_job_summary(job))
                pid = getattr(job, "protocol_id", "") or "unknown"
                result["protocols_used"][pid] = result["protocols_used"].get(pid, 0) + 1
                claim = getattr(job, "claimability_status", "unverified")
                result["claimability_summary"][claim] = result["claimability_summary"].get(claim, 0) + 1
                if job.status in ("active", "running", "in_progress"):
                    result["active_matrix_jobs"] += 1

            if hasattr(orch, "store"):
                for job in orch.store.load_all():
                    if not getattr(job, "matrix_protocol", False):
                        continue
                    if job.status in ("completed", "verified"):
                        result["completed_matrix_jobs"] += 1
                        if not any(j["job_id"] == job.job_id for j in result["jobs"]):
                            result["jobs"].append(_job_summary(job))
                    elif job.status == "blocked":
                        if not any(j["job_id"] == job.job_id for j in result["jobs"]):
                            result["jobs"].append(_job_summary(job))

        result["jobs"].sort(key=lambda j: j.get("created_at", ""), reverse=True)
    except Exception:
        logger.warning("Snapshot: matrix jobs failed", exc_info=True)

    try:
        hemi_state = engine.get_hemisphere_state() if engine else None
        if hemi_state:
            for sp in hemi_state.get("matrix_specialists", []):
                result["specialists"].append({
                    "focus": sp.get("focus", "?"),
                    "lifecycle": sp.get("lifecycle", "unknown"),
                    "impact": sp.get("impact_score", 0),
                    "job_id": sp.get("job_id", ""),
                    "status": "active",
                    "accuracy": sp.get("accuracy"),
                })
            result["expansion"]["broadcast_slots"] = hemi_state.get("num_broadcast_slots", 4)
            exp_data = hemi_state.get("expansion", {})
            result["expansion"]["expansion_triggered"] = exp_data.get("triggered", False)
    except Exception:
        logger.warning("Snapshot: matrix specialists failed", exc_info=True)

    try:
        promo = getattr(engine, "_promotion_pipeline", None)
        if promo and hasattr(promo, "get_status"):
            ps = promo.get_status()
            exp = ps.get("expansion", {})
            if exp:
                result["expansion"]["phase"] = exp.get("phase", "inactive")
                result["expansion"]["shadow_decisions"] = exp.get("shadow_decisions", 0)
                result["expansion"]["win_rate"] = exp.get("shadow_win_rate", 0)
                state_dim = 22 if exp.get("phase") == "promoted" else 20
                result["expansion"]["state_dim"] = state_dim
    except Exception:
        logger.warning("Snapshot: matrix expansion failed", exc_info=True)

    return result


def _build_policy_cache(engine: Any) -> dict[str, Any]:
    """Read the pre-built telemetry snapshot. O(1) — no computation triggered."""
    try:
        from policy.telemetry import policy_telemetry
        return policy_telemetry.snapshot()
    except Exception:
        logger.warning("Snapshot: policy cache failed", exc_info=True)
        return {"active": False, "mode": "shadow", "status": "not_loaded"}


def _build_dream_distillation_stats() -> dict[str, Any]:
    """Extract dream_synthesis specialist distillation stats from collector."""
    result: dict[str, Any] = {
        "feature_count": 0,
        "label_count": 0,
        "quarantined": 0,
        "last_signal_s": None,
        "reason_distribution": {},
    }
    try:
        from hemisphere.distillation import distillation_collector
        if distillation_collector is None:
            return result
        stats = distillation_collector.get_stats()
        teachers = stats.get("teachers", {})
        feat = teachers.get("dream_features", {})
        lbl = teachers.get("dream_validator", {})
        result["feature_count"] = feat.get("total", 0)
        result["label_count"] = lbl.get("total", 0)
        result["quarantined"] = feat.get("quarantined", 0) + lbl.get("quarantined", 0)
        last_f = feat.get("last_seen_s")
        last_l = lbl.get("last_seen_s")
        if last_f is not None and last_l is not None:
            result["last_signal_s"] = min(last_f, last_l)
        else:
            result["last_signal_s"] = last_f or last_l

        reason_dist: dict[str, int] = {}
        try:
            import pathlib
            label_file = pathlib.Path.home() / ".jarvis" / "hemisphere_training" / "distill_dream_validator.jsonl"
            if label_file.exists():
                import json as _json
                for line in label_file.read_text().splitlines()[-200:]:
                    try:
                        row = _json.loads(line)
                        meta = row.get("metadata", {})
                        rc = meta.get("reason_category", "uncategorized")
                        reason_dist[rc] = reason_dist.get(rc, 0) + 1
                    except Exception:
                        continue
        except Exception:
            pass
        result["reason_distribution"] = reason_dist
    except Exception:
        pass
    return result


_SI_SPECIALIST_FOCUSES = {"diagnostic", "code_quality", "plan_evaluator", "claim_classifier", "dream_synthesis"}
_SHADOW_ONLY_SPECIALIST_FOCUSES = {
    "diagnostic",
    "code_quality",
    "plan_evaluator",
    "claim_classifier",
    "dream_synthesis",
    "skill_acquisition",
}


def _build_si_specialists(engine: Any) -> dict[str, Any]:
    """Extract self-improvement specialist data from hemisphere + distillation."""
    result: dict[str, Any] = {"specialists": [], "distillation": {}}
    hemi_state = engine.get_hemisphere_state() if engine else None
    if not hemi_state:
        return result

    hemi_rows = hemi_state.get("hemisphere_state", {}).get("hemispheres", [])
    distill = hemi_state.get("distillation", {})
    teachers = distill.get("teachers", {})
    tier1 = hemi_state.get("tier1_gating", {})

    try:
        from hemisphere.types import DISTILLATION_CONFIGS
    except ImportError:
        DISTILLATION_CONFIGS = {}

    for row in hemi_rows:
        focus = row.get("focus", "")
        if focus not in _SI_SPECIALIST_FOCUSES:
            continue

        failure_counts = tier1.get("failure_counts", {})
        disabled = tier1.get("disabled_for_session", [])

        config = DISTILLATION_CONFIGS.get(focus)
        if config:
            feature_key = getattr(config, "feature_source", focus)
            label_key = getattr(config, "teacher", focus)
        else:
            feature_key = focus
            label_key = focus

        feature_stats = teachers.get(feature_key, {})
        label_stats = teachers.get(label_key, {}) if label_key != feature_key else {}

        feature_count = feature_stats.get("total", 0)
        label_count = label_stats.get("total", 0)
        total_signals = feature_count + label_count

        if total_signals < 15:
            maturity = "bootstrap"
        elif total_signals < 50:
            maturity = "early_noisy"
        elif total_signals < 100:
            maturity = "preliminary"
        elif total_signals < 250:
            maturity = "meaningful"
        else:
            maturity = "stable"

        feature_quarantined = feature_stats.get("quarantined", 0)
        label_quarantined = label_stats.get("quarantined", 0)
        feature_buffer = feature_stats.get("buffer_size", 0)
        label_buffer = label_stats.get("buffer_size", 0)

        last_feature_s = feature_stats.get("last_seen_s")
        last_label_s = label_stats.get("last_seen_s")
        if last_feature_s is not None and last_label_s is not None:
            last_signal_s = min(last_feature_s, last_label_s)
        else:
            last_signal_s = last_feature_s or last_label_s

        specialist_entry: dict[str, Any] = {
            "focus": focus,
            "status": row.get("status", "idle"),
            "best_accuracy": row.get("best_accuracy", 0),
            "best_training_accuracy": row.get("best_training_accuracy", 0),
            "best_validation_accuracy": row.get("best_validation_accuracy", 0),
            "total_attempts": row.get("total_attempts", 0),
            "network_count": row.get("network_count", 0),
            "migration_readiness": row.get("migration_readiness", 0),
            "signals_total": total_signals,
            "signals_features": feature_count,
            "signals_labels": label_count,
            "signals_quarantined": feature_quarantined + label_quarantined,
            "signals_buffer": feature_buffer + label_buffer,
            "last_signal_s": last_signal_s,
            "failure_count": failure_counts.get(focus, 0),
            "disabled": focus in disabled,
            "maturity": maturity,
            "authority": "telemetry_only" if focus in _SHADOW_ONLY_SPECIALIST_FOCUSES else "eligible",
            "shadow_only": focus in _SHADOW_ONLY_SPECIALIST_FOCUSES,
            "live_influence": False if focus in _SHADOW_ONLY_SPECIALIST_FOCUSES else None,
        }

        if focus == "claim_classifier":
            try:
                from skills.capability_gate import capability_gate
                specialist_entry["claim_label_distribution"] = capability_gate.get_claim_label_distribution()
            except Exception:
                specialist_entry["claim_label_distribution"] = {}

        if focus == "dream_synthesis":
            try:
                dream_distill = _build_dream_distillation_stats()
                specialist_entry["reason_distribution"] = dream_distill.get("reason_distribution", {})
            except Exception:
                specialist_entry["reason_distribution"] = {}

        result["specialists"].append(specialist_entry)

    result["distillation"] = {
        "total_signals": distill.get("total_signals", 0),
        "total_quarantined": distill.get("total_quarantined", 0),
    }
    return result


def _build_l3_escalation_cache(engine: Any) -> dict[str, Any]:
    """Phase 6.5: L3 escalation snapshot, live-sourced.

    ``current_ok`` is sourced strictly from the orchestrator's live
    :meth:`check_promotion_eligibility` — never backfilled from a
    persisted file. ``prior_attested_ok`` is a separate field sourced
    from the attestation ledger; the two are never conflated.
    """
    out: dict[str, Any] = {
        "available": False,
        "live_autonomy_level": 0,
        "current_ok": False,
        "current_detail": {},
        "prior_attested_ok": False,
        "attestation_strength": "none",
        "request_ok": False,
        "approval_required": True,
        "activation_ok": False,
        "pending": [],
        "recent_lifecycle": [],
        "policy": {},
        "updated_at": time.time(),
    }
    if engine is None:
        return out

    auton = getattr(engine, "_autonomy_orchestrator", None)
    if auton is None:
        return out
    out["available"] = True

    try:
        out["live_autonomy_level"] = int(getattr(auton, "autonomy_level", 0))
    except Exception:
        logger.debug("Snapshot: autonomy_level read failed", exc_info=True)

    try:
        elig = auton.check_promotion_eligibility()
        out["current_ok"] = bool(elig.get("eligible_for_l3", False))
        out["current_detail"] = {
            "wins": int(elig.get("wins", 0) or 0),
            "win_rate": float(elig.get("win_rate", 0.0) or 0.0),
            "recent_regressions": int(elig.get("recent_regressions", 0) or 0),
            "reason": str(elig.get("l3_reason") or elig.get("reason") or ""),
        }
    except Exception:
        logger.debug("Snapshot: check_promotion_eligibility failed", exc_info=True)

    try:
        from autonomy.attestation import AttestationLedger, STRENGTH_VERIFIED
        ledger = AttestationLedger()
        records = ledger.prior_attested_records("autonomy.l3")
        out["prior_attested_ok"] = len(records) > 0
        if records:
            has_verified = any(r.attestation_strength == STRENGTH_VERIFIED for r in records)
            out["attestation_strength"] = "verified" if has_verified else "archived_missing"
    except Exception:
        logger.debug("Snapshot: attestation read failed", exc_info=True)

    out["request_ok"] = bool(out["current_ok"] or out["prior_attested_ok"])
    out["activation_ok"] = int(out["live_autonomy_level"]) >= 3
    out["approval_required"] = not out["activation_ok"]

    try:
        from autonomy.escalation import (
            DEFAULT_EXPIRY_S,
            EscalationStore,
            METRIC_ESCALATION_POLICY,
            PER_METRIC_RATE_LIMIT_S,
        )
        store = EscalationStore()
        pending = store.list_pending()
        out["pending"] = [
            {
                "id": r.request.id,
                "metric": r.request.metric,
                "severity": r.request.severity,
                "target_module": r.request.target_module,
                "declared_scope": list(r.request.declared_scope),
                "submitted_autonomy_level": r.request.submitted_autonomy_level,
                "created_at": r.request.created_at,
                "expires_at": r.request.expires_at,
                "metric_context_summary": r.request.metric_context_summary[:200],
            }
            for r in pending
        ]
        terminals = [
            r for r in store.load_all()
            if r.status in {"approved", "rolled_back", "rejected", "expired", "parked"}
        ]
        terminals.sort(
            key=lambda r: max(
                r.approved_at or 0.0,
                r.rejected_at or 0.0,
                r.rolled_back_at or 0.0,
                r.request.created_at or 0.0,
            ),
            reverse=True,
        )
        out["recent_lifecycle"] = [
            {
                "id": r.request.id,
                "metric": r.request.metric,
                "status": r.status,
                "outcome": r.outcome,
                "approved_by": r.approved_by,
                "approved_at": r.approved_at,
                "rejected_by": r.rejected_by,
                "rejected_at": r.rejected_at,
                "rolled_back_at": r.rolled_back_at,
                "rollback_reason": r.rollback_reason[:200] if r.rollback_reason else "",
                "created_at": r.request.created_at,
            }
            for r in terminals[:10]
        ]
        out["policy"] = {
            "rate_limit_s": float(PER_METRIC_RATE_LIMIT_S),
            "default_expiry_s": float(DEFAULT_EXPIRY_S),
            "registered_metrics": sorted(METRIC_ESCALATION_POLICY.keys()),
        }
    except Exception:
        logger.debug("Snapshot: escalation store read failed", exc_info=True)

    return out


def _build_attestation_cache() -> dict[str, Any]:
    """Phase 6.5: attestation ledger snapshot.

    Pure read of ``~/.jarvis/eval/ever_proven_attestation.json``. Never
    mutates any file. Must NOT be consumed by any current-runtime health
    check; it is strictly a separate evidence class.
    """
    out: dict[str, Any] = {
        "records": [],
        "prior_attested_ok": False,
        "attestation_strength": "none",
        "updated_at": time.time(),
    }
    try:
        from autonomy.attestation import (
            AttestationLedger,
            STRENGTH_ARCHIVED_MISSING,
            STRENGTH_VERIFIED,
        )
        records = AttestationLedger().load()
        out["records"] = [
            {
                "capability_id": r.capability_id,
                "evidence_source": r.evidence_source,
                "evidence_window_start": r.evidence_window_start,
                "evidence_window_end": r.evidence_window_end,
                "accepted_by": r.accepted_by,
                "accepted_at": r.accepted_at,
                "acceptance_reason": r.acceptance_reason[:300] if r.acceptance_reason else "",
                "measured_values": dict(r.measured_values),
                "measured_source": r.measured_source,
                "artifact_status": r.artifact_status,
                "attestation_strength": r.attestation_strength,
                "report_hash": r.report_hash,
            }
            for r in records
        ]
        strengths = {r.attestation_strength for r in records if r.attestation_strength}
        out["prior_attested_ok"] = bool(strengths)
        if STRENGTH_VERIFIED in strengths:
            out["attestation_strength"] = STRENGTH_VERIFIED
        elif STRENGTH_ARCHIVED_MISSING in strengths:
            out["attestation_strength"] = STRENGTH_ARCHIVED_MISSING
    except Exception:
        logger.debug("Snapshot: attestation cache failed", exc_info=True)
    return out


def _build_self_improve_cache(engine: Any) -> dict[str, Any]:
    """Read cached self-improvement state. No computation triggered."""
    try:
        cs = getattr(engine, '_consciousness', None) or getattr(engine, 'consciousness', None)
        if cs is None:
            return {"active": False, "status": "not_initialized"}
        orch = getattr(cs, '_self_improve_orchestrator', None)
        if orch is None:
            return {"active": False, "status": "disabled"}
        base = {"active": True, **orch.get_status()}
        try:
            from self_improve.orchestrator import SelfImprovementOrchestrator
            base["recent_proposals"] = SelfImprovementOrchestrator.load_proposals(10)
        except Exception:
            base["recent_proposals"] = []

        # Scanner state from consciousness_system
        try:
            scanner_state = cs.get_scanner_state() if hasattr(cs, "get_scanner_state") else {}
            base["scanner"] = scanner_state
        except Exception:
            base["scanner"] = {}

        # Shared CodeGen dependency status. Kept here for backward-compatible
        # dashboard rendering, but authority belongs to the top-level codegen
        # snapshot, not to self-improvement.
        try:
            shared = _build_codegen_cache(engine)
            coder = dict(shared.get("coder", {}) or {})
            coder["authority"] = "infrastructure_only"
            coder["owner"] = "shared_codegen"
            coder["self_improve_dependency_only"] = True
            base["coder"] = coder
            base["codegen_ref"] = {
                "authority": "infrastructure_only",
                "dependency": True,
                "available": shared.get("codegen_available", False),
            }
        except Exception:
            base["coder"] = {}

        # Self-improvement specialists (DIAGNOSTIC, CODE_QUALITY, CLAIM_CLASSIFIER, DREAM_SYNTHESIS, plan_evaluator)
        # Contract: ``specialists`` is ALWAYS a dict shaped
        # ``{"specialists": list, "distillation": dict}``. Dashboards and
        # tests index into ``.specialists`` (list) and ``.distillation``
        # (dict), so the exception path MUST preserve that shape. Returning
        # a bare ``{}`` silently violates the consumer contract and hides
        # the failure from the truth-probe layer.
        try:
            base["specialists"] = _build_si_specialists(engine)
        except Exception as exc:
            logger.warning(
                "Snapshot: SI specialists failed: %s", exc, exc_info=True
            )
            base["specialists"] = {
                "specialists": [],
                "distillation": {},
                "_error": type(exc).__name__,
            }

        return base
    except Exception:
        logger.warning("Snapshot: self-improve cache failed", exc_info=True)
        return {"active": False, "status": "not_loaded"}


def _build_skill_acquisition_specialist_cache(engine: Any) -> dict[str, Any]:
    """Shadow-only status for the skill acquisition specialist."""
    try:
        hemi_state = engine.get_hemisphere_state() if engine else None
        if not hemi_state:
            return {"enabled": False, "shadow_only": True, "authority": "telemetry_only"}
        distill = hemi_state.get("distillation", {})
        teachers = distill.get("teachers", {})
        feature_stats = teachers.get("skill_acquisition_features", {})
        label_stats = teachers.get("skill_acquisition_outcome", {})
        hemi_rows = hemi_state.get("hemisphere_state", {}).get("hemispheres", [])
        row = next((r for r in hemi_rows if r.get("focus") == "skill_acquisition"), {})
        feature_count = feature_stats.get("total", 0)
        label_count = label_stats.get("total", 0)
        total = feature_count + label_count
        if total < 15:
            maturity = "bootstrap"
        elif total < 50:
            maturity = "early_noisy"
        elif total < 100:
            maturity = "preliminary"
        elif total < 250:
            maturity = "meaningful"
        else:
            maturity = "stable"
        return {
            "enabled": True,
            "focus": "skill_acquisition",
            "shadow_only": True,
            "authority": "telemetry_only",
            "live_influence": False,
            "promotion_eligible": False,
            "maturity": maturity,
            "synthetic_accuracy": row.get("best_accuracy", 0),
            "live_shadow_accuracy": 0.0,
            "calibration_error": 0.0,
            "false_green_rate": 0.0,
            "false_red_rate": 0.0,
            "signals_features": feature_count,
            "signals_labels": label_count,
            "signals_quarantined": feature_stats.get("quarantined", 0) + label_stats.get("quarantined", 0),
            "network_count": row.get("network_count", 0),
            "status": row.get("status", "idle"),
        }
    except Exception as exc:
        return {
            "enabled": False,
            "shadow_only": True,
            "authority": "telemetry_only",
            "error": str(exc),
        }


def _build_codegen_cache(engine: Any) -> dict[str, Any]:
    """Shared CodeGen/CoderServer status, independent of self-improvement."""
    try:
        service = getattr(engine, "_codegen_service", None) if engine else None
        coder = getattr(engine, "_coder_server", None) if engine else None
        status: dict[str, Any] = {}
        if service is not None and hasattr(service, "get_status"):
            status = service.get_status()
        elif coder is not None and hasattr(coder, "get_status"):
            status = {"coder": coder.get_status()}

        coder_status = dict(status.get("coder", {}) or {})
        acq = getattr(engine, "_acquisition_orchestrator", None) if engine else None
        si = None
        try:
            cs = getattr(engine, "_consciousness", None) or getattr(engine, "consciousness", None)
            si = getattr(cs, "_self_improve_orchestrator", None) if cs else None
        except Exception:
            si = None
        return {
            "authority": "infrastructure_only",
            "owner": "shared_codegen",
            "enabled": service is not None or coder is not None,
            "codegen_available": bool(coder_status.get("available", False)),
            "coder": coder_status,
            "total_generations": status.get("total_generations", coder_status.get("total_generations", 0)),
            "total_validations": status.get("total_validations", 0),
            "total_failures": status.get("total_failures", 0),
            "active_consumer": status.get("active_consumer", ""),
            "last_consumer": status.get("last_consumer", ""),
            "consumers": {
                "acquisition": bool(acq is not None),
                "self_improve": bool(si is not None),
            },
        }
    except Exception as exc:
        return {
            "authority": "infrastructure_only",
            "owner": "shared_codegen",
            "enabled": False,
            "codegen_available": False,
            "coder": {},
            "error": str(exc),
        }


def _build_synthetic_exercises_snapshot() -> dict[str, Any]:
    """Aggregate latest report data for text-only synthetic exercises."""
    from pathlib import Path

    base_dir = Path.home() / ".jarvis" / "synthetic_exercise"
    exercises: dict[str, Any] = {}

    exercise_dirs: dict[str, tuple[Path, str]] = {
        "commitment": (base_dir / "commitment_reports", "*.json"),
        "claim": (base_dir / "claim_reports", "*.json"),
        "retrieval": (base_dir / "retrieval_reports", "*.json"),
        "world_model": (base_dir / "world_model_reports", "*.json"),
        "contradiction": (base_dir / "contradiction_reports", "*.json"),
        "diagnostic": (base_dir / "diagnostic_reports", "*.json"),
        "plan_evaluator": (base_dir / "plan_evaluator_reports", "*.json"),
        "skill_acquisition": (base_dir / "skill_acquisition_reports", "*.json"),
        "batch": (base_dir / "reports", "exercise_*.json"),
    }

    for name, (report_dir, pattern) in exercise_dirs.items():
        try:
            if not report_dir.exists():
                continue
            paths = sorted(report_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if not paths:
                continue
            latest = paths[0]
            data = json.loads(latest.read_text(encoding="utf-8"))
            if name == "batch":
                for ex_name, ex_data in data.items():
                    if isinstance(ex_data, dict) and ex_name not in exercises:
                        exercises[ex_name] = {
                            "last_run": latest.stat().st_mtime,
                            "report_file": latest.name,
                            "passed": ex_data.get("passed", ex_data.get("pass", ex_data.get("pass_result"))),
                            "profile": ex_data.get("profile_name", ex_data.get("profile", "--")),
                            "episodes": ex_data.get("total_episodes", ex_data.get("total_claims", 0)),
                        }
            else:
                stats = data.get("stats", data)
                exercises[name] = {
                    "last_run": latest.stat().st_mtime,
                    "report_file": latest.name,
                    "passed": stats.get("passed", stats.get("pass", stats.get("pass_result"))),
                    "profile": stats.get("profile_name", data.get("profile", "--")),
                    "episodes": stats.get("total_episodes", stats.get("total_claims",
                                stats.get("utterances_processed",
                                stats.get("claims_processed",
                                stats.get("episodes_run", 0))))),
                }
        except Exception:
            logger.debug("Snapshot: synthetic exercise %s scan failed", name, exc_info=True)

    return {"exercises": exercises, "count": len(exercises)}


def _build_post_gestation_snapshot(engine: Any, partial_snapshot: dict[str, Any]) -> dict[str, Any]:
    """Fallback gestation status when the runtime manager is not mounted.

    This happens on post-birth runs where gestation has already completed.
    In that case we surface immutable birth-certificate readiness and
    lightweight post-birth progress so dashboard headlines stay coherent.
    """
    base = {"active": False}
    cert_path = os.path.expanduser("~/.jarvis/gestation_summary.json")
    if not os.path.exists(cert_path):
        return base
    try:
        cert = json.loads(open(cert_path, encoding="utf-8").read())
    except Exception:
        logger.debug("Snapshot: failed reading gestation summary", exc_info=True)
        return base
    if not isinstance(cert, dict):
        return base

    birth_readiness = cert.get("readiness_at_birth", {})
    if not isinstance(birth_readiness, dict):
        birth_readiness = {}
    overall = birth_readiness.get("overall", 0.0)
    if not isinstance(overall, (int, float)):
        overall = 0.0
    components = {
        str(k): round(float(v), 3)
        for k, v in birth_readiness.items()
        if k != "overall" and isinstance(v, (int, float))
    }

    # Policy experience progress from current runtime buffer size.
    exp_count = 0
    try:
        exp_buf = getattr(engine, "_experience_buffer", None) if engine else None
        if exp_buf is not None:
            exp_count = len(exp_buf)
    except Exception:
        exp_count = 0
    policy_progress = min(1.0, exp_count / 50.0)

    # Personality emergence progress from trait deviation around neutral 0.5.
    trait_deviation = 0.0
    try:
        rollback = (partial_snapshot.get("personality") or {}).get("rollback", {})
        traits = rollback.get("current_traits", {}) if isinstance(rollback, dict) else {}
        vals = [float(v) for v in (traits or {}).values() if isinstance(v, (int, float))]
        if vals:
            trait_deviation = sum(abs(v - 0.5) for v in vals) / len(vals)
    except Exception:
        trait_deviation = 0.0
    personality_progress = min(1.0, trait_deviation / 0.3) if trait_deviation > 0 else 0.0

    # Loop-integrity progress from autonomy delta measurements.
    measured_deltas = 0
    try:
        measured_deltas = int(
            (((partial_snapshot.get("autonomy") or {}).get("delta_tracker") or {}).get("total_measured", 0) or 0
        )
        )
    except Exception:
        measured_deltas = 0
    loop_progress = min(1.0, measured_deltas / 10.0)

    return {
        "active": False,
        "graduated": True,
        "phase": 3,
        "phase_name": "identity_formation",
        "started_at": float(cert.get("gestation_started", 0.0) or 0.0),
        "graduated_at": float(cert.get("gestation_completed", 0.0) or 0.0),
        "elapsed_s": 0.0,
        "directives_issued": int(cert.get("directives_completed", 0) or 0),
        "directives_completed": int(cert.get("directives_completed", 0) or 0),
        "research_jobs_completed": int(cert.get("research_jobs_completed", 0) or 0),
        "phase_completions": {"self_study": 0, "knowledge": 0, "bootcamp": 0},
        "backpressure_active": False,
        "readiness": {
            "overall": round(float(overall), 3),
            "components": components,
            "met_minimum_duration": True,
            "recommendation": "graduated",
        },
        "readiness_source": "birth_certificate",
        "birth_snapshot": {
            "instance_id": str(cert.get("instance_id", "") or ""),
            "gestation_started": float(cert.get("gestation_started", 0.0) or 0.0),
            "gestation_completed": float(cert.get("gestation_completed", 0.0) or 0.0),
            "duration_s": float(cert.get("duration_s", 0.0) or 0.0),
            "readiness_at_birth": birth_readiness,
        },
        "post_birth_progress": {
            "policy_experience": round(policy_progress, 3),
            "policy_experience_count": int(exp_count),
            "personality_emergence": round(personality_progress, 3),
            "trait_deviation": round(trait_deviation, 3),
            "loop_integrity": round(loop_progress, 3),
            "measured_deltas": int(measured_deltas),
            "updated_at": time.time(),
        },
        "person_detected": False,
        "person_sustained_s": 0.0,
        "network_healthy": True,
        "first_contact_armed": False,
        "self_study_remaining": 0,
        "knowledge_remaining": 0,
        "bootcamp_remaining": 0,
    }
