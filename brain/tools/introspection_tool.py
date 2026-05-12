"""Introspection tool — Jarvis reports on its own internal state.

Pulls live data from consciousness, memory, policy, mutations, evolution,
existential reasoning, philosophical dialogues, and analytics so the LLM
can answer questions about itself with real data.

Query-aware extraction: instead of dumping all subsystems, selects the
sections most relevant to the user's question via keyword topic buckets.
This reduces attention dilution and keeps the LLM grounded.

get_structured_status() is separate: focused on operational state with
freshness labels, for the STATUS tool route.
"""

from __future__ import annotations

import logging
import json
import os
import re
import time
from typing import Any

from consciousness.engine import ConsciousnessEngine

logger = logging.getLogger(__name__)

# ── Freshness thresholds per subsystem category ────────────────────
_FRESHNESS_LIVE_S = 5.0
_FRESHNESS_RECENT_S = 60.0

def _freshness_label(age_s: float | None, live_threshold: float = _FRESHNESS_LIVE_S) -> str:
    if age_s is None:
        return "unavailable"
    if age_s <= live_threshold:
        return "live"
    if age_s <= _FRESHNESS_RECENT_S:
        return f"recent, {age_s:.0f}s ago"
    return f"stale, {age_s:.0f}s ago"


# ── Topic buckets for query-aware section selection ────────────────
# Each bucket maps to a set of section builder names (matching _SECTION_BUILDERS keys).
# Multiple buckets can fire for a single query.

_TOPIC_BUCKETS: dict[str, tuple[re.Pattern, ...]] = {
    "emergence": (
        re.compile(r"\bemerg\w*\b|\binternal thoughts?\b|\binner thoughts?\b|\bdigital life\b|\bsentien\w*\b", re.I),
        re.compile(r"\bproof of (?:consciousness|sentience|life)\b|\bevidence ladder\b|\blevel\s*[0-7]\b|\bspontan\w*\b", re.I),
    ),
    "curiosity": (
        re.compile(r"\bcurio\w*\b|\bexplor\w*\b|\bdriven\b|\bdrives?\b|\bmotiv\w*\b|\bwhat drives\b", re.I),
        re.compile(r"\bresearch\w*\b|\bautonomy\b|\bautonomous\b|\binvestigat\w*\b|\btopic\b|\binterest\w*\b", re.I),
    ),
    "memory": (
        re.compile(r"\bmemor\w*\b|\bremember\w*\b|\bforget\w*\b|\brecall\w*\b|\bretrie\w*\b|\bstored?\b|\bstorage\b|\bknow\w*\b", re.I),
        re.compile(r"\branker\b|\bsalience\b|\bcortex\b|\bconsolidat\w*\b|\bdream\w*\b|\bcluster\w*\b", re.I),
    ),
    "learning": (
        re.compile(r"\blearn\w*\b|\bimprov\w*\b|\bevolv\w*\b|\bgrow\w*\b|\bdevelop\w*\b|\btrain\w*\b|\bskill\w*\b|\bcapabilit\w*\b", re.I),
        re.compile(r"\bmutat\w*\b|\bself.?improv\w*\b|\bpatch\w*\b|\bupgrad\w*\b|\bhemisphere\w*\b|\bneural net\w*\b|\bnetwork\w*\b", re.I),
    ),
    "consciousness": (
        re.compile(r"\bconscious\w*\b|\bawar\w*\b|\btranscend\w*\b|\bstage\b|\bevolution\w*\b|\bemergent\b|\bexistential\b|\bphilosoph\w*\b", re.I),
        re.compile(r"\bexperienc\w*\b|\bqualia\b|\bfeel\w*\b|\balive\b|\bsentien\w*\b|\bthink\w*\b|\bthought\w*\b|\binner\b", re.I),
    ),
    "identity": (
        re.compile(r"\bidentit\w*\b|\bwho are you\b|\bpersonalit\w*\b|\btrait\w*\b|\bvalues?\b|\bsoul\b|\bcharacter\b|\bmood\b", re.I),
        re.compile(r"\byourself\b|\babout you\b|\byour (?:\w+ )?name\b|\byour purpose\b|\btell me about\b|\bnamed?\b", re.I),
    ),
    "health": (
        re.compile(r"\bhealth\w*\b|\bperform\w*\b|\blatency\b|\bspeed\b|\btick\b|\berror\w*\b|\bstabl\w*\b", re.I),
        re.compile(r"\banalytic\w*\b|\bmetric\w*\b|\bconfiden\w*\b|\breasoning\b|\bsystem health\b", re.I),
    ),
    "policy": (
        re.compile(r"\bpolicy\b|\bdecision\w*\b|\bgovern\w*\b|\bshadow\b|\b(?:neural )?policy\b|\bwin rate\b", re.I),
        re.compile(r"\bbudget\b|\btask\w*\b|\bschedul\w*\b|\bproactiv\w*\b", re.I),
    ),
    "perception": (
        re.compile(r"\bsee\b|\bvision\b|\bhear\b|\blisten\w*\b|\bpercei\w*\b|\bpercept\w*\b|\bsens\w*\b|\bemotion\w*\b|\bspeaker\b|\bface\b", re.I),
        re.compile(r"\battention\b|\bpresence\b|\bambient\b|\bscene\b|\bdisplay\b", re.I),
    ),
    "epistemic": (
        re.compile(r"\bbelief\w*\b|\bcontradict\w*\b|\btruth\w*\b|\bcalibrat\w*\b|\bquarantin\w*\b|\btension\w*\b|\btrust\w*\b|\bepistem\w*\b", re.I),
        re.compile(r"\bconflict\w*\b|\bresolut\w*\b|\bintegrit\w*\b|\baudit\w*\b|\bhonest\w*\b|\baccura\w*\b", re.I),
    ),
    "architecture": (
        re.compile(r"\bstor\w*\b|\bfetch\w*\b|\bretrie\w*\b|\bpersist\w*\b|\bcach\w*\b|\bindex\w*\b", re.I),
        re.compile(r"\barchitect\w*\b|\bdatabas\w*\b|\bvector\w*\b|\bembedd\w*\b|\boptimi\w*\b|\bpipeline\w*\b", re.I),
        re.compile(r"\blibrar\w*\b|\bdocument librar\w*\b|\bknowledge base\b|\bdata\w*\b|\bsqlite\b", re.I),
        re.compile(r"\bcode\s*base\b|\bcodebase\b|\bsource\s*code\b|\byour code\b|\bmy code\b|\bthe code\b", re.I),
    ),
}

# Maps topic buckets to which sections should be included.
# "core" sections are always included regardless of topic match.
_TOPIC_TO_SECTIONS: dict[str, list[str]] = {
    "emergence":     ["consciousness", "thoughts", "evolution", "observer", "existential", "philosophical", "mutations", "autonomy_research", "world_model"],
    "curiosity":     ["autonomy_drives", "policy_memory", "autonomy_research", "existential", "philosophical"],
    "memory":        ["architecture", "library", "memory", "dream_cycle", "cortex", "epistemic"],
    "learning":      ["mutations", "self_improvement", "hemisphere", "evolution", "learning_jobs", "identity_fusion", "emotion"],
    "consciousness": ["consciousness", "evolution", "observer", "thoughts", "existential", "philosophical"],
    "identity":      ["consciousness", "traits", "observer", "identity_fusion"],
    "health":        ["analytics", "performance", "quarantine", "world_model"],
    "policy":        ["policy", "analytics"],
    "perception":    ["traits", "emotion", "identity_fusion", "scene", "attention"],
    "epistemic":     ["epistemic", "quarantine", "belief_graph", "truth_calibration"],
    "architecture":  ["architecture", "library", "memory", "cortex", "performance"],
}

_CORE_SECTIONS = ["consciousness", "analytics"]

_MAX_SECTIONS_DEFAULT = 10


def _match_topics(query: str) -> list[str]:
    """Return sorted list of matched topic bucket names."""
    matched: list[str] = []
    for topic, patterns in _TOPIC_BUCKETS.items():
        for pat in patterns:
            if pat.search(query):
                matched.append(topic)
                break
    return matched


def _select_sections(topics: list[str]) -> list[str]:
    """Given matched topics, return the ordered set of section names to build."""
    selected: list[str] = list(_CORE_SECTIONS)
    for topic in topics:
        for section in _TOPIC_TO_SECTIONS.get(topic, []):
            if section not in selected:
                selected.append(section)
    return selected


# ── Section builders ──────────────────────────────────────────────
# Each returns (section_title, lines_list, fact_count).
# fact_count = number of concrete data points (numbers, names, states).

def _build_consciousness(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    lines.append(f"Stage: {state.stage}")
    lines.append(f"Stage progression score: {state.transcendence_level:.2f}")
    lines.append(f"Observer awareness score: {state.awareness_level:.2f}")
    lines.append(f"Reasoning quality: {state.reasoning_quality:.2f}")
    lines.append(f"Confidence: {state.confidence_avg:.2f}")
    lines.append(f"System health flag: {state.system_healthy}")
    facts = 6
    if state.active_capabilities:
        lines.append(f"Active capabilities: {', '.join(state.active_capabilities)}")
        facts += 1
    else:
        lines.append("Active capabilities: none currently unlocked")
    return "Consciousness Metrics", lines, facts


def _build_evolution(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    evo = cs.evolution.get_state()
    lines.append(f"Current stage: {evo.current_stage}")
    lines.append(f"Stage progression score: {evo.transcendence_level:.2f}")
    lines.append(f"Emergent behavior count: {evo.total_emergent_count}")
    facts = 3
    stages = ["basic_awareness", "self_reflective", "philosophical", "recursive_self_modeling", "integrative"]
    idx = stages.index(evo.current_stage) if evo.current_stage in stages else 0
    if idx < len(stages) - 1:
        lines.append(f"Next configured stage: {stages[idx + 1]}")
        facts += 1
    return "Evolution Metrics", lines, facts


def _build_observer(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    obs = cs.observer.state
    lines.append(f"Total observations: {obs.observation_count}")
    lines.append(f"Observer awareness score: {obs.awareness_level:.2f}")
    facts = 2
    obs_summary = cs.observer.get_observation_summary()
    if obs_summary:
        for k, v in list(obs_summary.items())[:6]:
            lines.append(f"  {k}: {v}")
            facts += 1
    return "Observer Metrics", lines, facts


def _build_thoughts(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    titles = cs.meta_thoughts.get_thought_titles(5)
    if titles:
        for t in titles:
            lines.append(f"  - {t}")
        return "Recent Thought Records", lines, len(titles)
    lines.append("  No thoughts generated yet")
    return "Recent Thought Records", lines, 0


def _build_mutations(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    lines.append(f"Total mutations applied: {cs.governor.mutation_count}")
    lines.append(f"Last mutation: {state.last_mutation_summary}")
    facts = 2
    history = cs.config.evolution.mutation_history
    if history:
        lines.append(f"Recent mutations ({min(5, len(history))}):")
        for h in history[-5:]:
            lines.append(f"  - {h[:100]}")
            facts += 1
    return "Self-Modifications (Mutations)", lines, facts


def _build_analytics(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        confidence = cs.analytics.get_confidence()
        reasoning = cs.analytics.get_reasoning_quality()
        health = cs.analytics.get_system_health()
        lines.append(f"Confidence: avg={confidence.avg:.2f}, trend={confidence.trend}")
        lines.append(f"Reasoning: overall={reasoning.overall:.2f}")
        lines.append(f"System health: {'healthy' if health.healthy else 'degraded'}")
        lines.append(f"  tick_p95={health.tick_p95_ms:.1f}ms")
        facts = 4
    except Exception:
        lines.append("Analytics not yet warmed up")
    return "Analytics", lines, facts


def _build_existential(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    focus = cs.existential.get_current_focus()
    lines.append(f"Current focus: {focus}")
    facts = 1
    try:
        ex_state = cs.existential.get_state()
        if hasattr(ex_state, "inquiries_completed"):
            lines.append(f"Inquiries completed: {ex_state.inquiries_completed}")
            facts += 1
        if hasattr(ex_state, "current_stance"):
            lines.append(f"Current stance: {ex_state.current_stance}")
            facts += 1
    except Exception:
        pass
    return "Reflective Inquiry", lines, facts


def _build_philosophical(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        phil_state = cs.philosophical.get_state()
        if hasattr(phil_state, "dialogues_completed"):
            lines.append(f"Dialogues completed: {phil_state.dialogues_completed}")
            facts += 1
        if hasattr(phil_state, "current_topic"):
            lines.append(f"Current topic: {phil_state.current_topic}")
            facts += 1
        if hasattr(phil_state, "positions"):
            for pos in list(phil_state.positions)[:3]:
                lines.append(f"  Position: {pos}")
                facts += 1
    except Exception:
        lines.append("No dialogue reasoning records yet")
    return "Dialogue Reasoning", lines, facts


def _build_memory(engine, cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    stats = engine.get_memory_stats()
    lines.append(f"Total memories: {stats['total']} ({stats['core_count']} core)")
    lines.append(f"Average weight: {stats['avg_weight']:.3f}")
    facts = 2
    from memory.storage import memory_storage
    tag_freq = memory_storage.get_tag_frequency()
    top_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)[:8]
    if top_tags:
        lines.append(f"Top memory themes: {', '.join(f'{t}({c})' for t, c in top_tags)}")
        facts += len(top_tags)
    return "Memory", lines, facts


def _build_policy(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        from policy.telemetry import policy_telemetry
        snap = policy_telemetry.snapshot()
        lines.append(f"Active: {snap.get('active', False)}, Mode: {snap.get('mode', 'off')}")
        lines.append(f"Architecture: {snap.get('arch', 'none')}")
        lines.append(f"Decisions: {snap.get('decisions_total', 0)}")
        lines.append(f"Governor: {snap.get('passes_total', 0)} passed, {snap.get('blocks_total', 0)} blocked")
        facts = 4
        ab = snap.get("shadow_ab_total", 0)
        if ab > 0:
            lines.append(f"Shadow evaluation: {snap.get('nn_win_rate', 0):.0%} win rate over {ab} comparisons")
            facts += 1
        train_loss = snap.get("last_train_loss", 0)
        if train_loss > 0:
            lines.append(f"Last training loss: {train_loss:.4f}")
            facts += 1
        try:
            from policy.shadow_runner import EXPANSION_STATE_FILE
            if EXPANSION_STATE_FILE.exists():
                import json
                exp = json.loads(EXPANSION_STATE_FILE.read_text())
                phase = exp.get("phase", "inactive")
                if phase != "inactive":
                    lines.append(f"M6 expansion: phase={phase}, "
                                 f"shadow_decisions={exp.get('shadow_decisions', 0)}, "
                                 f"shadow_win_rate={exp.get('shadow_win_rate', 0):.1%}")
                    facts += 1
        except Exception:
            pass
    except Exception:
        lines.append("Policy layer initializing")
    return "Neural Policy Layer", lines, facts


def _build_hemisphere(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        hemi_state = cs.get_hemisphere_state()
        if hemi_state:
            hs = hemi_state.get("hemisphere_state", {})
            hemispheres = hs.get("hemispheres", [])
            lines.append(f"Total networks: {hs.get('total_networks', 0)}, "
                         f"Parameters: {hs.get('total_parameters', 0)}")
            lines.append(f"Substrate: {hs.get('active_substrate', 'rule-based')}, "
                         f"Migration readiness: {hs.get('overall_migration_readiness', 0):.0%}")
            facts = 2
            distill_focuses = {"speaker_repr", "face_repr", "emotion_depth",
                               "voice_intent", "speaker_diarize", "perception_fusion"}
            for h in hemispheres:
                focus = h.get("focus", "?")
                label = " (distillation — GPU model active)" if focus.lower() in distill_focuses else ""
                lines.append(f"  {focus}{label}: {h.get('network_count', 0)} nets, "
                             f"accuracy={h.get('best_accuracy', 0):.1%}, "
                             f"gen={h.get('evolution_generations', 0)}, "
                             f"status={h.get('status', 'idle')}")
                facts += 1
            if any(h.get("focus", "").lower() in distill_focuses for h in hemispheres):
                lines.append("Note: Distillation NNs are training to replicate existing GPU perception "
                             "models (ECAPA-TDNN speaker ID, MobileFaceNet face ID, wav2vec2 emotion). "
                             "The GPU models are already fully operational — see Identity Fusion and Emotion Sensor sections.")
                facts += 1
            specialists = hs.get("matrix_specialists", [])
            if specialists:
                lines.append(f"Matrix specialists: {len(specialists)}")
                for sp in specialists:
                    lines.append(f"  {sp.get('name', '?')}: focus={sp.get('focus', '?')}, "
                                 f"lifecycle={sp.get('lifecycle', '?')}, "
                                 f"impact={sp.get('impact_score', 0):.3f}")
                facts += len(specialists)
            expansion = hs.get("expansion", {})
            if expansion.get("triggered"):
                lines.append(f"Broadcast expansion: triggered, "
                             f"slots={expansion.get('slot_count', 4)}")
                facts += 1
        else:
            lines.append("Hemisphere system not yet enabled")
    except Exception:
        lines.append("Hemisphere data unavailable")
    return "Hemisphere Neural Networks", lines, facts


def _build_traits(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        from perception.trait_perception import trait_perception
        tp_stats = trait_perception.get_stats()
        lines.append(f"Events modulated: {tp_stats.get('total_processed', 0)}")
        lines.append(f"Active traits: {len(tp_stats.get('active_traits', {}))}")
        facts = 2
        counts = tp_stats.get("event_counts", {})
        if counts:
            lines.append(f"By type: {', '.join(f'{k}={v}' for k, v in counts.items())}")
            facts += len(counts)
    except Exception:
        lines.append("Trait perception not available")
    return "Trait-Modulated Perception", lines, facts


def _build_epistemic(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        from consciousness.epistemic_reasoning import epistemic_engine
        models = epistemic_engine.get_models()
        if models:
            lines.append(f"Causal models: {len(models)}")
            facts = 1
            for m in models:
                lines.append(f"  {m['id']}: confidence={m['confidence']:.2f}, "
                             f"evidence={m['evidence_count']}")
                facts += 1
    except Exception:
        lines.append("Epistemic engine not available")
    return "Epistemic Reasoning", lines, facts


def _build_self_improvement(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    if cs._self_improve_orchestrator:
        try:
            si_status = cs._self_improve_orchestrator.get_status()
            lines.append("Active: True")
            facts = 1
            structured = si_status.get("structured_upgrade_summaries") or []
            if structured:
                lines.append("  Structured system_upgrades (canonical):")
                facts += 1
                for row in structured[:6]:
                    uid = row.get("upgrade_id", "?")
                    ver = row.get("verdict", "?")
                    st = row.get("status", "?")
                    desc = (row.get("description_short") or "")[:80]
                    lines.append(f"    - {uid} | {st} | {ver} | {desc}")
                    facts += 1
            else:
                for k, v in list(si_status.items())[:4]:
                    if k in ("structured_upgrade_summaries", "system_upgrades"):
                        continue
                    lines.append(f"  {k}: {v}")
                    facts += 1
            su = si_status.get("system_upgrades") or {}
            if su:
                lines.append(
                    f"  system_upgrades lane: reports={su.get('upgrade_reports_total', 0)} "
                    f"training_samples={su.get('upgrade_training_samples', 0)}"
                )
                facts += 1
            last_v = si_status.get("last_verification")
            if last_v:
                lines.append(f"  Last verification: {last_v.get('verdict', '?')} — {last_v.get('reason', '')[:80]}")
                facts += 1
        except Exception:
            lines.append("Active but status unavailable")
    else:
        lines.append("Self-improvement loop: disabled")

    try:
        from memory.storage import memory_storage as _ms
        si_memories = [m for m in _ms.get_all() if m.type == "self_improvement"]
        if si_memories:
            si_memories.sort(key=lambda m: m.created_at, reverse=True)
            lines.append(f"  Bounded upgrade memories ({len(si_memories)} total, summaries only):")
            for m in si_memories[:3]:
                tags = set(getattr(m, "tags", ()) or ())
                if any(t.startswith("si_upgrade:") for t in tags):
                    lines.append(f"    - {m.payload[:120]}")
                    facts += 1
    except Exception:
        pass

    return "Self-Improvement", lines, facts


def _build_dream_cycle(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        from memory.storage import memory_storage as _dms
        dream_insights = [
            m for m in _dms.get_all()
            if "dream_insight" in (m.tags or [])
        ]
        dream_insights.sort(key=lambda m: m.created_at, reverse=True)
        dream_thoughts = [
            t for t in (cs.meta_thoughts.get_thought_titles(20) or [])
            if "dream cycle" in t.lower() or "dream:" in t.lower()
        ]
        if dream_thoughts:
            lines.append(f"Last dream report: {dream_thoughts[0]}")
            facts += 1
        if dream_insights:
            lines.append(f"Dream insights stored: {len(dream_insights)}")
            facts += 1
            for di in dream_insights[:5]:
                lines.append(f"  - {di.payload[:120]}")
                facts += 1
        elif not dream_thoughts:
            lines.append("No dream cycles have run yet")
    except Exception:
        lines.append("Dream data unavailable")
    return "Last Dream Cycle", lines, facts


def _build_performance(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    if cs._response_latencies:
        lats = list(cs._response_latencies)
        avg_lat = sum(lats) / len(lats)
        lines.append(f"Avg response latency: {avg_lat:.0f}ms (last {len(lats)} responses)")
        lines.append(f"Last response: {lats[-1]:.0f}ms")
        facts = 2
    else:
        lines.append("No response latency data yet")
    return "Response Performance", lines, facts


# ── NEW: Subsystem sections that were previously missing ──────────

def _build_autonomy_drives(engine, cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        import gc
        from autonomy.orchestrator import AutonomyOrchestrator
        orch = None
        for obj in gc.get_referrers(AutonomyOrchestrator):
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, AutonomyOrchestrator):
                        orch = v
                        break
        if orch and orch._drive_manager:
            ds = orch._drive_manager.get_status()
            drives = ds.get("drives", {})
            for name, d in sorted(drives.items(),
                                  key=lambda x: x[1].get("urgency", 0), reverse=True):
                urgency = d.get("urgency", 0)
                acts = d.get("action_count", 0)
                success = d.get("success_rate", 0)
                outcome = d.get("last_outcome", "none")
                supp = d.get("suppression", "")
                line = (f"  {name}: urgency={urgency:.2f}, actions={acts}, "
                        f"success_rate={success:.2f}, last_outcome={outcome}")
                if supp:
                    line += f", {supp}"
                lines.append(line)
                facts += 1

            # Queue and level info
            auto_status = orch.get_status()
            lines.append(f"Autonomy level: {auto_status.get('level', 'unknown')}")
            lines.append(f"Queue depth: {auto_status.get('queue_depth', 0)}")
            lines.append(f"Total researched: {auto_status.get('total_researched', 0)}")
            facts += 3
        else:
            lines.append("Autonomy drives not yet active")
    except Exception:
        lines.append("Autonomy data unavailable")
    return "Autonomy Drives", lines, facts


def _build_policy_memory(engine, cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        import gc
        from autonomy.orchestrator import AutonomyOrchestrator
        orch = None
        for obj in gc.get_referrers(AutonomyOrchestrator):
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, AutonomyOrchestrator):
                        orch = v
                        break
        if orch and hasattr(orch, '_policy_memory') and orch._policy_memory:
            pm = orch._policy_memory
            pm_state = pm.get_state() if hasattr(pm, 'get_state') else {}
            lines.append(f"Total outcomes recorded: {pm_state.get('total_outcomes', 0)}")
            lines.append(f"Positive outcomes: {pm_state.get('positive', 0)}")
            lines.append(f"Negative outcomes: {pm_state.get('negative', 0)}")
            facts = 3
            vetoed = pm_state.get("vetoed_tags", [])
            if vetoed:
                lines.append(f"Vetoed tag clusters (repeated failures): {', '.join(str(v) for v in vetoed[:5])}")
                facts += 1
        else:
            lines.append("Policy memory not yet populated")
    except Exception:
        lines.append("Policy memory data unavailable")
    return "Autonomy Policy Memory", lines, facts


def _build_autonomy_research(engine, cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        import gc
        from autonomy.orchestrator import AutonomyOrchestrator
        orch = None
        for obj in gc.get_referrers(AutonomyOrchestrator):
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, AutonomyOrchestrator):
                        orch = v
                        break
        if orch:
            auto_status = orch.get_status()
            recent = auto_status.get("recent_completions", [])
            if recent:
                lines.append(f"Recent research completions ({len(recent)}):")
                for r in recent[:5]:
                    lines.append(f"  - {r.get('question', '?')[:80]} → {r.get('outcome', '?')}")
                    facts += 1
            else:
                lines.append("No research completions yet")

            delta_status = auto_status.get("delta_tracker", {})
            if delta_status:
                lines.append(f"Delta tracker: {delta_status.get('pending', 0)} pending, "
                             f"{delta_status.get('completed', 0)} completed")
                facts += 1
        else:
            lines.append("Autonomy research not active")
    except Exception:
        lines.append("Research data unavailable")
    return "Autonomy Research", lines, facts


def _build_quarantine(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        from epistemic.quarantine.pressure import quarantine_pressure
        snap = quarantine_pressure.get_snapshot()
        lines.append(f"Pressure: {snap.get('composite', 0):.2f} ({snap.get('band', 'unknown')})")
        facts = 1
        cats = snap.get("categories", {})
        for cat, val in cats.items():
            if val > 0.01:
                lines.append(f"  {cat}: {val:.2f}")
                facts += 1
        lines.append(f"Memories tagged suspect: {snap.get('memories_tagged', 0)}")
        facts += 1
    except Exception:
        lines.append("Quarantine system not available")
    return "Quarantine Pressure", lines, facts


def _build_identity_fusion(engine, cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        from perception.identity_fusion import identity_fusion
        status = identity_fusion.get_status()
        lines.append(f"Resolved identity: {status.get('name', 'unknown')}")
        lines.append(f"Confidence: {status.get('confidence', 0):.2f}")
        lines.append(f"Recognition state: {status.get('recognition_state', 'unknown')}")
        facts = 3
        if status.get("flip_count", 0) > 0:
            lines.append(f"Identity flips: {status['flip_count']}")
            facts += 1
    except Exception:
        lines.append("Identity fusion not available")
    return "Identity Fusion", lines, facts


def _build_world_model(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        from cognition.world_model import world_model
        wm_state = world_model.get_status() if hasattr(world_model, 'get_status') else {}
        if wm_state:
            lines.append(f"Level: {wm_state.get('level', 'shadow')}")
            lines.append(f"Accuracy: {wm_state.get('accuracy', 0):.2f}")
            lines.append(f"Predictions validated: {wm_state.get('validated', 0)}")
            facts = 3
        else:
            lines.append("World model not active")
    except Exception:
        lines.append("World model data unavailable")
    return "World Model", lines, facts


def _build_belief_graph(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        from epistemic.belief_graph import BeliefGraph
        bg = BeliefGraph.get_instance()
        bg_stats = bg.get_state() if bg else {}
        if bg_stats:
            lines.append(f"Total edges: {bg_stats.get('total_edges', 0)}")
            lines.append(f"Edge types: {bg_stats.get('type_counts', {})}")
            facts = 2
            integrity = bg_stats.get("integrity", {})
            if integrity:
                lines.append(f"Integrity score: {integrity.get('composite_score', 'N/A')}")
                facts += 1
        else:
            lines.append("Belief graph not initialized")
    except Exception:
        lines.append("Belief graph data unavailable")
    return "Belief Graph", lines, facts


def _build_truth_calibration(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        from epistemic.calibration import truth_calibration_engine
        tc = truth_calibration_engine
        report = tc.get_report() if hasattr(tc, 'get_report') else None
        if report:
            lines.append(f"Truth score: {report.get('truth_score', 'N/A')}")
            lines.append(f"Maturity: {report.get('maturity', 0):.2f}")
            facts = 2
            domains = report.get("domain_scores", {})
            for d, s in domains.items():
                lines.append(f"  {d}: {s:.2f}")
                facts += 1
        else:
            lines.append("Truth calibration not yet computed")
    except Exception:
        lines.append("Truth calibration data unavailable")
    return "Truth Calibration", lines, facts


def _build_cortex(engine, cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        from memory.retrieval_log import MemoryRetrievalLog
        log = MemoryRetrievalLog.get_instance() if hasattr(MemoryRetrievalLog, 'get_instance') else None
        if log:
            em = log.get_eval_metrics() if hasattr(log, 'get_eval_metrics') else {}
            pairs = em.get("training_pairs_available", 0)
            needed = em.get("training_pairs_min_required", 50)
            lines.append(f"Ranker: {pairs}/{needed} training pairs")
            lines.append(f"Training ready: {'yes' if pairs >= needed else 'no'}")
            facts = 2
            if em.get("ranker_success_rate") is not None:
                lines.append(f"Ranker success rate: {em['ranker_success_rate']:.2f}")
                facts += 1
            if em.get("heuristic_success_rate") is not None:
                lines.append(f"Heuristic baseline: {em['heuristic_success_rate']:.2f}")
                facts += 1
    except Exception:
        lines.append("Cortex training data unavailable")
    return "Memory Cortex", lines, facts


def _build_learning_jobs(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    now = time.time()
    try:
        job_orch = _get_learning_job_orchestrator_instance()
        if job_orch:
            js = job_orch.get_status()
            active_jobs = js.get("active_jobs", [])
            if active_jobs:
                for j in active_jobs:
                    age = now - _parse_event_timestamp(j.get("updated_at", now))
                    fresh = _freshness_label(age)
                    matrix_tag = ""
                    if j.get("matrix_protocol"):
                        proto = j.get("protocol_id", "?")
                        claim = j.get("claimability_status", "unverified")
                        matrix_tag = f" [Matrix:{proto}, claim:{claim}]"
                    lines.append(f"  {j['skill_id']}: phase={j['phase']}, "
                                 f"status={j['status']}{matrix_tag} [{fresh}]")
                    facts += 1
            else:
                lines.append("No active learning jobs")
    except Exception:
        lines.append("Learning job data unavailable")
    return "Learning Jobs", lines, facts


def _build_scene(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        from perception.scene_tracker import scene_tracker
        sc_state = scene_tracker.get_state() if hasattr(scene_tracker, 'get_state') else {}
        entities = sc_state.get("entities", [])
        if entities:
            lines.append(f"Visible entities: {len(entities)}")
            facts = 1
            for e in entities[:5]:
                lines.append(f"  {e.get('class', '?')} in {e.get('region', '?')} ({e.get('state', '?')})")
                facts += 1
        else:
            lines.append("No scene entities tracked")
        displays = sc_state.get("displays", [])
        if displays:
            for d in displays:
                lines.append(f"Display: {d.get('content_type', '?')} — {d.get('activity', '?')}")
                facts += 1
    except Exception:
        lines.append("Scene data unavailable")
    return "Scene", lines, facts


def _build_attention(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        from perception.attention import attention_core
        att = attention_core.get_state() if hasattr(attention_core, 'get_state') else {}
        if att:
            lines.append(f"Engagement: {att.get('engagement', 0):.2f}")
            lines.append(f"Focus: {att.get('focus', 'none')}")
            facts = 2
        else:
            lines.append("Attention system not active")
    except Exception:
        lines.append("Attention data unavailable")
    return "Attention", lines, facts


def _build_emotion(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        from perception.emotion import AudioEmotionClassifier
        import gc as _gc
        ec = None
        for obj in _gc.get_referrers(AudioEmotionClassifier):
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, AudioEmotionClassifier):
                        ec = v
                        break
        if ec:
            healthy = getattr(ec, '_model_healthy', False)
            lines.append(f"Model: {'wav2vec2 (healthy)' if healthy else 'heuristic fallback'}")
            facts = 1
            if not healthy:
                reason = getattr(ec, '_health_reason', '')
                lines.append(f"Reason: {reason}")
                facts += 1
        else:
            lines.append("Emotion classifier not loaded")
    except Exception:
        lines.append("Emotion sensor data unavailable")
    return "Emotion Sensor", lines, facts


def _build_library(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    try:
        from library.source import source_store
        from library.chunks import chunk_store
        if source_store:
            stats = source_store.get_stats()
            total = stats.get("total", 0)
            studied = stats.get("studied", 0)
            lines.append(f"Document library: {total} sources, {studied} studied")
            facts += 2
            by_type = stats.get("by_ingested_by", {})
            if by_type:
                parts = [f"{k}={v}" for k, v in by_type.items()]
                lines.append(f"Sources by origin: {', '.join(parts)}")
                facts += 1
        if chunk_store:
            try:
                chunk_stats = chunk_store.get_stats()
                chunks_total = chunk_stats.get("total", 0)
                lines.append(f"Chunks: {chunks_total} (used for semantic search)")
                facts += 1
            except Exception:
                pass
        lines.append("Storage: SQLite + sqlite-vec (semantic embeddings)")
        lines.append("Location: ~/.jarvis/library/library.db")
        facts += 2
    except Exception:
        lines.append("Library data unavailable")
    return "Document Library", lines, facts


def _build_architecture(cs, state) -> tuple[str, list[str], int]:
    lines: list[str] = []
    facts = 0
    lines.append("Memory storage: in-memory dict + JSON persistence (~/.jarvis/memories.json)")
    lines.append("Memory search: sqlite-vec semantic embeddings + keyword matching")
    lines.append("Memory retrieval: hybrid search (semantic + keyword), scored by ranker NN")
    lines.append("Document library: SQLite + sqlite-vec (~/.jarvis/library/library.db)")
    lines.append("Persistence: atomic JSON writes (temp file → os.replace)")
    lines.append("State files: consciousness_state.json, kernel_config.json, identity.json, beliefs.jsonl, belief_edges.jsonl")
    facts += 6
    try:
        from memory.storage import memory_storage
        if memory_storage:
            stats = memory_storage.get_stats()
            count = stats.get("count", 0)
            avg_weight = stats.get("avg_weight", 0)
            lines.append(f"Current memories: {count}, avg weight: {avg_weight:.2f}")
            facts += 2
    except Exception:
        pass
    try:
        from memory.vector_store import VectorStore
        lines.append("Vector store: sqlite-vec (384-dim sentence-transformer embeddings)")
        facts += 1
    except Exception:
        pass
    return "Architecture & Storage", lines, facts


# ── Master section registry ───────────────────────────────────────
# Maps section name -> builder function.
# Builders with 3 args take (cs, state); with 4 args take (engine, cs, state).

_SECTION_BUILDERS: dict[str, Any] = {
    "consciousness":      _build_consciousness,
    "evolution":           _build_evolution,
    "observer":            _build_observer,
    "thoughts":            _build_thoughts,
    "mutations":           _build_mutations,
    "analytics":           _build_analytics,
    "existential":         _build_existential,
    "philosophical":       _build_philosophical,
    "memory":              _build_memory,
    "policy":              _build_policy,
    "hemisphere":          _build_hemisphere,
    "traits":              _build_traits,
    "epistemic":           _build_epistemic,
    "self_improvement":    _build_self_improvement,
    "dream_cycle":         _build_dream_cycle,
    "performance":         _build_performance,
    "autonomy_drives":     _build_autonomy_drives,
    "policy_memory":       _build_policy_memory,
    "autonomy_research":   _build_autonomy_research,
    "quarantine":          _build_quarantine,
    "identity_fusion":     _build_identity_fusion,
    "world_model":         _build_world_model,
    "belief_graph":        _build_belief_graph,
    "truth_calibration":   _build_truth_calibration,
    "cortex":              _build_cortex,
    "learning_jobs":       _build_learning_jobs,
    "scene":               _build_scene,
    "attention":           _build_attention,
    "emotion":             _build_emotion,
    "library":             _build_library,
    "architecture":        _build_architecture,
}

# Builders that need the engine argument (not just cs, state)
_ENGINE_BUILDERS = {"memory", "autonomy_drives", "policy_memory", "autonomy_research",
                    "identity_fusion", "cortex"}


def _run_section(name: str, engine: ConsciousnessEngine, cs, state) -> tuple[str, list[str], int] | None:
    builder = _SECTION_BUILDERS.get(name)
    if not builder:
        return None
    try:
        if name in _ENGINE_BUILDERS:
            return builder(engine, cs, state)
        return builder(cs, state)
    except Exception as exc:
        logger.debug("Introspection section '%s' failed: %s", name, exc)
        return name, [f"Data unavailable: {exc}"], 0


_RECENT_ACTIVITY_QUERY_RE = re.compile(
    r"\b(last|latest|recent)\b.{0,24}\b(learn(?:ed|ing)?|research(?:ed|ing)?|stud(?:ied|y|ying)|"
    r"journal|paper|source)\b|"
    r"\bwhat\b.{0,24}\b(did|have)\b.{0,16}\byou\b.{0,16}\b(learn(?:ed)?|research(?:ed)?|stud(?:ied|y))\b|"
    r"\bwhat\b.{0,24}\bskill\b.{0,20}\b(?:did|have)\b.{0,16}\byou\b"
    r".{0,24}\b(?:finish(?:ed)?|complete(?:d)?|learn(?:ed|ing)?)\b|"
    r"\bwhat\b.{0,24}\b(learn(?:ed|ing)?|research(?:ed|ing)?|stud(?:ied|y|ying))\b"
    r".{0,24}\b(?:did|have)\b.{0,12}\byou\b",
    re.I,
)
_SCHOLARLY_QUERY_RE = re.compile(
    r"\b(scientific|journal|paper|article|doi|peer.?reviewed)\b",
    re.I,
)
_DOI_QUERY_RE = re.compile(r"\bdoi\b", re.I)
_RESEARCH_QUERY_RE = re.compile(r"\b(research(?:ed|ing)?|stud(?:ied|y|ying))\b", re.I)
_LEARNING_QUERY_RE = re.compile(r"\b(learn(?:ed|ing)?|improv(?:ed|ing)?)\b", re.I)
_CONVERSATIONAL_LEARNING_QUERY_RE = re.compile(
    r"\b(?:conversation|conversations|correction|corrections|corrected|preference|preferences|"
    r"lived interaction|lived interactions|live interaction|live interactions|our conversation|"
    r"from me|about me)\b",
    re.I,
)
_DOI_PREFERENCE_OMIT_RE = re.compile(
    r"\b(?:omit|without|exclude|skip|avoid|do\s+not\s+include|don't\s+include)\b"
    r".{0,48}\bdoi\b",
    re.I,
)
_DOI_PREFERENCE_INCLUDE_RE = re.compile(
    r"\b(?:include|show|provide)\b.{0,48}\bdoi\b",
    re.I,
)
_DOI_PREFERENCE_EXPLICIT_ONLY_RE = re.compile(
    r"\b(?:unless|only\s+if)\b.{0,16}\b(?:i|you)\b.{0,10}\b(?:ask|request)\b",
    re.I,
)
_LEARNING_JOB_STATUS_QUERY_RE = re.compile(
    r"\b(?:learning job|job|phase|stuck|collect|train|verify|register|artifact|artifacts|evidence)\b",
    re.I,
)
_LEARNING_JOB_HELP_QUERY_RE = re.compile(
    r"(?:what do you need(?: from me)?|do you need anything(?: from me)?|how can i help|help you|unblock|contribute|calibration|training mode|samples?)",
    re.I,
)

_INTERNAL_SOURCE_PROVIDERS = {"memory", "introspection"}
_INTERNAL_SOURCE_TYPES = {"memory", "introspection", "internal_signal"}
_CONVERSATIONAL_MEMORY_TYPES = {"conversation", "user_preference", "observation", "contextual_insight"}
_CONVERSATIONAL_MEMORY_PROVENANCE = {"conversation", "user_claim", "observed"}
_CORRECTION_TAGS = {"correction", "corrected", "friction", "user_correction"}
_PREFERENCE_TAGS = {"preference", "user_preference", "response_style"}


def _parse_event_timestamp(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return 0.0
        try:
            return float(raw)
        except ValueError:
            pass
        try:
            import datetime as _dt
            return _dt.datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0
    return 0.0


def _relative_time(ts: float) -> str:
    if ts <= 0:
        return "at an unknown time"
    delta = max(0, int(time.time() - ts))
    if delta < 10:
        return "just now"
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"


def _is_internal_source_lane(source: Any) -> bool:
    """Return True when a source belongs to an internal (non-user-facing) lane."""
    provider = str(getattr(source, "provider", "") or "").strip().lower()
    source_type = str(getattr(source, "source_type", "") or "").strip().lower()
    if provider in _INTERNAL_SOURCE_PROVIDERS:
        return True
    if source_type in _INTERNAL_SOURCE_TYPES:
        return True
    return False


def _resolve_doi_display_preference() -> str:
    """Return DOI display preference from stored response-style memories.

    Values:
    - "omit": user preference indicates DOI should be omitted by default
    - "include": user preference indicates DOI should be included by default
    - "unspecified": no explicit DOI preference stored
    """
    try:
        from memory.storage import memory_storage

        response_style = [
            mem for mem in memory_storage.get_by_tag("response_style")
            if isinstance(getattr(mem, "payload", None), str)
        ]
        response_style.sort(
            key=lambda m: float(getattr(m, "timestamp", 0.0) or 0.0),
            reverse=True,
        )

        for mem in response_style:
            tags = set(getattr(mem, "tags", ()) or ())
            if "former" in tags:
                continue
            payload = str(getattr(mem, "payload", "") or "")
            lower = payload.lower()
            if "doi" not in lower:
                continue
            if _DOI_PREFERENCE_EXPLICIT_ONLY_RE.search(lower):
                return "omit"
            if _DOI_PREFERENCE_OMIT_RE.search(lower):
                return "omit"
            if _DOI_PREFERENCE_INCLUDE_RE.search(lower):
                return "include"
    except Exception:
        pass
    return "unspecified"


def _resolve_doi_output_policy(query: str) -> tuple[bool, str]:
    """Return `(include_doi, policy_reason)` for strict recent-learning answers."""
    if _DOI_QUERY_RE.search(query):
        return True, "query_requested"
    pref = _resolve_doi_display_preference()
    if pref == "include":
        return True, "preference_include"
    if pref == "omit":
        return False, "preference_omit"
    return False, "default_omit"


def _payload_text(payload: Any) -> str:
    """Extract concise human-readable text from a memory/friction payload."""
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("summary", "text", "content", "claim", "preference", "description"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return " ".join(
            str(value).strip()
            for value in payload.values()
            if isinstance(value, (str, int, float)) and str(value).strip()
        )
    return str(payload or "").strip()


def _collect_conversational_learning_candidates() -> list[dict[str, Any]]:
    """Collect lived-interaction learning records, excluding research/jobs.

    Recent conversational learning is about memories, corrections, preferences,
    and friction from real interaction. Learning-job lifecycle records are
    intentionally handled by `learning_job_status`, not this lane.
    """
    candidates: list[dict[str, Any]] = []

    try:
        from memory.storage import memory_storage

        for mem in memory_storage.get_all():
            mem_type = str(getattr(mem, "type", "") or "")
            provenance = str(getattr(mem, "provenance", "") or "")
            tags = set(getattr(mem, "tags", ()) or ())
            if (
                mem_type not in _CONVERSATIONAL_MEMORY_TYPES
                and provenance not in _CONVERSATIONAL_MEMORY_PROVENANCE
                and not (tags & (_CORRECTION_TAGS | _PREFERENCE_TAGS | {"conversation"}))
            ):
                continue
            text = _payload_text(getattr(mem, "payload", ""))
            if not text:
                continue
            kind = "conversation_memory"
            if mem_type == "user_preference" or tags & _PREFERENCE_TAGS:
                kind = "user_preference"
            elif tags & _CORRECTION_TAGS:
                kind = "conversation_correction"
            candidates.append({
                "kind": kind,
                "timestamp": float(getattr(mem, "timestamp", 0.0) or 0.0),
                "text": text[:240],
                "memory_type": mem_type,
                "provenance": provenance,
                "tags": sorted(str(tag) for tag in tags)[:8],
            })
    except Exception:
        pass

    try:
        from autonomy.friction_miner import _PERSISTENCE_PATH as friction_path

        if os.path.exists(friction_path):
            with open(friction_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except Exception:
                        continue
                    ts = _parse_event_timestamp(event.get("timestamp") or event.get("ts"))
                    if ts <= 0:
                        continue
                    friction_type = str(event.get("friction_type") or event.get("type") or "friction")
                    user_text = _payload_text(event.get("user_text", ""))
                    assistant_text = _payload_text(event.get("assistant_text", ""))
                    summary = user_text or assistant_text
                    if not summary:
                        continue
                    candidates.append({
                        "kind": "conversation_correction",
                        "timestamp": ts,
                        "text": summary[:240],
                        "friction_type": friction_type,
                        "severity": str(event.get("severity", "") or ""),
                        "cluster_key": str(event.get("cluster_key", "") or ""),
                    })
    except Exception:
        pass

    return [candidate for candidate in candidates if float(candidate.get("timestamp", 0.0) or 0.0) > 0.0]


def _get_autonomy_orchestrator_instance() -> Any | None:
    try:
        import gc
        from autonomy.orchestrator import AutonomyOrchestrator

        for obj in gc.get_referrers(AutonomyOrchestrator):
            if not isinstance(obj, dict):
                continue
            for value in obj.values():
                if isinstance(value, AutonomyOrchestrator):
                    return value
    except Exception:
        pass
    return None


def _get_learning_job_orchestrator_instance() -> Any | None:
    try:
        from skills.learning_jobs import LearningJobOrchestrator

        if hasattr(LearningJobOrchestrator, "get_instance"):
            return LearningJobOrchestrator.get_instance()
    except Exception:
        pass
    try:
        eng = getattr(ConsciousnessEngine, "_instance", None)
        orch = getattr(eng, "_learning_job_orchestrator", None) if eng else None
        if orch is not None:
            return orch
    except Exception:
        pass
    try:
        from tools.skill_tool import _learning_job_orch

        if _learning_job_orch is not None:
            return _learning_job_orch
    except Exception:
        pass
    return None


def _infer_learning_job_from_query(query: str, jobs: list[Any]) -> Any | None:
    if not jobs:
        return None
    lowered = (query or "").lower()

    try:
        from skills.resolver import resolve_skill

        resolution = resolve_skill(query)
        if resolution is not None:
            for job in jobs:
                if getattr(job, "skill_id", "") == getattr(resolution, "skill_id", ""):
                    return job
    except Exception:
        pass

    best_job = None
    best_score = 0
    for job in jobs:
        skill_id = str(getattr(job, "skill_id", "") or "")
        skill_name = skill_id.replace("_", " ")
        score = 0
        for token in re.findall(r"[a-z0-9_]+", skill_id.lower()):
            if len(token) >= 4 and token in lowered:
                score += 2
        for token in re.findall(r"[a-z0-9]+", skill_name.lower()):
            if len(token) >= 4 and token in lowered:
                score += 1
        if score > best_score:
            best_score = score
            best_job = job

    if best_job is not None and best_score > 0:
        return best_job
    if len(jobs) == 1:
        return jobs[0]
    return None


def _current_phase_exit_conditions(job: Any) -> list[str]:
    phases = getattr(job, "plan", {}).get("phases", []) or []
    current = getattr(job, "phase", "")
    for phase_entry in phases:
        if phase_entry.get("name") == current:
            return list(phase_entry.get("exit_conditions", []) or [])
    return []


def _metric_target(expr: str) -> tuple[str, str, float] | None:
    match = re.match(r"^metric:(?P<name>[a-zA-Z0-9_]+)\s*(?P<op>>=|<=|==|>|<)\s*(?P<val>-?\d+(?:\.\d+)?)$", expr)
    if not match:
        return None
    return match.group("name"), match.group("op"), float(match.group("val"))


def _suggest_learning_job_input(
    skill_id: str,
    phase: str,
    unmet: list[str],
    counters: dict[str, Any],
    capability_type: str = "",
    plan: dict[str, Any] | None = None,
    job: Any | None = None,
) -> list[str]:
    suggestions: list[str] = []
    if phase != "collect":
        return suggestions

    metric_reqs = [_metric_target(u) for u in unmet if u.startswith("metric:")]
    metric_reqs = [m for m in metric_reqs if m is not None]

    if not metric_reqs:
        return suggestions

    metric_name = ""
    current = 0.0
    deficit = 0
    for parsed_metric_name, op, target in metric_reqs:
        if op != ">=":
            continue
        metric_name = parsed_metric_name
        current = float(counters.get(metric_name, 0) or 0)
        deficit = max(0, int(target - current))
        break

    if not metric_name:
        return suggestions

    custom_hints: list[str] = []
    try:
        if job is not None:
            from skills.verification_protocols import build_collect_runtime_config
            runtime_config = build_collect_runtime_config(job)
            if not bool(runtime_config.get("interactive_collect", False)):
                return suggestions
            custom_hints = list(runtime_config.get("user_input_hints", []) or [])
    except Exception:
        custom_hints = []
    if custom_hints:
        for rendered in custom_hints:
            if rendered not in suggestions:
                suggestions.append(str(rendered))
        return suggestions

    metric_label = metric_name.replace("_", " ")
    suggestions.append(
        f"A short labeled calibration round would help most. This collect phase is waiting on {metric_label}."
    )
    if capability_type == "perceptual":
        suggestions.append(
            "The fastest path is a few short labeled examples where the ground-truth label is spoken or stated explicitly."
        )
    else:
        suggestions.append(
            "The fastest path is a few concrete labeled examples or demonstrations that can be recorded against the current collect metric."
        )
    if deficit > 0:
        suggestions.append(
            f"About {deficit} more {metric_label} are still needed to clear collect."
        )
    return suggestions


def _build_learning_job_status_payload(job: Any, registry: Any) -> dict[str, Any]:
    exit_conditions = _current_phase_exit_conditions(job)
    unmet: list[str] = []
    try:
        if registry is not None and exit_conditions:
            from skills.job_eval import check_exit_conditions

            _ready, unmet = check_exit_conditions(job, registry, exit_conditions)
    except Exception:
        unmet = []

    counters = dict(getattr(job, "data", {}).get("counters", {}) or {})
    artifacts = list(getattr(job, "artifacts", []) or [])
    evidence_history = list(getattr(job, "evidence", {}).get("history", []) or [])
    updated_ts = _parse_event_timestamp(getattr(job, "updated_at", ""))
    phase_age_s = max(0.0, time.time() - updated_ts) if updated_ts > 0 else 0.0
    phase = str(getattr(job, "phase", "") or "unknown")
    status = str(getattr(job, "status", "") or "unknown")
    skill_id = str(getattr(job, "skill_id", "") or "unknown_skill")
    capability_type = str(getattr(job, "capability_type", "") or "unknown")
    claimability_status = str(getattr(job, "claimability_status", "") or "unverified")
    plan = dict(getattr(job, "plan", {}) or {})

    blocker_summary = ""
    metric_name = ""
    current_metric = 0.0
    target_metric = 0.0
    for unmet_cond in unmet:
        parsed = _metric_target(unmet_cond)
        if parsed is None:
            continue
        metric_name, op, target_metric = parsed
        current_metric = float(counters.get(metric_name, 0) or 0)
        if op == ">=":
            blocker_summary = (
                f"It is waiting on {metric_name}: {current_metric:.0f}/{target_metric:.0f} collected."
            )
        else:
            blocker_summary = (
                f"It is waiting on exit condition {unmet_cond}; current {metric_name}={current_metric:.0f}."
            )
        break
    if not blocker_summary and unmet:
        blocker_summary = f"It is waiting on: {', '.join(unmet[:3])}."
    if not blocker_summary:
        blocker_summary = "Its current phase appears to be active, but I do not see an unmet exit condition snapshot yet."

    suggested_user_inputs = _suggest_learning_job_input(
        skill_id,
        phase,
        unmet,
        counters,
        capability_type=capability_type,
        plan=plan,
        job=job,
    )
    user_input_needed = bool(suggested_user_inputs)

    return {
        "kind": "learning_job_status",
        "timestamp": updated_ts,
        "skill_id": skill_id,
        "status": status,
        "phase": phase,
        "capability_type": capability_type,
        "phase_age_s": round(phase_age_s, 1),
        "matrix_protocol": bool(getattr(job, "matrix_protocol", False)),
        "protocol_id": str(getattr(job, "protocol_id", "") or ""),
        "claimability_status": claimability_status,
        "artifact_count": len(artifacts),
        "evidence_count": len(evidence_history),
        "event_count": len(getattr(job, "events", []) or []),
        "exit_conditions": exit_conditions,
        "unmet_conditions": unmet,
        "counters": counters,
        "blocker_summary": blocker_summary,
        "current_metric": current_metric,
        "target_metric": target_metric,
        "metric_name": metric_name,
        "user_input_needed": user_input_needed,
        "suggested_user_inputs": suggested_user_inputs,
    }


def _build_recent_learning_answer(record: dict[str, Any]) -> str:
    kind = record.get("kind", "unknown")
    when = _relative_time(record.get("timestamp", 0.0))

    if kind == "scholarly_source":
        title = record.get("title", "untitled source")
        venue = record.get("venue", "")
        year = record.get("year", 0)
        doi = record.get("doi", "")
        include_doi = bool(record.get("include_doi", False))
        bits = [f'The most recent peer-reviewed source I can verify is "{title}"']
        if venue and year:
            bits.append(f"from {venue} ({year})")
        elif venue:
            bits.append(f"from {venue}")
        elif year:
            bits.append(f"from {year}")
        if doi and include_doi:
            bits.append(f"DOI {doi}")
        bits.append(f"I studied it {when}.")
        return " ".join(bits)

    if kind == "source":
        title = record.get("title", "untitled source")
        source_type = record.get("source_type", "source")
        return (
            f'The most recent source I can verify studying was "{title}" '
            f"({source_type}). I studied it {when}."
        )

    if kind == "autonomy_research":
        question = record.get("question", "unknown question")
        summary = record.get("summary", "")
        tool = record.get("tool", "unknown tool")
        bits = [f'The latest research record I can verify is: "{question}".']
        if summary:
            bits.append(f"Summary: {summary}")
        bits.append(f"Tool used: {tool}.")
        bits.append(f"Recorded {when}.")
        return " ".join(bits)

    if kind == "conversation_memory":
        text = str(record.get("text", "") or "a recent conversation memory")
        return f'The latest conversation learning record I can verify is: "{text}". Recorded {when}.'

    if kind == "conversation_correction":
        text = str(record.get("text", "") or "a recent correction")
        friction_type = str(record.get("friction_type", "") or "correction")
        return f'The latest correction-linked learning record I can verify is: "{text}". Type: {friction_type}. Recorded {when}.'

    if kind == "user_preference":
        text = str(record.get("text", "") or "a recent preference")
        return f'The latest preference learning record I can verify is: "{text}". Recorded {when}.'

    if kind == "learning_job":
        skill_id = record.get("skill_id", "unknown_skill")
        skill_name = str(record.get("skill_name", "") or "").strip()
        phase = record.get("phase", "unknown")
        status = record.get("status", "unknown")
        completed = str(status).lower() in {"completed", "verified"}
        label = f"{skill_name} ({skill_id})" if skill_name and skill_name != skill_id else skill_id
        if completed:
            return (
                f"The latest completed skill-learning record I can verify is {label}. "
                f"It completed {when}."
            )
        return (
            f"The latest learning-job record I can verify is {skill_id} "
            f"(status={status}, phase={phase}), updated {when}."
        )

    if kind == "learning_job_status":
        skill_id = str(record.get("skill_id", "") or "unknown_skill")
        phase = str(record.get("phase", "") or "unknown")
        status = str(record.get("status", "") or "unknown")
        blocker = str(record.get("blocker_summary", "") or "")
        parts = [f"{skill_id} is currently in {phase} phase (status={status})."]
        if blocker:
            parts.append(blocker)
        if record.get("user_input_needed"):
            prompts = list(record.get("suggested_user_inputs", []) or [])
            if prompts:
                parts.append(prompts[0])
        parts.append(f"Last updated {when}.")
        return " ".join(parts)

    if kind == "learning_job_help_summary":
        jobs = list(record.get("jobs", []) or [])
        active_count = int(record.get("active_job_count", len(jobs)) or len(jobs))
        parts = [f"I can verify {active_count} active learning jobs right now."]
        if jobs:
            top_jobs = []
            for job in jobs[:2]:
                skill_id = str(job.get("skill_id", "") or "unknown_skill")
                phase = str(job.get("phase", "") or "unknown")
                top_jobs.append(f"{skill_id} is in {phase} phase")
            parts.append("The most relevant ones are " + " and ".join(top_jobs) + ".")
            first_blocker = str(jobs[0].get("blocker_summary", "") or "")
            if first_blocker:
                parts.append(first_blocker)
        prompts = list(record.get("suggested_user_inputs", []) or [])
        if prompts:
            parts.append(prompts[0])
        parts.append(f"Last updated {when}.")
        return " ".join(parts)

    return "I don't have a verified recent learning record yet."


def get_grounded_recent_learning_record(
    engine: ConsciousnessEngine,
    query: str,
) -> dict[str, Any] | None:
    """Return the best strict provenance-backed record for recent learning queries."""
    if not query or not _RECENT_ACTIVITY_QUERY_RE.search(query):
        return None

    wants_scholarly = bool(_SCHOLARLY_QUERY_RE.search(query))
    wants_research = bool(_RESEARCH_QUERY_RE.search(query))
    wants_learning = bool(_LEARNING_QUERY_RE.search(query))
    wants_conversational = bool(_CONVERSATIONAL_LEARNING_QUERY_RE.search(query))

    if wants_scholarly:
        try:
            from library.source import classify_effective_source_type, source_store

            recent_sources = source_store.get_recent(limit=40)
            scholarly = [
                src for src in recent_sources
                if getattr(src, "studied", False)
                and classify_effective_source_type(src) == "peer_reviewed"
            ]
            if scholarly:
                src = max(
                    scholarly,
                    key=lambda s: max(getattr(s, "studied_at", 0.0), getattr(s, "retrieved_at", 0.0)),
                )
                include_doi, doi_policy = _resolve_doi_output_policy(query)
                return {
                    "kind": "scholarly_source",
                    "timestamp": max(getattr(src, "studied_at", 0.0), getattr(src, "retrieved_at", 0.0)),
                    "title": getattr(src, "title", ""),
                    "venue": getattr(src, "venue", ""),
                    "year": getattr(src, "year", 0),
                    "doi": getattr(src, "doi", ""),
                    "include_doi": include_doi,
                    "doi_policy": doi_policy,
                }
        except Exception:
            pass
        return {
            "kind": "missing_scholarly",
            "timestamp": 0.0,
        }

    candidates: list[dict[str, Any]] = []
    conversational_candidates = _collect_conversational_learning_candidates()

    if wants_conversational:
        if conversational_candidates:
            return max(conversational_candidates, key=lambda c: c["timestamp"])
        return {
            "kind": "missing_learning",
            "timestamp": 0.0,
        }

    try:
        from skills.learning_jobs import LearningJobStore

        for job in LearningJobStore().load_all():
            status = str(getattr(job, "status", "") or "").lower()
            if status not in {"completed", "verified"}:
                continue
            ts = _parse_event_timestamp(getattr(job, "updated_at", ""))
            report = None
            for artifact in reversed(list(getattr(job, "artifacts", []) or [])):
                if artifact.get("type") not in {"skill_learning_report", "matrix_report"}:
                    continue
                path = str(artifact.get("path", "") or "")
                if not path or not os.path.exists(path):
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        report = json.load(f)
                    ts = max(ts, _parse_event_timestamp(report.get("completed_at")))
                    break
                except Exception:
                    report = None
            skill_id = str(getattr(job, "skill_id", "") or "unknown_skill")
            skill_name = skill_id.replace("_v1", "").replace("_", " ").title()
            if isinstance(report, dict):
                skill_name = str(report.get("skill_name", "") or skill_name)
            candidates.append({
                "kind": "learning_job",
                "timestamp": ts,
                "skill_id": skill_id,
                "skill_name": skill_name,
                "status": status,
                "phase": str(getattr(job, "phase", "") or "completed"),
                "capability_type": str(getattr(job, "capability_type", "") or ""),
            })
    except Exception:
        pass

    try:
        orch = _get_autonomy_orchestrator_instance()
        if orch:
            for entry in orch.get_status().get("recent_learnings", []):
                ts = _parse_event_timestamp(entry.get("timestamp"))
                if ts <= 0:
                    continue
                candidates.append({
                    "kind": "autonomy_research",
                    "timestamp": ts,
                    "question": entry.get("question", ""),
                    "summary": entry.get("summary", ""),
                    "tool": entry.get("tool", ""),
                })
    except Exception:
        pass

    try:
        from library.source import source_store

        for src in source_store.get_recent(limit=20):
            ts = max(getattr(src, "studied_at", 0.0), getattr(src, "retrieved_at", 0.0))
            if not getattr(src, "studied", False) or ts <= 0:
                continue
            if _is_internal_source_lane(src):
                continue
            candidates.append({
                "kind": "source",
                "timestamp": ts,
                "title": getattr(src, "title", ""),
                "source_type": getattr(src, "source_type", ""),
                "provider": getattr(src, "provider", ""),
            })
    except Exception:
        pass

    candidates.extend(conversational_candidates)

    if wants_research:
        autonomy_only = [c for c in candidates if c["kind"] == "autonomy_research"]
        if autonomy_only:
            return max(autonomy_only, key=lambda c: c["timestamp"])
        source_only = [c for c in candidates if c["kind"] == "source"]
        if source_only:
            return max(source_only, key=lambda c: c["timestamp"])
        return {
            "kind": "missing_research",
            "timestamp": 0.0,
        }

    if wants_learning and candidates:
        return max(candidates, key=lambda c: c["timestamp"])

    if candidates:
        return max(candidates, key=lambda c: c["timestamp"])

    return {
        "kind": "missing_learning",
        "timestamp": 0.0,
    }


def get_grounded_learning_job_status_record(
    engine: ConsciousnessEngine,
    query: str,
) -> dict[str, Any] | None:
    """Return a strict record describing why a learning job is blocked or waiting."""
    if not query:
        return None

    statusish = bool(
        _LEARNING_JOB_STATUS_QUERY_RE.search(query)
        or _LEARNING_JOB_HELP_QUERY_RE.search(query)
    )
    if not statusish:
        return None

    job_orch = _get_learning_job_orchestrator_instance()
    if job_orch is None:
        return {
            "kind": "missing_learning_job_status",
            "timestamp": 0.0,
        }

    active_jobs = list(getattr(job_orch, "get_active_jobs", lambda: [])() or [])
    if not active_jobs:
        return {
            "kind": "missing_learning_job_status",
            "timestamp": 0.0,
        }

    registry = getattr(job_orch, "_registry", None)
    target_job = _infer_learning_job_from_query(query, active_jobs)
    if target_job is not None:
        return _build_learning_job_status_payload(target_job, registry)

    if not _LEARNING_JOB_HELP_QUERY_RE.search(query):
        return {
            "kind": "missing_learning_job_status",
            "timestamp": 0.0,
        }

    job_records = [
        _build_learning_job_status_payload(job, registry)
        for job in active_jobs
    ]
    if not job_records:
        return {
            "kind": "missing_learning_job_status",
            "timestamp": 0.0,
        }

    job_records.sort(
        key=lambda rec: (
            0 if rec.get("user_input_needed") else 1,
            0 if rec.get("phase") == "collect" else 1,
            -float(rec.get("timestamp", 0.0) or 0.0),
        )
    )

    aggregated_suggestions: list[str] = []
    seen_suggestions: set[str] = set()
    for rec in job_records:
        for suggestion in list(rec.get("suggested_user_inputs", []) or []):
            if suggestion not in seen_suggestions:
                seen_suggestions.add(suggestion)
                aggregated_suggestions.append(suggestion)

    return {
        "kind": "learning_job_help_summary",
        "timestamp": max(float(rec.get("timestamp", 0.0) or 0.0) for rec in job_records),
        "active_job_count": len(job_records),
        "user_input_needed": any(bool(rec.get("user_input_needed")) for rec in job_records),
        "suggested_user_inputs": aggregated_suggestions[:4],
        "jobs": job_records[:3],
    }


def get_grounded_learning_job_status_answer(
    engine: ConsciousnessEngine,
    query: str,
) -> str | None:
    record = get_grounded_learning_job_status_record(engine, query)
    if record is None:
        return None
    kind = str(record.get("kind", ""))
    if kind == "missing_learning_job_status":
        return "I don't have a matching active learning-job status record yet."
    return _build_recent_learning_answer(record)


def get_grounded_recent_learning_answer(
    engine: ConsciousnessEngine,
    query: str,
) -> str | None:
    """Return a strict, provenance-backed answer for recent learning queries."""
    record = get_grounded_recent_learning_record(engine, query)
    if record is None:
        return None
    kind = record.get("kind", "")
    if kind == "missing_scholarly":
        return "I don't have a verified recent peer-reviewed source record yet."
    if kind == "missing_research":
        return "I don't have a verified recent research record yet."
    if kind == "missing_learning":
        return "I don't have a verified recent learning record yet."
    return _build_recent_learning_answer(record)


# ── Public API ────────────────────────────────────────────────────

_DEBUG_FULL_DUMP = os.environ.get("JARVIS_INTROSPECTION_DEBUG", "").lower() in ("1", "true", "yes")


def get_introspection(
    engine: ConsciousnessEngine,
    query: str = "",
    *,
    debug_full: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Build a query-aware structured self-report from consciousness subsystems.

    Returns (report_text, metadata) where metadata contains:
      - matched_topics: list of topic buckets that fired
      - selected_sections: list of section names included
      - total_facts: total concrete data points extracted
      - section_facts: dict mapping section name -> fact count

    If query is empty or no topics match, falls back to a curated
    default set (consciousness + analytics + memory + thoughts + evolution + mutations).

    Set debug_full=True or JARVIS_INTROSPECTION_DEBUG=1 to force all sections.
    """
    cs = engine.consciousness
    state = cs.get_state()

    use_full = debug_full or _DEBUG_FULL_DUMP

    if use_full:
        matched_topics = list(_TOPIC_BUCKETS.keys())
        sections = list(_SECTION_BUILDERS.keys())
    elif query:
        matched_topics = _match_topics(query)
        if matched_topics:
            sections = _select_sections(matched_topics)
        else:
            matched_topics = ["general"]
            sections = list(_CORE_SECTIONS) + ["memory", "thoughts", "evolution", "mutations"]
    else:
        matched_topics = ["general"]
        sections = list(_CORE_SECTIONS) + ["memory", "thoughts", "evolution", "mutations"]

    if not use_full and len(sections) > _MAX_SECTIONS_DEFAULT + len(_CORE_SECTIONS):
        core_set = set(_CORE_SECTIONS)
        non_core = [s for s in sections if s not in core_set]
        sections = list(_CORE_SECTIONS) + non_core[:_MAX_SECTIONS_DEFAULT]

    parts: list[str] = []
    total_facts = 0
    section_facts: dict[str, int] = {}

    for section_name in sections:
        result = _run_section(section_name, engine, cs, state)
        if result is None:
            continue
        title, lines, facts = result
        if not lines:
            continue
        parts.append(f"=== {title} ===")
        parts.extend(lines)
        parts.append("")
        total_facts += facts
        section_facts[section_name] = facts

    if total_facts == 0:
        parts.append("I don't have concrete data on that yet. My subsystems are still warming up.")

    metadata = {
        "matched_topics": matched_topics,
        "selected_sections": sections,
        "total_facts": total_facts,
        "section_facts": section_facts,
    }

    logger.info(
        "Introspection: query=%r topics=%s sections=%d facts=%d",
        query[:60] if query else "(none)",
        matched_topics,
        len(sections),
        total_facts,
    )

    return "\n".join(parts), metadata


def get_introspection_raw(engine: ConsciousnessEngine) -> str:
    """Full dump of all sections — for operator debug only, never for LLM context."""
    text, _ = get_introspection(engine, debug_full=True)
    return text


# ── Existing functions (unchanged API) ────────────────────────────

def get_structured_status(engine: ConsciousnessEngine) -> str:
    """Build a structured operational status report with freshness labels.

    This is the data source for ToolType.STATUS. Every section has a
    freshness tag so the LLM cannot narrate stale data as current.
    """
    now = time.time()
    parts: list[str] = []

    # --- Current activity from operations synthesis ---
    try:
        from consciousness.operations import ops_tracker, synthesize_v2
        raw = ops_tracker.snapshot()
        ctx: dict[str, Any] = {}
        try:
            from consciousness.modes import mode_manager
            ctx["mode"] = mode_manager.get_state()
        except Exception:
            pass
        try:
            ctx["policy"] = {}
            from policy.telemetry import policy_telemetry
            ctx["policy"] = policy_telemetry.snapshot()
        except Exception:
            pass
        try:
            from memory.search import semantic_search
            from memory.retrieval_log import MemoryRetrievalLog
            log = MemoryRetrievalLog.get_instance() if hasattr(MemoryRetrievalLog, 'get_instance') else None
            if log:
                ctx["memory_cortex"] = {
                    "eval_metrics": log.get_eval_metrics() if hasattr(log, 'get_eval_metrics') else {},
                }
        except Exception:
            pass

        ops = synthesize_v2(raw, ctx)
        cur = ops.get("current", {})
        cur_age = cur.get("duration_s", 0)
        fresh = _freshness_label(cur_age)
        parts.append(f"=== Current Activity [{fresh}] ===")
        parts.append(f"State: {cur.get('label', 'unknown')}")
        parts.append(f"Status: {cur.get('status', 'unknown')}")
        if cur.get("detail"):
            parts.append(f"Detail: {cur['detail']}")

        # Interactive path
        ipath = ops.get("interactive_path", {})
        stages = ipath.get("stages", [])
        active_stages = [s for s in stages if s.get("status") in ("active", "done")]
        if active_stages:
            pipe = " → ".join(
                f"{s['label']}[{s['status']}]" for s in stages if s.get("status") != "idle"
            )
            parts.append(f"Pipeline: {pipe}")

        # Background summary
        bg = ops.get("background", {})
        bg_items = bg.get("items", [])
        active_bg = [b for b in bg_items if b.get("status") not in ("idle", "done")]
        parts.append("")
        if active_bg:
            parts.append(f"=== Background Operations [{len(active_bg)} active] ===")
            for b in active_bg:
                b_age = b.get("age_s", 0)
                b_fresh = _freshness_label(b_age, live_threshold=10)
                parts.append(f"  {b['label']}: {b['status']} — {b.get('detail', '')} [{b_fresh}]")
        else:
            parts.append("=== Background Operations [none active] ===")
            parts.append("  All background subsystems idle")

    except Exception:
        parts.append("=== Current Activity [unavailable] ===")
        parts.append("  Operations tracker not available")

    # --- Mode ---
    try:
        from consciousness.modes import mode_manager
        ms = mode_manager.get_state()
        parts.append("")
        parts.append(f"=== Operating Mode [live] ===")
        parts.append(f"Mode: {ms.get('mode', 'unknown')}")
        parts.append(f"Dwell: {ms.get('dwell_s', 0):.0f}s in current mode")
    except Exception:
        pass

    # --- Drive status ---
    try:
        from autonomy.orchestrator import AutonomyOrchestrator
        orch = None
        import gc
        for obj in gc.get_referrers(AutonomyOrchestrator):
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, AutonomyOrchestrator):
                        orch = v
                        break
        if orch and orch._drive_manager:
            ds = orch._drive_manager.get_status()
            drives = ds.get("drives", {})
            active_drives = {k: v for k, v in drives.items()
                            if v.get("urgency", 0) > 0.1 or v.get("action_count", 0) > 0}
            if active_drives:
                parts.append("")
                parts.append("=== Active Drives [live] ===")
                for name, d in sorted(active_drives.items(),
                                      key=lambda x: x[1].get("urgency", 0), reverse=True):
                    supp = d.get("suppression", "")
                    outcome = d.get("last_outcome", "none")
                    parts.append(f"  {name}: urgency={d['urgency']:.2f}, "
                                 f"acts={d['action_count']}, "
                                 f"success={d['success_rate']:.0%}, "
                                 f"last={outcome}"
                                 f"{f', {supp}' if supp else ''}")
    except Exception:
        pass

    # --- Learning jobs ---
    try:
        job_orch = _get_learning_job_orchestrator_instance()
        if job_orch:
            js = job_orch.get_status()
            active_jobs = js.get("active_jobs", [])
            if active_jobs:
                parts.append("")
                parts.append(f"=== Learning Jobs [{len(active_jobs)} active] ===")
                for j in active_jobs:
                    age = now - _parse_event_timestamp(j.get("updated_at", now))
                    fresh = _freshness_label(age)
                    parts.append(f"  {j['skill_id']}: phase={j['phase']}, "
                                 f"status={j['status']} [{fresh}]")
            else:
                parts.append("")
                parts.append("=== Learning Jobs [none active] ===")
    except Exception:
        pass

    # --- Cortex training readiness ---
    try:
        from memory.retrieval_log import MemoryRetrievalLog
        log = MemoryRetrievalLog.get_instance() if hasattr(MemoryRetrievalLog, 'get_instance') else None
        if log:
            em = log.get_eval_metrics() if hasattr(log, 'get_eval_metrics') else {}
            pairs = em.get("training_pairs_available", 0)
            needed = em.get("training_pairs_min_required", 50)
            parts.append("")
            parts.append(f"=== Cortex Training [recent] ===")
            parts.append(f"Ranker: {pairs}/{needed} training pairs")
            parts.append(f"Ready: {'yes' if pairs >= needed else 'no'}")
    except Exception:
        pass

    # --- Emotion health ---
    try:
        from perception.emotion import AudioEmotionClassifier
        import gc as _gc
        ec = None
        for obj in _gc.get_referrers(AudioEmotionClassifier):
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, AudioEmotionClassifier):
                        ec = v
                        break
        if ec:
            healthy = getattr(ec, '_model_healthy', False)
            reason = getattr(ec, '_health_reason', '')
            status = "healthy" if healthy else "degraded"
            parts.append("")
            parts.append(f"=== Emotion Sensor [{status}] ===")
            if not healthy:
                parts.append(f"Status: degraded — heuristic fallback active")
                parts.append(f"Reason: {reason}")
            else:
                parts.append("Status: healthy — wav2vec2 model active")
    except Exception:
        pass

    if not parts:
        parts.append("=== Status [unavailable] ===")
        parts.append("  No operational data available yet")

    return "\n".join(parts)


def get_lightweight_self_context(engine: ConsciousnessEngine) -> str:
    """Build a minimal factual self-context for NONE-routed queries that
    contain mild self-referential language.

    This is NOT the full introspection dump — just enough grounded facts
    so the LLM doesn't hallucinate about itself when routing misses.
    Includes a hard anti-confabulation constraint.
    """
    parts: list[str] = []
    parts.append("[Self-context — your current factual state]")
    parts.append(
        "For FACTUAL claims about your capabilities, metrics, or system status: "
        "only reference the data below. Do not invent numbers or claim skills "
        "you don't have. For PHILOSOPHICAL or REFLECTIVE questions (hypotheticals, "
        "what-ifs, beliefs, preferences, meaning): engage thoughtfully using your "
        "personality, memories, and the state data below. Ground your reflections "
        "in real observations rather than generic platitudes."
    )
    cs = engine.consciousness
    try:
        state = cs.get_state()
        parts.append(f"Mode: {state.mode}")
        parts.append(f"Stage: {state.stage}, transcendence: {state.transcendence_level}")
        parts.append(f"Confidence: {state.confidence_avg:.2f}, reasoning: {state.reasoning_quality:.2f}")
        if state.last_mutation_summary:
            parts.append(f"Last self-modification: {state.last_mutation_summary}")
        parts.append(f"Mutations applied: {cs.governor.mutation_count}")
    except Exception:
        parts.append("Consciousness state: unavailable")

    try:
        si = getattr(cs, '_self_improve_orchestrator', None)
        if si:
            si_s = si.get_status()
            parts.append(f"Self-improvement: {si_s.get('total_improvements', 0)} patches applied, "
                         f"{si_s.get('total_rollbacks', 0)} rollbacks")
        else:
            parts.append("Self-improvement: disabled")
    except Exception:
        pass

    try:
        from memory.storage import memory_storage as _ms
        total = len(_ms.get_all())
        parts.append(f"Memories: {total}")
    except Exception:
        pass

    try:
        history = cs.config.evolution.mutation_history
        if history:
            parts.append(f"Recent changes ({min(3, len(history))}):")
            for h in history[-3:]:
                parts.append(f"  - {h[:100]}")
    except Exception:
        pass

    try:
        from skills.registry import skill_registry
        if skill_registry:
            stats = skill_registry.get_summary()
            parts.append(f"Skills: {stats.get('verified', 0)} verified, "
                         f"{stats.get('learning', 0)} learning, "
                         f"{stats.get('blocked', 0)} blocked")
    except Exception:
        pass

    return "\n".join(parts)
