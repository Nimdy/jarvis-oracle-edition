"""Phase 6.4 — Explainability Layer.

Builds human-readable provenance traces from internal metadata so users
can understand *why* they got a particular answer and *where* it came from.

Three capabilities:
  1. build_provenance_trace() — structured trace from _language_example_seed
  2. build_evidence_chain()   — causal chain from attribution ledger
  3. cite_sources()           — memory-backed citation list

All functions are stateless and safe to call from the hot path.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Verdict → human-readable source description ─────────────────────────

_VERDICT_DESCRIPTIONS: dict[str, str] = {
    "grounded_tool_data": "Live system data (status tool)",
    "grounded_memory_context": "Verified memory records",
    "grounded_codebase_answer": "Codebase analysis",
    "grounded_identity_status": "Biometric identity verification",
    "strict_learning_job_grounded": "Verified learning job records",
    "strict_provenance_grounded": "Verified provenance chain",
    "reflective_introspection": "Self-reflective analysis (capability-gated)",
    "bounded_introspection": "Bounded articulation (deterministic)",
    "registry_grounded_capability_status": "Capability registry lookup",
    "registry_grounded_skill_status": "Skill registry lookup",
    "native_dream_artifact": "Dream processing artifact",
    "none_route_general_conversation": "General conversation with LLM articulation, capability-gated",
}

_RESPONSE_CLASS_LABELS: dict[str, str] = {
    "self_status": "System Status Report",
    "self_introspection": "Self-Reflective Analysis",
    "recent_learning": "Recent Learning Summary",
    "recent_research": "Recent Research Findings",
    "memory_recall": "Memory Retrieval",
    "identity_answer": "Identity Verification",
    "capability_status": "Capability Status",
    "system_explanation": "Codebase Explanation",
    "learning_job_status": "Learning Job Status",
    "general_conversation": "General Conversation",
}


def build_provenance_trace(
    seed: dict[str, Any] | None,
    *,
    reply: str = "",
    conversation_id: str = "",
    ledger_entry_id: str = "",
) -> dict[str, Any]:
    """Build a structured provenance trace from a language example seed.

    Returns a dict suitable for inclusion in response metadata and
    dashboard rendering. Always safe to call — returns minimal trace
    on missing/empty seed.
    """
    if not seed:
        return {
            "available": False,
            "summary": "No provenance data available for this response.",
        }

    verdict = seed.get("provenance_verdict", "unknown")
    response_class = seed.get("response_class", "")
    confidence = seed.get("confidence", 0.0)
    native_used = seed.get("native_used", False)
    safety_flags = seed.get("safety_flags", [])
    meaning_frame = seed.get("meaning_frame", {})

    # Build human-readable source description
    source_desc = _VERDICT_DESCRIPTIONS.get(verdict, "")
    if not source_desc:
        if verdict.startswith("grounded_"):
            source_desc = "Grounded data source"
        elif verdict.startswith("native_"):
            source_desc = "Native bounded path"
        elif verdict.startswith("negative:"):
            source_desc = "Negative example (training data)"
        else:
            source_desc = "Unclassified source"

    class_label = _RESPONSE_CLASS_LABELS.get(response_class, response_class or "general")

    # Extract facts from meaning frame if available
    facts = meaning_frame.get("facts", [])
    fact_count = meaning_frame.get("fact_count", len(facts))

    # Build source citations from grounding payload
    grounding = seed.get("grounding_payload")
    citations = _extract_citations(grounding, response_class)

    trace: dict[str, Any] = {
        "available": True,
        "provenance_verdict": verdict,
        "source_description": source_desc,
        "response_class": response_class,
        "response_class_label": class_label,
        "confidence": round(confidence, 3),
        "native_used": native_used,
        "fact_count": fact_count,
        "citations": citations,
        "safety_flags": safety_flags,
        "summary": _build_summary(source_desc, class_label, confidence, native_used, fact_count),
    }

    if conversation_id:
        trace["conversation_id"] = conversation_id
    if ledger_entry_id:
        trace["ledger_entry_id"] = ledger_entry_id

    return trace


def _build_summary(
    source_desc: str,
    class_label: str,
    confidence: float,
    native_used: bool,
    fact_count: int,
) -> str:
    """Build a one-line human-readable summary."""
    path = "bounded articulation" if native_used else "LLM generation"
    conf_pct = f"{confidence * 100:.0f}%"
    parts = [f"{class_label} via {path} ({conf_pct} confidence)"]
    if source_desc:
        parts.append(f"Source: {source_desc}")
    if fact_count > 0:
        parts.append(f"{fact_count} verified facts")
    return ". ".join(parts) + "."


def _extract_citations(
    grounding: Any,
    response_class: str,
) -> list[dict[str, str]]:
    """Extract structured citations from grounding payload."""
    citations: list[dict[str, str]] = []

    if not grounding:
        return citations

    if isinstance(grounding, dict):
        # Memory recall citations
        if response_class == "memory_recall":
            mem_ctx = grounding.get("memory_context", "")
            if mem_ctx:
                citations.append({
                    "type": "memory",
                    "label": "Memory search results",
                    "detail": mem_ctx[:200] + ("..." if len(mem_ctx) > 200 else ""),
                })

        # Status tool citations
        elif response_class == "self_status":
            for key in ("autonomy", "health", "memory", "reasoning"):
                if key in grounding:
                    val = grounding[key]
                    if isinstance(val, dict):
                        citations.append({
                            "type": "system_data",
                            "label": f"Live {key} data",
                            "detail": str(val)[:150],
                        })
                    else:
                        citations.append({
                            "type": "system_data",
                            "label": f"Live {key} data",
                            "detail": str(val)[:150],
                        })

        # Learning job citations
        elif response_class in ("recent_learning", "recent_research", "learning_job_status"):
            if "jobs" in grounding:
                for job in grounding["jobs"][:3]:
                    if isinstance(job, dict):
                        citations.append({
                            "type": "learning_job",
                            "label": job.get("topic", "learning job"),
                            "detail": job.get("status", ""),
                        })
            elif "topic" in grounding:
                citations.append({
                    "type": "learning_job",
                    "label": grounding.get("topic", ""),
                    "detail": grounding.get("status", ""),
                })

        # Identity citations
        elif response_class == "identity_answer":
            if "name" in grounding or "speaker" in grounding:
                citations.append({
                    "type": "identity",
                    "label": "Biometric match",
                    "detail": f"Speaker: {grounding.get('name', grounding.get('speaker', 'unknown'))}",
                })

        # Capability/skill registry citations
        elif response_class == "capability_status":
            if "skills" in grounding:
                for skill in grounding["skills"][:5]:
                    if isinstance(skill, dict):
                        citations.append({
                            "type": "registry",
                            "label": skill.get("name", "skill"),
                            "detail": skill.get("status", ""),
                        })

    elif isinstance(grounding, str) and grounding:
        citations.append({
            "type": "text",
            "label": "Grounding context",
            "detail": grounding[:200] + ("..." if len(grounding) > 200 else ""),
        })

    return citations


def build_evidence_chain(
    conversation_id: str,
    ledger_entry_id: str = "",
) -> dict[str, Any]:
    """Build a full evidence chain from the attribution ledger.

    If ledger_entry_id is provided, traces from that entry's root chain.
    Otherwise, finds the most recent conversation entry for conversation_id.

    Returns a dict with chain entries and human-readable narrative.
    """
    try:
        from consciousness.attribution_ledger import attribution_ledger
    except Exception:
        return {"available": False, "reason": "Attribution ledger not available"}

    root_id = ledger_entry_id

    # If no specific entry, find the root for this conversation
    if not root_id and conversation_id:
        recent = attribution_ledger.query(
            subsystem="conversation",
            event_type="user_message",
            limit=20,
        )
        for entry in recent:
            if entry.get("conversation_id") == conversation_id:
                root_id = entry.get("root_entry_id") or entry.get("entry_id", "")
                break

    if not root_id:
        return {"available": False, "reason": "No ledger entries found for this conversation"}

    chain = attribution_ledger.get_chain(root_id)
    if not chain:
        return {"available": False, "reason": "Chain is empty"}

    # Build human-readable steps
    steps: list[dict[str, Any]] = []
    for entry in chain:
        step = {
            "entry_id": entry.get("entry_id", ""),
            "subsystem": entry.get("subsystem", ""),
            "event_type": entry.get("event_type", ""),
            "timestamp": entry.get("ts", 0),
            "confidence": entry.get("confidence", 0),
            "outcome": entry.get("outcome", "pending"),
            "narrative": _narrate_step(entry),
        }
        if entry.get("data"):
            # Include selected data fields, not the full blob
            data = entry["data"]
            step["detail"] = {
                k: v for k, v in data.items()
                if k in ("text", "tool", "reply_len", "latency_ms", "topic",
                         "reason", "query", "emotion", "tool_route")
                and v
            }
        steps.append(step)

    return {
        "available": True,
        "root_entry_id": root_id,
        "conversation_id": conversation_id,
        "step_count": len(steps),
        "steps": steps,
        "narrative": _build_chain_narrative(steps),
    }


def _narrate_step(entry: dict[str, Any]) -> str:
    """Convert a ledger entry into a human-readable sentence."""
    subsystem = entry.get("subsystem", "system")
    event_type = entry.get("event_type", "event")
    outcome = entry.get("outcome", "pending")
    data = entry.get("data", {})

    _NARRATIONS: dict[str, str] = {
        "conversation:user_message": "User sent a message",
        "conversation:response_complete": "System generated a response",
        "capability_gate:claim_blocked": "Capability gate blocked an unverified claim",
        "capability_gate:claim_rewritten": "Capability gate rewrote a claim",
        "learning_jobs:job_created": "A learning job was created",
        "learning_jobs:job_completed": "A learning job completed",
        "autonomy:intent_selected": "Autonomy selected an action intent",
        "autonomy:intent_executed": "Autonomy executed an action",
        "memory:retrieval": "Memory retrieval was performed",
        "memory:store": "A new memory was stored",
    }

    key = f"{subsystem}:{event_type}"
    base = _NARRATIONS.get(key, f"{subsystem} recorded {event_type}")

    # Enrich with data
    if "text" in data:
        text_preview = data["text"][:80]
        base += f': "{text_preview}"'
    elif "tool" in data:
        base += f" (tool: {data['tool']})"

    if outcome and outcome != "pending":
        base += f" [{outcome}]"

    return base


def _build_chain_narrative(steps: list[dict[str, Any]]) -> str:
    """Build a multi-line narrative from chain steps."""
    if not steps:
        return "No evidence chain available."

    lines: list[str] = []
    for i, step in enumerate(steps, 1):
        lines.append(f"{i}. {step['narrative']}")

    return "\n".join(lines)


def cite_sources(
    seed: dict[str, Any] | None,
    *,
    memory_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Build a citation list for memory-backed and grounded responses.

    Returns a list of citation dicts with type, label, provenance, and detail.
    """
    citations: list[dict[str, str]] = []

    # Citations from the language example seed
    if seed:
        trace_citations = _extract_citations(
            seed.get("grounding_payload"),
            seed.get("response_class", ""),
        )
        citations.extend(trace_citations)

    # Citations from memory retrieval results
    if memory_results:
        for mem in memory_results[:10]:
            if not isinstance(mem, dict):
                continue
            mem_id = str(mem.get("id", ""))
            payload = mem.get("payload", mem.get("formatted", ""))
            provenance = mem.get("provenance", "unknown")
            weight = mem.get("weight", 0.0)

            citations.append({
                "type": "memory",
                "label": f"Memory {mem_id[:8]}" if mem_id else "Memory record",
                "provenance": provenance,
                "detail": str(payload)[:150] + ("..." if len(str(payload)) > 150 else ""),
                "weight": str(round(weight, 3)) if weight else "",
            })

    return citations


# ── Compact trace for response metadata (hot path) ──────────────────────

def compact_trace(seed: dict[str, Any] | None) -> dict[str, Any]:
    """Build a minimal provenance summary for inclusion in response events.

    Kept small to avoid bloating the event bus / broadcast payloads.
    """
    if not seed:
        return {
            "provenance": "fallback_unclassified",
            "source": "fallback:missing_language_seed",
            "confidence": 0.0,
            "native": False,
            "response_class": "unknown",
            "fallback": True,
        }

    verdict = seed.get("provenance_verdict", "unknown")
    return {
        "provenance": verdict,
        "source": _VERDICT_DESCRIPTIONS.get(verdict, ""),
        "confidence": round(seed.get("confidence", 0.0), 3),
        "native": seed.get("native_used", False),
        "response_class": seed.get("response_class", ""),
    }
