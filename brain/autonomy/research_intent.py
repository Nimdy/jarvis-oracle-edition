"""Research intent and result data structures for autonomous learning.

ResearchIntent: what Jarvis wants to learn and why.
ResearchResult: what was found, with provenance and confidence.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


ResearchScope = Literal["local_only", "external_ok"]
IntentStatus = Literal["queued", "running", "blocked", "completed", "failed", "cancelled"]
ToolHint = Literal["web", "academic", "codebase", "memory", "introspection", "any"]


@dataclass
class ResearchIntent:
    """A bounded, inspectable research request born from internal curiosity."""

    question: str
    source_event: str
    source_hint: ToolHint = "any"
    priority: float = 0.5
    scope: ResearchScope = "local_only"
    max_results: int = 5
    max_tokens: int = 2000
    timeout_s: float = 30.0
    tag_cluster: tuple[str, ...] = ()
    trigger_count: int = 1

    id: str = field(default_factory=lambda: f"ri_{uuid.uuid4().hex[:12]}")
    status: IntentStatus = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    reason: str = ""
    result: ResearchResult | None = None
    blocked_reason: str = ""
    goal_id: str = ""
    task_id: str = ""
    shadow_planner_event: str = ""
    shadow_planner_utility: float = 0.0
    shadow_planner_goal_alignment: float = 0.0
    shadow_planner_recommendation: str = ""
    shadow_planner_reason: str = ""
    golden_trace_id: str = ""
    golden_command_id: str = ""
    golden_status: str = "none"
    # ── SPARK_DESIGN §2.2 / §3 component 2 — grounding provenance (default-safe) ──
    # When this intent was seeded by a belief_validation_curiosity tension-thought,
    # belief_id names the belief whose grounding is being validated and
    # validation_target is the externally-answerable target. Both default to ""
    # so persisted JSONL stays backward-compatible (readers tolerate absence).
    # P3 uses them to emit THOUGHT_VALIDATION_OUTCOME (the teacher signal) on
    # completion, carrying whether the validation grounded/refuted the belief.
    belief_id: str = ""
    validation_target: str = ""

    def elapsed_s(self) -> float:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        if self.started_at:
            return time.time() - self.started_at
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "question": self.question,
            "source_event": self.source_event,
            "source_hint": self.source_hint,
            "priority": self.priority,
            "scope": self.scope,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "reason": self.reason,
            "tag_cluster": list(self.tag_cluster),
            "trigger_count": self.trigger_count,
            "blocked_reason": self.blocked_reason,
            "goal_id": self.goal_id,
            "task_id": self.task_id,
            "shadow_planner_event": self.shadow_planner_event,
            "shadow_planner_utility": self.shadow_planner_utility,
            "shadow_planner_goal_alignment": self.shadow_planner_goal_alignment,
            "shadow_planner_recommendation": self.shadow_planner_recommendation,
            "shadow_planner_reason": self.shadow_planner_reason,
            "golden_trace_id": self.golden_trace_id,
            "golden_command_id": self.golden_command_id,
            "golden_status": self.golden_status,
            "elapsed_s": self.elapsed_s(),
        }
        # SPARK §3 component 2 — only emit grounding provenance when present so
        # existing intents (the common case) keep their exact serialised shape.
        if self.belief_id:
            d["belief_id"] = self.belief_id
        if self.validation_target:
            d["validation_target"] = self.validation_target
        if self.result:
            d["result"] = self.result.to_dict()
        return d


# ---------------------------------------------------------------------------
# SPARK_DESIGN §2.2 / §3 component 2 / §8 P3 — the teacher signal.
#
# THOUGHT_VALIDATION_OUTCOME is emitted on a tension-seeded ResearchIntent's
# completion. It is the missing external-only teacher signal for the
# tension-thought selector (and the grounding drive). HONESTY GUARDRAIL
# (SPARK §7): the outcome is set by an EXTERNAL validator, never self-scored —
# a finding only counts as "grounded" if it is source-cited and external-scope.
# "Being corrected counts as success": a refutation still has grounded=True.
# ---------------------------------------------------------------------------

# Source types that count as an external, citeable validator (SPARK §6: web /
# academic). codebase / memory / introspection are internal — they cannot
# ground a belief against external truth, so they never set grounded=True.
_EXTERNAL_SOURCE_TYPES = frozenset({"peer_reviewed", "preprint", "web"})


def is_external_grounding(intent: "ResearchIntent", result: "ResearchResult | None") -> bool:
    """True iff ``result`` is an EXTERNAL, source-cited validation of the intent.

    External-only (SPARK §7): requires external scope, success, and at least one
    finding whose ``source_type`` is an external validator with a citation
    (url / doi). Internal tools (codebase/memory/introspection) never qualify —
    they cannot ground a belief against outside truth. View-only; never mutates.
    """
    if result is None or not getattr(result, "success", False):
        return False
    if getattr(intent, "scope", "local_only") != "external_ok":
        return False
    for f in getattr(result, "findings", []) or []:
        st = (getattr(f, "source_type", "") or "").lower()
        cited = bool(getattr(f, "url", "") or getattr(f, "doi", ""))
        if st in _EXTERNAL_SOURCE_TYPES and cited:
            return True
    return False


def emit_thought_validation_outcome(
    intent: "ResearchIntent",
    result: "ResearchResult | None",
    *,
    refuted: bool | None = None,
) -> dict[str, Any] | None:
    """Emit THOUGHT_VALIDATION_OUTCOME for a completed tension-seeded intent.

    Returns the emitted payload (for logging/tests), or None if the intent was
    NOT tension-seeded (no ``belief_id``) — in which case nothing is emitted and
    DEFAULT behavior is unchanged.

    ``grounded`` is decided EXTERNAL-ONLY via :func:`is_external_grounding`
    (never self-scored). ``refuted`` (optional) records whether the external
    evidence contradicted the belief; a refutation still counts as grounded
    (SPARK §7 — being corrected is success). The tension-thought promotion gate
    consumes this as its teacher signal.

    Safe by construction: never raises into the caller; the event bus / promotion
    update is best-effort.
    """
    belief_id = getattr(intent, "belief_id", "") or ""
    if not belief_id:
        return None  # not tension-seeded — emit nothing, change nothing.

    grounded = is_external_grounding(intent, result)
    payload: dict[str, Any] = {
        "intent_id": getattr(intent, "id", ""),
        "belief_id": belief_id,
        "validation_target": getattr(intent, "validation_target", "") or "",
        "source_event": getattr(intent, "source_event", "") or "",
        "tool_used": getattr(result, "tool_used", "") if result is not None else "",
        "finding_count": len(getattr(result, "findings", []) or []) if result is not None else 0,
        "scope": getattr(intent, "scope", "local_only"),
        "grounded": grounded,
        "refuted": bool(refuted) if refuted is not None else False,
        "refuted_known": refuted is not None,
        "timestamp": time.time(),
    }

    try:
        from consciousness.events import event_bus, THOUGHT_VALIDATION_OUTCOME
        event_bus.emit(THOUGHT_VALIDATION_OUTCOME, **payload)
    except Exception:
        pass

    # Feed the external-only teacher signal to the tension-thought promotion
    # gate (still pure shadow at P3 — this only accrues the outcomes that EARN
    # promotion later; it flips no lever now).
    try:
        from consciousness.meta_cognitive_thoughts import TensionThoughtPromotion
        TensionThoughtPromotion.get_instance().record_validation_outcome(grounded)
    except Exception:
        pass

    # NATIVE-COGNITION SEED (FINISH_ROADMAP native pivot, step 1): capture this
    # reasoning->outcome pairing as a distillation training signal. Pure SHADOW
    # accumulation — no NN, no behavior, no promotion. It is the first training
    # data from which a native reasoning specialist could LATER be distilled
    # (jump-before-dunk: collect the reps before the muscle can exist). This is the
    # opposite of qwen-verb-hacking — it grows JARVIS's own cognition, not a guard.
    #
    # Phase 0 (#3) enrichment: snapshot the reasoning-substrate grounding-coherence
    # signal (the view-only ReasoningEncoder reading the live belief field) AT
    # outcome time, so the recorded pair becomes reasoning-STATE -> outcome, not just
    # belief -> outcome. That is the feature the future native reasoner trains on:
    # "when my substrate read like THIS, did grounding land?" Best-effort, view-only.
    reasoning_signal: float | None = None
    try:
        from hemisphere.reasoning_encoder import compute_live_signal
        reasoning_signal = round(compute_live_signal(), 4)
    except Exception:
        reasoning_signal = None
    try:
        from hemisphere.distillation import distillation_collector
        distillation_collector.record(
            teacher="reasoning_validation",
            signal_type="belief_validation",
            data={
                "belief_id": belief_id,
                "validation_target": payload["validation_target"],
                "source_event": payload["source_event"],
                "tool_used": payload["tool_used"],
                "finding_count": payload["finding_count"],
                "scope": payload["scope"],
                "grounded": bool(grounded),
                "refuted": payload["refuted"],
                "reasoning_signal": reasoning_signal,
            },
            metadata={"native_pivot": "reasoning_seed",
                      "outcome": "grounded" if grounded else "ungrounded",
                      "reasoning_signal": reasoning_signal},
            origin="live",
            fidelity=1.0,
        )
    except Exception:
        pass

    return payload


@dataclass
class ResearchFinding:
    """A single piece of discovered information with provenance."""

    content: str
    provenance: str
    confidence: float = 0.5
    url: str = ""
    file_path: str = ""
    line_range: str = ""
    doi: str = ""
    doi_url: str = ""
    authors: str = ""
    year: int = 0
    venue: str = ""
    citation_count: int = 0
    influential_citation_count: int = 0
    source_type: str = ""  # peer_reviewed, preprint, web, codebase, memory
    source_provider: str = ""  # semantic_scholar, crossref, ddg
    open_access_pdf_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "content": self.content[:500],
            "provenance": self.provenance,
            "confidence": self.confidence,
            "url": self.url,
            "file_path": self.file_path,
            "line_range": self.line_range,
        }
        if self.doi:
            d["doi"] = self.doi
            d["doi_url"] = self.doi_url
        if self.authors:
            d["authors"] = self.authors
        if self.year:
            d["year"] = self.year
        if self.venue:
            d["venue"] = self.venue
        if self.citation_count:
            d["citation_count"] = self.citation_count
        if self.influential_citation_count:
            d["influential_citation_count"] = self.influential_citation_count
        if self.source_type:
            d["source_type"] = self.source_type
        if self.source_provider:
            d["source_provider"] = self.source_provider
        return d


@dataclass
class ResearchResult:
    """The outcome of executing a ResearchIntent."""

    intent_id: str
    tool_used: str
    findings: list[ResearchFinding] = field(default_factory=list)
    summary: str = ""
    raw_query: str = ""
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "tool_used": self.tool_used,
            "summary": self.summary[:500],
            "finding_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings[:5]],
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
        }
