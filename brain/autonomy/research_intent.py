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
        if self.result:
            d["result"] = self.result.to_dict()
        return d


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
