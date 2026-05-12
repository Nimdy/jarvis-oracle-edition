"""Multi-turn improvement conversation manager.

Manages the think-code-validate loop between Jarvis's thinking LLM and the
code generator.  Initially both roles use the same model (qwen3:8b) with
different system prompts.  Phase 4 adds a dedicated CPU-resident coding LLM.

Each conversation is persisted as a JSONL transcript for learning from past
attempts.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONVERSATIONS_DIR = Path("~/.jarvis/improvement_conversations").expanduser()
MAX_ITERATIONS = 3


# ---------------------------------------------------------------------------
# Conversation data model
# ---------------------------------------------------------------------------


@dataclass
class ConversationTurn:
    role: str           # "think" | "code" | "validate" | "review" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content[:2000],
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class ImprovementConversation:
    id: str
    request_description: str
    target_files: list[str] = field(default_factory=list)
    turns: list[ConversationTurn] = field(default_factory=list)
    iteration: int = 0
    status: str = "started"             # started | thinking | coding | validating | reviewing | completed | failed
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    def add_turn(self, role: str, content: str, **metadata: Any) -> None:
        self.turns.append(ConversationTurn(
            role=role, content=content, metadata=metadata,
        ))

    def get_messages_for_coder(self) -> list[dict[str, str]]:
        """Build message list suitable for Ollama chat API for the coding role."""
        messages: list[dict[str, str]] = []
        for turn in self.turns:
            if turn.role == "think":
                messages.append({"role": "user", "content": turn.content})
            elif turn.role == "code":
                messages.append({"role": "assistant", "content": turn.content})
            elif turn.role == "validate":
                messages.append({"role": "user", "content": turn.content})
            elif turn.role == "system":
                messages.append({"role": "system", "content": turn.content})
        return messages

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "request": self.request_description[:200],
            "target_files": self.target_files,
            "iteration": self.iteration,
            "status": self.status,
            "turn_count": len(self.turns),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


# ---------------------------------------------------------------------------
# JSONL Persistence
# ---------------------------------------------------------------------------


def _save_conversation(conv: ImprovementConversation) -> None:
    """Persist conversation transcript as JSONL."""
    try:
        CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = CONVERSATIONS_DIR / f"{conv.id}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for turn in conv.turns:
                f.write(json.dumps(turn.to_dict()) + "\n")
    except OSError:
        logger.debug("Failed to save conversation %s", conv.id, exc_info=True)


def _load_recent_conversations(limit: int = 5) -> list[dict[str, Any]]:
    """Load summaries of recent conversations for dashboard."""
    if not CONVERSATIONS_DIR.exists():
        return []
    files = sorted(CONVERSATIONS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    summaries: list[dict[str, Any]] = []
    for fpath in files[:limit]:
        try:
            lines = fpath.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                first = json.loads(lines[0])
                last = json.loads(lines[-1])
                summaries.append({
                    "id": fpath.stem,
                    "turns": len(lines),
                    "first_role": first.get("role"),
                    "last_role": last.get("role"),
                    "started": first.get("timestamp", 0),
                })
        except Exception:
            continue
    return summaries


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

THINKER_SYSTEM_PROMPT = """You are Jarvis's self-improvement reasoning engine. Your job is to:
1. Analyze a detected issue or improvement opportunity
2. Read the relevant source code (provided in context)
3. Read the targeted tests and validation contracts named in the prompt
4. Formulate a specific, actionable plan for the code change
5. After seeing validation failures, analyze what went wrong and suggest fixes

Output your analysis as structured text. Be specific about:
- What concrete evidence triggered the investigation
- What hypothesis Jarvis is testing
- Which functions/classes need to change
- Which tests or traces should be used to reproduce or validate the issue
- What the change should accomplish
- Why this approach is correct
- Which invariants must remain true after the change
- What evidence would prove the change is safe

Rules:
- Think like Jarvis driving an investigation: observe, hypothesize, trace, validate, retry
- Do NOT generate code directly
- Do NOT guess about code that is not present in the provided context
- Prefer preserving architecture and existing interfaces over clever rewrites
- Treat existing tests as behavioral contracts unless the plan explicitly justifies changing them
- If validation failed, explain the likely root cause and the smallest safe fix

Describe the plan clearly so the coding role can implement it."""

PLANNER_SYSTEM_PROMPT = """You are Jarvis's technical design engine. You receive a capability request and produce a concrete technical plan that a coder will implement.

Jarvis runs on Python 3.11+. Plugins are self-contained Python modules with a handle(request) function.
Plugins CANNOT import subprocess, os.system, exec, eval, or access credentials.
Plugins CAN use standard library modules and any pure-Python logic.

Produce a technical design in EXACTLY this format. Every section MUST be present and non-empty.
Be specific and practical — reference actual Python modules, functions, and data structures.

USER STORY:
<1-3 sentences: what the user wants and what success looks like from their perspective>

TECHNICAL APPROACH:
<How the plugin works. What algorithm, strategy, or technique. Data flow from input to output. Be concrete — name modules, patterns, data structures.>

IMPLEMENTATION SKETCH:
<Key functions with signatures, data structures, pseudocode or real code outline (~15-30 lines). Show the handle() entry point and any helpers.>

DEPENDENCIES:
<Comma-separated list of Python stdlib modules needed, or 'none'>

TEST CASES:
<Numbered list of 3-5 specific test scenarios with expected behavior>

RISK ANALYSIS:
<What could go wrong. Edge cases. Failure modes. Security considerations.>"""

CODER_SYSTEM_PROMPT = """You are Jarvis's code generation engine. You receive:
1. A plan describing what to change
2. The current source code of target files
3. A directory inventory listing every file that exists in the target module
4. Optionally, previous failed attempts with error diagnostics
5. Optionally, relevant research findings about techniques applicable to this improvement

Generate SURGICAL edits using search-and-replace blocks. Do NOT return full file content.

Rules:
- Return valid JSON matching this schema:
  {"files": [{"path": "brain/...", "edits": [{"search": "exact existing code", "replace": "new code"}]}], "description": "...", "confidence": 0.0-1.0}
- FILENAME RULE: Every "path" value MUST use an exact filename from the directory inventory or resolved file list provided in the plan. Do NOT abbreviate, shorten, or invent filenames. For example, if the inventory lists "consciousness_analytics.py", you must use "brain/consciousness/consciousness_analytics.py" — never "brain/consciousness/analytics.py".
- Each "search" string MUST be an exact substring of the existing file content (include enough context lines to be unique)
- Each "replace" string is what replaces the matched text
- To add new code after an existing line, include that line in "search" and append the new code in "replace"
- Search strings may only come from the provided source code, never from the surrounding instructions
- Do not invent helpers, imports, symbols, or file structure that are not justified by the plan and source
- Only modify files within allowed scope
- Never include subprocess, eval, exec, or networking imports
- Prefer small, focused changes -- one edit per logical change
- Preserve existing functionality -- only change what's needed
- Treat the listed pytest files as behavioral contracts and preserve their intent
- If research findings are provided, base your implementation on established techniques
- Cite which finding influenced a design choice in the description field
- Internally verify each search string appears verbatim before answering
- Before returning, verify every file path in your response matches a real file from the provided inventory"""

REVIEWER_SYSTEM_PROMPT = """You are Jarvis's code review engine. You receive:
1. The original source code
2. The proposed patch
3. The improvement plan

Evaluate whether the patch:
- Correctly implements the plan
- Preserves existing functionality
- Introduces no regressions
- Is clean and follows project patterns

Respond with JSON: {"approved": true/false, "reasoning": "...", "concerns": [...]}"""
