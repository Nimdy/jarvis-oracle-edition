"""Golden Words command authority contract and parser.

This module defines the deterministic operator command plane:
    Jarvis, GOLDEN COMMAND <EXACT COMMAND>

Rules:
- Prefix is case-insensitive.
- Command body is exact-match after boundary cleanup + whitespace collapse.
- No fuzzy aliases, no synonym expansion, no natural-language fallback.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, Literal

GoldenAuthorityClass = Literal["informational", "control", "privileged", "destructive"]
GoldenStatus = Literal["none", "invalid", "blocked", "unauthorized", "executed"]

_BOUNDARY_PUNCT = " \t\r\n.,;:!?\"'`~*_()[]{}<>-–—"
_MULTI_WS_RE = re.compile(r"\s+")
_INNER_SEPARATOR_RE = re.compile(r"[.,;:!?]+")
_PREFIX_RE = re.compile(
    r"^\s*(?:jarvis)\s*[,.;:!?-]*\s*golden\s+command\b",
    re.IGNORECASE,
)
_PREFIX_BARE_RE = re.compile(
    r"^\s*[,.;:!?-]*\s*golden\s+command\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GoldenCommandDefinition:
    command_id: str
    canonical_body: str
    target_route: str
    authority_class: GoldenAuthorityClass
    requires_confirmation: bool = False
    allows_side_effects: bool = False
    allowed_subsystems: tuple[str, ...] = ()
    expected_arguments: tuple[str, ...] = ()
    audit_level: str = "standard"
    operation: str = ""
    default_args: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class GoldenCommandContext:
    trace_id: str
    issued_at: float
    expires_at: float
    operator_text: str
    command_body_raw: str
    normalized_body: str
    command_id: str
    canonical_body: str
    authority_class: GoldenAuthorityClass
    recognized_prefix: bool
    recognized_exactly: bool
    golden_status: GoldenStatus
    block_reason: str = ""
    requires_confirmation: bool = False
    allows_side_effects: bool = False
    allowed_subsystems: tuple[str, ...] = ()
    expected_arguments: tuple[str, ...] = ()
    audit_level: str = "standard"
    operation: str = ""
    argument_text: str = ""

    def with_outcome(self, status: GoldenStatus, block_reason: str = "") -> GoldenCommandContext:
        return replace(self, golden_status=status, block_reason=block_reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "operator_text": self.operator_text,
            "command_body_raw": self.command_body_raw,
            "normalized_body": self.normalized_body,
            "command_id": self.command_id,
            "canonical_body": self.canonical_body,
            "authority_class": self.authority_class,
            "recognized_prefix": self.recognized_prefix,
            "recognized_exactly": self.recognized_exactly,
            "golden_status": self.golden_status,
            "block_reason": self.block_reason,
            "requires_confirmation": self.requires_confirmation,
            "allows_side_effects": self.allows_side_effects,
            "allowed_subsystems": list(self.allowed_subsystems),
            "expected_arguments": list(self.expected_arguments),
            "audit_level": self.audit_level,
            "operation": self.operation,
            "argument_text": self.argument_text,
        }


@dataclass(frozen=True)
class GoldenParseResult:
    context: GoldenCommandContext
    command: GoldenCommandDefinition | None


_COMMANDS: tuple[GoldenCommandDefinition, ...] = (
    GoldenCommandDefinition(
        command_id="GW_STATUS",
        canonical_body="STATUS",
        target_route="STATUS",
        authority_class="informational",
        allowed_subsystems=("consciousness", "operations"),
        operation="status",
    ),
    GoldenCommandDefinition(
        command_id="GW_INTROSPECTION_STATUS",
        canonical_body="INTROSPECTION STATUS",
        target_route="INTROSPECTION",
        authority_class="informational",
        allowed_subsystems=("consciousness", "memory", "skills"),
        operation="introspection_status",
    ),
    GoldenCommandDefinition(
        command_id="GW_MEMORY_STATUS",
        canonical_body="MEMORY STATUS",
        target_route="MEMORY",
        authority_class="informational",
        allowed_subsystems=("memory",),
        operation="memory_status",
        default_args=(("action", "summary"),),
    ),
    GoldenCommandDefinition(
        command_id="GW_VISION_STATUS",
        canonical_body="VISION STATUS",
        target_route="VISION",
        authority_class="informational",
        allowed_subsystems=("perception",),
        operation="vision_status",
    ),
    GoldenCommandDefinition(
        command_id="GW_CODEBASE_STATUS",
        canonical_body="CODEBASE STATUS",
        target_route="CODEBASE",
        authority_class="informational",
        allowed_subsystems=("codebase",),
        operation="codebase_status",
    ),
    GoldenCommandDefinition(
        command_id="GW_GOAL_STATUS",
        canonical_body="GOAL STATUS",
        target_route="INTROSPECTION",
        authority_class="informational",
        allowed_subsystems=("goals", "autonomy"),
        operation="goal_status",
        default_args=(("golden_query_override", "what are your pending goals"),),
    ),
    GoldenCommandDefinition(
        command_id="GW_RESEARCH_WEB",
        canonical_body="RESEARCH WEB",
        target_route="WEB_SEARCH",
        authority_class="privileged",
        allowed_subsystems=("autonomy", "tools"),
        operation="research_web",
        default_args=(("golden_query_override", "jarvis system reliability best practices"),),
    ),
    GoldenCommandDefinition(
        command_id="GW_RESEARCH_ACADEMIC",
        canonical_body="RESEARCH ACADEMIC",
        target_route="ACADEMIC_SEARCH",
        authority_class="privileged",
        allowed_subsystems=("autonomy", "tools"),
        operation="research_academic",
        default_args=(("golden_query_override", "autonomous agent reliability and calibration"),),
    ),
    GoldenCommandDefinition(
        command_id="GW_VERIFY_CODEBASE",
        canonical_body="VERIFY CODEBASE",
        target_route="CODEBASE",
        authority_class="privileged",
        allowed_subsystems=("codebase", "self_improve"),
        operation="verify_codebase",
        default_args=(("golden_query_override", "verify current codebase health and high-risk defects"),),
    ),
    GoldenCommandDefinition(
        command_id="GW_GOAL_PAUSE",
        canonical_body="GOAL PAUSE",
        target_route="NONE",
        authority_class="control",
        allows_side_effects=True,
        allowed_subsystems=("goals",),
        operation="goal_pause",
    ),
    GoldenCommandDefinition(
        command_id="GW_GOAL_RESUME",
        canonical_body="GOAL RESUME",
        target_route="NONE",
        authority_class="control",
        allows_side_effects=True,
        allowed_subsystems=("goals",),
        operation="goal_resume",
    ),
    GoldenCommandDefinition(
        command_id="GW_CANCEL_CURRENT_TASK",
        canonical_body="CANCEL CURRENT TASK",
        target_route="NONE",
        authority_class="control",
        allows_side_effects=True,
        allowed_subsystems=("goals", "autonomy"),
        operation="cancel_current_task",
    ),
    GoldenCommandDefinition(
        command_id="GW_SELF_IMPROVE_DRY_RUN",
        canonical_body="SELF IMPROVE DRY RUN",
        target_route="SELF_IMPROVE",
        authority_class="privileged",
        allowed_subsystems=("self_improve",),
        operation="self_improve_dry_run",
    ),
    GoldenCommandDefinition(
        command_id="GW_SELF_IMPROVE_EXECUTE_CONFIRM",
        canonical_body="SELF IMPROVE EXECUTE CONFIRM",
        target_route="SELF_IMPROVE",
        authority_class="destructive",
        allows_side_effects=True,
        allowed_subsystems=("self_improve",),
        operation="self_improve_execute",
    ),
    GoldenCommandDefinition(
        command_id="GW_ACQUIRE",
        canonical_body="ACQUIRE",
        target_route="ACQUISITION",
        authority_class="privileged",
        allows_side_effects=True,
        allowed_subsystems=("acquisition",),
        operation="acquire",
        expected_arguments=("intent_text",),
    ),
    GoldenCommandDefinition(
        command_id="GW_ACQUISITION_STATUS",
        canonical_body="ACQUISITION STATUS",
        target_route="ACQUISITION",
        authority_class="informational",
        allowed_subsystems=("acquisition",),
        operation="acquisition_status",
    ),
)

_BY_CANONICAL_BODY = {
    cmd.canonical_body: cmd
    for cmd in _COMMANDS
}


def _strip_boundary_punctuation(text: str) -> str:
    return text.strip(_BOUNDARY_PUNCT)


def normalize_command_body(command_body: str) -> str:
    stripped = _strip_boundary_punctuation(command_body or "")
    if not stripped:
        return ""
    # STT often inserts comma/period pauses between command tokens.
    # Treat these separators as spaces while still requiring exact canonical words.
    normalized = _INNER_SEPARATOR_RE.sub(" ", stripped)
    return _MULTI_WS_RE.sub(" ", normalized).strip()


def parse_golden_command(
    user_text: str,
    *,
    ttl_s: float = 120.0,
    allow_bare_prefix: bool = False,
) -> GoldenParseResult | None:
    """Parse Golden command prefix and exact command body.

    Returns:
        - None when no Golden prefix is present.
        - GoldenParseResult(command=None) when prefix exists but body is invalid.
        - GoldenParseResult(command=...) for exact recognized commands.
    """
    source = user_text or ""
    match = _PREFIX_RE.match(source)
    if match is None and allow_bare_prefix:
        match = _PREFIX_BARE_RE.match(source)
    if not match:
        return None

    now = time.time()
    body_raw = source[match.end():]
    normalized_body = normalize_command_body(body_raw)

    trace_id = f"gw_{uuid.uuid4().hex[:12]}"
    expires_at = now + max(1.0, float(ttl_s))

    if not normalized_body:
        ctx = GoldenCommandContext(
            trace_id=trace_id,
            issued_at=now,
            expires_at=expires_at,
            operator_text=source.strip(),
            command_body_raw=body_raw,
            normalized_body="",
            command_id="",
            canonical_body="",
            authority_class="control",
            recognized_prefix=True,
            recognized_exactly=False,
            golden_status="invalid",
            block_reason="missing_command_body",
        )
        return GoldenParseResult(context=ctx, command=None)

    canonical_lookup = normalized_body.upper()
    command = _BY_CANONICAL_BODY.get(canonical_lookup)

    # Prefix match for commands that expect trailing arguments (e.g. ACQUIRE <text>)
    argument_text = ""
    if command is None:
        for cmd in _COMMANDS:
            if cmd.expected_arguments and canonical_lookup.startswith(cmd.canonical_body + " "):
                command = cmd
                argument_text = normalized_body[len(cmd.canonical_body):].strip()
                break

    if command is None:
        ctx = GoldenCommandContext(
            trace_id=trace_id,
            issued_at=now,
            expires_at=expires_at,
            operator_text=source.strip(),
            command_body_raw=body_raw,
            normalized_body=normalized_body,
            command_id="",
            canonical_body="",
            authority_class="control",
            recognized_prefix=True,
            recognized_exactly=False,
            golden_status="invalid",
            block_reason="unknown_command_body",
        )
        return GoldenParseResult(context=ctx, command=None)

    ctx = GoldenCommandContext(
        trace_id=trace_id,
        issued_at=now,
        expires_at=expires_at,
        operator_text=source.strip(),
        command_body_raw=body_raw,
        normalized_body=normalized_body,
        command_id=command.command_id,
        canonical_body=command.canonical_body,
        authority_class=command.authority_class,
        recognized_prefix=True,
        recognized_exactly=True,
        golden_status="executed",
        requires_confirmation=command.requires_confirmation,
        allows_side_effects=command.allows_side_effects,
        allowed_subsystems=command.allowed_subsystems,
        expected_arguments=command.expected_arguments,
        audit_level=command.audit_level,
        operation=command.operation,
        argument_text=argument_text,
    )
    return GoldenParseResult(context=ctx, command=command)


def list_canonical_commands() -> list[str]:
    return [cmd.canonical_body for cmd in _COMMANDS]


def with_golden_outcome(
    context: GoldenCommandContext | None,
    *,
    status: GoldenStatus,
    block_reason: str = "",
) -> GoldenCommandContext | None:
    if context is None:
        return None
    return context.with_outcome(status=status, block_reason=block_reason)

