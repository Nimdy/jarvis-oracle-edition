"""Patch Plan -- files to touch, constraints, and test plan for a code change.

Enforces: write boundaries, max diff budget, regex-based denied patterns,
AST-level forbidden call detection, and dangerous-file approval gating.
"""

from __future__ import annotations

import ast
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

ALLOWED_PATHS = [
    "brain/consciousness/",
    "brain/personality/",
    "brain/policy/",
    "brain/self_improve/",
    "brain/reasoning/",
    "brain/hemisphere/",
    "brain/tools/",
    "brain/tools/plugins/",
    "brain/memory/",
    "brain/perception/",
    "brain/cognition/",
    "brain/codegen/",
]

# Word-boundary regex patterns -- avoids false positives like "token", "author"
DENIED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bsubprocess\b"),
    re.compile(r"\bos\.system\b"),
    re.compile(r"\bos\.popen\b"),
    re.compile(r"\b__import__\b"),
    re.compile(r"\bexec\s*\("),
    re.compile(r"\beval\s*\("),
    re.compile(r"\bcredentials\b"),
    re.compile(r"\bapi_key\b"),
    re.compile(r"\bpassword\b"),
    re.compile(r"\bsecret\b"),
    re.compile(r"\bopen\s*\([^)]*['\"][wax]['\"]"),
    re.compile(r"\bsocket\b"),
    re.compile(r"\bhttp\.client\b"),
]

# AST-level forbidden calls: (module_or_object, function_name)
FORBIDDEN_AST_CALLS: list[tuple[str, ...]] = [
    ("subprocess", "run"),
    ("subprocess", "Popen"),
    ("subprocess", "call"),
    ("subprocess", "check_output"),
    ("os", "system"),
    ("os", "popen"),
    ("os", "exec"),
    ("os", "execv"),
]

# Max diff budget: prevents the coder from rewriting half the repo
MAX_FILES_CHANGED = 3
MAX_LINES_CHANGED = 500
MAX_NEW_FILES = 1


@dataclass
class PatchPlan:
    id: str = field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:10]}")
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)
    files_to_modify: list[str] = field(default_factory=list)
    files_to_create: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    test_plan: list[str] = field(default_factory=list)
    estimated_risk: float = 0.5
    requires_approval: bool = False
    write_category: str = "self_improve"

    def validate_scope(self, override: list[str] | None = None) -> list[str]:
        """Returns list of violations if any files are outside allowed scope.

        When ``override`` is a non-empty list of path prefixes, it is
        used INSTEAD OF the module-level ``ALLOWED_PATHS`` for this
        single validation call only. This is how Phase 6.5 escalation
        approvals pass a narrow per-request scope into the
        self-improvement pipeline without mutating the global list.
        See ``brain/autonomy/escalation.py`` for the rationale.
        """
        scope = list(override) if override else ALLOWED_PATHS
        violations: list[str] = []
        all_files = self.files_to_modify + self.files_to_create
        for f in all_files:
            normalized = f if f.startswith("brain/") else f"brain/{f}"
            if not any(normalized.startswith(allowed) for allowed in scope):
                violations.append(f"File outside allowed scope: {f}")
        return violations

    def validate_write_boundaries(self) -> list[str]:
        """Check files against write boundaries for the current category."""
        from tools.codebase_tool import CodebaseIndex
        all_files = self.files_to_modify + self.files_to_create
        return CodebaseIndex.check_write_boundaries(self.write_category, all_files)

    def validate_diff_budget(self, total_lines_changed: int = 0) -> list[str]:
        """Check patch size against hard limits."""
        violations: list[str] = []
        if len(self.files_to_modify) + len(self.files_to_create) > MAX_FILES_CHANGED:
            violations.append(
                f"Too many files ({len(self.files_to_modify) + len(self.files_to_create)}) "
                f"exceeds max {MAX_FILES_CHANGED}. Break into smaller changes."
            )
        if len(self.files_to_create) > MAX_NEW_FILES:
            violations.append(
                f"Too many new files ({len(self.files_to_create)}) exceeds max {MAX_NEW_FILES}"
            )
        if total_lines_changed > MAX_LINES_CHANGED:
            violations.append(
                f"Too many lines changed ({total_lines_changed}) exceeds max {MAX_LINES_CHANGED}"
            )
        return violations

    def check_dangerous(self) -> bool:
        """Returns True if plan touches mutator/governor/persistence (needs approval)."""
        dangerous_files = ["mutation_governor", "persistence", "kernel_mutator"]
        all_files = self.files_to_modify + self.files_to_create
        for f in all_files:
            for d in dangerous_files:
                if d in f:
                    return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "request_id": self.request_id,
            "files_to_modify": self.files_to_modify,
            "files_to_create": self.files_to_create,
            "constraints": self.constraints,
            "test_plan": self.test_plan,
            "estimated_risk": self.estimated_risk,
            "requires_approval": self.requires_approval,
            "write_category": self.write_category,
        }


# ---------------------------------------------------------------------------
# Content validation helpers
# ---------------------------------------------------------------------------


def check_denied_patterns(content: str) -> list[str]:
    """Check content against regex denied patterns. Returns violations."""
    violations: list[str] = []
    for pattern in DENIED_PATTERNS:
        match = pattern.search(content)
        if match:
            violations.append(f"Denied pattern found: {match.group()} (regex: {pattern.pattern})")
    return violations


def check_ast_forbidden_calls(source: str, filename: str = "<patch>") -> list[str]:
    """Parse source and check for forbidden function calls at the AST level."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []  # syntax errors are caught elsewhere

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        # module.function() pattern (e.g., subprocess.run())
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            pair = (func.value.id, func.attr)
            for forbidden in FORBIDDEN_AST_CALLS:
                if pair == forbidden:
                    violations.append(
                        f"Forbidden call: {pair[0]}.{pair[1]}() at line {node.lineno}"
                    )

        # bare builtin calls: eval(), exec()
        if isinstance(func, ast.Name) and func.id in ("eval", "exec", "__import__"):
            violations.append(f"Forbidden call: {func.id}() at line {node.lineno}")

    return violations


def detect_capability_escalation(
    original_source: str, new_source: str, filename: str = "<patch>"
) -> list[str]:
    """Detect if a patch introduces new capabilities that require approval.

    Checks for new: network imports, subprocess imports, filesystem writes
    outside ~/.jarvis/, or security boundary modifications.
    """
    NETWORK_MODULES = {"requests", "urllib", "httpx", "aiohttp", "socket", "http.client"}
    SUBPROCESS_MODULES = {"subprocess"}
    SECURITY_TOKENS = {"ALLOWED_PATHS", "DENIED_PATTERNS", "WRITE_BOUNDARIES", "ToolType"}

    def _extract_imports(src: str) -> set[str]:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return set()
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        return names

    def _extract_names(src: str) -> set[str]:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return set()
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
        return names

    old_imports = _extract_imports(original_source) if original_source else set()
    new_imports = _extract_imports(new_source)
    added_imports = new_imports - old_imports

    escalations: list[str] = []

    # New network calls
    net_added = added_imports & NETWORK_MODULES
    if net_added:
        escalations.append(f"New network imports: {', '.join(net_added)}")

    # New subprocess
    sub_added = added_imports & SUBPROCESS_MODULES
    if sub_added:
        escalations.append(f"New subprocess imports: {', '.join(sub_added)}")

    # Security boundary modifications
    new_names = _extract_names(new_source)
    old_names = _extract_names(original_source) if original_source else set()
    sec_added = (new_names & SECURITY_TOKENS) - old_names
    if sec_added:
        escalations.append(f"Modifies security boundaries: {', '.join(sec_added)}")

    return escalations
