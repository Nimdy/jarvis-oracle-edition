"""Code Patch -- unified diff representation returned by AI providers.

Includes original_content population (via codebase_tool), AST validation,
denied-pattern checking, and diff budget enforcement.
"""

from __future__ import annotations

import ast
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from self_improve.patch_plan import (
    check_ast_forbidden_calls,
    check_denied_patterns,
    detect_capability_escalation,
    MAX_FILES_CHANGED,
    MAX_LINES_CHANGED,
    MAX_NEW_FILES,
)


@dataclass
class FileDiff:
    path: str
    original_content: str = ""
    new_content: str = ""
    diff: str = ""


@dataclass
class CodePatch:
    id: str = field(default_factory=lambda: f"patch_{uuid.uuid4().hex[:10]}")
    plan_id: str = ""
    timestamp: float = field(default_factory=time.time)
    provider: str = ""
    files: list[FileDiff] = field(default_factory=list)
    description: str = ""
    test_instructions: str = ""
    confidence: float = 0.5
    requires_approval: bool = False

    def validate(self) -> list[str]:
        """Run all validation checks: denied patterns, AST forbidden calls, syntax.

        When original_content is available, denied-pattern checks run only on
        *new* lines (lines present in new_content but not in original_content)
        to avoid false positives from pre-existing code.
        """
        violations: list[str] = []
        for fd in self.files:
            content = fd.new_content or fd.diff
            if not content:
                continue

            if fd.original_content and fd.new_content:
                old_lines = set(fd.original_content.splitlines())
                new_only = "\n".join(
                    line for line in fd.new_content.splitlines()
                    if line not in old_lines
                )
                violations.extend(
                    f"{fd.path}: {v}" for v in check_denied_patterns(new_only)
                )
            else:
                violations.extend(
                    f"{fd.path}: {v}" for v in check_denied_patterns(content)
                )

            if fd.new_content:
                violations.extend(
                    f"{fd.path}: {v}" for v in check_ast_forbidden_calls(fd.new_content, fd.path)
                )

        return violations

    def validate_syntax(self) -> list[str]:
        """Check that all new_content files parse as valid Python."""
        errors: list[str] = []
        for fd in self.files:
            if not fd.new_content:
                continue
            try:
                ast.parse(fd.new_content, filename=fd.path)
            except SyntaxError as exc:
                errors.append(f"{fd.path}:{exc.lineno}: {exc.msg}")
        return errors

    def check_capability_escalation(self) -> list[str]:
        """Detect if any file diff introduces new capabilities needing approval."""
        escalations: list[str] = []
        for fd in self.files:
            if fd.new_content:
                esc = detect_capability_escalation(
                    fd.original_content, fd.new_content, fd.path
                )
                escalations.extend(f"{fd.path}: {e}" for e in esc)
        if escalations:
            self.requires_approval = True
        return escalations

    def validate_diff_budget(self) -> list[str]:
        """Check patch size against hard limits."""
        violations: list[str] = []
        n_files = len(self.files)
        n_new = sum(1 for fd in self.files if not fd.original_content and fd.new_content)

        if n_files > MAX_FILES_CHANGED:
            violations.append(f"Too many files ({n_files}), max {MAX_FILES_CHANGED}")

        if n_new > MAX_NEW_FILES:
            violations.append(f"Too many new files ({n_new}), max {MAX_NEW_FILES}")

        total_lines = 0
        for fd in self.files:
            if fd.new_content and fd.original_content:
                old_lines = fd.original_content.count("\n") + 1
                new_lines = fd.new_content.count("\n") + 1
                total_lines += abs(new_lines - old_lines) + _count_changed_lines(
                    fd.original_content, fd.new_content
                )
            elif fd.new_content:
                total_lines += fd.new_content.count("\n") + 1

        if total_lines > MAX_LINES_CHANGED:
            violations.append(f"Too many lines changed ({total_lines}), max {MAX_LINES_CHANGED}")

        return violations

    def populate_original_content(self) -> None:
        """Read original file contents via codebase_tool for diffing."""
        try:
            from tools.codebase_tool import codebase_index
        except ImportError:
            return

        for fd in self.files:
            if fd.original_content:
                continue
            rel = fd.path.replace("brain/", "")
            raw = codebase_index.read_file(rel)
            if raw:
                # strip line numbers from read_file output
                lines = []
                for line in raw.splitlines():
                    if "|" in line[:6]:
                        lines.append(line.split("|", 1)[1])
                    else:
                        lines.append(line)
                fd.original_content = "\n".join(lines)

    def file_count(self) -> int:
        return len(self.files)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "plan_id": self.plan_id,
            "provider": self.provider,
            "file_count": self.file_count(),
            "description": self.description,
            "confidence": self.confidence,
            "requires_approval": self.requires_approval,
            "files": [{"path": f.path, "has_diff": bool(f.diff), "has_original": bool(f.original_content)} for f in self.files],
        }


def _count_changed_lines(old: str, new: str) -> int:
    """Quick line-level diff count without importing difflib."""
    old_lines = set(old.splitlines())
    new_lines = set(new.splitlines())
    return len(new_lines - old_lines)
