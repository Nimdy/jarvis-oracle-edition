"""Tests for self_improve/code_patch.py — differential validation and safety.

The critical behavior here is the DIFFERENTIAL denied-pattern check:
when original_content is available, only NEW lines are scanned for violations.
This prevents false positives from pre-existing code while still catching
injected dangers. A bug in this differential logic could either:
  - Miss real violations (dangerous)
  - False-positive on pre-existing code (annoying but safe)

Covers:
  - validate() differential analysis (new lines only when original available)
  - validate() full scan (when no original)
  - validate_syntax() AST parsing
  - check_capability_escalation() side-effect on requires_approval
  - validate_diff_budget() file/line counting
  - _count_changed_lines() helper
  - to_dict() serialization
  - FileDiff dataclass
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from self_improve.code_patch import CodePatch, FileDiff, _count_changed_lines


# ---------------------------------------------------------------------------
# validate() — differential denied-pattern analysis
# ---------------------------------------------------------------------------

class TestValidateDifferential:
    """When original_content is present, only NEW lines should be scanned."""

    def test_preexisting_subprocess_not_flagged(self):
        fd = FileDiff(
            path="brain/test.py",
            original_content="import subprocess\nx = 1\n",
            new_content="import subprocess\nx = 2\n",
        )
        patch = CodePatch(files=[fd])
        violations = patch.validate()
        denied = [v for v in violations if "Denied pattern" in v]
        assert len(denied) == 0, f"Pre-existing subprocess flagged: {denied}"

    def test_new_subprocess_is_flagged(self):
        fd = FileDiff(
            path="brain/test.py",
            original_content="x = 1\n",
            new_content="x = 1\nimport subprocess\n",
        )
        patch = CodePatch(files=[fd])
        violations = patch.validate()
        assert any("subprocess" in v.lower() for v in violations)

    def test_new_eval_in_diff_flagged(self):
        fd = FileDiff(
            path="brain/test.py",
            original_content="x = 1\n",
            new_content="x = 1\nresult = eval(expr)\n",
        )
        patch = CodePatch(files=[fd])
        violations = patch.validate()
        assert len(violations) >= 1

    def test_no_original_full_scan(self):
        fd = FileDiff(
            path="brain/test.py",
            original_content="",
            new_content="import subprocess\n",
        )
        patch = CodePatch(files=[fd])
        violations = patch.validate()
        assert any("subprocess" in v.lower() for v in violations)

    def test_clean_diff_passes(self):
        fd = FileDiff(
            path="brain/test.py",
            original_content="x = 1\n",
            new_content="x = 2\ny = 3\n",
        )
        patch = CodePatch(files=[fd])
        assert patch.validate() == []

    def test_empty_files_passes(self):
        patch = CodePatch(files=[])
        assert patch.validate() == []

    def test_empty_content_skipped(self):
        fd = FileDiff(path="brain/test.py", original_content="", new_content="")
        patch = CodePatch(files=[fd])
        assert patch.validate() == []


# ---------------------------------------------------------------------------
# validate() — AST forbidden calls (always run on full new_content)
# ---------------------------------------------------------------------------

class TestValidateASTCalls:
    def test_subprocess_run_in_new_content(self):
        fd = FileDiff(
            path="brain/test.py",
            original_content="x = 1\n",
            new_content="import subprocess\nsubprocess.run(['ls'])\n",
        )
        patch = CodePatch(files=[fd])
        violations = patch.validate()
        assert any("subprocess.run" in v for v in violations)

    def test_bare_eval_in_new_content(self):
        fd = FileDiff(
            path="brain/test.py",
            original_content="",
            new_content="result = eval('1+1')\n",
        )
        patch = CodePatch(files=[fd])
        violations = patch.validate()
        assert any("eval" in v for v in violations)


# ---------------------------------------------------------------------------
# validate_syntax()
# ---------------------------------------------------------------------------

class TestValidateSyntax:
    def test_valid_python_passes(self):
        fd = FileDiff(path="brain/test.py", new_content="x = 1\ndef foo():\n    return 42\n")
        patch = CodePatch(files=[fd])
        assert patch.validate_syntax() == []

    def test_syntax_error_caught(self):
        fd = FileDiff(path="brain/test.py", new_content="def foo(:\n    pass\n")
        patch = CodePatch(files=[fd])
        errors = patch.validate_syntax()
        assert len(errors) >= 1
        assert "brain/test.py" in errors[0]

    def test_multiple_files_one_bad(self):
        good = FileDiff(path="brain/a.py", new_content="x = 1\n")
        bad = FileDiff(path="brain/b.py", new_content="def foo(:\n")
        patch = CodePatch(files=[good, bad])
        errors = patch.validate_syntax()
        assert len(errors) == 1
        assert "brain/b.py" in errors[0]

    def test_empty_content_skipped(self):
        fd = FileDiff(path="brain/test.py", new_content="")
        patch = CodePatch(files=[fd])
        assert patch.validate_syntax() == []


# ---------------------------------------------------------------------------
# check_capability_escalation()
# ---------------------------------------------------------------------------

class TestCheckCapabilityEscalation:
    def test_sets_requires_approval(self):
        fd = FileDiff(
            path="brain/test.py",
            original_content="x = 1\n",
            new_content="import requests\nx = 1\n",
        )
        patch = CodePatch(files=[fd])
        assert patch.requires_approval is False
        esc = patch.check_capability_escalation()
        assert len(esc) >= 1
        assert patch.requires_approval is True

    def test_no_escalation_no_flag(self):
        fd = FileDiff(
            path="brain/test.py",
            original_content="x = 1\n",
            new_content="x = 2\n",
        )
        patch = CodePatch(files=[fd])
        esc = patch.check_capability_escalation()
        assert esc == []
        assert patch.requires_approval is False


# ---------------------------------------------------------------------------
# validate_diff_budget()
# ---------------------------------------------------------------------------

class TestValidateDiffBudget:
    def test_within_budget(self):
        fd = FileDiff(
            path="brain/test.py",
            original_content="x = 1\n",
            new_content="x = 2\n",
        )
        patch = CodePatch(files=[fd])
        assert patch.validate_diff_budget() == []

    def test_too_many_files(self):
        files = [
            FileDiff(path=f"brain/f{i}.py", original_content="x=1", new_content="x=2")
            for i in range(4)
        ]
        patch = CodePatch(files=files)
        v = patch.validate_diff_budget()
        assert any("too many files" in x.lower() for x in v)

    def test_too_many_new_files(self):
        files = [
            FileDiff(path=f"brain/new{i}.py", original_content="", new_content="x=1\n")
            for i in range(2)
        ]
        patch = CodePatch(files=files)
        v = patch.validate_diff_budget()
        assert any("too many new files" in x.lower() for x in v)

    def test_too_many_lines(self):
        big_content = "\n".join(f"line_{i} = {i}" for i in range(600))
        fd = FileDiff(path="brain/big.py", original_content="", new_content=big_content)
        patch = CodePatch(files=[fd])
        v = patch.validate_diff_budget()
        assert any("too many lines" in x.lower() for x in v)


# ---------------------------------------------------------------------------
# _count_changed_lines()
# ---------------------------------------------------------------------------

class TestCountChangedLines:
    def test_identical(self):
        assert _count_changed_lines("x = 1\n", "x = 1\n") == 0

    def test_one_line_changed(self):
        assert _count_changed_lines("x = 1\n", "x = 2\n") == 1

    def test_line_added(self):
        assert _count_changed_lines("x = 1\n", "x = 1\ny = 2\n") == 1

    def test_empty_old(self):
        assert _count_changed_lines("", "x = 1\ny = 2\n") == 2


# ---------------------------------------------------------------------------
# to_dict() and FileDiff
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_shape(self):
        fd = FileDiff(path="brain/test.py", new_content="x=1\n", original_content="x=0\n")
        patch = CodePatch(files=[fd], description="test patch")
        d = patch.to_dict()
        assert d["file_count"] == 1
        assert d["description"] == "test patch"
        assert "files" in d
        assert d["files"][0]["path"] == "brain/test.py"
        assert d["files"][0]["has_original"] is True

    def test_id_prefix(self):
        patch = CodePatch()
        assert patch.id.startswith("patch_")

    def test_json_serializable(self):
        import json
        patch = CodePatch(files=[FileDiff(path="brain/x.py")])
        json.dumps(patch.to_dict())

    def test_filediff_defaults(self):
        fd = FileDiff(path="brain/test.py")
        assert fd.original_content == ""
        assert fd.new_content == ""
        assert fd.diff == ""
