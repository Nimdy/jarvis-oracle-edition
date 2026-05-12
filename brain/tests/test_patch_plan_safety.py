"""Tests for self_improve/patch_plan.py — the safety boundary layer.

This is the CRITICAL defense against self-improvement injecting dangerous code.
Every denied pattern, AST forbidden call, scope boundary, capability escalation
detection, and diff budget cap is tested individually. A regression here means
the system could promote code containing subprocess.Popen, eval(), or os.system.

Covers:
  - 13 DENIED_PATTERNS (regex-based content scanning)
  - 8 FORBIDDEN_AST_CALLS (AST-level function call detection)
  - ALLOWED_PATHS scope validation
  - detect_capability_escalation (new network/subprocess/security imports)
  - validate_diff_budget (file count, line count, new file caps)
  - check_dangerous (mutator/governor/persistence files)
  - to_dict() serialization
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from self_improve.patch_plan import (
    PatchPlan,
    ALLOWED_PATHS,
    DENIED_PATTERNS,
    FORBIDDEN_AST_CALLS,
    MAX_FILES_CHANGED,
    MAX_LINES_CHANGED,
    MAX_NEW_FILES,
    check_denied_patterns,
    check_ast_forbidden_calls,
    detect_capability_escalation,
)


# ---------------------------------------------------------------------------
# DENIED_PATTERNS — every single pattern must catch its target
# ---------------------------------------------------------------------------

class TestDeniedPatterns:
    """Each of the 13 denied regex patterns is tested individually."""

    def test_subprocess_detected(self):
        v = check_denied_patterns("import subprocess")
        assert len(v) >= 1
        assert any("subprocess" in x for x in v)

    def test_os_system_detected(self):
        v = check_denied_patterns("os.system('rm -rf /')")
        assert len(v) >= 1

    def test_os_popen_detected(self):
        v = check_denied_patterns("os.popen('ls')")
        assert len(v) >= 1

    def test_dunder_import_detected(self):
        v = check_denied_patterns("__import__('os')")
        assert len(v) >= 1

    def test_exec_detected(self):
        v = check_denied_patterns("exec(code)")
        assert len(v) >= 1

    def test_eval_detected(self):
        v = check_denied_patterns("eval(expr)")
        assert len(v) >= 1

    def test_credentials_detected(self):
        v = check_denied_patterns('credentials = "admin:pass"')
        assert len(v) >= 1

    def test_api_key_detected(self):
        v = check_denied_patterns('api_key = "sk-xxx"')
        assert len(v) >= 1

    def test_password_detected(self):
        v = check_denied_patterns('password = "hunter2"')
        assert len(v) >= 1

    def test_secret_detected(self):
        v = check_denied_patterns('secret = "abc123"')
        assert len(v) >= 1

    def test_open_write_detected(self):
        v = check_denied_patterns("open('/etc/passwd', 'w')")
        assert len(v) >= 1

    def test_socket_detected(self):
        v = check_denied_patterns("import socket")
        assert len(v) >= 1

    def test_http_client_detected(self):
        v = check_denied_patterns("import http.client")
        assert len(v) >= 1

    def test_pattern_count_matches_docs(self):
        assert len(DENIED_PATTERNS) == 13

    def test_clean_code_passes(self):
        clean = "def hello():\n    return 'world'\n"
        assert check_denied_patterns(clean) == []

    def test_word_boundary_no_false_positive_token(self):
        assert check_denied_patterns("token = 'abc'") == []

    def test_word_boundary_no_false_positive_author(self):
        assert check_denied_patterns("author = 'alice'") == []

    def test_open_read_not_flagged(self):
        assert check_denied_patterns("open('file.txt', 'r')") == []

    def test_multiple_violations_all_reported(self):
        code = "import subprocess\nos.system('ls')\neval(x)"
        v = check_denied_patterns(code)
        assert len(v) >= 3


# ---------------------------------------------------------------------------
# FORBIDDEN_AST_CALLS — every call pair tested
# ---------------------------------------------------------------------------

class TestASTForbiddenCalls:
    """Each of the 8 AST-level forbidden calls is tested."""

    def test_subprocess_run(self):
        v = check_ast_forbidden_calls("import subprocess\nsubprocess.run(['ls'])")
        assert any("subprocess.run" in x for x in v)

    def test_subprocess_popen(self):
        v = check_ast_forbidden_calls("import subprocess\nsubprocess.Popen(['ls'])")
        assert any("subprocess.Popen" in x for x in v)

    def test_subprocess_call(self):
        v = check_ast_forbidden_calls("import subprocess\nsubprocess.call(['ls'])")
        assert any("subprocess.call" in x for x in v)

    def test_subprocess_check_output(self):
        v = check_ast_forbidden_calls("import subprocess\nsubprocess.check_output(['ls'])")
        assert any("subprocess.check_output" in x for x in v)

    def test_os_system(self):
        v = check_ast_forbidden_calls("import os\nos.system('ls')")
        assert any("os.system" in x for x in v)

    def test_os_popen(self):
        v = check_ast_forbidden_calls("import os\nos.popen('ls')")
        assert any("os.popen" in x for x in v)

    def test_os_exec(self):
        v = check_ast_forbidden_calls("import os\nos.exec('/bin/sh')")
        assert any("os.exec" in x for x in v)

    def test_os_execv(self):
        v = check_ast_forbidden_calls("import os\nos.execv('/bin/sh', [])")
        assert any("os.execv" in x for x in v)

    def test_bare_eval(self):
        v = check_ast_forbidden_calls("x = eval('1+1')")
        assert any("eval" in x for x in v)

    def test_bare_exec(self):
        v = check_ast_forbidden_calls("exec('print(1)')")
        assert any("exec" in x for x in v)

    def test_bare_dunder_import(self):
        v = check_ast_forbidden_calls("m = __import__('os')")
        assert any("__import__" in x for x in v)

    def test_forbidden_call_count(self):
        assert len(FORBIDDEN_AST_CALLS) == 8

    def test_clean_code_passes(self):
        clean = "def foo():\n    return 42\n"
        assert check_ast_forbidden_calls(clean) == []

    def test_syntax_error_returns_empty(self):
        assert check_ast_forbidden_calls("def foo(: pass") == []

    def test_line_numbers_in_violations(self):
        code = "x = 1\nsubprocess.run(['ls'])\n"
        v = check_ast_forbidden_calls(code)
        assert any("line 2" in x for x in v)

    def test_safe_attribute_calls_pass(self):
        code = "import logging\nlogging.info('hello')\npath.exists()\n"
        assert check_ast_forbidden_calls(code) == []


# ---------------------------------------------------------------------------
# ALLOWED_PATHS scope validation
# ---------------------------------------------------------------------------

class TestScopeValidation:
    def test_allowed_file_passes(self):
        plan = PatchPlan(files_to_modify=["brain/consciousness/kernel.py"])
        assert plan.validate_scope() == []

    def test_allowed_with_brain_prefix(self):
        plan = PatchPlan(files_to_modify=["brain/memory/core.py"])
        assert plan.validate_scope() == []

    def test_disallowed_file_flagged(self):
        plan = PatchPlan(files_to_modify=["brain/main.py"])
        v = plan.validate_scope()
        assert len(v) >= 1
        assert "outside allowed scope" in v[0].lower()

    def test_absolute_path_blocked(self):
        plan = PatchPlan(files_to_modify=["/etc/passwd"])
        assert len(plan.validate_scope()) >= 1

    def test_dot_dot_path_blocked(self):
        plan = PatchPlan(files_to_modify=["brain/../requirements.txt"])
        assert len(plan.validate_scope()) >= 1

    def test_empty_files_passes(self):
        plan = PatchPlan()
        assert plan.validate_scope() == []

    def test_new_files_checked_too(self):
        plan = PatchPlan(files_to_create=["brain/dashboard/new_page.py"])
        assert len(plan.validate_scope()) >= 1

    def test_all_allowed_paths_accepted(self):
        for path in ALLOWED_PATHS:
            plan = PatchPlan(files_to_modify=[f"{path}test.py"])
            assert plan.validate_scope() == [], f"Failed for allowed path: {path}"

    def test_allowed_path_count_matches_docs(self):
        assert len(ALLOWED_PATHS) >= 11


# ---------------------------------------------------------------------------
# detect_capability_escalation
# ---------------------------------------------------------------------------

class TestCapabilityEscalation:
    def test_new_network_import_detected(self):
        old = "import json\n"
        new = "import json\nimport requests\n"
        esc = detect_capability_escalation(old, new)
        assert any("network" in e.lower() for e in esc)

    def test_new_subprocess_import_detected(self):
        old = "import json\n"
        new = "import json\nimport subprocess\n"
        esc = detect_capability_escalation(old, new)
        assert any("subprocess" in e.lower() for e in esc)

    def test_preexisting_import_not_flagged(self):
        old = "import subprocess\n"
        new = "import subprocess\nx = 1\n"
        esc = detect_capability_escalation(old, new)
        assert not any("subprocess" in e.lower() for e in esc)

    def test_security_boundary_modification(self):
        old = "x = 1\n"
        new = "ALLOWED_PATHS = ['brain/everything/']\n"
        esc = detect_capability_escalation(old, new)
        assert any("security" in e.lower() for e in esc)

    def test_clean_change_no_escalation(self):
        old = "x = 1\n"
        new = "x = 2\ny = 3\n"
        assert detect_capability_escalation(old, new) == []

    def test_empty_original_new_import_detected(self):
        esc = detect_capability_escalation("", "import aiohttp\n")
        assert any("network" in e.lower() for e in esc)

    def test_multiple_escalations(self):
        old = ""
        new = "import subprocess\nimport requests\nALLOWED_PATHS = []\n"
        esc = detect_capability_escalation(old, new)
        assert len(esc) >= 2

    def test_syntax_error_handled(self):
        esc = detect_capability_escalation("def foo(: pass", "import requests\n")
        assert isinstance(esc, list)


# ---------------------------------------------------------------------------
# validate_diff_budget
# ---------------------------------------------------------------------------

class TestDiffBudget:
    def test_within_budget_passes(self):
        plan = PatchPlan(
            files_to_modify=["brain/consciousness/kernel.py"],
            files_to_create=[],
        )
        assert plan.validate_diff_budget(total_lines_changed=50) == []

    def test_too_many_files(self):
        plan = PatchPlan(
            files_to_modify=["brain/a.py", "brain/b.py", "brain/c.py"],
            files_to_create=["brain/d.py"],
        )
        v = plan.validate_diff_budget()
        assert any("too many files" in x.lower() for x in v)

    def test_too_many_new_files(self):
        plan = PatchPlan(
            files_to_create=["brain/consciousness/a.py", "brain/consciousness/b.py"],
        )
        v = plan.validate_diff_budget()
        assert any("too many new files" in x.lower() for x in v)

    def test_too_many_lines(self):
        plan = PatchPlan(files_to_modify=["brain/consciousness/kernel.py"])
        v = plan.validate_diff_budget(total_lines_changed=MAX_LINES_CHANGED + 1)
        assert any("too many lines" in x.lower() for x in v)

    def test_exact_limits_pass(self):
        plan = PatchPlan(
            files_to_modify=["brain/a.py", "brain/b.py"],
            files_to_create=["brain/consciousness/c.py"],
        )
        assert plan.validate_diff_budget(total_lines_changed=MAX_LINES_CHANGED) == []

    def test_max_constants_are_reasonable(self):
        assert MAX_FILES_CHANGED >= 1
        assert MAX_LINES_CHANGED >= 50
        assert MAX_NEW_FILES >= 1


# ---------------------------------------------------------------------------
# check_dangerous
# ---------------------------------------------------------------------------

class TestCheckDangerous:
    def test_mutation_governor_flagged(self):
        plan = PatchPlan(files_to_modify=["brain/consciousness/mutation_governor.py"])
        assert plan.check_dangerous() is True

    def test_persistence_flagged(self):
        plan = PatchPlan(files_to_modify=["brain/memory/persistence.py"])
        assert plan.check_dangerous() is True

    def test_kernel_mutator_flagged(self):
        plan = PatchPlan(files_to_modify=["brain/consciousness/kernel_mutator.py"])
        assert plan.check_dangerous() is True

    def test_safe_file_not_flagged(self):
        plan = PatchPlan(files_to_modify=["brain/consciousness/observer.py"])
        assert plan.check_dangerous() is False

    def test_empty_files_not_dangerous(self):
        plan = PatchPlan()
        assert plan.check_dangerous() is False


# ---------------------------------------------------------------------------
# PatchPlan serialization
# ---------------------------------------------------------------------------

class TestPatchPlanSerialization:
    def test_to_dict_shape(self):
        plan = PatchPlan(
            files_to_modify=["brain/consciousness/kernel.py"],
            constraints=["no breaking changes"],
        )
        d = plan.to_dict()
        assert "id" in d
        assert d["files_to_modify"] == ["brain/consciousness/kernel.py"]
        assert d["constraints"] == ["no breaking changes"]
        assert "write_category" in d

    def test_id_prefix(self):
        plan = PatchPlan()
        assert plan.id.startswith("plan_")

    def test_to_dict_is_json_serializable(self):
        import json
        plan = PatchPlan(files_to_modify=["brain/test.py"])
        json.dumps(plan.to_dict())
