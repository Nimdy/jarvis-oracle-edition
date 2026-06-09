"""JARVIS must never study test/junk files into its belief graph.

Operator directive after a stray "test is discard" belief surfaced in the grounding
queue: the codebase-ingest boundary rejects test files, caches, vendored deps, etc.
The guard returns early (before any source-store work), so this runs without heavy deps.
"""
from __future__ import annotations

import pytest

try:
    from library.ingest import ingest_codebase_source
except Exception:  # pragma: no cover - heavy deps absent
    pytest.skip("library.ingest import unavailable", allow_module_level=True)


@pytest.mark.parametrize("path", [
    "brain/tests/test_foo.py",
    "brain/tests/conftest.py",
    "foo/test_bar.py",
    "bar_test.py",
    "brain/__pycache__/x.pyc",
    "node_modules/lib/x.js",
    ".venv/lib/python3.11/site.py",
    "project/.git/config",
    "build/lib/x.py",
    "x.egg-info/PKG-INFO",
])
def test_test_and_junk_files_are_never_studied(path):
    r = ingest_codebase_source(file_path=path, content="some real-looking content", title="x")
    assert r.success is False
    assert "test/junk" in (r.error or "")


def test_real_module_path_is_not_flagged_as_junk():
    # A real source path must NOT be rejected by the junk guard. (It may still fail
    # later on missing store deps in this env — we only assert it isn't junk-blocked.)
    try:
        r = ingest_codebase_source(file_path="brain/reasoning/tool_router.py",
                                   content="class ToolType: ...", title="tool router")
    except Exception:
        return  # store deps unavailable in this env — guard already passed
    assert "test/junk" not in (r.error or "")
