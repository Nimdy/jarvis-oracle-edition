"""Gestation self-study teaches the AST, not markdown books."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from consciousness.gestation import _TIER_A_FILES, _TIER_B_FILES, SELF_STUDY_DIRECTIVES


def test_tier_lists_are_python_only():
    for rel, _title in _TIER_A_FILES + _TIER_B_FILES:
        assert rel.endswith(".py"), rel
        assert not rel.endswith(".md"), rel
        assert not rel.startswith("docs/"), rel


def test_tier_a_includes_pi_senses():
    paths = [rel for rel, _ in _TIER_A_FILES]
    assert "pi/main.py" in paths
    assert "AGENTS.md" not in paths
    assert "ARCHITECTURE.md" not in paths


def test_self_study_directives_query_codebase_not_docs():
    assert SELF_STUDY_DIRECTIVES
    assert all(d.tool_hint == "codebase" for d in SELF_STUDY_DIRECTIVES)


def test_ingest_skips_markdown_even_if_listed():
    src = Path(__file__).resolve().parents[1].joinpath("consciousness/gestation.py").read_text(
        encoding="utf-8",
    )
    idx = src.find("def _ingest_codebase_to_library")
    body = src[idx:idx + 2500]
    assert 'rel_path.endswith(".md")' in body
    assert "library_index.init" in body
    assert 'domain_tags="codebase,self_knowledge,architecture"' not in body


def test_codebase_index_resolves_pi_and_brain_paths():
    from tools.codebase_tool import CodebaseIndex

    idx = CodebaseIndex()
    brain_py = idx._resolve_source("consciousness/engine.py")
    pi_py = idx._resolve_source("pi/main.py")
    assert brain_py.name == "engine.py"
    assert "brain" in str(brain_py).replace("\\", "/")
    assert pi_py.name == "main.py"
    assert str(pi_py).replace("\\", "/").endswith("pi/main.py")
