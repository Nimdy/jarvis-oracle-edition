from pathlib import Path

from library import self_study_filter


def test_allow_library_reingest_accepts_runtime_file(monkeypatch):
    monkeypatch.setattr(self_study_filter, "is_git_tracked", lambda project_root, repo_rel: True)

    allowed = self_study_filter.allow_library_reingest(
        Path("/tmp/project"),
        "brain/reasoning/context.py",
    )

    assert allowed is True


def test_allow_library_reingest_rejects_tests_even_if_tracked(monkeypatch):
    monkeypatch.setattr(self_study_filter, "is_git_tracked", lambda project_root, repo_rel: True)

    allowed = self_study_filter.allow_library_reingest(
        Path("/tmp/project"),
        "brain/tests/test_capability_gate.py",
    )

    assert allowed is False


def test_allow_library_reingest_rejects_non_whitelisted_docs(monkeypatch):
    monkeypatch.setattr(self_study_filter, "is_git_tracked", lambda project_root, repo_rel: True)

    allowed = self_study_filter.allow_library_reingest(
        Path("/tmp/project"),
        "docs/Jarvis_Live_Brain_Trust_Audit_Prompt.md",
    )

    assert allowed is False
