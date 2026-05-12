"""Guards for codebase self-study ingestion.

Jarvis should only ingest shipped runtime/docs into the Library. Local dev
artifacts like tests, scratch files, and untracked workspace churn should not
become self-knowledge.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

SELF_STUDY_ALLOWED_DOCS = {
    "AGENTS.md",
    "ARCHITECTURE.md",
    "docs/SYSTEM_OVERVIEW.md",
    "docs/COMPANION_TRAINING_PLAYBOOK.md",
}

SELF_STUDY_ALLOWED_RUNTIME_PREFIXES = (
    "brain/autonomy/",
    "brain/cognition/",
    "brain/consciousness/",
    "brain/dashboard/",
    "brain/epistemic/",
    "brain/goals/",
    "brain/hemisphere/",
    "brain/identity/",
    "brain/jarvis_eval/",
    "brain/library/",
    "brain/memory/",
    "brain/perception/",
    "brain/personality/",
    "brain/policy/",
    "brain/reasoning/",
    "brain/self_improve/",
    "brain/skills/",
    "brain/tools/",
)


def is_git_tracked(project_root: Path, repo_rel: str) -> bool:
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--error-unmatch", repo_rel],
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def allow_library_reingest(project_root: Path, repo_rel: str) -> bool:
    """Gate codebase self-study to shipped runtime/docs, not local dev churn."""
    normalized = repo_rel.strip().lstrip("./")
    if not normalized:
        return False
    if normalized.startswith(("brain/tests/", "brain/scripts/")):
        return False
    if "/tests/" in normalized or normalized.endswith("/tests"):
        return False
    if normalized in SELF_STUDY_ALLOWED_DOCS:
        return is_git_tracked(project_root, normalized)
    if normalized.startswith(SELF_STUDY_ALLOWED_RUNTIME_PREFIXES):
        return is_git_tracked(project_root, normalized)
    return False
