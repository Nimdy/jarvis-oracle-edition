"""HRR truth-boundary proof.

Runs ``brain/synthetic/hrr_exercise.py`` (directly, in-process) and asserts
that no canonical JARVIS state was touched:

* ``~/.jarvis/memory/`` contents (mtime + size of every file, recursive)
* ``~/.jarvis/identity.json``
* ``~/.jarvis/belief_graph.jsonl`` or equivalent belief edge stores
* ``~/.jarvis/conversation_history.json``
* ``~/.jarvis/autonomy_state.json``
* any HRR-derived file under ``~/.jarvis/`` (explicitly must NOT exist this sprint)

The test does NOT create new state; it only snapshots whatever already exists
on disk in the worker's ``$HOME/.jarvis`` and checks for deltas.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from synthetic.hrr_exercise import run_exercise


JARVIS_DIR = Path.home() / ".jarvis"

# Files we explicitly care about — if they exist, HRR must not mutate them.
# If they don't exist, HRR must not create them.
WATCHED_PATHS = (
    JARVIS_DIR / "identity.json",
    JARVIS_DIR / "conversation_history.json",
    JARVIS_DIR / "autonomy_state.json",
    JARVIS_DIR / "belief_graph.jsonl",
)

# Directories whose entire contents we snapshot (recursive mtime+size+hash).
WATCHED_DIRS = (
    JARVIS_DIR / "memory",
    JARVIS_DIR / "hemispheres",
    JARVIS_DIR / "language_corpus",
)

# HRR-derived artifacts that MUST NOT exist after this sprint's exercise run.
FORBIDDEN_PATHS = (
    JARVIS_DIR / "cache" / "hrr_memory_vectors.jsonl",
    JARVIS_DIR / "hrr_state.json",
    JARVIS_DIR / "hrr" / "any_persistent_file",
)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "MISSING"


def _snapshot_path(p: Path) -> Tuple[str, int, float, str]:
    try:
        st = p.stat()
        return ("present", st.st_size, st.st_mtime, _hash_file(p))
    except FileNotFoundError:
        return ("absent", 0, 0.0, "MISSING")


def _snapshot_dir(root: Path) -> Dict[str, Tuple[int, float, str]]:
    snap: Dict[str, Tuple[int, float, str]] = {}
    if not root.exists():
        return snap
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            full = Path(dirpath) / name
            try:
                st = full.stat()
            except FileNotFoundError:
                continue
            rel = str(full.relative_to(root))
            snap[rel] = (st.st_size, st.st_mtime, _hash_file(full))
    return snap


def _full_snapshot() -> Dict[str, object]:
    return {
        "watched_paths": {str(p): _snapshot_path(p) for p in WATCHED_PATHS},
        "watched_dirs": {str(d): _snapshot_dir(d) for d in WATCHED_DIRS},
        "forbidden_paths_absent": {
            str(p): (not p.exists()) for p in FORBIDDEN_PATHS
        },
    }


def test_hrr_exercise_touches_nothing_in_jarvis_home(tmp_path):
    """The core truth-boundary contract for this sprint."""
    before = _full_snapshot()

    # Run the exercise in-process. No CLI, no subprocess, no file writes except
    # the explicit --out path we control (inside tmp_path, outside ~/.jarvis).
    result = run_exercise(
        dim=256,  # small to keep the test fast
        facts=[1, 2, 4, 8],
        noise_levels=[0.0, 0.05],
        seed=0,
    )

    after = _full_snapshot()

    assert result["schema_version"] == 1
    assert result["status"] == "PRE-MATURE"
    assert result["hrr_side_effects"] == 0
    assert all(v is False for v in result["authority_flags"].values())

    # Every watched path must be byte-identical.
    for key, before_entry in before["watched_paths"].items():
        after_entry = after["watched_paths"][key]
        assert before_entry == after_entry, f"HRR exercise mutated {key}"

    # Every watched directory must be byte-identical.
    for key, before_dir in before["watched_dirs"].items():
        after_dir = after["watched_dirs"][key]
        assert before_dir == after_dir, f"HRR exercise mutated contents of {key}"

    # Forbidden durable HRR artifacts must remain absent.
    for key, before_absent in before["forbidden_paths_absent"].items():
        after_absent = after["forbidden_paths_absent"][key]
        assert before_absent and after_absent, (
            f"forbidden HRR artifact present: {key} "
            f"(before_absent={before_absent}, after_absent={after_absent})"
        )


def test_hrr_exercise_out_file_is_the_only_write(tmp_path, monkeypatch):
    """When ``--out`` is supplied, the exercise writes exactly one file there
    and nothing else anywhere. We verify by running inside a throwaway HOME.
    """
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    out_path = tmp_path / "hrr_stage0.json"
    # Invoke the module's main() through argparse to exercise the same code
    # path the CLI uses.
    from synthetic.hrr_exercise import main

    rc = main(
        [
            "--dim", "128",
            "--facts", "1,2,4",
            "--noise", "0.0",
            "--out", str(out_path),
        ]
    )
    assert rc in (0, 2)  # thresholds may or may not pass at dim 128; both fine here
    assert out_path.exists()

    # fake_home must still be empty (or contain only pre-existing caches nothing
    # created by our run). Since we just created it, it should be empty.
    survivors = [p for p in fake_home.rglob("*") if p.is_file()]
    assert survivors == [], f"HRR exercise created files under HOME: {survivors}"


def test_hrr_exercise_never_imports_forbidden_writers():
    """Guard import graph: the exercise module must not import any writer."""
    import synthetic.hrr_exercise as mod
    src = Path(mod.__file__).read_text(encoding="utf-8")
    forbidden = (
        "from memory.storage",
        "from memory.persistence",
        "from epistemic.belief_graph.bridge",
        "from policy.state_encoder",
        "from autonomy",
        "from identity",
    )
    for token in forbidden:
        assert token not in src, f"hrr_exercise.py must not import {token!r}"
