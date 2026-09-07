"""Compare on-disk brain .py mtimes to process start.

Used by GET /api/system/code-freshness so the operator can Restart after
sync-desktop without guessing whether this PID loaded the new files.
"""
from __future__ import annotations

import os
import time
from typing import Any

SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".git", ".venv", "venv", "env", "node_modules",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "improvement_snapshots", "kernel_snapshots", "hemispheres",
    "policy_models", "synthetic_exercise",
})
EXTENSIONS: tuple[str, ...] = (".py",)
MAX_FILES: int = 40


def scan_code_freshness(
    brain_root: str,
    process_started_ts: float,
    *,
    max_files: int = MAX_FILES,
) -> dict[str, Any]:
    """Walk brain_root for .py files newer than process_started_ts. Never raises."""
    newest_mtime = 0.0
    newest_file = ""
    file_count = 0
    stale_acc: list[dict[str, Any]] = []
    try:
        for dirpath, dirnames, filenames in os.walk(brain_root, followlinks=False):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") or d in (".jarvis",)]
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fname in filenames:
                if not fname.endswith(EXTENSIONS):
                    continue
                full = os.path.join(dirpath, fname)
                try:
                    mtime = os.path.getmtime(full)
                except OSError:
                    continue
                file_count += 1
                rel = os.path.relpath(full, brain_root)
                if mtime > newest_mtime:
                    newest_mtime = mtime
                    newest_file = rel
                if mtime > process_started_ts:
                    stale_acc.append({
                        "path": rel,
                        "mtime": mtime,
                        "age_s": round(mtime - process_started_ts, 1),
                    })
    except Exception:
        return {
            "process_started_ts": process_started_ts,
            "newest_mtime": 0.0,
            "newest_file": "",
            "file_count": 0,
            "stale_count": 0,
            "stale_files": [],
            "is_stale": False,
            "stale_age_s": 0.0,
            "scan_ok": False,
        }

    stale_acc.sort(key=lambda row: -float(row["mtime"]))
    is_stale = newest_mtime > process_started_ts
    stale_age_s = max(0.0, newest_mtime - process_started_ts) if is_stale else 0.0
    return {
        "process_started_ts": process_started_ts,
        "newest_mtime": newest_mtime,
        "newest_file": newest_file,
        "file_count": file_count,
        "stale_count": len(stale_acc),
        "stale_files": stale_acc[:max_files],
        "is_stale": is_stale,
        "stale_age_s": stale_age_s,
        "scan_ok": True,
        "now": time.time(),
    }
