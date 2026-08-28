"""On-disk .py newer than this PID — dashboard restart signal."""
from __future__ import annotations

import os
import time

from dashboard.code_freshness import scan_code_freshness


def test_scan_lists_py_newer_than_process_start(tmp_path):
    old = tmp_path / "old.py"
    new = tmp_path / "new.py"
    old.write_text("old\n", encoding="utf-8")
    new.write_text("new\n", encoding="utf-8")
    started = time.time()
    os.utime(old, (started - 200, started - 200))
    os.utime(new, (started + 5, started + 5))

    data = scan_code_freshness(str(tmp_path), started)
    assert data["scan_ok"] is True
    assert data["is_stale"] is True
    assert data["stale_count"] == 1
    paths = [row["path"] for row in data["stale_files"]]
    assert "new.py" in paths
    assert "old.py" not in paths
    assert data["newest_file"] == "new.py"


def test_scan_in_sync_when_all_files_older(tmp_path):
    f = tmp_path / "boot.py"
    f.write_text("x\n", encoding="utf-8")
    started = time.time()
    os.utime(f, (started - 50, started - 50))

    data = scan_code_freshness(str(tmp_path), started)
    assert data["scan_ok"] is True
    assert data["is_stale"] is False
    assert data["stale_count"] == 0
    assert data["stale_files"] == []


def test_v2_banner_css_does_not_force_hidden():
    """Lived: .banner-code{display:none} + JS display='' never painted."""
    from pathlib import Path
    import re
    root = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "v2"
    css = (root / "v2.css").read_text(encoding="utf-8")
    m = re.search(r"\.banner-code\{[^}]*\}", css)
    assert m, "missing .banner-code rule"
    compact = m.group(0).replace(" ", "")
    assert "display:none" not in compact
    js = (root / "shared.js").read_text(encoding="utf-8")
    assert "b.style.display='block'" in js
    assert "b.style.display='';" not in js
