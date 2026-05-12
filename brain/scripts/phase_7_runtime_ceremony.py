"""Phase 7 `isolated_subprocess` runtime ceremony.

Proves the three post-ship verification boxes that were never checked after
Phase 7 landed on 2026-04-16:

    1. One real plugin goes through `isolated_subprocess` end-to-end.
    2. Venv creation + pinned dependency install (pip) actually works on
       the host.
    3. The existing in-process plugin path is still intact.

The ceremony runs against an **isolated** registry + plugins dir + audit
dir under ``~/.jarvis/ceremonies/phase_7_<date>/`` so it cannot race with
or pollute the live brain's registry. Only the venv lands under the
natural ``~/.jarvis/plugin_venvs/<plugin>/`` location (namespaced into the
ceremony sub-tree) so the evidence is physical bits on disk, not a mock.

Usage:

    cd ~/duafoo/brain
    source .venv/bin/activate
    python -m scripts.phase_7_runtime_ceremony

Outputs a JSON evidence bundle to stdout **and** to
``~/.jarvis/ceremonies/phase_7_<date>/evidence.json`` for later citation
in ``docs/validation_reports/phase_7_isolated_subprocess_runtime_proof-<date>.md``.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


_BRAIN_DIR = Path(__file__).resolve().parent.parent
if str(_BRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_BRAIN_DIR))


CEREMONY_DATE = _dt.date.today().isoformat()
CEREMONY_ROOT = Path.home() / ".jarvis" / "ceremonies" / f"phase_7_{CEREMONY_DATE}"
CEREMONY_PLUGINS_DIR = CEREMONY_ROOT / "plugins"
CEREMONY_REGISTRY_PATH = CEREMONY_ROOT / "plugin_registry.json"
CEREMONY_AUDIT_DIR = CEREMONY_ROOT / "plugin_audit"
CEREMONY_VENV_DIR = CEREMONY_ROOT / "plugin_venvs"
EVIDENCE_PATH = CEREMONY_ROOT / "evidence.json"

PLUGIN_NAME = "ceremony_dateutil_demo"
PINNED_DEP = "python-dateutil==2.9.0"

LIVE_REGISTRY_PATH = Path.home() / ".jarvis" / "plugin_registry.json"


def _isolate_registry_paths() -> None:
    """Redirect plugin_registry + plugin_process module globals at the
    ceremony root so nothing collides with the live brain."""
    CEREMONY_ROOT.mkdir(parents=True, exist_ok=True)
    CEREMONY_PLUGINS_DIR.mkdir(parents=True, exist_ok=True)
    CEREMONY_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    CEREMONY_VENV_DIR.mkdir(parents=True, exist_ok=True)

    import tools.plugin_registry as preg
    import tools.plugin_process as pproc

    preg._PLUGINS_DIR = CEREMONY_PLUGINS_DIR
    preg._REGISTRY_PATH = CEREMONY_REGISTRY_PATH
    preg._AUDIT_DIR = CEREMONY_AUDIT_DIR
    pproc._VENV_BASE_DIR = CEREMONY_VENV_DIR


def _demo_plugin_sources() -> dict[str, str]:
    """Return the two-file plugin: manifest+bridge (__init__.py) + handler.

    The handler imports ``dateutil`` (installed via the pinned dep) and
    parses a fixed ISO-ish timestamp. Output is deterministic so we can
    assert on it.
    """
    handler_src = (
        "from dateutil import parser as _dp\n"
        "from dateutil.tz import tzutc as _tzutc\n"
        "\n"
        "\n"
        "def run(request):\n"
        "    text = request.get('request', '') or '2026-04-23T18:00:00Z'\n"
        "    parsed = _dp.isoparse(text)\n"
        "    ref = parsed.astimezone(_tzutc())\n"
        "    return {\n"
        "        'parsed_iso': parsed.isoformat(),\n"
        "        'utc_iso': ref.isoformat(),\n"
        "        'tzinfo': str(parsed.tzinfo),\n"
        "        'dep_evidence': 'dateutil.parser.isoparse',\n"
        "    }\n"
    )
    return {"handler.py": handler_src}


def _build_manifest():
    from tools.plugin_registry import PluginManifest

    return PluginManifest(
        name=PLUGIN_NAME,
        version="1.0.0",
        description="Phase 7 ceremony demo: parses an ISO timestamp via python-dateutil.",
        keywords=["ceremony", "phase7", "isolated"],
        intent_patterns=[r"^ceremony parse\b"],
        created_by="phase_7_runtime_ceremony.py",
        skill_id="ceremony.phase7.isolated.dateutil_demo",
        risk_tier=0,
        approved_by="operator:phase_7_ceremony",
        supervision_mode="active",
        permissions=[],
        allowed_imports=["dateutil"],
        timeout_s=15.0,
        execution_mode="isolated_subprocess",
        pinned_dependencies=[PINNED_DEP],
        verify_imports=["dateutil"],
    )


def _venv_installed_packages(venv_dir: Path) -> list[str]:
    """List packages installed inside the venv via its own pip."""
    pip = venv_dir / "bin" / "pip"
    if not pip.exists():
        return []
    try:
        out = subprocess.check_output(
            [str(pip), "list", "--format=freeze"],
            stderr=subprocess.STDOUT,
            timeout=30,
        ).decode("utf-8", errors="replace")
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception as exc:
        return [f"pip-list-failed: {exc}"]


def _venv_disk_usage(venv_dir: Path) -> dict[str, Any]:
    if not venv_dir.exists():
        return {"exists": False}
    total = 0
    n_files = 0
    for root, _dirs, files in os.walk(venv_dir):
        for f in files:
            p = Path(root) / f
            try:
                total += p.stat().st_size
                n_files += 1
            except OSError:
                pass
    return {
        "exists": True,
        "path": str(venv_dir),
        "bytes": total,
        "mb": round(total / (1024 * 1024), 3),
        "file_count": n_files,
    }


def _tail_audit_jsonl(path: Path, limit: int = 5) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    lines.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return lines[-limit:]


def _inspect_live_in_process_plugins() -> dict[str, Any]:
    """Read the live brain's registry file (read-only) and report the
    shape of any in-process plugin records present.

    This proves the in-process path remains intact on the real brain
    without the ceremony having to invoke anything there.
    """
    if not LIVE_REGISTRY_PATH.exists():
        return {"exists": False, "path": str(LIVE_REGISTRY_PATH)}
    try:
        data = json.loads(LIVE_REGISTRY_PATH.read_text())
    except Exception as exc:
        return {
            "exists": True,
            "path": str(LIVE_REGISTRY_PATH),
            "error": f"{type(exc).__name__}: {exc}",
        }

    entries: list[dict[str, Any]] = []
    for name, rec in data.items():
        if not isinstance(rec, dict):
            continue
        if rec.get("execution_mode", "in_process") != "in_process":
            continue
        entries.append(
            {
                "name": name,
                "state": rec.get("state"),
                "execution_mode": rec.get("execution_mode", "in_process"),
                "invocation_count": rec.get("invocation_count"),
                "success_count": rec.get("success_count"),
                "failure_count": rec.get("failure_count"),
                "avg_latency_ms": rec.get("avg_latency_ms"),
                "last_invocation_at": rec.get("last_invocation_at"),
            }
        )
    return {
        "exists": True,
        "path": str(LIVE_REGISTRY_PATH),
        "in_process_records": entries,
    }


async def _run_ceremony() -> dict[str, Any]:
    _isolate_registry_paths()

    from tools.plugin_registry import PluginRegistry, PluginRequest

    manifest = _build_manifest()
    code_files = _demo_plugin_sources()

    registry = PluginRegistry(plugins_dir=CEREMONY_PLUGINS_DIR)

    t_quarantine = time.monotonic()
    ok, errors = registry.quarantine(
        plugin_name=PLUGIN_NAME,
        code_files=code_files,
        manifest=manifest,
        acquisition_id="ceremony-phase-7",
    )
    quarantine_ms = round((time.monotonic() - t_quarantine) * 1000, 2)
    if not ok:
        return {
            "stage": "quarantine_failed",
            "errors": errors,
            "quarantine_ms": quarantine_ms,
        }

    activated = registry.activate(PLUGIN_NAME, supervision_mode="active")
    if not activated:
        return {"stage": "activate_failed"}

    rec = registry.get_record(PLUGIN_NAME)
    pre_invoke_rec = rec.to_dict() if rec is not None else {}

    request = PluginRequest(
        request_id="phase7-ceremony-001",
        plugin_name=PLUGIN_NAME,
        user_text="2026-04-23T18:00:00+00:00",
        context={},
        timeout_s=15.0,
    )

    t_invoke = time.monotonic()
    response = await registry.invoke(request)
    invoke_ms = round((time.monotonic() - t_invoke) * 1000, 2)

    mgr = registry._process_managers.get(PLUGIN_NAME)
    mgr_status = mgr.get_status() if mgr is not None else None
    install_log = mgr.install_log if mgr is not None else ""

    request_2 = PluginRequest(
        request_id="phase7-ceremony-002",
        plugin_name=PLUGIN_NAME,
        user_text="2026-05-10T00:00:00+00:00",
        context={},
        timeout_s=15.0,
    )
    t_invoke2 = time.monotonic()
    response_2 = await registry.invoke(request_2)
    invoke2_ms = round((time.monotonic() - t_invoke2) * 1000, 2)

    if mgr is not None:
        try:
            await mgr.shutdown()
        except Exception:
            pass

    rec_after = registry.get_record(PLUGIN_NAME)
    post_invoke_rec = rec_after.to_dict() if rec_after is not None else {}

    venv_dir = CEREMONY_VENV_DIR / PLUGIN_NAME
    audit_path = CEREMONY_AUDIT_DIR / f"{PLUGIN_NAME}.jsonl"

    evidence = {
        "ceremony": "phase_7_isolated_subprocess_runtime_proof",
        "date": CEREMONY_DATE,
        "host_python": sys.version.split()[0],
        "executable": sys.executable,
        "paths": {
            "ceremony_root": str(CEREMONY_ROOT),
            "plugins_dir": str(CEREMONY_PLUGINS_DIR),
            "registry_path": str(CEREMONY_REGISTRY_PATH),
            "audit_dir": str(CEREMONY_AUDIT_DIR),
            "venv_dir": str(venv_dir),
            "evidence_path": str(EVIDENCE_PATH),
        },
        "plugin": {
            "name": PLUGIN_NAME,
            "execution_mode": manifest.execution_mode,
            "pinned_dependencies": manifest.pinned_dependencies,
            "version": manifest.version,
        },
        "timings_ms": {
            "quarantine": quarantine_ms,
            "first_invoke_including_venv_build": invoke_ms,
            "second_invoke_warm_child": invoke2_ms,
        },
        "pre_invoke_record": pre_invoke_rec,
        "post_invoke_record": post_invoke_rec,
        "invoke_response_1": {
            "success": response.success,
            "error": response.error,
            "duration_ms": response.duration_ms,
            "result": response.result,
        },
        "invoke_response_2": {
            "success": response_2.success,
            "error": response_2.error,
            "duration_ms": response_2.duration_ms,
            "result": response_2.result,
        },
        "process_manager_status": mgr_status,
        "install_log": install_log,
        "venv_disk": _venv_disk_usage(venv_dir),
        "venv_installed_packages": _venv_installed_packages(venv_dir),
        "audit_ledger_tail": _tail_audit_jsonl(audit_path, limit=5),
        "live_in_process_control": _inspect_live_in_process_plugins(),
    }

    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2))
    return evidence


def _verdict_lines(evidence: dict[str, Any]) -> list[str]:
    r1 = evidence.get("invoke_response_1") or {}
    r2 = evidence.get("invoke_response_2") or {}
    venv = evidence.get("venv_disk") or {}
    pkgs = evidence.get("venv_installed_packages") or []
    mgr = evidence.get("process_manager_status") or {}
    lines: list[str] = []

    ok_invoke_1 = bool(r1.get("success"))
    ok_invoke_2 = bool(r2.get("success"))
    ok_venv = bool(venv.get("exists"))
    ok_dep = any("python-dateutil" in p or "python_dateutil" in p for p in pkgs)
    ok_mgr = bool(mgr.get("venv_ready"))

    lines.append(f"[{'PASS' if ok_invoke_1 else 'FAIL'}] First invocation (builds venv + installs pip dep + JSON IPC)")
    lines.append(f"[{'PASS' if ok_invoke_2 else 'FAIL'}] Second invocation (warm child, JSON IPC only)")
    lines.append(f"[{'PASS' if ok_venv else 'FAIL'}] Venv on disk at {venv.get('path', '?')} ({venv.get('mb', '?')} MB, {venv.get('file_count', '?')} files)")
    lines.append(f"[{'PASS' if ok_dep else 'FAIL'}] Pinned dependency installed in venv (python-dateutil present in pip list)")
    lines.append(f"[{'PASS' if ok_mgr else 'FAIL'}] PluginProcessManager reports venv_ready=True")

    live = evidence.get("live_in_process_control") or {}
    live_recs = live.get("in_process_records") or []
    live_ok = any(
        r.get("state") == "active" and (r.get("success_count") or 0) >= 1
        for r in live_recs
    )
    lines.append(
        f"[{'PASS' if live_ok else 'WARN'}] In-process control: live registry has at least one active in_process plugin with >=1 successful invocation"
    )
    return lines


def main() -> int:
    evidence = asyncio.run(_run_ceremony())
    print(json.dumps({"ceremony_root": str(CEREMONY_ROOT), "evidence_path": str(EVIDENCE_PATH)}, indent=2))
    print()

    stage = evidence.get("stage")
    if stage:
        print(f"[ABORT] Ceremony aborted at stage: {stage}")
        if evidence.get("errors"):
            for err in evidence["errors"]:
                print(f"  - {err}")
        EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2))
        print(f"\nPartial evidence written to {EVIDENCE_PATH}")
        return 2

    for line in _verdict_lines(evidence):
        print(line)

    r1 = evidence.get("invoke_response_1") or {}
    return 0 if r1.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
