#!/usr/bin/env python3
"""Generate a runtime validation pack from the dashboard full snapshot.

Usage examples:
  python -m scripts.run_validation_pack
  python -m scripts.run_validation_pack --host 192.168.1.222 --port 9200
  python -m scripts.run_validation_pack --strict
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError, HTTPError
from urllib.request import urlopen

from jarvis_eval.validation_pack import (
    build_runtime_validation_report,
    render_validation_markdown,
)


def _fetch_full_snapshot(host: str, port: int, timeout_s: float) -> dict[str, Any]:
    url = f"http://{host}:{port}/api/full-snapshot"
    with urlopen(url, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected /api/full-snapshot response shape")
    return payload


def _print_report(report: dict[str, Any]) -> None:
    validation = report.get("validation", {}) if isinstance(report, dict) else {}
    checks = validation.get("checks", []) if isinstance(validation, dict) else []
    status = str(report.get("status", "unknown")).lower()
    emoji = {"ready": "PASS", "caution": "WARN", "blocked": "FAIL"}.get(status, "INFO")

    print(f"[{emoji}] Runtime Validation Pack: {status.upper()}")
    print(f"  Ready for next items: {bool(report.get('ready_for_next_items', False))}")
    print(f"  Ready for continuation: {bool(report.get('ready_for_continuation', False))}")
    if report.get("continuation_mode"):
        print(f"  Continuation mode: {report.get('continuation_mode')}")
    print(f"  Next action: {report.get('next_action', '--')}")
    if report.get("continuation_action") and report.get("continuation_action") != report.get("next_action"):
        print(f"  Continuation action: {report.get('continuation_action')}")
    print(
        "  Checks: "
        f"{validation.get('checks_passing', '--')}/{validation.get('checks_total', '--')} current, "
        f"{validation.get('checks_ever_met', '--')} ever, "
        f"{validation.get('checks_regressed', '--')} regressed"
    )
    phase5 = None
    for check in checks:
        if check.get("id") == "phase5_proof_chain":
            phase5 = check
            break
    if isinstance(phase5, dict):
        phase5_icon = "PASS" if phase5.get("current_ok") else ("EVER" if phase5.get("ever_ok") else "FAIL")
        print(f"  Phase 5 proof chain: {phase5_icon} | now: {phase5.get('current_detail', '--')} | ever: {phase5.get('ever_detail', '--')}")
    targets = (
        validation.get("language_evidence_targets", [])
        if isinstance(validation.get("language_evidence_targets"), list)
        else []
    )
    if targets:
        print("  Language evidence targets:")
        for target in targets:
            if not isinstance(target, dict):
                continue
            cls = str(target.get("response_class", "--"))
            cur = "PASS" if bool(target.get("current_ok", False)) else "FAIL"
            ever = "YES" if bool(target.get("ever_ok", False)) else "NO"
            count = int(target.get("count", 0) or 0)
            target_count = int(target.get("target_count", 0) or 0)
            est = "~" if bool(target.get("count_estimated", False)) else ""
            gap = int(target.get("gap", 0) or 0)
            color = str(target.get("color", "unknown") or "unknown")
            reason = str(target.get("gate_reason", "ok") or "ok")
            print(
                f"    - {cls}: {cur} (ever={ever}) "
                f"count={est}{count}/{target_count} gap={gap} color={color} reason={reason}"
            )
    route_class = (
        validation.get("language_route_class_baselines", [])
        if isinstance(validation.get("language_route_class_baselines"), list)
        else []
    )
    if route_class:
        print("  Language baseline route/class:")
        for row in route_class:
            if not isinstance(row, dict):
                continue
            route = str(row.get("route", "--"))
            cls = str(row.get("response_class", "--"))
            cur = "PASS" if bool(row.get("current_ok", False)) else "FAIL"
            ever = "YES" if bool(row.get("ever_ok", False)) else "NO"
            count = int(row.get("count", 0) or 0)
            print(f"    - {route} -> {cls}: {cur} (ever={ever}) count={count}")
    print("")

    for check in checks:
        current_ok = bool(check.get("current_ok", False))
        ever_ok = bool(check.get("ever_ok", False))
        critical = bool(check.get("critical", False))
        if current_ok:
            icon = "[PASS]"
        elif ever_ok:
            icon = "[EVER]"
        else:
            icon = "[FAIL]"
        crit = " [CRITICAL]" if critical else ""
        print(f"{icon}{crit} {check.get('label', check.get('id', '--'))}")
        print(f"    now : {check.get('current_detail', '--')}")
        print(f"    ever: {check.get('ever_detail', '--')}")


def _write_artifacts(report: dict[str, Any], output_dir: str) -> tuple[str, str]:
    out_dir = Path(os.path.expanduser(output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"validation-pack-{ts}"
    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True, sort_keys=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_validation_markdown(report))

    return str(json_path), str(md_path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a runtime validation pack report.")
    p.add_argument("--host", default="127.0.0.1", help="Dashboard host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=9200, help="Dashboard port (default: 9200)")
    p.add_argument("--timeout", type=float, default=8.0, help="HTTP timeout in seconds")
    p.add_argument(
        "--output-dir",
        default="~/.jarvis/eval/validation_reports",
        help="Output directory for JSON/Markdown artifacts",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write report files; print to stdout only",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if report status is blocked",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        snapshot = _fetch_full_snapshot(args.host, args.port, args.timeout)
    except (URLError, HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Failed to fetch snapshot: {exc}")
        return 2
    except Exception as exc:  # defensive: keep tool usable under unexpected conditions
        print(f"[ERROR] Unexpected snapshot fetch failure: {exc}")
        return 2

    report = build_runtime_validation_report(snapshot)
    _print_report(report)

    if not args.no_write:
        try:
            json_path, md_path = _write_artifacts(report, args.output_dir)
            print("")
            print(f"[INFO] JSON: {json_path}")
            print(f"[INFO] MD  : {md_path}")
        except Exception as exc:
            print(f"[WARN] Failed to write artifacts: {exc}")
            return 1

    if args.strict and str(report.get("status", "")).lower() == "blocked":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
