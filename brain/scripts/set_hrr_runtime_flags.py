#!/usr/bin/env python3
"""Manage the persistent HRR runtime-flag file.

This is the operator-facing tool for the P5.1 runtime-flag persistence
layer. It writes ``~/.jarvis/runtime_flags.json`` (or the path pointed
at by ``JARVIS_RUNTIME_FLAGS``) so an operator's brain can keep the HRR
shadow + P5 mental-world lane enabled across restarts without exporting
shell environment variables every time.

It does **not** flip the dashboard status marker. ``spatial_hrr_mental_world``
remains ``PRE-MATURE`` regardless of what this script writes — promotion
remains a governance act, never an operator switch.

Usage examples::

    # Enable HRR + P5 mental-world on this brain (still requires restart).
    PYTHONPATH=brain python brain/scripts/set_hrr_runtime_flags.py --enable

    # Disable both lanes for the next restart.
    PYTHONPATH=brain python brain/scripts/set_hrr_runtime_flags.py --disable

    # Inspect what the next boot will read (default → file → env layered view).
    PYTHONPATH=brain python brain/scripts/set_hrr_runtime_flags.py --status

The script prints the resolved boot config (with provenance) at the end
of every successful run, so the operator can confirm what would happen
on the next restart before sending SIGTERM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


def _import_runtime_config():
    """Import :class:`HRRRuntimeConfig` lazily so the script works under PYTHONPATH=brain."""
    try:
        from library.vsa.runtime_config import (  # type: ignore[import-not-found]
            DEFAULT_RUNTIME_FLAGS_PATH,
            HRRRuntimeConfig,
            RUNTIME_FLAGS_PATH_ENV,
            _resolve_runtime_flags_path,
        )
    except ImportError as exc:
        sys.stderr.write(
            f"set_hrr_runtime_flags: cannot import library.vsa.runtime_config: {exc}\n"
            "Make sure to run with PYTHONPATH=brain.\n"
        )
        raise SystemExit(2) from exc
    return (
        HRRRuntimeConfig,
        DEFAULT_RUNTIME_FLAGS_PATH,
        RUNTIME_FLAGS_PATH_ENV,
        _resolve_runtime_flags_path,
    )


def _read_existing(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return dict(data)


def _write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_status(path: Path) -> None:
    HRRRuntimeConfig, _, env_var, _ = _import_runtime_config()
    cfg = HRRRuntimeConfig.from_env()
    file_payload = _read_existing(path)
    print("=== HRR runtime-flag status ===")
    print(f"  runtime_flags path: {path}")
    print(f"  override env var:   {env_var}={os.environ.get(env_var) or '(unset)'}")
    print(f"  file exists:        {path.exists()}")
    if path.exists():
        print(f"  file payload:       {json.dumps(file_payload, sort_keys=True)}")
    if cfg.runtime_flags_error:
        print(f"  file error:         {cfg.runtime_flags_error}")
    print("  resolved boot config:")
    print(f"    enabled                            = {cfg.enabled}  (source={cfg.enabled_source})")
    print(
        "    spatial_scene_enabled              = "
        f"{cfg.spatial_scene_enabled}  (source={cfg.spatial_scene_enabled_source})"
    )
    print(
        "    spatial_scene_active (twin gate)   = "
        f"{cfg.spatial_scene_active}"
    )
    print(f"    dim                                = {cfg.dim}")
    print(f"    sample_every_ticks                 = {cfg.sample_every_ticks}")
    print(
        "    spatial_scene_sample_every_ticks   = "
        f"{cfg.spatial_scene_sample_every_ticks}"
    )
    print("  flag_sources:")
    for k, v in sorted(cfg.flag_sources_dict.items()):
        print(f"    {k}: {v}")


def main(argv: list[str] | None = None) -> int:
    HRRRuntimeConfig, default_path, _, _ = _import_runtime_config()

    parser = argparse.ArgumentParser(
        description="Manage the persistent HRR runtime-flag file (P5.1)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--enable",
        action="store_true",
        help="Set enable_hrr_shadow=true AND enable_hrr_spatial_scene=true.",
    )
    group.add_argument(
        "--disable",
        action="store_true",
        help="Set both flags to false (keys retained for transparency).",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Print the resolved boot config (default → file → env) and exit.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help=(
            "Override the runtime_flags file location. Defaults to the "
            "JARVIS_RUNTIME_FLAGS env if set, otherwise ~/.jarvis/runtime_flags.json."
        ),
    )
    parser.add_argument(
        "--sample-every-ticks",
        type=int,
        default=None,
        help="Optional: persist hrr_sample_every_ticks (world shadow cadence).",
    )
    parser.add_argument(
        "--spatial-scene-sample-every-ticks",
        type=int,
        default=None,
        help="Optional: persist hrr_spatial_scene_sample_every_ticks (P5 cadence).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Optional: persist hrr_shadow_dim.",
    )
    args = parser.parse_args(argv)

    target = (
        args.path
        if args.path is not None
        else (
            Path(os.environ.get("JARVIS_RUNTIME_FLAGS"))
            if os.environ.get("JARVIS_RUNTIME_FLAGS")
            else default_path
        )
    )

    if args.status:
        _print_status(target)
        return 0

    payload = _read_existing(target)
    if args.enable:
        payload["enable_hrr_shadow"] = True
        payload["enable_hrr_spatial_scene"] = True
    elif args.disable:
        payload["enable_hrr_shadow"] = False
        payload["enable_hrr_spatial_scene"] = False

    if args.sample_every_ticks is not None:
        payload["hrr_sample_every_ticks"] = max(1, int(args.sample_every_ticks))
    if args.spatial_scene_sample_every_ticks is not None:
        payload["hrr_spatial_scene_sample_every_ticks"] = max(
            1, int(args.spatial_scene_sample_every_ticks)
        )
    if args.dim is not None:
        payload["hrr_shadow_dim"] = max(16, int(args.dim))

    _write(target, payload)
    print(f"wrote runtime flags to {target}:")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print()
    print(
        "NOTE: the running brain will not pick up these flags until the "
        "supervisor / main.py is restarted. Status marker "
        "spatial_hrr_mental_world remains PRE-MATURE regardless."
    )
    print()
    _print_status(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
