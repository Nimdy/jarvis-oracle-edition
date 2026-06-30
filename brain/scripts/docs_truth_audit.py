#!/usr/bin/env python3
"""Read-only documentation truth audit for JARVIS docs surfaces.

The audit compares static dashboard/docs claims against source-of-truth code and
optional live dashboard endpoints. It never writes runtime state and is safe to
run after a fresh brain or Pi setup.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"
SEVERITY_FAIL = "fail"

STATUS_MARKER_LITERALS = {"SHIPPED", "PARTIAL", "PRE-MATURE", "DEFERRED"}

STATIC_PAGE_ROUTES: dict[str, str] = {
    "/": "index.html",
    "/docs": "docs.html",
    "/history": "history.html",
    "/api-reference": "api.html",
    "/science": "science.html",
    "/showcase": "showcase.html",
    "/capability-pipeline": "self_improve.html",
    "/self-improve": "self_improve.html",
    "/hrr": "hrr.html",
    "/hrr-scene": "hrr_scene.html",
    "/maturity": "maturity.html",
    "/learning": "learning.html",
}

FRESHNESS_REQUIRED_PAGES = set(STATIC_PAGE_ROUTES.values())


@dataclass
class Finding:
    code: str
    severity: str
    path: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "path": self.path,
            "message": self.message,
            "evidence": dict(self.evidence),
        }


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def _count_pvl_contracts(repo_root: Path) -> tuple[int | None, int | None, str | None]:
    process_contracts = repo_root / "brain" / "jarvis_eval" / "process_contracts.py"
    if not process_contracts.exists():
        return None, None, "brain/jarvis_eval/process_contracts.py missing"
    namespace: dict[str, Any] = {}
    try:
        exec(compile(_read(process_contracts), str(process_contracts), "exec"), namespace)
        contracts = namespace.get("ALL_CONTRACTS", [])
        groups = {getattr(c, "group", None) for c in contracts}
        groups.discard(None)
        return len(contracts), len(groups), None
    except Exception as exc:  # pragma: no cover - defensive for broken checkouts
        return None, None, f"{type(exc).__name__}: {exc}"


def _extract_app_api_routes(repo_root: Path) -> set[tuple[str, str]]:
    app_path = repo_root / "brain" / "dashboard" / "app.py"
    tree = ast.parse(_read(app_path), filename=str(app_path))
    routes: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Attribute):
                continue
            if not isinstance(dec.func.value, ast.Name) or dec.func.value.id != "app":
                continue
            method = dec.func.attr.upper()
            if method not in {"GET", "POST", "DELETE", "PUT", "PATCH"}:
                continue
            if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
                path = dec.args[0].value
                if path.startswith("/api/"):
                    routes.add((method, path))
    return routes


def _extract_app_page_routes(repo_root: Path) -> set[str]:
    app_path = repo_root / "brain" / "dashboard" / "app.py"
    tree = ast.parse(_read(app_path), filename=str(app_path))
    routes: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Attribute):
                continue
            if not isinstance(dec.func.value, ast.Name) or dec.func.value.id != "app":
                continue
            if dec.func.attr.lower() != "get":
                continue
            if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
                path = dec.args[0].value
                if not path.startswith("/api/") and path not in {"/static"}:
                    routes.add(path)
    return routes


def _extract_doc_api_routes(repo_root: Path) -> set[tuple[str, str]]:
    api_html = repo_root / "brain" / "dashboard" / "static" / "api.html"
    text = _read(api_html)
    routes = {
        (method, path)
        for method, path in re.findall(
            r'data-method="([^"]+)"\s+data-path="([^"]+)"', text
        )
        if path.startswith("/api/")
    }
    return routes


def _extract_build_status_pages(repo_root: Path) -> set[str]:
    app_path = repo_root / "brain" / "dashboard" / "app.py"
    text = _read(app_path)
    block = re.search(r"pages\s*=\s*\[(?P<body>.*?)\]", text, re.S)
    if not block:
        return set()
    return set(re.findall(r'"([^"]+\.html)"', block.group("body")))


def _extract_integrity_layers_from_docs_js(repo_root: Path) -> list[str]:
    docs_js = _read(repo_root / "brain" / "dashboard" / "static" / "docs.js")
    block = re.search(r"var layers = \[(?P<body>.*?)\n\s*\];", docs_js, re.S)
    if not block:
        return []
    return re.findall(r"label:\s*'([^']+)'", block.group("body"))


def _extract_integrity_layers_from_docs_html(repo_root: Path) -> list[str]:
    docs_html = _read(repo_root / "brain" / "dashboard" / "static" / "docs.html")
    section = re.search(
        r'<section id="epistemic-stack".*?</section>', docs_html, re.S
    )
    if not section:
        return []
    return re.findall(r"<tr><td[^>]*>(L\d+[A-Z]?)</td>", section.group(0))


def _enum_values(repo_root: Path, enum_name: str) -> set[str]:
    path = repo_root / "brain" / "hemisphere" / "types.py"
    tree = ast.parse(_read(path), filename=str(path))
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == enum_name:
            for item in node.body:
                if isinstance(item, ast.Assign) and isinstance(item.value, ast.Constant):
                    if isinstance(item.value.value, str):
                        values.add(item.value.value)
    return values


def _scan_static_claims(repo_root: Path, findings: list[Finding], expected_pvl: int | None, expected_groups: int | None) -> None:
    static_dir = repo_root / "brain" / "dashboard" / "static"
    current_facing = [
        repo_root / "AGENTS.md",
        repo_root / "docs" / "SYSTEM_OVERVIEW.md",
        repo_root / "docs" / "MASTER_ROADMAP.md",
        repo_root / "docs" / "MATURITY_GATES_REFERENCE.md",
        repo_root / "brain" / "dashboard" / "static" / "docs.js",
    ]
    current_facing.extend(sorted(static_dir.glob("*.html")))
    stale_pvl = re.compile(r"\b91 contracts\b")
    stale_l11 = re.compile(r"L0[–-]L11")
    stale_half_life = re.compile(r"half-life 45s")
    stale_api_routes = re.compile(r">\s*137\s*</span><span class=\"a-stat-label\">Brain API Routes")
    stale_auth_protected = re.compile(r">\s*\d+\s*</span><span class=\"a-stat-label\">Auth Protected")
    overclaim = re.compile(
        r"\b(achieved\s+AGI|achieved\s+ASI|superhuman\s+performance)\b",
        re.I,
    )
    for path in current_facing:
        if not path.exists():
            continue
        rel = str(path.relative_to(repo_root))
        text = _read(path)
        if stale_pvl.search(text):
            findings.append(Finding(
                "static.stale_pvl_count",
                SEVERITY_FAIL,
                rel,
                "Current-facing docs still mention 91 PVL contracts.",
                {"expected_contracts": expected_pvl, "expected_groups": expected_groups},
            ))
        if stale_l11.search(text):
            findings.append(Finding(
                "static.stale_integrity_range",
                SEVERITY_FAIL,
                rel,
                "Current-facing docs still describe the integrity stack as L0-L11.",
                {"expected": "L0-L12 plus L3A/L3B"},
            ))
        if stale_half_life.search(text):
            findings.append(Finding(
                "static.stale_identity_half_life",
                SEVERITY_FAIL,
                rel,
                "Current-facing docs still claim identity persistence half-life is 45s.",
                {"expected": "90s"},
            ))
        if stale_api_routes.search(text):
            findings.append(Finding(
                "static.stale_api_route_count",
                SEVERITY_FAIL,
                rel,
                "API reference still hard-codes stale Brain API route count.",
                {"expected_source": "FastAPI route inventory"},
            ))
        if stale_auth_protected.search(text):
            findings.append(Finding(
                "static.stale_auth_protected_count",
                SEVERITY_WARN,
                rel,
                "API reference hard-codes an auth-protected route count; verify against lock markers before release.",
            ))
        if overclaim.search(text):
            findings.append(Finding(
                "static.release_overclaim_phrase",
                SEVERITY_INFO,
                rel,
                "Overclaim phrase appears; verify it is used only in a negated/disallowed context.",
            ))


def _check_static_page_inventory(repo_root: Path, findings: list[Finding]) -> None:
    static_dir = repo_root / "brain" / "dashboard" / "static"
    app_routes = _extract_app_page_routes(repo_root)
    build_pages = _extract_build_status_pages(repo_root)
    expected_routes = set(STATIC_PAGE_ROUTES)
    missing_routes = sorted(expected_routes - app_routes)
    if missing_routes:
        findings.append(Finding(
            "static_pages.missing_routes",
            SEVERITY_FAIL,
            "brain/dashboard/app.py",
            "Static documentation routes are missing from FastAPI.",
            {"missing_routes": missing_routes},
        ))
    missing_files = sorted({
        page for page in set(STATIC_PAGE_ROUTES.values())
        if not (static_dir / page).exists()
    })
    if missing_files:
        findings.append(Finding(
            "static_pages.missing_files",
            SEVERITY_FAIL,
            "brain/dashboard/static",
            "Static documentation page files are missing.",
            {"missing_files": missing_files},
        ))
    missing_build = sorted(set(STATIC_PAGE_ROUTES.values()) - build_pages)
    if missing_build:
        findings.append(Finding(
            "static_pages.missing_build_status_inventory",
            SEVERITY_FAIL,
            "brain/dashboard/app.py:/api/meta/build-status",
            "Static documentation pages are missing from build-status inventory.",
            {"missing_pages": missing_build},
        ))
    html_files = {p.name for p in static_dir.glob("*.html")}
    unserved = sorted(html_files - set(STATIC_PAGE_ROUTES.values()))
    if unserved:
        findings.append(Finding(
            "static_pages.unserved_html",
            SEVERITY_WARN,
            "brain/dashboard/static",
            "HTML files exist without an explicit documentation route.",
            {"unserved": unserved},
        ))
    for page in sorted(FRESHNESS_REQUIRED_PAGES):
        path = static_dir / page
        if not path.exists():
            continue
        text = _read(path)
        if "j-freshness-banner" not in text or "freshness-banner.js" not in text:
            findings.append(Finding(
                "static_pages.missing_freshness_banner",
                SEVERITY_FAIL,
                str(path.relative_to(repo_root)),
                "Static documentation page lacks freshness banner wiring.",
            ))


def _known_status_marker_keys(repo_root: Path) -> set[str]:
    app_text = _read(repo_root / "brain" / "dashboard" / "app.py")
    marker_block = re.search(r"markers\s*=\s*\{(?P<body>.*?)\n\s*\}", app_text, re.S)
    if not marker_block:
        return set()
    return set(re.findall(r'"([^"]+)"\s*:', marker_block.group("body")))


def _check_status_marker_usage(repo_root: Path, findings: list[Finding]) -> None:
    known = _known_status_marker_keys(repo_root)
    static_dir = repo_root / "brain" / "dashboard" / "static"
    for path in sorted(static_dir.glob("*.html")):
        text = _read(path)
        for marker in re.findall(r'data-status-marker="([^"]+)"', text):
            if marker not in STATUS_MARKER_LITERALS and known and marker not in known:
                findings.append(Finding(
                    "static_pages.unknown_status_marker",
                    SEVERITY_FAIL,
                    str(path.relative_to(repo_root)),
                    "Static page references an unknown status marker key.",
                    {"marker": marker},
                ))


def _check_api_reference_stats(repo_root: Path, findings: list[Finding]) -> None:
    api_html = repo_root / "brain" / "dashboard" / "static" / "api.html"
    text = _read(api_html)
    route_count = len(_extract_app_api_routes(repo_root))
    match = re.search(
        r'<span class="a-stat-num">(\d+)</span><span class="a-stat-label">Brain API Routes</span>',
        text,
    )
    if match and int(match.group(1)) != route_count:
        findings.append(Finding(
            "api_reference.route_count_mismatch",
            SEVERITY_FAIL,
            "brain/dashboard/static/api.html",
            "API reference Brain API route count does not match FastAPI inventory.",
            {"documented": int(match.group(1)), "actual": route_count},
        ))
    missing_locks: list[str] = []
    for match in re.finditer(
        r'<div class="a-endpoint"[^>]*data-method="(POST|PUT|PATCH|DELETE)"[^>]*data-path="([^"]+)"(?P<body>.*?)(?=<div class="a-endpoint"|<!-- ═|</main>)',
        text,
        re.S,
    ):
        body = match.group("body")
        if "a-ep-lock" not in body:
            missing_locks.append(f"{match.group(1)} {match.group(2)}")
    if missing_locks:
        findings.append(Finding(
            "api_reference.mutating_routes_missing_lock",
            SEVERITY_FAIL,
            "brain/dashboard/static/api.html",
            "Mutating API reference entries must show the protected-route lock indicator.",
            {"missing_locks": missing_locks[:50], "count": len(missing_locks)},
        ))


def _check_pvl_count(repo_root: Path, findings: list[Finding]) -> tuple[int | None, int | None]:
    count, groups, error = _count_pvl_contracts(repo_root)
    if error:
        findings.append(Finding(
            "source.pvl_count_unavailable",
            SEVERITY_WARN,
            "brain/jarvis_eval/process_contracts.py",
            "Could not load PVL source contract count.",
            {"error": error},
        ))
        return count, groups
    if count != 114:
        findings.append(Finding(
            "source.pvl_count_changed",
            SEVERITY_WARN,
            "brain/jarvis_eval/process_contracts.py",
            "PVL source contract count changed; update static docs and this audit expectation.",
            {"actual_contracts": count, "actual_groups": groups, "expected_contracts": 114},
        ))
    return count, groups


def _check_api_reference(repo_root: Path, findings: list[Finding]) -> None:
    app_routes = _extract_app_api_routes(repo_root)
    doc_routes = _extract_doc_api_routes(repo_root)
    missing = sorted(app_routes - doc_routes)
    phantom = sorted(doc_routes - app_routes)
    if missing:
        findings.append(Finding(
            "api_reference.missing_routes",
            SEVERITY_FAIL,
            "brain/dashboard/static/api.html",
            "API reference is missing FastAPI routes.",
            {"missing": missing[:50], "count": len(missing)},
        ))
    if phantom:
        findings.append(Finding(
            "api_reference.phantom_routes",
            SEVERITY_FAIL,
            "brain/dashboard/static/api.html",
            "API reference documents routes that are not registered by FastAPI.",
            {"phantom": phantom[:50], "count": len(phantom)},
        ))


def _check_integrity_diagram(repo_root: Path, findings: list[Finding]) -> None:
    diagram_layers = _extract_integrity_layers_from_docs_js(repo_root)
    table_layers = _extract_integrity_layers_from_docs_html(repo_root)
    expected = ["L0", "L1", "L2", "L3", "L3A", "L3B", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12"]
    if diagram_layers != expected:
        findings.append(Finding(
            "docs_diagram.integrity_layers_mismatch",
            SEVERITY_FAIL,
            "brain/dashboard/static/docs.js",
            "Interactive architecture diagram integrity layers do not match canonical L0-L12 + L3A/L3B order.",
            {"actual": diagram_layers, "expected": expected},
        ))
    if table_layers != expected:
        findings.append(Finding(
            "docs_html.integrity_table_mismatch",
            SEVERITY_FAIL,
            "brain/dashboard/static/docs.html",
            "Epistemic integrity table does not match canonical L0-L12 + L3A/L3B order.",
            {"actual": table_layers, "expected": expected},
        ))


def _check_specialist_claims(repo_root: Path, findings: list[Finding]) -> None:
    focus_values = _enum_values(repo_root, "HemisphereFocus")
    docs_html = _read(repo_root / "brain" / "dashboard" / "static" / "docs.html")
    claimed = set(re.findall(r"<tr><td>([a-z_]+)</td><td>.*?</td><td>.*?</td></tr>", docs_html))
    tier_claims = {c for c in claimed if c in {
        "speaker_repr", "face_repr", "emotion_depth", "voice_intent",
        "speaker_diarize", "perception_fusion", "plan_evaluator",
        "diagnostic", "code_quality", "claim_classifier", "dream_synthesis",
        "skill_acquisition", "language_style", "hrr_encoder",
        "thought_trigger_selector",
    }}
    unknown = sorted(tier_claims - focus_values)
    if unknown:
        findings.append(Finding(
            "docs.specialist_claim_unknown_focus",
            SEVERITY_FAIL,
            "brain/dashboard/static/docs.html",
            "Docs list specialist focus names not declared in HemisphereFocus.",
            {"unknown": unknown},
        ))


def _fetch_json(url: str, timeout_s: float) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with urlopen(url, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if isinstance(data, dict):
            return data, None
        return None, "response was not a JSON object"
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _check_live_dashboard(host: str, port: int, timeout_s: float, findings: list[Finding]) -> dict[str, Any]:
    live: dict[str, Any] = {"checked": False, "host": host, "port": port}
    base = f"http://{host}:{port}"
    markers, error = _fetch_json(f"{base}/api/meta/status-markers", timeout_s)
    if error:
        findings.append(Finding(
            "live.dashboard_unreachable",
            SEVERITY_WARN,
            f"{base}/api/meta/status-markers",
            "Live dashboard checks skipped because dashboard endpoint is unreachable.",
            {"error": error},
        ))
        live["error"] = error
        return live
    live["checked"] = True
    marker_map = markers.get("markers", {}) if isinstance(markers, dict) else {}
    if marker_map.get("intention_resolver") != "PRE-MATURE":
        findings.append(Finding(
            "live.intention_resolver_marker",
            SEVERITY_FAIL,
            "/api/meta/status-markers",
            "IntentionResolver marker should remain PRE-MATURE until promoted by evidence gates.",
            {"actual": marker_map.get("intention_resolver")},
        ))
    full_snapshot, error = _fetch_json(f"{base}/api/full-snapshot", timeout_s)
    if error:
        findings.append(Finding(
            "live.full_snapshot_unreachable",
            SEVERITY_WARN,
            "/api/full-snapshot",
            "Could not fetch live full snapshot.",
            {"error": error},
        ))
    else:
        live["full_snapshot_keys"] = sorted(full_snapshot.keys())[:100]
    return live


_CANONICAL_STACK = [
    "L0", "L1", "L2", "L3", "L3A", "L3B", "L4", "L5",
    "L6", "L7", "L8", "L9", "L10", "L11", "L12",
]
_VALID_STATUS = {
    "shipped", "live", "shadow", "dormant", "gated",
    "partial", "planned", "absent", "signal-failure",
}


def _check_subsystem_registry(repo_root: Path, findings: list[Finding]) -> None:
    """Lock the architecture manifest against drift.

    The manifest (subsystem_registry.json) is JARVIS's code-grounded self-description.
    This check fails CI if it drifts from the code: a cited home-file vanished/moved,
    the 15-layer integrity stack changed order, or a status fell outside the enum.
    """
    reg_path = repo_root / "brain" / "subsystem_registry.json"
    if not reg_path.exists():
        findings.append(Finding(
            "registry.missing", SEVERITY_WARN, "brain/subsystem_registry.json",
            "subsystem_registry.json not found — architecture manifest is not locked.", {},
        ))
        return
    try:
        reg = json.loads(_read(reg_path))
    except (ValueError, OSError) as exc:
        findings.append(Finding(
            "registry.unparseable", SEVERITY_FAIL, "brain/subsystem_registry.json",
            f"subsystem_registry.json is not valid JSON: {exc}", {},
        ))
        return

    subs = reg.get("subsystems", []) or []

    # 1. Home-file existence — the core anti-drift check.
    missing: list[str] = []
    for s in subs:
        for hf in (s.get("home_files") or []):
            tok = re.split(r"[ :(]", str(hf).strip(), 1)[0]
            if "/" not in tok:  # descriptive, not a path
                continue
            if not (repo_root / tok).exists():
                missing.append(f"{s.get('id')}: {tok}")
    if missing:
        findings.append(Finding(
            "registry.home_file_missing", SEVERITY_FAIL, "brain/subsystem_registry.json",
            "Manifest cites home files that no longer exist (drift — code moved or registry is stale).",
            {"count": len(missing), "examples": missing[:25]},
        ))

    # 2. Integrity stack must be the canonical 15 in order.
    stack_ids = [e.get("id") for e in (reg.get("integrity_stack") or [])]
    if stack_ids != _CANONICAL_STACK:
        findings.append(Finding(
            "registry.integrity_stack_drift", SEVERITY_FAIL, "brain/subsystem_registry.json",
            "Integrity stack is not the canonical 15-entry L0-L12 + L3A/L3B order.",
            {"actual": stack_ids, "expected": _CANONICAL_STACK},
        ))

    # 3. Status enum validity.
    bad = [f"{s.get('id')}={s.get('status')}" for s in subs if s.get("status") not in _VALID_STATUS]
    if bad:
        findings.append(Finding(
            "registry.bad_status", SEVERITY_WARN, "brain/subsystem_registry.json",
            "Subsystem status values outside the canonical enum.",
            {"examples": bad[:20]},
        ))

    findings.append(Finding(
        "registry.summary", SEVERITY_INFO, "brain/subsystem_registry.json",
        f"{len(subs)} subsystems, {len(stack_ids)}-entry integrity stack, "
        f"{len(reg.get('audit_assertions') or [])} assertions.",
        {},
    ))


def _check_nn_fleet_registry(repo_root: Path, findings: list[Finding]) -> None:
    """Lock the NN fleet registry against drift + surface inference-orphaned real NNs.

    nn_fleet_registry.json tracks every NN (toward hundreds). This fails CI if a cited home
    file vanished, and WARNS with the count of NNs whose inference is ORPHANED/BROKEN so that
    'feed wired, inference orphaned' regressions stay visible and cannot silently grow.
    """
    reg_path = repo_root / "brain" / "nn_fleet_registry.json"
    if not reg_path.exists():
        findings.append(Finding(
            "nnfleet.missing", SEVERITY_WARN, "brain/nn_fleet_registry.json",
            "nn_fleet_registry.json not found — NN fleet is not tracked.", {},
        ))
        return
    try:
        reg = json.loads(_read(reg_path))
    except (ValueError, OSError) as exc:
        findings.append(Finding(
            "nnfleet.unparseable", SEVERITY_FAIL, "brain/nn_fleet_registry.json",
            f"nn_fleet_registry.json is not valid JSON: {exc}", {},
        ))
        return
    recs = reg.get("records", []) or []
    missing: list[str] = []
    orphaned: list[str] = []
    for r in recs:
        home = str(r.get("home") or "").strip()
        if home and "/" in home and not (repo_root / "brain" / home).exists():
            missing.append(f"{r.get('name')}: {home}")
        if str(r.get("wiring_confirmed")) in ("ORPHANED", "BROKEN"):
            orphaned.append(str(r.get("name")))
    if missing:
        findings.append(Finding(
            "nnfleet.home_file_missing", SEVERITY_FAIL, "brain/nn_fleet_registry.json",
            "NN registry cites home files that no longer exist (drift).",
            {"count": len(missing), "examples": missing[:25]},
        ))
    if orphaned:
        findings.append(Finding(
            "nnfleet.inference_orphaned", SEVERITY_WARN, "brain/nn_fleet_registry.json",
            "Real NNs whose inference output nothing consumes (or broken) — doing no work despite existing.",
            {"count": len(orphaned), "nns": orphaned},
        ))


_CONNECTOME_EXPECTED_DYNAMIC = 6


def _check_connectome_emit_map(repo_root: Path, findings: list[Finding]) -> None:
    """Lock the live connectome's code-derived emitter map (dashboard/connectome.py). It attributes each
    edge's emitter by scanning .emit() sites; dynamic (variable) emit sites are unattributable ->
    RELAYED_UNKNOWN. FAIL if the scan parses ~nothing (regression that would blank the emitter side);
    WARN if the unattributable count drifts up (a new emitter that the connectome can't ground)."""
    brain = repo_root / "brain"
    events_py = brain / "consciousness" / "events.py"
    const_re = re.compile(r'^([A-Z][A-Z0-9_]+)\s*=\s*["\']([a-z_]+:[a-z_0-9]+)["\']', re.M)
    emit_re = re.compile(r'\.emit(?:_event)?\(\s*([A-Z][A-Z0-9_]+|["\'][a-z_]+:[a-z_0-9]+["\'])')
    const: dict[str, str] = {}
    try:
        for m in const_re.finditer(_read(events_py)):
            const[m.group(1)] = m.group(2)
    except OSError:
        pass
    if len(const) < 50:
        findings.append(Finding(
            "connectome.const_parse", SEVERITY_FAIL, "brain/consciousness/events.py",
            "Connectome event-const table parsed too few entries — emitter attribution would blank out.",
            {"parsed": len(const)}))
        return
    events: set[str] = set()
    dynamic = 0
    for f in brain.rglob("*.py"):
        sp = str(f)
        if "/tests/" in sp or "__pycache__" in sp:
            continue
        try:
            txt = _read(f)
        except OSError:
            continue
        if ".emit" not in txt:
            continue
        for m in emit_re.finditer(txt):
            tok = m.group(1)
            if tok[0] in "\"'":
                events.add(tok.strip("\"'"))
            elif tok in const:
                events.add(const[tok])
            else:
                dynamic += 1
    if len(events) < 60:
        findings.append(Finding(
            "connectome.emit_scan", SEVERITY_FAIL, "brain/",
            "Connectome emit-site scan resolved too few events — likely a parse regression.",
            {"resolved": len(events)}))
    if dynamic != _CONNECTOME_EXPECTED_DYNAMIC:
        findings.append(Finding(
            "connectome.dynamic_emit_drift", SEVERITY_WARN, "brain/",
            f"Connectome unattributable (dynamic) emit-site count changed "
            f"{_CONNECTOME_EXPECTED_DYNAMIC}->{dynamic} — new RELAYED_UNKNOWN edges; confirm the live "
            "connectome still grounds emitters (bump the expected count if intended).",
            {"expected": _CONNECTOME_EXPECTED_DYNAMIC, "actual": dynamic}))


def run_audit(repo_root: Path, *, host: str | None = None, port: int = 9200, timeout_s: float = 5.0) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    findings: list[Finding] = []
    pvl_count, pvl_groups = _check_pvl_count(repo_root, findings)
    _check_static_page_inventory(repo_root, findings)
    _check_integrity_diagram(repo_root, findings)
    _check_api_reference(repo_root, findings)
    _check_api_reference_stats(repo_root, findings)
    _check_status_marker_usage(repo_root, findings)
    _check_specialist_claims(repo_root, findings)
    _scan_static_claims(repo_root, findings, pvl_count, pvl_groups)
    _check_subsystem_registry(repo_root, findings)
    _check_nn_fleet_registry(repo_root, findings)
    _check_connectome_emit_map(repo_root, findings)
    live: dict[str, Any] = {"checked": False}
    if host:
        live = _check_live_dashboard(host, port, timeout_s, findings)
    return {
        "ok": not any(f.severity == SEVERITY_FAIL for f in findings),
        "repo_root": str(repo_root),
        "source_truth": {
            "pvl_contracts": pvl_count,
            "pvl_groups": pvl_groups,
            "api_routes": len(_extract_app_api_routes(repo_root)),
            "documented_api_routes": len(_extract_doc_api_routes(repo_root)),
        },
        "live": live,
        "findings": [f.to_dict() for f in findings],
        "summary": {
            "fail": sum(1 for f in findings if f.severity == SEVERITY_FAIL),
            "warn": sum(1 for f in findings if f.severity == SEVERITY_WARN),
            "info": sum(1 for f in findings if f.severity == SEVERITY_INFO),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only docs truth audit")
    parser.add_argument("--repo-root", default=str(_repo_root_from_script()))
    parser.add_argument("--host", default=None, help="Optional live dashboard host")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on fail findings")
    args = parser.parse_args(argv)

    report = run_audit(Path(args.repo_root), host=args.host, port=args.port, timeout_s=args.timeout)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        summary = report["summary"]
        print(f"[{status}] Docs Truth Audit")
        print(
            f"  Findings: {summary['fail']} fail / {summary['warn']} warn / {summary['info']} info"
        )
        print(
            "  Source truth: "
            f"PVL={report['source_truth']['pvl_contracts']} contracts, "
            f"API={report['source_truth']['api_routes']} registered / "
            f"{report['source_truth']['documented_api_routes']} documented"
        )
        for finding in report["findings"]:
            print(f"[{finding['severity'].upper()}] {finding['code']} @ {finding['path']}")
            print(f"  {finding['message']}")
    if args.strict and not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
