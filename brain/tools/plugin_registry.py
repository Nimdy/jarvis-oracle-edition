"""Plugin Registry — lifecycle manager for dynamically generated tool plugins.

Quarantine-first lifecycle: quarantined → shadow → supervised → active → disabled.
Generated plugins land in quarantined state (exist on disk, NOT routable).
Only after review, approval, and shadow-mode validation do they become callable.

Safety model:
  - Import allowlist (validated via AST at quarantine time)
  - Runtime timeout (per-invocation, default 30s)
  - Permission envelope (declared capabilities matched to actual behavior)
  - Per-plugin audit trail (append-only JSONL)
  - Version pinning + rollback/disable
  - Circuit breaker (3 crashes in 10 min = auto-disable)
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PLUGINS_DIR = Path(__file__).parent / "plugins"
_JARVIS_DIR = Path.home() / ".jarvis"
_REGISTRY_PATH = _JARVIS_DIR / "plugin_registry.json"
_AUDIT_DIR = _JARVIS_DIR / "plugin_audit"

CIRCUIT_BREAKER_FAILURES = 3
CIRCUIT_BREAKER_WINDOW_S = 600  # 10 minutes

ALWAYS_ALLOWED_IMPORTS = frozenset({
    "json", "re", "datetime", "pathlib", "os.path", "typing", "dataclasses",
    "collections", "math", "hashlib", "base64", "urllib.parse", "html", "csv",
    "io", "time", "functools", "itertools", "enum", "abc", "logging",
    "random", "string", "textwrap", "copy", "operator", "decimal", "fractions",
    "statistics", "uuid",
})

TIER1_IMPORTS = frozenset({
    "requests", "httpx", "bs4", "lxml", "yaml", "toml",
})

TIER2_IMPORTS = frozenset({
    "aiohttp", "paramiko", "ftplib",
})

NEVER_ALLOWED_IMPORTS = frozenset({
    "subprocess", "os.system", "os.popen", "eval", "exec", "__import__",
    "socket", "ctypes", "importlib",
})

PluginState = str  # "quarantined" | "shadow" | "supervised" | "active" | "disabled"


# ---------------------------------------------------------------------------
# Plugin manifest / request / response
# ---------------------------------------------------------------------------

@dataclass
class PluginManifest:
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    intent_patterns: list[str] = field(default_factory=list)
    created_by: str = ""
    skill_id: str = ""
    risk_tier: int = 0
    approved_by: str = ""
    supervision_mode: str = "shadow"
    permissions: list[str] = field(default_factory=list)
    allowed_imports: list[str] = field(default_factory=list)
    timeout_s: float = 30.0
    created_at: str = ""
    execution_mode: str = "in_process"  # "in_process" | "isolated_subprocess"
    pinned_dependencies: list[str] = field(default_factory=list)
    verify_imports: list[str] = field(default_factory=list)
    invocation_schema_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PluginManifest:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class PluginRequest:
    """Stable request envelope — same shape in-process or over IPC."""
    request_id: str = ""
    plugin_name: str = ""
    user_text: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 30.0


@dataclass
class PluginResponse:
    """Stable response envelope."""
    request_id: str = ""
    plugin_name: str = ""
    success: bool = False
    result: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: float = 0.0
    audit_entry: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Plugin metadata (persisted in registry)
# ---------------------------------------------------------------------------

@dataclass
class PluginRecord:
    name: str = ""
    state: PluginState = "quarantined"
    version: str = "1.0.0"
    fallback_version: str = ""
    supervision_mode: str = "shadow"
    risk_tier: int = 0
    approved_by: str = ""
    acquisition_id: str = ""
    # Capability authority (shadow-first, reversible): the skill this plugin is a
    # version of (the general grouping key — at most one version is `active` per
    # skill_id), and the last-known-good version it displaced when promoted (the
    # floor to fall back to on demote). See docs/CAPABILITY_AUTHORITY_DESIGN.md.
    skill_id: str = ""
    generation: int = 0                  # 0 = original; N = the Nth improvement (for plain labels)
    prior_authoritative: str = ""        # plugin name that was active before this one
    last_authoritative_at: float = 0.0   # last time THIS plugin was the active version
    code_hash: str = ""
    execution_mode: str = "in_process"  # "in_process" | "isolated_subprocess"
    venv_ready: bool = False

    # Runtime stats
    invocation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_invocation_at: float = 0.0
    avg_latency_ms: float = 0.0

    # Circuit breaker
    recent_failures: list[float] = field(default_factory=list)

    # Lifecycle timestamps
    activated_at: float = 0.0

    # Upgrade tracking
    upgrade_candidacy: str = "none"  # none | stale_dep | failure_rate | doc_update
    doc_artifact_ids: list[str] = field(default_factory=list)

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PluginRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# PluginRegistry
# ---------------------------------------------------------------------------

class PluginRegistry:
    """Manages the plugin lifecycle: quarantine, shadow, supervised, active, disabled."""

    def __init__(self, plugins_dir: Path | None = None) -> None:
        self._plugins_dir = plugins_dir or _PLUGINS_DIR
        self._plugins_dir.mkdir(parents=True, exist_ok=True)
        _AUDIT_DIR.mkdir(parents=True, exist_ok=True)

        self._records: dict[str, PluginRecord] = {}
        self._handlers: dict[str, Callable] = {}
        self._compiled_patterns: dict[str, list[re.Pattern[str]]] = {}
        self._process_managers: dict[str, Any] = {}

        self._load_registry()
        self._discover_plugins()

    # ── Persistence ────────────────────────────────────────────────────

    def _load_registry(self) -> None:
        if _REGISTRY_PATH.exists():
            try:
                data = json.loads(_REGISTRY_PATH.read_text())
                for name, rec_dict in data.items():
                    self._records[name] = PluginRecord.from_dict(rec_dict)
            except Exception:
                logger.warning("Failed to load plugin registry")

    def _save_registry(self) -> None:
        try:
            from memory.persistence import atomic_write_json
            data = {name: rec.to_dict() for name, rec in self._records.items()}
            atomic_write_json(_REGISTRY_PATH, data)
        except Exception:
            logger.warning("Failed to save plugin registry")

    # ── Discovery ──────────────────────────────────────────────────────

    def _discover_plugins(self) -> None:
        """Scan plugins directory for existing plugin packages."""
        if not self._plugins_dir.exists():
            return
        for entry in self._plugins_dir.iterdir():
            if entry.is_dir() and (entry / "__init__.py").exists():
                name = entry.name
                if name.startswith("_"):
                    continue
                if name not in self._records:
                    self._records[name] = PluginRecord(name=name, state="quarantined")
                self._try_load_handler(name)

    def _try_load_handler(self, name: str) -> None:
        """Attempt to load a plugin handler (for active/shadow/supervised plugins).

        For isolated_subprocess plugins, only intent patterns are loaded from the
        manifest on disk (via safe AST extraction). The handler is NOT loaded into
        the brain process — invocation goes through PluginProcessManager instead.
        """
        rec = self._records.get(name)
        if not rec or rec.state in ("quarantined", "disabled"):
            return

        plugin_dir = self._plugins_dir / name
        init_file = plugin_dir / "__init__.py"
        if not init_file.exists():
            return

        if rec.execution_mode == "isolated_subprocess":
            try:
                content = init_file.read_text()
                manifest_dict = self._extract_manifest_safe(content)
                if manifest_dict and manifest_dict.get("intent_patterns"):
                    self._compiled_patterns[name] = [
                        re.compile(p, re.I) for p in manifest_dict["intent_patterns"]
                    ]
            except Exception:
                logger.warning("Failed to load manifest for isolated plugin: %s", name, exc_info=True)
            return

        try:
            import importlib.util, sys as _sys
            pkg_name = f"_jarvis_plugins.{name}"
            spec = importlib.util.spec_from_file_location(
                pkg_name, init_file,
                submodule_search_locations=[str(plugin_dir)],
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                _sys.modules[pkg_name] = mod
                spec.loader.exec_module(mod)
                handler = getattr(mod, "handle", None)
                manifest_dict = getattr(mod, "PLUGIN_MANIFEST", {})
                if handler:
                    self._handlers[name] = handler
                if manifest_dict.get("intent_patterns"):
                    self._compiled_patterns[name] = [
                        re.compile(p, re.I) for p in manifest_dict["intent_patterns"]
                    ]
        except Exception:
            logger.warning("Failed to load plugin handler: %s", name, exc_info=True)

    # ── Quarantine (Phase 4 core) ──────────────────────────────────────

    def quarantine(
        self,
        plugin_name: str,
        code_files: dict[str, str],
        manifest: PluginManifest,
        acquisition_id: str = "",
    ) -> tuple[bool, list[str]]:
        """Deploy a generated plugin to disk in quarantined state.

        Returns (success, validation_errors).
        The plugin is NOT routable until activated.
        """
        errors = self._validate_plugin(code_files, manifest)
        if errors:
            return False, errors

        # Write to disk
        plugin_dir = self._plugins_dir / plugin_name
        plugin_dir.mkdir(parents=True, exist_ok=True)

        for filename, content in code_files.items():
            (plugin_dir / filename).write_text(content)

        # Write manifest + handle() bridge to __init__.py if not present
        if "__init__.py" not in code_files:
            init_content = (
                f"PLUGIN_MANIFEST = {json.dumps(manifest.to_dict(), indent=2)}\n\n\n"
                "async def handle(text: str, context: dict) -> dict:\n"
                "    try:\n"
                "        from .handler import run\n"
                "    except Exception:\n"
                '        return {"output": "Plugin handler not available"}\n'
                "    try:\n"
                '        return {"output": run({"request": text})}\n'
                "    except Exception as exc:\n"
                '        return {"output": f"Plugin execution failed: {exc}"}\n'
            )
            (plugin_dir / "__init__.py").write_text(init_content)

        # Write VERSION
        (plugin_dir / "VERSION").write_text(manifest.version + "\n")

        # Create record
        code_hash = hashlib.sha256(
            json.dumps(code_files, sort_keys=True).encode()
        ).hexdigest()[:16]

        rec = PluginRecord(
            name=plugin_name,
            state="quarantined",
            version=manifest.version,
            risk_tier=manifest.risk_tier,
            acquisition_id=acquisition_id,
            skill_id=getattr(manifest, "skill_id", "") or "",
            code_hash=code_hash,
            supervision_mode=manifest.supervision_mode,
            execution_mode=manifest.execution_mode,
        )
        self._records[plugin_name] = rec
        self._save_registry()

        logger.info("Plugin quarantined: %s (v%s, hash=%s)", plugin_name, manifest.version, code_hash)
        return True, []

    def _validate_plugin(self, code_files: dict[str, str], manifest: PluginManifest) -> list[str]:
        """Validate a plugin before quarantine."""
        errors: list[str] = []

        # AST parse all Python files
        for filename, content in code_files.items():
            if filename.endswith(".py"):
                try:
                    ast.parse(content, filename=filename)
                except SyntaxError as exc:
                    errors.append(f"Syntax error in {filename}: {exc}")

        # Import allowlist validation
        for filename, content in code_files.items():
            if filename.endswith(".py"):
                import_errors = self._check_imports(content, filename, manifest)
                errors.extend(import_errors)

        # Denied patterns from patch_plan
        try:
            from self_improve.patch_plan import check_denied_patterns
            for filename, content in code_files.items():
                if filename.endswith(".py"):
                    denied = check_denied_patterns(content)
                    for d in denied:
                        errors.append(f"Denied pattern in {filename}: {d}")
        except Exception:
            pass

        if not manifest.name:
            errors.append("Plugin manifest must have a name")

        # Execution mode validation
        if manifest.execution_mode not in ("in_process", "isolated_subprocess"):
            errors.append(
                f"Invalid execution_mode '{manifest.execution_mode}': "
                "must be 'in_process' or 'isolated_subprocess'"
            )

        # In-process plugins must not declare pinned dependencies
        if manifest.execution_mode == "in_process" and manifest.pinned_dependencies:
            errors.append(
                "in_process plugins must not declare pinned_dependencies "
                "(use isolated_subprocess for external packages)"
            )

        # Pinned dependency format: must be exact pins (package==x.y.z)
        _PIN_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9._-]*==\d+(\.\d+)*$")
        for dep in manifest.pinned_dependencies:
            if not _PIN_RE.match(dep):
                errors.append(
                    f"Dependency '{dep}' must be an exact pin (package==x.y.z). "
                    "Floating versions (>=, ~=, >, <) are not allowed."
                )

        # setup_commands are explicitly forbidden in v1
        if hasattr(manifest, "setup_commands") and getattr(manifest, "setup_commands", None):
            errors.append("setup_commands are forbidden in invocation_schema_version 1")

        return errors

    def _check_imports(self, source: str, filename: str, manifest: PluginManifest) -> list[str]:
        """Validate imports against allowlist."""
        errors: list[str] = []
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError:
            return errors

        declared = set(manifest.allowed_imports)
        all_allowed = ALWAYS_ALLOWED_IMPORTS | TIER1_IMPORTS | TIER2_IMPORTS | declared

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split(".")[0]
                    if mod in NEVER_ALLOWED_IMPORTS:
                        errors.append(f"Forbidden import in {filename}: {alias.name}")
                    elif mod not in all_allowed and mod not in declared:
                        errors.append(f"Undeclared import in {filename}: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue  # intra-package relative import -- always allowed
                if node.module:
                    mod = node.module.split(".")[0]
                    if mod in NEVER_ALLOWED_IMPORTS:
                        errors.append(f"Forbidden import in {filename}: from {node.module}")
                    elif mod not in all_allowed and mod not in declared:
                        errors.append(f"Undeclared import in {filename}: from {node.module}")

        return errors

    # ── Activation / State Transitions ─────────────────────────────────

    def activate(self, plugin_name: str, supervision_mode: str = "shadow") -> bool:
        """Move plugin from quarantined to shadow/supervised/active."""
        rec = self._records.get(plugin_name)
        if not rec:
            return False
        if rec.state not in ("quarantined", "disabled"):
            return False

        rec.state = supervision_mode if supervision_mode in ("shadow", "supervised", "active") else "shadow"
        rec.supervision_mode = supervision_mode
        rec.activated_at = time.time()
        rec.updated_at = time.time()
        self._save_registry()
        self._try_load_handler(plugin_name)

        logger.info("Plugin activated: %s -> %s", plugin_name, rec.state)
        return True

    def disable(self, plugin_name: str, reason: str = "") -> bool:
        """Disable a plugin."""
        rec = self._records.get(plugin_name)
        if not rec:
            return False
        rec.state = "disabled"
        rec.updated_at = time.time()
        self._save_registry()
        self._handlers.pop(plugin_name, None)
        self._compiled_patterns.pop(plugin_name, None)

        self._audit_log(plugin_name, {"action": "disabled", "reason": reason})
        logger.info("Plugin disabled: %s (reason: %s)", plugin_name, reason)
        return True

    def rollback(self, plugin_name: str) -> bool:
        """Revert to fallback version."""
        rec = self._records.get(plugin_name)
        if not rec or not rec.fallback_version:
            return False

        rec.version = rec.fallback_version
        rec.fallback_version = ""
        rec.state = "quarantined"
        rec.updated_at = time.time()
        self._save_registry()
        self._handlers.pop(plugin_name, None)

        self._audit_log(plugin_name, {"action": "rollback", "to_version": rec.version})
        logger.info("Plugin rolled back: %s -> v%s", plugin_name, rec.version)
        return True

    def promote(self, plugin_name: str) -> bool:
        """Promote a plugin to the next supervision tier."""
        rec = self._records.get(plugin_name)
        if not rec:
            return False

        transitions = {
            "shadow": "supervised",
            "supervised": "active",
        }
        new_state = transitions.get(rec.state)
        if not new_state:
            return False

        rec.state = new_state
        rec.supervision_mode = new_state
        rec.updated_at = time.time()
        self._save_registry()

        self._audit_log(plugin_name, {"action": "promoted", "to_state": new_state})
        logger.info("Plugin promoted: %s -> %s", plugin_name, new_state)
        return True

    # ── Capability authority (shadow-first, reversible) ────────────────
    # General, skill-agnostic: at most one version per skill_id is `active`.
    # Promotion raises authority (owner-gated); demotion lowers it (owner OR auto).
    # See docs/CAPABILITY_AUTHORITY_DESIGN.md.

    _SET_SKILL = "set_skill_id"  # noqa

    def set_skill_id(self, plugin_name: str, skill_id: str, generation: int = 0) -> bool:
        """Bind a plugin record to its skill (the version-grouping key) and its generation
        (0 = original, N = the Nth improvement) for plain-language labels."""
        rec = self._records.get(plugin_name)
        if not rec:
            return False
        rec.skill_id = skill_id or ""
        rec.generation = int(generation or 0)
        rec.updated_at = time.time()
        self._save_registry()
        return True

    def set_intent_patterns(self, plugin_name: str, patterns: list[str]) -> bool:
        """Set/replace a plugin's intent trigger patterns (the data-owned trigger) and
        recompile them live. Anchored to the skill via the record; self-cleaning on removal."""
        rec = self._records.get(plugin_name)
        if not rec:
            return False
        valid: list[str] = []
        for p in patterns or []:
            if not isinstance(p, str) or not p or len(p) > 200:
                continue
            try:
                re.compile(p, re.I)
                valid.append(p)
            except re.error:
                pass
        if not valid:
            return False
        self._compiled_patterns[plugin_name] = [re.compile(p, re.I) for p in valid]
        # persist into the stored manifest so it survives restart
        try:
            mdir = self._plugins_dir / plugin_name
            mpath = mdir / "manifest.json"
            if mpath.exists():
                m = json.loads(mpath.read_text())
                m["intent_patterns"] = valid
                from memory.persistence import atomic_write_json
                atomic_write_json(mpath, m)
        except Exception:
            pass
        rec.updated_at = time.time()
        self._save_registry()
        self._audit_log(plugin_name, {"action": "intent_patterns_set", "count": len(valid)})
        return True

    def versions_for_skill(self, skill_id: str) -> list[PluginRecord]:
        """All plugin versions bound to a skill, newest-activated first."""
        if not skill_id:
            return []
        recs = [r for r in self._records.values() if getattr(r, "skill_id", "") == skill_id]
        recs.sort(key=lambda r: (r.last_authoritative_at, r.activated_at, r.created_at), reverse=True)
        return recs

    def active_for_skill(self, skill_id: str) -> PluginRecord | None:
        """The single authoritative version for a skill, if any."""
        for r in self.versions_for_skill(skill_id):
            if r.state == "active":
                return r
        return None

    def _known_good_floor(self, skill_id: str, exclude: str) -> PluginRecord | None:
        """The version to fall back to when the active one is demoted: the most
        recently-authoritative healthy sibling (shadow/supervised), excluding `exclude`."""
        if not skill_id:
            return None
        best = None
        for r in self.versions_for_skill(skill_id):
            if r.name == exclude:
                continue
            if r.state in ("shadow", "supervised") and r.last_authoritative_at > 0:
                if best is None or r.last_authoritative_at > best.last_authoritative_at:
                    best = r
        return best

    def make_authoritative(self, plugin_name: str, approved_by: str = "", reason: str = "") -> bool:
        """Owner-gated: make this version the ACTIVE (authoritative) one for its skill.

        Atomic per skill — demotes the current active version to shadow and records it on
        the new version as its ``prior_authoritative`` floor. Raises authority, so the
        caller must already be the operator (the HTTP route is api-key gated).
        """
        rec = self._records.get(plugin_name)
        if not rec:
            return False
        if rec.state not in ("shadow", "supervised", "active"):
            return False  # must be eligible (not quarantined/disabled)

        skill_id = getattr(rec, "skill_id", "") or ""
        displaced = self.active_for_skill(skill_id) if skill_id else None
        if displaced and displaced.name != plugin_name:
            displaced.state = "shadow"
            displaced.supervision_mode = "shadow"
            displaced.last_authoritative_at = displaced.last_authoritative_at or time.time()
            displaced.updated_at = time.time()
            rec.prior_authoritative = displaced.name
            self._audit_log(displaced.name, {
                "action": "authority_changed", "from": "active", "to": "shadow",
                "actor": approved_by or "owner", "reason": f"displaced by {plugin_name}",
            })

        rec.state = "active"
        rec.supervision_mode = "active"
        rec.approved_by = approved_by or rec.approved_by
        rec.activated_at = rec.activated_at or time.time()
        rec.last_authoritative_at = time.time()
        rec.updated_at = time.time()
        self._save_registry()
        self._try_load_handler(plugin_name)
        self._audit_log(plugin_name, {
            "action": "authority_changed", "from": "shadow", "to": "active",
            "actor": approved_by or "owner", "reason": reason or "made authoritative",
            "skill_id": skill_id, "floor": rec.prior_authoritative,
        })
        self._emit_authority_event(rec, "active", approved_by or "owner", reason)
        logger.info("Capability authority: %s -> ACTIVE for skill '%s' (floor=%s)",
                    plugin_name, skill_id, rec.prior_authoritative or "none")
        return True

    def demote(self, plugin_name: str, reason: str = "", actor: str = "owner") -> dict[str, Any]:
        """Lower authority: active/supervised -> shadow (reversible circuit breaker).

        NOT owner-gated (lowering authority is always safe — the asymmetric gate). When the
        demoted plugin was the skill's active version, the last-known-good floor is restored
        to active so the skill keeps serving on the prior trusted version; if there is no
        floor the skill goes dormant (no active) — safe by design.
        """
        rec = self._records.get(plugin_name)
        if not rec:
            return {"ok": False, "error": "not_found"}
        if rec.state not in ("active", "supervised"):
            return {"ok": False, "error": f"not authoritative (state={rec.state})"}

        was_active = rec.state == "active"
        skill_id = getattr(rec, "skill_id", "") or ""
        if was_active:
            rec.last_authoritative_at = time.time()
        rec.state = "shadow"
        rec.supervision_mode = "shadow"
        rec.updated_at = time.time()
        self._audit_log(plugin_name, {
            "action": "authority_changed", "from": "active" if was_active else "supervised",
            "to": "shadow", "actor": actor, "reason": reason or "demoted",
        })
        self._emit_authority_event(rec, "shadow", actor, reason)

        report: dict[str, Any] = {"ok": True, "demoted": plugin_name, "actor": actor,
                                  "fell_back_to": None, "dormant": False}
        if was_active and skill_id:
            floor = self._known_good_floor(skill_id, exclude=plugin_name)
            if floor is not None:
                floor.state = "active"
                floor.supervision_mode = "active"
                floor.last_authoritative_at = time.time()
                floor.updated_at = time.time()
                self._try_load_handler(floor.name)
                self._audit_log(floor.name, {
                    "action": "authority_changed", "from": "shadow", "to": "active",
                    "actor": "auto:fallback", "reason": f"restored after {plugin_name} demoted",
                })
                self._emit_authority_event(floor, "active", "auto:fallback", "known-good floor restored")
                report["fell_back_to"] = floor.name
            else:
                report["dormant"] = True
        self._save_registry()
        logger.info("Capability authority: %s demoted to shadow by %s (fallback=%s, dormant=%s)",
                    plugin_name, actor, report["fell_back_to"], report["dormant"])
        return report

    def _emit_authority_event(self, rec: PluginRecord, to_state: str, actor: str, reason: str) -> None:
        try:
            from consciousness.events import event_bus
            event_bus.emit("plugin:authority_changed", {
                "plugin_name": rec.name, "skill_id": getattr(rec, "skill_id", ""),
                "to_state": to_state, "actor": actor, "reason": reason,
            })
        except Exception:
            pass

    # ── Invocation ─────────────────────────────────────────────────────

    async def invoke(self, request: PluginRequest) -> PluginResponse:
        """Invoke a plugin with full safety wrapper.

        Routes to in-process handler or subprocess manager based on
        the plugin's execution_mode. No silent fallback between modes.
        """
        rec = self._records.get(request.plugin_name)
        if not rec or rec.state in ("quarantined", "disabled"):
            return PluginResponse(
                request_id=request.request_id,
                plugin_name=request.plugin_name,
                success=False,
                error=f"Plugin not available (state={rec.state if rec else 'unknown'})",
            )

        import asyncio
        t0 = time.monotonic()

        if rec.execution_mode == "isolated_subprocess":
            response = await self._invoke_subprocess(request, rec, t0)
        else:
            response = await self._invoke_in_process(request, rec, t0)

        # Audit (both paths)
        response.audit_entry = {
            "plugin_name": request.plugin_name,
            "request_id": request.request_id,
            "success": response.success,
            "duration_ms": round(response.duration_ms, 1),
            "error": response.error,
            "supervision_mode": rec.supervision_mode,
            "execution_mode": rec.execution_mode,
            "timestamp": time.time(),
        }
        self._audit_log(request.plugin_name, response.audit_entry)

        rec.updated_at = time.time()
        self._save_registry()
        return response

    async def _invoke_in_process(
        self, request: PluginRequest, rec: PluginRecord, t0: float
    ) -> PluginResponse:
        """In-process invocation via importlib-loaded handler."""
        handler = self._handlers.get(request.plugin_name)
        if not handler:
            return PluginResponse(
                request_id=request.request_id,
                plugin_name=request.plugin_name,
                success=False,
                error="Plugin handler not loaded",
            )

        import asyncio
        try:
            timeout = request.timeout_s or rec.supervision_mode == "bounded" and 10.0 or 30.0
            if asyncio.iscoroutinefunction(handler):
                result = await asyncio.wait_for(
                    handler(request.user_text, request.context),
                    timeout=timeout,
                )
            else:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, handler, request.user_text, request.context),
                    timeout=timeout,
                )

            duration_ms = (time.monotonic() - t0) * 1000
            rec.invocation_count += 1
            rec.success_count += 1
            rec.last_invocation_at = time.time()
            rec.avg_latency_ms = (rec.avg_latency_ms * (rec.invocation_count - 1) + duration_ms) / rec.invocation_count

            response = PluginResponse(
                request_id=request.request_id,
                plugin_name=request.plugin_name,
                success=True,
                result=result if isinstance(result, dict) else {"output": str(result)},
                duration_ms=duration_ms,
            )

            if rec.state == "shadow":
                response.result = None

            return response

        except asyncio.TimeoutError:
            duration_ms = (time.monotonic() - t0) * 1000
            rec.failure_count += 1
            self._record_failure(rec)
            return PluginResponse(
                request_id=request.request_id,
                plugin_name=request.plugin_name,
                success=False,
                error=f"Plugin timed out after {request.timeout_s}s",
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            rec.failure_count += 1
            self._record_failure(rec)
            return PluginResponse(
                request_id=request.request_id,
                plugin_name=request.plugin_name,
                success=False,
                error=str(exc),
                duration_ms=duration_ms,
            )

    async def _invoke_subprocess(
        self, request: PluginRequest, rec: PluginRecord, t0: float
    ) -> PluginResponse:
        """Subprocess invocation via PluginProcessManager. No silent fallback."""
        mgr = self._process_managers.get(request.plugin_name)
        if mgr is None:
            from tools.plugin_process import PluginProcessManager
            plugin_dir = self._plugins_dir / request.plugin_name
            if not plugin_dir.exists():
                return PluginResponse(
                    request_id=request.request_id,
                    plugin_name=request.plugin_name,
                    success=False,
                    error=f"Plugin directory not found: {plugin_dir}",
                )
            manifest_dict = self._extract_manifest_safe(
                (plugin_dir / "__init__.py").read_text()
            ) if (plugin_dir / "__init__.py").exists() else {}
            pinned = manifest_dict.get("pinned_dependencies", []) if manifest_dict else []
            verify_imports = manifest_dict.get("verify_imports", []) if manifest_dict else []
            mgr = PluginProcessManager(
                plugin_name=request.plugin_name,
                plugin_dir=plugin_dir,
                pinned_dependencies=pinned,
                verify_imports=verify_imports,
            )
            self._process_managers[request.plugin_name] = mgr

        req_dict = {
            "request_id": request.request_id,
            "user_text": request.user_text,
            "context": request.context,
            "timeout_s": request.timeout_s or 30.0,
        }
        resp_dict = await mgr.invoke(req_dict)

        if mgr.venv_ready and not rec.venv_ready:
            rec.venv_ready = True

        duration_ms = (time.monotonic() - t0) * 1000
        success = resp_dict.get("success", False)

        if success:
            rec.invocation_count += 1
            rec.success_count += 1
            rec.last_invocation_at = time.time()
            rec.avg_latency_ms = (rec.avg_latency_ms * (rec.invocation_count - 1) + duration_ms) / rec.invocation_count
        else:
            rec.failure_count += 1
            self._record_failure(rec)

        result = resp_dict.get("result")
        if rec.state == "shadow":
            result = None

        return PluginResponse(
            request_id=request.request_id,
            plugin_name=request.plugin_name,
            success=success,
            result=result,
            error=resp_dict.get("error"),
            duration_ms=duration_ms,
        )

    def _record_failure(self, rec: PluginRecord) -> None:
        """Track failure for circuit breaker."""
        now = time.time()
        rec.recent_failures.append(now)
        cutoff = now - CIRCUIT_BREAKER_WINDOW_S
        rec.recent_failures = [t for t in rec.recent_failures if t > cutoff]

        if len(rec.recent_failures) >= CIRCUIT_BREAKER_FAILURES:
            logger.warning("Circuit breaker triggered for plugin: %s", rec.name)
            skill_id = getattr(rec, "skill_id", "") or ""
            floor = self._known_good_floor(skill_id, exclude=rec.name) if skill_id else None
            if rec.state in ("active", "supervised") and floor is not None:
                # Asymmetric auto-demote: when there's a known-good floor to fall back to,
                # pull the misbehaving LIVE capability back to shadow and restore the floor,
                # autonomously — lowering authority is always safe, so no human is needed to
                # stop the bleeding and the skill keeps serving on the prior trusted version.
                rec.recent_failures = []
                self.demote(rec.name, reason="circuit_breaker", actor="auto:circuit_breaker")
            else:
                # No safe floor to fall back to (standalone / first version): disable the
                # broken plugin entirely rather than leave a known-bad one loaded.
                rec.state = "disabled"
                self._handlers.pop(rec.name, None)
                self._compiled_patterns.pop(rec.name, None)
                try:
                    from consciousness.events import event_bus
                    event_bus.emit("plugin:disabled", {
                        "plugin_name": rec.name,
                        "reason": "circuit_breaker",
                        "failures": len(rec.recent_failures),
                    })
                except Exception:
                    pass

    # ── Routing ────────────────────────────────────────────────────────

    def match(self, text: str) -> str | None:
        """Check if any active plugin matches the text. Returns plugin name or None."""
        for name, patterns in self._compiled_patterns.items():
            rec = self._records.get(name)
            if not rec or rec.state not in ("active", "supervised"):
                continue
            for pat in patterns:
                if pat.search(text):
                    return name
        return None

    def get_routes(self) -> list[dict[str, Any]]:
        """Return routing info for all active plugins."""
        routes = []
        for name, rec in self._records.items():
            if rec.state in ("active", "supervised", "shadow"):
                routes.append({
                    "name": name,
                    "state": rec.state,
                    "supervision_mode": rec.supervision_mode,
                    "patterns": [p.pattern for p in self._compiled_patterns.get(name, [])],
                })
        return routes

    # ── Query ──────────────────────────────────────────────────────────

    def get_record(self, plugin_name: str) -> PluginRecord | None:
        """Get a single plugin record by name."""
        return self._records.get(plugin_name)

    def discover(self) -> list[PluginManifest]:
        """List all plugin manifests from disk.

        Uses AST inspection + ast.literal_eval to extract PLUGIN_MANIFEST
        safely without executing plugin code.
        """
        manifests = []
        for entry in self._plugins_dir.iterdir():
            if entry.is_dir() and (entry / "__init__.py").exists():
                try:
                    content = (entry / "__init__.py").read_text()
                    manifest_dict = self._extract_manifest_safe(content)
                    if manifest_dict:
                        manifests.append(PluginManifest.from_dict(manifest_dict))
                except Exception:
                    pass
        return manifests

    @staticmethod
    def _extract_manifest_safe(source: str) -> dict[str, Any] | None:
        """Extract PLUGIN_MANIFEST from source using AST, without executing code."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "PLUGIN_MANIFEST":
                        try:
                            return ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            return None
        return None

    def get_handler(self, plugin_name: str) -> Callable | None:
        return self._handlers.get(plugin_name)

    def get_status(self) -> dict[str, Any]:
        """Dashboard status."""
        by_state: dict[str, int] = {}
        for rec in self._records.values():
            by_state[rec.state] = by_state.get(rec.state, 0) + 1

        plugins_list = []
        for rec in self._records.values():
            entry = {
                "name": rec.name,
                "state": rec.state,
                "version": rec.version,
                "risk_tier": rec.risk_tier,
                "invocation_count": rec.invocation_count,
                "success_count": rec.success_count,
                "success_rate": round(rec.success_count / max(rec.invocation_count, 1), 3),
                "avg_latency_ms": round(rec.avg_latency_ms, 1),
                "supervision_mode": rec.supervision_mode,
                "acquisition_id": rec.acquisition_id,
                # capability authority (version grouping + floor)
                "skill_id": getattr(rec, "skill_id", ""),
                "generation": getattr(rec, "generation", 0),
                "prior_authoritative": getattr(rec, "prior_authoritative", ""),
                "last_authoritative_at": getattr(rec, "last_authoritative_at", 0.0),
                "execution_mode": rec.execution_mode,
                "venv_ready": rec.venv_ready,
            }
            mgr = self._process_managers.get(rec.name)
            if mgr is not None:
                entry["subprocess"] = mgr.get_status()
            plugins_list.append(entry)

        subprocess_count = sum(
            1 for r in self._records.values() if r.execution_mode == "isolated_subprocess"
        )
        subprocess_running = sum(
            1 for m in self._process_managers.values() if m.is_running
        )

        return {
            "total_plugins": len(self._records),
            "by_state": by_state,
            "subprocess_count": subprocess_count,
            "subprocess_running": subprocess_running,
            "plugins": plugins_list,
            "routes": self.get_routes(),
        }

    # ── Audit Trail ────────────────────────────────────────────────────

    def _audit_log(self, plugin_name: str, entry: dict[str, Any]) -> None:
        """Append to per-plugin audit JSONL."""
        try:
            audit_file = _AUDIT_DIR / f"{plugin_name}.jsonl"
            entry["_ts"] = time.time()
            with open(audit_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

            # Rotation: 10MB max
            if audit_file.stat().st_size > 10 * 1024 * 1024:
                rotated = audit_file.with_suffix(".jsonl.1")
                if rotated.exists():
                    rotated.unlink()
                audit_file.rename(rotated)
        except Exception:
            pass

    def log_invocation(self, plugin_name: str, input_data: Any, output_data: Any,
                       duration: float, error: str | None = None) -> None:
        """External audit logging API."""
        self._audit_log(plugin_name, {
            "action": "invocation",
            "input_summary": str(input_data)[:200],
            "output_summary": str(output_data)[:200] if output_data else None,
            "duration_ms": round(duration, 1),
            "error": error,
        })


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry_singleton: PluginRegistry | None = None


def get_plugin_registry() -> PluginRegistry:
    """Return the shared PluginRegistry singleton."""
    global _registry_singleton
    if _registry_singleton is None:
        _registry_singleton = PluginRegistry()
    return _registry_singleton
