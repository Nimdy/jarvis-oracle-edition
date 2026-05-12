"""Phase E Language Kernel Artifact Identity (P1.5 scaffolding).

Adds an addressable-artifact wrapper over the existing Phase C
checkpoint at
:mod:`reasoning.language_phasec` (``CHECKPOINT_PATH``). This is NOT a
new student model and does NOT activate anything. It only lets the
system:

  1. Compute a stable identity for the current Phase C checkpoint
     (content-hash + monotonic version + promoted-at timestamp).
  2. Record each registered identity in a durable registry so the
     existing :class:`LanguagePromotionGovernor` can reference which
     artifact id is live.
  3. Roll back to a prior registered artifact id by restoring its
     snapshot on disk — distinct from the existing per-class
     gate-regression rollback at ``language_promotion.py`` which only
     flips ``shadow|canary|live`` levels.

Explicit non-goals:

  - No auto-promotion. ``register_current_checkpoint`` is invoked
    explicitly by the Phase D governor or an operator-facing API. The
    constructor alone never mutates live state.
  - No bypass of the Phase D guard. Artifact rollback restores the
    checkpoint payload on disk but does NOT change
    ``LanguagePromotionGovernor`` levels. The governor continues to
    enforce its own shadow/canary/live gating.
  - No new student model. This wraps the existing checkpoint JSON.

Status: **PRE-MATURE**. On a fresh brain there is no Phase C checkpoint
yet, so ``get_live_artifact()`` will return ``None`` and the dashboard
tab will display that explicitly.

Persistence:
  - Registry: ``~/.jarvis/language_kernel/registry.json``
  - Snapshots: ``~/.jarvis/language_kernel/snapshots/<artifact_id>.json``

Each snapshot is an exact byte-copy of the Phase C checkpoint at
registration time. We keep the snapshot (not a diff) so rollback is
simple and auditable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Defaults + paths
# --------------------------------------------------------------------------

_DEFAULT_JARVIS_DIR = Path.home() / ".jarvis"
_DEFAULT_KERNEL_DIR = _DEFAULT_JARVIS_DIR / "language_kernel"
_DEFAULT_REGISTRY_PATH = _DEFAULT_KERNEL_DIR / "registry.json"
_DEFAULT_SNAPSHOT_DIR = _DEFAULT_KERNEL_DIR / "snapshots"

# Default source checkpoint (Phase C). Lazily imported to avoid a
# reasoning <-> language import cycle at package load time.
_DEFAULT_CHECKPOINT_PATH_HINT = (
    _DEFAULT_JARVIS_DIR / "language_corpus" / "phase_c" / "student_checkpoint.json"
)


MAX_HISTORY_ENTRIES = 50


# --------------------------------------------------------------------------
# Artifact record
# --------------------------------------------------------------------------


@dataclass
class LanguageKernelArtifact:
    """Immutable record of a registered Phase C checkpoint identity.

    ``artifact_id`` is ``phasec-v{version}-{hash[:12]}``. ``hash`` is
    the sha256 of the checkpoint JSON bytes at registration time.
    ``snapshot_path`` is the absolute path to the stored byte-copy
    (string, for JSON-safety).
    """

    artifact_id: str
    version: int
    hash: str
    promoted_at: float
    source_path: str
    snapshot_path: str
    checkpoint_schema_version: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LanguageKernelArtifact":
        return cls(
            artifact_id=str(d["artifact_id"]),
            version=int(d["version"]),
            hash=str(d["hash"]),
            promoted_at=float(d["promoted_at"]),
            source_path=str(d["source_path"]),
            snapshot_path=str(d["snapshot_path"]),
            checkpoint_schema_version=(
                int(d["checkpoint_schema_version"])
                if d.get("checkpoint_schema_version") is not None
                else None
            ),
            notes=str(d.get("notes", "")),
        )


# --------------------------------------------------------------------------
# Persisted registry state
# --------------------------------------------------------------------------


@dataclass
class _RegistryState:
    artifacts: list[LanguageKernelArtifact] = field(default_factory=list)
    live_artifact_id: str | None = None
    last_rollback_ts: float = 0.0
    rollback_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": [a.to_dict() for a in self.artifacts],
            "live_artifact_id": self.live_artifact_id,
            "last_rollback_ts": self.last_rollback_ts,
            "rollback_history": list(self.rollback_history),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "_RegistryState":
        return cls(
            artifacts=[
                LanguageKernelArtifact.from_dict(a)
                for a in (d.get("artifacts") or [])
            ],
            live_artifact_id=d.get("live_artifact_id"),
            last_rollback_ts=float(d.get("last_rollback_ts", 0.0) or 0.0),
            rollback_history=list(d.get("rollback_history", []) or []),
        )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


class LanguageKernelRegistry:
    """Durable registry for Phase C checkpoint artifact identities.

    Thread-safe. All disk writes use ``tmp + rename`` for atomicity.
    """

    def __init__(
        self,
        registry_path: Path | str = _DEFAULT_REGISTRY_PATH,
        snapshot_dir: Path | str = _DEFAULT_SNAPSHOT_DIR,
        checkpoint_path: Path | str | None = None,
    ) -> None:
        self._registry_path = Path(registry_path)
        self._snapshot_dir = Path(snapshot_dir)
        self._checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        self._lock = threading.Lock()
        self._state = self._load()

    # ----------------------------------------------------------------------
    # Resolution helpers
    # ----------------------------------------------------------------------

    def _resolve_checkpoint_path(self) -> Path:
        if self._checkpoint_path is not None:
            return self._checkpoint_path
        try:
            from reasoning.language_phasec import CHECKPOINT_PATH

            return Path(CHECKPOINT_PATH)
        except Exception:
            logger.debug(
                "language_kernel: could not import CHECKPOINT_PATH, "
                "falling back to default hint",
                exc_info=True,
            )
            return _DEFAULT_CHECKPOINT_PATH_HINT

    # ----------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------

    def _load(self) -> _RegistryState:
        if not self._registry_path.exists():
            return _RegistryState()
        try:
            return _RegistryState.from_dict(
                json.loads(self._registry_path.read_text(encoding="utf-8"))
            )
        except Exception:
            logger.warning(
                "language_kernel: corrupt registry at %s; resetting",
                self._registry_path,
                exc_info=True,
            )
            return _RegistryState()

    def _save(self) -> None:
        try:
            self._registry_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._registry_path.with_suffix(
                self._registry_path.suffix + ".tmp"
            )
            tmp.write_text(
                json.dumps(self._state.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(tmp, self._registry_path)
        except Exception:
            logger.exception("language_kernel: failed to persist registry")

    # ----------------------------------------------------------------------
    # Registration / rollback
    # ----------------------------------------------------------------------

    @staticmethod
    def _hash_bytes(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _build_artifact_id(version: int, digest: str) -> str:
        return f"phasec-v{version}-{digest[:12]}"

    def register_current_checkpoint(
        self, notes: str = ""
    ) -> LanguageKernelArtifact | None:
        """Register the current Phase C checkpoint as a new artifact.

        Returns the new artifact on success, or ``None`` if no
        checkpoint exists on disk (a fresh brain returns ``None`` —
        explicitly PRE-MATURE, not a bug).

        If the current checkpoint hash matches the live artifact, this
        is a no-op and returns the existing live artifact.
        """
        with self._lock:
            src = self._resolve_checkpoint_path()
            if not src.exists():
                logger.info(
                    "language_kernel: no Phase C checkpoint at %s "
                    "(PRE-MATURE; nothing to register)",
                    src,
                )
                return None
            try:
                payload = src.read_bytes()
            except Exception:
                logger.exception("language_kernel: failed to read %s", src)
                return None

            digest = self._hash_bytes(payload)

            # No-op if current checkpoint already matches the live artifact.
            live = self._get_live_locked()
            if live is not None and live.hash == digest:
                return live

            # Parse schema version out of the payload if present (non-fatal
            # if absent or malformed).
            schema_version: int | None = None
            try:
                parsed = json.loads(payload)
                sv = parsed.get("schema_version") if isinstance(parsed, dict) else None
                if sv is not None:
                    schema_version = int(sv)
            except Exception:
                pass

            next_version = len(self._state.artifacts) + 1
            artifact_id = self._build_artifact_id(next_version, digest)

            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = self._snapshot_dir / f"{artifact_id}.json"
            try:
                snapshot_tmp = snapshot_path.with_suffix(".json.tmp")
                snapshot_tmp.write_bytes(payload)
                os.replace(snapshot_tmp, snapshot_path)
            except Exception:
                logger.exception(
                    "language_kernel: failed to write snapshot %s",
                    snapshot_path,
                )
                return None

            artifact = LanguageKernelArtifact(
                artifact_id=artifact_id,
                version=next_version,
                hash=digest,
                promoted_at=time.time(),
                source_path=str(src),
                snapshot_path=str(snapshot_path),
                checkpoint_schema_version=schema_version,
                notes=notes,
            )
            self._state.artifacts.append(artifact)
            self._state.live_artifact_id = artifact_id
            self._save()
            logger.info(
                "language_kernel: registered %s (hash=%s, schema=%s)",
                artifact_id, digest[:12], schema_version,
            )
            return artifact

    def rollback_to(
        self, artifact_id: str, *, reason: str = "manual"
    ) -> LanguageKernelArtifact | None:
        """Revert the Phase C checkpoint on disk to a prior artifact.

        This is distinct from the per-class gate-regression rollback in
        :class:`LanguagePromotionGovernor`. Artifact rollback restores
        the checkpoint bytes; it does NOT flip gate levels. The Phase D
        governor continues to enforce its own gating independently.

        Returns the restored artifact, or ``None`` if the id is unknown
        or the snapshot file is missing.
        """
        with self._lock:
            target = next(
                (a for a in self._state.artifacts if a.artifact_id == artifact_id),
                None,
            )
            if target is None:
                logger.warning(
                    "language_kernel: rollback target %s not in registry",
                    artifact_id,
                )
                return None
            snapshot = Path(target.snapshot_path)
            if not snapshot.exists():
                logger.error(
                    "language_kernel: snapshot %s missing; cannot roll back",
                    snapshot,
                )
                return None

            src = self._resolve_checkpoint_path()
            try:
                src.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snapshot, src)
            except Exception:
                logger.exception(
                    "language_kernel: failed to restore %s -> %s",
                    snapshot, src,
                )
                return None

            prev_live = self._state.live_artifact_id
            now = time.time()
            self._state.live_artifact_id = artifact_id
            self._state.last_rollback_ts = now
            self._state.rollback_history.append({
                "from_artifact_id": prev_live,
                "to_artifact_id": artifact_id,
                "ts": now,
                "reason": reason,
            })
            if len(self._state.rollback_history) > MAX_HISTORY_ENTRIES:
                self._state.rollback_history = (
                    self._state.rollback_history[-MAX_HISTORY_ENTRIES:]
                )
            self._save()
            logger.warning(
                "language_kernel: rolled back %s -> %s (reason=%s)",
                prev_live, artifact_id, reason,
            )
            return target

    # ----------------------------------------------------------------------
    # Reads
    # ----------------------------------------------------------------------

    def _get_live_locked(self) -> LanguageKernelArtifact | None:
        if self._state.live_artifact_id is None:
            return None
        return next(
            (
                a for a in self._state.artifacts
                if a.artifact_id == self._state.live_artifact_id
            ),
            None,
        )

    def get_live_artifact(self) -> LanguageKernelArtifact | None:
        with self._lock:
            return self._get_live_locked()

    def list_artifacts(self) -> list[LanguageKernelArtifact]:
        with self._lock:
            return list(self._state.artifacts)

    def get_state(self) -> dict[str, Any]:
        """Read-only dashboard-facing view."""
        with self._lock:
            live = self._get_live_locked()
            checkpoint = self._resolve_checkpoint_path()
            checkpoint_exists = checkpoint.exists()

            # Compute live-hash of the current on-disk checkpoint for the
            # "on-disk-matches-live-artifact" invariant.
            on_disk_hash: str | None = None
            if checkpoint_exists:
                try:
                    on_disk_hash = self._hash_bytes(checkpoint.read_bytes())
                except Exception:
                    logger.debug(
                        "language_kernel: hash compute failed",
                        exc_info=True,
                    )
                    on_disk_hash = None

            matches_live = (
                live is not None
                and on_disk_hash is not None
                and on_disk_hash == live.hash
            )

            return {
                "status": "pre_mature" if live is None else "registered",
                "live_artifact": live.to_dict() if live else None,
                "total_artifacts": len(self._state.artifacts),
                "artifacts": [a.to_dict() for a in self._state.artifacts[-10:]],
                "checkpoint_path": str(checkpoint),
                "checkpoint_exists": checkpoint_exists,
                "on_disk_hash": on_disk_hash,
                "matches_live_artifact": matches_live,
                "last_rollback_ts": (
                    self._state.last_rollback_ts or None
                ),
                "rollback_history": list(self._state.rollback_history[-10:]),
                "note": (
                    "Phase E kernel identity is PRE-MATURE on a fresh brain. "
                    "It wraps the existing Phase C checkpoint and never bypasses "
                    "the Phase D promotion governor."
                ),
            }


# --------------------------------------------------------------------------
# Module-level singleton
# --------------------------------------------------------------------------


_registry: LanguageKernelRegistry | None = None
_registry_lock = threading.Lock()


def get_language_kernel_registry() -> LanguageKernelRegistry:
    """Return the process-wide :class:`LanguageKernelRegistry` singleton."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = LanguageKernelRegistry()
        return _registry


def set_language_kernel_registry(registry: LanguageKernelRegistry | None) -> None:
    """Override the singleton (used by tests)."""
    global _registry
    with _registry_lock:
        _registry = registry
