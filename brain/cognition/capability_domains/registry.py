"""Matrix v2 — Capability Domain registry (Phase 1: isolation substrate).

Owns the lifecycle of isolated, deletable Capability Domains. Each domain gets its
own directory under the registry root; ALL of a domain's data lives there and
nowhere else. ``delete()`` removes that directory — a clean ablation with **zero
residue** in core memory or sibling domains (the brain-injury analogy, made an
automated guarantee by the tests). Read-only/observational; grants no authority.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
import uuid
from pathlib import Path

from cognition.capability_domains.domain import CapabilityDomain

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = Path.home() / ".jarvis" / "domains"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(name: str) -> str:
    s = _SLUG_RE.sub("_", (name or "").strip().lower()).strip("_")
    return (s or "domain")[:32]


class CapabilityDomainRegistry:
    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root) if root else _DEFAULT_ROOT
        self._root.mkdir(parents=True, exist_ok=True)
        self._index = self._root / "registry.json"
        self._domains: dict[str, CapabilityDomain] = {}
        self._load()

    # ── persistence ──
    def _load(self) -> None:
        if not self._index.exists():
            return
        try:
            data = json.loads(self._index.read_text())
            for rec in data.get("domains", []):
                dom = CapabilityDomain.from_dict(rec)
                self._domains[dom.domain_id] = dom
        except Exception:
            logger.exception("CapabilityDomainRegistry load failed")

    def _persist(self) -> None:
        try:
            tmp = self._index.with_suffix(".tmp")
            tmp.write_text(json.dumps(
                {"domains": [d.to_dict() for d in self._domains.values()]}, indent=2,
            ))
            tmp.replace(self._index)
        except Exception:
            logger.exception("CapabilityDomainRegistry persist failed")

    # ── lifecycle ──
    def create(self, name: str, kind: str = "document") -> CapabilityDomain:
        """Register a new domain with its own isolated directory."""
        did = f"dom_{_slug(name)}_{uuid.uuid4().hex[:6]}"
        root_dir = self._root / did
        root_dir.mkdir(parents=True, exist_ok=True)
        dom = CapabilityDomain(
            domain_id=did, name=name, kind=kind,
            root_dir=str(root_dir),
            knowledge_db=str(root_dir / "knowledge.db"),
            memory_path=str(root_dir / "memory.json"),
        )
        self._domains[did] = dom
        self._persist()
        logger.info("CapabilityDomain created: %s (%s, kind=%s)", did, name, kind)
        return dom

    def get(self, domain_id: str) -> CapabilityDomain | None:
        return self._domains.get(domain_id)

    def list(self) -> list[CapabilityDomain]:
        return list(self._domains.values())

    def update(self, dom: CapabilityDomain) -> None:
        dom.updated_at = time.time()
        self._domains[dom.domain_id] = dom
        self._persist()

    def _under_root(self, p: Path) -> bool:
        """Safety: only ever delete paths inside our own registry root."""
        try:
            root = self._root.resolve()
            target = p.resolve()
            return target != root and root in target.parents
        except Exception:
            return False

    def delete(self, domain_id: str) -> bool:
        """Clean ablation: drop the domain's isolated dir + registry entry.

        Zero residue elsewhere. Refuses to remove anything outside the registry
        root (defensive). Returns True if a domain was deleted.
        """
        dom = self._domains.pop(domain_id, None)
        if dom is None:
            return False
        rd = Path(dom.root_dir)
        if rd.exists() and self._under_root(rd):
            shutil.rmtree(rd, ignore_errors=True)
        elif rd.exists():
            logger.warning("Refusing to delete out-of-root domain dir: %s", rd)
        self._persist()
        logger.info("CapabilityDomain deleted (clean ablation): %s", domain_id)
        return True

    def status(self) -> dict:
        """Observability summary for /api/domains."""
        doms = self.list()
        by_status: dict[str, int] = {}
        for d in doms:
            by_status[d.status] = by_status.get(d.status, 0) + 1
        return {
            "count": len(doms),
            "by_status": by_status,
            "domains": [d.public_view() for d in doms],
        }


_registry: CapabilityDomainRegistry | None = None


def get_capability_domain_registry() -> CapabilityDomainRegistry:
    global _registry
    if _registry is None:
        _registry = CapabilityDomainRegistry()
    return _registry
