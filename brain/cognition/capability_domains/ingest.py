"""Matrix v2 Phase 2 — ingest knowledge into a domain's ISOLATED store.

Takes text / .md / .txt / .pdf into the domain's own knowledge store only. Never
writes to core memory or the shared library — that is the anti-pollution invariant
(net §4.5). Provenance is tagged ``ingested`` (asymmetric gate: ingested knowledge
is "know about", never "can do"). The domain's tallies + status update via the
registry.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess

from cognition.capability_domains.store import DomainKnowledgeStore

logger = logging.getLogger(__name__)

_PARA_RE = re.compile(r"\n\s*\n")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
SUPPORTED_EXT = (".txt", ".md", ".markdown", ".text", ".rst", ".pdf")


def chunk_text(content: str, max_chars: int = 600) -> list[str]:
    """Deterministic chunker: paragraph-first, then pack sentences up to max_chars."""
    chunks: list[str] = []
    for para in _PARA_RE.split(content or ""):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        buf = ""
        for sent in _SENT_RE.split(para):
            sent = sent.strip()
            if not sent:
                continue
            if len(buf) + len(sent) + 1 > max_chars and buf:
                chunks.append(buf.strip())
                buf = sent
            else:
                buf = (buf + " " + sent).strip()
        if buf.strip():
            chunks.append(buf.strip())
    return chunks


def _read_pdf(path: str, max_chars: int = 200_000) -> str:
    try:
        proc = subprocess.run(["pdftotext", path, "-"], capture_output=True, timeout=30)
        if proc.returncode == 0:
            return proc.stdout.decode("utf-8", errors="ignore")[:max_chars]
    except Exception:
        logger.debug("pdftotext failed for %s", path, exc_info=True)
    return ""


def ingest_text(registry, domain, title: str, content: str,
                provenance: str = "ingested") -> int:
    """Chunk + store *content* into the domain's isolated store. Returns chunk count."""
    chunks = chunk_text(content)
    if not chunks:
        return 0
    store = DomainKnowledgeStore(domain.knowledge_db)
    try:
        source_id = re.sub(r"[^a-z0-9]+", "_", (title or "source").lower())[:48]
        n = store.add_chunks(source_id, title or source_id, chunks, provenance=provenance)
        domain.source_count = store.source_count()
        domain.chunk_count = store.count()
        domain.provenance[provenance] = domain.provenance.get(provenance, 0) + n
        if domain.status == "created":
            domain.status = "ingesting"
        registry.update(domain)
        return n
    finally:
        store.close()


def ingest_file(registry, domain, path: str) -> int:
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXT:
        return 0
    if ext == ".pdf":
        content = _read_pdf(path)
    else:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            logger.debug("read failed for %s", path, exc_info=True)
            return 0
    if not content.strip():
        return 0
    return ingest_text(registry, domain, os.path.basename(path), content)


def ingest_folder(registry, domain, folder: str) -> dict:
    """Ingest every supported file under *folder* into the domain. Returns a summary."""
    files_ingested, chunks = 0, 0
    for root, _dirs, names in os.walk(folder):
        for name in sorted(names):
            if os.path.splitext(name)[1].lower() in SUPPORTED_EXT:
                n = ingest_file(registry, domain, os.path.join(root, name))
                if n > 0:
                    files_ingested += 1
                    chunks += n
    return {"files_ingested": files_ingested, "chunks": chunks,
            "domain_id": domain.domain_id}
