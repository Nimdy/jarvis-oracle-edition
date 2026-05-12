"""Manual ingestion — user-curated sources enter the library via this module.

Supports three modes:
  1. Paste (raw text / note)
  2. URL fetch (HTML → plain text, with SSRF guards)
  3. File upload (text/PDF content passed as string)

All ingested sources go through the same Source → Chunk → Embed pipeline as
autonomous acquisition, but are tagged ``ingested_by="user"`` and
``trust_tier="curated"`` with a conservative starting ``quality_score=0.35``.
"""

from __future__ import annotations

import hashlib
import html as html_mod
import ipaddress
import logging
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from library.source import Source, SourceStore, make_source_id, source_store
from library.chunks import Chunk, ChunkStore, chunk_store, chunk_text
from library.index import LibraryIndex, library_index

logger = logging.getLogger(__name__)

MAX_CONTENT_BYTES = 500_000  # ~500 KB text cap
MAX_CHUNKS_PER_SOURCE = 200
URL_CONNECT_TIMEOUT = 5
URL_READ_TIMEOUT = 15

USER_QUALITY_SCORE = 0.35


@dataclass
class IngestResult:
    success: bool
    source_id: str = ""
    chunk_count: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "source_id": self.source_id,
            "chunk_count": self.chunk_count,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------

_BLOCKED_HOSTS = {"localhost", "metadata.google.internal", "169.254.169.254"}


def _is_private_ip(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
        return (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
        )
    except ValueError:
        return True


def _validate_url(url: str) -> str | None:
    """Return an error string if the URL is unsafe, else None."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "Only http/https URLs are allowed"
    host = parsed.hostname or ""
    if not host:
        return "No hostname found in URL"
    lower_host = host.lower()
    if lower_host in _BLOCKED_HOSTS:
        return f"Blocked hostname: {host}"
    if lower_host.endswith((".local", ".internal")):
        return f"Blocked domain suffix: {host}"
    try:
        resolved = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _family, _type, _proto, _canon, sockaddr in resolved:
            ip_str = sockaddr[0]
            if _is_private_ip(ip_str):
                return f"Hostname {host} resolves to private IP {ip_str}"
    except socket.gaierror:
        return f"DNS resolution failed for {host}"
    return None


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s{3,}")
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE
)


def _strip_html(raw_html: str) -> str:
    text = _SCRIPT_STYLE_RE.sub(" ", raw_html)
    text = _TAG_RE.sub(" ", text)
    text = html_mod.unescape(text)
    text = _WHITESPACE_RE.sub("\n\n", text)
    return text.strip()


def _extract_pdf_text(raw_bytes: bytes, max_chars: int = 5000) -> tuple[str, str]:
    """Extract text from raw PDF bytes via pdftotext. Returns (text, error)."""
    try:
        proc = subprocess.run(
            ["pdftotext", "-", "-"],
            input=raw_bytes, capture_output=True, timeout=30,
        )
        if proc.returncode == 0 and proc.stdout:
            text = proc.stdout.decode("utf-8", errors="replace").strip()
            if len(text) > 200:
                return text[:max_chars], ""
            return "", "pdf_extraction_too_short"
        return "", f"pdftotext_exit_{proc.returncode}"
    except FileNotFoundError:
        return "", "pdftotext_not_installed"
    except subprocess.TimeoutExpired:
        return "", "pdftotext_timeout"
    except Exception as exc:
        return "", f"pdftotext_error:{exc}"


# ---------------------------------------------------------------------------
# URL fetching (Content-Type aware)
# ---------------------------------------------------------------------------

_TEXTUAL_CONTENT_TYPES = ("text/plain", "text/xml", "application/json", "application/xml")


def _fetch_url(url: str) -> tuple[str, str]:
    """Fetch URL content with Content-Type routing.

    Routes PDF responses through pdftotext, strips HTML, rejects binary.
    Returns (text, error_msg). Error empty on success.
    """
    import urllib.request
    import urllib.error

    err = _validate_url(url)
    if err:
        return "", err

    req = urllib.request.Request(url, headers={
        "User-Agent": "Jarvis-LibraryBot/1.0 (research ingest)",
    })

    try:
        with urllib.request.urlopen(
            req, timeout=URL_READ_TIMEOUT
        ) as resp:
            content_type = resp.headers.get("Content-Type", "").lower()
            raw = resp.read(MAX_CONTENT_BYTES + 1)
            if len(raw) > MAX_CONTENT_BYTES:
                raw = raw[:MAX_CONTENT_BYTES]

            if "pdf" in content_type or raw[:5] == b"%PDF-":
                text, pdf_err = _extract_pdf_text(raw)
                if pdf_err:
                    return "", f"pdf_extraction_failed:{pdf_err}"
                return text, ""

            if "html" in content_type:
                text = _strip_html(raw.decode("utf-8", errors="replace"))
                return text, ""

            if any(t in content_type for t in _TEXTUAL_CONTENT_TYPES):
                return raw.decode("utf-8", errors="replace"), ""

            return "", f"unsupported_content_type:{content_type}"
    except urllib.error.HTTPError as exc:
        return "", f"HTTP {exc.code}: {exc.reason}"
    except urllib.error.URLError as exc:
        return "", f"URL error: {exc.reason}"
    except Exception as exc:
        return "", f"Fetch failed: {exc}"


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_manual_source(
    content: str = "",
    url: str = "",
    title: str = "",
    source_type: str = "user_note",
    domain_tags: str = "",
    study_now: bool = False,
    ingested_by: str = "user",
    quality_score: float = 0.0,
    content_depth: str = "",
    trust_tier: str = "",
) -> IngestResult:
    """Ingest a user-provided source into the library.

    Either ``content`` (for paste/file) or ``url`` (for fetch) must be given.
    Returns an IngestResult with the new source_id and chunk count.
    """
    # Resolve content
    canonical_domain = ""
    if url and not content:
        source_type = "url"
        content, err = _fetch_url(url)
        if err:
            return IngestResult(success=False, error=err)
        canonical_domain = _extract_domain(url)

    if not content or not content.strip():
        return IngestResult(success=False, error="No content provided")

    if len(content.encode("utf-8", errors="replace")) > MAX_CONTENT_BYTES:
        content = content[:MAX_CONTENT_BYTES // 2]

    if not title:
        title = content[:120].split("\n")[0].strip() or "Untitled"

    source_id = make_source_id(url=url, title=title)

    if source_store.exists(source_id):
        return IngestResult(success=False, source_id=source_id,
                            error="Source already exists in library")

    source = Source(
        source_id=source_id,
        source_type=source_type,
        retrieved_at=time.time(),
        url=url,
        title=title,
        content_text=content[:5000],
        license_flags="user_provided",
        quality_score=quality_score if quality_score > 0 else USER_QUALITY_SCORE,
        provider="manual",
        ingested_by=ingested_by,
        trust_tier=trust_tier or "curated",
        domain_tags=domain_tags,
        canonical_domain=canonical_domain,
        content_depth=content_depth,
    )

    if not source_store.add(source):
        return IngestResult(success=False, error="Failed to write source to DB")

    # Chunk
    chunks = chunk_text(content, source_id)
    if len(chunks) > MAX_CHUNKS_PER_SOURCE:
        chunks = chunks[:MAX_CHUNKS_PER_SOURCE]
    added = chunk_store.add_many(chunks)

    # Embed
    indexed = 0
    if library_index.available:
        for c in chunks:
            if library_index.add_chunk(c.chunk_id, c.source_id, c.text):
                indexed += 1

    logger.info(
        "Manual ingest complete: source=%s chunks=%d indexed=%d tags=%s",
        source_id, added, indexed, domain_tags,
    )

    if study_now:
        _trigger_study(source_id)

    return IngestResult(
        success=True,
        source_id=source_id,
        chunk_count=added,
    )


def ingest_codebase_source(
    file_path: str,
    content: str,
    title: str,
    domain_tags: str = "codebase,self_knowledge",
    content_hash: str = "",
) -> IngestResult:
    """Ingest a codebase source file into the library.

    Uses a deterministic source_id from the repo-relative file path so
    re-ingestion is idempotent:
    - Same hash as stored: skip (no work)
    - Different hash: replace old chunks/embeddings with new content
    """
    if not content or not content.strip():
        return IngestResult(success=False, error="No content provided")

    if not content_hash:
        content_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:32]

    normalized_path = file_path.replace("\\", "/").strip("/")
    source_id = f"src_{hashlib.sha256(normalized_path.encode()).hexdigest()[:16]}"

    hash_tag = f"content_hash:{content_hash}"

    existing = source_store.get(source_id)
    if existing is not None:
        existing_hash = ""
        for flag in (existing.license_flags or "").split(","):
            if flag.strip().startswith("content_hash:"):
                existing_hash = flag.strip().split(":", 1)[1]
                break
        if existing_hash == content_hash:
            return IngestResult(success=True, source_id=source_id,
                                chunk_count=0, error="unchanged")
        library_index.remove_source(source_id)
        _delete_chunks_for_source(source_id)

    if not title:
        title = normalized_path.rsplit("/", 1)[-1]

    source = Source(
        source_id=source_id,
        source_type="codebase",
        retrieved_at=time.time(),
        title=title,
        content_text=content[:10000],
        license_flags=hash_tag,
        quality_score=0.95,
        provider="codebase",
        ingested_by="gestation",
        trust_tier="verified",
        domain_tags=domain_tags,
        content_depth="full_text",
    )

    if not source_store.add(source):
        return IngestResult(success=False, error="Failed to write source to DB")

    chunks = chunk_text(content, source_id, chunk_type="source_code")
    if len(chunks) > MAX_CHUNKS_PER_SOURCE:
        chunks = chunks[:MAX_CHUNKS_PER_SOURCE]
    added = chunk_store.add_many(chunks)

    indexed = 0
    if library_index.available:
        for c in chunks:
            if library_index.add_chunk(c.chunk_id, c.source_id, c.text):
                indexed += 1

    logger.info(
        "Codebase ingest: path=%s source=%s chunks=%d indexed=%d hash=%s",
        normalized_path, source_id, added, indexed, content_hash[:12],
    )

    return IngestResult(success=True, source_id=source_id, chunk_count=added)


def _delete_chunks_for_source(source_id: str) -> int:
    """Delete all chunks belonging to a source from the chunk store."""
    try:
        conn = chunk_store._ensure_init()
        from library.db import LIBRARY_WRITE_LOCK
        with LIBRARY_WRITE_LOCK:
            cursor = conn.execute(
                "DELETE FROM chunks WHERE source_id = ?", (source_id,)
            )
            conn.commit()
            return cursor.rowcount
    except Exception:
        logger.exception("Failed to delete chunks for %s", source_id)
        return 0


def _trigger_study(source_id: str) -> None:
    """Kick off study in a background thread."""
    import threading

    def _run() -> None:
        try:
            from library.study import study_source
            study_source(source_id)
        except Exception:
            logger.exception("Background study failed for %s", source_id)

    t = threading.Thread(target=_run, name=f"study-{source_id[:12]}", daemon=True)
    t.start()
