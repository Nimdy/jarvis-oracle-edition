"""Batch textbook ingestion — discover chapters from a TOC page and ingest each.

Supports any multi-page textbook or documentation site. Auto-detects site type
(Sphinx/MathJax, pdf2htmlEX, generic) and applies the appropriate content
sanitizer before ingestion. Each page becomes a separate library Source so
chunking, study, and Blue Diamond graduation work per-page.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

from library.content_sanitizer import (
    SanitizedContent,
    detect_site_type,
    sanitize,
)

logger = logging.getLogger(__name__)

FETCH_DELAY_S = 2.0
FETCH_TIMEOUT_S = 20
MAX_CONTENT_BYTES = 500_000
MIN_QUALITY_SCORE = 0.5
TEXTBOOK_QUALITY_SCORE = 0.70
USER_AGENT = "Jarvis-LibraryBot/1.0 (textbook ingest)"


@dataclass
class ChapterInfo:
    url: str
    title: str
    order: int = 0


@dataclass
class ChapterResult:
    url: str
    title: str
    success: bool
    skipped: bool = False
    skip_reason: str = ""
    quality_score: float = 0.0
    math_preserved: int = 0
    code_preserved: int = 0
    chunk_count: int = 0
    source_id: str = ""
    error: str = ""


@dataclass
class BatchIngestResult:
    success: bool
    title: str
    toc_url: str
    site_type: str = ""
    chapters_discovered: int = 0
    chapters_ingested: int = 0
    chapters_skipped: int = 0
    chapters_failed: int = 0
    total_math: int = 0
    total_code: int = 0
    total_chunks: int = 0
    results: list[ChapterResult] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "title": self.title,
            "toc_url": self.toc_url,
            "site_type": self.site_type,
            "chapters_discovered": self.chapters_discovered,
            "chapters_ingested": self.chapters_ingested,
            "chapters_skipped": self.chapters_skipped,
            "chapters_failed": self.chapters_failed,
            "total_math": self.total_math,
            "total_code": self.total_code,
            "total_chunks": self.total_chunks,
            "results": [
                {
                    "url": r.url,
                    "title": r.title,
                    "success": r.success,
                    "skipped": r.skipped,
                    "skip_reason": r.skip_reason,
                    "quality_score": r.quality_score,
                    "math_preserved": r.math_preserved,
                    "code_preserved": r.code_preserved,
                    "chunk_count": r.chunk_count,
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# URL fetching
# ---------------------------------------------------------------------------

def _fetch_raw_html(url: str) -> tuple[str, str]:
    """Fetch raw HTML from a URL. Returns (html, error). Error empty on success."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
            raw = resp.read(MAX_CONTENT_BYTES + 1)
            if len(raw) > MAX_CONTENT_BYTES:
                raw = raw[:MAX_CONTENT_BYTES]
            return raw.decode("utf-8", errors="replace"), ""
    except urllib.error.HTTPError as exc:
        return "", f"HTTP {exc.code}: {exc.reason}"
    except urllib.error.URLError as exc:
        return "", f"URL error: {exc.reason}"
    except Exception as exc:
        return "", f"Fetch failed: {exc}"


# ---------------------------------------------------------------------------
# TOC discovery
# ---------------------------------------------------------------------------

_HREF_RE = re.compile(r'href="([^"]+)"', re.IGNORECASE)
_LINK_TEXT_RE = re.compile(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")

_SKIP_PATTERNS = [
    re.compile(r"index\.html$"),
    re.compile(r"genindex"),
    re.compile(r"search\.html$"),
    re.compile(r"^#"),
    re.compile(r"^mailto:"),
    re.compile(r"^javascript:"),
    re.compile(r"\.(pdf|zip|tar|gz|png|jpg|svg)$", re.IGNORECASE),
]


def discover_chapter_urls(toc_url: str) -> list[ChapterInfo]:
    """Discover chapter/page URLs from a table-of-contents page."""
    html, err = _fetch_raw_html(toc_url)
    if err:
        logger.error("Failed to fetch TOC at %s: %s", toc_url, err)
        return []

    parsed_base = urlparse(toc_url)
    base_domain = parsed_base.hostname or ""

    chapters: list[ChapterInfo] = []
    seen_urls: set[str] = set()
    order = 0

    for match in _LINK_TEXT_RE.finditer(html):
        href = match.group(1)
        link_text = _TAG_STRIP_RE.sub("", match.group(2)).strip()

        if any(p.search(href) for p in _SKIP_PATTERNS):
            continue

        full_url = urljoin(toc_url, href)
        link_domain = urlparse(full_url).hostname or ""
        if link_domain != base_domain:
            continue

        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        if not _is_content_link(href, full_url, base_domain):
            continue

        title = link_text or href.rsplit("/", 1)[-1].replace(".html", "").replace("-", " ").title()
        chapters.append(ChapterInfo(url=full_url, title=title, order=order))
        order += 1

    logger.info("Discovered %d chapter URLs from %s", len(chapters), toc_url)
    return chapters


def _is_content_link(href: str, full_url: str, base_domain: str) -> bool:
    """Heuristic: is this link likely a content page vs navigation/resource?"""
    if "chapter" in href.lower() or "/contents/" in href:
        return True
    if href.endswith(".html") and "/" in href:
        return True
    if href.endswith("/") and not href.startswith("http"):
        path = href.strip("/")
        if path and "/" in path and not path.startswith(("_", ".")):
            return True
    return False


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest_textbook(
    toc_url: str,
    title: str = "",
    domain_tags: str = "",
    study_now: bool = False,
    dry_run: bool = False,
) -> BatchIngestResult:
    """Discover and ingest all chapters from a textbook TOC page.

    When ``dry_run=True``, fetches and sanitizes content but stores nothing.
    """
    result = BatchIngestResult(success=False, title=title, toc_url=toc_url)

    chapters = discover_chapter_urls(toc_url)
    if not chapters:
        result.error = "No chapter URLs discovered from TOC"
        return result

    result.chapters_discovered = len(chapters)

    toc_html, _ = _fetch_raw_html(toc_url)
    site_type = detect_site_type(toc_html) if toc_html else "generic"
    result.site_type = site_type

    if not title:
        title = _extract_title(toc_html) or urlparse(toc_url).hostname or "Untitled Textbook"
        result.title = title

    textbook_id = hashlib.sha256(toc_url.encode()).hexdigest()[:12]
    base_tags = domain_tags or "textbook"
    if "textbook" not in base_tags:
        base_tags = f"{base_tags},textbook"

    for i, chapter in enumerate(chapters):
        if i > 0:
            time.sleep(FETCH_DELAY_S)

        ch_result = _ingest_one_chapter(
            chapter=chapter,
            site_type=site_type,
            textbook_id=textbook_id,
            textbook_title=title,
            base_tags=base_tags,
            study_now=study_now,
            dry_run=dry_run,
        )
        result.results.append(ch_result)

        if ch_result.skipped:
            result.chapters_skipped += 1
        elif ch_result.success:
            result.chapters_ingested += 1
            result.total_math += ch_result.math_preserved
            result.total_code += ch_result.code_preserved
            result.total_chunks += ch_result.chunk_count
        else:
            result.chapters_failed += 1

        logger.info(
            "Chapter %d/%d: %s — %s (quality=%.2f, math=%d, code=%d)",
            i + 1, len(chapters), chapter.title,
            "ingested" if ch_result.success else ("skipped" if ch_result.skipped else "failed"),
            ch_result.quality_score, ch_result.math_preserved, ch_result.code_preserved,
        )

    result.success = result.chapters_ingested > 0
    return result


def _ingest_one_chapter(
    chapter: ChapterInfo,
    site_type: str,
    textbook_id: str,
    textbook_title: str,
    base_tags: str,
    study_now: bool,
    dry_run: bool,
) -> ChapterResult:
    """Fetch, sanitize, and optionally ingest a single chapter."""
    ch_result = ChapterResult(url=chapter.url, title=chapter.title, success=False)

    raw_html, err = _fetch_raw_html(chapter.url)
    if err:
        ch_result.error = err
        return ch_result

    sanitized = sanitize(raw_html, site_type=site_type)
    ch_result.quality_score = sanitized.quality_score
    ch_result.math_preserved = sanitized.math_blocks_preserved
    ch_result.code_preserved = sanitized.code_blocks_preserved

    if sanitized.quality_score < MIN_QUALITY_SCORE:
        ch_result.skipped = True
        ch_result.skip_reason = f"quality_too_low:{sanitized.quality_score:.2f}"
        return ch_result

    if not sanitized.text or len(sanitized.text.strip()) < 200:
        ch_result.skipped = True
        ch_result.skip_reason = "content_too_short"
        return ch_result

    tags = f"{base_tags},textbook_id:{textbook_id}"

    if dry_run:
        from library.chunks import chunk_text
        chunks = chunk_text(sanitized.text, f"dry_run_{textbook_id}")
        ch_result.chunk_count = len(chunks)
        ch_result.success = True
        return ch_result

    try:
        from library.ingest import ingest_manual_source
        ingest_result = ingest_manual_source(
            content=sanitized.text,
            url=chapter.url,
            title=f"{textbook_title} — {chapter.title}",
            source_type="textbook_chapter",
            domain_tags=tags,
            study_now=study_now,
            ingested_by="textbook_ingest",
            quality_score=TEXTBOOK_QUALITY_SCORE,
            content_depth="full_text",
            trust_tier="curated",
        )
        ch_result.success = ingest_result.success
        ch_result.chunk_count = ingest_result.chunk_count
        ch_result.source_id = ingest_result.source_id
        if not ingest_result.success:
            ch_result.error = ingest_result.error
    except Exception as exc:
        ch_result.error = str(exc)
        logger.exception("Failed to ingest chapter: %s", chapter.url)

    return ch_result


def _extract_title(html: str) -> str:
    """Extract page title from HTML."""
    m = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
    if m:
        title = m.group(1).strip()
        for suffix in [" documentation", " &#8212;", " —", " - "]:
            if suffix in title:
                title = title.split(suffix)[0].strip()
        return title
    return ""
