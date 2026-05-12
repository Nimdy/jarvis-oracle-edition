"""Source storage — first-class persistent store for research findings.

Every research result (academic paper, web page, codebase snippet) becomes a
Source object with full provenance metadata.  Sources are stored in SQLite
alongside chunks and embeddings in a single library.db file.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from library.db import get_connection, LIBRARY_WRITE_LOCK, LIBRARY_DB_PATH

logger = logging.getLogger(__name__)


@dataclass
class Source:
    source_id: str
    source_type: str            # "doi", "url", "local_file", "codebase", "memory", "introspection"
    retrieved_at: float
    url: str = ""
    doi: str = ""
    title: str = ""
    authors: str = ""
    year: int = 0
    venue: str = ""
    citation_count: int = 0
    content_text: str = ""      # abstract/snippet — what we're allowed to store
    license_flags: str = ""     # "open_access", "fair_use_abstract", "unknown"
    quality_score: float = 0.0  # 0-1, from confidence scoring
    provider: str = ""          # "semantic_scholar", "crossref", "ddg", "codebase"
    studied: bool = False       # True once the study pipeline has processed this
    studied_at: float = 0.0
    study_error: str = ""       # last error if study failed
    study_attempts: int = 0     # prevents thrash on persistent failures
    study_next_attempt_at: float = 0.0  # epoch seconds; exponential backoff
    ingested_by: str = "autonomous"  # "autonomous" | "user" | "admin"
    trust_tier: str = "unverified"   # "unverified" | "curated" | "verified"
    domain_tags: str = ""            # comma-separated: "mechanic,automotive"
    canonical_domain: str = ""       # derived from URL host
    content_depth: str = ""          # "title_only", "abstract", "tldr", "full_text", "metadata_only"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_source_id(doi: str = "", url: str = "", title: str = "") -> str:
    """Stable deterministic ID from the best available identifier."""
    key = doi or url or title
    if not key:
        key = f"unknown_{time.time()}"
    return f"src_{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def classify_effective_source_type(source: Source | dict[str, Any]) -> str:
    """Collapse raw storage types into truthier user-facing buckets."""
    if isinstance(source, dict):
        source_type = str(source.get("source_type", "") or "").strip()
        url = str(source.get("url", "") or "").strip()
        doi = str(source.get("doi", "") or "").strip()
        provider = str(source.get("provider", "") or "").strip().lower()
    else:
        source_type = str(getattr(source, "source_type", "") or "").strip()
        url = str(getattr(source, "url", "") or "").strip()
        doi = str(getattr(source, "doi", "") or "").strip()
        provider = str(getattr(source, "provider", "") or "").strip().lower()

    if provider in {"memory", "introspection"} or source_type in {"memory", "introspection"}:
        return "internal_signal"
    if source_type == "codebase":
        return "codebase"
    if source_type in {"peer_reviewed", "doi"} or doi:
        return "peer_reviewed"
    if source_type in {"url", "web"}:
        return "web" if url else "manual_text"
    if source_type in {"user_note", "local_file"}:
        return source_type
    return source_type or "unknown"


class SourceStore:
    """SQLite-backed CRUD for Source objects.  Single library.db file."""

    _instance: SourceStore | None = None

    def __init__(self, db_path: str | Path = "") -> None:
        self._db_path = Path(db_path) if db_path else LIBRARY_DB_PATH
        self._conn: sqlite3.Connection | None = None
        self._initialized = False

    @classmethod
    def get_instance(cls) -> SourceStore:
        if cls._instance is None:
            cls._instance = SourceStore()
        return cls._instance

    def init(self) -> None:
        if self._initialized:
            return
        self._conn = get_connection()
        self._create_tables()
        self._migrate_schema()
        self._initialized = True
        logger.info("SourceStore initialized")

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                retrieved_at REAL NOT NULL,
                url TEXT DEFAULT '',
                doi TEXT DEFAULT '',
                title TEXT DEFAULT '',
                authors TEXT DEFAULT '',
                year INTEGER DEFAULT 0,
                venue TEXT DEFAULT '',
                citation_count INTEGER DEFAULT 0,
                content_text TEXT DEFAULT '',
                license_flags TEXT DEFAULT '',
                quality_score REAL DEFAULT 0.0,
                provider TEXT DEFAULT '',
                studied INTEGER DEFAULT 0,
                studied_at REAL DEFAULT 0.0,
                study_error TEXT DEFAULT '',
                study_attempts INTEGER DEFAULT 0,
                study_next_attempt_at REAL DEFAULT 0.0,
                ingested_by TEXT DEFAULT 'autonomous',
                trust_tier TEXT DEFAULT 'unverified',
                domain_tags TEXT DEFAULT '',
                canonical_domain TEXT DEFAULT '',
                content_depth TEXT DEFAULT ''
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sources_doi ON sources(doi) WHERE doi != ''"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sources_studied ON sources(studied)"
        )
        self._conn.commit()

    def _migrate_schema(self) -> None:
        """Add columns that may be missing from older library.db files."""
        assert self._conn is not None
        existing = {
            row[1] for row in self._conn.execute("PRAGMA table_info(sources)").fetchall()
        }
        migrations = [
            ("study_error", "TEXT DEFAULT ''"),
            ("study_attempts", "INTEGER DEFAULT 0"),
            ("study_next_attempt_at", "REAL DEFAULT 0.0"),
            ("ingested_by", "TEXT DEFAULT 'autonomous'"),
            ("trust_tier", "TEXT DEFAULT 'unverified'"),
            ("domain_tags", "TEXT DEFAULT ''"),
            ("canonical_domain", "TEXT DEFAULT ''"),
            ("content_depth", "TEXT DEFAULT ''"),
        ]
        for col_name, col_def in migrations:
            if col_name not in existing:
                self._conn.execute(f"ALTER TABLE sources ADD COLUMN {col_name} {col_def}")
                logger.info("Migrated sources table: added column %s", col_name)
        self._conn.commit()

        self._backfill_domain_tags()

    def _backfill_domain_tags(self) -> None:
        """One-time backfill: derive domain_tags from venue + source_type for existing sources."""
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT source_id, source_type, venue FROM sources WHERE domain_tags = '' OR domain_tags IS NULL"
        ).fetchall()
        if not rows:
            return
        updated = 0
        for row in rows:
            parts: list[str] = []
            venue = row[2] or ""
            stype = row[1] or ""
            if venue:
                parts.append(venue.lower().replace(" ", "_"))
            if stype and stype not in ("url", "web", ""):
                parts.append(stype)
            if not parts:
                continue
            tags = ",".join(parts)[:200]
            self._conn.execute(
                "UPDATE sources SET domain_tags = ? WHERE source_id = ?",
                (tags, row[0]),
            )
            updated += 1
        if updated:
            self._conn.commit()
            logger.info("Backfilled domain_tags for %d sources", updated)

    def _ensure_init(self) -> sqlite3.Connection:
        if not self._initialized:
            self.init()
        assert self._conn is not None
        return self._conn

    # -- CRUD (writes locked, reads unlocked) -------------------------------

    def add(self, source: Source) -> bool:
        conn = self._ensure_init()
        with LIBRARY_WRITE_LOCK:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO sources (
                        source_id, source_type, retrieved_at, url, doi, title,
                        authors, year, venue, citation_count, content_text,
                        license_flags, quality_score, provider, studied, studied_at,
                        study_error, study_attempts, study_next_attempt_at,
                        ingested_by, trust_tier, domain_tags, canonical_domain,
                        content_depth
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    source.source_id, source.source_type, source.retrieved_at,
                    source.url, source.doi, source.title, source.authors,
                    source.year, source.venue, source.citation_count,
                    source.content_text, source.license_flags, source.quality_score,
                    source.provider, int(source.studied), source.studied_at,
                    source.study_error, source.study_attempts,
                    source.study_next_attempt_at,
                    source.ingested_by, source.trust_tier, source.domain_tags,
                    source.canonical_domain, source.content_depth,
                ))
                conn.commit()
                return True
            except Exception:
                logger.exception("Failed to add source %s", source.source_id)
                return False

    def get(self, source_id: str) -> Source | None:
        conn = self._ensure_init()
        row = conn.execute(
            "SELECT * FROM sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_source(row)

    def get_by_doi(self, doi: str) -> Source | None:
        conn = self._ensure_init()
        row = conn.execute(
            "SELECT * FROM sources WHERE doi = ?", (doi,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_source(row)

    def get_unstudied(self, limit: int = 10) -> list[Source]:
        conn = self._ensure_init()
        now = time.time()
        rows = conn.execute(
            "SELECT * FROM sources WHERE studied = 0 AND study_attempts < 3 "
            "AND study_next_attempt_at <= ? "
            "ORDER BY retrieved_at DESC LIMIT ?",
            (now, limit),
        ).fetchall()
        return [self._row_to_source(r) for r in rows]

    def mark_studied(self, source_id: str) -> bool:
        conn = self._ensure_init()
        with LIBRARY_WRITE_LOCK:
            try:
                conn.execute(
                    "UPDATE sources SET studied = 1, studied_at = ?, study_error = '' "
                    "WHERE source_id = ?",
                    (time.time(), source_id),
                )
                conn.commit()
                return True
            except Exception:
                logger.exception("Failed to mark source %s as studied", source_id)
                return False

    def mark_study_error(self, source_id: str, error: str) -> bool:
        conn = self._ensure_init()
        with LIBRARY_WRITE_LOCK:
            try:
                row = conn.execute(
                    "SELECT study_attempts FROM sources WHERE source_id = ?",
                    (source_id,),
                ).fetchone()
                attempts = (row[0] if row else 0) + 1
                backoff_s = {1: 300, 2: 1800}.get(attempts, 21600)
                next_at = time.time() + backoff_s
                conn.execute(
                    "UPDATE sources SET study_error = ?, "
                    "study_attempts = ?, study_next_attempt_at = ? "
                    "WHERE source_id = ?",
                    (error[:500], attempts, next_at, source_id),
                )
                conn.commit()
                return True
            except Exception:
                return False

    def update_quality(self, source_id: str, quality_score: float) -> bool:
        conn = self._ensure_init()
        with LIBRARY_WRITE_LOCK:
            try:
                conn.execute(
                    "UPDATE sources SET quality_score = ? WHERE source_id = ?",
                    (quality_score, source_id),
                )
                conn.commit()
                return True
            except Exception:
                return False

    def adjust_quality(self, source_id: str, delta: float) -> bool:
        """Adjust quality_score by a delta, clamped to [0, 1]."""
        conn = self._ensure_init()
        with LIBRARY_WRITE_LOCK:
            try:
                conn.execute(
                    "UPDATE sources SET quality_score = "
                    "MAX(0.0, MIN(1.0, quality_score + ?)) WHERE source_id = ?",
                    (delta, source_id),
                )
                conn.commit()
                return True
            except Exception:
                return False

    def exists(self, source_id: str) -> bool:
        conn = self._ensure_init()
        row = conn.execute(
            "SELECT 1 FROM sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        return row is not None

    def count(self) -> int:
        conn = self._ensure_init()
        row = conn.execute("SELECT COUNT(*) FROM sources").fetchone()
        return row[0] if row else 0

    def get_recent(self, limit: int = 20) -> list[Source]:
        conn = self._ensure_init()
        rows = conn.execute(
            "SELECT * FROM sources ORDER BY retrieved_at DESC LIMIT ?", (limit,),
        ).fetchall()
        return [self._row_to_source(r) for r in rows]

    def search_text(self, query: str, limit: int = 10) -> list[Source]:
        """Basic keyword search over title + content_text."""
        conn = self._ensure_init()
        pattern = f"%{query}%"
        rows = conn.execute(
            "SELECT * FROM sources WHERE title LIKE ? OR content_text LIKE ? "
            "ORDER BY quality_score DESC LIMIT ?",
            (pattern, pattern, limit),
        ).fetchall()
        return [self._row_to_source(r) for r in rows]

    def get_stats(self) -> dict[str, Any]:
        conn = self._ensure_init()
        total = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        studied = conn.execute(
            "SELECT COUNT(*) FROM sources WHERE studied = 1"
        ).fetchone()[0]
        failed = conn.execute(
            "SELECT COUNT(*) FROM sources WHERE study_attempts >= 3 AND studied = 0"
        ).fetchone()[0]
        backoff = conn.execute(
            "SELECT COUNT(*) FROM sources WHERE study_next_attempt_at > ? AND studied = 0",
            (time.time(),),
        ).fetchone()[0]
        by_type_raw: dict[str, int] = {}
        for row in conn.execute(
            "SELECT source_type, COUNT(*) FROM sources GROUP BY source_type"
        ).fetchall():
            by_type_raw[row[0]] = row[1]
        by_type_effective: dict[str, int] = {}
        try:
            for row in conn.execute(
                "SELECT source_type, url, doi FROM sources"
            ).fetchall():
                effective = classify_effective_source_type({
                    "source_type": row[0],
                    "url": row[1],
                    "doi": row[2],
                })
                by_type_effective[effective] = by_type_effective.get(effective, 0) + 1
        except Exception:
            by_type_effective = dict(by_type_raw)
        by_ingested: dict[str, int] = {}
        try:
            for row in conn.execute(
                "SELECT ingested_by, COUNT(*) FROM sources GROUP BY ingested_by"
            ).fetchall():
                by_ingested[row[0]] = row[1]
        except Exception:
            pass
        top_domains: list[dict[str, Any]] = []
        try:
            tag_counts: dict[str, int] = {}
            for row in conn.execute(
                "SELECT domain_tags FROM sources WHERE domain_tags != ''"
            ).fetchall():
                for tag in row[0].split(","):
                    tag = tag.strip()
                    if tag:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
            for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:15]:
                top_domains.append({"tags": tag, "count": count})
        except Exception:
            pass
        by_depth: dict[str, int] = {}
        try:
            for row in conn.execute(
                "SELECT content_depth, COUNT(*) FROM sources GROUP BY content_depth"
            ).fetchall():
                depth = row[0] or "unknown"
                by_depth[depth] = row[1]
        except Exception:
            pass
        return {
            "total": total,
            "studied": studied,
            "unstudied": total - studied - failed,
            "failed": failed,
            "backoff_pending": backoff,
            "by_type": by_type_effective,
            "by_type_effective": by_type_effective,
            "by_type_raw": by_type_raw,
            "by_ingested_by": by_ingested,
            "top_domains": top_domains,
            "by_content_depth": by_depth,
        }

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _row_to_source(row: sqlite3.Row) -> Source:
        return Source(
            source_id=row["source_id"],
            source_type=row["source_type"],
            retrieved_at=row["retrieved_at"],
            url=row["url"],
            doi=row["doi"],
            title=row["title"],
            authors=row["authors"],
            year=row["year"],
            venue=row["venue"],
            citation_count=row["citation_count"],
            content_text=row["content_text"],
            license_flags=row["license_flags"],
            quality_score=row["quality_score"],
            provider=row["provider"],
            studied=bool(row["studied"]),
            studied_at=row["studied_at"],
            study_error=row["study_error"] if "study_error" in row.keys() else "",
            study_attempts=row["study_attempts"] if "study_attempts" in row.keys() else 0,
            study_next_attempt_at=row["study_next_attempt_at"] if "study_next_attempt_at" in row.keys() else 0.0,
            ingested_by=row["ingested_by"] if "ingested_by" in row.keys() else "autonomous",
            trust_tier=row["trust_tier"] if "trust_tier" in row.keys() else "unverified",
            domain_tags=row["domain_tags"] if "domain_tags" in row.keys() else "",
            canonical_domain=row["canonical_domain"] if "canonical_domain" in row.keys() else "",
            content_depth=row["content_depth"] if "content_depth" in row.keys() else "",
        )

    def list_sources(
        self,
        limit: int = 20,
        offset: int = 0,
        ingested_by: str = "",
    ) -> list[Source]:
        """Return sources ordered by retrieved_at desc, optionally filtered."""
        conn = self._ensure_init()
        if ingested_by:
            rows = conn.execute(
                "SELECT * FROM sources WHERE ingested_by = ? ORDER BY retrieved_at DESC LIMIT ? OFFSET ?",
                (ingested_by, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM sources ORDER BY retrieved_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_source(r) for r in rows]

    def get_source(self, source_id: str) -> Source | None:
        """Return a single source by ID."""
        conn = self._ensure_init()
        row = conn.execute(
            "SELECT * FROM sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        if row:
            return self._row_to_source(row)
        return None

    def close(self) -> None:
        self._conn = None
        self._initialized = False


source_store = SourceStore.get_instance()
