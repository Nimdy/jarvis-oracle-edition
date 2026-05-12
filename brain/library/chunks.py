"""Chunk storage — content chunking and persistence for the document library.

Sources are split into chunks (256-512 token target) for granular retrieval.
Academic abstracts (typically short) become a single chunk.  Longer content is
split at paragraph boundaries.  All chunks share the same library.db as sources.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from library.db import get_connection, LIBRARY_WRITE_LOCK, LIBRARY_DB_PATH

logger = logging.getLogger(__name__)

TARGET_CHUNK_TOKENS = 384
MAX_CHUNK_TOKENS = 600
MIN_CHUNK_TOKENS = 40


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    text: str
    offset: int = 0            # position in source
    concepts: list[str] = field(default_factory=list)
    chunk_type: str = ""       # "abstract", "method", "result", "definition", "code", "snippet"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_chunk_id(source_id: str, offset: int) -> str:
    key = f"{source_id}:{offset}"
    return f"chk_{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def chunk_text(text: str, source_id: str, chunk_type: str = "") -> list[Chunk]:
    """Split text into chunks targeting TARGET_CHUNK_TOKENS.

    For short text (< MAX_CHUNK_TOKENS words), returns a single chunk.
    For longer text, splits at paragraph boundaries (\\n\\n), then at sentence
    boundaries if paragraphs are too large.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if len(words) <= MAX_CHUNK_TOKENS:
        return [Chunk(
            chunk_id=make_chunk_id(source_id, 0),
            source_id=source_id,
            text=text.strip(),
            offset=0,
            chunk_type=chunk_type or _classify_chunk(text),
        )]

    paragraphs = text.split("\n\n")
    chunks: list[Chunk] = []
    current_buf: list[str] = []
    current_len = 0
    offset = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_words = len(para.split())

        if current_len + para_words > MAX_CHUNK_TOKENS and current_buf:
            chunk_text_str = "\n\n".join(current_buf)
            chunks.append(Chunk(
                chunk_id=make_chunk_id(source_id, offset),
                source_id=source_id,
                text=chunk_text_str,
                offset=offset,
                chunk_type=chunk_type or _classify_chunk(chunk_text_str),
            ))
            offset += 1
            current_buf = []
            current_len = 0

        if para_words > MAX_CHUNK_TOKENS:
            sentences = _split_sentences(para)
            for sent in sentences:
                sent_words = len(sent.split())
                if current_len + sent_words > MAX_CHUNK_TOKENS and current_buf:
                    chunk_text_str = " ".join(current_buf)
                    chunks.append(Chunk(
                        chunk_id=make_chunk_id(source_id, offset),
                        source_id=source_id,
                        text=chunk_text_str,
                        offset=offset,
                        chunk_type=chunk_type or _classify_chunk(chunk_text_str),
                    ))
                    offset += 1
                    current_buf = []
                    current_len = 0
                current_buf.append(sent)
                current_len += sent_words
        else:
            current_buf.append(para)
            current_len += para_words

    if current_buf and current_len >= MIN_CHUNK_TOKENS:
        chunk_text_str = "\n\n".join(current_buf)
        chunks.append(Chunk(
            chunk_id=make_chunk_id(source_id, offset),
            source_id=source_id,
            text=chunk_text_str,
            offset=offset,
            chunk_type=chunk_type or _classify_chunk(chunk_text_str),
        ))

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Rough sentence splitting on '. ' boundaries."""
    parts = []
    for segment in text.replace("? ", "?|").replace("! ", "!|").replace(". ", ".|").split("|"):
        segment = segment.strip()
        if segment:
            parts.append(segment)
    return parts


def _classify_chunk(text: str) -> str:
    """Heuristic chunk type classification."""
    lower = text.lower()
    if any(kw in lower for kw in ("we propose", "we present", "we introduce", "this paper")):
        return "abstract"
    if any(kw in lower for kw in ("algorithm", "procedure", "step 1", "method")):
        return "method"
    if any(kw in lower for kw in ("result", "achieves", "outperforms", "accuracy")):
        return "result"
    if any(kw in lower for kw in ("defined as", "definition", "refers to")):
        return "definition"
    if any(kw in lower for kw in ("def ", "class ", "import ", "function")):
        return "code"
    return "snippet"


class ChunkStore:
    """SQLite-backed CRUD for Chunk objects.  Shares library.db with SourceStore."""

    _instance: ChunkStore | None = None

    def __init__(self, db_path: str = "") -> None:
        self._db_path = db_path or str(LIBRARY_DB_PATH)
        self._conn: sqlite3.Connection | None = None
        self._initialized = False

    @classmethod
    def get_instance(cls) -> ChunkStore:
        if cls._instance is None:
            cls._instance = ChunkStore()
        return cls._instance

    def init(self) -> None:
        if self._initialized:
            return
        self._conn = get_connection()
        self._create_tables()
        self._initialized = True
        logger.info("ChunkStore initialized")

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                text TEXT NOT NULL,
                offset_idx INTEGER DEFAULT 0,
                concepts TEXT DEFAULT '',
                chunk_type TEXT DEFAULT '',
                created_at REAL DEFAULT 0.0,
                FOREIGN KEY (source_id) REFERENCES sources(source_id)
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id)"
        )
        self._conn.commit()

    def _ensure_init(self) -> sqlite3.Connection:
        if not self._initialized:
            self.init()
        assert self._conn is not None
        return self._conn

    # -- Writes (locked) ----------------------------------------------------

    def add(self, chunk: Chunk) -> bool:
        conn = self._ensure_init()
        with LIBRARY_WRITE_LOCK:
            try:
                concepts_str = ",".join(chunk.concepts) if chunk.concepts else ""
                conn.execute("""
                    INSERT OR REPLACE INTO chunks
                        (chunk_id, source_id, text, offset_idx, concepts, chunk_type, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.chunk_id, chunk.source_id, chunk.text, chunk.offset,
                    concepts_str, chunk.chunk_type, chunk.created_at,
                ))
                conn.commit()
                return True
            except Exception:
                logger.exception("Failed to add chunk %s", chunk.chunk_id)
                return False

    def add_many(self, chunks: list[Chunk]) -> int:
        """Batch insert chunks in a single transaction.  Returns count added."""
        if not chunks:
            return 0
        conn = self._ensure_init()
        rows = [
            (
                c.chunk_id, c.source_id, c.text, c.offset,
                ",".join(c.concepts) if c.concepts else "",
                c.chunk_type, c.created_at,
            )
            for c in chunks
        ]
        with LIBRARY_WRITE_LOCK:
            try:
                conn.executemany("""
                    INSERT OR REPLACE INTO chunks
                        (chunk_id, source_id, text, offset_idx, concepts, chunk_type, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, rows)
                conn.commit()
                return len(rows)
            except Exception:
                logger.exception("Failed to batch-add %d chunks", len(chunks))
                return 0

    def update_concepts(self, chunk_id: str, concepts: list[str]) -> bool:
        """Update the concept tags for a chunk after study."""
        conn = self._ensure_init()
        with LIBRARY_WRITE_LOCK:
            try:
                conn.execute(
                    "UPDATE chunks SET concepts = ? WHERE chunk_id = ?",
                    (",".join(concepts), chunk_id),
                )
                conn.commit()
                return True
            except Exception:
                return False

    # -- Reads (unlocked) ---------------------------------------------------

    def get(self, chunk_id: str) -> Chunk | None:
        conn = self._ensure_init()
        row = conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    def get_for_source(self, source_id: str) -> list[Chunk]:
        conn = self._ensure_init()
        rows = conn.execute(
            "SELECT * FROM chunks WHERE source_id = ? ORDER BY offset_idx",
            (source_id,),
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_many(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        conn = self._ensure_init()
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = conn.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def search_text(self, query: str, limit: int = 10) -> list[Chunk]:
        """Basic keyword search over chunk text."""
        conn = self._ensure_init()
        pattern = f"%{query}%"
        rows = conn.execute(
            "SELECT * FROM chunks WHERE text LIKE ? LIMIT ?",
            (pattern, limit),
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def count(self) -> int:
        conn = self._ensure_init()
        row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        concepts_str = row["concepts"] or ""
        return Chunk(
            chunk_id=row["chunk_id"],
            source_id=row["source_id"],
            text=row["text"],
            offset=row["offset_idx"],
            concepts=[c for c in concepts_str.split(",") if c],
            chunk_type=row["chunk_type"],
            created_at=row["created_at"],
        )

    def get_stats(self) -> dict[str, Any]:
        conn = self._ensure_init()
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        with_concepts = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE concepts != ''"
        ).fetchone()[0]
        sources = conn.execute(
            "SELECT COUNT(DISTINCT source_id) FROM chunks"
        ).fetchone()[0]
        return {
            "total": total,
            "with_concepts": with_concepts,
            "sources": sources,
        }

    def close(self) -> None:
        self._conn = None
        self._initialized = False


chunk_store = ChunkStore.get_instance()
