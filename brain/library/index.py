"""Library index — sqlite-vec backed semantic search over library chunks.

Provides embedding + retrieval for chunks stored in library.db.  Shares the
same database file as SourceStore and ChunkStore.  Uses all-MiniLM-L6-v2
(384-dim) for embeddings, matching the memory vector store's model.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from pathlib import Path
from typing import Any

from library.db import get_connection, LIBRARY_WRITE_LOCK, LIBRARY_DB_PATH

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384

try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False
    logger.info("sqlite-vec not available — library semantic search disabled")

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    logger.info("sentence-transformers not available — library semantic search disabled")


def _serialize_f32(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class LibraryIndex:
    """sqlite-vec semantic index for library chunks.

    The virtual table `vec_chunks` lives in the same library.db alongside the
    `sources` and `chunks` tables.  A companion `chunk_vectors` metadata table
    maps chunk_ids to rowids used by the virtual table.
    """

    _instance: LibraryIndex | None = None

    def __init__(self, db_path: str = "", device: str = "cpu") -> None:
        self._db_path = db_path or str(LIBRARY_DB_PATH)
        self._model: Any = None
        self._conn: sqlite3.Connection | None = None
        self._device = device
        self.available = False

    @classmethod
    def get_instance(cls) -> LibraryIndex:
        if cls._instance is None:
            cls._instance = LibraryIndex()
        return cls._instance

    def init(self, device: str = "") -> None:
        if self.available:
            return
        if device:
            self._device = device

        if not SQLITE_VEC_AVAILABLE or not SBERT_AVAILABLE:
            logger.warning("LibraryIndex unavailable: sqlite-vec=%s sbert=%s",
                           SQLITE_VEC_AVAILABLE, SBERT_AVAILABLE)
            return

        try:
            from config import get_models_dir
            _cache = str(get_models_dir() / "huggingface")
            try:
                self._model = SentenceTransformer("all-MiniLM-L6-v2", device=self._device,
                                                  cache_folder=_cache, local_files_only=True)
            except Exception:
                logger.info("SentenceTransformer local cache miss, downloading all-MiniLM-L6-v2")
                self._model = SentenceTransformer("all-MiniLM-L6-v2", device=self._device,
                                                  cache_folder=_cache)
            self._init_db()
            self.available = True
            count = self._count()
            logger.info("LibraryIndex ready: %s (%d vectors, device=%s)",
                        self._db_path, count, self._device)
        except Exception as exc:
            logger.error("Failed to initialize LibraryIndex: %s", exc)

    def _init_db(self) -> None:
        self._conn = get_connection()
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_vectors (
                chunk_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                text_preview TEXT DEFAULT ''
            )
        """)
        self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks
            USING vec0(embedding float[{EMBEDDING_DIM}])
        """)
        self._conn.commit()

    def _ensure_init(self) -> sqlite3.Connection:
        if not self.available:
            self.init()
        assert self._conn is not None
        return self._conn

    # -- Embedding ----------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        if not self._model:
            return []
        return self._model.encode(text).tolist()

    # -- Writes (locked) ----------------------------------------------------

    def add_chunk(self, chunk_id: str, source_id: str, text: str) -> bool:
        if not self.available:
            return False
        try:
            embedding = self.embed(text)
            if not embedding:
                return False

            conn = self._ensure_init()
            preview = text[:200] if len(text) > 200 else text

            with LIBRARY_WRITE_LOCK:
                existing = conn.execute(
                    "SELECT rowid FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,)
                ).fetchone()

                if existing:
                    rowid = existing[0]
                    conn.execute(
                        "UPDATE chunk_vectors SET source_id=?, text_preview=? WHERE chunk_id=?",
                        (source_id, preview, chunk_id),
                    )
                    conn.execute(
                        "UPDATE vec_chunks SET embedding=? WHERE rowid=?",
                        (_serialize_f32(embedding), rowid),
                    )
                else:
                    conn.execute(
                        "INSERT INTO chunk_vectors (chunk_id, source_id, text_preview) VALUES (?, ?, ?)",
                        (chunk_id, source_id, preview),
                    )
                    new_rowid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    conn.execute(
                        "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                        (new_rowid, _serialize_f32(embedding)),
                    )

                conn.commit()
            return True
        except Exception as exc:
            logger.error("Failed to index chunk %s: %s", chunk_id, exc)
            try:
                conn = self._ensure_init()
                conn.rollback()
            except Exception:
                pass
            return False

    def add_chunks_batch(self, chunks: list[dict[str, str]]) -> int:
        """Batch-index chunks.  Each dict needs: chunk_id, source_id, text."""
        indexed = 0
        for c in chunks:
            if self.add_chunk(c["chunk_id"], c["source_id"], c["text"]):
                indexed += 1
        return indexed

    def remove_chunk(self, chunk_id: str) -> bool:
        if not self.available:
            return False
        try:
            conn = self._ensure_init()
            with LIBRARY_WRITE_LOCK:
                row = conn.execute(
                    "SELECT rowid FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,)
                ).fetchone()
                if not row:
                    return False
                conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (row[0],))
                conn.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
                conn.commit()
            return True
        except Exception:
            return False

    def remove_source(self, source_id: str) -> int:
        """Remove all chunk embeddings for a source."""
        if not self.available:
            return 0
        try:
            conn = self._ensure_init()
            with LIBRARY_WRITE_LOCK:
                rows = conn.execute(
                    "SELECT rowid FROM chunk_vectors WHERE source_id = ?", (source_id,)
                ).fetchall()
                for row in rows:
                    conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (row[0],))
                conn.execute(
                    "DELETE FROM chunk_vectors WHERE source_id = ?", (source_id,)
                )
                conn.commit()
            return len(rows)
        except Exception:
            return 0

    # -- Reads (unlocked) ---------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_id_filter: str = "",
    ) -> list[dict[str, Any]]:
        """Semantic search for chunks closest to query."""
        if not self.available:
            return []
        try:
            query_emb = self.embed(query)
            if not query_emb:
                return []

            conn = self._ensure_init()

            if source_id_filter:
                rows = conn.execute("""
                    SELECT
                        cv.chunk_id,
                        cv.source_id,
                        cv.text_preview,
                        vec_distance_cosine(vc.embedding, ?) as distance
                    FROM vec_chunks vc
                    JOIN chunk_vectors cv ON cv.rowid = vc.rowid
                    WHERE cv.source_id = ?
                    ORDER BY distance
                    LIMIT ?
                """, (_serialize_f32(query_emb), source_id_filter, top_k)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT
                        cv.chunk_id,
                        cv.source_id,
                        cv.text_preview,
                        vec_distance_cosine(vc.embedding, ?) as distance
                    FROM vec_chunks vc
                    JOIN chunk_vectors cv ON cv.rowid = vc.rowid
                    ORDER BY distance
                    LIMIT ?
                """, (_serialize_f32(query_emb), top_k)).fetchall()

            return [
                {
                    "chunk_id": r[0],
                    "source_id": r[1],
                    "text_preview": r[2],
                    "distance": r[3],
                    "similarity": 1.0 - r[3],
                }
                for r in rows
            ]
        except Exception as exc:
            logger.error("Library search failed: %s", exc)
            return []

    def _count(self) -> int:
        try:
            return self._conn.execute(
                "SELECT COUNT(*) FROM chunk_vectors"
            ).fetchone()[0]
        except Exception:
            return 0

    def get_stats(self) -> dict[str, Any]:
        if not self.available:
            return {"available": False, "total_chunks": 0}
        try:
            total = self._count()
            sources = self._conn.execute(
                "SELECT COUNT(DISTINCT source_id) FROM chunk_vectors"
            ).fetchone()[0]
            return {
                "available": True,
                "total_chunks": total,
                "total_sources": sources,
            }
        except Exception:
            return {"available": self.available, "total_chunks": 0}

    def close(self) -> None:
        self._conn = None
        self.available = False


library_index = LibraryIndex.get_instance()
