"""Vector memory store — sqlite-vec backed semantic search.

Stores sentence embeddings alongside memory IDs for meaning-based retrieval.
Uses all-MiniLM-L6-v2 (384-dim) for embeddings on the laptop GPU.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
DEFAULT_DB_PATH = JARVIS_DIR / "vector_memory.db"

try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False
    logger.info("sqlite-vec not available — semantic search disabled")

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    logger.info("sentence-transformers not available — semantic search disabled")


def _serialize_f32(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class VectorStore:
    """SQLite-vec backed vector store for semantic memory search."""

    def __init__(
        self,
        db_path: str = "",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        device: str = "cpu",
    ):
        self._db_path = db_path or str(DEFAULT_DB_PATH)
        self._dim = embedding_dim
        self._model = None
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self.available = False

        if not SQLITE_VEC_AVAILABLE or not SBERT_AVAILABLE:
            return

        try:
            from config import get_models_dir
            _cache = str(get_models_dir() / "huggingface")
            try:
                self._model = SentenceTransformer(embedding_model, device=device,
                                                  cache_folder=_cache, local_files_only=True)
            except Exception:
                logger.info("SentenceTransformer local cache miss, downloading %s", embedding_model)
                self._model = SentenceTransformer(embedding_model, device=device, cache_folder=_cache)
            self._init_db()
            self.available = True
            count = self._count()
            logger.info("Vector store ready: %s (%d vectors, dim=%d, device=%s)",
                        self._db_path, count, self._dim, device)
        except Exception as exc:
            logger.error("Failed to initialize vector store: %s", exc)

    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_vectors (
                memory_id TEXT PRIMARY KEY,
                text_content TEXT NOT NULL,
                memory_type TEXT DEFAULT '',
                weight REAL DEFAULT 0.5,
                created_at REAL DEFAULT 0.0
            )
        """)
        self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_memory
            USING vec0(embedding float[{self._dim}])
        """)
        self._conn.commit()

    def embed(self, text: str) -> list[float]:
        if not self._model:
            return []
        return self._model.encode(text).tolist()

    def _ensure_conn(self) -> bool:
        """Reconnect if the SQLite connection was lost (e.g. during sleep)."""
        if self._conn is not None:
            try:
                self._conn.execute("SELECT 1")
                return True
            except Exception:
                logger.warning("VectorStore connection stale, reconnecting")
                self._conn = None

        try:
            self._init_db()
            logger.info("VectorStore reconnected to %s", self._db_path)
            return True
        except Exception as exc:
            logger.error("VectorStore reconnection failed: %s", exc)
            return False

    def add(
        self,
        memory_id: str,
        text: str,
        memory_type: str = "",
        weight: float = 0.5,
    ) -> bool:
        """Add or update a memory's vector embedding."""
        if not self.available:
            return False
        try:
            embedding = self.embed(text)
            if not embedding:
                return False

            with self._lock:
                if not self._ensure_conn():
                    return False
                rowid = self._get_rowid_unlocked(memory_id)
                if rowid is not None:
                    self._conn.execute(
                        "UPDATE memory_vectors SET text_content=?, memory_type=?, weight=? WHERE memory_id=?",
                        (text, memory_type, weight, memory_id),
                    )
                    self._conn.execute(
                        "UPDATE vec_memory SET embedding=? WHERE rowid=?",
                        (_serialize_f32(embedding), rowid),
                    )
                else:
                    self._conn.execute(
                        "INSERT INTO memory_vectors (memory_id, text_content, memory_type, weight, created_at) VALUES (?, ?, ?, ?, ?)",
                        (memory_id, text, memory_type, weight, time.time()),
                    )
                    new_rowid = self._conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    self._conn.execute(
                        "INSERT INTO vec_memory (rowid, embedding) VALUES (?, ?)",
                        (new_rowid, _serialize_f32(embedding)),
                    )
                self._conn.commit()
            return True
        except Exception as exc:
            logger.error("Failed to add vector: %s", exc)
            return False

    def search(self, query: str, top_k: int = 5, min_weight: float = 0.0) -> list[dict[str, Any]]:
        """Semantic search: find memories closest to query by meaning."""
        if not self.available:
            return []
        try:
            query_emb = self.embed(query)
            if not query_emb:
                return []

            with self._lock:
                if not self._ensure_conn():
                    return []
                rows = self._conn.execute("""
                    SELECT
                        mv.memory_id,
                        mv.text_content,
                        mv.memory_type,
                        mv.weight,
                        vec_distance_cosine(vm.embedding, ?) as distance
                    FROM vec_memory vm
                    JOIN memory_vectors mv ON mv.rowid = vm.rowid
                    WHERE mv.weight >= ?
                    ORDER BY distance
                    LIMIT ?
                """, (_serialize_f32(query_emb), min_weight, top_k)).fetchall()

            return [
                {
                    "memory_id": r[0],
                    "text": r[1],
                    "type": r[2],
                    "weight": r[3],
                    "distance": r[4],
                    "similarity": 1.0 - r[4],
                }
                for r in rows
            ]
        except Exception as exc:
            logger.error("Vector search failed: %s", exc)
            return []

    def remove(self, memory_id: str) -> bool:
        if not self.available:
            return False
        try:
            with self._lock:
                if not self._ensure_conn():
                    return False
                rowid = self._get_rowid_unlocked(memory_id)
                if rowid is None:
                    return False
                self._conn.execute("DELETE FROM vec_memory WHERE rowid=?", (rowid,))
                self._conn.execute("DELETE FROM memory_vectors WHERE memory_id=?", (memory_id,))
                self._conn.commit()
            return True
        except Exception:
            return False

    def rebuild_from_memories(self, memories: list[Any]) -> int:
        """Rebuild the entire vector index from memory objects."""
        if not self.available:
            return 0

        count = 0
        for mem in memories:
            try:
                from memory.search import _extract_embedding_text
                text = _extract_embedding_text(mem.payload)
            except ImportError:
                text = mem.payload if isinstance(mem.payload, str) else str(mem.payload)
            if self.add(mem.id, text, mem.type, mem.weight):
                count += 1

        logger.info("Rebuilt vector store: %d memories indexed", count)
        return count

    def _get_rowid_unlocked(self, memory_id: str) -> int | None:
        """Return rowid for a memory_id. Caller must hold self._lock."""
        row = self._conn.execute(
            "SELECT rowid FROM memory_vectors WHERE memory_id=?", (memory_id,),
        ).fetchone()
        return row[0] if row else None

    def _count(self) -> int:
        try:
            with self._lock:
                if not self._ensure_conn():
                    return 0
                return self._conn.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0]
        except Exception:
            return 0

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
