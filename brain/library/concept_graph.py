"""Concept graph — lightweight co-occurrence graph of extracted concepts.

Concepts are nodes; edges represent co-occurrence within the same chunk.
Backed by two SQLite tables in library.db.  All concept names are lowercased.

Use:
  - concept_graph.add_concepts(chunk_id, concepts) during study
  - concept_graph.get_related(concept) for neighborhood queries
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

from library.db import get_connection, LIBRARY_WRITE_LOCK, LIBRARY_DB_PATH

logger = logging.getLogger(__name__)


class ConceptGraph:
    """SQLite-backed concept co-occurrence graph in library.db."""

    _instance: ConceptGraph | None = None

    def __init__(self, db_path: str = "") -> None:
        self._db_path = db_path or str(LIBRARY_DB_PATH)
        self._conn: sqlite3.Connection | None = None
        self._initialized = False

    @classmethod
    def get_instance(cls) -> ConceptGraph:
        if cls._instance is None:
            cls._instance = ConceptGraph()
        return cls._instance

    def init(self) -> None:
        if self._initialized:
            return
        self._conn = get_connection()
        self._create_tables()
        self._initialized = True
        logger.info("ConceptGraph initialized")

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                name TEXT PRIMARY KEY,
                first_seen_at REAL NOT NULL,
                last_seen_at REAL NOT NULL,
                count INTEGER DEFAULT 1
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS concept_edges (
                a TEXT NOT NULL,
                b TEXT NOT NULL,
                count INTEGER DEFAULT 1,
                last_seen_at REAL NOT NULL,
                PRIMARY KEY (a, b)
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_a ON concept_edges(a)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_b ON concept_edges(b)"
        )
        self._conn.commit()

    def _ensure_init(self) -> sqlite3.Connection:
        if not self._initialized:
            self.init()
        assert self._conn is not None
        return self._conn

    def add_concepts(self, chunk_id: str, concepts: list[str]) -> None:
        """Record concept occurrences and co-occurrence edges from a chunk.

        All concepts are lowercased.  Edges are created for every pair within
        the chunk (order-normalized: a < b).
        """
        if len(concepts) < 1:
            return

        conn = self._ensure_init()
        now = time.time()
        normalized = sorted(set(c.lower().strip() for c in concepts if c.strip()))
        if not normalized:
            return

        with LIBRARY_WRITE_LOCK:
            try:
                for name in normalized:
                    conn.execute("""
                        INSERT INTO concepts (name, first_seen_at, last_seen_at, count)
                        VALUES (?, ?, ?, 1)
                        ON CONFLICT(name) DO UPDATE SET
                            last_seen_at = excluded.last_seen_at,
                            count = count + 1
                    """, (name, now, now))

                for i in range(len(normalized)):
                    for j in range(i + 1, len(normalized)):
                        a, b = normalized[i], normalized[j]
                        if a > b:
                            a, b = b, a
                        conn.execute("""
                            INSERT INTO concept_edges (a, b, count, last_seen_at)
                            VALUES (?, ?, 1, ?)
                            ON CONFLICT(a, b) DO UPDATE SET
                                count = count + 1,
                                last_seen_at = excluded.last_seen_at
                        """, (a, b, now))

                conn.commit()
            except Exception as exc:
                logger.debug("Failed to update concept graph: %s", exc)

    def get_related(self, concept: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get concepts co-occurring with the given concept, ordered by count."""
        conn = self._ensure_init()
        concept = concept.lower().strip()
        rows = conn.execute("""
            SELECT
                CASE WHEN a = ? THEN b ELSE a END as neighbor,
                count,
                last_seen_at
            FROM concept_edges
            WHERE a = ? OR b = ?
            ORDER BY count DESC
            LIMIT ?
        """, (concept, concept, concept, limit)).fetchall()
        return [
            {"concept": r["neighbor"], "count": r["count"], "last_seen": r["last_seen_at"]}
            for r in rows
        ]

    def get_top_concepts(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get the most frequently seen concepts."""
        conn = self._ensure_init()
        rows = conn.execute(
            "SELECT name, count, first_seen_at, last_seen_at FROM concepts "
            "ORDER BY count DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "name": r["name"],
                "count": r["count"],
                "first_seen": r["first_seen_at"],
                "last_seen": r["last_seen_at"],
            }
            for r in rows
        ]

    def get_stats(self) -> dict[str, Any]:
        conn = self._ensure_init()
        concepts_count = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        edges_count = conn.execute("SELECT COUNT(*) FROM concept_edges").fetchone()[0]
        top_concepts: list[dict[str, Any]] = []
        for row in conn.execute(
            "SELECT name, count FROM concepts ORDER BY count DESC LIMIT 15"
        ).fetchall():
            top_concepts.append({"name": row[0], "count": row[1]})
        return {
            "total_concepts": concepts_count,
            "total_edges": edges_count,
            "top_concepts": top_concepts,
        }

    def close(self) -> None:
        self._conn = None
        self._initialized = False


concept_graph = ConceptGraph.get_instance()
