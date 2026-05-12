"""Shared library database connection and write lock.

All library modules (SourceStore, ChunkStore, LibraryIndex, ConceptGraph) use
the same SQLite file (library.db).  SQLite only allows one writer at a time,
even in WAL mode.  This module provides:

  - get_connection() -> a single shared sqlite3.Connection (check_same_thread=False)
  - LIBRARY_WRITE_LOCK -> a threading.Lock that ALL write operations must hold

This prevents "database is locked" errors when ingestion, study, and context
resolution happen concurrently across different singletons.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
LIBRARY_DIR = JARVIS_DIR / "library"
LIBRARY_DB_PATH = LIBRARY_DIR / "library.db"

LIBRARY_WRITE_LOCK = threading.Lock()

_conn: sqlite3.Connection | None = None
_conn_lock = threading.Lock()


def get_connection() -> sqlite3.Connection:
    """Return the shared library.db connection (created once, reused)."""
    global _conn
    if _conn is not None:
        return _conn
    with _conn_lock:
        if _conn is not None:
            return _conn
        LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(LIBRARY_DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA busy_timeout=5000")
        logger.info("Library DB connection opened: %s", LIBRARY_DB_PATH)
        return _conn


def close_connection() -> None:
    """Close the shared connection (for clean shutdown)."""
    global _conn
    with _conn_lock:
        if _conn:
            _conn.close()
            _conn = None
