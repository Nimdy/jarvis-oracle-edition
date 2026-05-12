"""Document Library — persistent source retention, chunking, and indexed retrieval.

This module provides the knowledge storage layer that sits between raw research
results and the memory system.  Research findings are stored as Source objects
with full provenance; content is chunked and embedded for retrieval; memories
become lightweight pointers (claim + source_ids + chunk_ids) instead of flat
text blobs.
"""

from library.db import get_connection, LIBRARY_WRITE_LOCK, close_connection
from library.source import Source, SourceStore, source_store
from library.chunks import Chunk, ChunkStore, chunk_store
from library.index import LibraryIndex, library_index
from library.concept_graph import ConceptGraph, concept_graph
from library.telemetry import RetrievalTelemetry, retrieval_telemetry
from library.ingest import ingest_manual_source, IngestResult

__all__ = [
    "get_connection", "LIBRARY_WRITE_LOCK", "close_connection",
    "Source", "SourceStore", "source_store",
    "Chunk", "ChunkStore", "chunk_store",
    "LibraryIndex", "library_index",
    "ConceptGraph", "concept_graph",
    "RetrievalTelemetry", "retrieval_telemetry",
    "ingest_manual_source", "IngestResult",
]
