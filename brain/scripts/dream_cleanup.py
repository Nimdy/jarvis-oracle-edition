#!/usr/bin/env python3
"""One-time dream data cleanup (Phase 1.5 of Dream Observer Build Plan).

Run after deploying Phase 1.1-1.4 code patches.

Actions:
  A) Downweight existing dream insight memories to 0.4
  B) Supersede dream-derived beliefs and remove their graph edges

Usage:
  cd brain && python -m scripts.dream_cleanup [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def cleanup_dream_insights(*, dry_run: bool) -> int:
    """Step A: Reset inflated dream insight weights to 0.4."""
    from memory.storage import MemoryStorage

    storage = MemoryStorage()

    try:
        from memory.persistence import load_memories
        load_memories(storage)
    except Exception as e:
        logger.error("Failed to load memories: %s", e)
        return 0

    dream_insights = storage.get_by_tag("dream_insight")
    logger.info("Found %d dream insight memories", len(dream_insights))

    reset_count = 0
    for mem in dream_insights:
        if mem.weight > 0.45:
            factor = 0.4 / mem.weight
            if dry_run:
                logger.info("  [DRY-RUN] Would downweight %s: %.3f -> 0.4", mem.id[:12], mem.weight)
            else:
                ok = storage.downweight(mem.id, weight_factor=factor, decay_rate_factor=1.0)
                if ok:
                    logger.info("  Downweighted %s: %.3f -> 0.4", mem.id[:12], mem.weight)
            reset_count += 1
        else:
            logger.info("  Skipping %s: weight %.3f already at or below 0.45", mem.id[:12], mem.weight)

    logger.info("Step A: %d dream insights %s", reset_count, "would be reset" if dry_run else "reset")
    return reset_count


def cleanup_dream_beliefs(*, dry_run: bool) -> int:
    """Step B: Supersede dream-derived beliefs and remove their graph edges."""
    from epistemic.contradiction_engine import ContradictionEngine

    engine = ContradictionEngine.get_instance()
    store = engine._belief_store

    try:
        store.rehydrate()
    except Exception:
        pass

    edge_store = None
    try:
        from epistemic.belief_graph import BeliefGraph
        bg = BeliefGraph.get_instance()
        edge_store = bg._edge_store
    except Exception:
        logger.warning("Could not load belief graph edge store; edges won't be cleaned")

    active = store.get_active_beliefs()
    dream_beliefs = [b for b in active if b.canonical_subject.startswith("dream_insight_")]
    logger.info("Found %d dream-derived beliefs out of %d active", len(dream_beliefs), len(active))

    superseded_count = 0
    edges_removed = 0
    for b in dream_beliefs:
        if dry_run:
            logger.info("  [DRY-RUN] Would supersede belief %s (subject=%s)", b.belief_id[:12], b.canonical_subject[:40])
        else:
            store.update_resolution(b.belief_id, "superseded")
            logger.info("  Superseded belief %s (subject=%s)", b.belief_id[:12], b.canonical_subject[:40])

        if edge_store:
            if dry_run:
                from_edges = edge_store._outgoing.get(b.belief_id, set())
                to_edges = edge_store._incoming.get(b.belief_id, set())
                n = len(from_edges | to_edges)
                logger.info("    [DRY-RUN] Would remove %d edges", n)
                edges_removed += n
            else:
                n = edge_store.remove_edges_for_belief(b.belief_id)
                logger.info("    Removed %d edges", n)
                edges_removed += n

        superseded_count += 1

    if not dry_run and superseded_count > 0:
        store.persist_full()
        logger.info("Persisted belief store (full rewrite)")

    logger.info(
        "Step B: %d beliefs %s, %d edges %s",
        superseded_count,
        "would be superseded" if dry_run else "superseded",
        edges_removed,
        "would be removed" if dry_run else "removed",
    )
    return superseded_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Dream data cleanup (Phase 1.5)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying them")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Dream Data Cleanup — Phase 1.5")
    logger.info("Mode: %s", "DRY RUN" if args.dry_run else "LIVE")
    logger.info("=" * 60)

    insights_reset = cleanup_dream_insights(dry_run=args.dry_run)
    beliefs_superseded = cleanup_dream_beliefs(dry_run=args.dry_run)

    logger.info("")
    logger.info("Summary: %d insights reset, %d beliefs superseded", insights_reset, beliefs_superseded)
    if args.dry_run:
        logger.info("(No changes made — pass without --dry-run to apply)")


if __name__ == "__main__":
    main()
