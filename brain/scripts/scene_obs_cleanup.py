#!/usr/bin/env python3
"""One-time cleanup of raw VLM scene observation memories.

These memories were created by an unguarded engine.remember() call in
perception_orchestrator._analyze_scene() that bypassed CueGate and
SpatialMemoryGate.  The VLM descriptions already flow into the
WorldModel via SceneTracker; storing them as memories is redundant.

Actions:
  A) Remove never-accessed "Scene observation:" memories
  B) Downweight accessed ones to 0.10 so natural decay handles them

Usage:
  cd brain && python -m scripts.scene_obs_cleanup [--dry-run]
"""
from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SCENE_PREFIX = "Scene observation:"


def cleanup_scene_observations(*, dry_run: bool) -> tuple[int, int]:
    """Remove or downweight raw VLM scene observation memories."""
    from memory.persistence import MemoryPersistence
    from memory.storage import memory_storage

    mp = MemoryPersistence()
    loaded = mp.load()
    if loaded == 0:
        logger.error("No memories loaded — nothing to clean")
        return 0, 0

    all_mems = memory_storage.get_all()
    scene_mems = [
        m for m in all_mems
        if m.type == "observation"
        and isinstance(m.payload, str)
        and m.payload.startswith(SCENE_PREFIX)
    ]

    logger.info("Total memories: %d", len(all_mems))
    logger.info("Scene observation memories: %d", len(scene_mems))

    removed = 0
    downweighted = 0

    for mem in scene_mems:
        accessed = getattr(mem, "access_count", 0) or 0
        if accessed == 0:
            if dry_run:
                logger.info(
                    "  [DRY-RUN] Would remove %s (w=%.3f, payload=%s)",
                    mem.id[:12], mem.weight, mem.payload[:80],
                )
            else:
                memory_storage.remove(mem.id)
                logger.info("  Removed %s (w=%.3f)", mem.id[:12], mem.weight)
            removed += 1
        else:
            target = 0.10
            if mem.weight > target:
                factor = target / mem.weight
                if dry_run:
                    logger.info(
                        "  [DRY-RUN] Would downweight %s: %.3f -> %.3f (accessed %d times)",
                        mem.id[:12], mem.weight, target, accessed,
                    )
                else:
                    memory_storage.downweight(mem.id, weight_factor=factor, decay_rate_factor=1.0)
                    logger.info(
                        "  Downweighted %s: %.3f -> %.3f (accessed %d times)",
                        mem.id[:12], mem.weight, target, accessed,
                    )
                downweighted += 1
            else:
                logger.info(
                    "  Skipping %s: weight %.3f already <= %.3f (accessed %d times)",
                    mem.id[:12], mem.weight, target, accessed,
                )

    if not dry_run and (removed or downweighted):
        if mp.save():
            logger.info("Persisted memory store to disk")
        else:
            logger.error("FAILED to persist memory store!")

    return removed, downweighted


def main() -> None:
    parser = argparse.ArgumentParser(description="Scene observation memory cleanup")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Scene Observation Memory Cleanup")
    logger.info("Mode: %s", "DRY RUN" if args.dry_run else "LIVE")
    logger.info("=" * 60)

    removed, downweighted = cleanup_scene_observations(dry_run=args.dry_run)

    logger.info("")
    logger.info("Summary: %d removed, %d downweighted", removed, downweighted)
    if args.dry_run:
        logger.info("(No changes made — pass without --dry-run to apply)")


if __name__ == "__main__":
    main()
