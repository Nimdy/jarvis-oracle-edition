#!/usr/bin/env python3
"""Purge hollow consolidation memories and exact duplicates.

Run WHILE JARVIS IS STOPPED to clean up dead-weight memories:
  1. Backs up memories.json before any changes
  2. Removes consolidation memories with no real content after stripping meta-headers
  3. Deduplicates exact payload matches (keeps highest-weight copy)
  4. Reports before/after stats

Usage:
    python scripts/purge_hollow_memories.py              # dry-run (default)
    python scripts/purge_hollow_memories.py --apply       # actually modify
    python scripts/purge_hollow_memories.py --apply --no-backup  # skip backup (not recommended)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

JARVIS_DIR = Path.home() / ".jarvis"
MEMORIES_FILE = JARVIS_DIR / "memories.json"


def strip_meta_headers(text: str) -> str:
    text = re.sub(r'\[Consolidated from \d+ memories\]\s*', '', text)
    text = re.sub(r'\[Dream artifact: [^\]]*\]\s*', '', text)
    text = re.sub(r'Consolidation: \w+\s*\(\d+ memories, coherence=[\d.]+\)\s*', '', text)
    text = re.sub(r'\[\+\d+ more\]\s*', '', text)
    text = re.sub(r'\s*\|\s*', ' ', text).strip()
    return text


def get_text_content(memory: dict) -> str:
    payload = memory.get("payload", "")
    if isinstance(payload, dict):
        return payload.get("text", "") or payload.get("summary", "")
    return str(payload) if payload else ""


def is_hollow_consolidation(memory: dict) -> bool:
    if memory.get("provenance") != "consolidation":
        return False
    text = get_text_content(memory)
    if not text:
        return True
    stripped = strip_meta_headers(text)
    return len(stripped) < 20


def payload_hash(memory: dict) -> str:
    payload = memory.get("payload", "")
    raw = json.dumps(payload, sort_keys=True, default=str) if isinstance(payload, dict) else str(payload)
    return hashlib.md5(raw.encode()).hexdigest()


def analyze(memories: list[dict]) -> dict:
    by_type = {}
    by_prov = {}
    hollow = []
    useful = []

    for m in memories:
        t = m.get("type", "unknown")
        p = m.get("provenance", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
        by_prov[p] = by_prov.get(p, 0) + 1

        if is_hollow_consolidation(m):
            hollow.append(m)
        else:
            useful.append(m)

    hash_groups: dict[str, list[dict]] = {}
    for m in useful:
        h = payload_hash(m)
        if h not in hash_groups:
            hash_groups[h] = []
        hash_groups[h].append(m)

    duplicates = []
    deduplicated = []
    for h, group in hash_groups.items():
        group.sort(key=lambda m: m.get("weight", 0), reverse=True)
        deduplicated.append(group[0])
        duplicates.extend(group[1:])

    weights = [m.get("weight", 0) for m in memories]
    access_counts = [m.get("access_count", 0) for m in memories]

    return {
        "total": len(memories),
        "by_type": dict(sorted(by_type.items(), key=lambda x: -x[1])),
        "by_provenance": dict(sorted(by_prov.items(), key=lambda x: -x[1])),
        "hollow_count": len(hollow),
        "duplicate_count": len(duplicates),
        "kept_count": len(deduplicated),
        "removed_total": len(hollow) + len(duplicates),
        "hollow_ids": [m.get("id", "?") for m in hollow],
        "duplicate_ids": [m.get("id", "?") for m in duplicates],
        "kept_memories": deduplicated,
        "avg_weight": sum(weights) / len(weights) if weights else 0,
        "never_accessed": sum(1 for a in access_counts if a == 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Purge hollow consolidation memories")
    parser.add_argument("--apply", action="store_true", help="Actually modify memories.json")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup (not recommended)")
    args = parser.parse_args()

    if not MEMORIES_FILE.exists():
        print(f"ERROR: {MEMORIES_FILE} not found")
        sys.exit(1)

    with open(MEMORIES_FILE) as f:
        memories = json.load(f)

    if not isinstance(memories, list):
        print("ERROR: memories.json is not a list")
        sys.exit(1)

    print(f"{'=' * 60}")
    print(f"Memory Purge {'(DRY RUN)' if not args.apply else '** APPLYING **'}")
    print(f"{'=' * 60}")
    print(f"Source: {MEMORIES_FILE}")
    print(f"Total memories: {len(memories)}")
    print()

    result = analyze(memories)

    print("=== BEFORE ===")
    print(f"  Total: {result['total']}")
    print(f"  By type: {result['by_type']}")
    print(f"  By provenance: {result['by_provenance']}")
    print(f"  Avg weight: {result['avg_weight']:.3f}")
    print(f"  Never accessed: {result['never_accessed']}")
    print()

    print("=== REMOVALS ===")
    print(f"  Hollow consolidations: {result['hollow_count']}")
    print(f"  Exact duplicates:      {result['duplicate_count']}")
    print(f"  Total to remove:       {result['removed_total']}")
    print()

    print("=== AFTER ===")
    print(f"  Remaining: {result['kept_count']}")
    print(f"  Space freed: {result['removed_total']} slots ({result['removed_total'] * 100 / max(result['total'], 1):.1f}%)")
    print()

    if result['kept_count'] > 0:
        kept_weights = [m.get("weight", 0) for m in result['kept_memories']]
        print(f"  New avg weight: {sum(kept_weights) / len(kept_weights):.3f}")
        kept_types = {}
        for m in result['kept_memories']:
            t = m.get("type", "unknown")
            kept_types[t] = kept_types.get(t, 0) + 1
        print(f"  New type dist: {dict(sorted(kept_types.items(), key=lambda x: -x[1]))}")

    if not args.apply:
        print()
        print("DRY RUN — no changes made. Use --apply to execute.")
        return

    if not args.no_backup:
        backup_path = MEMORIES_FILE.with_suffix(f".backup-{int(time.time())}.json")
        print(f"\nBacking up to {backup_path}")
        shutil.copy2(MEMORIES_FILE, backup_path)

    print(f"\nWriting {result['kept_count']} memories to {MEMORIES_FILE}")
    tmp = MEMORIES_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(result['kept_memories'], f, indent=None, default=str)
    os.replace(str(tmp), str(MEMORIES_FILE))

    print("DONE — purge complete.")
    print(f"Removed {result['removed_total']} memories ({result['hollow_count']} hollow + {result['duplicate_count']} duplicates)")
    print(f"Remaining: {result['kept_count']} memories")


if __name__ == "__main__":
    main()
