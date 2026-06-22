"""Spatial-episodic memory store — the mind's-eye "album" (P5, shadow, zero-authority).

Gives the HRR spatial mental-world lane a DURABLE, revisitable record of the
worlds it has seen — WITHOUT ever creating a path for synthetic spatial guesses
to become canonical memory.

SAFETY CONTRACT (non-negotiable — see docs deep-dive 2026-05-30):
  * ZERO-AUTHORITY shadow store. Every record carries AUTHORITY_FLAGS pinned
    false. It NEVER writes canonical memory, emits MEMORY_WRITE, touches the
    MemoryGate/CueGate, or influences policy/belief/autonomy/identity. It is a
    photo album, not testimony.
  * STRUCTURALLY ISOLATED: stdlib only. It imports NO canonical writers
    (engine.remember, the canonical memory storage/persistence layer, the event bus) and NO
    policy/belief/autonomy/identity/HRR modules. The ONLY bridge from this album
    to canonical memory is the separate, human-reviewed SpatialMemoryGate ->
    engine.remember path, which is intentionally NOT wired here.
  * NEVER stores a raw HRR vector — only the canonical, vector-free scene graph
    (graph.to_dict()) plus a provenance envelope. The HRR world is re-encodable
    losslessly on demand from {graph, dim, seed, vocab_version}; the vector is
    born transient in-process and never hits disk.
  * Capture is GATED UPSTREAM: the HRR shadow only calls capture() when the
    album sub-gate (ENABLE_HRR_SPATIAL_ALBUM, default OFF) AND the P5 twin gate
    are active. This module simply writes what it is handed, defensively
    vector-stripped, and never raises.

Records are append-only JSONL under ~/.jarvis/spatial/episodic/<session>.jsonl.
A restored world is always stamped loaded_from_store=True so it can never be
mistaken for a live observation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger("jarvis.spatial.episodic")

# Mirror cognition.mental_world.AUTHORITY_FLAGS VERBATIM. Kept local so this
# module imports nothing from the HRR lane and stays structurally isolated.
AUTHORITY_FLAGS: Dict[str, bool] = {
    "writes_memory": False,
    "writes_beliefs": False,
    "influences_policy": False,
    "influences_autonomy": False,
    "soul_integrity_influence": False,
    "llm_raw_vector_exposure": False,
    "no_raw_vectors_in_api": True,
}

LANE = "spatial_hrr_mental_world"
STATUS_MARKER = "PRE-MATURE"

# Any key that could carry a raw HRR vector — defensively dropped on write.
_VECTOR_KEYS = frozenset({"vector", "ndarray", "composite", "raw_vector"})

_DEFAULT_BASE = Path.home() / ".jarvis" / "spatial" / "episodic"
_DEFAULT_RETENTION_DAYS = 30
_PRUNE_EVERY = 200  # run retention sweep every N captures (cheap)


def _strip_vectors(obj: Any) -> Any:
    """Recursively drop any key that could carry a raw HRR vector.

    Defensive: the HRR shadow already strips the vector before handing us the
    payload, but the album must NEVER persist a vector under any circumstance.
    """
    if isinstance(obj, dict):
        return {k: _strip_vectors(v) for k, v in obj.items() if k not in _VECTOR_KEYS}
    if isinstance(obj, list):
        return [_strip_vectors(v) for v in obj]
    return obj


# Position quantization for the dedup fingerprint (metres). Coarse enough to
# ignore detector jitter, fine enough that real movement registers as a change.
_POS_QUANTUM_M = 0.5


def _fingerprint(world: Dict[str, Any]) -> str:
    """Content hash of a world's MEANINGFUL structure — entity states + coarse
    positions + relations — ignoring jitter and timestamps.

    Used to skip storing near-duplicate worlds: a static scene must not waste
    the album. Two worlds with the same entities in the same states, the same
    coarse positions, and the same relations collapse to one fingerprint, so a
    motionless office room is stored once, not every sample. Any real change
    (entity appears/vanishes/changes state, moves >~0.5 m, or a relation flips)
    yields a new fingerprint and a fresh stored world.
    """
    ents = []
    for e in (world.get("entities") or []):
        pos = e.get("position_room_m")
        if isinstance(pos, (list, tuple)) and pos:
            try:
                cpos = tuple(round(float(c) / _POS_QUANTUM_M) * _POS_QUANTUM_M for c in pos)
            except (TypeError, ValueError):
                cpos = None
        else:
            cpos = None
        ents.append((e.get("entity_id"), e.get("state"), cpos))
    rels = sorted(
        (r.get("source"), r.get("relation_type"), r.get("target"))
        for r in (world.get("relations") or [])
    )
    key = json.dumps(
        [sorted(ents, key=lambda x: str(x[0])), rels],
        default=str, separators=(",", ":"),
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class SpatialEpisodicStore:
    """Append-only, zero-authority durable store for spatial mental-world graphs."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        session_id: Optional[str] = None,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
    ) -> None:
        self._base = Path(base_dir) if base_dir is not None else _DEFAULT_BASE
        self._session_id = session_id or self._new_session_id()
        self._retention_days = max(1, int(retention_days))
        self._capture_count = 0
        self._deduped = 0              # near-duplicate worlds skipped (waste avoided)
        self._last_fingerprint = None  # content hash of the last STORED world
        self._ensured = False  # lazy: gate-OFF means zero disk touch

    @staticmethod
    def _new_session_id() -> str:
        # Boot-session id: time-ordered + random suffix; distinguishes this run
        # from worlds loaded back from the store.
        return f"s{int(time.time())}_{uuid.uuid4().hex[:8]}"

    @property
    def session_id(self) -> str:
        return self._session_id

    def _session_path(self, session_id: Optional[str] = None) -> Path:
        return self._base / f"{session_id or self._session_id}.jsonl"

    def _ensure_dir(self) -> None:
        if not self._ensured:
            self._base.mkdir(parents=True, exist_ok=True)
            self._ensured = True

    def capture(
        self,
        scene_payload: Dict[str, Any],
        hrr_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Append one OBSERVED-world record. Returns True on write.

        NEVER raises: failing to remember a world must never disturb the tick.
        The caller (HRR shadow) only invokes this when the album sub-gate is on,
        and only with derive_scene_graph OUTPUT (observed worlds) — never with
        imagined navigation graphs.
        """
        try:
            payload = _strip_vectors(dict(scene_payload))
            # Dedup: skip a world that is structurally identical to the last one
            # we STORED (static scene → one record, not a pile of duplicates).
            fp = _fingerprint(payload)
            if fp == self._last_fingerprint:
                self._deduped += 1
                return False
            tick = payload.get("tick")
            src = payload.get("source") if isinstance(payload.get("source"), dict) else {}
            record = {
                "world_id": f"{self._session_id}:{tick}",
                "session_id": self._session_id,
                "captured_ts": round(time.time(), 3),
                "lane": LANE,
                "status": STATUS_MARKER,
                "authority": dict(AUTHORITY_FLAGS),   # self-describes as zero-authority
                "loaded_from_store": False,           # this is a fresh capture, not a replay
                "fingerprint": fp,                    # content hash (dedup + distinct-world id)
                "calibration_version": src.get("calibration_version"),
                "hrr_config": dict(hrr_config) if hrr_config else None,
                "world": payload,                     # the canonical, vector-free graph
            }
            self._ensure_dir()
            line = json.dumps(record, separators=(",", ":"), default=str)
            with open(self._session_path(), "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._last_fingerprint = fp
            self._capture_count += 1
            if self._capture_count % _PRUNE_EVERY == 0:
                self._prune_old()
            return True
        except Exception as exc:  # never fatal
            logger.debug("spatial album capture skipped: %s", exc)
            return False

    # -- read side (read-only; used by later observability stones) -----------

    def list_sessions(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            if not self._base.exists():
                return out
            for p in sorted(self._base.glob("*.jsonl")):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        n = sum(1 for _ in f)
                    out.append({"session_id": p.stem, "records": n, "mtime": p.stat().st_mtime})
                except Exception:
                    continue
        except Exception:
            pass
        return out

    def load_session(self, session_id: str) -> List[Dict[str, Any]]:
        return list(self._iter_records(session_id))

    def load_world(self, world_id: str) -> Optional[Dict[str, Any]]:
        try:
            session_id = world_id.split(":", 1)[0]
        except Exception:
            return None
        for rec in self._iter_records(session_id):
            if rec.get("world_id") == world_id:
                return rec
        return None

    def _iter_records(self, session_id: str) -> Iterator[Dict[str, Any]]:
        path = self._session_path(session_id)
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # skip malformed/partial trailing line (never fatal)
                    rec["loaded_from_store"] = True  # a restored world is NEVER live
                    yield rec
        except Exception:
            return

    def _prune_old(self) -> None:
        try:
            if not self._base.exists():
                return
            cutoff = time.time() - self._retention_days * 86400
            for p in self._base.glob("*.jsonl"):
                try:
                    if p.stat().st_mtime < cutoff:
                        p.unlink()
                except Exception:
                    continue
        except Exception:
            pass


__all__ = ["SpatialEpisodicStore", "AUTHORITY_FLAGS", "LANE", "STATUS_MARKER"]
