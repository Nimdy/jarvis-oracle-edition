"""OSV voice distillation SEED (shadow) — accumulate faithful (grounded -> warm) TEACHER pairs that a
future NATIVE voice NN will distill from, so JARVIS grows into her own voice instead of leaning on the
baseline LLM. See the native-NN-voice directive: the baseline LLM is the TEACHER, never the destination.

Routed through the CANONICAL hemisphere DistillationCollector (teacher="native_voice") rather than a
bespoke corpus — the NN-fleet audit found that reinventing capture/dedup/quarantine/rotation was
duplication. By recording into the collector we inherit, for free: time-bucket dedup, fidelity
quarantine, lived/synthetic split, per-teacher ring buffer, JSONL persistence + rotation, and
get_stats() — the same machinery every other specialist uses, and where the native_voice student will
read its training batch.

How it stays honest:
  * The live self-description is unchanged — the deterministic grounded text is what's spoken. The
    baseline LLM NEVER speaks live here; it only generates training targets (teacher).
  * The teacher re-voice runs through the revoice firewall (no invented numbers / no confab / no
    refusal); ONLY verified-faithful pairs are recorded, so the corpus carries zero confabulation.
  * Never writes a memory. Never raises.
"""

from __future__ import annotations

import logging
from typing import Any

from cognition.self_view.revoice import revoice_self_view

logger = logging.getLogger(__name__)

TEACHER = "native_voice"
SIGNAL_TYPE = "grounded_to_voiced"

# Since-boot teacher fidelity tally (in-memory; the NN lab reads this). Bounded by construction.
_fidelity: dict[str, Any] = {"attempts": 0, "logged": 0, "rejected": {}}


def _collector():
    try:
        from hemisphere.distillation import distillation_collector
        return distillation_collector
    except Exception:
        logger.debug("distillation collector unavailable", exc_info=True)
        return None


async def capture_teacher_pair(grounded_text: str, kind: str, llm_client: Any,
                               *, persona_hint: str = "") -> dict[str, Any]:
    """Best-effort: generate a VERIFIED teacher re-voice and record it as a native_voice teacher
    signal in the canonical DistillationCollector. Returns a status dict. Never raises; the live
    reply is unaffected (this is shadow)."""
    status: dict[str, Any] = {"logged": False, "reason": ""}
    try:
        col = _collector()
        if not grounded_text or llm_client is None or col is None:
            status["reason"] = "no_input_or_client_or_collector"
            return status

        _fidelity["attempts"] += 1
        voiced, meta = await revoice_self_view(grounded_text, llm_client, persona_hint=persona_hint)
        if not meta.get("used_revoice"):
            r = str(meta.get("reason", "?")).split(":", 1)[0]
            _fidelity["rejected"][r] = _fidelity["rejected"].get(r, 0) + 1
            status["reason"] = f"teacher_rejected:{meta.get('reason')}"
            return status

        col.record(
            teacher=TEACHER,
            signal_type=SIGNAL_TYPE,
            data={"kind": kind, "grounded": grounded_text, "voiced": voiced},
            metadata={"persona": persona_hint},
            origin="conversation",   # real self-view turn -> lived (not synthetic)
            fidelity=1.0,            # passed the revoice firewall
        )
        _fidelity["logged"] += 1
        status["logged"] = True
        status["reason"] = "ok"
        return status
    except Exception as e:
        status["reason"] = f"error:{type(e).__name__}"
        logger.debug("voice-seed capture failed", exc_info=True)
        return status


def by_kind() -> dict[str, int]:
    counts: dict[str, int] = {}
    col = _collector()
    if col is None:
        return counts
    try:
        for sig in col.get_recent_signals(TEACHER, n=500):
            k = sig.get("kind", "?")
            counts[k] = counts.get(k, 0) + 1
    except Exception:
        pass
    return counts


def recent_pairs(n: int = 8) -> list[dict[str, Any]]:
    """Most recent captured (grounded -> warm) pairs, truncated for display. Read-only."""
    out: list[dict[str, Any]] = []
    col = _collector()
    if col is None:
        return out
    try:
        for sig in col.get_recent_signals(TEACHER, n=n):
            out.append({"ts": sig.get("ts"), "kind": sig.get("kind"),
                        "grounded": (sig.get("grounded") or "")[:260],
                        "voiced": (sig.get("voiced") or "")[:420]})
    except Exception:
        pass
    return out


def seed_stats() -> dict[str, Any]:
    """Read-only observability for the native_voice teacher corpus (what the NN lab reads)."""
    col = _collector()
    teacher_stats: dict[str, Any] = {}
    if col is not None:
        try:
            teacher_stats = (col.get_stats() or {}).get("teachers", {}).get(TEACHER, {}) or {}
        except Exception:
            pass
    att, logged = _fidelity["attempts"], _fidelity["logged"]
    return {
        "teacher": TEACHER,
        "via": "hemisphere.DistillationCollector",
        "entries": teacher_stats.get("total", 0),
        "lived": teacher_stats.get("lived", 0),
        "synthetic": teacher_stats.get("synthetic", 0),
        "quarantined": teacher_stats.get("quarantined", 0),
        "by_kind": by_kind(),
        "fidelity": {
            "attempts": att,
            "logged": logged,
            "rejected": dict(_fidelity["rejected"]),
            "pass_rate": round(logged / att, 3) if att else None,
        },
    }
