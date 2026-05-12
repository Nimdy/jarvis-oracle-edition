"""Guarded runtime-consumption policy for language promotion levels.

Phase D runtime bridge invariants:
- feature-flagged and default OFF
- read-only promotion access (`get_level` / `get_summary`)
- no gate evaluation on request hot path
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_VALID_MODES = {"off", "canary", "full"}
_PROMOTED_LEVELS = {"canary", "live"}


@dataclass(frozen=True)
class RuntimeLanguagePolicy:
    enabled: bool = False
    rollout_mode: str = "off"
    canary_classes: frozenset[str] = frozenset()


def _normalize_mode(mode: str) -> str:
    m = str(mode or "off").strip().lower()
    return m if m in _VALID_MODES else "off"


def load_runtime_language_policy() -> RuntimeLanguagePolicy:
    """Load runtime bridge policy from config with fail-closed defaults."""
    try:
        from config import BrainConfig

        cfg = BrainConfig()
        rt = getattr(cfg, "language_runtime", None)
        if rt is None:
            return RuntimeLanguagePolicy()
        enabled = bool(getattr(rt, "enable_promotion_bridge", False))
        rollout_mode = _normalize_mode(str(getattr(rt, "rollout_mode", "off") or "off"))
        canary_raw = getattr(rt, "canary_classes", ()) or ()
        canary = frozenset(str(c).strip() for c in canary_raw if str(c).strip())
        return RuntimeLanguagePolicy(
            enabled=enabled,
            rollout_mode=rollout_mode,
            canary_classes=canary,
        )
    except Exception:
        return RuntimeLanguagePolicy()


def _get_promotion_summary_row(response_class: str) -> dict[str, Any]:
    try:
        from jarvis_eval.language_promotion import LanguagePromotionGovernor

        summary = LanguagePromotionGovernor.get_instance().get_summary()
        row = summary.get(response_class, {}) if isinstance(summary, dict) else {}
        return row if isinstance(row, dict) else {}
    except Exception:
        return {}


def get_promotion_level(response_class: str) -> str:
    """Read class promotion level without mutating promotion state."""
    try:
        from jarvis_eval.language_promotion import LanguagePromotionGovernor

        level = str(LanguagePromotionGovernor.get_instance().get_level(response_class) or "shadow")
    except Exception:
        level = "shadow"
    if level not in {"shadow", "canary", "live"}:
        return "shadow"
    return level


def decide_runtime_consumption(
    response_class: str,
    *,
    native_candidate: bool,
    strict_native: bool = False,
    policy: RuntimeLanguagePolicy | None = None,
) -> dict[str, Any]:
    """Decide whether native/bounded output is allowed this turn."""
    cls = str(response_class or "unknown")
    pol = policy or load_runtime_language_policy()
    level = get_promotion_level(cls)
    row = _get_promotion_summary_row(cls)
    gate_color = str(row.get("color", "") or "")

    allow_native = bool(native_candidate)
    reason = "native_not_candidate"

    if strict_native:
        allow_native = bool(native_candidate)
        reason = "strict_native_invariant"
    elif not native_candidate:
        allow_native = False
        reason = "native_not_candidate"
    elif not pol.enabled or pol.rollout_mode == "off":
        allow_native = True
        reason = "bridge_disabled"
    elif pol.rollout_mode == "canary" and cls not in pol.canary_classes:
        allow_native = False
        reason = "class_not_in_canary_rollout"
    elif level in _PROMOTED_LEVELS:
        allow_native = True
        reason = f"promoted_level_{level}"
    else:
        allow_native = False
        reason = f"unpromoted_level_{level}"

    blocked_by_guard = (
        bool(native_candidate)
        and not allow_native
        and not strict_native
        and pol.enabled
        and pol.rollout_mode != "off"
    )

    runtime_live = bool(
        native_candidate
        and allow_native
        and not strict_native
        and pol.enabled
        and pol.rollout_mode != "off"
    )
    unpromoted_live_attempt = bool(
        native_candidate
        and not strict_native
        and pol.enabled
        and pol.rollout_mode != "off"
        and level not in _PROMOTED_LEVELS
    )

    return {
        "response_class": cls,
        "bridge_enabled": bool(pol.enabled),
        "rollout_mode": pol.rollout_mode,
        "canary_classes": sorted(pol.canary_classes),
        "promotion_level": level,
        "gate_color": gate_color,
        "native_candidate": bool(native_candidate),
        "native_allowed": bool(allow_native),
        "strict_native": bool(strict_native),
        "blocked_by_guard": bool(blocked_by_guard),
        "forced_llm": bool(blocked_by_guard),
        "runtime_live": bool(runtime_live),
        "unpromoted_live_attempt": bool(unpromoted_live_attempt),
        "reason": reason,
    }

