"""Intent classification for the Capability Acquisition Pipeline.

Routes user requests into outcome classes that determine which lanes activate.
Uses existing SkillResolver templates first, then keyword/pattern heuristics,
then LLM fallback.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    outcome_class: str          # knowledge_only | skill_creation | plugin_creation | core_upgrade | specialist_nn | hardware_integration | mixed
    confidence: float           # 0.0–1.0
    required_lanes: list[str]
    risk_tier: int              # 0–3
    reasoning: str = ""
    classified_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Lane sets per outcome class
# ---------------------------------------------------------------------------

_LANE_MAP: dict[str, list[str]] = {
    "knowledge_only": ["evidence_grounding", "truth"],
    "skill_creation": ["evidence_grounding", "doc_resolution", "planning", "implementation",
                       "skill_registration", "verification", "deployment", "truth"],
    "plugin_creation": ["evidence_grounding", "doc_resolution", "planning", "plan_review",
                        "implementation", "environment_setup", "plugin_quarantine",
                        "verification", "plugin_activation", "deployment", "truth"],
    "core_upgrade": ["evidence_grounding", "doc_resolution", "planning", "plan_review",
                     "implementation", "self_improve", "verification", "deployment", "truth"],
    "specialist_nn": ["evidence_grounding", "planning", "matrix_specialist", "verification", "truth"],
    "hardware_integration": ["evidence_grounding", "doc_resolution", "planning", "plan_review",
                             "implementation", "verification", "deployment", "truth"],
    "mixed": ["evidence_grounding", "doc_resolution", "planning", "plan_review",
              "implementation", "verification", "deployment", "truth"],
}

_RISK_MAP: dict[str, int] = {
    "knowledge_only": 0,
    "skill_creation": 1,
    "plugin_creation": 2,
    "core_upgrade": 2,
    "specialist_nn": 1,
    "hardware_integration": 3,
    "mixed": 2,
}


# ---------------------------------------------------------------------------
# Pattern banks
# ---------------------------------------------------------------------------

_KNOWLEDGE_PATTERNS = [
    re.compile(r"\b(?:what|how|why|explain|tell me about|describe|summarize)\b.*\b(?:work|mean|do|is)\b", re.I),
    re.compile(r"\b(?:learn about|understand|research)\b", re.I),
]

_PLUGIN_PATTERNS = [
    re.compile(r"\b(?:build|create|make|write|develop|implement)\b.*\b(?:tool|plugin|service|scraper|bot|connector|adapter|integration)\b", re.I),
    re.compile(r"\b(?:scrape|crawl|extract|fetch|pull data from)\b.*\b(?:web|url|page|site|api)\b", re.I),
]

_CORE_UPGRADE_PATTERNS = [
    re.compile(r"\b(?:improve yourself|fix your|upgrade your|modify your|change your|update your)\b", re.I),
    re.compile(r"\b(?:self[- ]improve|self[- ]modify|self[- ]upgrade)\b", re.I),
]

_HARDWARE_PATTERNS = [
    re.compile(r"\b(?:gpio|relay|motor|servo|sensor|actuator|robot|robotic|pi5|raspberry|arm control)\b", re.I),
    re.compile(r"\b(?:hardware|board|pin|driver|i2c|spi|uart)\b", re.I),
]

_SKILL_PATTERNS = [
    re.compile(r"\b(?:learn (?:how )?to|teach yourself|acquire the (?:skill|ability))\b", re.I),
    re.compile(r"\b(?:learn|train)\b.*\b(?:skill|ability|capability)\b", re.I),
]

_SPECIALIST_PATTERNS = [
    re.compile(r"\b(?:train a|create a|build a)\b.*\b(?:specialist|neural|network|nn|model|classifier|detector)\b", re.I),
    re.compile(r"\b(?:matrix protocol|matrix specialist)\b", re.I),
]


# ---------------------------------------------------------------------------
# IntentClassifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """Classifies user requests into outcome classes for acquisition routing.

    Classification priority:
      1. Existing SkillResolver templates — if a template matches, we know the type.
      2. Keyword/pattern heuristics — regex banks for each outcome class.
      3. LLM fallback — structured output classification (future).
    """

    def classify(self, user_text: str, context: dict[str, Any] | None = None) -> ClassificationResult:
        """Classify a user request into an outcome class."""
        text = user_text.strip()
        if not text:
            return ClassificationResult(
                outcome_class="knowledge_only",
                confidence=0.0,
                required_lanes=_LANE_MAP["knowledge_only"],
                risk_tier=0,
                reasoning="empty input",
            )

        # 1. Check SkillResolver templates
        sr_result = self._check_skill_resolver(text)
        if sr_result:
            return sr_result

        # 2. Pattern-based classification
        return self._classify_by_patterns(text)

    def _check_skill_resolver(self, text: str) -> ClassificationResult | None:
        """Check if existing SkillResolver has a matching template."""
        try:
            from skills.resolver import SkillResolver
            resolver = SkillResolver()
            resolution = resolver.resolve(text)
            if resolution and resolution.skill_id and resolution.skill_id != "unknown":
                if resolution.capability_type in ("perceptual", "control"):
                    outcome = "skill_creation"
                else:
                    outcome = "skill_creation"
                return ClassificationResult(
                    outcome_class=outcome,
                    confidence=0.85,
                    required_lanes=_LANE_MAP[outcome],
                    risk_tier=_RISK_MAP[outcome],
                    reasoning=f"SkillResolver matched: {resolution.skill_id}",
                )
        except Exception:
            pass
        return None

    def _classify_by_patterns(self, text: str) -> ClassificationResult:
        """Pattern-based classification using regex banks."""
        scores: dict[str, float] = {
            "knowledge_only": 0.0,
            "plugin_creation": 0.0,
            "core_upgrade": 0.0,
            "hardware_integration": 0.0,
            "skill_creation": 0.0,
            "specialist_nn": 0.0,
        }

        for p in _KNOWLEDGE_PATTERNS:
            if p.search(text):
                scores["knowledge_only"] += 0.4

        for p in _PLUGIN_PATTERNS:
            if p.search(text):
                scores["plugin_creation"] += 0.5

        for p in _CORE_UPGRADE_PATTERNS:
            if p.search(text):
                scores["core_upgrade"] += 0.6

        for p in _HARDWARE_PATTERNS:
            if p.search(text):
                scores["hardware_integration"] += 0.5

        for p in _SKILL_PATTERNS:
            if p.search(text):
                scores["skill_creation"] += 0.4

        for p in _SPECIALIST_PATTERNS:
            if p.search(text):
                scores["specialist_nn"] += 0.5

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = min(scores[best], 1.0)

        if confidence < 0.2:
            best = "knowledge_only"
            confidence = 0.3
            reasoning = "no strong pattern match — defaulting to knowledge"
        else:
            reasoning = f"pattern match: {best} (score={confidence:.2f})"

        return ClassificationResult(
            outcome_class=best,
            confidence=confidence,
            required_lanes=_LANE_MAP[best],
            risk_tier=_RISK_MAP[best],
            reasoning=reasoning,
        )

    def get_status(self) -> dict[str, Any]:
        return {
            "available": True,
            "pattern_banks": {
                "knowledge": len(_KNOWLEDGE_PATTERNS),
                "plugin": len(_PLUGIN_PATTERNS),
                "core_upgrade": len(_CORE_UPGRADE_PATTERNS),
                "hardware": len(_HARDWARE_PATTERNS),
                "skill": len(_SKILL_PATTERNS),
                "specialist": len(_SPECIALIST_PATTERNS),
            },
        }
