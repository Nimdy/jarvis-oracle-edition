"""Layer 3B Display Classifier — separates monitor pixels from room reality.

A monitor is a physical object in the room, but what it displays is NOT
physical reality. "Game enemy on screen" must never become "enemy detected
in room." This module:

1. Identifies display surface entities (tv, laptop, monitor)
2. Masks display-interior detections from physical tracking (done in SceneTracker)
3. Classifies display content from VLM descriptions using keyword heuristics
4. Produces DisplayContentSummary with activity labels

All display-derived observations use provenance="model_inference" and
memory type "contextual_insight", never "observation".
"""

from __future__ import annotations

import re
import time
from typing import Any

from perception.scene_types import (
    DISPLAY_SURFACE_LABELS,
    ContentType,
    DisplayContentSummary,
    DisplaySurface,
)

_CONTENT_PATTERNS: list[tuple[ContentType, list[str]]] = [
    ("game", [
        r"\bgame\b", r"\bgaming\b", r"\bhud\b", r"\bcrosshair\b",
        r"\bminimap\b", r"\bhealth.?bar\b", r"\binventory\b",
        r"\bfps\b", r"\brespawn\b", r"\bammo\b", r"\bscore.?board\b",
        r"\bcontroller\b.*\bscreen\b", r"\bvideo.?game\b",
    ]),
    ("code_editor", [
        r"\bcode\b", r"\bcoding\b", r"\beditor\b", r"\bide\b",
        r"\bsyntax\b", r"\bfunction\b", r"\bvariable\b",
        r"\bimport\b", r"\bclass\b.*\bdef\b", r"\bvscode\b",
        r"\bsource.?code\b", r"\bprogramming\b",
    ]),
    ("terminal", [
        r"\bterminal\b", r"\bconsole\b", r"\bshell\b", r"\bcommand.?line\b",
        r"\bprompt\b.*\$", r"\bbash\b", r"\bcli\b",
    ]),
    ("browser", [
        r"\bbrowser\b", r"\bwebsite\b", r"\bweb.?page\b", r"\burl\b",
        r"\baddress.?bar\b", r"\btab\b.*\bbrowser\b", r"\bchrome\b",
        r"\bfirefox\b", r"\bsearch.?engine\b",
    ]),
    ("video", [
        r"\bvideo\b", r"\bmovie\b", r"\bstreaming\b", r"\byoutube\b",
        r"\bplayer\b.*\bcontrols?\b", r"\bsubtitles?\b", r"\btimeline\b",
        r"\bplayback\b", r"\bcinematic\b",
    ]),
    ("chat_app", [
        r"\bchat\b", r"\bmessag(e|ing)\b", r"\bdiscord\b", r"\bslack\b",
        r"\bteams\b", r"\bconversation\b.*\bwindow\b",
    ]),
    ("document", [
        r"\bdocument\b", r"\bword\b", r"\bspreadsheet\b", r"\bpdf\b",
        r"\bparagraph\b", r"\btext.?editor\b", r"\bwriting\b",
    ]),
    ("dashboard", [
        r"\bdashboard\b", r"\bmetrics?\b", r"\bgraphs?\b", r"\bcharts?\b",
        r"\bmonitor(ing)?\b.*\bpanel\b", r"\bstatus\b.*\bboard\b",
    ]),
]

_ACTIVITY_MAP: dict[ContentType, str] = {
    "game": "gaming",
    "code_editor": "coding",
    "terminal": "coding",
    "browser": "browsing",
    "video": "watching",
    "chat_app": "chatting",
    "document": "reading",
    "dashboard": "monitoring",
    "mixed_ui": "",
    "unknown": "",
}


class DisplayClassifier:
    """Classifies display content from VLM scene descriptions."""

    def __init__(self) -> None:
        self._compiled: list[tuple[ContentType, list[re.Pattern]]] = [
            (ct, [re.compile(p, re.IGNORECASE) for p in patterns])
            for ct, patterns in _CONTENT_PATTERNS
        ]
        self._last_classifications: dict[str, DisplayContentSummary] = {}

    def classify_from_description(
        self,
        description: str,
        surfaces: list[DisplaySurface],
    ) -> list[DisplayContentSummary]:
        """Classify display content from a VLM text description.

        Returns one DisplayContentSummary per known display surface.
        If no surfaces are known, returns at most one summary for
        the dominant content type detected in the description.
        """
        if not description:
            return []

        desc_lower = description.lower()
        content_type, confidence = self._classify_text(desc_lower)

        now = time.time()
        results: list[DisplayContentSummary] = []

        if surfaces:
            for ds in surfaces:
                summary = DisplayContentSummary(
                    surface_id=ds.surface_id,
                    observed_at=now,
                    content_type=content_type,
                    confidence=confidence,
                    activity_label=_ACTIVITY_MAP.get(content_type, ""),
                    activity_confidence=confidence * 0.9,
                    semantic_summary=self._build_semantic_summary(content_type, description),
                )
                results.append(summary)
                self._last_classifications[ds.surface_id] = summary
        elif content_type != "unknown":
            summary = DisplayContentSummary(
                surface_id="inferred",
                observed_at=now,
                content_type=content_type,
                confidence=confidence * 0.7,
                activity_label=_ACTIVITY_MAP.get(content_type, ""),
                activity_confidence=confidence * 0.6,
                semantic_summary=self._build_semantic_summary(content_type, description),
            )
            results.append(summary)

        return results

    def get_last_classifications(self) -> dict[str, DisplayContentSummary]:
        return dict(self._last_classifications)

    def _classify_text(self, text: str) -> tuple[ContentType, float]:
        scores: dict[ContentType, float] = {}
        for ct, patterns in self._compiled:
            hits = sum(1 for p in patterns if p.search(text))
            if hits > 0:
                scores[ct] = min(0.95, 0.3 + hits * 0.15)

        if not scores:
            return "unknown", 0.0

        best_ct = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best_ct, scores[best_ct]

    @staticmethod
    def _build_semantic_summary(content_type: ContentType, description: str) -> str:
        if content_type == "unknown":
            return ""
        activity = _ACTIVITY_MAP.get(content_type, content_type)
        label = f"{content_type.replace('_', ' ')} content"
        if activity:
            return f"Display shows {label} ({activity} activity)"
        return f"Display shows {label}"
