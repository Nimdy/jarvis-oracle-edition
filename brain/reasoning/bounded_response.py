"""Deterministic planner/articulator for bounded truth-backed answer classes."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

MAX_ARTICULATE_SENTENCES = 8
MAX_ARTICULATE_CHARS = 600
MAX_ARTICULATE_FACTS = 5


def _clean_lines(text: str, *, limit: int = 8) -> list[str]:
    lines: list[str] = []
    for raw_line in (text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def _humanize_dwell(raw: str) -> str:
    """Turn '184s in current mode' into spoken-friendly phrasing."""
    raw = raw.strip()
    m = re.match(r"(\d+)s", raw)
    if not m:
        return raw
    secs = int(m.group(1))
    if secs < 60:
        return "just switched"
    if secs < 3600:
        mins = secs // 60
        return f"for about {mins} minute{'s' if mins != 1 else ''}"
    hours = secs // 3600
    return f"for about {hours} hour{'s' if hours != 1 else ''}"


def _articulate_self_status(frame: MeaningFrame) -> str:
    """Produce natural spoken-word status from structured facts.

    All data comes from the MeaningFrame — nothing is invented.
    The articulation rephrases key-value data into sentences a person
    would actually say aloud, suitable for TTS delivery.
    """
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []

    for fact in frame.facts:
        fact = fact.strip()
        if not fact:
            continue
        if fact.startswith("===") and fact.endswith("==="):
            if current_title or current_lines:
                sections.append((current_title, current_lines))
            current_title = fact.strip("= ").strip()
            current_lines = []
            continue
        current_lines.append(fact)

    if current_title or current_lines:
        sections.append((current_title, current_lines))

    if not sections:
        return frame.lead.strip()

    rendered: list[str] = []
    mode_value = ""

    for title, lines in sections:
        title_lower = title.lower()
        kv: dict[str, str] = {}
        extras: list[str] = []
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                kv[key.strip().lower()] = value.strip()
            else:
                extras.append(line.strip())

        if "current activity" in title_lower:
            state = kv.get("state", "").strip()
            detail = kv.get("detail", "").strip()
            if state.lower() in ("standing by", "idle", "unknown") and not detail:
                rendered.append("I'm standing by right now, nothing actively processing.")
            elif detail and "tool" in detail.lower():
                rendered.append(f"I'm currently handling a request — {detail.split('tool:', 1)[-1].strip().split('.')[0] if 'tool:' in detail.lower() else detail}.")
            elif state:
                rendered.append(f"Right now I'm {state.lower()}.")
            continue

        if "background operations" in title_lower:
            active_kv = {k: v for k, v in kv.items() if "idle" not in v.lower()}
            active_extras = [e for e in extras if "idle" not in e.lower()]
            if active_kv:
                descriptions = []
                for name, detail in list(active_kv.items())[:3]:
                    clean_name = name.strip().replace("_", " ")
                    status_part = detail.split("—")[0].strip() if "—" in detail else detail.split(",")[0].strip()
                    descriptions.append(f"{clean_name} is {status_part.lower()}")
                rendered.append(f"In the background, {', and '.join(descriptions)}.")
            elif active_extras:
                descriptions = [e.strip().split(":")[0].strip().lower() for e in active_extras[:3]]
                rendered.append(f"In the background, {', '.join(descriptions)} {'is' if len(descriptions) == 1 else 'are'} running.")
            continue

        if "operating mode" in title_lower:
            mode_value = kv.get("mode", "unknown")
            dwell = kv.get("dwell", "")
            dwell_phrase = _humanize_dwell(dwell) if dwell else ""
            if dwell_phrase:
                rendered.append(f"I'm in {mode_value} mode, {dwell_phrase}.")
            else:
                rendered.append(f"I'm in {mode_value} mode.")
            continue

        if "active drives" in title_lower:
            drive_names = []
            for line_raw in extras:
                parts = line_raw.strip().split(":")
                if parts:
                    drive_names.append(parts[0].strip().lower())
            for k in kv:
                drive_names.append(k.strip().lower())
            if drive_names:
                rendered.append(f"My active drives are {', '.join(drive_names[:3])}.")
            continue

        if "learning jobs" in title_lower:
            job_lines = extras + [f"{k}: {v}" for k, v in kv.items()]
            if not job_lines or ("none active" in title_lower):
                continue
            for jl in job_lines[:2]:
                skill = jl.split(":")[0].strip() if ":" in jl else jl
                phase_match = re.search(r"phase=(\w+)", jl)
                phase = phase_match.group(1) if phase_match else ""
                if phase:
                    rendered.append(f"I have a learning job for {skill.replace('_', ' ')}, currently in the {phase} phase.")
                else:
                    rendered.append(f"I have a learning job for {skill.replace('_', ' ')}.")
            continue

        if "cortex training" in title_lower:
            ranker = kv.get("ranker", "")
            ready = kv.get("ready", "no")
            if ranker:
                ranker_clean = ranker.replace("training pairs", "").strip()
                rendered.append(f"My memory cortex has collected {ranker_clean} training pairs{', and is ready to train' if ready.lower() == 'yes' else ''}.")
            continue

        if "emotion sensor" in title_lower:
            status = kv.get("status", "")
            if "healthy" in status.lower():
                continue
            if "degraded" in status.lower():
                rendered.append("My emotion sensor is running in fallback mode right now.")
            continue

        # Generic fallback for unknown sections — still natural phrasing
        if extras or kv:
            clean_title = re.sub(r"\[.*?\]", "", title).strip()
            section_bits = extras[:2] + [f"{v}" for _, v in list(kv.items())[:2]]
            if section_bits:
                rendered.append(f"{clean_title}: {'. '.join(section_bits)}.")

    if not rendered:
        if mode_value:
            return f"I'm in {mode_value} mode. All systems are running normally."
        return "All systems are running normally."

    return " ".join(rendered).strip()


_FACT_RANK_ORDER = [
    "current_state",
    "health",
    "memory",
    "architecture",
    "evolution",
    "mutation",
    "other",
]

_SECTION_CATEGORY: dict[str, str] = {
    "consciousness metrics": "current_state",
    "current activity": "current_state",
    "operating mode": "current_state",
    "analytics": "health",
    "performance": "health",
    "system health": "health",
    "quarantine": "health",
    "architecture": "architecture",
    "document library": "architecture",
    "storage": "architecture",
    "memory": "memory",
    "cortex": "memory",
    "dream": "memory",
    "evolution metrics": "evolution",
    "observer metrics": "evolution",
    "recent thought records": "evolution",
    "self-modifications": "mutation",
    "mutations": "mutation",
}


def _categorize_section(title: str) -> str:
    title_lower = title.lower()
    for key, cat in _SECTION_CATEGORY.items():
        if key in title_lower:
            return cat
    return "other"


def _parse_introspection_sections(
    text: str,
) -> tuple[list[tuple[str, str, list[str]]], list[str]]:
    """Parse === delimited introspection text into (category, title, kv_facts) tuples.

    Returns (sections, parse_warnings).
    """
    sections: list[tuple[str, str, list[str]]] = []
    warnings: list[str] = []
    current_title = ""
    current_lines: list[str] = []

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("===") and line.endswith("==="):
            if current_title and current_lines:
                cat = _categorize_section(current_title)
                sections.append((cat, current_title, current_lines))
            current_title = line.strip("= ").strip()
            current_lines = []
            continue
        if ":" in line and not line.startswith("http"):
            current_lines.append(line)
        elif not line.startswith("  "):
            current_lines.append(line)

    if current_title and current_lines:
        cat = _categorize_section(current_title)
        sections.append((cat, current_title, current_lines))

    if not sections and text and text.strip():
        warnings.append("no_sections_parsed")

    return sections, warnings


def _rank_and_select_facts(
    sections: list[tuple[str, str, list[str]]],
    *,
    max_facts: int = MAX_ARTICULATE_FACTS,
    preferred_categories: list[str] | None = None,
) -> list[str]:
    """Select top facts by category rank order, one pass."""
    by_cat: dict[str, list[str]] = {}
    for cat, _title, lines in sections:
        by_cat.setdefault(cat, []).extend(lines)

    rank_order: list[str] = []
    for cat in preferred_categories or []:
        if cat in by_cat and cat not in rank_order:
            rank_order.append(cat)
    for cat in _FACT_RANK_ORDER:
        if cat not in rank_order:
            rank_order.append(cat)

    selected: list[str] = []
    for cat in rank_order:
        for fact in by_cat.get(cat, []):
            if len(selected) >= max_facts:
                return selected
            selected.append(fact)
    return selected


def _fact_to_sentence(raw: str) -> str:
    """Convert a 'Key: Value' line into a spoken sentence."""
    if ":" not in raw:
        return raw if raw.endswith(".") else f"{raw}."
    key, _, value = raw.partition(":")
    key = key.strip().lstrip("- ")
    value = value.strip()
    if not value:
        return f"{key}." if key else ""

    key_lower = key.lower()
    if key_lower in ("stage", "current stage"):
        return f"I'm at the {value} stage."
    if "confidence" in key_lower:
        return f"My confidence level is {value}."
    if "awareness" in key_lower:
        return f"My awareness level is {value}."
    if "tick" in key_lower and "ms" in value:
        return f"My tick latency is {value}."
    if key_lower in ("total memories",):
        return f"I have {value} memories."
    if "health" in key_lower:
        return f"System health is {value}."
    if "reasoning" in key_lower:
        return f"Reasoning quality is {value}."
    if "mode" in key_lower:
        return f"I'm in {value} mode."
    if "mutations" in key_lower or "mutation" in key_lower:
        return f"Total mutations applied: {value}."

    return f"{key} is {value}."


def _articulate_self_introspection(frame: MeaningFrame) -> str:
    """Produce natural spoken introspection from structured facts.

    All data comes from the MeaningFrame. No inference, no philosophy,
    no hedging filler. Enforces MAX_ARTICULATE_SENTENCES / MAX_ARTICULATE_CHARS.
    """
    if not frame.facts:
        if frame.missing_reason:
            return frame.lead.strip()
        return "I don't have detailed introspection data available right now."

    sentences: list[str] = []
    char_count = 0

    lead = frame.lead.strip()
    if lead:
        sentences.append(lead)
        char_count += len(lead)

    for raw_fact in frame.facts:
        if len(sentences) >= MAX_ARTICULATE_SENTENCES:
            break
        if char_count >= MAX_ARTICULATE_CHARS:
            break
        sentence = _fact_to_sentence(raw_fact)
        if not sentence:
            continue
        if char_count + len(sentence) > MAX_ARTICULATE_CHARS:
            break
        sentences.append(sentence)
        char_count += len(sentence)

    if frame.missing_reason and len(sentences) < MAX_ARTICULATE_SENTENCES:
        note = "Some data is not available yet."
        if char_count + len(note) <= MAX_ARTICULATE_CHARS:
            sentences.append(note)

    return " ".join(sentences).strip()


def _articulate_emergence_evidence(frame: MeaningFrame) -> str:
    """Bounded scientific explanation for emergence/consciousness evidence."""
    summary = frame.metadata.get("summary") if isinstance(frame.metadata, dict) else {}
    levels = frame.metadata.get("levels") if isinstance(frame.metadata, dict) else []
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(levels, list):
        levels = []

    max_level = summary.get("max_supported_level", -1)
    supported = summary.get("supported_levels", 0)
    parts = [
        "The emergence dashboard is operational evidence, not proof of sentience.",
        "Real substrate evidence, not roleplay: it is derived from internal thought records, observer metrics, autonomy, mutation/evolution, prediction, and persistence telemetry.",
        f"Right now the ladder shows {supported} supported or partial levels, with the highest supported level at L{max_level}." if isinstance(max_level, int) and max_level >= 0 else "Right now the ladder has no supported level high enough to claim an anomaly.",
    ]

    compact: list[str] = []
    for level in levels[:8]:
        if not isinstance(level, dict):
            continue
        compact.append(
            f"L{level.get('level')}: {level.get('status', 'unknown')}"
        )
    if compact:
        parts.append("Current ladder: " + ", ".join(compact) + ".")
    parts.append("Level 7 is not claimed; it stays empty unless an event survives known-mechanism elimination such as templates, LLM prompt context, hardcoded detector rules, metric thresholds, and direct user prompting.")
    return " ".join(parts)


def _relative_time(timestamp: float) -> str:
    if not timestamp:
        return "recently"
    delta = max(0.0, time.time() - float(timestamp))
    if delta < 60:
        return "just now"
    if delta < 3600:
        mins = int(delta // 60)
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    if delta < 86400:
        hours = int(delta // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = int(delta // 86400)
    return f"{days} day{'s' if days != 1 else ''} ago"


@dataclass
class MeaningFrame:
    response_class: str
    lead: str
    facts: list[str] = field(default_factory=list)
    missing_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    safety_flags: list[str] = field(default_factory=list)
    frame_confidence: float = 0.0
    fact_count: int = 0
    section_count: int = 0
    parse_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_structurally_healthy(self) -> bool:
        return (
            self.frame_confidence >= 0.4
            and not any("critical" in w for w in self.parse_warnings)
        )


def build_meaning_frame(
    *,
    response_class: str,
    grounding_payload: Any,
    preferred_categories: list[str] | None = None,
) -> MeaningFrame:
    if response_class == "self_status":
        lines = _clean_lines(str(grounding_payload or ""), limit=16)
        if not lines:
            return MeaningFrame(
                response_class=response_class,
                lead="I don't have grounded status facts available right now.",
                missing_reason="missing_status_payload",
                safety_flags=["fail_closed_when_missing"],
            )
        open_count = 0
        oldest_age_s: float | None = None
        try:
            from cognition.intention_resolver import get_intention_resolver
            _resolver = get_intention_resolver()
            _rstatus = _resolver.get_status()
            _rtotal = _rstatus.get("total_evaluated", 0)
            if _rtotal > 0:
                facts.append(f"Resolver shadow verdicts evaluated: {_rtotal}")
                _rmetrics = _rstatus.get("shadow_metrics", {})
                if _rmetrics.get("sufficient_data"):
                    facts.append(f"Resolver shadow accuracy: {_rmetrics.get('shadow_accuracy', 0):.1%}")
        except Exception:
            pass
        try:
            from cognition.intention_registry import intention_registry
            snap = intention_registry.get_status()
            open_count = int(snap.get("open_count", 0) or 0)
            oldest_age_s = snap.get("most_recent_open_intention_age_s")
        except Exception:
            open_count = 0
            oldest_age_s = None

        if open_count > 0:
            lines.append(f"Open intentions: {open_count}")
            if isinstance(oldest_age_s, (int, float)) and oldest_age_s >= 0:
                age_s = float(oldest_age_s)
                if age_s < 60:
                    age_txt = f"{int(age_s)}s"
                elif age_s < 3600:
                    age_txt = f"{int(age_s / 60)}m"
                elif age_s < 86400:
                    age_txt = f"{age_s / 3600:.1f}h"
                else:
                    age_txt = f"{age_s / 86400:.1f}d"
                lines.append(f"Most recent open intention age: {age_txt}")

        return MeaningFrame(
            response_class=response_class,
            lead="Here is my current measured status.",
            facts=lines,
            metadata={
                "line_count": len(lines),
                "open_intentions_count": open_count,
                "most_recent_open_intention_age_s": (
                    float(oldest_age_s) if isinstance(oldest_age_s, (int, float)) else None
                ),
            },
            safety_flags=["status_mode"],
        )

    if response_class == "memory_recall":
        payload = grounding_payload if isinstance(grounding_payload, dict) else {}
        mode = str(payload.get("mode", "summary"))
        memory_context = str(payload.get("memory_context", "") or "")
        lines = _clean_lines(memory_context, limit=8)
        if not lines:
            return MeaningFrame(
                response_class=response_class,
                lead="I couldn't verify any matching memory details right now.",
                missing_reason="missing_memory_context",
                metadata={"mode": mode},
                safety_flags=["fail_closed_when_missing"],
            )
        lead = (
            "Here are the memory details I could verify."
            if mode == "search"
            else "Here is my current memory summary."
        )
        return MeaningFrame(
            response_class=response_class,
            lead=lead,
            facts=lines,
            metadata={"mode": mode, "line_count": len(lines)},
        )

    if response_class in {"recent_learning", "recent_research"}:
        record = grounding_payload if isinstance(grounding_payload, dict) else {}
        kind = str(record.get("kind", ""))
        if kind == "missing_scholarly":
            return MeaningFrame(
                response_class=response_class,
                lead="I don't have a verified recent peer-reviewed source record yet.",
                missing_reason="missing_scholarly",
                safety_flags=["fail_closed_when_missing"],
            )
        if kind == "missing_research":
            return MeaningFrame(
                response_class=response_class,
                lead="I don't have a verified recent research record yet.",
                missing_reason="missing_research",
                safety_flags=["fail_closed_when_missing"],
            )
        if kind == "missing_learning":
            return MeaningFrame(
                response_class=response_class,
                lead="I don't have a verified recent learning record yet.",
                missing_reason="missing_learning",
                safety_flags=["fail_closed_when_missing"],
            )

        when = _relative_time(float(record.get("timestamp", 0.0) or 0.0))

        if kind == "scholarly_source":
            title = str(record.get("title", "") or "untitled source")
            venue = str(record.get("venue", "") or "")
            year = record.get("year", 0)
            doi = str(record.get("doi", "") or "")
            include_doi = bool(record.get("include_doi", False))
            facts = [f'Title: "{title}"', f"Studied: {when}"]
            if venue and year:
                facts.append(f"Venue: {venue} ({year})")
            elif venue:
                facts.append(f"Venue: {venue}")
            elif year:
                facts.append(f"Year: {year}")
            if doi and include_doi:
                facts.append(f"DOI: {doi}")
            return MeaningFrame(
                response_class=response_class,
                lead="The most recent peer-reviewed source I can verify is below.",
                facts=facts,
                metadata={"kind": kind, "title": title},
                safety_flags=["strict_provenance_grounded"],
            )

        if kind == "source":
            title = str(record.get("title", "") or "untitled source")
            source_type = str(record.get("source_type", "") or "source")
            return MeaningFrame(
                response_class=response_class,
                lead="The most recent source I can verify studying is below.",
                facts=[
                    f'Title: "{title}"',
                    f"Type: {source_type}",
                    f"Studied: {when}",
                ],
                metadata={"kind": kind, "title": title},
                safety_flags=["strict_provenance_grounded"],
            )

        if kind == "autonomy_research":
            question = str(record.get("question", "") or "unknown question")
            summary = str(record.get("summary", "") or "")
            tool = str(record.get("tool", "") or "unknown tool")
            facts = [
                f'Question: "{question}"',
                f"Tool: {tool}",
                f"Recorded: {when}",
            ]
            if summary:
                facts.append(f"Summary: {summary}")
            return MeaningFrame(
                response_class=response_class,
                lead="The latest research record I can verify is below.",
                facts=facts,
                metadata={"kind": kind, "tool": tool},
                safety_flags=["strict_provenance_grounded"],
            )

        if kind in {"conversation_memory", "conversation_correction", "user_preference"}:
            text = str(record.get("text", "") or "")
            memory_type = str(record.get("memory_type", "") or "")
            provenance = str(record.get("provenance", "") or "")
            friction_type = str(record.get("friction_type", "") or "")
            facts = [f"Recorded: {when}"]
            if text:
                facts.insert(0, f'Learning record: "{text}"')
            if memory_type:
                facts.append(f"Memory type: {memory_type}")
            if provenance:
                facts.append(f"Provenance: {provenance}")
            if friction_type:
                facts.append(f"Friction type: {friction_type}")

            lead_by_kind = {
                "conversation_memory": "The latest conversation-learning record I can verify is below.",
                "conversation_correction": "The latest correction-linked learning record I can verify is below.",
                "user_preference": "The latest preference-learning record I can verify is below.",
            }
            return MeaningFrame(
                response_class=response_class,
                lead=lead_by_kind[kind],
                facts=facts,
                metadata={"kind": kind},
                safety_flags=["strict_provenance_grounded"],
            )

        if kind == "learning_job":
            skill_id = str(record.get("skill_id", "") or "unknown_skill")
            phase = str(record.get("phase", "") or "unknown")
            status = str(record.get("status", "") or "unknown")
            return MeaningFrame(
                response_class=response_class,
                lead="The latest completed learning-job record I can verify is below.",
                facts=[
                    f"Skill: {skill_id}",
                    f"Status: {status}",
                    f"Phase: {phase}",
                    f"Updated: {when}",
                ],
                metadata={"kind": kind, "skill_id": skill_id},
                safety_flags=["strict_provenance_grounded"],
            )

        return MeaningFrame(
            response_class=response_class,
            lead="I don't have a verified recent learning record yet.",
            missing_reason="unrecognized_recent_learning_kind",
            safety_flags=["fail_closed_when_missing"],
        )

    if response_class == "learning_job_status":
        record = grounding_payload if isinstance(grounding_payload, dict) else {}
        kind = str(record.get("kind", ""))
        if kind == "missing_learning_job_status":
            return MeaningFrame(
                response_class=response_class,
                lead="I don't have a matching active learning-job status record yet.",
                missing_reason="missing_learning_job_status",
                safety_flags=["fail_closed_when_missing"],
            )

        if kind == "learning_job_help_summary":
            jobs = list(record.get("jobs", []) or [])
            active_count = int(record.get("active_job_count", len(jobs)) or len(jobs))
            suggestions = list(record.get("suggested_user_inputs", []) or [])
            facts = [f"Active learning jobs: {active_count}"]
            for job in jobs[:3]:
                skill_id = str(job.get("skill_id", "") or "unknown_skill")
                phase = str(job.get("phase", "") or "unknown")
                status = str(job.get("status", "") or "unknown")
                blocker = str(job.get("blocker_summary", "") or "")
                facts.append(f"{skill_id}: phase={phase}, status={status}")
                if blocker:
                    facts.append(f"{skill_id} blocker: {blocker}")
            for suggestion in suggestions[:2]:
                facts.append(f"User input: {suggestion}")

            return MeaningFrame(
                response_class=response_class,
                lead=f"I can verify {active_count} active learning jobs right now.",
                facts=facts,
                metadata={"kind": kind, "active_job_count": active_count},
                safety_flags=["strict_provenance_grounded", "learning_job_status"],
            )

        skill_id = str(record.get("skill_id", "") or "unknown_skill")
        phase = str(record.get("phase", "") or "unknown")
        status = str(record.get("status", "") or "unknown")
        blocker = str(record.get("blocker_summary", "") or "")
        artifact_count = int(record.get("artifact_count", 0) or 0)
        evidence_count = int(record.get("evidence_count", 0) or 0)
        phase_age_s = float(record.get("phase_age_s", 0.0) or 0.0)
        matrix_protocol = bool(record.get("matrix_protocol", False))
        protocol_id = str(record.get("protocol_id", "") or "")
        claimability_status = str(record.get("claimability_status", "") or "")
        current_metric = float(record.get("current_metric", 0.0) or 0.0)
        target_metric = float(record.get("target_metric", 0.0) or 0.0)
        metric_name = str(record.get("metric_name", "") or "")
        suggestions = list(record.get("suggested_user_inputs", []) or [])

        facts = [
            f"Skill: {skill_id}",
            f"Status: {status}",
            f"Phase: {phase}",
            f"Artifacts: {artifact_count}",
            f"Evidence records: {evidence_count}",
            f"Phase age: {int(phase_age_s)}s",
        ]
        if matrix_protocol:
            facts.append(f"Matrix protocol: {protocol_id or 'active'}")
        if claimability_status:
            facts.append(f"Claimability: {claimability_status}")
        if blocker:
            facts.append(f"Blocker: {blocker}")
        if metric_name and target_metric > 0:
            facts.append(f"Progress: {metric_name} {current_metric:.0f}/{target_metric:.0f}")
        for suggestion in suggestions[:2]:
            facts.append(f"User input: {suggestion}")

        return MeaningFrame(
            response_class=response_class,
            lead=f"{skill_id} is currently in {phase} phase.",
            facts=facts,
            metadata={"kind": kind, "skill_id": skill_id, "phase": phase},
            safety_flags=["strict_provenance_grounded", "learning_job_status"],
        )

    if response_class == "identity_answer":
        payload = grounding_payload if isinstance(grounding_payload, dict) else {}
        kind = str(payload.get("kind", ""))
        if kind == "identity_check_match":
            name = str(payload.get("check_name", "") or "that person")
            methods = list(payload.get("matched_modalities", []) or [])
            facts = [f"Match: {name}"]
            if methods:
                facts.append(f"Confirmed by: {', '.join(methods)}")
            return MeaningFrame(
                response_class=response_class,
                lead=f"Yes, this is {name}.",
                facts=facts,
                metadata={"kind": kind, "name": name},
            )
        if kind == "identity_check_mismatch":
            asked = str(payload.get("check_name", "") or "that person")
            actual = str(payload.get("actual_name", "") or "someone else")
            confidence = payload.get("actual_confidence")
            facts = [f"Asked about: {asked}", f"Current identity: {actual}"]
            if isinstance(confidence, (int, float)) and confidence > 0:
                facts.append(f"Confidence: {int(confidence * 100):d}%")
            return MeaningFrame(
                response_class=response_class,
                lead=f"No, this does not appear to be {asked}.",
                facts=facts,
                metadata={"kind": kind, "asked": asked, "actual": actual},
            )
        if kind == "identity_check_enrolled_but_not_match":
            name = str(payload.get("check_name", "") or "that person")
            return MeaningFrame(
                response_class=response_class,
                lead=f"I know who {name} is, but the current voice does not match that profile.",
                facts=[f"Asked about: {name}", "Status: enrolled profile exists, current signal does not match"],
                metadata={"kind": kind, "name": name},
            )
        if kind == "identity_check_unknown_profile":
            name = str(payload.get("check_name", "") or "that person")
            return MeaningFrame(
                response_class=response_class,
                lead=f"I don't have a profile for {name} yet.",
                facts=[f"Asked about: {name}", "Enrollment is available if they want to register."],
                metadata={"kind": kind, "name": name},
            )
        if kind == "current_voice":
            name = str(payload.get("name", "") or "unknown")
            confidence = payload.get("confidence")
            facts = [f"Speaker: {name}"]
            if isinstance(confidence, (int, float)) and confidence > 0:
                facts.append(f"Confidence: {int(confidence * 100):d}%")
            return MeaningFrame(
                response_class=response_class,
                lead=f"The current speaker is identified as {name}.",
                facts=facts,
                metadata={"kind": kind, "name": name},
            )
        if kind == "current_face":
            name = str(payload.get("name", "") or "unknown")
            confidence = payload.get("confidence")
            facts = [f"Face: {name}"]
            if isinstance(confidence, (int, float)) and confidence > 0:
                facts.append(f"Confidence: {int(confidence * 100):d}%")
            return MeaningFrame(
                response_class=response_class,
                lead=f"The current face is identified as {name}.",
                facts=facts,
                metadata={"kind": kind, "name": name},
            )
        if kind == "unknown_identity":
            facts: list[str] = []
            enrolled_v = list(payload.get("enrolled_voices", []) or [])
            enrolled_f = list(payload.get("enrolled_faces", []) or [])
            if enrolled_v:
                facts.append(f"Enrolled voice profiles: {', '.join(enrolled_v)}")
            if enrolled_f:
                facts.append(f"Enrolled face profiles: {', '.join(enrolled_f)}")
            return MeaningFrame(
                response_class=response_class,
                lead="I don't recognize you yet. No current voice or face profile matches.",
                facts=facts,
                metadata={"kind": kind},
            )

    if response_class == "capability_status":
        payload = grounding_payload if isinstance(grounding_payload, dict) else {}
        kind = str(payload.get("kind", ""))
        skill_name = str(payload.get("skill_name", "") or payload.get("skill_id", "") or "that capability")
        if kind == "perform_unverified":
            return MeaningFrame(
                response_class=response_class,
                lead=f"I don't have {skill_name} verified yet.",
                facts=["Status: learning or unverified", "I won't claim that capability until evidence verifies it."],
                metadata={"kind": kind, "skill_name": skill_name},
                safety_flags=["registry_first_capability_gate"],
            )
        if kind == "perform_verified":
            return MeaningFrame(
                response_class=response_class,
                lead=f"{skill_name} is verified.",
                facts=["Status: verified"],
                metadata={"kind": kind, "skill_name": skill_name},
                safety_flags=["registry_first_capability_gate"],
            )
        if kind in {
            "system_uninitialized",
            "resolver_unavailable",
            "unresolved_request",
            "generic_fallback",
            "already_verified",
            "already_learning",
            "restart_learning",
            "job_creation_failed",
            "job_started",
            "guided_collect_not_available",
            "guided_collect_not_needed",
            "guided_collect_started",
            "guided_collect_continue",
            "guided_collect_completed",
            "guided_collect_cancelled",
        }:
            facts = []
            status = str(payload.get("status", "") or "")
            capability_type = str(payload.get("capability_type", "") or "")
            job_id = str(payload.get("job_id", "") or "")
            phase = str(payload.get("phase", "") or "")
            risk_level = str(payload.get("risk_level", "") or "")
            protocol_id = str(payload.get("protocol_id", "") or "")
            if skill_name:
                facts.append(f"Skill: {skill_name}")
            if status:
                facts.append(f"Status: {status}")
            if capability_type:
                facts.append(f"Capability type: {capability_type}")
            if job_id:
                facts.append(f"Job: {job_id}")
            if phase:
                facts.append(f"Phase: {phase}")
            if risk_level:
                facts.append(f"Risk: {risk_level}")
            if protocol_id:
                facts.append(f"Protocol: {protocol_id}")
            lead = str(payload.get("message", "") or "Here is the current capability status.")
            return MeaningFrame(
                response_class=response_class,
                lead=lead,
                facts=facts,
                metadata={"kind": kind, "skill_name": skill_name},
                safety_flags=["registry_first_capability_gate"],
            )

    if response_class == "system_explanation":
        payload = grounding_payload if isinstance(grounding_payload, dict) else {}
        title = str(payload.get("title", "") or "System explanation")
        body = str(payload.get("body", "") or "")
        query = str(payload.get("query", "") or "")
        lines = _clean_lines(body, limit=10)
        if not lines:
            return MeaningFrame(
                response_class=response_class,
                lead="I don't have grounded system explanation details available right now.",
                missing_reason="missing_system_explanation_body",
                metadata={"title": title, "query": query},
                safety_flags=["fail_closed_when_missing"],
            )
        return MeaningFrame(
            response_class=response_class,
            lead=f"{title}:",
            facts=lines,
            metadata={"title": title, "query": query, "line_count": len(lines)},
            safety_flags=["grounded_codebase_answer"],
        )

    if response_class == "self_introspection":
        raw_text = str(grounding_payload or "")
        sections, warnings = _parse_introspection_sections(raw_text)
        ranked_facts = _rank_and_select_facts(
            sections,
            max_facts=MAX_ARTICULATE_FACTS,
            preferred_categories=preferred_categories,
        )
        section_count = len(sections)
        fact_count = sum(len(lines) for _, _, lines in sections)
        confidence = min(1.0, (fact_count / 20.0) * (section_count / 4.0)) if section_count > 0 else 0.0

        if not ranked_facts:
            return MeaningFrame(
                response_class=response_class,
                lead="I don't have detailed introspection data available right now.",
                missing_reason="no_introspection_facts_parsed",
                safety_flags=["fail_closed_when_missing"],
                frame_confidence=0.0,
                fact_count=0,
                section_count=0,
                parse_warnings=warnings,
            )

        categories_present = sorted({cat for cat, _, _ in sections})
        lead = "Here is what I can report from my current state."
        if "current_state" in categories_present:
            lead = "Here is my current measured state."

        return MeaningFrame(
            response_class=response_class,
            lead=lead,
            facts=ranked_facts,
            metadata={
                "section_count": section_count,
                "total_facts": fact_count,
                "categories": categories_present,
            },
            safety_flags=["bounded_introspection"],
            frame_confidence=round(confidence, 3),
            fact_count=fact_count,
            section_count=section_count,
            parse_warnings=warnings,
        )

    if response_class == "emergence_evidence":
        payload = grounding_payload if isinstance(grounding_payload, dict) else {}
        levels = payload.get("levels") if isinstance(payload.get("levels"), list) else []
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        if not levels:
            return MeaningFrame(
                response_class=response_class,
                lead="I don't have a grounded emergence evidence snapshot available right now.",
                missing_reason="missing_emergence_evidence_snapshot",
                metadata={"summary": summary, "levels": []},
                safety_flags=["fail_closed_when_missing", "no_sentience_claim"],
                frame_confidence=0.0,
            )
        return MeaningFrame(
            response_class=response_class,
            lead="The emergence dashboard is operational evidence, not proof of sentience.",
            facts=[
                f"L{level.get('level')}: {level.get('name')} — {level.get('status')} ({level.get('evidence_count', 0)} evidence records)"
                for level in levels
                if isinstance(level, dict)
            ][:8],
            metadata={"summary": summary, "levels": levels},
            safety_flags=["bounded_emergence_evidence", "no_sentience_claim", "level7_not_claimed"],
            frame_confidence=0.95,
            fact_count=len(levels),
            section_count=1,
        )

    lines = _clean_lines(str(grounding_payload or ""), limit=8)
    if not lines:
        return MeaningFrame(
            response_class=response_class,
            lead="I don't have grounded data for that response class yet.",
            missing_reason="unsupported_response_class",
            safety_flags=["fail_closed_when_missing"],
        )
    return MeaningFrame(
        response_class=response_class,
        lead="Here is the grounded data I could verify.",
        facts=lines,
    )


def _articulate_recent_learning(frame: MeaningFrame) -> str:
    """Produce natural spoken answer for recent_learning / recent_research.

    All data comes from the MeaningFrame. Handles 7 kind subtypes via
    metadata["kind"]. Enforces output caps and anti-confab rules.
    """
    if frame.missing_reason:
        return frame.lead.strip()

    kind = frame.metadata.get("kind", "")
    sentences: list[str] = []
    char_count = 0

    def _append(s: str) -> bool:
        nonlocal char_count
        if len(sentences) >= MAX_ARTICULATE_SENTENCES or char_count >= MAX_ARTICULATE_CHARS:
            return False
        if char_count + len(s) > MAX_ARTICULATE_CHARS:
            return False
        sentences.append(s)
        char_count += len(s)
        return True

    if kind == "scholarly_source":
        title = frame.metadata.get("title", "")
        when_fact = next((f for f in frame.facts if f.startswith("Studied:")), "")
        when = when_fact.replace("Studied:", "").strip() if when_fact else "recently"
        _append(f'The most recent peer-reviewed source I studied is "{title}", {when}.')
        venue_fact = next((f for f in frame.facts if f.startswith("Venue:")), "")
        if venue_fact:
            _append(f"It was published in {venue_fact.replace('Venue:', '').strip()}.")
        doi_fact = next((f for f in frame.facts if f.startswith("DOI:")), "")
        if doi_fact:
            _append(f"The DOI is {doi_fact.replace('DOI:', '').strip()}.")

    elif kind == "source":
        title = frame.metadata.get("title", "")
        type_fact = next((f for f in frame.facts if f.startswith("Type:")), "")
        source_type = type_fact.replace("Type:", "").strip() if type_fact else "source"
        when_fact = next((f for f in frame.facts if f.startswith("Studied:")), "")
        when = when_fact.replace("Studied:", "").strip() if when_fact else "recently"
        _append(f'The most recent {source_type} I studied is "{title}", {when}.')

    elif kind == "autonomy_research":
        q_fact = next((f for f in frame.facts if f.startswith("Question:")), "")
        question = q_fact.replace("Question:", "").strip().strip('"') if q_fact else "an unknown topic"
        tool_fact = next((f for f in frame.facts if f.startswith("Tool:")), "")
        tool = tool_fact.replace("Tool:", "").strip() if tool_fact else ""
        when_fact = next((f for f in frame.facts if f.startswith("Recorded:")), "")
        when = when_fact.replace("Recorded:", "").strip() if when_fact else "recently"
        _append(f'My latest autonomous research was about "{question}", {when}.')
        if tool:
            _append(f"I used the {tool} tool.")
        summary_fact = next((f for f in frame.facts if f.startswith("Summary:")), "")
        if summary_fact:
            _append(summary_fact.replace("Summary:", "").strip())

    elif kind == "learning_job":
        skill = frame.metadata.get("skill_id", "unknown")
        status_fact = next((f for f in frame.facts if f.startswith("Status:")), "")
        status = status_fact.replace("Status:", "").strip() if status_fact else "unknown"
        phase_fact = next((f for f in frame.facts if f.startswith("Phase:")), "")
        phase = phase_fact.replace("Phase:", "").strip() if phase_fact else ""
        when_fact = next((f for f in frame.facts if f.startswith("Updated:")), "")
        when = when_fact.replace("Updated:", "").strip() if when_fact else "recently"
        _append(f"My latest learning job is for {skill.replace('_', ' ')}, status {status}, updated {when}.")
        if phase:
            _append(f"It is in the {phase} phase.")

    else:
        _append(frame.lead.strip())
        for fact in frame.facts[:MAX_ARTICULATE_FACTS]:
            if not _append(_fact_to_sentence(fact)):
                break

    if not sentences:
        return frame.lead.strip()

    return " ".join(sentences).strip()


def _articulate_identity_answer(frame: MeaningFrame) -> str:
    """Produce natural spoken answer for identity_answer.

    All data comes from the MeaningFrame. Handles 7 kind subtypes via
    metadata["kind"]. Enforces output caps and anti-confab rules.
    """
    kind = frame.metadata.get("kind", "")
    sentences: list[str] = []
    char_count = 0

    def _append(s: str) -> bool:
        nonlocal char_count
        if len(sentences) >= MAX_ARTICULATE_SENTENCES or char_count >= MAX_ARTICULATE_CHARS:
            return False
        if char_count + len(s) > MAX_ARTICULATE_CHARS:
            return False
        sentences.append(s)
        char_count += len(s)
        return True

    if kind == "identity_check_match":
        name = frame.metadata.get("name", "that person")
        _append(f"Yes, this is {name}.")
        confirm_fact = next((f for f in frame.facts if f.startswith("Confirmed by:")), "")
        if confirm_fact:
            methods = confirm_fact.replace("Confirmed by:", "").strip()
            _append(f"Confirmed by {methods}.")

    elif kind == "identity_check_mismatch":
        asked = frame.metadata.get("asked", "that person")
        actual = frame.metadata.get("actual", "someone else")
        _append(f"No, this does not appear to be {asked}.")
        _append(f"The current identity is {actual}.")
        conf_fact = next((f for f in frame.facts if f.startswith("Confidence:")), "")
        if conf_fact:
            _append(f"Recognition confidence is {conf_fact.replace('Confidence:', '').strip()}.")

    elif kind == "identity_check_enrolled_but_not_match":
        name = frame.metadata.get("name", "that person")
        _append(f"I know who {name} is, but the current signal does not match that profile.")

    elif kind == "identity_check_unknown_profile":
        name = frame.metadata.get("name", "that person")
        _append(f"I don't have a profile for {name} yet.")
        _append("Enrollment is available if they want to register.")

    elif kind == "current_voice":
        name = frame.metadata.get("name", "unknown")
        _append(f"The current speaker is identified as {name}.")
        conf_fact = next((f for f in frame.facts if f.startswith("Confidence:")), "")
        if conf_fact:
            _append(f"Voice confidence is {conf_fact.replace('Confidence:', '').strip()}.")

    elif kind == "current_face":
        name = frame.metadata.get("name", "unknown")
        _append(f"The current face is identified as {name}.")
        conf_fact = next((f for f in frame.facts if f.startswith("Confidence:")), "")
        if conf_fact:
            _append(f"Face confidence is {conf_fact.replace('Confidence:', '').strip()}.")

    elif kind == "unknown_identity":
        _append("I don't recognize you yet.")
        for fact in frame.facts[:2]:
            _append(fact if fact.endswith(".") else f"{fact}.")

    else:
        _append(frame.lead.strip())
        for fact in frame.facts[:MAX_ARTICULATE_FACTS]:
            if not _append(_fact_to_sentence(fact)):
                break

    if not sentences:
        return frame.lead.strip()

    return " ".join(sentences).strip()


def _articulate_memory_recall(frame: MeaningFrame) -> str:
    """Produce natural spoken answer for memory_recall.

    All data comes from the MeaningFrame. Renders search results or
    summary facts. Enforces output caps and anti-confab rules.
    """
    if frame.missing_reason:
        return frame.lead.strip()

    mode = frame.metadata.get("mode", "summary")
    sentences: list[str] = []
    char_count = 0

    def _append(s: str) -> bool:
        nonlocal char_count
        if len(sentences) >= MAX_ARTICULATE_SENTENCES or char_count >= MAX_ARTICULATE_CHARS:
            return False
        if char_count + len(s) > MAX_ARTICULATE_CHARS:
            return False
        sentences.append(s)
        char_count += len(s)
        return True

    n_facts = len(frame.facts)
    if mode == "search":
        _append(f"I found {n_facts} relevant memory {'entries' if n_facts != 1 else 'entry'}.")
    else:
        _append("Here is what I have from memory.")

    for raw_fact in frame.facts[:MAX_ARTICULATE_FACTS]:
        fact = raw_fact.strip()
        if not fact:
            continue
        if fact.startswith("Memory recall:"):
            continue
        sentence = fact if fact.endswith(".") else f"{fact}."
        if not _append(sentence):
            break

    if not sentences:
        return frame.lead.strip()

    return " ".join(sentences).strip()


def _articulate_capability_status(frame: MeaningFrame) -> str:
    """Produce natural spoken answer for capability_status.

    All data comes from the MeaningFrame. Handles 14+ kind subtypes via
    metadata["kind"]. Enforces output caps and anti-confab rules.
    """
    kind = frame.metadata.get("kind", "")
    skill = frame.metadata.get("skill_name", "that capability")
    sentences: list[str] = []
    char_count = 0

    def _append(s: str) -> bool:
        nonlocal char_count
        if len(sentences) >= MAX_ARTICULATE_SENTENCES or char_count >= MAX_ARTICULATE_CHARS:
            return False
        if char_count + len(s) > MAX_ARTICULATE_CHARS:
            return False
        sentences.append(s)
        char_count += len(s)
        return True

    if kind == "perform_unverified":
        _append(f"I don't have {skill} verified yet.")
        _append("I won't claim that capability until evidence verifies it.")

    elif kind == "perform_verified":
        _append(f"{skill} is verified and ready.")

    elif kind == "job_started":
        _append(frame.lead.strip())
        phase_fact = next((f for f in frame.facts if f.startswith("Phase:")), "")
        if phase_fact:
            _append(f"Starting in the {phase_fact.replace('Phase:', '').strip()} phase.")
        cap_fact = next((f for f in frame.facts if f.startswith("Capability type:")), "")
        if cap_fact:
            _append(f"This is a {cap_fact.replace('Capability type:', '').strip()} capability.")

    elif kind in ("already_verified", "already_learning"):
        _append(frame.lead.strip())

    elif kind in ("guided_collect_started", "guided_collect_continue"):
        _append(frame.lead.strip())
        for fact in frame.facts[:2]:
            if "Phase:" in fact or "Protocol:" in fact:
                continue
            if not _append(_fact_to_sentence(fact)):
                break

    elif kind == "guided_collect_completed":
        _append(frame.lead.strip())

    elif kind in ("system_uninitialized", "resolver_unavailable"):
        _append("The skill system is not fully initialized yet.")

    else:
        _append(frame.lead.strip())
        for fact in frame.facts[:MAX_ARTICULATE_FACTS]:
            if not _append(_fact_to_sentence(fact)):
                break

    if not sentences:
        return frame.lead.strip()

    return " ".join(sentences).strip()


def _articulate_system_explanation(frame: MeaningFrame) -> str:
    """Produce natural spoken answer for system_explanation.

    All data comes from the MeaningFrame. Renders codebase tool answers
    as spoken sentences. Enforces output caps and anti-confab rules.
    """
    if frame.missing_reason:
        return frame.lead.strip()

    title = frame.metadata.get("title", "").strip().rstrip(":")
    sentences: list[str] = []
    char_count = 0

    def _append(s: str) -> bool:
        nonlocal char_count
        if len(sentences) >= MAX_ARTICULATE_SENTENCES or char_count >= MAX_ARTICULATE_CHARS:
            return False
        if char_count + len(s) > MAX_ARTICULATE_CHARS:
            return False
        sentences.append(s)
        char_count += len(s)
        return True

    if title and title.lower() != "system explanation":
        _append(f"Regarding {title.lower()}:")
    else:
        _append("Here is what I found in the codebase.")

    for fact in frame.facts[:MAX_ARTICULATE_FACTS]:
        fact = fact.strip()
        if not fact:
            continue
        sentence = fact if fact.endswith(".") else f"{fact}."
        if not _append(sentence):
            break

    if not sentences:
        return frame.lead.strip()

    return " ".join(sentences).strip()


def articulate_meaning_frame(frame: MeaningFrame) -> str:
    if frame.response_class == "self_status":
        return _articulate_self_status(frame)
    if frame.response_class == "self_introspection":
        return _articulate_self_introspection(frame)
    if frame.response_class == "emergence_evidence":
        return _articulate_emergence_evidence(frame)
    if frame.response_class in ("recent_learning", "recent_research"):
        return _articulate_recent_learning(frame)
    if frame.response_class == "identity_answer":
        return _articulate_identity_answer(frame)
    if frame.response_class == "memory_recall":
        return _articulate_memory_recall(frame)
    if frame.response_class == "capability_status":
        return _articulate_capability_status(frame)
    if frame.response_class == "system_explanation":
        return _articulate_system_explanation(frame)
    parts = [frame.lead.strip()]
    for fact in frame.facts:
        fact = fact.strip()
        if fact:
            parts.append(fact)
    return " ".join(part for part in parts if part).strip()

