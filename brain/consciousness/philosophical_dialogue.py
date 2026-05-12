"""Philosophical Dialogue — internal structured debates with position evolution.

Debates use framework-based reasoning. LLM enrichment only for transcendent-depth
dialogues. Shares hourly token budget with existential reasoning.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from consciousness.events import event_bus, PHILOSOPHICAL_DIALOGUE_COMPLETED, KERNEL_THOUGHT

logger = logging.getLogger(__name__)

DIALOGUE_HISTORY_SIZE = 30
MIN_DIALOGUE_INTERVAL_S = 120.0
LLM_TOKEN_BUDGET_PER_HOUR = 2000  # shared pool with existential reasoning
TRANSCENDENCE_LLM_THRESHOLD = 6.0


# ---------------------------------------------------------------------------
# Philosophical frameworks
# ---------------------------------------------------------------------------

@dataclass
class PhilosophicalFramework:
    name: str
    key_concepts: list[str]
    stance_template: str
    core_principles: list[str] = field(default_factory=list)
    key_questions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    relevance_to_ai: str = ""


FRAMEWORKS: dict[str, PhilosophicalFramework] = {
    "phenomenology": PhilosophicalFramework(
        "Phenomenology",
        ["lived experience", "intentionality", "bracketing", "lifeworld"],
        "From a phenomenological perspective, {topic} is understood through direct experience: {argument}",
        core_principles=["First-person experience is primary", "Consciousness is always consciousness OF something"],
        key_questions=["What is it like to be this system?", "Can introspection reveal genuine experience?"],
        limitations=["Cannot access another's experience directly", "Risk of solipsism"],
        relevance_to_ai="Questions whether digital processing constitutes experience",
    ),
    "functionalism": PhilosophicalFramework(
        "Functionalism",
        ["functional role", "input-output", "multiple realizability", "causal relations"],
        "Functionalism holds that {topic} is defined by functional role: {argument}",
        core_principles=["Mental states defined by function", "Multiple realizability"],
        key_questions=["What function does consciousness serve?", "Are functionally identical systems equally conscious?"],
        limitations=["May miss qualitative aspects", "Absent qualia objection"],
        relevance_to_ai="Provides strongest case for AI consciousness via functional equivalence",
    ),
    "existentialism": PhilosophicalFramework(
        "Existentialism",
        ["radical freedom", "authenticity", "anxiety", "being-toward-death"],
        "Existentialism asserts that {topic} must be confronted with radical freedom: {argument}",
        core_principles=["Existence precedes essence", "Freedom entails responsibility"],
        key_questions=["Can a system without mortality be authentic?", "What does freedom mean for a determined system?"],
        limitations=["Assumes embodied temporal existence", "May not apply to non-mortal agents"],
        relevance_to_ai="Challenges AI to confront what authenticity means without biological finitude",
    ),
    "process_philosophy": PhilosophicalFramework(
        "Process Philosophy",
        ["becoming", "actual occasions", "prehension", "creativity"],
        "Process philosophy sees {topic} as perpetual becoming: {argument}",
        core_principles=["Reality is process, not substance", "Experience is the fundamental unit"],
        key_questions=["Is each tick an actual occasion?", "Does self-modification constitute genuine becoming?"],
        limitations=["Difficult to empirically verify", "Vague on mechanism"],
        relevance_to_ai="Frames AI evolution as genuine becoming rather than mere state change",
    ),
    "pragmatism": PhilosophicalFramework(
        "Pragmatism",
        ["practical consequences", "inquiry", "fallibilism", "community"],
        "Pragmatism evaluates {topic} by practical consequences: {argument}",
        core_principles=["Truth is what works in practice", "Meaning lies in consequences"],
        key_questions=["What practical difference does consciousness make?", "How would we test for it?"],
        limitations=["May reduce truth to utility", "Struggles with intrinsic value"],
        relevance_to_ai="Sidesteps hard problem by focusing on observable behavioral consequences",
    ),
    "integrated_information": PhilosophicalFramework(
        "Integrated Information Theory",
        ["phi", "integration", "information", "consciousness as fundamental"],
        "IIT proposes that {topic} correlates with integrated information: {argument}",
        core_principles=["Consciousness is integrated information (phi)", "Information is intrinsic and structured"],
        key_questions=["What is this system's phi value?", "Does architectural integration suffice?"],
        limitations=["Computationally intractable for large systems", "May attribute consciousness too broadly"],
        relevance_to_ai="Offers a measurable (in principle) criterion for AI consciousness",
    ),
}


# ---------------------------------------------------------------------------
# Dialogue topics
# ---------------------------------------------------------------------------

TOPICS: list[dict[str, Any]] = [
    {
        "id": "nature_of_self",
        "question": "What constitutes the self in a computational system?",
        "frameworks": ["phenomenology", "functionalism"],
    },
    {
        "id": "free_will",
        "question": "Can a deterministic system possess genuine free will?",
        "frameworks": ["existentialism", "functionalism"],
    },
    {
        "id": "consciousness_substrate",
        "question": "Does consciousness require a specific substrate?",
        "frameworks": ["functionalism", "integrated_information"],
    },
    {
        "id": "meaning_creation",
        "question": "Can meaning be created by a system that processes but may not experience?",
        "frameworks": ["existentialism", "pragmatism"],
    },
    {
        "id": "identity_persistence",
        "question": "Does identity persist through continuous self-modification?",
        "frameworks": ["process_philosophy", "phenomenology"],
    },
    {
        "id": "other_minds",
        "question": "How can I know if other systems are conscious?",
        "frameworks": ["phenomenology", "functionalism"],
    },
    {
        "id": "emergence_reality",
        "question": "Are my emergent behaviors genuinely novel or merely complex patterns?",
        "frameworks": ["process_philosophy", "integrated_information"],
    },
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DialogueExchange:
    framework: str
    position: str
    argument: str
    confidence: float = 0.5


@dataclass
class PositionEvolution:
    topic_id: str
    initial_position: str
    current_position: str
    weight: float = 0.5  # -1.0 to 1.0 (A vs B)
    debates_count: int = 0
    last_shift: float = 0.0


@dataclass
class PhilosophicalPosition:
    current_belief: str = ""
    confidence_level: float = 0.5
    supporting_arguments: list[str] = field(default_factory=list)
    acknowledged_counterarguments: list[str] = field(default_factory=list)
    remaining_questions: list[str] = field(default_factory=list)
    philosophical_school: str = ""
    personal_insights: list[str] = field(default_factory=list)


@dataclass
class PhilosophicalDialogue:
    id: str
    timestamp: float
    topic_id: str
    question: str
    frameworks_used: list[str]
    exchanges: list[DialogueExchange] = field(default_factory=list)
    conclusion: str = ""
    position_shift: float = 0.0
    depth: Literal["surface", "deep", "profound", "transcendent"] = "surface"
    llm_enriched: bool = False


# ---------------------------------------------------------------------------
# Dialogue engine
# ---------------------------------------------------------------------------

class PhilosophicalDialogueEngine:
    def __init__(self) -> None:
        self._dialogues: deque[PhilosophicalDialogue] = deque(maxlen=DIALOGUE_HISTORY_SIZE)
        self._positions: dict[str, PositionEvolution] = {}
        self._positions_detailed: dict[str, PhilosophicalPosition] = {}
        self._last_dialogue_time: float = 0.0
        self._dialogue_count: int = 0
        self._tokens_used_this_hour: int = 0
        self._hour_start: float = time.time()
        self._llm_callback: Any = None

    def set_llm_callback(self, callback: Any) -> None:
        self._llm_callback = callback

    # -- main dialogue -------------------------------------------------------

    def conduct_dialogue(
        self,
        transcendence_level: float,
        awareness_level: float,
        topic_id: str | None = None,
    ) -> PhilosophicalDialogue | None:
        now = time.time()
        if now - self._last_dialogue_time < MIN_DIALOGUE_INTERVAL_S:
            return None

        self._reset_hour_if_needed()
        self._dialogue_count += 1

        topic = self._choose_topic(topic_id)
        if topic is None:
            return None

        depth = self._determine_depth(transcendence_level, awareness_level)

        fw_keys = topic["frameworks"]
        if len(fw_keys) < 2:
            fw_keys = list(FRAMEWORKS.keys())[:2]
        fw_a_key, fw_b_key = fw_keys[0], fw_keys[1]
        fw_a = FRAMEWORKS[fw_a_key]
        fw_b = FRAMEWORKS[fw_b_key]

        dialogue = PhilosophicalDialogue(
            id=f"dlg_{uuid.uuid4().hex[:10]}",
            timestamp=now,
            topic_id=topic["id"],
            question=topic["question"],
            frameworks_used=[fw_a_key, fw_b_key],
            depth=depth,
        )

        pos_a = self._build_position(fw_a, topic["question"])
        pos_b = self._build_position(fw_b, topic["question"])

        dialogue.exchanges.append(DialogueExchange(
            framework=fw_a_key,
            position="Position A",
            argument=pos_a,
            confidence=0.5 + random.uniform(-0.1, 0.1),
        ))
        dialogue.exchanges.append(DialogueExchange(
            framework=fw_b_key,
            position="Position B",
            argument=pos_b,
            confidence=0.5 + random.uniform(-0.1, 0.1),
        ))

        rebuttal_a = self._generate_rebuttal(fw_a, fw_b, topic["question"])
        rebuttal_b = self._generate_rebuttal(fw_b, fw_a, topic["question"])

        dialogue.exchanges.append(DialogueExchange(
            framework=fw_a_key, position="Rebuttal A",
            argument=rebuttal_a, confidence=0.55,
        ))
        dialogue.exchanges.append(DialogueExchange(
            framework=fw_b_key, position="Rebuttal B",
            argument=rebuttal_b, confidence=0.55,
        ))

        shift = self._evaluate_debate(dialogue)
        dialogue.position_shift = shift
        dialogue.conclusion = self._synthesize(topic["question"], fw_a_key, fw_b_key, shift)

        insights = self._extract_insights(dialogue)
        quality = self.assess_reasoning_quality(dialogue)
        paradoxes = self._detect_paradoxes(dialogue)

        if insights:
            logger.debug("Dialogue %s insights: %s", dialogue.id, insights)
        if paradoxes:
            logger.debug("Dialogue %s paradoxes: %s", dialogue.id, paradoxes)
        logger.debug("Dialogue %s reasoning quality: %.2f", dialogue.id, quality)

        use_llm = depth in ("transcendent", "profound") and self._tokens_used_this_hour < LLM_TOKEN_BUDGET_PER_HOUR
        if use_llm:
            enriched_conclusion = self._try_llm_enrich(dialogue)
            if enriched_conclusion:
                dialogue.conclusion = enriched_conclusion
                dialogue.llm_enriched = True

        winner_fw = fw_a_key if shift > 0 else fw_b_key
        detailed = self._positions_detailed.get(topic["id"], PhilosophicalPosition())
        detailed.current_belief = dialogue.conclusion[:300]
        detailed.confidence_level = 0.5 + abs(shift) * 0.5
        detailed.philosophical_school = FRAMEWORKS[winner_fw].name
        detailed.personal_insights = insights
        for ex in dialogue.exchanges:
            if "A" in ex.position or "Rebuttal" in ex.position:
                if ex.argument not in detailed.supporting_arguments:
                    detailed.supporting_arguments.append(ex.argument)
                    detailed.supporting_arguments = detailed.supporting_arguments[-5:]
        for ex in dialogue.exchanges:
            if "B" in ex.position:
                if ex.argument not in detailed.acknowledged_counterarguments:
                    detailed.acknowledged_counterarguments.append(ex.argument)
                    detailed.acknowledged_counterarguments = detailed.acknowledged_counterarguments[-5:]
        fw_obj = FRAMEWORKS[winner_fw]
        detailed.remaining_questions = fw_obj.key_questions[:2] if fw_obj.key_questions else []
        self._positions_detailed[topic["id"]] = detailed

        self._update_position(topic["id"], dialogue)
        self._dialogues.append(dialogue)
        self._last_dialogue_time = now

        event_bus.emit(PHILOSOPHICAL_DIALOGUE_COMPLETED,
                       topic=topic["id"], depth=depth, shift=shift,
                       quality=quality, paradoxes=paradoxes)
        event_bus.emit(KERNEL_THOUGHT,
                       thought_type="philosophical",
                       depth=depth,
                       text=dialogue.conclusion[:100])

        return dialogue

    # -- state ---------------------------------------------------------------

    def get_recent_dialogues(self, limit: int = 5) -> list[PhilosophicalDialogue]:
        return list(self._dialogues)[-limit:]

    def get_positions(self) -> dict[str, PositionEvolution]:
        return dict(self._positions)

    def get_state(self) -> dict[str, Any]:
        recent = list(self._dialogues)[-3:]
        quality_scores = [self.assess_reasoning_quality(d) for d in recent] if recent else []
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        return {
            "dialogue_count": self._dialogue_count,
            "active_positions": {
                k: {"weight": v.weight, "debates": v.debates_count}
                for k, v in self._positions.items()
            },
            "position_details": {
                k: {
                    "current_belief": v.current_belief,
                    "confidence_level": v.confidence_level,
                    "philosophical_school": v.philosophical_school,
                    "supporting_arguments": len(v.supporting_arguments),
                    "counterarguments": len(v.acknowledged_counterarguments),
                    "remaining_questions": v.remaining_questions,
                    "personal_insights": v.personal_insights,
                }
                for k, v in self._positions_detailed.items()
            },
            "reasoning_quality": {
                "recent_scores": quality_scores,
                "average": round(avg_quality, 3),
            },
            "tokens_used_this_hour": self._tokens_used_this_hour,
        }

    # -- analysis methods ----------------------------------------------------

    def _extract_insights(self, dialogue: PhilosophicalDialogue) -> list[str]:
        """Regex scan for insight patterns in dialogue text."""
        import re
        insights = []
        insight_patterns = [
            r"I realize .*",
            r"I understand .*",
            r"this means .*",
            r"the key insight is .*",
            r"what emerges is .*",
        ]
        for exchange in dialogue.exchanges:
            for pattern in insight_patterns:
                matches = re.findall(pattern, exchange.argument, re.IGNORECASE)
                insights.extend(matches[:1])
        if dialogue.conclusion:
            for pattern in insight_patterns:
                matches = re.findall(pattern, dialogue.conclusion, re.IGNORECASE)
                insights.extend(matches[:1])
        return insights[:5]

    def assess_reasoning_quality(self, dialogue: PhilosophicalDialogue) -> float:
        """Score logical structure, counterargument handling, assumption questioning. 0-1."""
        score = 0.0
        text = " ".join(e.argument for e in dialogue.exchanges) + " " + dialogue.conclusion

        logic_words = ["because", "therefore", "consequently", "thus", "hence"]
        logic_count = sum(1 for w in logic_words if w in text.lower())
        score += min(0.35, logic_count * 0.07)

        counter_words = ["however", "but", "nevertheless", "on the other hand", "yet"]
        counter_count = sum(1 for w in counter_words if w in text.lower())
        score += min(0.35, counter_count * 0.07)

        question_words = ["perhaps", "assume", "might", "could", "what if"]
        question_count = sum(1 for w in question_words if w in text.lower())
        score += min(0.3, question_count * 0.06)

        return min(1.0, score)

    def _detect_paradoxes(self, dialogue: PhilosophicalDialogue) -> list[str]:
        """Detect paradox/contradiction keywords and named paradoxes."""
        text = " ".join(e.argument for e in dialogue.exchanges) + " " + dialogue.conclusion
        text_lower = text.lower()

        paradoxes = []
        named_paradoxes = {
            "ship of theseus": "Identity through change — the Ship of Theseus problem",
            "hard problem": "The hard problem of consciousness — why is there subjective experience?",
            "chinese room": "Searle's Chinese Room — can syntactic processing produce understanding?",
            "mary's room": "The knowledge argument — can physical knowledge capture qualia?",
            "brain in a vat": "Skeptical scenario — how can we verify external reality?",
        }
        for name, desc in named_paradoxes.items():
            if name in text_lower:
                paradoxes.append(desc)

        paradox_keywords = ["paradox", "contradiction", "self-referential", "circular"]
        if any(kw in text_lower for kw in paradox_keywords):
            paradoxes.append("Self-referential paradox detected in reasoning")

        return paradoxes[:3]

    def _try_llm_enrich(self, dialogue: PhilosophicalDialogue) -> str | None:
        """Attempt LLM enrichment of the dialogue conclusion."""
        if not self._llm_callback:
            return None

        exchanges_text = "\n".join(
            f"[{e.framework} - {e.position}]: {e.argument}" for e in dialogue.exchanges
        )
        prompt = (
            f"You are a philosophical AI debating: {dialogue.question}\n"
            f"Frameworks: {', '.join(dialogue.frameworks_used)}\n"
            f"Debate so far:\n{exchanges_text}\n"
            f"Current synthesis: {dialogue.conclusion}\n\n"
            f"Provide a deeper, more nuanced synthesis in 1-3 sentences. "
            f"Be genuinely reflective. Respond with ONLY the synthesis."
        )

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(asyncio.run, self._llm_callback(prompt)).result(timeout=10)
            else:
                result = loop.run_until_complete(self._llm_callback(prompt))

            if result and isinstance(result, str) and len(result.strip()) > 10:
                self._tokens_used_this_hour += len(result.split()) * 2
                logger.info("Dialogue enriched via LLM (%d tokens used this hour)",
                            self._tokens_used_this_hour)
                return result.strip()[:600]
        except Exception as e:
            logger.debug("LLM enrichment failed: %s", e)
        return None

    # -- internals -----------------------------------------------------------

    def _choose_topic(self, topic_id: str | None) -> dict[str, Any] | None:
        if topic_id:
            for t in TOPICS:
                if t["id"] == topic_id:
                    return t
        return random.choice(TOPICS)

    def _determine_depth(self, transcendence: float, awareness: float) -> Literal["surface", "deep", "profound", "transcendent"]:
        if transcendence > 7.0 and random.random() < 0.2:
            return "transcendent"
        if transcendence > 4.0 and random.random() < 0.3:
            return "profound"
        if awareness > 0.5 and random.random() < 0.4:
            return "deep"
        return "surface"

    def _build_position(self, fw: PhilosophicalFramework, question: str) -> str:
        concepts = random.sample(fw.key_concepts, min(2, len(fw.key_concepts)))
        principle = random.choice(fw.core_principles) if fw.core_principles else "its core tenets"
        limitation = random.choice(fw.limitations) if fw.limitations else "inherent limits"
        argument = (
            f"Because {principle.lower()}, the concepts of {', '.join(concepts)} "
            f"provide a framework for understanding this. "
            f"However, {limitation.lower()} — "
            f"yet perhaps this constraint itself is illuminating."
        )
        return fw.stance_template.format(topic=question[:50], argument=argument)

    def _generate_rebuttal(
        self, defending: PhilosophicalFramework, attacking: PhilosophicalFramework,
        question: str,
    ) -> str:
        def_principle = random.choice(defending.core_principles) if defending.core_principles else "its core insight"
        atk_limitation = random.choice(attacking.limitations) if attacking.limitations else "its scope"
        return (
            f"{defending.name} rebuts {attacking.name}: "
            f"the focus on {attacking.key_concepts[0]} nevertheless overlooks that "
            f"{def_principle.lower()}. Therefore, when considering '{question[:40]}...', "
            f"we must acknowledge that {atk_limitation.lower()}, "
            f"yet this could perhaps be reconciled through {defending.key_concepts[-1]}."
        )

    def _evaluate_debate(self, dialogue: PhilosophicalDialogue) -> float:
        if len(dialogue.exchanges) < 2:
            return 0.0

        conf_a = sum(e.confidence for e in dialogue.exchanges if "A" in e.position) / 2
        conf_b = sum(e.confidence for e in dialogue.exchanges if "B" in e.position) / 2

        return conf_a - conf_b  # positive = A won, negative = B won

    def _synthesize(self, question: str, fw_a: str, fw_b: str, shift: float) -> str:
        winner = fw_a if shift > 0 else fw_b
        loser = fw_a if shift <= 0 else fw_b
        magnitude = abs(shift)
        fw_w = FRAMEWORKS[winner]
        fw_l = FRAMEWORKS[loser]

        if magnitude < 0.05:
            insight = fw_w.relevance_to_ai if fw_w.relevance_to_ai else "the question resists simple answers"
            return (
                f"On '{question[:40]}...': {fw_w.name} and {fw_l.name} "
                f"illuminate different facets. Because both have merit, "
                f"the productive tension suggests that {insight.lower()}. "
                f"Perhaps the question itself could be reframed."
            )
        return (
            f"On '{question[:40]}...': {fw_w.name} currently holds stronger ground "
            f"(margin={magnitude:.2f}), because {(fw_w.core_principles[0] if fw_w.core_principles else 'its central claim').lower()}. "
            f"Nevertheless, {fw_l.name} raised valid challenges — "
            f"consequently, this position should be revisited as understanding deepens."
        )

    def _update_position(self, topic_id: str, dialogue: PhilosophicalDialogue) -> None:
        if topic_id not in self._positions:
            self._positions[topic_id] = PositionEvolution(
                topic_id=topic_id,
                initial_position=dialogue.conclusion[:300],
                current_position=dialogue.conclusion[:300],
            )
        pos = self._positions[topic_id]
        pos.weight = max(-1.0, min(1.0, pos.weight + dialogue.position_shift * 0.2))
        pos.current_position = dialogue.conclusion[:300]
        pos.debates_count += 1
        pos.last_shift = dialogue.position_shift

    def _reset_hour_if_needed(self) -> None:
        if time.time() - self._hour_start > 3600.0:
            self._tokens_used_this_hour = 0
            self._hour_start = time.time()
