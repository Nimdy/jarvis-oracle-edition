"""Episodic memory — multi-turn conversation episodes as coherent narratives.

Tracks conversation episodes with topic, emotion arc, outcome, and linked memories.
Enables: "Last time we discussed X, you seemed frustrated about Y, and we decided Z."
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JARVIS_DIR = Path.home() / ".jarvis"
DEFAULT_EPISODES_PATH = JARVIS_DIR / "episodes.json"
MAX_EPISODES = 200
EPISODE_TIMEOUT_S = 300.0  # 5 minutes of silence ends an episode
MIN_TURNS_FOR_EPISODE = 2


@dataclass
class EpisodeTurn:
    role: str  # "user" or "assistant"
    text: str
    emotion: str = "neutral"
    timestamp: float = 0.0
    conversation_id: str = ""
    trace_id: str = ""
    request_id: str = ""
    output_id: str = ""
    conversation_entry_id: str = ""
    response_entry_id: str = ""
    root_entry_id: str = ""


@dataclass
class Episode:
    id: str
    topic: str = ""
    turns: list[EpisodeTurn] = field(default_factory=list)
    emotion_arc: list[str] = field(default_factory=list)
    outcome: str = ""
    speaker: str = "unknown"
    memory_ids: list[str] = field(default_factory=list)
    started_at: float = 0.0
    ended_at: float = 0.0
    is_active: bool = True
    summary: str = ""

    def duration_s(self) -> float:
        if self.ended_at > 0:
            return self.ended_at - self.started_at
        return time.time() - self.started_at

    def turn_count(self) -> int:
        return len(self.turns)

    def add_turn(
        self,
        role: str,
        text: str,
        emotion: str = "neutral",
        conversation_id: str = "",
        trace_id: str = "",
        request_id: str = "",
        output_id: str = "",
        conversation_entry_id: str = "",
        response_entry_id: str = "",
        root_entry_id: str = "",
    ) -> None:
        self.turns.append(EpisodeTurn(
            role=role,
            text=text,
            emotion=emotion,
            timestamp=time.time(),
            conversation_id=conversation_id,
            trace_id=trace_id,
            request_id=request_id,
            output_id=output_id,
            conversation_entry_id=conversation_entry_id,
            response_entry_id=response_entry_id,
            root_entry_id=root_entry_id,
        ))
        if emotion != "neutral":
            self.emotion_arc.append(emotion)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "turns": [
                {
                    "role": t.role,
                    "text": t.text,
                    "emotion": t.emotion,
                    "timestamp": t.timestamp,
                    "conversation_id": t.conversation_id,
                    "trace_id": t.trace_id,
                    "request_id": t.request_id,
                    "output_id": t.output_id,
                    "conversation_entry_id": t.conversation_entry_id,
                    "response_entry_id": t.response_entry_id,
                    "root_entry_id": t.root_entry_id,
                }
                for t in self.turns
            ],
            "emotion_arc": self.emotion_arc,
            "outcome": self.outcome,
            "speaker": self.speaker,
            "memory_ids": self.memory_ids,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "is_active": self.is_active,
            "summary": self.summary,
        }


class EpisodicMemory:
    """Manages conversation episodes for narrative continuity."""

    def __init__(self, persist_path: str = "", max_episodes: int = MAX_EPISODES):
        self._episodes: deque[Episode] = deque(maxlen=max_episodes)
        self._active_episode: Episode | None = None
        self._persist_path = persist_path or str(DEFAULT_EPISODES_PATH)
        self._episode_counter = 0
        self.load()

    def start_episode(self, speaker: str = "unknown") -> Episode:
        """Start a new conversation episode."""
        if self._active_episode and self._active_episode.turn_count() >= MIN_TURNS_FOR_EPISODE:
            self.end_episode()
        elif self._active_episode:
            self._active_episode.is_active = False

        self._episode_counter += 1
        episode = Episode(
            id=f"ep_{self._episode_counter:05d}",
            speaker=speaker,
            started_at=time.time(),
        )
        self._active_episode = episode
        self._episodes.append(episode)
        logger.debug("Started episode %s", episode.id)
        return episode

    def add_user_turn(
        self,
        text: str,
        emotion: str = "neutral",
        speaker: str = "",
        conversation_id: str = "",
        trace_id: str = "",
        request_id: str = "",
        conversation_entry_id: str = "",
        root_entry_id: str = "",
    ) -> Episode:
        """Add a user message to the current episode, starting one if needed."""
        now = time.time()
        if self._active_episode:
            last_turn_time = self._active_episode.turns[-1].timestamp if self._active_episode.turns else self._active_episode.started_at
            if now - last_turn_time > EPISODE_TIMEOUT_S:
                self.end_episode()
                self._active_episode = None

        if not self._active_episode:
            self.start_episode(speaker=speaker or "unknown")

        self._active_episode.add_turn(
            "user",
            text,
            emotion,
            conversation_id=conversation_id,
            trace_id=trace_id,
            request_id=request_id,
            conversation_entry_id=conversation_entry_id,
            root_entry_id=root_entry_id,
        )
        if speaker and speaker != "unknown":
            self._active_episode.speaker = speaker
        return self._active_episode

    def add_assistant_turn(
        self,
        text: str,
        conversation_id: str = "",
        trace_id: str = "",
        request_id: str = "",
        output_id: str = "",
        conversation_entry_id: str = "",
        response_entry_id: str = "",
        root_entry_id: str = "",
    ) -> Episode | None:
        """Add an assistant response to the current episode."""
        if not self._active_episode:
            return None
        self._active_episode.add_turn(
            "assistant",
            text,
            conversation_id=conversation_id,
            trace_id=trace_id,
            request_id=request_id,
            output_id=output_id,
            conversation_entry_id=conversation_entry_id,
            response_entry_id=response_entry_id,
            root_entry_id=root_entry_id,
        )
        return self._active_episode

    def remove_last_user_turn_if_match(self, text: str) -> bool:
        """Remove trailing user turn when caller wants a memory-neutral turn.

        Returns True only if the active episode's last turn is a user turn and
        matches ``text`` (trimmed). If the active episode becomes empty, it is
        dropped from the in-memory queue.
        """
        ep = self._active_episode
        if not ep or not ep.turns:
            return False
        last = ep.turns[-1]
        if last.role != "user":
            return False
        if text and last.text.strip() != text.strip():
            return False
        ep.turns.pop()
        if ep.turns:
            return True
        if self._episodes and self._episodes[-1].id == ep.id:
            self._episodes.pop()
        self._active_episode = None
        return True

    def end_episode(self, outcome: str = "") -> Episode | None:
        """End the current episode and generate a summary."""
        if not self._active_episode:
            return None

        ep = self._active_episode
        ep.is_active = False
        ep.ended_at = time.time()
        if outcome:
            ep.outcome = outcome

        if not ep.topic and ep.turns:
            ep.topic = self._extract_topic(ep)

        if not ep.summary:
            ep.summary = self._generate_summary(ep)

        self._active_episode = None
        self.save()
        self._index_episode(ep)
        logger.debug("Ended episode %s: %s", ep.id, ep.topic[:50])
        return ep

    def get_active_episode(self) -> Episode | None:
        return self._active_episode

    def get_recent_episodes(self, count: int = 5) -> list[Episode]:
        completed = [e for e in self._episodes if not e.is_active and e.turn_count() >= MIN_TURNS_FOR_EPISODE]
        return list(completed)[-count:]

    def find_episodes_about(self, topic: str, limit: int = 3) -> list[Episode]:
        """Find past episodes related to a topic (keyword match)."""
        topic_lower = topic.lower()
        matches = []
        for ep in reversed(list(self._episodes)):
            if ep.is_active:
                continue
            text_blob = (ep.topic + " " + ep.summary + " " + " ".join(t.text for t in ep.turns)).lower()
            if topic_lower in text_blob:
                matches.append(ep)
                if len(matches) >= limit:
                    break
        return matches

    def find_episodes_semantic(self, query: str, limit: int = 3) -> list[Episode]:
        """Find past episodes by semantic similarity using the vector store."""
        from memory.search import get_vector_store
        vs = get_vector_store()
        if not vs or not vs.available:
            return self.find_episodes_about(query, limit)

        results = vs.search(query, top_k=limit * 2, min_weight=0.0)
        ep_map = {ep.id: ep for ep in self._episodes if not ep.is_active}
        matches = []
        for r in results:
            mid = r.get("memory_id", "")
            if mid.startswith("episode:"):
                ep_id = mid[len("episode:"):]
                if ep_id in ep_map and ep_map[ep_id] not in matches:
                    matches.append(ep_map[ep_id])
                    if len(matches) >= limit:
                        break
        if not matches:
            return self.find_episodes_about(query, limit)
        return matches

    def get_conversation_context(self, max_episodes: int = 3) -> str:
        """Build a concise context string from recent episodes for LLM injection."""
        recent = self.get_recent_episodes(max_episodes)
        if not recent:
            return ""

        parts = []
        for ep in recent:
            speaker = f" with {ep.speaker}" if ep.speaker != "unknown" else ""
            emotions = ", ".join(ep.emotion_arc[:3]) if ep.emotion_arc else "neutral"
            parts.append(f"- [{ep.topic}]{speaker} ({emotions}): {ep.summary[:200]}")

        return "Recent conversation history:\n" + "\n".join(parts)

    def get_recent_context(self, max_turns: int = 4) -> str:
        """Return the last N turns from the active episode as a compact string."""
        if not self._active_episode:
            return ""
        turns = self._active_episode.turns[-max_turns:]
        lines = []
        for t in turns:
            role = "User" if t.role == "user" else "Jarvis"
            lines.append(f"{role}: {t.text[:300]}")
        return "\n".join(lines)

    def search_episodes(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        """Search episodes by keyword. Returns dicts for tool output."""
        matches = self.find_episodes_about(query, limit)
        results = []
        for ep in matches:
            results.append({
                "id": ep.id,
                "topic": ep.topic,
                "summary": ep.summary[:200],
                "started": time.strftime("%Y-%m-%d %H:%M", time.localtime(ep.started_at)),
                "turns": ep.turn_count(),
            })
        return results

    def get_episode_count(self) -> int:
        return len(self._episodes)

    def link_memory(self, memory_id: str) -> None:
        """Link a memory to the current episode."""
        if self._active_episode and memory_id not in self._active_episode.memory_ids:
            self._active_episode.memory_ids.append(memory_id)

    def get_stats(self) -> dict[str, Any]:
        total = len(self._episodes)
        completed = sum(1 for e in self._episodes if not e.is_active)
        avg_turns = (sum(e.turn_count() for e in self._episodes) / total) if total > 0 else 0
        return {
            "total_episodes": total,
            "completed": completed,
            "active": self._active_episode.id if self._active_episode else None,
            "avg_turns": round(avg_turns, 1),
        }

    # --- Persistence ---

    def save(self) -> None:
        try:
            from memory.persistence import atomic_write_json
            data = [ep.to_dict() for ep in self._episodes]
            atomic_write_json(self._persist_path, data, default=str, indent=2)
        except Exception as exc:
            logger.error("Failed to save episodes: %s", exc)

    def load(self) -> int:
        if not os.path.exists(self._persist_path):
            return 0
        try:
            with open(self._persist_path) as f:
                data = json.load(f)

            count = 0
            for ep_data in data:
                turns = [
                    EpisodeTurn(
                        role=t["role"], text=t["text"],
                        emotion=t.get("emotion", "neutral"),
                        timestamp=t.get("timestamp", 0.0),
                        conversation_id=t.get("conversation_id", ""),
                        trace_id=t.get("trace_id", ""),
                        request_id=t.get("request_id", ""),
                        output_id=t.get("output_id", ""),
                        conversation_entry_id=t.get("conversation_entry_id", ""),
                        response_entry_id=t.get("response_entry_id", ""),
                        root_entry_id=t.get("root_entry_id", ""),
                    )
                    for t in ep_data.get("turns", [])
                ]
                ep = Episode(
                    id=ep_data["id"],
                    topic=ep_data.get("topic", ""),
                    turns=turns,
                    emotion_arc=ep_data.get("emotion_arc", []),
                    outcome=ep_data.get("outcome", ""),
                    speaker=ep_data.get("speaker", "unknown"),
                    memory_ids=ep_data.get("memory_ids", []),
                    started_at=ep_data.get("started_at", 0.0),
                    ended_at=ep_data.get("ended_at", 0.0),
                    is_active=False,
                    summary=ep_data.get("summary", ""),
                )
                self._episodes.append(ep)
                count += 1

                num = ep.id.split("_")[-1] if "_" in ep.id else "0"
                try:
                    self._episode_counter = max(self._episode_counter, int(num))
                except ValueError:
                    pass

            logger.info("Loaded %d episodes from %s", count, self._persist_path)
            return count
        except Exception as exc:
            logger.error("Failed to load episodes: %s", exc)
            return 0

    async def summarize_episode_llm(self, episode: Episode, ollama_client) -> str | None:
        """Use the LLM to generate a concise episode summary. Returns None on failure."""
        if not episode.turns or not ollama_client:
            return None
        try:
            transcript = "\n".join(
                f"{t.role}: {t.text[:500]}" for t in episode.turns[:50]
            )
            prompt = (
                "Summarize this conversation in 1-2 sentences. "
                "Focus on what was discussed and the outcome. Be concise.\n\n"
                f"{transcript}"
            )
            summary = await ollama_client.chat(
                [{"role": "user", "content": prompt}],
                system="You are a concise summarizer. Output only the summary, nothing else.",
            )
            summary = summary.strip()
            if summary and len(summary) < 500:
                episode.summary = summary
                self.save()
                self._index_episode(episode)
                logger.info("LLM summary for %s: %s", episode.id, summary[:80])
                return summary
        except Exception as exc:
            logger.debug("LLM episode summarization failed: %s", exc)
        return None

    @staticmethod
    def _index_episode(ep: Episode) -> None:
        """Index an episode's summary into the vector store for semantic retrieval."""
        if not ep.summary or ep.is_active:
            return
        try:
            from memory.search import get_vector_store
            vs = get_vector_store()
            if vs and vs.available:
                text = f"{ep.topic}. {ep.summary}"
                vs.add(f"episode:{ep.id}", text, "episode", 0.6)
        except Exception as exc:
            logger.debug("Failed to index episode %s: %s", ep.id, exc)

    # --- Helpers ---

    @staticmethod
    def _extract_topic(episode: Episode) -> str:
        if not episode.turns:
            return "general"
        user_msgs = [t.text for t in episode.turns if t.role == "user"]
        if not user_msgs:
            return "general"
        first = user_msgs[0]
        key_phrases = []
        for msg in user_msgs[:3]:
            words = msg.split()
            skip = {"what", "how", "why", "can", "you", "tell", "me", "about",
                    "the", "a", "is", "are", "do", "does", "i", "my", "please"}
            meaningful = [w for w in words if w.lower().strip("?.,!") not in skip and len(w) > 2]
            key_phrases.extend(meaningful[:4])
        if key_phrases:
            return " ".join(key_phrases[:6])
        return " ".join(first.split()[:8])

    @staticmethod
    def _generate_summary(episode: Episode) -> str:
        user_msgs = [t.text for t in episode.turns if t.role == "user"]
        assistant_msgs = [t.text for t in episode.turns if t.role == "assistant"]

        parts = []
        if user_msgs:
            parts.append(f"User asked about: {user_msgs[0][:200]}")
        if len(user_msgs) > 1:
            follow_ups = [m[:120] for m in user_msgs[1:3]]
            parts.append(f"Follow-ups: {'; '.join(follow_ups)}")
        if assistant_msgs:
            last = assistant_msgs[-1]
            parts.append(f"Resolved with: {last[:200]}")
        if episode.emotion_arc:
            parts.append(f"Mood: {' → '.join(episode.emotion_arc[:4])}")

        return ". ".join(parts) if parts else "Brief interaction"
