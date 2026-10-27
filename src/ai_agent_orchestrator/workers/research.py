"""ResearchWorker: deterministic, offline lookups against a bundled knowledge base.

No network access, ever -- this is the same "offline fixture" shape used
elsewhere in the portfolio, scoped here to one worker type among several.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_agent_orchestrator.knowledge_base import find_topic, load_knowledge_base
from ai_agent_orchestrator.workers.base import Worker

if TYPE_CHECKING:
    from ai_agent_orchestrator.dag import DAG, Subtask


class ResearchWorker(Worker):
    capability = "research"

    def run(self, subtask: Subtask, dag: DAG) -> dict[str, Any]:
        del dag  # research needs no upstream results
        topic = str(subtask.params.get("topic", "")).strip()
        if not topic:
            raise ValueError("ResearchWorker requires a non-empty 'topic' param.")

        key = find_topic(topic)
        if key is None:
            available = sorted(load_knowledge_base().keys())
            return {
                "topic": topic,
                "found": False,
                "message": (
                    f"No offline knowledge-base entry found for '{topic}'. "
                    f"Known topics: {', '.join(available)}."
                ),
            }

        entry = load_knowledge_base()[key]
        return {
            "topic": key,
            "found": True,
            "title": entry["title"],
            "category": entry["category"],
            "summary": entry["summary"],
            "key_points": list(entry["key_points"]),
        }
