"""WritingWorker: compiles upstream subtask results into formatted text.

This worker does no generation of its own -- it is a deterministic
formatter that reads the completed results of its declared dependencies
straight out of the in-memory DAG and arranges them into cited prose. The
Synthesizer (a separate, LLM-backed stage) runs after the whole DAG resolves
and may draw on this worker's output as one of several inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_agent_orchestrator.workers.base import Worker

if TYPE_CHECKING:
    from ai_agent_orchestrator.dag import DAG, Subtask


class WritingWorker(Worker):
    capability = "writing"

    def run(self, subtask: Subtask, dag: DAG) -> dict[str, Any]:
        instructions = str(subtask.params.get("instructions") or "Summarize the findings.")
        title = subtask.params.get("title")

        sections: list[str] = []
        sources: list[str] = []
        for dep_id in subtask.depends_on:
            dep = dag.subtasks[dep_id]
            sources.append(dep_id)
            sections.append(_render_section(dep_id, dep.capability, dep.result))

        parts: list[str] = []
        if title:
            parts.append(f"# {title}")
        parts.append(instructions)
        if sections:
            parts.append("")
            parts.extend(sections)
        compiled_text = "\n".join(parts).strip()

        return {"instructions": instructions, "compiled_text": compiled_text, "sources": sources}


def _render_section(dep_id: str, capability: str, result: dict[str, Any] | None) -> str:
    result = result or {}
    if capability == "research":
        if result.get("found"):
            points = "; ".join(result.get("key_points", []))
            title, summary = result.get("title"), result.get("summary")
            return f"[{dep_id}] {title}: {summary} Key points: {points}."
        return f"[{dep_id}] {result.get('message', 'No information found.')}"
    if capability == "analysis":
        return f"[{dep_id}] Computed {result.get('expression')} = {result.get('value')}."
    # A writing subtask depending on another writing subtask: inline its text.
    return f"[{dep_id}] {result.get('compiled_text', '')}"
