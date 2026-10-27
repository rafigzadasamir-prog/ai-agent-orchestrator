"""Synthesizer agent: the final, cited answer over a resolved DAG."""

from __future__ import annotations

from ai_agent_orchestrator.dag import DAG
from ai_agent_orchestrator.llm.base import LLMBackend, SubtaskResultView


class Synthesizer:
    def __init__(self, llm: LLMBackend) -> None:
        self.llm = llm

    def synthesize(self, request: str, dag: DAG) -> tuple[str, list[str]]:
        """Combine every successfully completed subtask result into one answer.

        Returns ``(final_answer, citations)`` where ``citations`` is the
        ordered list of subtask ids the answer draws on.
        """
        views = [
            SubtaskResultView(
                id=sid,
                description=dag.subtasks[sid].description,
                capability=dag.subtasks[sid].capability,
                result=dag.subtasks[sid].result,
            )
            for sid in dag.order
            if dag.subtasks[sid].status == "done"
        ]
        final_answer = self.llm.generate_synthesis(request, views)
        return final_answer, [v.id for v in views]
