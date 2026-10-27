"""Planner agent: turns a free-form request into a validated task DAG."""

from __future__ import annotations

from ai_agent_orchestrator.dag import CAPABILITIES, DAG
from ai_agent_orchestrator.llm.base import LLMBackend


class Planner:
    def __init__(self, llm: LLMBackend) -> None:
        self.llm = llm

    def decompose(self, request: str) -> DAG:
        specs = self.llm.generate_plan(request, CAPABILITIES)
        return DAG.from_specs(specs)
