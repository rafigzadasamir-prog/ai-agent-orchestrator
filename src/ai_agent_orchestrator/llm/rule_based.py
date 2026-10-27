"""Deterministic, fully offline LLM backend.

``RuleBasedLLM`` never makes a network call. It decomposes a free-form
request into a DAG using a handful of pattern-matching passes (which
knowledge-base topics are named, which "average of ..." style calculations
are requested, where the "write ..." instruction begins) and synthesizes the
final answer by deterministically combining completed subtask results. This
is what the test suite and CI run against, and what the whole pipeline falls
back to whenever ``ANTHROPIC_API_KEY`` is not set.
"""

from __future__ import annotations

import re
from typing import Any

from ai_agent_orchestrator.knowledge_base import topics_mentioned_in
from ai_agent_orchestrator.llm.base import LLMBackend, SubtaskResultView

# Keyword -> safe-evaluator function name (see workers.analysis.SAFE_FUNCTIONS).
_AGGREGATE_KEYWORDS = {
    "average": "avg",
    "mean": "avg",
    "sum": "sum",
    "total": "sum",
    "product": "product",
    "maximum": "max",
    "max": "max",
    "minimum": "min",
    "min": "min",
    "median": "median",
}
_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(_AGGREGATE_KEYWORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
_DIRECT_EXPR_PATTERN = re.compile(
    r"\b(?:calculate|compute)\b\s+([0-9][0-9\s.+\-*/()]*[0-9)])", re.IGNORECASE
)
_WRITE_SPLIT_PATTERN = re.compile(r",?\s*(?:and\s+)?(?:then\s+)?\bwrite\b", re.IGNORECASE)


class RuleBasedLLM(LLMBackend):
    name = "rule-based"

    def generate_plan(self, request: str, capabilities: tuple[str, ...]) -> list[dict[str, Any]]:
        del capabilities  # the rule-based planner only ever emits research/analysis/writing
        head, writing_instructions = self._split_writing_instruction(request)

        specs: list[dict[str, Any]] = []
        research_ids = self._research_specs(head, specs)
        analysis_ids = self._analysis_specs(head, specs)

        if not specs:
            # Nothing to research or compute: the whole request is a direct
            # writing task with no upstream dependencies.
            specs.append(
                {
                    "id": "W1",
                    "description": f"Write a response to: {request}",
                    "capability": "writing",
                    "depends_on": [],
                    "params": {"instructions": request, "title": None},
                }
            )
            return specs

        writing_deps = research_ids + analysis_ids
        specs.append(
            {
                "id": "W1",
                "description": "Compile the research and analysis findings into a final write-up.",
                "capability": "writing",
                "depends_on": writing_deps,
                "params": {"instructions": writing_instructions, "title": None},
            }
        )
        return specs

    def generate_synthesis(self, request: str, results: list[SubtaskResultView]) -> str:
        if not results:
            return (
                "No subtasks completed successfully, so no answer could be "
                "synthesized for this request."
            )

        writing_results = [r for r in results if r.capability == "writing" and r.result]
        lines = [f"Request: {request}", ""]

        if writing_results:
            lines.append(writing_results[-1].result["compiled_text"])
        else:
            lines.append("Findings from the completed subtasks:")
            for r in results:
                lines.append(f"- [{r.id}] {_one_line_summary(r)}")

        citation_ids = ", ".join(r.id for r in results)
        lines.append("")
        lines.append(f"(Synthesized from subtasks: {citation_ids}.)")
        return "\n".join(lines)

    # -- plan construction helpers -----------------------------------------

    @staticmethod
    def _split_writing_instruction(request: str) -> tuple[str, str]:
        match = _WRITE_SPLIT_PATTERN.search(request)
        if not match:
            return request, "Write a clear, well-structured summary of all findings."
        head = request[: match.start()]
        tail = "Write" + request[match.end() :]
        return head, tail.strip() or "Write a clear, well-structured summary of all findings."

    @staticmethod
    def _research_specs(head: str, specs: list[dict[str, Any]]) -> list[str]:
        topics = topics_mentioned_in(head)
        ids = []
        for i, topic in enumerate(topics, start=1):
            subtask_id = f"R{i}"
            specs.append(
                {
                    "id": subtask_id,
                    "description": f"Research {topic}.",
                    "capability": "research",
                    "depends_on": [],
                    "params": {"topic": topic},
                }
            )
            ids.append(subtask_id)
        return ids

    @staticmethod
    def _analysis_specs(head: str, specs: list[dict[str, Any]]) -> list[str]:
        ids: list[str] = []
        keyword_matches = list(_KEYWORD_PATTERN.finditer(head))
        for i, match in enumerate(keyword_matches, start=1):
            start = match.end()
            end = keyword_matches[i].start() if i < len(keyword_matches) else len(head)
            segment = head[start:end]
            numbers = _NUMBER_PATTERN.findall(segment)
            if not numbers:
                continue
            func = _AGGREGATE_KEYWORDS[match.group(1).lower()]
            expression = f"{func}([{', '.join(numbers)}])"
            subtask_id = f"A{i}"
            op_name = match.group(1).lower()
            numbers_text = ", ".join(numbers)
            specs.append(
                {
                    "id": subtask_id,
                    "description": f"Calculate the {op_name} of {numbers_text}.",
                    "capability": "analysis",
                    "depends_on": [],
                    "params": {"expression": expression},
                }
            )
            ids.append(subtask_id)

        if not ids:
            direct = _DIRECT_EXPR_PATTERN.search(head)
            if direct:
                expression = direct.group(1).strip()
                specs.append(
                    {
                        "id": "A1",
                        "description": f"Calculate {expression}.",
                        "capability": "analysis",
                        "depends_on": [],
                        "params": {"expression": expression},
                    }
                )
                ids.append("A1")
        return ids


def _one_line_summary(result: SubtaskResultView) -> str:
    payload = result.result or {}
    if result.capability == "research":
        if payload.get("found"):
            return str(payload.get("summary", ""))
        return str(payload.get("message", "Topic not found."))
    if result.capability == "analysis":
        return f"{payload.get('expression', '')} = {payload.get('value', '')}"
    return str(payload.get("compiled_text", ""))
