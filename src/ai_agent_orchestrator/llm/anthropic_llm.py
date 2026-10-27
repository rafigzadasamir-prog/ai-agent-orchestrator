"""Live Anthropic-backed planning and synthesis.

Only ever instantiated when ``ANTHROPIC_API_KEY`` is set (see
``config.get_llm_backend``); no test or CI path constructs this class, so
the suite never makes a network call.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ai_agent_orchestrator.llm.base import LLMBackend, SubtaskResultView

DEFAULT_MODEL = "claude-opus-5"

_PLAN_SYSTEM_PROMPT = """You are the Planner in a multi-agent task-orchestration system.
Decompose the user's request into a DAG of subtasks. Respond with ONLY a JSON array
(no prose, no markdown fences). Each element must be an object with exactly these keys:
  - "id": a short unique string id, e.g. "R1", "A1", "W1"
  - "description": one sentence describing the subtask
  - "capability": one of {capabilities}
  - "depends_on": a list of ids of subtasks that must complete first (ids must already
     appear earlier in the array; use [] if there are no dependencies)
  - "params": an object of capability-specific arguments:
      - research: {{"topic": "<topic to look up>"}}
      - analysis: {{"expression": "<arithmetic expression, e.g. avg([1, 2, 3])>"}}
      - writing: {{"instructions": "<what to write>", "title": "<optional title or null>"}}
The plan must end with exactly one "writing" subtask that depends on every other subtask,
so its output can serve as the final combined write-up."""

_SYNTHESIS_SYSTEM_PROMPT = """You are the Synthesizer in a multi-agent task-orchestration
system. You are given the original request and the results of every subtask that
completed successfully. Write a final, well-organized answer to the request. Cite the
subtask id in square brackets (e.g. "[R1]") immediately after any claim drawn from it.
Respond with plain text only."""


class AnthropicLLM(LLMBackend):
    name = "anthropic"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        import anthropic  # imported lazily so the dependency is optional at import time

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate_plan(self, request: str, capabilities: tuple[str, ...]) -> list[dict[str, Any]]:
        system = _PLAN_SYSTEM_PROMPT.format(capabilities=", ".join(capabilities))
        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": request}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return _parse_plan_json(text)

    def generate_synthesis(self, request: str, results: list[SubtaskResultView]) -> str:
        payload = {
            "request": request,
            "results": [
                {
                    "id": r.id,
                    "description": r.description,
                    "capability": r.capability,
                    "result": r.result,
                }
                for r in results
            ],
        }
        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=_SYNTHESIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()


def _parse_plan_json(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    # Tolerate an accidental ```json ... ``` fence even though the prompt forbids it.
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        plan = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Planner LLM did not return valid JSON: {text!r}") from exc
    if not isinstance(plan, list):
        raise ValueError(f"Planner LLM must return a JSON array, got: {type(plan).__name__}")
    return plan
