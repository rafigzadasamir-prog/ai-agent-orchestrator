"""LLM backend abstraction.

Both the Planner (decomposition) and the Synthesizer (final write-up) talk to
an ``LLMBackend`` instead of a concrete provider. This keeps the two real
implementations -- a live ``AnthropicLLM`` and a deterministic offline
``RuleBasedLLM`` -- fully interchangeable, so the orchestration pipeline is
identical regardless of which one is wired in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class SubtaskResultView:
    """Read-only view of a completed subtask, as handed to the Synthesizer."""

    id: str
    description: str
    capability: str
    result: dict[str, Any] | None


class LLMBackend(ABC):
    """A planning + synthesis backend. Implementations must be deterministic
    enough to be swapped in for offline testing (``RuleBasedLLM``) or genuinely
    generative (``AnthropicLLM``)."""

    name: str

    @abstractmethod
    def generate_plan(self, request: str, capabilities: tuple[str, ...]) -> list[dict[str, Any]]:
        """Decompose ``request`` into an ordered list of subtask specs.

        Each spec is a dict with keys: id, description, capability,
        depends_on (list of ids already defined earlier in the list), and
        params (capability-specific arguments for the Worker).
        """

    @abstractmethod
    def generate_synthesis(self, request: str, results: list[SubtaskResultView]) -> str:
        """Produce the final combined answer from completed subtask results.

        Implementations should cite subtask ids (e.g. "[R1]") inline so the
        reader can trace each claim back to the subtask that produced it.
        """
