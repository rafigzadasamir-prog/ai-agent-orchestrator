"""Worker agent contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_agent_orchestrator.dag import DAG, Subtask


class Worker(ABC):
    """Executes one subtask. Implementations must be synchronous and pure
    with respect to their inputs (``subtask.params`` and completed upstream
    results reachable through ``dag``) so they can run in a worker thread."""

    capability: str

    @abstractmethod
    def run(self, subtask: Subtask, dag: DAG) -> dict[str, Any]:
        """Return a JSON-serializable result dict, or raise on failure."""
