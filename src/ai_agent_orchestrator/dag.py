"""The task DAG: subtasks, dependency validation, and readiness queries.

A plan produced by the Planner is a directed acyclic graph of ``Subtask``
nodes. This module owns the data model plus the structural validation
(unknown capability, missing/duplicate ids, cycles) that must happen before
the scheduler is allowed to touch the graph.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

Capability = Literal["research", "analysis", "writing"]
CAPABILITIES: tuple[Capability, ...] = ("research", "analysis", "writing")

SubtaskStatus = Literal["pending", "running", "done", "failed"]


class DAGValidationError(ValueError):
    """Raised when a planner-produced spec cannot form a valid DAG."""


@dataclass
class Subtask:
    """A single node in the task DAG."""

    id: str
    description: str
    capability: str
    depends_on: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    status: SubtaskStatus = "pending"
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "capability": self.capability,
            "depends_on": list(self.depends_on),
            "params": dict(self.params),
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class DAG:
    """A validated collection of subtasks keyed by id, in planner order."""

    subtasks: dict[str, Subtask]
    order: list[str]

    @classmethod
    def from_specs(cls, specs: list[dict[str, Any]]) -> DAG:
        """Build and validate a DAG from planner-produced subtask specs.

        Raises ``DAGValidationError`` for an empty plan, duplicate/unknown
        ids, an unknown capability, a dependency on a non-existent subtask,
        or a dependency cycle.
        """
        if not specs:
            raise DAGValidationError("A plan must contain at least one subtask.")

        subtasks: dict[str, Subtask] = {}
        order: list[str] = []
        for spec in specs:
            subtask_id = spec.get("id")
            if not subtask_id or not isinstance(subtask_id, str):
                raise DAGValidationError(f"Subtask spec is missing a valid id: {spec!r}")
            if subtask_id in subtasks:
                raise DAGValidationError(f"Duplicate subtask id: {subtask_id!r}")
            capability = spec.get("capability")
            if capability not in CAPABILITIES:
                raise DAGValidationError(
                    f"Subtask {subtask_id!r} has unknown capability {capability!r}; "
                    f"expected one of {CAPABILITIES}"
                )
            depends_on = list(spec.get("depends_on") or [])
            if subtask_id in depends_on:
                raise DAGValidationError(f"Subtask {subtask_id!r} cannot depend on itself.")
            subtasks[subtask_id] = Subtask(
                id=subtask_id,
                description=str(spec.get("description", "")),
                capability=capability,
                depends_on=depends_on,
                params=dict(spec.get("params") or {}),
            )
            order.append(subtask_id)

        for subtask in subtasks.values():
            for dep in subtask.depends_on:
                if dep not in subtasks:
                    raise DAGValidationError(
                        f"Subtask {subtask.id!r} depends on unknown subtask {dep!r}."
                    )

        cls._check_acyclic(subtasks)
        return cls(subtasks=subtasks, order=order)

    @staticmethod
    def _check_acyclic(subtasks: dict[str, Subtask]) -> None:
        """Kahn's algorithm: if we can't consume every node, there is a cycle."""
        in_degree = {sid: 0 for sid in subtasks}
        for subtask in subtasks.values():
            in_degree[subtask.id] += len(subtask.depends_on)

        # dependents[x] = subtasks that depend on x
        dependents: dict[str, list[str]] = {sid: [] for sid in subtasks}
        for subtask in subtasks.values():
            for dep in subtask.depends_on:
                dependents[dep].append(subtask.id)

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            current = queue.pop()
            visited += 1
            for dependent in dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if visited != len(subtasks):
            cyclic = sorted(sid for sid, deg in in_degree.items() if deg > 0)
            raise DAGValidationError(f"Dependency cycle detected among subtasks: {cyclic}")

    def ready_subtasks(self) -> list[Subtask]:
        """Pending subtasks whose dependencies have all completed successfully."""
        ready = []
        for sid in self.order:
            subtask = self.subtasks[sid]
            if subtask.status != "pending":
                continue
            if all(self.subtasks[dep].status == "done" for dep in subtask.depends_on):
                ready.append(subtask)
        return ready

    def blocked_pending(self) -> list[Subtask]:
        """Pending subtasks that can never become ready because a dependency failed."""
        blocked = []
        for sid in self.order:
            subtask = self.subtasks[sid]
            if subtask.status != "pending":
                continue
            if any(self.subtasks[dep].status == "failed" for dep in subtask.depends_on):
                blocked.append(subtask)
        return blocked

    def is_terminal(self) -> bool:
        return all(st.status in ("done", "failed") for st in self.subtasks.values())

    def to_list(self) -> list[dict[str, Any]]:
        return [self.subtasks[sid].to_dict() for sid in self.order]
