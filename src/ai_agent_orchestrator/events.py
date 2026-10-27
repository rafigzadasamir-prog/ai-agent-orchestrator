"""Orchestration event contract.

Every state change the scheduler makes is emitted as an ``OrchestrationEvent``.
The same event objects are persisted to the Blackboard and broadcast live over
the dashboard's WebSocket, so this is the single source of truth for the
wire/storage shape of an event.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

EVENT_TYPES = (
    "run_started",
    "dag_created",
    "subtask_started",
    "subtask_completed",
    "subtask_failed",
    "run_completed",
    "run_failed",
)


@dataclass
class OrchestrationEvent:
    run_id: str
    seq: int
    type: str
    subtask_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.type not in EVENT_TYPES:
            raise ValueError(f"Unknown event type: {self.type!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "seq": self.seq,
            "type": self.type,
            "subtask_id": self.subtask_id,
            "data": self.data,
            "timestamp": self.timestamp,
        }
