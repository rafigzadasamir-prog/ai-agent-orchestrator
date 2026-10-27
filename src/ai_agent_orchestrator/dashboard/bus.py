"""A tiny async pub/sub bus fanning orchestration events out to WebSocket clients."""

from __future__ import annotations

import asyncio

from ai_agent_orchestrator.events import OrchestrationEvent


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    async def publish(self, event: OrchestrationEvent) -> None:
        for queue in list(self._subscribers):
            await queue.put(event.to_dict())
