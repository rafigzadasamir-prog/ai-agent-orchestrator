"""The Orchestrator: walks the DAG, dispatches ready subtasks, resolves it.

On each iteration it dispatches every subtask whose dependencies are all
satisfied, concurrently, to the matching Worker. If a subtask fails, every
subtask that transitively depends on it is marked failed too (never
silently dropped), so the DAG always reaches a terminal state. Once no
subtask is pending, the Synthesizer produces the final answer and the Run
is persisted as complete.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from ai_agent_orchestrator.blackboard import Blackboard
from ai_agent_orchestrator.dag import DAG, Subtask
from ai_agent_orchestrator.events import OrchestrationEvent
from ai_agent_orchestrator.planner import Planner
from ai_agent_orchestrator.synthesizer import Synthesizer
from ai_agent_orchestrator.workers.base import Worker

EventSink = Callable[[OrchestrationEvent], Awaitable[None]]


class Orchestrator:
    def __init__(
        self,
        planner: Planner,
        workers: dict[str, Worker],
        synthesizer: Synthesizer,
        blackboard: Blackboard,
        worker_delay: float = 0.0,
    ) -> None:
        self.planner = planner
        self.workers = workers
        self.synthesizer = synthesizer
        self.blackboard = blackboard
        self.worker_delay = worker_delay

    async def run(
        self,
        request: str,
        run_id: str | None = None,
        on_event: EventSink | None = None,
    ) -> dict[str, Any]:
        run_id = run_id or uuid.uuid4().hex[:12]
        seq_counter = {"n": 0}

        async def emit(event_type: str, subtask_id: str | None = None, **data: Any) -> None:
            seq_counter["n"] += 1
            event = OrchestrationEvent(
                run_id=run_id, seq=seq_counter["n"], type=event_type, subtask_id=subtask_id,
                data=data,
            )
            self.blackboard.record_event(event)
            if on_event is not None:
                await on_event(event)

        await emit("run_started", request=request)

        try:
            dag = self.planner.decompose(request)
        except Exception as exc:
            self.blackboard.complete_run(run_id, "failed", f"Planning failed: {exc}", [])
            await emit("run_failed", error=str(exc))
            raise

        self.blackboard.create_run(run_id, request, dag)
        await emit("dag_created", subtasks=dag.to_list())

        while not dag.is_terminal():
            ready = dag.ready_subtasks()
            if ready:
                await self._dispatch(run_id, dag, ready, emit)
                continue

            blocked = dag.blocked_pending()
            if not blocked:
                # Nothing ready and nothing blocked, yet not terminal: should be
                # unreachable given DAG validation, but fail safe instead of hanging.
                break
            for subtask in blocked:
                subtask.status = "failed"
                subtask.error = "Skipped: an upstream dependency failed."
                subtask.completed_at = time.time()
                self.blackboard.update_subtask(run_id, subtask.id, subtask.to_dict())
                await emit("subtask_failed", subtask_id=subtask.id, error=subtask.error)

        final_answer, citations = self.synthesizer.synthesize(request, dag)
        any_done = any(st.status == "done" for st in dag.subtasks.values())
        run_status = "completed" if any_done else "failed"
        self.blackboard.complete_run(run_id, run_status, final_answer, citations)
        await emit("run_completed", status=run_status, citations=citations)

        result = self.blackboard.get_run(run_id)
        assert result is not None
        return result

    async def _dispatch(
        self, run_id: str, dag: DAG, ready: list[Subtask], emit: Callable[..., Awaitable[None]]
    ) -> None:
        for subtask in ready:
            subtask.status = "running"
            subtask.started_at = time.time()
            self.blackboard.update_subtask(run_id, subtask.id, subtask.to_dict())
            await emit("subtask_started", subtask_id=subtask.id, capability=subtask.capability)

        await asyncio.gather(*(self._execute(subtask, dag) for subtask in ready))

        for subtask in ready:
            self.blackboard.update_subtask(run_id, subtask.id, subtask.to_dict())
            if subtask.status == "done":
                await emit(
                    "subtask_completed",
                    subtask_id=subtask.id,
                    result=subtask.result,
                    duration=(subtask.completed_at or 0) - (subtask.started_at or 0),
                )
            else:
                await emit("subtask_failed", subtask_id=subtask.id, error=subtask.error)

    async def _execute(self, subtask: Subtask, dag: DAG) -> None:
        worker = self.workers.get(subtask.capability)
        try:
            if worker is None:
                raise ValueError(f"No worker registered for capability {subtask.capability!r}")
            if self.worker_delay:
                await asyncio.sleep(self.worker_delay)
            result = await asyncio.to_thread(worker.run, subtask, dag)
            subtask.result = result
            subtask.status = "done"
        except Exception as exc:  # a failing worker must not crash the whole run
            subtask.error = str(exc)
            subtask.status = "failed"
        finally:
            subtask.completed_at = time.time()
