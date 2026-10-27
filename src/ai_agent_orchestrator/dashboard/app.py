"""The FastAPI dashboard: WebSocket event stream + run HTTP endpoints.

``create_app`` is an application factory, deliberately with no eager
module-level ``app = create_app()`` call: constructing an app touches disk
(it opens the Blackboard's SQLite file) and would run on every import,
including from the test suite. Tests call ``create_app`` with an isolated
temp-file database; production serving goes through
``ai_agent_orchestrator.dashboard.server:app``, the one place that
instantiates the default app.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from ai_agent_orchestrator.blackboard import Blackboard
from ai_agent_orchestrator.config import default_db_path, get_llm_backend, worker_delay_seconds
from ai_agent_orchestrator.dashboard.bus import EventBus
from ai_agent_orchestrator.planner import Planner
from ai_agent_orchestrator.scheduler import Orchestrator
from ai_agent_orchestrator.synthesizer import Synthesizer
from ai_agent_orchestrator.workers import default_workers

_STATIC_DIR = Path(__file__).parent / "static"


class RunRequest(BaseModel):
    request: str


def create_app(db_path: str | None = None, worker_delay: float | None = None) -> FastAPI:
    llm = get_llm_backend()
    blackboard = Blackboard(db_path or default_db_path())
    orchestrator = Orchestrator(
        planner=Planner(llm),
        workers=default_workers(),
        synthesizer=Synthesizer(llm),
        blackboard=blackboard,
        worker_delay=worker_delay_seconds() if worker_delay is None else worker_delay,
    )
    bus = EventBus()

    app = FastAPI(title="AI Agent Orchestrator", version="0.1.0")
    app.state.blackboard = blackboard
    app.state.orchestrator = orchestrator
    app.state.bus = bus

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((_STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.post("/run")
    async def start_run(payload: RunRequest) -> JSONResponse:
        request_text = payload.request.strip()
        if not request_text:
            return JSONResponse({"detail": "request must not be empty"}, status_code=422)

        run_id = uuid.uuid4().hex[:12]

        async def execute() -> None:
            # Any failure is already captured in the persisted run record
            # (status="failed"); swallow it here so the background task
            # doesn't log an unhandled exception.
            with contextlib.suppress(Exception):
                await orchestrator.run(request_text, run_id=run_id, on_event=bus.publish)

        task = asyncio.create_task(execute())
        app.state.background_tasks = getattr(app.state, "background_tasks", set())
        app.state.background_tasks.add(task)
        task.add_done_callback(app.state.background_tasks.discard)

        # Briefly yield so /runs/{id} is guaranteed a row immediately after this
        # returns, without waiting for the whole orchestration to finish.
        await asyncio.sleep(0)
        return JSONResponse({"run_id": run_id, "status": "started"})

    @app.get("/runs")
    async def list_runs(limit: int = 50) -> list[dict[str, Any]]:
        return blackboard.list_runs(limit=limit)

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str) -> JSONResponse:
        record = blackboard.get_run(run_id)
        if record is None:
            return JSONResponse({"detail": "run not found"}, status_code=404)
        return JSONResponse(record)

    @app.websocket("/ws")
    async def ws_events(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = bus.subscribe()
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event)
        except WebSocketDisconnect:
            pass
        finally:
            bus.unsubscribe(queue)

    return app
