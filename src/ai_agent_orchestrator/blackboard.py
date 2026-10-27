"""The Blackboard: shared run state, kept in memory and mirrored to SQLite.

Every write goes through ``_memory`` (so a live run's in-process state is
always current) and through a fresh SQLite connection (so a crashed or
restarted process can still answer ``get_run`` for any run that was ever
started). A new connection is opened per call rather than held open, which
keeps this class trivially safe to share across concurrent runs on the
single-threaded asyncio event loop.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from ai_agent_orchestrator.dag import DAG
from ai_agent_orchestrator.events import OrchestrationEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    request TEXT NOT NULL,
    status TEXT NOT NULL,
    final_answer TEXT,
    citations_json TEXT,
    created_at REAL NOT NULL,
    completed_at REAL
);

CREATE TABLE IF NOT EXISTS subtasks (
    run_id TEXT NOT NULL,
    subtask_id TEXT NOT NULL,
    description TEXT NOT NULL,
    capability TEXT NOT NULL,
    depends_on_json TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    created_at REAL,
    started_at REAL,
    completed_at REAL,
    PRIMARY KEY (run_id, subtask_id)
);

CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    type TEXT NOT NULL,
    subtask_id TEXT,
    data_json TEXT NOT NULL,
    timestamp REAL NOT NULL,
    PRIMARY KEY (run_id, seq)
);
"""


class Blackboard:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._memory: dict[str, dict[str, Any]] = {}
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- writes --------------------------------------------------------

    def create_run(self, run_id: str, request: str, dag: DAG) -> None:
        now = time.time()
        record = {
            "run_id": run_id,
            "request": request,
            "status": "running",
            "final_answer": None,
            "citations": [],
            "created_at": now,
            "completed_at": None,
            "subtasks": {sid: dag.subtasks[sid].to_dict() for sid in dag.order},
            "subtask_order": list(dag.order),
            "events": [],
        }
        self._memory[run_id] = record
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, request, status, final_answer, citations_json, "
                "created_at, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, request, "running", None, json.dumps([]), now, None),
            )
            for sid in dag.order:
                subtask = dag.subtasks[sid]
                conn.execute(
                    "INSERT INTO subtasks (run_id, subtask_id, description, capability, "
                    "depends_on_json, status, result_json, error, created_at, started_at, "
                    "completed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        sid,
                        subtask.description,
                        subtask.capability,
                        json.dumps(subtask.depends_on),
                        subtask.status,
                        json.dumps(subtask.result),
                        subtask.error,
                        subtask.created_at,
                        subtask.started_at,
                        subtask.completed_at,
                    ),
                )

    def update_subtask(self, run_id: str, subtask_id: str, subtask_dict: dict[str, Any]) -> None:
        record = self._memory.setdefault(run_id, {"subtasks": {}, "subtask_order": []})
        record["subtasks"][subtask_id] = subtask_dict
        with self._connect() as conn:
            conn.execute(
                "UPDATE subtasks SET status=?, result_json=?, error=?, started_at=?, "
                "completed_at=? WHERE run_id=? AND subtask_id=?",
                (
                    subtask_dict["status"],
                    json.dumps(subtask_dict["result"]),
                    subtask_dict["error"],
                    subtask_dict["started_at"],
                    subtask_dict["completed_at"],
                    run_id,
                    subtask_id,
                ),
            )

    def record_event(self, event: OrchestrationEvent) -> None:
        record = self._memory.setdefault(event.run_id, {"subtasks": {}, "subtask_order": []})
        record.setdefault("events", []).append(event.to_dict())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events (run_id, seq, type, subtask_id, data_json, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event.run_id,
                    event.seq,
                    event.type,
                    event.subtask_id,
                    json.dumps(event.data),
                    event.timestamp,
                ),
            )

    def complete_run(
        self, run_id: str, status: str, final_answer: str, citations: list[str]
    ) -> None:
        now = time.time()
        record = self._memory.setdefault(run_id, {"subtasks": {}, "subtask_order": []})
        record.update(
            status=status, final_answer=final_answer, citations=citations, completed_at=now
        )
        with self._connect() as conn:
            # Upsert rather than plain UPDATE: if planning itself failed, create_run()
            # was never called, so there may be no row yet for this run_id.
            conn.execute(
                "INSERT INTO runs (run_id, request, status, final_answer, citations_json, "
                "created_at, completed_at) VALUES (?, '', ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET status=excluded.status, "
                "final_answer=excluded.final_answer, citations_json=excluded.citations_json, "
                "completed_at=excluded.completed_at",
                (run_id, status, final_answer, json.dumps(citations), now, now),
            )

    # -- reads -----------------------------------------------------------

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        if run_id in self._memory:
            record = self._memory[run_id]
            order = record.get("subtask_order") or sorted(record["subtasks"])
            return {
                "run_id": run_id,
                "request": record.get("request"),
                "status": record.get("status"),
                "final_answer": record.get("final_answer"),
                "citations": record.get("citations", []),
                "created_at": record.get("created_at"),
                "completed_at": record.get("completed_at"),
                "subtasks": [record["subtasks"][sid] for sid in order if sid in record["subtasks"]],
                "events": record.get("events", []),
            }
        return self._load_run_from_db(run_id)

    def _load_run_from_db(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            run_row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if run_row is None:
                return None
            subtask_rows = conn.execute(
                "SELECT * FROM subtasks WHERE run_id=? ORDER BY rowid", (run_id,)
            ).fetchall()
            event_rows = conn.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY seq", (run_id,)
            ).fetchall()

        return {
            "run_id": run_row["run_id"],
            "request": run_row["request"],
            "status": run_row["status"],
            "final_answer": run_row["final_answer"],
            "citations": json.loads(run_row["citations_json"] or "[]"),
            "created_at": run_row["created_at"],
            "completed_at": run_row["completed_at"],
            "subtasks": [
                {
                    "id": row["subtask_id"],
                    "description": row["description"],
                    "capability": row["capability"],
                    "depends_on": json.loads(row["depends_on_json"]),
                    "status": row["status"],
                    "result": json.loads(row["result_json"]) if row["result_json"] else None,
                    "error": row["error"],
                    "created_at": row["created_at"],
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                }
                for row in subtask_rows
            ],
            "events": [
                {
                    "run_id": row["run_id"],
                    "seq": row["seq"],
                    "type": row["type"],
                    "subtask_id": row["subtask_id"],
                    "data": json.loads(row["data_json"]),
                    "timestamp": row["timestamp"],
                }
                for row in event_rows
            ],
        }

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id, request, status, created_at, completed_at FROM runs "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
