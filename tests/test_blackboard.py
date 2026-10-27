from ai_agent_orchestrator.blackboard import Blackboard
from ai_agent_orchestrator.dag import DAG
from ai_agent_orchestrator.events import OrchestrationEvent


def make_dag():
    return DAG.from_specs(
        [
            {"id": "R1", "description": "Research X", "capability": "research",
             "depends_on": [], "params": {"topic": "redis"}},
            {"id": "W1", "description": "Write it up", "capability": "writing",
             "depends_on": ["R1"], "params": {"instructions": "go"}},
        ]
    )


def test_create_run_persists_request_and_pending_subtasks(blackboard):
    dag = make_dag()
    blackboard.create_run("run-1", "Tell me about redis", dag)

    record = blackboard.get_run("run-1")
    assert record["request"] == "Tell me about redis"
    assert record["status"] == "running"
    assert [s["id"] for s in record["subtasks"]] == ["R1", "W1"]
    assert all(s["status"] == "pending" for s in record["subtasks"])


def test_update_subtask_and_complete_run_round_trip(blackboard):
    dag = make_dag()
    blackboard.create_run("run-2", "Tell me about redis", dag)

    dag.subtasks["R1"].status = "done"
    dag.subtasks["R1"].result = {"found": True, "title": "Redis"}
    blackboard.update_subtask("run-2", "R1", dag.subtasks["R1"].to_dict())

    blackboard.record_event(
        OrchestrationEvent(run_id="run-2", seq=1, type="subtask_completed", subtask_id="R1")
    )
    blackboard.complete_run("run-2", "completed", "Redis is an in-memory store. [R1]", ["R1"])

    record = blackboard.get_run("run-2")
    assert record["status"] == "completed"
    assert record["final_answer"] == "Redis is an in-memory store. [R1]"
    assert record["citations"] == ["R1"]
    r1 = next(s for s in record["subtasks"] if s["id"] == "R1")
    assert r1["status"] == "done"
    assert r1["result"]["title"] == "Redis"
    assert len(record["events"]) == 1
    assert record["events"][0]["type"] == "subtask_completed"


def test_persistence_survives_a_simulated_restart(db_path):
    dag = make_dag()
    first = Blackboard(db_path)
    first.create_run("run-3", "Tell me about redis", dag)
    dag.subtasks["R1"].status = "done"
    dag.subtasks["R1"].result = {"found": True}
    first.update_subtask("run-3", "R1", dag.subtasks["R1"].to_dict())
    first.complete_run("run-3", "completed", "Final answer text", ["R1"])

    # A brand-new Blackboard instance against the same file simulates a
    # process restart: nothing may be served from the first instance's
    # in-memory cache.
    second = Blackboard(db_path)
    record = second.get_run("run-3")
    assert record is not None
    assert record["request"] == "Tell me about redis"
    assert record["status"] == "completed"
    assert record["final_answer"] == "Final answer text"
    r1 = next(s for s in record["subtasks"] if s["id"] == "R1")
    assert r1["status"] == "done"
    assert r1["result"] == {"found": True}


def test_get_run_returns_none_for_unknown_run(blackboard):
    assert blackboard.get_run("does-not-exist") is None


def test_list_runs_orders_newest_first(blackboard):
    import time

    dag = make_dag()
    blackboard.create_run("run-a", "first", dag)
    blackboard.complete_run("run-a", "completed", "answer", [])
    time.sleep(0.01)
    blackboard.create_run("run-b", "second", make_dag())
    blackboard.complete_run("run-b", "completed", "answer", [])

    runs = blackboard.list_runs()
    ids = [r["run_id"] for r in runs]
    assert ids.index("run-b") < ids.index("run-a")
