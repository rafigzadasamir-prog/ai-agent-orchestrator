from fastapi.testclient import TestClient

from ai_agent_orchestrator.dashboard.app import create_app


def make_client(tmp_path):
    app = create_app(db_path=str(tmp_path / "dashboard-test.db"), worker_delay=0.0)
    return TestClient(app)


def test_index_serves_the_dashboard_html(tmp_path):
    with make_client(tmp_path) as client:
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "AI Agent Orchestrator" in res.text
        assert "/ws" in res.text


def test_run_then_runs_by_id_http_contract(tmp_path):
    with make_client(tmp_path) as client:
        res = client.post("/run", json={"request": "Write a haiku about the sea."})
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "started"
        run_id = body["run_id"]
        assert run_id

        record = _wait_for_completion(client, run_id)
        assert record["status"] == "completed"
        assert record["request"] == "Write a haiku about the sea."
        assert len(record["subtasks"]) == 1
        assert record["final_answer"]


def test_run_rejects_an_empty_request(tmp_path):
    with make_client(tmp_path) as client:
        res = client.post("/run", json={"request": "   "})
        assert res.status_code == 422


def test_runs_by_id_404s_for_an_unknown_run(tmp_path):
    with make_client(tmp_path) as client:
        res = client.get("/runs/does-not-exist")
        assert res.status_code == 404


def test_runs_list_reflects_completed_runs(tmp_path):
    with make_client(tmp_path) as client:
        res = client.post("/run", json={"request": "Write a short note about rain."})
        run_id = res.json()["run_id"]
        _wait_for_completion(client, run_id)

        listing = client.get("/runs").json()
        assert any(r["run_id"] == run_id for r in listing)


def test_websocket_broadcasts_the_full_event_sequence_for_a_run(tmp_path):
    with make_client(tmp_path) as client, client.websocket_connect("/ws") as ws:
        res = client.post(
            "/run",
            json={
                "request": "Research redis and write a short note about it."
            },
        )
        run_id = res.json()["run_id"]

        seen_types = []
        while True:
            event = ws.receive_json()
            assert event["run_id"] == run_id
            seen_types.append(event["type"])
            if event["type"] in ("run_completed", "run_failed"):
                break

        assert seen_types[0] == "run_started"
        assert "dag_created" in seen_types
        assert "subtask_started" in seen_types
        assert "subtask_completed" in seen_types
        assert seen_types[-1] == "run_completed"


def _wait_for_completion(client, run_id, attempts=200):
    import time

    for _ in range(attempts):
        res = client.get(f"/runs/{run_id}")
        if res.status_code == 200:
            record = res.json()
            if record["status"] in ("completed", "failed"):
                return record
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} did not complete in time")
