import asyncio

from ai_agent_orchestrator.dag import DAG


class FixedPlanner:
    """A stand-in Planner that returns a pre-built DAG, for tests that need
    exact control over the graph shape rather than going through the
    rule-based regex decomposition."""

    def __init__(self, dag_factory):
        self._dag_factory = dag_factory

    def decompose(self, request):
        return self._dag_factory()


def collect_events(orchestrator, request, **run_kwargs):
    events = []

    async def on_event(event):
        events.append(event)

    record = asyncio.run(orchestrator.run(request, on_event=on_event, **run_kwargs))
    return record, events


def test_two_independent_subtasks_are_dispatched_together(orchestrator):
    record, events = collect_events(
        orchestrator,
        "Research postgresql and mongodb, and write a short comparison.",
    )
    event_types_with_ids = [(e.type, e.subtask_id) for e in events]

    started = [sid for (t, sid) in event_types_with_ids if t == "subtask_started"]
    completed_idx = {
        sid: i for i, (t, sid) in enumerate(event_types_with_ids) if t == "subtask_completed"
    }
    started_idx = {sid: i for i, (t, sid) in enumerate(event_types_with_ids) if t == "subtask_started"}

    assert set(started) >= {"R1", "R2"}
    # Both research subtasks must have been started before either completed:
    # that is the observable signature of concurrent dispatch within one batch.
    assert started_idx["R1"] < completed_idx["R2"]
    assert started_idx["R2"] < completed_idx["R1"]


def test_dependent_subtask_only_starts_after_all_its_dependencies_complete(orchestrator):
    request = (
        "Research the key differences between PostgreSQL and MongoDB, calculate the "
        "average of their relative performance scores (120, 95, 110, 130), and write "
        "an executive summary comparing them for a startup choosing a database."
    )
    record, events = collect_events(orchestrator, request)

    order = [(e.type, e.subtask_id) for e in events]
    completed_at = {sid: i for i, (t, sid) in enumerate(order) if t == "subtask_completed"}
    started_at = {sid: i for i, (t, sid) in enumerate(order) if t == "subtask_started"}

    assert "W1" in started_at
    for dep in ("R1", "R2", "A1"):
        assert completed_at[dep] < started_at["W1"]

    assert record["status"] == "completed"
    assert order[0] == ("run_started", None)
    assert order[-1][0] == "run_completed"


def test_a_failed_subtask_blocks_its_dependents_without_hanging(blackboard):
    from ai_agent_orchestrator.llm.rule_based import RuleBasedLLM
    from ai_agent_orchestrator.scheduler import Orchestrator
    from ai_agent_orchestrator.synthesizer import Synthesizer
    from ai_agent_orchestrator.workers import default_workers

    def make_dag():
        return DAG.from_specs(
            [
                {
                    "id": "A1", "description": "divide by zero", "capability": "analysis",
                    "depends_on": [], "params": {"expression": "1 / 0"},
                },
                {
                    "id": "W1", "description": "write it up", "capability": "writing",
                    "depends_on": ["A1"], "params": {"instructions": "go"},
                },
            ]
        )

    llm = RuleBasedLLM()
    orchestrator = Orchestrator(
        planner=FixedPlanner(make_dag),
        workers=default_workers(),
        synthesizer=Synthesizer(llm),
        blackboard=blackboard,
        worker_delay=0.0,
    )

    record, events = collect_events(orchestrator, "irrelevant request text")

    subtasks = {s["id"]: s for s in record["subtasks"]}
    assert subtasks["A1"]["status"] == "failed"
    assert subtasks["W1"]["status"] == "failed"
    assert "upstream dependency failed" in subtasks["W1"]["error"]
    assert record["status"] == "failed"

    failed_event_ids = [e.subtask_id for e in events if e.type == "subtask_failed"]
    assert failed_event_ids == ["A1", "W1"]


def test_run_id_can_be_specified_explicitly(orchestrator):
    record, _ = collect_events(
        orchestrator, "Write a haiku about rivers.", run_id="my-custom-run-id"
    )
    assert record["run_id"] == "my-custom-run-id"
