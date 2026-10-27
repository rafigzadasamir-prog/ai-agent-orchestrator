import pytest

from ai_agent_orchestrator.dag import DAG, DAGValidationError


def spec(id_, capability="research", depends_on=None, **params):
    return {
        "id": id_,
        "description": f"do {id_}",
        "capability": capability,
        "depends_on": depends_on or [],
        "params": params,
    }


def test_builds_a_valid_linear_dag():
    dag = DAG.from_specs(
        [
            spec("R1"),
            spec("A1", capability="analysis", depends_on=["R1"]),
            spec("W1", capability="writing", depends_on=["A1"]),
        ]
    )
    assert dag.order == ["R1", "A1", "W1"]
    assert dag.subtasks["W1"].depends_on == ["A1"]


def test_empty_plan_is_rejected():
    with pytest.raises(DAGValidationError):
        DAG.from_specs([])


def test_duplicate_id_is_rejected():
    with pytest.raises(DAGValidationError, match="Duplicate"):
        DAG.from_specs([spec("R1"), spec("R1")])


def test_unknown_capability_is_rejected():
    with pytest.raises(DAGValidationError, match="capability"):
        DAG.from_specs([spec("R1", capability="teleportation")])


def test_self_dependency_is_rejected():
    with pytest.raises(DAGValidationError, match="itself"):
        DAG.from_specs([spec("R1", depends_on=["R1"])])


def test_missing_dependency_is_rejected():
    with pytest.raises(DAGValidationError, match="unknown subtask"):
        DAG.from_specs([spec("W1", capability="writing", depends_on=["GHOST"])])


def test_cycle_is_rejected():
    with pytest.raises(DAGValidationError, match="cycle"):
        DAG.from_specs(
            [
                spec("A", depends_on=["B"]),
                spec("B", depends_on=["A"]),
            ]
        )


def test_longer_cycle_is_rejected():
    with pytest.raises(DAGValidationError, match="cycle"):
        DAG.from_specs(
            [
                spec("A", depends_on=["C"]),
                spec("B", depends_on=["A"]),
                spec("C", depends_on=["B"]),
            ]
        )


def test_ready_subtasks_only_includes_nodes_with_satisfied_dependencies():
    dag = DAG.from_specs(
        [
            spec("R1"),
            spec("A1", capability="analysis", depends_on=["R1"]),
        ]
    )
    assert [s.id for s in dag.ready_subtasks()] == ["R1"]

    dag.subtasks["R1"].status = "done"
    assert [s.id for s in dag.ready_subtasks()] == ["A1"]


def test_blocked_pending_detects_subtasks_behind_a_failed_dependency():
    dag = DAG.from_specs(
        [
            spec("R1"),
            spec("W1", capability="writing", depends_on=["R1"]),
        ]
    )
    dag.subtasks["R1"].status = "failed"
    assert [s.id for s in dag.blocked_pending()] == ["W1"]


def test_is_terminal_requires_every_subtask_to_be_done_or_failed():
    dag = DAG.from_specs([spec("R1"), spec("R2")])
    assert not dag.is_terminal()
    dag.subtasks["R1"].status = "done"
    assert not dag.is_terminal()
    dag.subtasks["R2"].status = "failed"
    assert dag.is_terminal()
