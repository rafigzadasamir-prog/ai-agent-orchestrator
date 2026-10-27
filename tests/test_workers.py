import pytest

from ai_agent_orchestrator.dag import DAG
from ai_agent_orchestrator.workers.analysis import AnalysisWorker, UnsafeExpressionError, safe_eval
from ai_agent_orchestrator.workers.research import ResearchWorker
from ai_agent_orchestrator.workers.writing import WritingWorker


def build_dag(specs):
    return DAG.from_specs(specs)


# -- ResearchWorker ---------------------------------------------------------


def test_research_worker_finds_a_known_topic():
    dag = build_dag(
        [{"id": "R1", "description": "", "capability": "research", "depends_on": [],
          "params": {"topic": "postgresql"}}]
    )
    result = ResearchWorker().run(dag.subtasks["R1"], dag)
    assert result["found"] is True
    assert result["title"] == "PostgreSQL"
    assert "key_points" in result and len(result["key_points"]) > 0


def test_research_worker_reports_unknown_topic_without_raising():
    dag = build_dag(
        [{"id": "R1", "description": "", "capability": "research", "depends_on": [],
          "params": {"topic": "cobol mainframes"}}]
    )
    result = ResearchWorker().run(dag.subtasks["R1"], dag)
    assert result["found"] is False
    assert "No offline knowledge-base entry" in result["message"]


def test_research_worker_requires_a_topic():
    dag = build_dag(
        [{"id": "R1", "description": "", "capability": "research", "depends_on": [],
          "params": {}}]
    )
    with pytest.raises(ValueError):
        ResearchWorker().run(dag.subtasks["R1"], dag)


# -- AnalysisWorker / safe_eval ----------------------------------------------


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("avg([120, 95, 110, 130])", 113.75),
        ("sum([1, 2, 3])", 6),
        ("max([4, 9, 2])", 9),
        ("min([4, 9, 2])", 2),
        ("median([1, 3, 2])", 2),
        ("product([2, 3, 4])", 24),
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("round(7 / 2)", 4),
    ],
)
def test_safe_eval_computes_expected_values(expression, expected):
    assert safe_eval(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('secrets.txt').read()",
        "[].__class__",
        "1 if True else 2",
        "lambda: 1",
        "a + 1",
    ],
)
def test_safe_eval_rejects_anything_outside_the_allow_list(expression):
    with pytest.raises(UnsafeExpressionError):
        safe_eval(expression)


def test_analysis_worker_returns_expression_and_value():
    dag = build_dag(
        [{"id": "A1", "description": "", "capability": "analysis", "depends_on": [],
          "params": {"expression": "avg([120, 95, 110, 130])"}}]
    )
    result = AnalysisWorker().run(dag.subtasks["A1"], dag)
    assert result == {"expression": "avg([120, 95, 110, 130])", "value": 113.75}


def test_analysis_worker_requires_expression_param():
    dag = build_dag(
        [{"id": "A1", "description": "", "capability": "analysis", "depends_on": [],
          "params": {}}]
    )
    with pytest.raises(ValueError):
        AnalysisWorker().run(dag.subtasks["A1"], dag)


# -- WritingWorker ------------------------------------------------------------


def test_writing_worker_compiles_and_cites_upstream_results():
    dag = build_dag(
        [
            {"id": "R1", "description": "", "capability": "research", "depends_on": [],
             "params": {"topic": "postgresql"}},
            {"id": "A1", "description": "", "capability": "analysis", "depends_on": [],
             "params": {"expression": "sum([1, 2])"}},
            {"id": "W1", "description": "", "capability": "writing",
             "depends_on": ["R1", "A1"],
             "params": {"instructions": "Write a short brief.", "title": "Brief"}},
        ]
    )
    dag.subtasks["R1"].result = ResearchWorker().run(dag.subtasks["R1"], dag)
    dag.subtasks["R1"].status = "done"
    dag.subtasks["A1"].result = AnalysisWorker().run(dag.subtasks["A1"], dag)
    dag.subtasks["A1"].status = "done"

    result = WritingWorker().run(dag.subtasks["W1"], dag)
    assert result["sources"] == ["R1", "A1"]
    assert "[R1]" in result["compiled_text"]
    assert "[A1]" in result["compiled_text"]
    assert "PostgreSQL" in result["compiled_text"]
    assert "# Brief" in result["compiled_text"]
