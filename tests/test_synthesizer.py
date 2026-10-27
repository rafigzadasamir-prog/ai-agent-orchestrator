from ai_agent_orchestrator.dag import DAG
from ai_agent_orchestrator.llm.rule_based import RuleBasedLLM
from ai_agent_orchestrator.synthesizer import Synthesizer
from ai_agent_orchestrator.workers.analysis import AnalysisWorker
from ai_agent_orchestrator.workers.research import ResearchWorker
from ai_agent_orchestrator.workers.writing import WritingWorker


def build_resolved_dag():
    dag = DAG.from_specs(
        [
            {"id": "R1", "description": "research postgresql", "capability": "research",
             "depends_on": [], "params": {"topic": "postgresql"}},
            {"id": "A1", "description": "average scores", "capability": "analysis",
             "depends_on": [], "params": {"expression": "avg([120, 95, 110, 130])"}},
            {"id": "W1", "description": "write summary", "capability": "writing",
             "depends_on": ["R1", "A1"], "params": {"instructions": "Write a summary."}},
        ]
    )
    dag.subtasks["R1"].result = ResearchWorker().run(dag.subtasks["R1"], dag)
    dag.subtasks["R1"].status = "done"
    dag.subtasks["A1"].result = AnalysisWorker().run(dag.subtasks["A1"], dag)
    dag.subtasks["A1"].status = "done"
    dag.subtasks["W1"].result = WritingWorker().run(dag.subtasks["W1"], dag)
    dag.subtasks["W1"].status = "done"
    return dag


def test_synthesize_cites_every_completed_subtask_in_order():
    synthesizer = Synthesizer(RuleBasedLLM())
    dag = build_resolved_dag()

    final_answer, citations = synthesizer.synthesize("Tell me about postgresql", dag)

    assert citations == ["R1", "A1", "W1"]
    assert "[R1]" in final_answer
    assert "[A1]" in final_answer
    assert "PostgreSQL" in final_answer
    assert "113.75" in final_answer


def test_synthesize_only_cites_subtasks_that_actually_completed():
    synthesizer = Synthesizer(RuleBasedLLM())
    dag = build_resolved_dag()
    dag.subtasks["A1"].status = "failed"
    dag.subtasks["A1"].result = None
    dag.subtasks["A1"].error = "boom"

    _, citations = synthesizer.synthesize("Tell me about postgresql", dag)
    assert "A1" not in citations
    assert citations == ["R1", "W1"]
