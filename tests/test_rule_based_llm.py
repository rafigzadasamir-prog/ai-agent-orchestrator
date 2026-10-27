from ai_agent_orchestrator.dag import DAG
from ai_agent_orchestrator.llm.base import SubtaskResultView
from ai_agent_orchestrator.llm.rule_based import RuleBasedLLM
from ai_agent_orchestrator.planner import Planner

CAPS = ("research", "analysis", "writing")


def test_plan_extracts_two_research_topics_one_analysis_and_a_dependent_writer():
    llm = RuleBasedLLM()
    request = (
        "Research the key differences between PostgreSQL and MongoDB, calculate the "
        "average of their relative performance scores (120, 95, 110, 130), and write "
        "an executive summary comparing them for a startup choosing a database."
    )
    specs = llm.generate_plan(request, CAPS)
    ids_by_capability = {s["id"]: s["capability"] for s in specs}

    research_ids = [sid for sid, cap in ids_by_capability.items() if cap == "research"]
    analysis_ids = [sid for sid, cap in ids_by_capability.items() if cap == "analysis"]
    writing_ids = [sid for sid, cap in ids_by_capability.items() if cap == "writing"]

    assert len(research_ids) == 2
    assert len(analysis_ids) == 1
    assert len(writing_ids) == 1

    topics = {s["id"]: s["params"]["topic"] for s in specs if s["capability"] == "research"}
    assert set(topics.values()) == {"postgresql", "mongodb"}

    analysis_spec = next(s for s in specs if s["capability"] == "analysis")
    assert analysis_spec["params"]["expression"] == "avg([120, 95, 110, 130])"

    writing_spec = next(s for s in specs if s["capability"] == "writing")
    assert set(writing_spec["depends_on"]) == set(research_ids) | set(analysis_ids)
    assert "executive summary" in writing_spec["params"]["instructions"].lower()

    # The whole thing must still validate as a DAG.
    dag = DAG.from_specs(specs)
    assert dag.ready_subtasks()  # the two research + one analysis tasks have no deps


def test_plan_with_no_topics_or_numbers_falls_back_to_a_single_writing_task():
    llm = RuleBasedLLM()
    specs = llm.generate_plan("Write a haiku about autumn leaves.", CAPS)
    assert len(specs) == 1
    assert specs[0]["capability"] == "writing"
    assert specs[0]["depends_on"] == []


def test_plan_handles_sum_keyword():
    llm = RuleBasedLLM()
    specs = llm.generate_plan(
        "Calculate the sum of 3, 4, 5 and write one sentence about it.", CAPS
    )
    analysis_spec = next(s for s in specs if s["capability"] == "analysis")
    assert analysis_spec["params"]["expression"] == "sum([3, 4, 5])"


def test_plan_handles_direct_calculate_expression_fallback():
    llm = RuleBasedLLM()
    specs = llm.generate_plan("calculate 6 * 7 and write the result down.", CAPS)
    analysis_spec = next(s for s in specs if s["capability"] == "analysis")
    assert analysis_spec["params"]["expression"] == "6 * 7"


def test_generate_synthesis_cites_every_completed_subtask():
    llm = RuleBasedLLM()
    results = [
        SubtaskResultView(
            id="R1", description="research postgresql", capability="research",
            result={"found": True, "title": "PostgreSQL", "summary": "A relational database.",
                     "key_points": ["ACID"]},
        ),
        SubtaskResultView(
            id="A1", description="average", capability="analysis",
            result={"expression": "avg([1, 2, 3])", "value": 2},
        ),
    ]
    text = llm.generate_synthesis("Tell me about postgresql", results)
    assert "[R1]" in text or "R1" in text
    assert "Synthesized from subtasks: R1, A1" in text


def test_generate_synthesis_handles_no_completed_subtasks():
    llm = RuleBasedLLM()
    text = llm.generate_synthesis("anything", [])
    assert "No subtasks completed" in text


def test_planner_decompose_returns_a_validated_dag():
    planner = Planner(RuleBasedLLM())
    dag = planner.decompose("Research redis and write a short note about it.")
    assert "R1" in dag.subtasks
    assert dag.subtasks["R1"].params["topic"] == "redis"
