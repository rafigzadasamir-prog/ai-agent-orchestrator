"""End-to-end: a real multi-step request through the full RuleBasedLLM pipeline.

Planner -> DAG -> Scheduler -> {ResearchWorker, AnalysisWorker, WritingWorker}
-> Blackboard -> Synthesizer, with zero network calls and a non-trivial
dependency chain (two independent research lookups and an independent
calculation, all gating one final writing subtask).
"""

import asyncio

from ai_agent_orchestrator.blackboard import Blackboard

REQUEST = (
    "Research the key differences between PostgreSQL and MongoDB, calculate the "
    "average of their relative performance scores (120, 95, 110, 130), and write "
    "an executive summary comparing them for a startup choosing a database."
)


def test_end_to_end_run_produces_a_correct_cited_answer(orchestrator, blackboard):
    record = asyncio.run(orchestrator.run(REQUEST, run_id="e2e-run"))

    assert record["status"] == "completed"
    assert record["run_id"] == "e2e-run"

    subtasks = {s["id"]: s for s in record["subtasks"]}
    assert len(subtasks) == 4  # R1, R2, A1, W1
    assert {s["capability"] for s in subtasks.values()} == {"research", "analysis", "writing"}
    assert all(s["status"] == "done" for s in subtasks.values())

    research_topics = {
        s["result"]["topic"] for s in subtasks.values() if s["capability"] == "research"
    }
    assert research_topics == {"postgresql", "mongodb"}

    analysis_subtask = next(s for s in subtasks.values() if s["capability"] == "analysis")
    assert analysis_subtask["result"]["value"] == 113.75

    writing_subtask = next(s for s in subtasks.values() if s["capability"] == "writing")
    assert set(writing_subtask["depends_on"]) == {
        sid for sid, s in subtasks.items() if s["capability"] in ("research", "analysis")
    }

    final_answer = record["final_answer"]
    assert "PostgreSQL" in final_answer
    assert "MongoDB" in final_answer
    assert "113.75" in final_answer
    assert set(record["citations"]) == set(subtasks.keys())

    # The run must be durably readable from a brand-new Blackboard instance
    # pointed at the same SQLite file, i.e. it actually survives a restart.
    reloaded = Blackboard(blackboard.db_path).get_run("e2e-run")
    assert reloaded["final_answer"] == final_answer
    assert reloaded["status"] == "completed"
    assert len(reloaded["subtasks"]) == 4
    assert len(reloaded["events"]) > 0
