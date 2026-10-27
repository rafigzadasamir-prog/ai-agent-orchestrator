import pytest

from ai_agent_orchestrator.blackboard import Blackboard
from ai_agent_orchestrator.llm.rule_based import RuleBasedLLM
from ai_agent_orchestrator.planner import Planner
from ai_agent_orchestrator.scheduler import Orchestrator
from ai_agent_orchestrator.synthesizer import Synthesizer
from ai_agent_orchestrator.workers import default_workers


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "orchestrator-test.db"


@pytest.fixture
def blackboard(db_path):
    return Blackboard(db_path)


@pytest.fixture
def orchestrator(blackboard):
    llm = RuleBasedLLM()
    return Orchestrator(
        planner=Planner(llm),
        workers=default_workers(),
        synthesizer=Synthesizer(llm),
        blackboard=blackboard,
        worker_delay=0.0,
    )
