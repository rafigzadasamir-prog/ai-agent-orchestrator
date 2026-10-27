from ai_agent_orchestrator.workers.analysis import AnalysisWorker
from ai_agent_orchestrator.workers.base import Worker
from ai_agent_orchestrator.workers.research import ResearchWorker
from ai_agent_orchestrator.workers.writing import WritingWorker

__all__ = ["Worker", "ResearchWorker", "AnalysisWorker", "WritingWorker"]


def default_workers() -> dict[str, Worker]:
    """The standard capability -> Worker registry used by the Orchestrator."""
    return {
        "research": ResearchWorker(),
        "analysis": AnalysisWorker(),
        "writing": WritingWorker(),
    }
