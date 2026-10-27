"""Runtime configuration: LLM backend selection and storage locations."""

from __future__ import annotations

import os
from pathlib import Path

from ai_agent_orchestrator.llm.base import LLMBackend


def default_db_path() -> str:
    """SQLite path for the Blackboard, overridable for tests/deployments."""
    return os.environ.get("ORCHESTRATOR_DB_PATH", str(Path.cwd() / "orchestrator.db"))


def worker_delay_seconds() -> float:
    """Artificial per-subtask delay so a live dashboard has something to watch.

    Zero by default (and in tests) for deterministic, fast execution; the
    dashboard and CLI opt into a small visible delay via this env var.
    """
    try:
        return float(os.environ.get("ORCHESTRATOR_WORKER_DELAY", "0"))
    except ValueError:
        return 0.0


def get_llm_backend() -> LLMBackend:
    """Real Anthropic backend when a key is configured, else the offline fallback.

    This mirrors the rest of the portfolio: every code path downstream of this
    function works identically against either backend, so the full pipeline is
    testable offline while still supporting a genuine LLM in production.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        from ai_agent_orchestrator.llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM(api_key=api_key)

    from ai_agent_orchestrator.llm.rule_based import RuleBasedLLM

    return RuleBasedLLM()
