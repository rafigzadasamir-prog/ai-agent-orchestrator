"""Production ASGI entry point: ``uvicorn ai_agent_orchestrator.dashboard.server:app``.

This is the only module that eagerly constructs the default application
(and therefore the only module that touches the real on-disk Blackboard on
import) -- kept separate from ``app.py`` so importing the factory for tests
never has that side effect.
"""

from ai_agent_orchestrator.dashboard.app import create_app

app = create_app()
