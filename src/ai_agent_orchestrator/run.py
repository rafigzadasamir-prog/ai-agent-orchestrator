"""Headless CLI entry point: ``python -m ai_agent_orchestrator.run "request"``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from ai_agent_orchestrator.blackboard import Blackboard
from ai_agent_orchestrator.config import default_db_path, get_llm_backend, worker_delay_seconds
from ai_agent_orchestrator.events import OrchestrationEvent
from ai_agent_orchestrator.planner import Planner
from ai_agent_orchestrator.scheduler import Orchestrator
from ai_agent_orchestrator.synthesizer import Synthesizer
from ai_agent_orchestrator.workers import default_workers


def build_orchestrator(
    db_path: str | None = None, worker_delay: float | None = None
) -> tuple[Orchestrator, Blackboard]:
    llm = get_llm_backend()
    blackboard = Blackboard(db_path or default_db_path())
    orchestrator = Orchestrator(
        planner=Planner(llm),
        workers=default_workers(),
        synthesizer=Synthesizer(llm),
        blackboard=blackboard,
        worker_delay=worker_delay_seconds() if worker_delay is None else worker_delay,
    )
    return orchestrator, blackboard


def _print_event(event: OrchestrationEvent) -> None:
    label = event.type.upper().replace("_", " ")
    suffix = f" [{event.subtask_id}]" if event.subtask_id else ""
    print(f"  {label}{suffix}")


async def _run_async(request: str, verbose: bool, as_json: bool) -> dict:
    orchestrator, _ = build_orchestrator()

    async def on_event(event: OrchestrationEvent) -> None:
        if verbose and not as_json:
            _print_event(event)

    return await orchestrator.run(request, on_event=on_event)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ai_agent_orchestrator.run",
        description="Decompose a request into a DAG, run it through the orchestrator, "
        "and print the synthesized final answer.",
    )
    parser.add_argument("request", help="Free-form request to decompose and execute.")
    parser.add_argument("--json", action="store_true", help="Print the full run record as JSON.")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress the live event trace."
    )
    args = parser.parse_args(argv)

    run_record = asyncio.run(_run_async(args.request, verbose=not args.quiet, as_json=args.json))

    if args.json:
        print(json.dumps(run_record, indent=2))
        return 0 if run_record["status"] == "completed" else 1

    print()
    print("=" * 72)
    print(f"Run {run_record['run_id']} -- status: {run_record['status']}")
    print("=" * 72)
    for subtask in run_record["subtasks"]:
        print(f"  [{subtask['id']}] ({subtask['capability']}) {subtask['status']}")
    print()
    print("Final answer:")
    print("-" * 72)
    print(run_record["final_answer"])
    print("-" * 72)
    print(f"Cited subtasks: {', '.join(run_record['citations'])}")
    return 0 if run_record["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
