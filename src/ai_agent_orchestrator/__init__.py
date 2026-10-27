"""Hierarchical multi-agent task orchestration platform.

A Planner decomposes a free-form request into a DAG of capability-tagged
subtasks. A Scheduler walks the DAG, dispatching every subtask whose
dependencies are satisfied to the matching Worker agent, and persists
results on a shared Blackboard. Once the DAG resolves, a Synthesizer
produces a final answer that cites which subtask results it drew on.
"""

__version__ = "0.1.0"
