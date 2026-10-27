"""AnalysisWorker: safe arithmetic over a Python-expression string.

Expressions are parsed with ``ast.parse`` and walked by a recursive
evaluator that only ever recognizes a small allow-list of node types,
operators, and function names. There is no ``eval``/``exec`` anywhere in
this module, so there is no way to reach attribute access, name lookups
outside the allow-list, imports, or any other form of code execution.
"""

from __future__ import annotations

import ast
import math
import operator
import statistics
from typing import TYPE_CHECKING, Any

from ai_agent_orchestrator.workers.base import Worker

if TYPE_CHECKING:
    from ai_agent_orchestrator.dag import DAG, Subtask

_BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
SAFE_FUNCTIONS: dict[str, Any] = {
    "sum": sum,
    "min": min,
    "max": max,
    "len": len,
    "round": round,
    "abs": abs,
    "avg": lambda xs: sum(xs) / len(xs),
    "median": statistics.median,
    "product": math.prod,
}


class UnsafeExpressionError(ValueError):
    """Raised when an expression uses anything outside the allow-list."""


def safe_eval(expression: str) -> float | int | list:
    """Evaluate an arithmetic expression using only vetted AST node types."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Could not parse expression: {expression!r}") from exc
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise UnsafeExpressionError(f"Disallowed constant: {node.value!r}")
    if isinstance(node, ast.List):
        return [_eval_node(el) for el in node.elts]
    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpressionError(f"Disallowed operator: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpressionError(f"Disallowed unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS:
            raise UnsafeExpressionError("Only the allow-listed functions may be called.")
        if node.keywords:
            raise UnsafeExpressionError("Keyword arguments are not supported.")
        args = [_eval_node(arg) for arg in node.args]
        return SAFE_FUNCTIONS[node.func.id](*args)
    raise UnsafeExpressionError(f"Disallowed expression element: {ast.dump(node)}")


class AnalysisWorker(Worker):
    capability = "analysis"

    def run(self, subtask: Subtask, dag: DAG) -> dict[str, Any]:
        del dag  # pure arithmetic over the given expression; no upstream data needed
        expression = subtask.params.get("expression")
        if not expression or not isinstance(expression, str):
            raise ValueError("AnalysisWorker requires a non-empty 'expression' string param.")
        value = safe_eval(expression)
        return {"expression": expression, "value": value}
