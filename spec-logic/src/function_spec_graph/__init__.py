"""Per-function LangGraph test-generation workflow."""

from .parser.graph_parser import collect_functions, FunctionInfo

__all__ = ["collect_functions", "FunctionInfo", "run_workflow"]


def run_workflow(*args, **kwargs):
    from .langgraph_workflow import run_workflow as _run_workflow
    return _run_workflow(*args, **kwargs)
