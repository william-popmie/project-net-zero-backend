from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from parser.graph_parser import FunctionInfo, collect_functions
from spec_logic.ai_spec_generator import generate_spec


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class FunctionState(TypedDict):
    function: FunctionInfo
    spec_code: str
    status: str    # "generated" | "failed"
    error: str
    engine: str    # "claude" or "crusoe"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def generate_spec_node(state: FunctionState) -> FunctionState:
    func = state["function"]
    engine = state.get("engine", "claude")
    print(f"  [generate] {func.qualified_name}  ({func.file_path}) ...")
    try:
        code = generate_spec(
            function_id=func.id,
            function_source=func.source_code,
            file_path=func.file_path,
            engine=engine,
        )
        state["spec_code"] = code
        state["status"] = "generated"
        print(f"  [generate] OK  ({len(code.splitlines())} lines)")
    except Exception as e:
        state["error"] = str(e)
        state["status"] = "failed"
        print(f"  [generate] FAILED: {e}")
    return state


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def _build_graph():
    wf = StateGraph(FunctionState)
    wf.add_node("generate", generate_spec_node)
    wf.set_entry_point("generate")
    wf.add_edge("generate", END)
    return wf.compile()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_workflow(
    project_root: Path,
    output_path: Path,
    engine: str = "claude",
) -> dict[str, Any]:
    root = project_root.resolve()

    print(f"\nScanning project: {root}")
    functions = collect_functions(root)
    print(f"\nFound {len(functions)} source functions\n")

    graph = _build_graph()
    results: list[dict[str, Any]] = []

    for i, func in enumerate(functions, 1):
        print(f"{'─' * 58}")
        print(f"[{i}/{len(functions)}]  {func.qualified_name}")
        print(f"         file : {func.file_path}  lines {func.line}–{func.end_line}")

        initial: FunctionState = {
            "function": func,
            "spec_code": "",
            "status": "pending",
            "error": "",
            "engine": engine,
        }
        try:
            final = graph.invoke(initial)
        except Exception as e:
            print(f"  [ERROR] graph crashed: {e}")
            final = {**initial, "status": "failed", "error": str(e)}

        results.append({
            "id": func.id,
            "name": func.name,
            "qualified_name": func.qualified_name,
            "file": func.file_path,
            "line": func.line,
            "end_line": func.end_line,
            "function_code": func.source_code,
            "spec_code": final.get("spec_code", ""),
            "status": final.get("status", "failed"),
            "error": final.get("error", ""),
        })

    generated = sum(1 for r in results if r["status"] == "generated")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\n{'─' * 58}")
    print(f"Done  {generated}/{len(results)} generated,  {failed} failed")

    output: dict[str, Any] = {
        "project_root": str(root),
        "generated_at": datetime.now().isoformat(),
        "functions": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Output → {output_path}")

    return output
