from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from .parser.graph_parser import (
    FunctionCollector,
    FunctionInfo,
    collect_function_call_map,
    collect_functions,
    discover_python_files,
    extract_function_source,
    is_test_file,
    parse_python_file,
    path_to_module_path,
)
from .parser.ai_spec_generator import determine_test_file_path, generate_tests


class FunctionState(TypedDict):
    project_root: Path
    function: FunctionInfo
    coverage_threshold: float
    max_iterations: int

    existing_test_code: str       # test code found or generated
    test_file_path: Path | None   # where tests live / will be written
    coverage_score: float         # 0–100
    iteration: int

    final_test_code: str          # populated when workflow reaches a terminal state
    status: str                   # "passed_existing" | "generated" | "failed" | "in_progress"
    errors: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_python_executable(project_root: Path) -> str:
    for candidate in (".venv/bin/python", "venv/bin/python", "env/bin/python"):
        python_path = project_root / candidate
        if python_path.exists():
            return str(python_path)
    # Fall back to the same interpreter that is running this code so that
    # pytest and coverage.py are guaranteed to be available.
    return sys.executable


# ---------------------------------------------------------------------------
# Node 1: lookup_tests_node
# ---------------------------------------------------------------------------

def lookup_tests_node(state: FunctionState) -> FunctionState:
    """Find existing tests that exercise this function."""
    project_root = state["project_root"]
    function = state["function"]
    func_name = function.name

    matching_test_code: list[str] = []
    test_file_path: Path | None = None

    for file_path in discover_python_files(project_root):
        # Use relative path so parent directories outside the project don't
        # accidentally trigger the test-file heuristic.
        if not is_test_file(file_path.relative_to(project_root)):
            continue

        module_tree = parse_python_file(file_path)
        if module_tree is None:
            continue

        module_path = path_to_module_path(project_root, file_path)
        call_map = collect_function_call_map(module_tree, module_path)

        collector = FunctionCollector(
            file_path=file_path.relative_to(project_root),
            module_path=module_path,
            kind="spec_function",
        )
        collector.visit(module_tree)

        for test_func in collector.collected:
            matched = False

            # Direct call detection
            called_names = call_map.get(test_func.id, set())
            if func_name in called_names:
                matched = True

            # Name heuristic: test name contains source function name as a complete word
            if not matched:
                test_name_lower = test_func.name.lower()
                for prefix in ("test_", "should_", "it_", "spec_"):
                    if test_name_lower.startswith(prefix):
                        test_name_lower = test_name_lower[len(prefix):]
                        break
                pattern = r"\b" + re.escape(func_name.lower()) + r"\b"
                if re.search(pattern, test_name_lower):
                    matched = True

            if matched:
                test_code = extract_function_source(file_path, test_func.qualified_name)
                if test_code:
                    matching_test_code.append(test_code)
                    if test_file_path is None:
                        test_file_path = file_path

    state["existing_test_code"] = "\n\n".join(matching_test_code)
    state["test_file_path"] = test_file_path

    if matching_test_code:
        rel = test_file_path.relative_to(project_root) if test_file_path else "?"
        print(f"  [lookup] found {len(matching_test_code)} existing test(s) in {rel}")
    else:
        print(f"  [lookup] no existing tests found")

    return state


# ---------------------------------------------------------------------------
# Node 2: evaluate_tests_node
# ---------------------------------------------------------------------------

def evaluate_tests_node(state: FunctionState) -> FunctionState:
    """Run existing/generated tests and measure per-function line coverage."""
    if not state["existing_test_code"]:
        # Nothing to evaluate yet — routing will send us to create
        return state

    project_root = state["project_root"]
    function = state["function"]
    test_file_path = state["test_file_path"]

    if test_file_path is None or not test_file_path.exists():
        state["coverage_score"] = 0.0
        return state

    python_exe = _detect_python_executable(project_root)

    # Top-level source package from the function's file path (e.g. "src")
    func_file = Path(function.file_path)
    cov_module = func_file.parts[0] if func_file.parts else "src"

    try:
        result = subprocess.run(
            [
                python_exe, "-m", "pytest",
                str(test_file_path),
                f"--cov={cov_module}",
                "--cov-report=json",
                "-q",
                "--tb=short",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  [evaluate] pytest exit {result.returncode}")
            if result.stdout.strip():
                print("  " + result.stdout.strip().splitlines()[-1])
            if result.stderr.strip():
                print("  " + result.stderr.strip().splitlines()[-1])

        coverage_file = project_root / "coverage.json"
        coverage_score = 0.0

        if coverage_file.exists():
            coverage_data = json.loads(coverage_file.read_text(encoding="utf-8"))
            files = coverage_data.get("files", {})

            # Match the coverage entry to this function's file
            func_file_posix = function.file_path
            file_cov = files.get(func_file_posix)
            if file_cov is None:
                for key, val in files.items():
                    if key.endswith(func_file_posix) or func_file_posix.endswith(key):
                        file_cov = val
                        break

            if file_cov is not None:
                executed_lines = set(file_cov.get("executed_lines", []))
                missing_lines = set(file_cov.get("missing_lines", []))
                # Only count lines that coverage.py considers executable
                # (executed + missing). This excludes blank lines, comments,
                # and docstrings that sit inside the function range.
                all_executable = executed_lines | missing_lines
                func_range = set(range(function.line, function.end_line + 1))
                func_executable = func_range & all_executable
                if func_executable:
                    coverage_score = len(func_executable & executed_lines) / len(func_executable) * 100
                else:
                    coverage_score = 100.0  # no executable lines — trivially covered

        state["coverage_score"] = coverage_score

        # Set terminal status when coverage is sufficient
        if coverage_score >= state["coverage_threshold"]:
            state["status"] = "passed_existing" if state["iteration"] == 0 else "generated"
            if test_file_path.exists():
                state["final_test_code"] = test_file_path.read_text(encoding="utf-8")

    except subprocess.TimeoutExpired:
        state["errors"].append(f"Test execution timeout for {function.id}")
        state["coverage_score"] = 0.0
    except Exception as e:
        state["errors"].append(f"Test execution error for {function.id}: {e}")
        state["coverage_score"] = 0.0

    # Mark as failed when we've run out of iterations
    if (
        state["coverage_score"] < state["coverage_threshold"]
        and state["iteration"] >= state["max_iterations"]
    ):
        state["status"] = "failed"
        if test_file_path and test_file_path.exists():
            state["final_test_code"] = test_file_path.read_text(encoding="utf-8")

    return state


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_after_evaluate(state: FunctionState) -> str:
    if not state["existing_test_code"]:
        # No tests yet
        if state["iteration"] >= state["max_iterations"]:
            return END
        return "create"

    if state["coverage_score"] >= state["coverage_threshold"]:
        return END

    if state["iteration"] >= state["max_iterations"]:
        return END

    return "create"


# ---------------------------------------------------------------------------
# Node 3: create_tests_node
# ---------------------------------------------------------------------------

def create_tests_node(state: FunctionState) -> FunctionState:
    """Generate new tests with Claude and write them to the test file."""
    project_root = state["project_root"]
    function = state["function"]

    try:
        new_test_code = generate_tests(
            function_id=function.id,
            function_code=function.source_code,
            source_file=function.file_path,
            project_root=project_root,
        )

        # Resolve test file path
        test_file_path = state["test_file_path"]
        if test_file_path is None:
            test_file_path = determine_test_file_path(project_root, function.file_path)

        if test_file_path.exists():
            existing_content = test_file_path.read_text(encoding="utf-8")

            # Avoid adding functions/classes that already exist
            try:
                existing_tree = ast.parse(existing_content)
                existing_names = {
                    node.name
                    for node in ast.walk(existing_tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                }
                new_tree = ast.parse(new_test_code)
                new_names = {
                    node.name
                    for node in ast.walk(new_tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                }
                if new_names - existing_names:
                    final_content = existing_content.rstrip() + "\n\n" + new_test_code + "\n"
                else:
                    final_content = existing_content
            except SyntaxError:
                final_content = existing_content.rstrip() + "\n\n" + new_test_code + "\n"
        else:
            test_file_path.parent.mkdir(parents=True, exist_ok=True)
            # Create __init__.py so pytest can import from subdirectories
            init_py = test_file_path.parent / "__init__.py"
            if not init_py.exists():
                init_py.write_text("", encoding="utf-8")
            final_content = new_test_code

        test_file_path.write_text(final_content, encoding="utf-8")

        state["existing_test_code"] = new_test_code
        state["test_file_path"] = test_file_path

    except Exception as e:
        state["errors"].append(f"Test generation error for {function.id}: {e}")

    # Always increment iteration to prevent infinite loops
    state["iteration"] = state["iteration"] + 1
    return state


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_per_function_graph():
    workflow = StateGraph(FunctionState)

    workflow.add_node("lookup", lookup_tests_node)
    workflow.add_node("evaluate", evaluate_tests_node)
    workflow.add_node("create", create_tests_node)

    workflow.set_entry_point("lookup")
    workflow.add_edge("lookup", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {
            "create": "create",
            END: END,
        },
    )

    workflow.add_edge("create", "evaluate")

    return workflow.compile()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_workflow(
    project_root: Path,
    output_path: Path,
    coverage_threshold: float = 80.0,
    max_iterations: int = 3,
) -> dict[str, Any]:
    """
    Run the per-function LangGraph workflow over an entire project.

    For each source function:
      1. lookup_node  – find existing tests
      2. evaluate_node – run pytest with per-function line coverage
      3. create_node  – ask Claude to generate tests (loops back to evaluate)

    Writes results to output_path as JSON and returns the output dict.
    """
    root = project_root.resolve()

    print(f"\nScanning project: {root}")
    functions = collect_functions(root)
    print(f"Found {len(functions)} source functions")

    # Populate source_code for every function up front
    for func in functions:
        func.source_code = extract_function_source(root / func.file_path, func.qualified_name)

    graph = _build_per_function_graph()
    results: list[dict[str, Any]] = []

    for i, func in enumerate(functions, 1):
        print(f"\n[{i}/{len(functions)}] Processing: {func.id}")

        if not func.source_code:
            print("  [!] Could not extract source, skipping")
            results.append(
                {
                    "id": func.id,
                    "name": func.name,
                    "file": func.file_path,
                    "function_code": "",
                    "test_code": "",
                    "test_file": None,
                    "coverage_score": 0.0,
                    "status": "failed",
                    "errors": ["Could not extract source code"],
                }
            )
            continue

        initial_state: FunctionState = {
            "project_root": root,
            "function": func,
            "coverage_threshold": coverage_threshold,
            "max_iterations": max_iterations,
            "existing_test_code": "",
            "test_file_path": None,
            "coverage_score": 0.0,
            "iteration": 0,
            "final_test_code": "",
            "status": "in_progress",
            "errors": [],
        }

        try:
            final_state = graph.invoke(initial_state)
        except Exception as e:
            print(f"  [ERROR] Workflow failed: {e}")
            results.append(
                {
                    "id": func.id,
                    "name": func.name,
                    "file": func.file_path,
                    "function_code": func.source_code,
                    "test_code": "",
                    "test_file": None,
                    "coverage_score": 0.0,
                    "status": "failed",
                    "errors": [str(e)],
                }
            )
            continue

        test_file = final_state.get("test_file_path")
        status = final_state.get("status", "failed")

        # Finalize status if still in_progress (e.g. max_iterations == 0)
        if status == "in_progress":
            score = final_state.get("coverage_score", 0.0)
            iters = final_state.get("iteration", 0)
            if score >= coverage_threshold:
                status = "passed_existing" if iters == 0 else "generated"
            else:
                status = "failed"

        score = final_state.get("coverage_score", 0.0)
        print(f"  Status: {status}, Coverage: {score:.1f}%")

        # Prefer final_test_code (full file); fall back to existing_test_code (matched functions)
        test_code_out = final_state.get("final_test_code") or final_state.get(
            "existing_test_code", ""
        )

        results.append(
            {
                "id": func.id,
                "name": func.name,
                "file": func.file_path,
                "function_code": func.source_code,
                "test_code": test_code_out,
                "test_file": str(test_file) if test_file else None,
                "coverage_score": score,
                "status": status,
                "errors": final_state.get("errors", []),
            }
        )

    output: dict[str, Any] = {
        "project_root": str(root),
        "generated_at": datetime.now().isoformat(),
        "coverage_threshold": coverage_threshold,
        "functions": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults written to: {output_path}")

    return output
