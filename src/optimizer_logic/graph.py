"""
graph.py — LangGraph pipeline for optimizing a single function.

Flow:
    measure_baseline
        → (skip?)      → finalize
        → (ok)         → optimize

    optimize (attempt += 1)
        → run_tests

    run_tests
        → passed                        → measure_optimized
        → failed AND attempt < max      → optimize (retry: test suite failed)
        → failed AND attempt >= max     → finalize (no improvement)

    measure_optimized
        → emissions < baseline          → finalize (success)
        → worse AND attempt < max       → optimize (retry: X% worse than baseline)
        → worse AND attempt >= max      → finalize (no improvement)

    finalize → END
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from .venv_runner import (
    build_project_dir,
    measure_emissions_via_pytest,
    replace_function_in_source,
    run_spec,
)

load_dotenv()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class OptimizerState(TypedDict):
    func_record: dict            # one entry from spec results.json
    project_root: str
    python_bin: Path             # venv python binary
    full_source: str             # full text of original source file (immutable reference)
    current_function_code: str   # function body with original file indentation
    current_full_source: str     # full source with current_function_code spliced in
    baseline_emissions: float
    optimized_emissions: Optional[float]
    test_passed: bool
    attempt: int                 # starts 0, incremented in optimize node
    max_attempts: int            # always 2
    last_test_output: str
    retry_reason: str            # human-readable reason for current retry
    success: bool
    skip_reason: Optional[str]
    use_hints: bool              # when True, inject AST-derived optimization hints into the prompt


# ---------------------------------------------------------------------------
# Feature-targeted optimization hints
# ---------------------------------------------------------------------------
# Guided by RandomForest feature importances measured on 58 optimized functions:
#   num_function_calls 0.264 | loc 0.212 | cyclomatic_complexity 0.161
#   num_arithmetic_ops 0.103 | has_string_ops 0.059 | has_torch 0.056

def _get_optimization_hints(source: str) -> list[str]:
    """Analyse source via AST and return targeted optimization hints."""
    hints: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return hints

    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    num_calls = len(calls)

    call_names: list[str] = []
    for c in calls:
        if isinstance(c.func, ast.Name):
            call_names.append(c.func.id)
        elif isinstance(c.func, ast.Attribute):
            call_names.append(c.func.attr)
    repeated = {n for n, cnt in Counter(call_names).items() if cnt > 1}

    arithmetic_ops = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, ast.BinOp)
        and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div,
                               ast.Mod, ast.Pow, ast.FloorDiv))
    )
    num_loops = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While)))
    loc = source.count("\n") + 1
    branches = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                           ast.With, ast.comprehension, ast.BoolOp))
    )
    uses_str_ops = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("join", "split", "replace", "format", "strip",
                             "startswith", "endswith", "find", "count",
                             "encode", "decode")
        for n in ast.walk(tree)
    )
    has_torch = any(
        isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "torch"
        for n in ast.walk(tree)
    )
    uses_numpy = any(
        isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id in ("np", "numpy")
        for n in ast.walk(tree)
    )
    has_list_comp = any(
        isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
        for n in ast.walk(tree)
    )

    # num_function_calls is the #1 predictor (importance 0.264)
    if num_calls > 5 and repeated:
        hints.append(
            f"Cache repeated calls — {', '.join(sorted(repeated)[:4])} are called "
            "multiple times. Store results in local variables or use functools.lru_cache."
        )
    elif num_calls > 10:
        hints.append(
            "Many function calls detected. Hoist repeated attribute lookups into "
            "local variables before any loop to reduce overhead."
        )

    # cyclomatic_complexity (importance 0.161)
    if branches > 6:
        hints.append(
            "High branching complexity. Consider a lookup dict or dispatch table "
            "instead of chained if/elif chains."
        )
    if loc > 40 and branches > 4:
        hints.append(
            "Long function with many branches. Use early returns to reduce nesting."
        )

    # num_arithmetic_ops (importance 0.103)
    if arithmetic_ops > 8 and num_loops > 0 and not uses_numpy:
        hints.append(
            "Heavy arithmetic inside loops. Consider NumPy vectorized operations "
            "(np.array / np.sum / np.dot) to eliminate Python loop overhead."
        )
    elif arithmetic_ops > 4 and num_loops == 0:
        hints.append(
            "Multiple arithmetic operations. Combine expressions and prefer "
            "integer arithmetic where possible."
        )

    if num_loops > 0 and not has_list_comp:
        hints.append(
            "Loops that build lists can often be replaced with list comprehensions "
            "or map(), which carry less bytecode overhead."
        )

    # has_string_ops (importance 0.059)
    if uses_str_ops:
        hints.append(
            "String operations detected. Use str.join() instead of += concatenation, "
            "pre-compile regex with re.compile(), prefer f-strings."
        )

    # has_torch (importance 0.056)
    if has_torch:
        hints.append(
            "PyTorch code detected. Prefer in-place operations (add_, mul_), "
            "avoid repeated .to(device) in loops, use torch.no_grad() for inference."
        )

    if uses_numpy and num_loops > 0:
        hints.append(
            "NumPy is available — replace remaining Python loops with NumPy "
            "broadcasting / ufuncs for maximum vectorization."
        )

    return hints


# ---------------------------------------------------------------------------
# Indentation helpers
# ---------------------------------------------------------------------------

def _extract_indented_function(full_source: str, start_line: int, end_line: int) -> str:
    """Return the lines [start_line..end_line] (1-indexed, inclusive) from full_source."""
    lines = full_source.splitlines()
    return "\n".join(lines[start_line - 1 : end_line])


def _get_indent(full_source: str, start_line: int) -> str:
    """Return the leading whitespace of the function definition line."""
    lines = full_source.splitlines()
    if start_line - 1 < len(lines):
        line = lines[start_line - 1]
        return line[: len(line) - len(line.lstrip())]
    return ""


def _apply_indent(code: str, indent: str) -> str:
    """Re-apply *indent* to every line of *code* if the first line lacks it."""
    if not indent:
        return code
    lines = code.splitlines()
    if not lines:
        return code
    if lines[0].startswith(indent):
        return code  # already correctly indented
    return "\n".join(indent + line for line in lines)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def measure_baseline(state: OptimizerState) -> dict:
    import tempfile, pathlib

    func_record = state["func_record"]
    python = state["python_bin"]
    full_source = state["full_source"]
    start_line = func_record.get("line", 1)
    end_line = func_record.get("end_line", start_line)

    print(f"[measure_baseline] {func_record['id']} ...")

    # Extract the function with its original indentation from the source file
    indented_function_code = _extract_indented_function(full_source, start_line, end_line)

    with tempfile.TemporaryDirectory(prefix="optimizer_baseline_") as tmp_str:
        tmp = pathlib.Path(tmp_str)
        build_project_dir(tmp, func_record, full_source)
        avg_emissions, all_passed = measure_emissions_via_pytest(python, tmp, runs=1)

    if not all_passed or avg_emissions == 0.0:
        reason = "tests failed" if not all_passed else "measurement returned 0"
        print(f"[measure_baseline] SKIP — baseline {reason}")
        return {
            "baseline_emissions": 0.0,
            "current_function_code": indented_function_code,
            "current_full_source": full_source,
            "skip_reason": "baseline_measurement_failed",
        }

    print(f"[measure_baseline] baseline = {avg_emissions:.2e} kg CO2eq")
    return {
        "baseline_emissions": avg_emissions,
        "current_function_code": indented_function_code,
        "current_full_source": full_source,
        "skip_reason": None,
    }


def optimize(state: OptimizerState) -> dict:
    attempt = state["attempt"] + 1
    func_record = state["func_record"]
    retry_reason = state.get("retry_reason", "")

    print("  generating new code")
    if attempt > 1 and retry_reason:
        print(f"  (retry {attempt}/{state['max_attempts']}: {retry_reason})")

    client = anthropic.Anthropic()

    # current_function_code already has the original file indentation
    current_code = state["current_function_code"]
    spec_code = func_record.get("spec_code", "")
    start_line = func_record.get("line", 1)
    indent = _get_indent(state["full_source"], start_line)

    user_content = (
        "Optimize the following Python function to reduce CPU usage and energy consumption.\n\n"
        "IMPORTANT rules:\n"
        "1. Return ONLY the optimized function inside a ```python code block.\n"
        "2. Preserve the EXACT leading indentation of every line (the function may be a class method).\n"
        "3. Do not add any explanation outside the code block.\n\n"
        f"Original function (keep the same indentation):\n```python\n{current_code}\n```"
    )

    # Inject AST-derived optimization hints when --hints is enabled
    if state.get("use_hints"):
        hints = _get_optimization_hints(current_code)
        if hints:
            hint_text = "\n".join(f"- {h}" for h in hints)
            user_content += (
                f"\n\nBased on static analysis of this function, focus on these "
                f"specific optimizations:\n{hint_text}"
            )

    if attempt > 1 and state.get("last_test_output"):
        user_content += (
            f"\n\nPrevious attempt failed ({retry_reason}). "
            f"Test output:\n```\n{state['last_test_output'][:2000]}\n```\n\n"
            f"The function must pass these tests:\n```python\n{spec_code}\n```"
        )

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=(
            "You are an expert Python performance engineer. "
            "When asked to optimize a function, return ONLY the optimized function "
            "inside a single ```python code block. "
            "Match the original indentation exactly — if the original has 4-space leading indent, keep it. "
            "No explanations, no other text."
        ),
        messages=[{"role": "user", "content": user_content}],
    )

    raw = message.content[0].text

    match = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    optimized_code = match.group(1).rstrip("\n") if match else raw.strip()

    # Re-apply original indentation if Claude stripped it
    optimized_code = _apply_indent(optimized_code, indent)

    # Splice back into full source using original line numbers
    end_line = func_record.get("end_line", start_line)
    new_full_source = replace_function_in_source(
        state["full_source"], optimized_code, start_line, end_line
    )

    return {
        "attempt": attempt,
        "current_function_code": optimized_code,
        "current_full_source": new_full_source,
    }


def run_tests(state: OptimizerState) -> dict:
    import tempfile, pathlib

    func_record = state["func_record"]
    python = state["python_bin"]
    print("  testing code with spec files")

    with tempfile.TemporaryDirectory(prefix="optimizer_test_") as tmp_str:
        tmp = pathlib.Path(tmp_str)
        build_project_dir(tmp, func_record, state["current_full_source"])
        passed, output = run_spec(python, tmp)

    if passed:
        print("  passed spec file")
    else:
        print("  failed spec file")

    return {"test_passed": passed, "last_test_output": output}


def measure_optimized(state: OptimizerState) -> dict:
    import tempfile, pathlib

    func_record = state["func_record"]
    python = state["python_bin"]
    baseline = state["baseline_emissions"]
    print("  testing code with codecarbon")

    with tempfile.TemporaryDirectory(prefix="optimizer_measure_") as tmp_str:
        tmp = pathlib.Path(tmp_str)
        build_project_dir(tmp, func_record, state["current_full_source"])
        avg_emissions, all_passed = measure_emissions_via_pytest(python, tmp, runs=1)

    if baseline > 0:
        pct = (avg_emissions - baseline) / baseline * 100
        if avg_emissions < baseline:
            print(f"  passed codecarbon (more efficient by {abs(pct):.1f}%)")
        else:
            print(f"  failed codecarbon (less efficient by {abs(pct):.1f}%)")
    else:
        print(f"  codecarbon result: {avg_emissions:.2e} kg CO2eq")

    return {"optimized_emissions": avg_emissions}


def finalize(state: OptimizerState) -> dict:
    skip_reason = state.get("skip_reason")
    if skip_reason:
        print(f"[finalize] SKIP — {skip_reason}")
        return {"success": False}

    baseline = state["baseline_emissions"]
    optimized = state.get("optimized_emissions")

    if optimized is not None and optimized < baseline:
        reduction = (baseline - optimized) / baseline * 100 if baseline > 0 else 0.0
        print(f"[finalize] SUCCESS — {baseline:.2e} -> {optimized:.2e} kg CO2eq ({reduction:.1f}% reduction)")
        return {"success": True}

    print(f"[finalize] NO IMPROVEMENT after {state['attempt']} attempt(s)")
    return {"success": False, "optimized_emissions": None}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_baseline(state: OptimizerState) -> str:
    if state.get("skip_reason"):
        return "finalize"
    return "optimize"


def route_after_tests(state: OptimizerState) -> str:
    if state["test_passed"]:
        return "measure_optimized"
    if state["attempt"] < state["max_attempts"]:
        return "optimize"
    return "finalize"


def route_after_measure_optimized(state: OptimizerState) -> str:
    optimized = state.get("optimized_emissions") or 0.0
    baseline = state["baseline_emissions"]

    if optimized < baseline:
        return "finalize"

    if state["attempt"] < state["max_attempts"]:
        pct = (optimized - baseline) / baseline * 100 if baseline > 0 else 0.0
        state["retry_reason"] = f"{pct:.1f}% worse emissions than baseline"
        return "optimize"

    return "finalize"


def set_retry_reason_emissions(state: OptimizerState) -> dict:
    """Intermediate node: set retry_reason before looping back to optimize from measure_optimized."""
    optimized = state.get("optimized_emissions") or 0.0
    baseline = state["baseline_emissions"]
    pct = (optimized - baseline) / baseline * 100 if baseline > 0 else 0.0
    return {"retry_reason": f"{pct:.1f}% worse emissions than baseline"}


def set_retry_reason_tests(state: OptimizerState) -> dict:
    """Intermediate node: set retry_reason before looping back to optimize from run_tests."""
    return {"retry_reason": "test suite failed"}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(OptimizerState)

    builder.add_node("measure_baseline", measure_baseline)
    builder.add_node("optimize", optimize)
    builder.add_node("run_tests", run_tests)
    builder.add_node("set_retry_tests", set_retry_reason_tests)
    builder.add_node("measure_optimized", measure_optimized)
    builder.add_node("set_retry_emissions", set_retry_reason_emissions)
    builder.add_node("finalize", finalize)

    builder.set_entry_point("measure_baseline")

    builder.add_conditional_edges(
        "measure_baseline",
        route_after_baseline,
        {"optimize": "optimize", "finalize": "finalize"},
    )
    builder.add_edge("optimize", "run_tests")

    def _route_tests(state: OptimizerState) -> str:
        if state["test_passed"]:
            return "measure_optimized"
        if state["attempt"] < state["max_attempts"]:
            return "set_retry_tests"
        return "finalize"

    builder.add_conditional_edges(
        "run_tests",
        _route_tests,
        {
            "measure_optimized": "measure_optimized",
            "set_retry_tests": "set_retry_tests",
            "finalize": "finalize",
        },
    )
    builder.add_edge("set_retry_tests", "optimize")

    def _route_emissions(state: OptimizerState) -> str:
        optimized = state.get("optimized_emissions") or 0.0
        baseline = state["baseline_emissions"]
        if optimized < baseline:
            return "finalize"
        if state["attempt"] < state["max_attempts"]:
            return "set_retry_emissions"
        return "finalize"

    builder.add_conditional_edges(
        "measure_optimized",
        _route_emissions,
        {
            "finalize": "finalize",
            "set_retry_emissions": "set_retry_emissions",
        },
    )
    builder.add_edge("set_retry_emissions", "optimize")

    builder.add_edge("finalize", END)

    return builder.compile()
