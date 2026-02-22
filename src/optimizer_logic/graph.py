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
        → failed AND attempt < max      → optimize (retry)
        → failed AND attempt >= max     → finalize (no improvement)

    measure_optimized
        → emissions < baseline          → finalize (success)
        → worse AND attempt < max       → optimize (retry)
        → worse AND attempt >= max      → finalize (no improvement)

    finalize → END
"""

from __future__ import annotations

import re
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
    current_function_code: str   # function body being optimized (just the method)
    current_full_source: str     # full source with current_function_code spliced in
    baseline_emissions: float
    optimized_emissions: Optional[float]
    test_passed: bool
    attempt: int                 # starts 0, incremented in optimize node
    max_attempts: int            # always 2
    last_test_output: str
    success: bool
    skip_reason: Optional[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def measure_baseline(state: OptimizerState) -> dict:
    print(f"[measure_baseline] {state['func_record']['id']} ...")
    import tempfile, pathlib

    func_record = state["func_record"]
    python = state["python_bin"]
    project_root = state["project_root"]
    full_source = state["full_source"]

    with tempfile.TemporaryDirectory(prefix="optimizer_baseline_") as tmp_str:
        tmp = pathlib.Path(tmp_str)
        build_project_dir(tmp, func_record, full_source)
        avg_emissions, all_passed = measure_emissions_via_pytest(python, tmp, runs=3)

    if not all_passed:
        print(f"[measure_baseline] tests failed during baseline — skipping")
        return {
            "baseline_emissions": 0.0,
            "current_function_code": func_record["function_code"],
            "current_full_source": full_source,
            "skip_reason": "baseline_measurement_failed",
        }

    if avg_emissions == 0.0:
        print(f"[measure_baseline] baseline emissions = 0 (measurement failed) — skipping")
        return {
            "baseline_emissions": 0.0,
            "current_function_code": func_record["function_code"],
            "current_full_source": full_source,
            "skip_reason": "baseline_measurement_failed",
        }

    print(f"[measure_baseline] baseline = {avg_emissions:.2e} kg CO2eq")
    return {
        "baseline_emissions": avg_emissions,
        "current_function_code": func_record["function_code"],
        "current_full_source": full_source,
        "skip_reason": None,
    }


def optimize(state: OptimizerState) -> dict:
    attempt = state["attempt"] + 1
    func_record = state["func_record"]
    print(f"[optimize] {func_record['id']} — attempt {attempt}/{state['max_attempts']}")

    client = anthropic.Anthropic()

    current_code = state["current_function_code"]
    spec_code = func_record.get("spec_code", "")

    user_content = (
        f"Optimize the following Python function to reduce CPU usage and energy consumption.\n\n"
        f"IMPORTANT: Return ONLY the optimized function inside a ```python code block.\n"
        f"Keep the EXACT same indentation level as the original (do not add or remove leading spaces).\n"
        f"Do not include any explanation.\n\n"
        f"Original function:\n```python\n{current_code}\n```"
    )

    if attempt > 1 and state.get("last_test_output"):
        user_content += (
            f"\n\nThe previous version failed tests or had worse emissions. "
            f"Test output:\n```\n{state['last_test_output']}\n```\n\n"
            f"The function must pass these tests:\n```python\n{spec_code}\n```"
        )

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        system=(
            "You are an expert Python performance engineer. "
            "When asked to optimize a function, return ONLY the optimized function "
            "inside a single ```python code block. "
            "Preserve the original indentation level exactly. No explanations, no other text."
        ),
        messages=[{"role": "user", "content": user_content}],
    )

    raw = message.content[0].text
    match = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    optimized_code = match.group(1).rstrip("\n") if match else raw.strip()

    print(f"[optimize] received optimized code ({len(optimized_code)} chars)")

    # Splice optimized code back into full source
    start_line = func_record.get("line", 1)
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
    print(f"[run_tests] {func_record['id']} ...")

    with tempfile.TemporaryDirectory(prefix="optimizer_test_") as tmp_str:
        tmp = pathlib.Path(tmp_str)
        build_project_dir(tmp, func_record, state["current_full_source"])
        passed, output = run_spec(python, tmp)

    print(f"[run_tests] {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(output[:800])
    return {"test_passed": passed, "last_test_output": output}


def measure_optimized(state: OptimizerState) -> dict:
    import tempfile, pathlib

    func_record = state["func_record"]
    python = state["python_bin"]
    print(f"[measure_optimized] {func_record['id']} ...")

    with tempfile.TemporaryDirectory(prefix="optimizer_measure_") as tmp_str:
        tmp = pathlib.Path(tmp_str)
        build_project_dir(tmp, func_record, state["current_full_source"])
        avg_emissions, all_passed = measure_emissions_via_pytest(python, tmp, runs=3)

    print(f"[measure_optimized] emissions = {avg_emissions:.2e} kg CO2eq")
    return {"optimized_emissions": avg_emissions}


def finalize(state: OptimizerState) -> dict:
    skip_reason = state.get("skip_reason")
    if skip_reason:
        print(f"[finalize] skipped: {skip_reason}")
        return {"success": False}

    baseline = state["baseline_emissions"]
    optimized = state.get("optimized_emissions")

    if optimized is not None and optimized < baseline:
        reduction = (baseline - optimized) / baseline * 100 if baseline > 0 else 0.0
        print(f"[finalize] SUCCESS — {baseline:.2e} → {optimized:.2e} kg ({reduction:.1f}% reduction)")
        return {"success": True}

    print(f"[finalize] no improvement after {state['attempt']} attempt(s)")
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
    optimized = state.get("optimized_emissions", 0.0) or 0.0
    baseline = state["baseline_emissions"]
    if optimized < baseline:
        return "finalize"
    if state["attempt"] < state["max_attempts"]:
        return "optimize"
    return "finalize"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(OptimizerState)

    builder.add_node("measure_baseline", measure_baseline)
    builder.add_node("optimize", optimize)
    builder.add_node("run_tests", run_tests)
    builder.add_node("measure_optimized", measure_optimized)
    builder.add_node("finalize", finalize)

    builder.set_entry_point("measure_baseline")

    builder.add_conditional_edges(
        "measure_baseline",
        route_after_baseline,
        {"optimize": "optimize", "finalize": "finalize"},
    )
    builder.add_edge("optimize", "run_tests")
    builder.add_conditional_edges(
        "run_tests",
        route_after_tests,
        {
            "measure_optimized": "measure_optimized",
            "optimize": "optimize",
            "finalize": "finalize",
        },
    )
    builder.add_conditional_edges(
        "measure_optimized",
        route_after_measure_optimized,
        {"finalize": "finalize", "optimize": "optimize"},
    )
    builder.add_edge("finalize", END)

    return builder.compile()
